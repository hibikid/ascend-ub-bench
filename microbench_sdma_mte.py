#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""
双机 DRAM BM 内存池上，SDMA 与 MTE（AI Core）两种传输引擎的 microbench。

数据格式：torch(shape=(N, 1, 576), device="npu")（默认 int32），N 由 --scales 扫描，
默认 N ∈ {64, 128, 256, 512, 1024, 2048}，最大档 (2048, 1, 576) 约 4.5 MiB。
方向：L2G（本地 NPU -> 对端 DRAM 池 GVA）与 G2L（对端 DRAM 池 GVA -> 本地 NPU）。

引擎由 HostDataOpSDMA::CopyG2G（src/hybm/csrc/data_operation/host/hybm_data_op_sdma.cpp）
的 flags 决定，两条路径都会真正执行传输：
  - flags = 0                 -> SDMA 引擎 SQE 拷贝（InitG2GStreamTask + HalSqTaskSend）
  - flags = COPY_EXTEND_FLAG  -> HybmCopyExtend -> hybm_copy_kernel
                                （AIV 内核，MTE2 GM->UB / MTE3 UB->GM，需 A3 超节点映射远端 DRAM）

指标：
  1. 单条时延/带宽（每个 scale，每条 方向 x 引擎 组合）：
     单次 copy_data 的端到端耗时（同步调用，内部已 AclrtSynchronizeStream / hStream->Synchronize）
  2. 批量吞吐（只在最大 scale）：copy_data_batch 聚合带宽
     （默认 --batch-mode random：从独立的批量源张量 (batch_rows=16384, 1, 576) 随机取 B 条
      (1,576) 行，scatter 到 peer 批量区 / 从 peer 批量区 gather 回 scratch；
      --batch-mode strided：从同一批量源按固定间隔取 B 行（--step-size 控制源行间隔，
      0=连续取 0,1,2,...，1=隔一行取 0,2,4,...），dst 保持连续；
      --batch-mode broadcast 则把整张最大 scale 的 src 广播到 peer DRAM 的 B 个连续切片 /
      从 B 个连续切片收进 dst）

角色（双机 --world_size 2）：
     rank0 = config store host + 纯内存持有方：建池 join 后循环 sleep，直到 Ctrl+C 退出。
             不碰 peer GVA、不执行任何拷贝，因此不会因 peer 切片未 import 而报错。
     rank1 = benchmark 驱动/计时方：join 后先 send_peer_ready——循环向 rank0 的 DRAM 池末尾
             写 READY_MAGIC，写成功即说明 rank0 切片已 import 进本机设备页表（能写就说明映射
             好了；JoinHandle 的 GroupGatherResult barrier 两边一起过，rank0 侧 import 也同步
             完成），之后对 peer_gva 的全部计时拷贝必然可达。
     动态组模式下任意一方的 join() 只等自己就返回，"把对端切片 import 进本机设备页表"是异步
     的——这正是"有时跑会出错"的根因；握手把它变成明确的等待或超时。

运行（两台机器各一个终端，先 rank0 后 rank1）：
  # rank0（config store host + 内存持有方，Ctrl+C 退出）：
  python3 microbench_sdma_mte.py --rank 0 --url tcp://<rank0-ip>:8570 --nic <rank0-数据面ip>
  # rank1（benchmark 驱动/计时）：
  python3 microbench_sdma_mte.py --rank 1 --url tcp://<rank0-ip>:8570 --nic <rank1-数据面ip>

单机自测（无第二台机，--world_size 1）：
  python3 microbench_sdma_mte.py --rank 0 --world_size 1 --url tcp://127.0.0.1:8570
"""
import argparse
import ctypes
import os
import random
import time

import torch
import torch_npu  # noqa: F401  （导入即完成 torch_npu 注册）
import memfabric_hybrid
from memfabric_hybrid import bm

from bm_mem_pool import BmDramPool

SHAPE = (2048, 1, 576)
COPY_EXTEND_FLAG = 1 << 1          # 见 src/smem/include/host/smem.h：走 hybm_copy_extend（MTE）
BLOCK_ALIGN = 64                   # hybm_copy_kernel.cpp 的 SINGLE_COPY_SLICE，长度须 64B 整数倍
DONE_BLOCK_BYTES = 64              # 握手/标记块大小，保证 64B 对齐
READY_MAGIC = 0x51DE9A7E           # 握手标记：rank1 join 后写入 rank0 DRAM 池末尾，< 2^31

DTYPES = {
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

ENGINES = (("SDMA", 0), ("MTE", COPY_EXTEND_FLAG))
DIRECTIONS = ("L2G", "G2L")


def check_extend_lib():
    """确认 MTE 路径的编译产物 libmf_hybm_copy_extend.so 可用（benchmark 驱动方 rank1 才需要）。"""
    lib_dir = os.environ.get("MEMFABRIC_HYBRID_EXTEND_LIB_PATH")
    if not lib_dir:
        raise RuntimeError(
            "MEMFABRIC_HYBRID_EXTEND_LIB_PATH 未设置，请先 source memfabric_hybrid 的 set_env.sh，"
            "或用 --extend-lib-path 指定 libmf_hybm_copy_extend.so 所在目录"
        )
    so_path = os.path.join(lib_dir, "libmf_hybm_copy_extend.so")
    if not os.path.exists(so_path):
        raise RuntimeError(
            f"not found: {so_path}，请确认 run 包已安装，或执行 examples/hybm_copy_extend/build.sh 从源码编译"
        )
    # 先全局加载 libascendcl，保证 kernel so 里的 acl 符号能解析（通常 torch_npu 已加载）
    try:
        ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
    lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
    assert hasattr(lib, "hybm_copy_extend"), f"{so_path} 缺少 hybm_copy_extend 符号"
    return so_path


def parse_args():
    ap = argparse.ArgumentParser(
        description="dual-node DRAM BM pool: SDMA vs MTE (COPY_EXTEND_FLAG) transfer microbench")
    ap.add_argument("--rank", type=int, default=0,
                    help="global rank id: 0(store host + memory holder) or 1(benchmark driver)")
    ap.add_argument("--world_size", type=int, default=2, choices=[1, 2],
                    help="1: single-machine smoke; 2: dual-machine (default)")
    ap.add_argument("--device", type=int, default=0, help="local NPU device id, default 0")
    ap.add_argument("--url", default="tcp://127.0.0.1:8570",
                    help="config store url, e.g. tcp://<rank0-ip>:8570")
    ap.add_argument("--nic", default="127.0.0.1", help="local host/device nic ip")
    ap.add_argument("--local-dram", type=int, default=1 << 30,
                    help="local DRAM contributed to pool, default 1GiB (2MiB aligned)")
    ap.add_argument("--max-dram", type=int, default=1 << 30,
                    help="max per-rank DRAM, default 1GiB (2MiB aligned)")
    ap.add_argument("--dtype", default="int32", choices=sorted(DTYPES),
                    help="tensor dtype (determines per-tensor bytes)")
    ap.add_argument("--iters", type=int, default=100, help="single-copy iterations per measurement")
    ap.add_argument("--warmup", type=int, default=10, help="single-copy warmup iterations")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="tensors per copy_data_batch call (batch region must fit local DRAM)")
    ap.add_argument("--batch-iters", type=int, default=20, help="batch-copy iterations per measurement")
    ap.add_argument("--batch-warmup", type=int, default=2, help="batch-copy warmup iterations")
    ap.add_argument("--batch-mode", default="random", choices=["random", "strided", "broadcast"],
                    help="batch copy pattern: 'random' (default) scatters B random (1,576) rows of the "
                         "dedicated batch-source tensor into the peer batch region; 'strided' copies B rows "
                         "with a fixed source interval (--step-size) into a contiguous peer region; "
                         "'broadcast' copies the largest-scale tensor to B consecutive slices")
    ap.add_argument("--batch-rows", type=int, default=16384,
                    help="first dim of the dedicated batch-source tensor (batch_rows, 1, 576); 'random' and "
                         "'strided' modes take rows from it")
    ap.add_argument("--step-size", type=int, default=0,
                    help="--batch-mode strided: interval (in (1,576) rows) between consecutive source rows, "
                         "0 = contiguous (rows 0,1,2,...), 1 = skip one (rows 0,2,4,...); destination stays "
                         "contiguous")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for --batch-mode random row selection")
    ap.add_argument("--scales", default="64,128,256,512,1024,2048",
                    help="comma-separated first-dim values for the single-copy scale sweep "
                         "(shape=(N,1,576)); batch throughput is measured on the largest scale only")
    ap.add_argument("--ready-timeout", type=int, default=120,
                    help="rank1 timeout (s) for the ready handshake: retries writing READY_MAGIC into "
                         "rank0's DRAM pool until rank0's slice is imported on this device")
    ap.add_argument("--extend-lib-path", default=None,
                    help="dir containing libmf_hybm_copy_extend.so, sets MEMFABRIC_HYBRID_EXTEND_LIB_PATH")
    return ap.parse_args()


def make_pattern(dtype, shape=SHAPE):
    """确定性填充：0..96，所有 dtype 精确可表示，保证回读逐元素相等。"""
    return base_pattern(dtype, shape).view(shape)


def tensor_nbytes(dtype, shape=SHAPE):
    """单个 shape tensor 的字节数，需为 64B 整数倍。"""
    numel = 1
    for s in shape:
        numel *= s
    return numel * torch.empty((), dtype=dtype).element_size()


def base_pattern(dtype, shape=SHAPE):
    """确定性 1-D 模式（0..96），reshape 成 shape。"""
    numel = 1
    for s in shape:
        numel *= s
    return (torch.arange(numel, dtype=torch.int64, device="npu") % 97).to(dtype)


def shape_for(scale):
    """scale 扫描用的 shape：第一维取 scale，其余两维固定 (1, 576)。"""
    return (scale, 1, 576)


def parse_scales(s):
    """解析 --scales "64,128,..."，去重升序，保证最后一档是最大规模（批量吞吐用它）。"""
    scales = [int(x) for x in s.split(",") if x.strip()]
    assert scales, "no scales given"
    assert all(x > 0 for x in scales), "scales must be positive"
    return sorted(set(scales))


def time_single_copy(pool, copy_type, flags, src_ptr, dst_ptr, nbytes, warmup, iters):
    """端到端同步单条拷贝耗时（copy_data 内部已同步，time.perf_counter 直接测得真实时延）。"""
    for _ in range(warmup):
        assert pool.copy(src_ptr, dst_ptr, nbytes, copy_type, flags) == 0, "copy_data failed"
    torch_npu.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        assert pool.copy(src_ptr, dst_ptr, nbytes, copy_type, flags) == 0, "copy_data failed"
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def time_batch_copy(pool, copy_type, flags, src_addrs, dst_addrs, sizes, count, warmup, iters):
    """批量同步拷贝耗时，返回单次 copy_data_batch 的平均耗时。"""
    for _ in range(warmup):
        assert pool.copy_batch(src_addrs, dst_addrs, sizes, count, copy_type, flags) == 0, "copy_data_batch failed"
    torch_npu.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        assert pool.copy_batch(src_addrs, dst_addrs, sizes, count, copy_type, flags) == 0, "copy_data_batch failed"
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def bench_one(pool, direction, engine, flags, src, dst, peer_gva, nbytes, args, with_batch=True,
              batch_src=None):
    """对单个 方向 x 引擎 组合做正确性校验 + 单条时延/带宽；with_batch 时再加批量吞吐。

    batch_src：random/strided 批量模式专用的批量源张量 (batch_rows, 1, 576)，独立于 scale 扫描的 src。
    """
    shape = tuple(src.shape)
    row = {"shape": shape, "direction": direction, "engine": engine,
           "latency_us": None, "single_bw_gbs": None, "batch_bw_gbs": None, "error": None}

    copy_type = bm.BmCopyType.L2G if direction == "L2G" else bm.BmCopyType.G2L
    src_ptr, dst_ptr = (src.data_ptr(), peer_gva) if direction == "L2G" else (peer_gva, dst.data_ptr())

    # 1) 正确性：L2G 写入 -> G2L 读回 -> 逐元素相等（写回读验整条链路，也顺带把 peer DRAM 填上模式）
    try:
        assert pool.copy(src.data_ptr(), peer_gva, nbytes, bm.BmCopyType.L2G, flags) == 0, "L2G verify failed"
        torch_npu.npu.synchronize()
        assert pool.copy(peer_gva, dst.data_ptr(), nbytes, bm.BmCopyType.G2L, flags) == 0, "G2L verify failed"
        torch_npu.npu.synchronize()
        if not torch.equal(dst, src):
            raise RuntimeError("round-trip data mismatch")
    except Exception as e:
        row["error"] = f"unsupported or verify failed: {e}"
        return row

    # 2) 单条时延 + 带宽（G2L 的数据已由步骤 1 的 L2G 写入就位）
    try:
        avg_s = time_single_copy(pool, copy_type, flags, src_ptr, dst_ptr, nbytes, args.warmup, args.iters)
        row["latency_us"] = avg_s * 1e6
        row["single_bw_gbs"] = nbytes / avg_s / 1e9
    except Exception as e:
        row["error"] = f"single-copy failed: {e}"
        return row

    if not with_batch:
        return row

    # 3) 批量吞吐：copy_data_batch 一次搬 B 条 (1,576) 行
    #    - random（默认）：从独立批量源 batch_src(batch_rows,1,576) 随机取 B 行，scatter 到 peer 批量区 /
    #      从 peer 批量区 gather 回本地 scratch
    #    - strided：从 batch_src 按固定间隔取 B 行（--step-size 控制源行间隔，0=连续取 0,1,2,...；
    #      1=隔一行取 0,2,4,...），dst 仍连续，scatter/gather 同上
    #    - broadcast：把整张 src 广播到 peer DRAM 的 B 个连续切片 / 从 B 个连续切片收进 dst
    try:
        count = args.batch_size
        row_b = nbytes // shape[0]            # 单条 (1,576) 行的字节数 = element_size * 576（src 与 batch_src 同 dtype，行字节相同）
        if args.batch_mode in ("random", "strided"):
            assert batch_src is not None, "random/strided batch mode requires batch_src"
            n_rows = batch_src.shape[0]                   # 从 (batch_rows, 1, 576) 的批量源里取行
            if args.batch_mode == "random":
                rng = random.Random(args.seed)
                idx = rng.sample(range(n_rows), count)
            else:  # strided：源行下标按固定间隔取（间隔 = step_size+1 行），dst 保持连续
                step = args.step_size + 1
                idx = [i * step for i in range(count)]
            idx_t = torch.tensor(idx, dtype=torch.int64, device="npu")
            batch_base = peer_gva + nbytes                 # 避开 offset 0（单条拷贝回读区）与末尾 64B 握手块
            src_rows = [batch_src.data_ptr() + i * row_b for i in idx]
            dst_rows = [batch_base + j * row_b for j in range(count)]
            sizes = [row_b] * count
            # 1) 正确性：所选行 scatter -> peer 批量区 -> gather 回 scratch -> 与 batch_src.index_select 逐元素相等
            #    （顺带把 peer 批量区填上所选行的数据，保证 G2L 计时读到有效数据）
            scratch = torch.empty(count, *shape[1:], dtype=batch_src.dtype, device="npu")
            assert scratch.data_ptr() % BLOCK_ALIGN == 0, f"scratch not {BLOCK_ALIGN}B aligned"
            assert pool.copy_batch(src_rows, dst_rows, sizes, count, bm.BmCopyType.L2G, flags) == 0, "batch L2G fill"
            torch_npu.npu.synchronize()
            got_rows = [scratch.data_ptr() + j * row_b for j in range(count)]
            assert pool.copy_batch(dst_rows, got_rows, sizes, count, bm.BmCopyType.G2L, flags) == 0, "batch G2L gather"
            torch_npu.npu.synchronize()
            if not torch.equal(scratch, batch_src.index_select(0, idx_t)):
                raise RuntimeError("batch round-trip mismatch")
            # 2) 计时：L2G 从 batch_src 所选行 scatter 到 peer；G2L 从 peer 批量区 gather 回 scratch
            #    （回写目标固定用 scratch，不污染 src/dst/batch_src，避免破坏后续组合的 step1 回读校验）
            if direction == "L2G":
                avg_b = time_batch_copy(pool, copy_type, flags, src_rows, dst_rows, sizes,
                                        count, args.batch_warmup, args.batch_iters)
            else:
                avg_b = time_batch_copy(pool, copy_type, flags, dst_rows, got_rows, sizes,
                                        count, args.batch_warmup, args.batch_iters)
            row["batch_bw_gbs"] = (count * row_b) / avg_b / 1e9
        else:  # broadcast
            src_addrs = [src.data_ptr()] * count
            dst_addrs = [peer_gva + i * nbytes for i in range(count)]
            sizes = [nbytes] * count
            # 先把 peer 的批量区域填上模式（不计时），保证 G2L 读到有效数据
            for _ in range(args.batch_warmup):
                assert pool.copy_batch(src_addrs, dst_addrs, sizes, count,
                                       bm.BmCopyType.L2G, flags) == 0, "L2G batch fill failed"
            if direction == "L2G":
                avg_b = time_batch_copy(pool, copy_type, flags, src_addrs, dst_addrs, sizes,
                                        count, args.batch_warmup, args.batch_iters)
            else:
                g2l_src = [peer_gva + i * nbytes for i in range(count)]
                g2l_dst = [dst.data_ptr()] * count
                avg_b = time_batch_copy(pool, copy_type, flags, g2l_src, g2l_dst, sizes,
                                        count, args.batch_warmup, args.batch_iters)
            row["batch_bw_gbs"] = (count * nbytes) / avg_b / 1e9
    except Exception as e:
        # 单条结果保留，批量降级为 N/A（例如 A5/x86 不支持 BatchCopyExtend）
        row["error"] = f"batch failed: {e}"
    return row


def print_table(rows, shape, nbytes, dtype_name):
    """最大规模的完整对比表：单条时延/带宽 + 批量吞吐 + MTE/SDMA 倍率。"""
    header = (f"shape={shape}, dtype={dtype_name}, nbytes={nbytes} ({nbytes / 1024 / 1024:.2f} MiB)\n"
              f"{'direction':<9} {'engine':<6} {'latency(us)':>12} {'single-BW(GB/s)':>16} {'batch-BW(GB/s)':>16}")
    print(header)
    print("-" * len(header))

    by_dir = {}
    for r in rows:
        lat = f"{r['latency_us']:.2f}" if r["latency_us"] is not None else "N/A"
        bw = f"{r['single_bw_gbs']:.2f}" if r["single_bw_gbs"] is not None else "N/A"
        bb = f"{r['batch_bw_gbs']:.2f}" if r["batch_bw_gbs"] is not None else "N/A"
        note = f"  [{r['error']}]" if r["error"] else ""
        print(f"{r['direction']:<9} {r['engine']:<6} {lat:>12} {bw:>16} {bb:>16}{note}")
        by_dir.setdefault(r["direction"], {})[r["engine"]] = r

    # MTE / SDMA 倍率（仅当两者都有结果时）
    for direction in DIRECTIONS:
        eng = by_dir.get(direction, {})
        s, m = eng.get("SDMA"), eng.get("MTE")
        if s and m and s["latency_us"] and m["latency_us"]:
            print(f">> {direction}: MTE vs SDMA latency {m['latency_us'] / s['latency_us']:.2f}x, "
                  f"single-BW {m['single_bw_gbs'] / s['single_bw_gbs']:.2f}x"
                  + (f", batch-BW {m['batch_bw_gbs'] / s['batch_bw_gbs']:.2f}x"
                     if s["batch_bw_gbs"] and m["batch_bw_gbs"] else ""))


def print_scale_table(rows, dtype_name):
    """scale 扫描总表：每个 shape 下各 方向 x 引擎 的单条时延/带宽。"""
    header = (f"scale sweep: single-copy latency & bandwidth (dtype={dtype_name})\n"
              f"{'shape':<16} {'direction':<9} {'engine':<6} {'latency(us)':>12} {'single-BW(GB/s)':>16}")
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda r: (r["shape"][0], r["direction"], r["engine"])):
        lat = f"{r['latency_us']:.2f}" if r["latency_us"] is not None else "N/A"
        bw = f"{r['single_bw_gbs']:.2f}" if r["single_bw_gbs"] is not None else "N/A"
        note = f"  [{r['error']}]" if r["error"] else ""
        shp = "x".join(str(s) for s in r["shape"])
        print(f"{shp:<16} {r['direction']:<9} {r['engine']:<6} {lat:>12} {bw:>16}{note}")


def send_peer_ready(pool, peer_gva, local_dram_size, timeout):
    """rank1 join 后向 rank0 的 DRAM 池末尾写 READY_MAGIC，完成"peer 已就绪"握手。

    写成功的前提是 rank0 的切片已 import 进**本机**设备页表（rank1 自己的 JoinHandle 里
    hybm_import + hybm_mmap 完成）；在此之前写会返回非 0，循环重试即可。
    关键：写成功同时意味着**两边**的 JoinHandle 都已完成——JoinHandle 里的
    GroupGatherResult barrier（smem_bm_entry.cpp GroupOpBarrier）是 rank0/rank1 一起过的，
    rank1 的 import 若做完，rank0 把 rank1 切片 import 进本机设备页表也必然同步做完。
    所以 rank1 收到成功返回后再对 peer_gva 做拷贝必然可达，不会撞未映射 GVA。
    失败只是 host 侧段校验返回非 0，不毒化设备流；最坏情况是明确的超时报错。
    """
    slot = peer_gva + local_dram_size - DONE_BLOCK_BYTES
    marker = torch.zeros(DONE_BLOCK_BYTES // 4, dtype=torch.int32, device="npu")
    marker[0] = READY_MAGIC
    torch_npu.npu.synchronize()
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        ret = pool.copy(marker.data_ptr(), slot, DONE_BLOCK_BYTES, bm.BmCopyType.L2G, 0)
        if ret != 0:  # SDMA 直访失败时回退 MTE
            ret = pool.copy(marker.data_ptr(), slot, DONE_BLOCK_BYTES,
                            bm.BmCopyType.L2G, COPY_EXTEND_FLAG)
        if ret == 0:
            torch_npu.npu.synchronize()
            print(f"[rank {pool.rank_id}] peer handshake sent (attempt {attempt})")
            return
        time.sleep(0.2)
    raise RuntimeError(
        f"[rank {pool.rank_id}] timeout {timeout}s sending ready handshake to rank0's DRAM "
        f"pool at 0x{slot:x}; check rank0 is running and has joined")


def run_benchmark(pool, args, peer_gva):
    """benchmark 主体：全部单条/批量计时的采集与表格输出。

    双机由 rank1 调用，单机自测（--world_size 1）由 rank0 调用；peer_gva 为对端
    （单机时自己）的 DRAM 池切片基址，本机对它的 L2G/G2L 拷贝就是被计时对象。
    调用前必须保证对端切片已 import 进本机设备页表（调用方负责握手；单机自测天然满足）。
    """
    if args.extend_lib_path:
        os.environ["MEMFABRIC_HYBRID_EXTEND_LIB_PATH"] = args.extend_lib_path
    extend_so = check_extend_lib()
    print(f"[rank {pool.rank_id}] MTE path lib: {extend_so}")

    dtype = DTYPES[args.dtype]
    scales = parse_scales(args.scales)
    largest = scales[-1]
    largest_nbytes = tensor_nbytes(dtype, shape_for(largest))

    # 批量吞吐只在最大 scale 上测；批量区域须给末尾 64B 握手块留位
    batch_row_nbytes = tensor_nbytes(dtype, (1, 576))      # 单条 (1,576) 行的字节数
    if args.batch_mode in ("random", "strided"):
        # random/strided：批量源是独立的 (batch_rows,1,576)，dst 连续放在 peer_gva + largest_nbytes 之后
        batch_region = largest_nbytes + args.batch_size * batch_row_nbytes
        if args.batch_mode == "random":
            assert args.batch_size <= args.batch_rows, "random: batch-size must be <= --batch-rows"
        else:
            last_row = (args.batch_size - 1) * (args.step_size + 1)  # 源行下标从 0 起按步进
            assert last_row < args.batch_rows, \
                f"strided: last source row {last_row} >= batch_rows {args.batch_rows}; " \
                f"reduce --batch-size or --step-size"
    else:
        batch_region = args.batch_size * largest_nbytes
    assert batch_region <= args.local_dram - DONE_BLOCK_BYTES, \
        f"batch region {batch_region}B exceeds local DRAM {args.local_dram}B minus handshake block; " \
        f"reduce --batch-size"
    print(f"[rank {pool.rank_id}] scales={scales}, iters={args.iters}, batch_size={args.batch_size} "
          f"x batch_iters={args.batch_iters} (batch_mode={args.batch_mode}, batch_rows={args.batch_rows}, "
          f"seed={args.seed})")

    # random/strided 批量模式专用的批量源张量 (batch_rows, 1, 576)，独立于 scale 扫描的单条拷贝张量
    batch_src = None
    if args.batch_mode in ("random", "strided"):
        batch_shape = (args.batch_rows, 1, 576)
        batch_src = make_pattern(dtype, batch_shape)
        assert batch_src.data_ptr() % BLOCK_ALIGN == 0, "batch_src not 64B aligned"
        torch_npu.npu.synchronize()
        print(f"[rank {pool.rank_id}] batch source tensor {batch_shape} "
              f"nbytes={tensor_nbytes(dtype, batch_shape) / 1024 / 1024:.2f} MiB, "
              f"ptr=0x{batch_src.data_ptr():x}")

    rows = []
    for scale in scales:
        shape = shape_for(scale)
        nbytes = tensor_nbytes(dtype, shape)
        assert nbytes % BLOCK_ALIGN == 0, f"shape {shape} nbytes {nbytes} not {BLOCK_ALIGN}B aligned"
        src = make_pattern(dtype, shape)
        dst = torch.empty(shape, dtype=dtype, device="npu")
        assert src.data_ptr() % BLOCK_ALIGN == 0, f"src not {BLOCK_ALIGN}B aligned"
        assert dst.data_ptr() % BLOCK_ALIGN == 0, f"dst not {BLOCK_ALIGN}B aligned"
        torch_npu.npu.synchronize()
        print(f"[rank {pool.rank_id}] scale {shape} nbytes={nbytes} "
              f"({nbytes / 1024 / 1024:.2f} MiB), src ptr=0x{src.data_ptr():x}")
        with_batch = (scale == largest)  # 只有最大 scale 附带批量吞吐
        for direction in DIRECTIONS:
            for engine, flags in ENGINES:
                row = bench_one(pool, direction, engine, flags, src, dst, peer_gva, nbytes, args,
                                with_batch, batch_src)
                rows.append(row)
                if row["error"]:
                    print(f"[rank {pool.rank_id}] {shape} {direction}/{engine}: {row['error']}")

    print_scale_table(rows, args.dtype)
    print_table([r for r in rows if r["shape"] == shape_for(largest)], shape_for(largest),
                largest_nbytes, args.dtype)


def run_rank0(args):
    """rank0 = config store host + 纯内存持有方（双机）；单机自测时跑 benchmark。

    双机：建池 join 后循环 sleep，直到 Ctrl+C 退出。不碰 peer GVA、不执行任何拷贝，
    所以不会出现 peer 切片未 import 时直接访问对端地址的竞态/报错。
    单机（--world_size 1）：对端即自己，建池后直接跑 benchmark（无需握手）。
    """
    torch_npu.npu.set_device(args.device)
    pool = BmDramPool(rank=args.rank, world_size=args.world_size, device=args.device,
                      store_url=args.url, nic=args.nic,
                      local_dram_size=args.local_dram, max_dram_size=args.max_dram,
                      start_store=True)
    try:
        pool.initialize()
        pool.create_pool(data_op_type=bm.BmDataOpType.SDMA)
        print(f"[rank {pool.rank_id}] DRAM pool joined (local={args.local_dram / (1 << 30):.2f}GiB, "
              f"max={args.max_dram / (1 << 30):.2f}GiB/rank)")
        if args.world_size == 1:
            peer_gva = pool.peer_gva(0, bm.BmMemType.HOST)   # 单机：对端就是自己
            print(f"[rank {pool.rank_id}] single-machine mode, peer == self (GVA=0x{peer_gva:x})")
            run_benchmark(pool, args, peer_gva)
            return
        print(f"[rank {pool.rank_id}] holding DRAM pool, Ctrl+C to exit ...")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print(f"[rank {pool.rank_id}] Ctrl+C received, exiting")
    finally:
        pool.destroy()
    print(f"[rank {pool.rank_id}] done.")


def run_rank1(args):
    """rank1 = benchmark 驱动/计时方。

    动态组模式下 join() 只等自己，"把 rank0 切片 import 进本机设备页表"是异步的。
    send_peer_ready 循环向 rank0 的 DRAM 池末尾写 READY_MAGIC，写成功即说明 import
    已完成（能写就说明映射好了；且 barrier 相互性保证 rank0 侧 import 也同步完成），
    之后对 peer_gva 的拷贝必然可达，再开始计时。
    """
    torch_npu.npu.set_device(args.device)
    pool = BmDramPool(rank=args.rank, world_size=args.world_size, device=args.device,
                      store_url=args.url, nic=args.nic,
                      local_dram_size=args.local_dram, max_dram_size=args.max_dram,
                      start_store=False)
    try:
        pool.initialize()
        pool.create_pool(data_op_type=bm.BmDataOpType.SDMA)
        print(f"[rank {pool.rank_id}] joined, waiting for rank0's DRAM slice to be importable ...")

        peer_gva = pool.peer_gva(0, bm.BmMemType.HOST)       # rank0 的切片基址
        print(f"[rank {pool.rank_id}] peer_rank=0, remote DRAM GVA=0x{peer_gva:x}")
        send_peer_ready(pool, peer_gva, args.local_dram, args.ready_timeout)

        run_benchmark(pool, args, peer_gva)
    finally:
        pool.destroy()
    print(f"[rank {pool.rank_id}] done.")


def main():
    args = parse_args()
    assert args.world_size in (1, 2) and args.rank < args.world_size, "invalid rank/world_size"
    assert args.local_dram % (2 << 20) == 0 and args.max_dram % (2 << 20) == 0, "DRAM size must be 2MiB aligned"
    assert args.iters > 0 and args.warmup >= 0 and args.batch_size > 0, "invalid iterations/batch-size"
    assert args.batch_rows > 0, "batch-rows must be positive"
    assert args.step_size >= 0, "step-size must be non-negative"
    try:
        if args.rank == 0:
            run_rank0(args)
        else:
            run_rank1(args)
    except Exception as e:
        print(f"[rank {args.rank}] FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
