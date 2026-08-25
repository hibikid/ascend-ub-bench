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
      --batch-mode broadcast 则把整张最大 scale 的 src 广播到 peer DRAM 的 B 个连续切片 /
      从 B 个连续切片收进 dst）

角色：rank0 为驱动方，对 rank1 的 DRAM 池 GVA 执行全部计时；rank1 join 后通过 G2L 拷贝引擎
     轮询完成标记并校验落盘数据（不 CPU 直读 GVA——910C/GVA_V4 下 DRAM 池 LVA 是设备侧地址，
     memmove 会段错误），确认后退出。

    动态组模式下 rank0 的 join() 只等自己就返回，rank1 的 DRAM 切片 import 是异步的；
    rank0 开测前先 wait_peer_ready 握手（小 L2G+G2L 回读直到 peer GVA 可达），
    避免 rank1 起得晚时第一笔拷贝撞上未映射的 GVA、或 rank1 未启动时糊里糊涂地挂。

运行（两台机器各一个终端）：
  # rank0（默认启动 config store）：
  python3 microbench_sdma_mte.py --rank 0 --url tcp://<rank0-ip>:8570 --nic <rank0-数据面ip>
  # rank1：
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
DONE_MAGIC = 0x5E7ABE5E            # 完成标记，int32 可表示（< 2^31）
DONE_BLOCK_BYTES = 64              # 完成标记块大小，保证 64B 对齐

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
    """确认 MTE 路径的编译产物 libmf_hybm_copy_extend.so 可用（rank0 才需要）。"""
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
    ap.add_argument("--rank", type=int, default=0, help="global rank id: 0(driver) or 1(verifier)")
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
    ap.add_argument("--batch-mode", default="random", choices=["random", "broadcast"],
                    help="batch copy pattern: 'random' (default) scatters B random (1,576) rows of the "
                         "dedicated batch-source tensor into the peer batch region; 'broadcast' copies the "
                         "largest-scale tensor to B consecutive slices")
    ap.add_argument("--batch-rows", type=int, default=16384,
                    help="first dim of the dedicated batch-source tensor (batch_rows, 1, 576); random mode "
                         "samples batch_size (1,576) rows from it")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for --batch-mode random row selection")
    ap.add_argument("--scales", default="64,128,256,512,1024,2048",
                    help="comma-separated first-dim values for the single-copy scale sweep "
                         "(shape=(N,1,576)); batch throughput is measured on the largest scale only")
    ap.add_argument("--poll-timeout", type=int, default=120,
                    help="rank1 timeout (s) waiting for the done marker from rank0")
    ap.add_argument("--ready-timeout", type=int, default=120,
                    help="rank0 timeout (s) waiting for the peer DRAM GVA to become reachable "
                         "(rank1 joined + imported on this device) before benchmarking")
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
    """解析 --scales "64,128,..."，去重升序，保证最后一档是最大规模（批量吞吐/完成标记用它）。"""
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

    batch_src：random 批量模式专用的批量源张量 (batch_rows, 1, 576)，独立于 scale 扫描的 src。
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
    #    - broadcast：把整张 src 广播到 peer DRAM 的 B 个连续切片 / 从 B 个连续切片收进 dst
    try:
        count = args.batch_size
        row_b = nbytes // shape[0]            # 单条 (1,576) 行的字节数 = element_size * 576（src 与 batch_src 同 dtype，行字节相同）
        if args.batch_mode == "random":
            assert batch_src is not None, "random batch mode requires batch_src"
            n_rows = batch_src.shape[0]                   # 从 (batch_rows, 1, 576) 的批量源里随机挑
            rng = random.Random(args.seed)
            idx = rng.sample(range(n_rows), count)
            idx_t = torch.tensor(idx, dtype=torch.int64, device="npu")
            batch_base = peer_gva + nbytes                 # 避开 offset 0（rank1 校验区）与末尾 done marker
            src_rows = [batch_src.data_ptr() + i * row_b for i in idx]
            dst_rows = [batch_base + j * row_b for j in range(count)]
            sizes = [row_b] * count
            # 1) 正确性：随机行 scatter -> peer 批量区 -> gather 回 scratch -> 与 batch_src.index_select 逐元素相等
            #    （顺带把 peer 批量区填上随机行的数据，保证 G2L 计时读到有效数据）
            scratch = torch.empty(count, *shape[1:], dtype=batch_src.dtype, device="npu")
            assert scratch.data_ptr() % BLOCK_ALIGN == 0, f"scratch not {BLOCK_ALIGN}B aligned"
            assert pool.copy_batch(src_rows, dst_rows, sizes, count, bm.BmCopyType.L2G, flags) == 0, "random L2G fill"
            torch_npu.npu.synchronize()
            got_rows = [scratch.data_ptr() + j * row_b for j in range(count)]
            assert pool.copy_batch(dst_rows, got_rows, sizes, count, bm.BmCopyType.G2L, flags) == 0, "random G2L gather"
            torch_npu.npu.synchronize()
            if not torch.equal(scratch, batch_src.index_select(0, idx_t)):
                raise RuntimeError("random-row batch round-trip mismatch")
            # 2) 计时：L2G 从 batch_src 随机行 scatter 到 peer；G2L 从 peer 批量区 gather 回 scratch
            #    （回写目标固定用 scratch，不污染 src/dst/batch_src，避免破坏后续组合的 step1 回读校验与 rank1 校验）
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


def write_done_marker(pool, peer_gva, local_dram_size):
    """把完成标记写到对端 DRAM 池末尾（64B 对齐块），rank1 轮询该块开头 4 字节。"""
    offset = local_dram_size - DONE_BLOCK_BYTES
    marker = torch.zeros(DONE_BLOCK_BYTES // 4, dtype=torch.int32, device="npu")
    marker[0] = DONE_MAGIC
    torch_npu.npu.synchronize()
    ret = pool.copy(marker.data_ptr(), peer_gva + offset, DONE_BLOCK_BYTES, bm.BmCopyType.L2G, 0)
    if ret != 0:  # SDMA 直访失败时回退 MTE，避免 rank1 干等超时
        print(f"[rank {pool.rank_id}] WARN: SDMA marker write failed ({ret}), trying MTE")
        ret = pool.copy(marker.data_ptr(), peer_gva + offset, DONE_BLOCK_BYTES,
                        bm.BmCopyType.L2G, COPY_EXTEND_FLAG)
    assert ret == 0, f"write done marker failed: {ret}"
    print(f"[rank {pool.rank_id}] done marker written at GVA 0x{peer_gva + offset:x}")


def wait_done_marker(pool, my_gva, local_dram_size, timeout):
    """rank1 用 G2L 拷贝引擎轮询自己 DRAM 池末尾的完成标记。

    不通过 gva_to_va + CPU memmove 直读本机 DRAM 池：A3/Ascend 910C + GVA_V4 下
    DRAM 池是 HybmVmmBasedSegment，其 LVA 由 HalMemAddressReserve 保留在设备侧地址空间，
    CPU 进程不可直读（memmove 会段错误）。改为 G2L 把标记块拷进 NPU buffer 再比对，
    全程走设备通路，与段类型无关，两种 segment 都能工作。
    """
    offset = local_dram_size - DONE_BLOCK_BYTES
    marker_gva = my_gva + offset
    buf = torch.zeros(DONE_BLOCK_BYTES // 4, dtype=torch.int32, device="npu")
    deadline = time.time() + timeout
    while time.time() < deadline:
        ret = pool.copy(marker_gva, buf.data_ptr(), DONE_BLOCK_BYTES, bm.BmCopyType.G2L, 0)
        assert ret == 0, f"marker poll copy(G2L) failed: {ret}"
        torch_npu.npu.synchronize()
        if buf[0].item() == DONE_MAGIC:
            print(f"[rank {pool.rank_id}] done marker received, benchmark on peer finished")
            return
        time.sleep(0.2)
    raise RuntimeError(f"timeout {timeout}s waiting for done marker (offset 0x{offset:x})")


def verify_peer_data(pool, my_gva, dtype, nbytes, shape=SHAPE):
    """rank1 校验 rank0 收尾时留在自己 DRAM 池开头的 src 模式（最大 scale）。

    G2L 把本机 DRAM 池 GVA 拷回 NPU 再逐元素比对（同 wait_done_marker 的原因，
    不 CPU 直读 GVA——VMM segment 下返回的是设备侧 LVA，memmove 会段错误）。
    """
    got = torch.empty(shape, dtype=dtype, device="npu")
    ret = pool.copy(my_gva, got.data_ptr(), nbytes, bm.BmCopyType.G2L, 0)
    if ret != 0:
        return False
    torch_npu.npu.synchronize()
    return torch.equal(got, make_pattern(dtype, shape))


def wait_peer_ready(pool, peer_gva, timeout):
    """rank0 join 后等待 peer 的 DRAM GVA 在本机设备上可达（rank1 join+import 完成）。

    BM 组引擎是**动态成员模式**（smem_net_group_engine.cpp：groupSize 按实际 join 数增长，
    见 UpdateBitmapFromRank / GroupJoin），rank0 的 join() 只等自己就返回，rank1 的 DRAM 切片
    是异步 hybm_import 进本机设备页表的。若 rank1 起得晚，rank0 一 join 就开测，第一笔
    copy 会撞上尚未映射的 GVA（拷贝失败/“有时跑会出错”）。这里循环做一次小的
    L2G 写 + G2L 读回（先后试 SDMA/MTE 两条路径），任一路径成功即说明 GVA 已映射，
    再开始计时；超时则明确报 peer 未就绪，而不是第一笔拷贝才炸。
    """
    probe = (torch.arange(256, dtype=torch.int64, device="npu") % 97).to(torch.int32)
    readback = torch.empty(256, dtype=torch.int32, device="npu")
    size = probe.nbytes  # 256 * 4B = 1024 B，64B 整数倍；data_ptr 512B 对齐
    assert probe.data_ptr() % BLOCK_ALIGN == 0 and readback.data_ptr() % BLOCK_ALIGN == 0
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        for flags in (0, COPY_EXTEND_FLAG):
            try:
                if pool.copy(probe.data_ptr(), peer_gva, size, bm.BmCopyType.L2G, flags) != 0:
                    continue
                torch_npu.npu.synchronize()
                if pool.copy(peer_gva, readback.data_ptr(), size, bm.BmCopyType.G2L, flags) != 0:
                    continue
                torch_npu.npu.synchronize()
                if torch.equal(readback, probe):
                    print(f"[rank {pool.rank_id}] peer GVA 0x{peer_gva:x} reachable "
                          f"(attempt {attempt}, flags={flags})")
                    return
            except Exception:
                continue
        time.sleep(0.2)
    raise RuntimeError(
        f"[rank {pool.rank_id}] timeout {timeout}s waiting for peer GVA 0x{peer_gva:x} "
        f"to become reachable; check rank1 is running and has joined "
        f"(its DRAM slice must be imported on this device)")


def run_rank0(args):
    if args.extend_lib_path:
        os.environ["MEMFABRIC_HYBRID_EXTEND_LIB_PATH"] = args.extend_lib_path
    extend_so = check_extend_lib()
    print(f"[rank {args.rank}] MTE path lib: {extend_so}")

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

        peer_rank = (pool.rank_id + 1) % args.world_size
        peer_gva = pool.peer_gva(peer_rank, bm.BmMemType.HOST)
        print(f"[rank {pool.rank_id}] peer_rank={peer_rank}, remote DRAM GVA=0x{peer_gva:x}")

        # 动态组模式下 join() 只等自己就返回，rank1 的 DRAM 切片 import 是异步的；
        # 先握手等 peer GVA 可达再开测，避免第一笔拷贝撞上尚未映射的 GVA
        # （rank1 起得晚时“有时跑会出错”、rank1 未启动时在这里就明确超时报错）
        if args.world_size > 1:
            wait_peer_ready(pool, peer_gva, args.ready_timeout)

        dtype = DTYPES[args.dtype]
        scales = parse_scales(args.scales)
        largest = scales[-1]
        largest_nbytes = tensor_nbytes(dtype, shape_for(largest))

        # 批量吞吐只在最大 scale 上测；批量区域须给末尾 64B 完成标记留位
        batch_row_nbytes = tensor_nbytes(dtype, (1, 576))      # 单条 (1,576) 行的字节数
        if args.batch_mode == "random":
            # 随机行模式：从独立的 (batch_rows, 1, 576) 批量源张量里挑 batch_size 行，
            # 批量区放在 peer_gva + largest_nbytes 之后（不碰 offset 0 的 rank1 校验区）
            batch_region = largest_nbytes + args.batch_size * batch_row_nbytes
            assert args.batch_size <= args.batch_rows, "batch-size must be <= --batch-rows for random-row mode"
        else:
            batch_region = args.batch_size * largest_nbytes
        assert batch_region <= args.local_dram - DONE_BLOCK_BYTES, \
            f"batch region {batch_region}B exceeds local DRAM {args.local_dram}B minus done-marker block; " \
            f"reduce --batch-size"
        print(f"[rank {pool.rank_id}] scales={scales}, iters={args.iters}, batch_size={args.batch_size} "
              f"x batch_iters={args.batch_iters} (batch_mode={args.batch_mode}, batch_rows={args.batch_rows}, "
              f"seed={args.seed})")

        # 随机行批量模式专用的批量源张量 (batch_rows, 1, 576)，独立于 scale 扫描的单条拷贝张量
        batch_src = None
        if args.batch_mode == "random":
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

        if args.world_size > 1:
            write_done_marker(pool, peer_gva, args.local_dram)

        print_scale_table(rows, args.dtype)
        print_table([r for r in rows if r["shape"] == shape_for(largest)], shape_for(largest),
                    largest_nbytes, args.dtype)
    finally:
        pool.destroy()
    print(f"[rank {pool.rank_id}] done.")


def run_rank1(args):
    torch_npu.npu.set_device(args.device)
    pool = BmDramPool(rank=args.rank, world_size=args.world_size, device=args.device,
                      store_url=args.url, nic=args.nic,
                      local_dram_size=args.local_dram, max_dram_size=args.max_dram,
                      start_store=False)
    try:
        pool.initialize()
        pool.create_pool(data_op_type=bm.BmDataOpType.SDMA)
        print(f"[rank {pool.rank_id}] joined, waiting for peer benchmark done marker ...")

        my_gva = pool.peer_gva(pool.rank_id, bm.BmMemType.HOST)
        wait_done_marker(pool, my_gva, args.local_dram, args.poll_timeout)

        # 校验 rank0 收尾时留在本机 DRAM 池开头的最大 scale 的 src 模式
        # （G2L 拷回 NPU 比对，双机 --dtype/--scales 需一致）
        scales = parse_scales(args.scales)
        shape = shape_for(scales[-1])
        dtype = DTYPES[args.dtype]
        nbytes = tensor_nbytes(dtype, shape)
        if verify_peer_data(pool, my_gva, dtype, nbytes, shape):
            print(f"[rank {pool.rank_id}] peer data verify OK: rank0's largest-scale tensor landed in my DRAM pool")
        else:
            print(f"[rank {pool.rank_id}] WARN: peer data verify MISMATCH")
    finally:
        pool.destroy()
    print(f"[rank {pool.rank_id}] done.")


def main():
    args = parse_args()
    assert args.world_size in (1, 2) and args.rank < args.world_size, "invalid rank/world_size"
    assert args.local_dram % (2 << 20) == 0 and args.max_dram % (2 << 20) == 0, "DRAM size must be 2MiB aligned"
    assert args.iters > 0 and args.warmup >= 0 and args.batch_size > 0, "invalid iterations/batch-size"
    assert args.batch_rows > 0, "batch-rows must be positive"
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
