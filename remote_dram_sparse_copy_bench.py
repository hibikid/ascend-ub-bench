#!/usr/bin/env python3
# coding=utf-8
"""Two-rank remote-DRAM sparse KV-copy benchmark.

rank0 creates a BSND KV tensor on its NPU and stages it in the rank0 HOST DRAM
slice of a MemFabric pool.  rank1 then gathers the same randomly chosen S-axis
tokens through four paths:

* copy_data_batch with SDMA;
* copy_data_batch with MTE / COPY_EXTEND_FLAG;
* the UniDexCopy AI Core kernel with 24 blocks;
* the UniDexCopy AI Core kernel with 48 blocks.

Only rank1 performs timed copies.  A result is checked against a deterministic
source pattern after every (topk, method) measurement.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch_npu  # noqa: F401  # Registers the NPU backend with torch.

import memfabric_hybrid as mf
from memfabric_hybrid import bm


WORLD_SIZE = 2
ONE_GIB = 1 << 30
STORE_PORT = 8573
DEFAULT_NIC_URL = "tcp://127.0.0.1:10005"
DATA_OP_TYPE = bm.BmDataOpType.SDMA
COPY_EXTEND_FLAG = 1 << 1

BATCH = 16
SRC_SEQ = 32 * 1024
NUM_HEADS = 1
HEAD_DIM = 576
TOPKS = (64, 128, 256, 512, 1024, 2048)
READY_BLOCK_BYTES = 64
SOURCE_READY_MAGIC = 0x39A6C1E5

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass
class BenchResult:
    topk: int
    total_bytes: int
    method: str
    avg_us: float
    bandwidth_gbs: float


def _nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return math.prod(shape) * torch.empty((), dtype=dtype).element_size()


def _hex(value: int) -> str:
    return f"0x{int(value):x}"


def _arange_on_device(end: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    return torch.arange(end, dtype=dtype, device=device)


def _stream_ptr(stream) -> int:
    """Extract the raw aclrtStream pointer, matching UniDexCopy's Python wrapper."""
    for attr in ("npu_stream", "stream_ptr", "cuda_stream"):
        if hasattr(stream, attr):
            value = getattr(stream, attr)
            return int(value() if callable(value) else value)
    raise RuntimeError("Unable to extract a raw stream pointer from torch NPU stream")


def _load_unidex_copy_raw():
    """Load the raw UniDexCopy launcher used inside unidex_copy_inplace.

    The high-level wrapper prints its complete launch configuration for every
    call.  The raw API launches the same ``unidex_copy`` kernel without that
    debug output, so host-side printing cannot contaminate the timed region.
    """
    try:
        from sgl_kernel_npu.sparsity_driven_kv_offload import unidex_copy_inplace_raw

        return unidex_copy_inplace_raw
    except Exception as first_exc:
        repo_root = Path(__file__).resolve().parent.parent
        fallback_dir = repo_root / "indexcopy" / "unindexcopykernel"
        if str(fallback_dir) not in sys.path:
            sys.path.insert(0, str(fallback_dir))
        try:
            from unindexcopykernel import unidex_copy_inplace_raw

            return unidex_copy_inplace_raw
        except Exception as second_exc:
            raise ImportError(
                "Failed to import unidex_copy_inplace_raw from "
                "sgl_kernel_npu.sparsity_driven_kv_offload or local fallback "
                f"{fallback_dir}.  The raw launcher is required to benchmark "
                "UniDexCopy without debug-print overhead."
            ) from second_exc


def _make_kv_values(
    token_indices: torch.Tensor,
    batch_indices: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a deterministic, token-identifying BSND payload.

    BF16/FP16 cannot represent every integer in [0, 32768) exactly.  Each row
    therefore stores the low and high bytes of S in separate channels, and its
    batch id in a third channel.  This makes a wrong source row detectable even
    after conversion to a 2-byte floating-point dtype.
    """
    if token_indices.dim() != 2:
        raise ValueError(f"token_indices must be [B, S_or_K], got {tuple(token_indices.shape)}")
    if batch_indices.numel() != token_indices.shape[0]:
        raise ValueError("batch_indices and token_indices disagree on batch size")

    device = str(token_indices.device)
    dim_slot = _arange_on_device(HEAD_DIM, torch.int32, device).remainder(4).reshape(1, 1, 1, HEAD_DIM)
    token = token_indices.to(torch.int64)
    low = token.remainder(256).to(torch.float32).reshape(*token.shape, 1, 1)
    high = torch.div(token, 256, rounding_mode="floor").remainder(256).to(torch.float32)
    high = high.reshape(*token.shape, 1, 1)
    batch = batch_indices.to(torch.float32).reshape(-1, 1, 1, 1)
    mixed = (low * 17 + high * 13 + batch * 7).remainder(256)

    values = torch.where(
        dim_slot == 0,
        low,
        torch.where(dim_slot == 1, high, torch.where(dim_slot == 2, batch, mixed)),
    )
    return values.to(dtype).contiguous()


def _build_rank0_kv(device: str, dtype: torch.dtype) -> torch.Tensor:
    """Create the full (B, S, N, D) source without a full-size FP32 temporary."""
    src_shape = (BATCH, SRC_SEQ, NUM_HEADS, HEAD_DIM)
    src = torch.empty(src_shape, dtype=dtype, device=device).contiguous()
    token_row = _arange_on_device(SRC_SEQ, torch.long, device).reshape(1, SRC_SEQ)
    for batch_id in range(BATCH):
        batch = torch.tensor([batch_id], dtype=torch.long, device=device)
        src[batch_id:batch_id + 1].copy_(_make_kv_values(token_row, batch, dtype))
    torch.npu.synchronize()
    return src


def _build_topk_indices(topk: int, seed: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one unique random Top-K list per batch, reproducibly on CPU."""
    generator = torch.Generator()
    generator.manual_seed(seed + topk)
    indices_cpu = torch.stack(
        [torch.randperm(SRC_SEQ, dtype=torch.long, generator=generator)[:topk] for _ in range(BATCH)]
    ).contiguous()
    return indices_cpu, indices_cpu.to(device).contiguous()


def _build_unidex_indices(topk_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Translate [B, K] token positions into the flattened UniDex row mapping."""
    topk = topk_indices.shape[1]
    device = str(topk_indices.device)
    batch_offsets = _arange_on_device(BATCH, torch.long, device).reshape(BATCH, 1) * SRC_SEQ
    src_index = (batch_offsets + topk_indices).reshape(-1).contiguous()
    dst_index = _arange_on_device(BATCH * topk, torch.long, device).contiguous()
    valid_mask = torch.ones(BATCH * topk, dtype=torch.bool, device=device)
    return src_index, dst_index, valid_mask


def _build_batch_addresses(
    peer_gva: int,
    dst: torch.Tensor,
    indices_cpu: torch.Tensor,
    row_bytes: int,
) -> tuple[list[int], list[int], list[int]]:
    """Build GVA source and contiguous NPU destination addresses for copy_data_batch."""
    count = indices_cpu.numel()
    topk = indices_cpu.shape[1]
    flat_tokens = indices_cpu.reshape(-1).tolist()
    src_addrs = [
        peer_gva + (((row_id // topk) * SRC_SEQ + token) * row_bytes)
        for row_id, token in enumerate(flat_tokens)
    ]
    dst_addrs = [dst.data_ptr() + row_id * row_bytes for row_id in range(count)]
    return src_addrs, dst_addrs, [row_bytes] * count


def _assert_equal(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if torch.equal(actual, expected):
        return

    mismatch = actual.ne(expected)
    mismatch_count = int(mismatch.sum().item())
    first = mismatch.nonzero(as_tuple=False)[:5].cpu().tolist()
    print(f"{label}: mismatch elements={mismatch_count}", flush=True)
    for index in first:
        index_tuple = tuple(index)
        print(
            f"  index={index_tuple}, actual={actual[index_tuple].item()}, "
            f"expected={expected[index_tuple].item()}",
            flush=True,
        )
    raise AssertionError(f"{label}: sparse-copy verification failed")


def _measure(
    launch: Callable[[], None],
    warmup: int,
    iters: int,
    total_bytes: int,
) -> tuple[float, float]:
    """Use microbench_sdma_mte.py's wall-clock style for a synchronous result."""
    for _ in range(warmup):
        launch()
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        launch()
    torch.npu.synchronize()
    avg_s = (time.perf_counter() - t0) / iters
    return avg_s * 1e6, total_bytes / avg_s / 1e9


def _initialize_bm(args: argparse.Namespace, rank: int) -> None:
    cfg = bm.BmConfig()
    cfg.auto_ranking = False
    cfg.rank_id = rank
    cfg.start_store = rank == 0
    cfg.set_nic(args.nic_url)
    store_url = f"tcp://{args.head_ip}:{args.store_port}"
    assert bm.initialize(store_url, WORLD_SIZE, args.device_id, cfg) == 0, "bm.initialize failed"


def _create_handle(args: argparse.Namespace, rank: int):
    store_url = f"tcp://{args.head_ip}:{args.store_port}"
    handle = bm.create2(
        id=args.pool_id,
        local_dram_size=args.pool_bytes,
        max_dram_size=args.pool_bytes,
        local_hbm_size=0,
        max_hbm_size=0,
        data_op_type=DATA_OP_TYPE,
    )
    print(f"[rank {rank}] joining pool via {store_url}", flush=True)
    assert handle is not None, "bm.create2 failed"
    assert handle.join() == 0, "pool join failed"
    return handle


def _cleanup(handle, bm_inited: bool, joined: bool) -> None:
    try:
        if handle is not None:
            if joined:
                try:
                    handle.leave()
                except Exception as exc:
                    print(f"leave failed during cleanup: {exc}", flush=True)
            try:
                handle.destroy()
            except Exception as exc:
                print(f"destroy failed during cleanup: {exc}", flush=True)
    finally:
        if bm_inited:
            bm.uninitialize()
        mf.uninitialize()


def _write_source_marker(
    handle,
    source_gva: int,
    pool_bytes: int,
    device: str,
    marker_value: int,
) -> None:
    """Write a 64-byte source-state marker in rank0's reserved pool tail."""
    marker = torch.zeros(READY_BLOCK_BYTES // 4, dtype=torch.int32, device=device)
    marker[0] = marker_value
    torch.npu.synchronize()
    slot = source_gva + pool_bytes - READY_BLOCK_BYTES
    ret = handle.copy_data(
        marker.data_ptr(), slot, READY_BLOCK_BYTES, bm.BmCopyType.L2GH, COPY_EXTEND_FLAG
    )
    assert ret == 0, f"source-ready marker write failed, ret={ret}, err={mf.get_last_err_msg()}"
    torch.npu.synchronize()


def _wait_for_source_ready(
    handle,
    peer_gva: int,
    pool_bytes: int,
    device: str,
    timeout: float,
) -> None:
    """Wait until rank0 both maps and fills its remote DRAM source slice."""
    marker = torch.empty(READY_BLOCK_BYTES // 4, dtype=torch.int32, device=device)
    slot = peer_gva + pool_bytes - READY_BLOCK_BYTES
    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        ret = handle.copy_data(slot, marker.data_ptr(), READY_BLOCK_BYTES, bm.BmCopyType.G2L, 0)
        if ret != 0:
            ret = handle.copy_data(
                slot, marker.data_ptr(), READY_BLOCK_BYTES, bm.BmCopyType.G2L, COPY_EXTEND_FLAG
            )
        if ret == 0:
            torch.npu.synchronize()
            if int(marker[0].item()) == SOURCE_READY_MAGIC:
                print(f"[rank 1] rank0 source ready after {attempt} poll(s)", flush=True)
                return
        time.sleep(0.2)
    raise TimeoutError(
        f"timed out after {timeout}s waiting for rank0 source-ready marker at {_hex(slot)}"
    )


def _run_rank0(args: argparse.Namespace, dtype: torch.dtype) -> None:
    src_shape = (BATCH, SRC_SEQ, NUM_HEADS, HEAD_DIM)
    source_bytes = _nbytes(src_shape, dtype)
    if source_bytes + READY_BLOCK_BYTES > args.pool_bytes:
        raise RuntimeError(
            f"pool_bytes={args.pool_bytes} cannot hold source={source_bytes} plus ready marker"
        )

    mf.set_log_level(args.log_level)
    assert mf.initialize() == 0, "mf.initialize failed"
    handle = None
    bm_inited = False
    joined = False
    try:
        _initialize_bm(args, rank=0)
        bm_inited = True
        handle = _create_handle(args, rank=0)
        joined = True

        source_gva = handle.peer_rank_ptr(0, bm.BmMemType.HOST)
        assert source_gva != 0, "peer_rank_ptr(rank=0, HOST) returned 0"
        # A fresh pool is not required to be zeroed.  Clear the reserved tail
        # before rank1 is allowed to observe a ready value from an old run.
        _write_source_marker(handle, source_gva, args.pool_bytes, args.device, marker_value=0)
        print(
            f"[rank 0] building source shape={src_shape}, dtype={args.dtype}, "
            f"bytes={source_bytes} ({source_bytes / 2**20:.2f} MiB)",
            flush=True,
        )
        src = _build_rank0_kv(args.device, dtype)

        ret = handle.copy_data(
            src.data_ptr(), source_gva, source_bytes, bm.BmCopyType.L2GH, COPY_EXTEND_FLAG
        )
        assert ret == 0, f"rank0 source offload failed, ret={ret}, err={mf.get_last_err_msg()}"
        torch.npu.synchronize()
        _write_source_marker(
            handle, source_gva, args.pool_bytes, args.device, marker_value=SOURCE_READY_MAGIC
        )
        print(
            f"[rank 0] source staged in DRAM GVA={_hex(source_gva)}; "
            "holding pool for rank1 (Ctrl+C to exit)",
            flush=True,
        )

        if args.rank0_hold_sec > 0:
            time.sleep(args.rank0_hold_sec)
        else:
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                print("[rank 0] interrupted", flush=True)
    finally:
        _cleanup(handle, bm_inited, joined)
    print("[rank 0] cleanup done", flush=True)


def _run_rank1(args: argparse.Namespace, dtype: torch.dtype) -> None:
    unidex_copy_raw = _load_unidex_copy_raw()
    src_shape = (BATCH, SRC_SEQ, NUM_HEADS, HEAD_DIM)
    row_bytes = _nbytes((1, NUM_HEADS, HEAD_DIM), dtype)
    src_rows = BATCH * SRC_SEQ

    mf.set_log_level(args.log_level)
    assert mf.initialize() == 0, "mf.initialize failed"
    handle = None
    bm_inited = False
    joined = False
    results: list[BenchResult] = []
    try:
        _initialize_bm(args, rank=1)
        bm_inited = True
        handle = _create_handle(args, rank=1)
        joined = True

        peer_gva = handle.peer_rank_ptr(0, bm.BmMemType.HOST)
        assert peer_gva != 0, "peer_rank_ptr(rank=0, HOST) returned 0"
        print(f"[rank 1] rank0 source GVA={_hex(peer_gva)}", flush=True)
        _wait_for_source_ready(handle, peer_gva, args.pool_bytes, args.device, args.ready_timeout)

        src_lva = handle.gva_to_va(peer_gva, bm.BmMemType.LOCAL_DEVICE)
        assert src_lva != 0, f"gva_to_va({_hex(peer_gva)}, LOCAL_DEVICE) failed"
        print(f"[rank 1] UniDexCopy source LVA={_hex(src_lva)}", flush=True)
        print(
            f"[rank 1] source={src_shape}, dtype={args.dtype}, row_bytes={row_bytes}, "
            f"warmup={args.warmup}, iters={args.iters}",
            flush=True,
        )

        stream_ptr = _stream_ptr(torch.npu.current_stream())
        batch_ids = _arange_on_device(BATCH, torch.long, args.device)
        for topk in args.topks:
            total_rows = BATCH * topk
            total_bytes = total_rows * row_bytes
            dst_shape = (BATCH, topk, NUM_HEADS, HEAD_DIM)
            indices_cpu, topk_indices = _build_topk_indices(topk, args.seed, args.device)
            expected = _make_kv_values(topk_indices, batch_ids, dtype)
            src_index, dst_index, valid_mask = _build_unidex_indices(topk_indices)

            print(
                f"[rank 1] topk={topk}, dst={dst_shape}, bytes={total_bytes} "
                f"({total_bytes / 2**20:.3f} MiB)",
                flush=True,
            )

            for method, flags in (("copy_batch_sdma", 0), ("copy_batch_mte", COPY_EXTEND_FLAG)):
                dst = torch.empty(dst_shape, dtype=dtype, device=args.device).contiguous()
                src_addrs, dst_addrs, sizes = _build_batch_addresses(
                    peer_gva, dst, indices_cpu, row_bytes
                )

                def launch_batch(
                    src_addrs=src_addrs,
                    dst_addrs=dst_addrs,
                    sizes=sizes,
                    flags=flags,
                ) -> None:
                    ret = handle.copy_data_batch(
                        src_addrs,
                        dst_addrs,
                        sizes,
                        total_rows,
                        bm.BmCopyType.G2L,
                        flags,
                    )
                    assert ret == 0, (
                        f"{method} failed, ret={ret}, err={mf.get_last_err_msg()}"
                    )

                avg_us, bandwidth_gbs = _measure(
                    launch_batch, args.warmup, args.iters, total_bytes
                )
                _assert_equal(dst, expected, f"topk={topk} {method}")
                results.append(BenchResult(topk, total_bytes, method, avg_us, bandwidth_gbs))

            for block_dim in (24, 48):
                method = f"unidex_copy_{block_dim}core"
                dst = torch.empty(dst_shape, dtype=dtype, device=args.device).contiguous()

                def launch_unidex(dst=dst, block_dim=block_dim) -> None:
                    unidex_copy_raw(
                        src_ptr=src_lva,
                        dst_ptr=dst.data_ptr(),
                        src_index_ptr=src_index.data_ptr(),
                        dst_index_ptr=dst_index.data_ptr(),
                        valid_mask_ptr=valid_mask.data_ptr(),
                        src_rows=src_rows,
                        dst_rows=total_rows,
                        block_bytes=row_bytes,
                        max_copy=total_rows,
                        stream_ptr=stream_ptr,
                        block_dim=block_dim,
                        sync=False,
                    )

                avg_us, bandwidth_gbs = _measure(
                    launch_unidex, args.warmup, args.iters, total_bytes
                )
                _assert_equal(dst, expected, f"topk={topk} {method}")
                results.append(BenchResult(topk, total_bytes, method, avg_us, bandwidth_gbs))
    finally:
        _cleanup(handle, bm_inited, joined)

    _print_results(results, args.dtype, args.iters)
    print("[rank 1] all sparse-copy measurements and verifications passed", flush=True)


def _print_results(results: list[BenchResult], dtype_name: str, iters: int) -> None:
    print(
        "\nremote DRAM sparse KV copy benchmark "
        f"(dtype={dtype_name}, timed iterations={iters}, all rows verified)",
        flush=True,
    )
    print(f"{'topk':>6} {'bytes(MiB)':>12} {'method':<24} {'avg(us)':>12} {'BW(GB/s)':>12}", flush=True)
    print("-" * 74, flush=True)
    for row in results:
        print(
            f"{row.topk:>6} {row.total_bytes / 2**20:>12.3f} {row.method:<24} "
            f"{row.avg_us:>12.2f} {row.bandwidth_gbs:>12.2f}",
            flush=True,
        )


def _parse_topks(value: str) -> tuple[int, ...]:
    topks = tuple(sorted(set(int(item.strip()) for item in value.split(",") if item.strip())))
    if not topks:
        raise argparse.ArgumentTypeError("--topks cannot be empty")
    if any(topk <= 0 or topk > SRC_SEQ for topk in topks):
        raise argparse.ArgumentTypeError(f"--topks values must be in [1, {SRC_SEQ}]")
    return topks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-rank remote DRAM sparse KV benchmark: batch-copy SDMA/MTE vs UniDexCopy."
    )
    parser.add_argument("rank", type=int, choices=(0, 1))
    parser.add_argument("head_ip", nargs="?", default="")
    parser.add_argument("--store-port", type=int, default=STORE_PORT)
    parser.add_argument("--device-id", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    parser.add_argument("--nic-url", default=DEFAULT_NIC_URL)
    parser.add_argument("--pool-id", type=int, default=0)
    parser.add_argument("--pool-bytes", type=int, default=ONE_GIB)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument("--topks", type=_parse_topks, default=TOPKS)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    parser.add_argument("--rank0-hold-sec", type=float, default=0.0)
    parser.add_argument("--log-level", type=int, default=3)
    args = parser.parse_args()

    if not args.head_ip:
        args.head_ip = input("Head node IP: ").strip()
    if not args.head_ip:
        raise RuntimeError("head_ip is required")
    if args.pool_bytes % (2 << 20) != 0:
        raise RuntimeError("--pool-bytes must be 2 MiB aligned")
    if args.warmup < 0 or args.iters <= 0:
        raise RuntimeError("--warmup must be >= 0 and --iters must be > 0")
    if args.ready_timeout <= 0:
        raise RuntimeError("--ready-timeout must be > 0")

    args.device = "npu"
    return args


def main() -> None:
    args = _parse_args()
    torch.npu.set_device(args.device_id)
    dtype = DTYPES[args.dtype]
    if args.rank == 0:
        _run_rank0(args, dtype)
    else:
        _run_rank1(args, dtype)


if __name__ == "__main__":
    main()
