#!/usr/bin/env python3
# coding=utf-8
"""Two-rank MemFabric DRAM pool sparse-copy test with UniDexCopy."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

import memfabric_hybrid as mf
from memfabric_hybrid import bm


STORE_PORT = 8572
WORLD_SIZE = 2
ONE_GIB = 1 << 30
DATA_OP_TYPE = bm.BmDataOpType.SDMA
DEFAULT_NIC_URL = "tcp://127.0.0.1:10005"
POST_JOIN_RANK1_SLEEP_SEC = 3.0
COPY_EXTEND_FLAG = 1 << 1

BATCH = 16
SRC_SEQ = 16 * 1024
TOPK = 2048
NUM_HEADS = 1
HEAD_DIM = 576
DTYPE = torch.bfloat16

SRC_SHAPE = (BATCH, SRC_SEQ, NUM_HEADS, HEAD_DIM)
DST_SHAPE = (BATCH, TOPK, NUM_HEADS, HEAD_DIM)
SRC_ADDRESS_NDIMS = 2
DST_ADDRESS_NDIMS = 2


def _nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return math.prod(shape) * torch.empty((), dtype=dtype).element_size()


def _hex(value: int) -> str:
    return f"0x{int(value):x}"


def _arange_on_device(end: int, dtype: torch.dtype, device: str | torch.device) -> torch.Tensor:
    return torch.arange(end, dtype=dtype).to(device)


def _load_unidex_copy_inplace():
    try:
        from sgl_kernel_npu.sparsity_driven_kv_offload import unidex_copy_inplace

        return unidex_copy_inplace
    except Exception as first_exc:
        repo_root = Path(__file__).resolve().parents[1]
        fallback_dir = repo_root.parent / "indexcopy" / "unindexcopykernel"
        if str(fallback_dir) not in sys.path:
            sys.path.insert(0, str(fallback_dir))
        try:
            from unindexcopykernel import unidex_copy_inplace

            return unidex_copy_inplace
        except Exception as second_exc:
            raise ImportError(
                "Failed to import unidex_copy_inplace from "
                "sgl_kernel_npu.sparsity_driven_kv_offload or local "
                f"fallback {fallback_dir}"
            ) from second_exc


def _build_rank0_kv(device: str) -> torch.Tensor:
    src = torch.empty(SRC_SHAPE, dtype=DTYPE, device=device).contiguous()
    seq = _arange_on_device(SRC_SEQ, torch.float32, device).reshape(SRC_SEQ, 1, 1)
    dim = _arange_on_device(HEAD_DIM, torch.float32, device).remainder(8).reshape(1, 1, HEAD_DIM)

    with torch.no_grad():
        for batch_id in range(BATCH):
            row = (seq + batch_id * 17).remainder(64) + dim
            src[batch_id].copy_(row.to(DTYPE))

    torch.npu.synchronize()
    return src


def _build_topk_indices(device: str, seed: int, unique_topk: bool) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    if not unique_topk:
        topk_cpu = torch.randint(0, SRC_SEQ, (BATCH, TOPK), dtype=torch.long, generator=generator)
        return topk_cpu.to(device).contiguous()

    rows = [
        torch.randperm(SRC_SEQ, dtype=torch.long, generator=generator)[:TOPK]
        for _ in range(BATCH)
    ]
    return torch.stack(rows, dim=0).to(device).contiguous()


def _build_unidex_indices(topk_indices: torch.Tensor):
    device = topk_indices.device
    batch_offsets = _arange_on_device(BATCH, torch.long, device).reshape(BATCH, 1) * SRC_SEQ
    src_index = (batch_offsets + topk_indices).reshape(-1).contiguous()
    dst_index = _arange_on_device(BATCH * TOPK, torch.long, device).contiguous()
    valid_mask = ((topk_indices >= 0) & (topk_indices < SRC_SEQ)).reshape(-1).contiguous()
    return src_index, dst_index, valid_mask


def _build_expected(topk_indices: torch.Tensor, device: str) -> torch.Tensor:
    batch = _arange_on_device(BATCH, torch.float32, device).reshape(BATCH, 1, 1, 1)
    dim = _arange_on_device(HEAD_DIM, torch.float32, device).remainder(8).reshape(1, 1, 1, HEAD_DIM)
    token = topk_indices.to(torch.float32).reshape(BATCH, TOPK, 1, 1)
    return ((batch * 17 + token).remainder(64) + dim).to(DTYPE).contiguous()


def _dump_first_batch_topk(
    topk_indices: torch.Tensor,
    actual: torch.Tensor,
    expected: torch.Tensor,
    count: int,
    head_dims: int,
) -> None:
    count = min(max(int(count), 0), TOPK)
    head_dims = min(max(int(head_dims), 1), HEAD_DIM)
    if count == 0:
        return

    first_batch_indices = topk_indices[0, :count].cpu().tolist()
    actual_rows = actual[0, :count, 0, :head_dims].to(torch.float32).cpu().tolist()
    expected_rows = expected[0, :count, 0, :head_dims].to(torch.float32).cpu().tolist()
    row_matches = actual[0, :count].eq(expected[0, :count]).reshape(count, -1).all(dim=1).cpu().tolist()

    print(
        f"[rank 1] first batch first {count} topk rows "
        f"(showing first {head_dims}/{HEAD_DIM} dims)",
        flush=True,
    )
    for pos, token_idx in enumerate(first_batch_indices):
        print(
            f"  topk_pos={pos}, token_idx={int(token_idx)}, match={bool(row_matches[pos])}, "
            f"actual={actual_rows[pos]}, expected={expected_rows[pos]}",
            flush=True,
        )


def _make_src_meta_tensor(device_kind: str) -> torch.Tensor:
    if device_kind == "meta":
        return torch.empty(SRC_SHAPE, dtype=DTYPE, device="meta")
    return torch.empty(SRC_SHAPE, dtype=DTYPE, device="cpu").contiguous()


def _assert_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if torch.equal(actual, expected):
        return

    mismatch = actual.ne(expected)
    mismatch_count = int(mismatch.sum().item())
    first = mismatch.nonzero(as_tuple=False)[:10].cpu().tolist()
    print(f"UniDex remote sparse copy mismatch count: {mismatch_count}", flush=True)
    for index in first:
        index_tuple = tuple(index)
        print(
            "mismatch "
            f"index={index_tuple}, actual={actual[index_tuple].item()}, "
            f"expected={expected[index_tuple].item()}",
            flush=True,
        )
    raise AssertionError("UniDex remote sparse copy verification failed")


def _initialize_bm(args: argparse.Namespace, rank: int) -> None:
    store_url = f"tcp://{args.head_ip}:{args.store_port}"
    cfg = bm.BmConfig()
    cfg.rank_id = rank
    cfg.start_store = rank == 0
    cfg.set_nic(args.nic_url)

    assert bm.initialize(store_url, WORLD_SIZE, args.device_id, cfg) == 0, "bm.initialize failed"


def _create_handle(args: argparse.Namespace, rank: int):
    store_url = f"tcp://{args.head_ip}:{args.store_port}"
    handle = bm.create2(
        id=args.pool_id,
        local_dram_size=args.pool_bytes,
        max_dram_size=args.pool_bytes,
        data_op_type=DATA_OP_TYPE,
    )
    print(f"[rank {rank}] join store={store_url}", flush=True)
    assert handle.join() == 0, "join failed"
    return handle


def _cleanup(handle, bm_inited: bool, joined: bool) -> None:
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
    if bm_inited:
        bm.uninitialize(0)
    mf.uninitialize()


def _run_rank0(args: argparse.Namespace) -> None:
    total_bytes = _nbytes(SRC_SHAPE, DTYPE)
    if args.pool_bytes < total_bytes:
        raise RuntimeError(f"pool_bytes={args.pool_bytes} is smaller than required {total_bytes}")

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

        gva_me = handle.peer_rank_ptr(0, bm.BmMemType.HOST)
        assert gva_me != 0, "peer_rank_ptr(rank=0, HOST) returned 0"
        print(f"[rank 0] local DRAM pool GVA={_hex(gva_me)}, bytes={total_bytes}", flush=True)

        src = _build_rank0_kv(args.device)
        ret = handle.copy_data(
            src.data_ptr(),
            gva_me,
            total_bytes,
            bm.BmCopyType.L2GH,
            COPY_EXTEND_FLAG,
        )
        assert ret == 0, f"L2GH offload failed, ret={ret}, err={mf.get_last_err_msg()}"
        torch.npu.current_stream().synchronize()
        torch.npu.synchronize()
        print("[rank 0] device KV tensor copied to DRAM pool base; waiting for rank 1", flush=True)

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


def _run_rank1(args: argparse.Namespace) -> None:
    unidex_copy_inplace = _load_unidex_copy_inplace()

    mf.set_log_level(args.log_level)
    assert mf.initialize() == 0, "mf.initialize failed"
    handle = None
    bm_inited = False
    joined = False
    try:
        _initialize_bm(args, rank=1)
        bm_inited = True
        handle = _create_handle(args, rank=1)
        joined = True

        time.sleep(args.post_join_sleep_sec)
        gva_peer0 = handle.peer_rank_ptr(0, bm.BmMemType.HOST)
        assert gva_peer0 != 0, "peer_rank_ptr(rank=0, HOST) returned 0"

        src_lva = handle.gva_to_va(gva_peer0, bm.BmMemType.LOCAL_DEVICE)
        assert src_lva != 0, f"gva_to_va({gva_peer0}, LOCAL_DEVICE) failed"
        print(f"[rank 1] peer0 HOST GVA={_hex(gva_peer0)}, local device VA={_hex(src_lva)}", flush=True)

        dst = torch.empty(DST_SHAPE, dtype=DTYPE, device=args.device).contiguous()
        topk_indices = _build_topk_indices(args.device, args.seed, args.unique_topk)
        src_index, dst_index, valid_mask = _build_unidex_indices(topk_indices)
        src_meta = _make_src_meta_tensor(args.src_meta_device)

        unidex_copy_inplace(
            src=src_meta,
            dst=dst,
            src_index=src_index,
            dst_index=dst_index,
            valid_mask=valid_mask,
            src_address_ndims=SRC_ADDRESS_NDIMS,
            dst_address_ndims=DST_ADDRESS_NDIMS,
            block_dim=args.block_dim,
            src_ptr=src_lva,
        )
        torch.npu.synchronize()

        if not args.skip_verify:
            expected = _build_expected(topk_indices, args.device)
            _dump_first_batch_topk(
                topk_indices,
                dst,
                expected,
                args.dump_first_batch_topk,
                args.dump_head_dims,
            )
            _assert_equal(dst, expected)

        print("[rank 1] UniDex remote DRAM sparse copy OK", flush=True)
    finally:
        _cleanup(handle, bm_inited, joined)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-rank MemFabric HOST DRAM pool sparse-copy test with UniDexCopy."
    )
    parser.add_argument("rank", type=int, choices=(0, 1))
    parser.add_argument("head_ip", nargs="?", default="")
    parser.add_argument("--store-port", type=int, default=STORE_PORT)
    parser.add_argument("--device-id", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    parser.add_argument("--nic-url", default=DEFAULT_NIC_URL)
    parser.add_argument("--pool-id", type=int, default=0)
    parser.add_argument("--pool-bytes", type=int, default=ONE_GIB)
    parser.add_argument("--log-level", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--block-dim", type=int, default=24)
    parser.add_argument("--post-join-sleep-sec", type=float, default=POST_JOIN_RANK1_SLEEP_SEC)
    parser.add_argument("--rank0-hold-sec", type=float, default=0.0)
    parser.add_argument("--src-meta-device", choices=("cpu", "meta"), default="cpu")
    parser.add_argument("--dump-first-batch-topk", type=int, default=16)
    parser.add_argument("--dump-head-dims", type=int, default=16)
    parser.add_argument("--unique-topk", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    if not args.head_ip:
        args.head_ip = input("Head node IP: ").strip()
    if not args.head_ip:
        raise RuntimeError("head_ip is required")

    args.device = "npu"
    return args


def main() -> None:
    args = _parse_args()
    torch.npu.set_device(args.device_id)
    if args.rank == 0:
        _run_rank0(args)
    else:
        _run_rank1(args)


if __name__ == "__main__":
    main()
