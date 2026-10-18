#!/usr/bin/env python3
"""HBM QoS benchmark using the repository's synchronous MemFabric copy paths.

K bounds outstanding host copy calls, NOT device queue depth. Device overlap and
path selection require a separate profiler trace. No backend fallback is used.
"""
import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import threading
import time

KiB, MiB, GiB = 1024, 1024**2, 1024**3
H_BYTES = 64 * KiB
FLAGS = {"sdma": 0, "mte": 1 << 1}


def check(ret, label):
    if ret != 0:
        raise RuntimeError(f"{label}: return code {ret}")


def quantiles(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {"count": len(values), "mean": sum(values) / len(values),
            **{f"p{p}": values[math.ceil(len(values) * p / 100) - 1]
               for p in (50, 95, 99)}}


@lru_cache(maxsize=128)
def permutation(tiles, seed):
    rng = random.Random(seed)
    multiplier = rng.randrange(1, tiles + 1)
    while math.gcd(multiplier, tiles) != 1:
        multiplier += 1
    return multiplier, rng.randrange(tiles)


def offsets(index, size, working_set, mode, seed):
    """Permute 1 MiB tiles; traverse bytes within each tile in order.

    P sizes divide the tile: all sizes visit the same logical byte sequence.
    H uses the same rule on its own independent working set.
    """
    per_tile = MiB // size
    tile, within = divmod(index, per_tile)
    tiles = working_set // MiB
    if mode == "random":
        multiplier, shift = permutation(tiles, seed)
        tile = (tile * multiplier + shift) % tiles
    return (tile % tiles) * MiB + within * size


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def csv_write(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def environment(args):
    def command(cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                           timeout=10, text=True).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            return f"unavailable: {exc}"
    return {"host": socket.gethostname(), "platform": platform.platform(),
            "python": platform.python_version(), "device": args.device,
            "git_commit": command(["git", "rev-parse", "HEAD"]),
            "git_status": command(["git", "status", "--short"]),
            "npu_smi": command(["npu-smi", "info"]),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "operator_metadata": json.loads(Path(args.metadata).read_text()) if args.metadata else {},
            "clock": vars(time.get_clock_info("perf_counter")),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class SimBackend:
    """Functional smoke test only. Sleep timings carry no hardware meaning."""
    def setup_thread(self):
        pass

    def allocate(self, size):
        return size

    def read(self, engine, region, offset, size, target):
        time.sleep(0.0001)

    def validate(self, engine, region, offset, size, target):
        self.read(engine, region, offset, size, target)


class HardwareBackend:
    def __init__(self, args):
        import torch
        import torch_npu
        import memfabric_hybrid as mf
        from memfabric_hybrid import bm
        self.torch, self.npu, self.mf, self.bm = torch, torch_npu.npu, mf, bm
        self.args, self.handle = args, None
        self.mf_ready = self.bm_ready = self.joined = False
        self.peer = 0
        self.setup_thread()

    def setup_thread(self):
        self.npu.set_device(self.args.device)

    def open(self):
        check(self.mf.initialize(), "mf.initialize")
        self.mf_ready = True
        cfg = self.bm.BmConfig()
        cfg.auto_ranking = False
        cfg.rank_id = self.args.rank
        cfg.start_store = self.args.rank == 0
        cfg.set_nic(f"tcp://{self.args.nic}:1234")
        check(self.bm.initialize(self.args.url, 2, self.args.device, cfg), "bm.initialize")
        self.bm_ready = True
        size = self.args.h_working_set + self.args.p_working_set
        self.handle = self.bm.create2(id=self.args.pool_id,
            local_dram_size=0, max_dram_size=0,
            local_hbm_size=size if self.args.rank == 0 else 0,
            max_hbm_size=size, data_op_type=self.bm.BmDataOpType.SDMA)
        if self.handle is None:
            raise RuntimeError("bm.create2 failed")
        check(self.handle.join(), "pool.join")
        self.joined = True
        self.peer = self.handle.peer_rank_ptr(0, self.bm.BmMemType.DEVICE)
        if not self.peer:
            raise RuntimeError("rank0 HBM GVA unavailable")

    def allocate(self, size):
        target = self.torch.empty(size, dtype=self.torch.uint8, device="npu")
        if target.data_ptr() % 64:
            raise RuntimeError("destination must be 64B aligned")
        return target

    def pattern(self, region, offset, size):
        # Address-sensitive pattern generated on CPU; initialization/QA only.
        return ((self.torch.arange(offset, offset + size, dtype=self.torch.int64)
                 + (7 if region == "H" else 31)) % 251).to(self.torch.uint8)

    def stage(self):
        target = self.allocate(MiB)
        for region, size, base in (("H", self.args.h_working_set, 0),
                                  ("P", self.args.p_working_set, self.args.h_working_set)):
            for offset in range(0, size, MiB):
                target.copy_(self.pattern(region, offset, MiB))
                self.npu.synchronize()
                check(self.handle.copy_data(target.data_ptr(), self.peer + base + offset,
                    MiB, self.bm.BmCopyType.L2G, FLAGS[self.args.stage_engine]), "HBM staging")
        self.npu.synchronize()

    def read(self, engine, region, offset, size, target):
        base = 0 if region == "H" else self.args.h_working_set
        check(self.handle.copy_data(self.peer + base + offset, target.data_ptr(), size,
                                   self.bm.BmCopyType.G2L, FLAGS[engine]), f"{engine} G2L")

    def validate(self, engine, region, offset, size, target):
        target.fill_(255)  # poison prevents an unchanged destination passing QA
        self.npu.synchronize()
        self.read(engine, region, offset, size, target)
        self.npu.synchronize()
        if not self.torch.equal(target.cpu(), self.pattern(region, offset, size)):
            raise RuntimeError(f"{engine} {region} data mismatch at offset {offset}")

    def close(self):
        try:
            if self.handle is not None:
                try:
                    if self.joined:
                        self.handle.leave()
                finally:
                    self.handle.destroy()
        finally:
            try:
                if self.bm_ready:
                    self.bm.uninitialize()
            finally:
                if self.mf_ready:
                    self.mf.uninitialize()


def measure(backend, args, config, warmup=False):
    """One H thread and K P threads, with one synchronous call per thread.

    Bookkeeping lock is never held across copy_data. No executor backlog exists.
    Counted outstanding intervals start BEFORE the API call (acceptance is hidden).
    """
    engine, mode, p_size, k = (config[x] for x in ("engine", "mode", "p_size", "k"))
    fixed = config["experiment"] == "E4" and not warmup
    finite_p = config["experiment"] in ("E3", "E4") and not warmup
    p_limit = args.p_total_bytes // p_size if finite_p else None
    h_limit = args.h_count if fixed else None
    targets = [backend.allocate(H_BYTES)] + [backend.allocate(p_size) for _ in range(k)]
    for region, size, target in [("H", H_BYTES, targets[0])] + [
            ("P", p_size, target) for target in targets[1:]]:
        for offset in (0, (args.h_working_set if region == "H" else args.p_working_set) - size):
            backend.validate(engine, region, offset, size, target)
    lock, go, stop = threading.Lock(), threading.Event(), threading.Event()
    ready = threading.Barrier(k + 2)
    h_rows, p_rows, inflight, errors = [], [], [], []
    state = {"p_next": 0, "active": 0, "p_done": 0, "h_done": 0, "start": 0}
    duration = args.warmup_seconds if warmup else args.seconds
    minimum = 0 if warmup else args.min_h_samples
    if fixed or finite_p:
        minimum = 0
    rate = args.target_gib_s * GiB if config["experiment"] == "E3" and not warmup else 0

    def thread_main(slot):
        try:
            backend.setup_thread()
            ready.wait(timeout=args.ready_timeout)
            go.wait()
            index = 0
            while not stop.is_set():
                now = time.perf_counter_ns()
                elapsed = (now - state["start"]) / 1e9
                if args.max_seconds and elapsed >= args.max_seconds:
                    with lock:
                        state["budget_limited"] = True
                    stop.set()
                    break
                if slot == 0:
                    with lock:
                        finished_p = finite_p and state["p_done"] == p_limit
                    if (h_limit is not None and index >= h_limit) or (finite_p and not fixed and finished_p):
                        break
                    if not finite_p and elapsed >= duration and index >= minimum:
                        stop.set()
                        break
                    size, region, ordinal = H_BYTES, "H", index
                    t_ready = time.perf_counter_ns()
                else:
                    with lock:
                        ordinal = state["p_next"]
                        if p_limit is not None and ordinal >= p_limit:
                            break
                        due = state["start"] + int(ordinal * p_size / rate * 1e9) if rate else now
                        if due <= now:
                            state["p_next"] += 1
                    if due > now:
                        stop.wait(min((due - now) / 1e9, 0.01))
                        continue
                    size, region = p_size, "P"
                    t_ready = time.perf_counter_ns()
                    with lock:
                        state["active"] += 1
                        inflight.append((time.perf_counter_ns(), state["active"], 0))
                working_set = args.h_working_set if slot == 0 else args.p_working_set
                offset = offsets(ordinal, size, working_set, mode, args.seed + (0 if slot == 0 else 1))
                t_call = time.perf_counter_ns()
                try:
                    backend.read(engine, region, offset, size, targets[slot])
                except BaseException as exc:
                    failed = time.perf_counter_ns()
                    with lock:
                        if slot == 0:
                            h_rows.append((ordinal, offset, size, t_ready, t_call, failed, failed, repr(exc)))
                        else:
                            p_rows.append((ordinal, slot - 1, offset, size, t_call, failed, failed, repr(exc)))
                    raise
                t_return = time.perf_counter_ns()
                # copy_data returns after synchronization; no event / submit timestamp exists.
                t_done = t_return
                if slot == 0:
                    h_rows.append((ordinal, offset, size, t_ready, t_call, t_return, t_done, "ok"))
                    with lock:
                        state["h_done"] += 1
                    index += 1
                    if args.h_gap_us:
                        stop.wait(args.h_gap_us / 1e6)
                else:
                    with lock:
                        p_rows.append((ordinal, slot - 1, offset, size, t_call, t_return, t_done, "ok"))
                        state["active"] -= 1
                        state["p_done"] += 1
                        inflight.append((time.perf_counter_ns(), state["active"], 0))
        except BaseException as exc:
            with lock:
                errors.append(repr(exc))
            stop.set()
            ready.abort()

    threads = [threading.Thread(target=thread_main, args=(i,), name=f"qos-{i}") for i in range(k + 1)]
    for thread in threads:
        thread.start()
    try:
        ready.wait(timeout=args.ready_timeout)
        state["start"] = time.perf_counter_ns()
        go.set()
    except threading.BrokenBarrierError:
        stop.set()
        go.set()
    finally:
        try:
            for thread in threads:
                thread.join()
        except BaseException:
            # Never tear down a pool while workers can still access its GVA.
            stop.set()
            go.set()
            for thread in threads:
                thread.join()
            raise
    end = time.perf_counter_ns()
    # Validate the actual last destination of each slot before QA overwrites it.
    if not errors:
        final = [("H", h_rows[-1][1], H_BYTES, targets[0])] if h_rows else []
        for slot in range(k):
            rows = [r for r in p_rows if r[1] == slot]
            if rows:
                final.append(("P", rows[-1][2], p_size, targets[slot + 1]))
        if isinstance(backend, HardwareBackend):
            for region, offset, size, target in final:
                if not backend.torch.equal(target.cpu(), backend.pattern(region, offset, size)):
                    errors.append(f"post-run mismatch: {region} offset={offset}")
        for region, size, target in [("H", H_BYTES, targets[0])] + [
                ("P", p_size, target) for target in targets[1:]]:
            try:
                backend.validate(engine, region, 0, size, target)
            except Exception as exc:
                errors.append(repr(exc))
    return {"start": state["start"], "end": end, "h": h_rows, "p": p_rows,
            "inflight": inflight, "errors": errors, "budget_limited": state.get("budget_limited", False)}


def summarize(data, args, config):
    h = [r for r in data["h"] if r[7] == "ok"]
    p = [r for r in data["p"] if r[7] == "ok"]
    start = data["start"]
    end = max([start] + [r[6] for r in h] + [r[6] for r in p])
    seconds = (end - start) / 1e9
    def window(begin, finish):
        hs = [r for r in h if r[3] >= begin and r[6] <= finish]
        ps = [r for r in p if begin <= r[6] <= finish]
        span = max(0, finish - begin) / 1e9
        return {"start_ns": begin, "end_ns": finish, "seconds": span,
                "h_e2e_us": quantiles([(r[6] - r[3]) / 1000 for r in hs]),
                "h_call_to_done_us": quantiles([(r[6] - r[4]) / 1000 for r in hs]),
                "h_admission_us": quantiles([(r[4] - r[3]) / 1000 for r in hs]),
                "h_call_us": quantiles([(r[5] - r[4]) / 1000 for r in hs]),
                "h_completed_per_s": len(hs) / span if span else 0,
                "p_completed": len(ps), "p_completed_bytes": sum(r[3] for r in ps),
                "p_submitted_bytes": sum(r[3] for r in data["p"] if begin <= r[4] <= finish),
                "p_submitted_gib_s": sum(r[3] for r in data["p"] if begin <= r[4] <= finish) / GiB / span if span else 0,
                "p_gib_s": sum(r[3] for r in ps) / GiB / span if span else 0,
                "p_completed_per_s": len(ps) / span if span else 0}
    full = window(start, end)
    common = None
    if h and p:
        common = window(max(h[0][4], min(r[4] for r in p)), min(h[-1][6], max(r[6] for r in p)))
    area, previous, active, maximum, distribution = 0, start, 0, 0, {}
    for timestamp, current, _ in sorted(data["inflight"]):
        timestamp = min(timestamp, end)
        delta = max(0, timestamp - previous)
        area += active * delta
        distribution[active] = distribution.get(active, 0) + delta
        active, previous = current, timestamp
        maximum = max(maximum, current)
    distribution[active] = distribution.get(active, 0) + max(0, end - previous)
    p_active_s = (max(r[6] for r in p) - min(r[4] for r in p)) / 1e9 if p else 0
    completed = (not data["errors"] and not data["budget_limited"] and bool(h))
    if config["experiment"] in ("E3", "E4"):
        completed &= sum(r[3] for r in p) == args.p_total_bytes
    if config["experiment"] == "E4":
        completed &= len(h) == args.h_count
    overlap = has_overlap([(r[4], r[6]) for r in h[:1000]], [(r[4], r[6]) for r in p])
    return {"status": "ok" if completed else "invalid_or_incomplete",
            "simulation": args.simulate, "device_overlap": "unverified",
            "interpretation": "functional simulation only" if args.simulate else
                "host-concurrent calls only; requires thread-safety, direct-path and device timeline verification",
            "path": "simulation" if args.simulate else "unverified; no requested-backend fallback",
            "host_call_overlap_observed_first_1000_h": overlap,
            "k_semantics": "outstanding synchronous host calls; acceptance time hidden",
            "sample_target_met": len(h) >= args.min_h_samples and seconds >= args.seconds,
            "full": full, "common_active": common,
            "common_h_fraction": common["h_e2e_us"]["count"] / len(h) if common and h else 0,
            "p_outstanding_time_weighted_mean": area / (end - start) if end > start else 0,
            "p_outstanding_max": maximum, "p_outstanding_duration_ns": distribution,
            "p_active_seconds": p_active_s,
            "p_bytes_per_active_second": sum(r[3] for r in p) / p_active_s if p_active_s else 0,
            "p_submitted_bytes": sum(r[3] for r in data["p"]),
            "h_failed": sum(r[7] != "ok" for r in data["h"]),
            "p_failed": sum(r[7] != "ok" for r in data["p"]),
            "total_completion_seconds": seconds, "budget_limited": data["budget_limited"],
            "errors": data["errors"]}


def merged_intervals(intervals):
    merged = []
    for begin, end in sorted(intervals):
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((begin, end))
    return merged


def has_overlap(h_intervals, p_intervals):
    p_intervals = merged_intervals(p_intervals)
    cursor = 0
    for begin, end in sorted(h_intervals):
        while cursor < len(p_intervals) and p_intervals[cursor][1] <= begin:
            cursor += 1
        if cursor < len(p_intervals) and p_intervals[cursor][0] < end:
            return True
    return False


def configurations(args):
    rng = random.Random(args.seed)
    for repeat in range(args.repeats):
        for engine in args.engines:
            for mode in args.modes:
                base = {"repeat": repeat, "engine": engine, "mode": mode,
                        "experiment": "E1", "strategy": "baseline_pre", "p_size": 64 * KiB, "k": 0}
                yield dict(base)
                jobs = []
                for experiment in args.experiments:
                    if experiment in ("E2", "E3"):
                        for size in args.p_sizes:
                            for k in (args.ks if experiment == "E2" else [args.k_high]):
                                jobs.append(dict(base, experiment=experiment, p_size=size, k=k, strategy="S0"))
                    elif experiment == "E4":
                        for strategy, k in (("S0", args.k_high), ("S1", args.k_low)):
                            jobs.append(dict(base, experiment="E4", p_size=args.e4_p_size, k=k, strategy=strategy))
                rng.shuffle(jobs)
                yield from jobs
                if jobs:
                    yield dict(base, strategy="baseline_post")


def comparisons(rows):
    """Pair within a repeat/backend/address mode; never pool raw quantiles."""
    result = []
    for row in rows:
        if row["status"] != "ok" or row["experiment"] == "E1":
            continue
        peers = [r for r in rows if r["status"] == "ok" and all(
            r[key] == row[key] for key in ("repeat", "engine", "mode"))]
        baseline = next((r for r in peers if r["strategy"] == "baseline_pre"), None)
        post = next((r for r in peers if r["strategy"] == "baseline_post"), None)
        window = row["common_active"] or row["full"]
        p99 = window["h_e2e_us"]["p99"]
        entry = {"run_id": row["run_id"], "comparison_window": "common_active" if row["common_active"] else "full",
                 "device_overlap": "unverified", "simulation": row["simulation"]}
        if baseline and p99 and baseline["full"]["h_e2e_us"]["p99"]:
            denominator = baseline["full"]["h_e2e_us"]["p99"]
            entry["h_p99_amplification"] = p99 / denominator
            if post:
                entry["baseline_post_over_pre_p99"] = post["full"]["h_e2e_us"]["p99"] / denominator
        if row["experiment"] == "E4" and row["strategy"] == "S1":
            s0 = next((r for r in peers if r["experiment"] == "E4" and r["strategy"] == "S0"), None)
            if s0:
                s0_window = s0["common_active"]
                if s0_window and row["common_active"]:
                    s0_p99 = s0_window["h_e2e_us"]["p99"]
                    if s0_p99 and p99:
                        entry["h_p99_improvement"] = 1 - p99 / s0_p99
                    if s0_window["p_gib_s"]:
                        entry["p_throughput_loss"] = 1 - window["p_gib_s"] / s0_window["p_gib_s"]
                if s0["total_completion_seconds"]:
                    entry["completion_time_change"] = row["total_completion_seconds"] / s0["total_completion_seconds"] - 1
        result.append(entry)
    return result


def run_suite(backend, args, peer_info):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    env = environment(args)
    if isinstance(backend, HardwareBackend):
        env.update(torch=backend.torch.__version__,
                   memfabric=getattr(backend.mf, "__version__", "unknown"))
    aggregate = []
    for number, config in enumerate(configurations(args)):
        run_id = f"{number:04d}_{config['experiment']}_{config['engine']}_{config['mode']}_{config['strategy']}_K{config['k']}_P{config['p_size']}"
        folder = root / run_id
        folder.mkdir()
        (folder / "profiling").mkdir()
        json_write(folder / "config.json", {"args": vars(args), "run": config,
            "environment": env, "source": peer_info,
            "backend": "copy_data(G2L, flags=0 or 2), synchronous; one call/request",
            "thread_count": config["k"] + 1, "software_pending_queue": 0,
            "stream_assignment": "internal to MemFabric; Python API exposes no stream",
            "mte_configuration": "inherited from installed extend library; not independently configurable",
            "cache_state": "uncontrolled", "completion": "synchronous return; t_done=t_return",
            "logging": "in-memory per-request tuples, CSV after timing; overhead included",
            "address_period_bytes": {"H": args.h_working_set, "P": args.p_working_set}})
        print(run_id, flush=True)
        try:
            if args.warmup_seconds:
                warm = measure(backend, args, config, warmup=True)
                if warm["errors"]:
                    raise RuntimeError(f"warmup failed: {warm['errors']}")
            data = measure(backend, args, config)
            summary = summarize(data, args, config)
            csv_write(folder / "h_requests.csv", ["id", "offset", "bytes", "t_ready_ns", "t_call_ns",
                      "t_return_ns", "t_done_ns", "status"], data["h"])
            csv_write(folder / "p_completions.csv", ["id", "slot", "offset", "bytes", "t_call_ns",
                      "t_return_ns", "t_done_ns", "status"], data["p"])
            csv_write(folder / "inflight.csv", ["timestamp_ns", "outstanding_host_calls", "software_pending"], data["inflight"])
            json_write(folder / "summary.json", summary)
            (folder / "validation.log").write_text("pre/post correctness: " + ("FAILED\n" if data["errors"] else "passed\n")
                + "Device overlap / direct path: unverified; collect separate profiling.\n"
                + "\n".join(data["errors"]))
            aggregate.append({"run_id": run_id, **config, **summary})
            json_write(root / "summary.json", aggregate)
            json_write(root / "comparisons.json", comparisons(aggregate))
            print(f"  {summary['status']} H={len(data['h'])} p99={summary['full']['h_e2e_us']['p99']} us "
                  f"P={summary['full']['p_gib_s']:.4f} GiB/s", flush=True)
            if summary["status"] != "ok":
                raise RuntimeError("invalid/incomplete run; inspect raw records and summary")
        except BaseException as exc:
            with (folder / "validation.log").open("a") as log:
                log.write(f"\nFAILED: {exc!r}\n")
            raise
    return root


def receive(stream):
    line = stream.readline(1024 * 1024)
    if not line:
        raise RuntimeError("control connection closed")
    message = json.loads(line)
    if "error" in message:
        raise RuntimeError(f"peer failure: {message['error']}")
    return message


def send(stream, message):
    stream.write((json.dumps(message) + "\n").encode())
    stream.flush()


def hardware_main(args):
    backend = HardwareBackend(args)
    listener = connection = stream = None
    try:
        if args.rank == 0:
            listener = socket.socket()
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.control_host, args.control_port))
            listener.listen(1)
            listener.settimeout(args.ready_timeout)
        backend.open()
        if args.rank == 0:
            print("D1/rank0: HBM pool joined; waiting for D0/rank1", flush=True)
            connection, _ = listener.accept()
        else:
            deadline = time.monotonic() + args.ready_timeout
            while True:
                try:
                    connection = socket.create_connection((args.source_host, args.control_port), timeout=5)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.2)
        connection.settimeout(args.control_timeout)
        stream = connection.makefile("rwb")
        layout = {"h_working_set": args.h_working_set, "p_working_set": args.p_working_set,
                  "pool_id": args.pool_id}
        if args.rank == 0:
            message = receive(stream)
            if message != layout:
                raise RuntimeError("source/requester pool layouts differ")
            backend.stage()
            info = environment(args)
            info["torch"] = backend.torch.__version__
            info["memfabric"] = getattr(backend.mf, "__version__", "unknown")
            send(stream, {"ready": True, "environment": info, "layout": layout})
            print("D1: source staged; holding HBM until requester completes", flush=True)
            if receive(stream) != {"done": True}:
                raise RuntimeError("unexpected requester completion message")
        else:
            send(stream, layout)
            peer_info = receive(stream)
            deadline = time.monotonic() + args.ready_timeout
            while not backend.handle.gva_to_va(backend.peer, backend.bm.BmMemType.LOCAL_DEVICE):
                if time.monotonic() >= deadline:
                    raise TimeoutError("rank0 HBM mapping not ready")
                time.sleep(0.2)
            run_suite(backend, args, peer_info)
            send(stream, {"done": True})
    except BaseException as exc:
        if stream is not None:
            try:
                send(stream, {"error": repr(exc)})
            except OSError:
                pass
        raise
    finally:
        if stream:
            stream.close()
        if connection:
            connection.close()
        if listener:
            listener.close()
        backend.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(0, 1), default=1, help="0=D1 source, 1=D0 requester")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--url", default="tcp://127.0.0.1:8570")
    parser.add_argument("--nic", default="127.0.0.1")
    parser.add_argument("--pool-id", type=int, default=0)
    parser.add_argument("--source-host", default="127.0.0.1")
    parser.add_argument("--control-host", default="0.0.0.0")
    parser.add_argument("--control-port", type=int, default=8571)
    parser.add_argument("--ready-timeout", type=float, default=120)
    parser.add_argument("--control-timeout", type=float, default=86400)
    parser.add_argument("--engines", nargs="+", choices=FLAGS, default=list(FLAGS))
    parser.add_argument("--stage-engine", choices=FLAGS, default="sdma")
    parser.add_argument("--modes", nargs="+", choices=("sequential", "random"), default=["sequential", "random"])
    parser.add_argument("--experiments", nargs="+", choices=("E1", "E2", "E3", "E4"), default=["E1", "E2"])
    parser.add_argument("--p-sizes", nargs="+", type=int, default=[4 * KiB, 64 * KiB, MiB])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--k-high", type=int, default=32)
    parser.add_argument("--k-low", type=int, default=2)
    parser.add_argument("--e4-p-size", type=int, default=64 * KiB)
    parser.add_argument("--h-working-set", type=int, default=64 * MiB)
    parser.add_argument("--p-working-set", type=int, default=256 * MiB)
    parser.add_argument("--h-gap-us", type=float, default=0)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--warmup-seconds", type=float, default=5)
    parser.add_argument("--min-h-samples", type=int, default=100000)
    parser.add_argument("--max-seconds", type=float, default=0, help="0=unlimited; partial rounds are invalid")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--h-count", type=int, default=100000)
    parser.add_argument("--p-total-bytes", type=int, default=8 * GiB)
    parser.add_argument("--target-gib-s", type=float, default=0, help="required for E3; no default inferred rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata", help="operator-supplied environment/path/profiling metadata JSON")
    parser.add_argument("--output", default=f"QoS/results/{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}")
    parser.add_argument("--simulate", action="store_true", help="CPU functional smoke test, no hardware conclusions")
    parser.add_argument("--dry-run", action="store_true", help="print matrix without hardware imports")
    args = parser.parse_args(argv)
    for name in ("h_working_set", "p_working_set"):
        if getattr(args, name) <= 0 or getattr(args, name) % (2 * MiB):
            parser.error(f"{name} must be a positive multiple of 2 MiB")
    if any(size not in (4 * KiB, 64 * KiB, MiB) for size in args.p_sizes + [args.e4_p_size]):
        parser.error("P sizes must be 4096, 65536 or 1048576")
    if any(k <= 0 for k in args.ks + [args.k_high, args.k_low]) or args.k_low > args.k_high:
        parser.error("K must be positive; k-low <= k-high")
    if (args.seconds <= 0 or args.warmup_seconds < 0 or args.max_seconds < 0 or args.h_gap_us < 0
            or args.min_h_samples <= 0 or args.h_count <= 0 or args.repeats <= 0
            or args.ready_timeout <= 0 or args.control_timeout <= 0):
        parser.error("invalid duration/count")
    if args.p_total_bytes <= 0 or args.p_total_bytes % MiB:
        parser.error("p-total-bytes must be a positive multiple of 1 MiB")
    if "E3" in args.experiments and args.target_gib_s <= 0:
        parser.error("E3 requires --target-gib-s, selected after independent capacity measurements")
    return args


def main():
    args = parse_args()
    if args.dry_run:
        print(json.dumps(list(configurations(args)), indent=2))
    elif args.simulate:
        run_suite(SimBackend(), args, {"simulation": True})
    else:
        hardware_main(args)


if __name__ == "__main__":
    main()
