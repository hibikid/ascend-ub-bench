"""CPU-only checks of workload semantics; no synthetic hardware conclusions."""
import importlib.util
import contextlib
import io
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("qos", Path(__file__).parents[1] / "qos_bench.py")
qos = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qos)


class WorkloadTests(unittest.TestCase):
    def args(self, *extra):
        return qos.parse_args(["--simulate", "--engines", "sdma", "--modes", "sequential",
            "--seconds", ".01", "--warmup-seconds", "0", "--min-h-samples", "10",
            "--repeats", "1", "--h-count", "30", "--p-total-bytes", str(qos.MiB),
            "--k-high", "2", "--k-low", "1", *extra])

    def config(self, experiment="E4", k=2, size=65536):
        return dict(experiment=experiment, engine="sdma", mode="random", p_size=size,
                    k=k, strategy="S0", repeat=0)

    def test_nearest_rank(self):
        self.assertEqual(qos.quantiles(range(1, 101))["p99"], 99)
        self.assertEqual(qos.quantiles([9])["p50"], 9)
        self.assertIsNone(qos.quantiles([])["p99"])

    def test_working_set_permutation_and_same_byte_sequence(self):
        for mode in ("sequential", "random"):
            working_set = 6 * qos.MiB
            tiles = [qos.offsets(i, qos.MiB, working_set, mode, 42) for i in range(6)]
            self.assertEqual(len(set(tiles)), 6)
            self.assertEqual(qos.offsets(6, qos.MiB, working_set, mode, 42), tiles[0])
            for size in (4096, 65536):
                small = [qos.offsets(i, size, working_set, mode, 42)
                         for i in range(working_set // size)]
                expected = [tile + offset for tile in tiles for offset in range(0, qos.MiB, size)]
                self.assertEqual(small, expected)
                self.assertTrue(all(0 <= offset <= working_set - size and offset % 64 == 0 for offset in small))

    def test_default_e2_matrix(self):
        args = qos.parse_args([])
        matrix = list(qos.configurations(args))
        self.assertEqual(sum(c["experiment"] == "E2" for c in matrix), 72 * 5)
        self.assertEqual(sum(c["experiment"] == "E1" for c in matrix), 8 * 5)

    def test_fixed_work_and_slots(self):
        args, config = self.args(), self.config()
        data = qos.measure(qos.SimBackend(), args, config)
        summary = qos.summarize(data, args, config)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(len(data["h"]), args.h_count)
        self.assertEqual(sum(r[3] for r in data["p"]), args.p_total_bytes)
        self.assertEqual(sorted(r[0] for r in data["p"]), list(range(16)))
        self.assertLessEqual(summary["p_outstanding_max"], 2)
        self.assertGreater(summary["p_outstanding_time_weighted_mean"], 0)
        self.assertEqual(data["inflight"][-1][1], 0)
        for slot in range(2):
            rows = sorted((r for r in data["p"] if r[1] == slot), key=lambda r: r[4])
            self.assertTrue(all(first[6] <= second[4] for first, second in zip(rows, rows[1:])))
        self.assertTrue(all(r[3] <= r[4] <= r[5] == r[6] for r in data["h"]))

    def test_e2_meets_both_stopping_conditions(self):
        args, config = self.args(), self.config("E2")
        data = qos.measure(qos.SimBackend(), args, config)
        summary = qos.summarize(data, args, config)
        self.assertTrue(summary["sample_target_met"])
        self.assertGreaterEqual(len(data["h"]), args.min_h_samples)

    def test_fixed_work_s0_s1_identical_address_multiset(self):
        args = self.args()
        runs = [qos.measure(qos.SimBackend(), args, self.config(k=k)) for k in (1, 2)]
        self.assertEqual([r[1] for r in runs[0]["h"]], [r[1] for r in runs[1]["h"]])
        self.assertEqual(sorted((r[0], r[2], r[3]) for r in runs[0]["p"]),
                         sorted((r[0], r[2], r[3]) for r in runs[1]["p"]))

    def test_failure_is_retained_and_invalidates_round(self):
        class FailingBackend(qos.SimBackend):
            def read(self, *args):
                raise RuntimeError("injected backend failure")
            def validate(self, *args):
                pass
        args, config = self.args(), self.config()
        data = qos.measure(FailingBackend(), args, config)
        self.assertTrue(data["errors"])
        self.assertTrue(any(r[-1] != "ok" for r in data["h"] + data["p"]))
        summary = qos.summarize(data, args, config)
        self.assertEqual(summary["status"], "invalid_or_incomplete")
        self.assertEqual(summary["full"]["p_completed_bytes"], 0)

    def test_timeout_is_not_success(self):
        args = self.args("--max-seconds", ".001", "--h-count", "100000")
        config = self.config()
        data = qos.measure(qos.SimBackend(), args, config)
        self.assertTrue(data["budget_limited"])
        self.assertEqual(qos.summarize(data, args, config)["status"], "invalid_or_incomplete")

    def test_e3_target_rate_required(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("--experiments", "E3")

    def test_e3_budget_and_rate_schedule(self):
        args = self.args("--experiments", "E3", "--target-gib-s", ".05")
        config = self.config("E3", size=65536)
        data = qos.measure(qos.SimBackend(), args, config)
        self.assertFalse(data["errors"])
        self.assertEqual(sum(r[3] for r in data["p"]), args.p_total_bytes)
        for row in data["p"]:
            due = data["start"] + int(row[0] * row[3] / (args.target_gib_s * qos.GiB) * 1e9)
            self.assertGreaterEqual(row[4], due)

    def test_comparisons_use_paired_repeat_and_common_window(self):
        def row(strategy, p99, bw=2, seconds=4):
            window = {"h_e2e_us": {"p99": p99}, "p_gib_s": bw}
            return {"status": "ok", "run_id": strategy, "repeat": 0,
                    "engine": "sdma", "mode": "random", "strategy": strategy,
                    "experiment": "E1" if strategy.startswith("baseline") else "E4",
                    "full": window, "common_active": window,
                    "total_completion_seconds": seconds, "simulation": True}
        result = qos.comparisons([row("baseline_pre", 10), row("baseline_post", 12),
                                  row("S0", 40), row("S1", 20, 1, 6)])
        self.assertEqual(result[-1]["h_p99_amplification"], 2)
        self.assertEqual(result[-1]["baseline_post_over_pre_p99"], 1.2)
        self.assertEqual(result[-1]["h_p99_improvement"], .5)
        self.assertEqual(result[-1]["p_throughput_loss"], .5)
        self.assertEqual(result[-1]["completion_time_change"], .5)

    def test_overlap(self):
        self.assertTrue(qos.has_overlap([(3, 5)], [(0, 4), (8, 10)]))
        self.assertFalse(qos.has_overlap([(4, 5)], [(0, 4), (8, 10)]))
        self.assertFalse(qos.has_overlap([], [(0, 4)]))


if __name__ == "__main__":
    unittest.main()
