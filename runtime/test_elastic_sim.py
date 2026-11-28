import unittest

from runtime import elastic_config as ec
from runtime import elastic_sim as es

SMALL = es.Preset(layers=8, row_bytes=256, index_bytes=256, page=16, hot_per_request=160, max_running=64,
                  budget_gib=0.3, max_tokens=1 << 20, base_ms=(10.0, 0.4), stock_hot_rows=1280)


def config(mode="steady", **controller):
    return ec.parse({"schema": ec.SCHEMA, "model": "minimax_m3", "mode": mode,
                     "layers": {"eviction_order": [[0, 1], [2, 3], [4, 5]]},
                     "profile": {"miss_rate": {"4096": {str(i): 0.02 * (i + 1) for i in range(8)}},
                                 "selected_rows": 256, "fixed_us": 20.0},
                     "controller": {"interval_steps": 4, "dwell_steps": 64, "horizon_steps": 8, **controller}})


def burst_then_quiet():
    burst = es.workload(40.0, 60, (3000, 6000), (64, 128), seed=1)
    quiet = es.workload(0.5, 10, (500, 800), (64, 128), seed=2)
    end = burst[-1].arrival + 30.0
    for i, r in enumerate(quiet):
        r.rid, r.arrival = 1000 + i, end + r.arrival
    return burst + quiet


class SimulatorTests(unittest.TestCase):
    def test_every_policy_finishes_the_trace(self):
        requests = es.workload(4.0, 40, (1000, 2500), (32, 96), seed=0)
        for policy in ("full_resident", "stock", "fixed:2", "hybrid", "elastic"):
            with self.subTest(policy):
                result = es.simulate(config(), SMALL, requests, policy)
                self.assertEqual(result.finished, 40)
                self.assertEqual(result.output_tokens, sum(r.output for r in requests))

    def test_elastic_offloads_under_a_burst_and_backfills_when_quiet(self):
        for mode in ("steady", "smooth"):
            with self.subTest(mode):
                result = es.simulate(config(mode), SMALL, burst_then_quiet(), "elastic", noise=0.0)
                self.assertEqual(result.finished, 70)
                self.assertGreater(max(result.levels), 0)
                self.assertGreaterEqual(result.transitions, 2)
                self.assertEqual(result.retracted, 0)

    def test_capacity_grows_with_the_offload_level(self):
        layout = es.layout_for(SMALL, 2)
        chunks = int(SMALL.budget_gib * (1 << 30)) // layout.chunk
        cfg = config()
        caps = [layout.max_tokens_for(chunks, hot_rows=SMALL.hot_per_request * 4, offloaded=cfg.offloaded(level))
                for level in range(4)]
        self.assertEqual(caps, sorted(caps))
        self.assertLess(caps[0], caps[-1])

    def test_summary_reports_goodput_against_the_slo(self):
        requests = es.workload(4.0, 20, (1000, 2000), (32, 64), seed=3)
        summary = es.simulate(config(), SMALL, requests, "elastic").summary({"tpot_ms": 1e9, "ttft_ms": None})
        self.assertEqual(summary["finished"], 20)
        self.assertGreater(summary["goodput_req_s"], 0)
        tight = es.simulate(config(), SMALL, requests, "elastic").summary({"tpot_ms": 0.1, "ttft_ms": None})
        self.assertEqual(tight["goodput_req_s"], 0)


if __name__ == "__main__":
    unittest.main()
