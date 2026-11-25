"""CPU tests of the combined study's decode window, real-batch evidence and paired verdicts."""
import unittest

from runtime import combined_measure as m

STEP, FIRST = 0.015625, 1.0       # binary fractions keep every window boundary exact


def trace(batch=4, tokens=(100, 100, 100, 50), stagger=0.125):
    """Request i starts at 0, gets its first token at FIRST + i * stagger, then one token per STEP."""
    out = []
    for i in range(batch):
        first = FIRST + i * stagger
        out.append(dict(alias=f"r{i}", start=0.0, successful=True, finish_reason="length",
                        events=[(first + k * STEP, 1) for k in range(tokens[i])]))
    return out


def block(gain, tail=-0.01, quality=True, order="AB"):
    window = lambda tps, p95: dict(tokens_per_s=tps, tpot_ms=dict(p50=p95 * 0.9, p95=p95, p99=p95 * 1.1))
    return dict(round=0, order=order, control=window(1000.0, 100.0),
                candidate=window(1000.0 * (1 + gain), 100.0 * (1 + tail)), quality_ok=quality)


class DecodeLog(unittest.TestCase):
    def test_sglang_decode_line_gives_real_batch_and_graph_replay(self):
        line = ("[2026-11-25 10:00:01 TP0] Decode batch, #running-req: 32, #token: 1048576, token usage: 0.52, "
                "cuda graph: True, gen throughput (token/s): 1234.56, #queue-req: 3")
        self.assertEqual(m.parse_decode_line(line), dict(running=32, graph=True, queue=3, gen_throughput=1234.56))
        eager = line.replace("Decode batch,", "Decode batch [17],").replace("True", "False")
        self.assertEqual(m.parse_decode_line(eager)["graph"], False)
        self.assertIsNone(m.parse_decode_line("[2026-11-25 10:00:01] Prefill batch, #new-seq: 4"))

    def test_window_samples_must_all_show_the_planned_batch_on_a_graph(self):
        line = lambda running, graph=True: dict(running=running, graph=graph, queue=0, gen_throughput=1.0)
        samples = [(t, line(4)) for t in (1.0, 1.5, 2.0, 2.5)] + [(9.0, line(3))]
        info, reasons = m.real_batch(samples, batch=4, start=1.0, end=2.5)
        self.assertEqual((info["samples"], info["running"], info["graph"], reasons), (4, [4], True, []))
        self.assertTrue(m.real_batch(samples, batch=4, start=1.0, end=9.0)[1])            # batch shrank
        self.assertTrue(m.real_batch([(1.0, line(4, False))] * 3, batch=4, start=0, end=2)[1])
        self.assertIn("decode log samples", m.real_batch(samples, batch=4, start=1.0, end=1.4)[1][0])


class SteadyWindow(unittest.TestCase):
    def test_window_opens_after_the_prefill_barrier_and_warmup_and_closes_at_the_first_finish(self):
        window, reasons = m.steady_window(trace(), batch=4, warmup_steps=5, min_steps=40)
        self.assertEqual(reasons, [])
        self.assertEqual((window["barrier_s"], window["start_s"], window["end_s"]), (1.375, 1.453125, 2.140625))
        self.assertEqual((window["steps"], window["delivered_tokens"], window["tokens_per_s"]), (44, 176, 256.0))
        self.assertEqual((window["tpot_ms"]["p50"], window["tpot_ms"]["p95"]), (15.625, 15.625))
        self.assertEqual((window["ttft_s"]["p50"], window["finish_reasons"]), (1.1875, {"length": 4}))

    def test_short_failed_or_missing_runs_are_rejected_not_rescaled(self):
        self.assertIn("needs 45", m.steady_window(trace(), batch=4, warmup_steps=5, min_steps=45)[1][0])
        self.assertIn("warmup", m.steady_window(trace(), batch=4, warmup_steps=50, min_steps=1)[1][0])
        self.assertIn("1 of 5", m.steady_window(trace(), batch=5, warmup_steps=5, min_steps=1)[1][0])
        failed = trace()
        failed[1]["successful"] = False
        window, reasons = m.steady_window(failed, batch=4, warmup_steps=5, min_steps=1)
        self.assertIsNone(window)
        self.assertIn("1 of 4", reasons[0])

    def test_output_mismatch_counts_missing_and_differing_hashes(self):
        a = {"x": "1", "y": "2"}
        self.assertEqual(m.output_mismatch(a, dict(a)), 0.0)
        self.assertEqual(m.output_mismatch(a, {"x": "1", "y": "3"}), 0.5)
        self.assertEqual(m.output_mismatch({"x": None, "y": "2"}, {"x": None, "y": "2"}), 0.5)
        self.assertEqual(m.output_mismatch(a, {"x": "1"}), 1.0)


class PairedVerdicts(unittest.TestCase):
    RULES = dict(min_gain=0.05, tpot_p95_regression_max=0.02, min_pairs=3)

    def verdict(self, gains, **kwargs):
        blocks = [block(g, **kwargs) for g in gains]
        return m.paired(blocks, **self.RULES)

    def test_t_interval_over_independent_blocks(self):
        low, high = m.t_interval([0.06, 0.07, 0.08])
        self.assertAlmostEqual(low, 0.07 - 4.303 * 0.01 / 3 ** 0.5)
        self.assertAlmostEqual(high, 0.07 + 4.303 * 0.01 / 3 ** 0.5)
        self.assertIsNone(m.t_interval([0.1]))

    def test_pass_fail_and_inconclusive_follow_the_preregistered_rules(self):
        cases = [((0.08, 0.09, 0.10), {}, "pass", None),
                 ((0.01, 0.06, 0.11), {}, "inconclusive", "includes zero"),
                 ((-0.05, -0.04, -0.06), {}, "fail", "negative result"),
                 ((0.010, 0.012, 0.011), {}, "fail", "excludes the target"),
                 ((0.08, 0.09, 0.10), dict(tail=0.05), "fail", "p95 TPOT"),
                 ((0.08, 0.09, 0.10), dict(quality=False), "fail", "quality"),
                 ((0.08, 0.09), {}, "inconclusive", "needs 3")]
        for gains, kwargs, verdict, reason in cases:
            with self.subTest(gains=gains, kwargs=kwargs):
                result = self.verdict(gains, **kwargs)
                self.assertEqual(result["verdict"], verdict)
                if reason is not None:
                    self.assertTrue(any(reason in r for r in result["reasons"]), result["reasons"])
        result = self.verdict((0.08, 0.09, 0.10))
        self.assertAlmostEqual(result["throughput_gain"]["mean"], 0.09)
        self.assertEqual(result["tpot_p95_change"]["tolerance"], 0.02)

    def test_acceptance_needs_every_baseline_at_enough_pressure_points(self):
        points = ("8-16Ki", "16-32Ki", "32-64Ki")
        ok = {"verdict": "pass"}
        results = {(p, base, "cand"): ok for p in points[:2] for base in ("stock", "static")}
        results[("32-64Ki", "stock", "cand")] = {"verdict": "inconclusive"}
        run = lambda r, n=2: m.acceptance(r, candidate="cand", baselines=("stock", "static"), points=points,
                                          min_points=n)
        self.assertEqual(run(results)["overall"], "pass")
        self.assertEqual(run(results, 3)["overall"], "inconclusive")
        self.assertEqual(run(results | {("32-64Ki", "static", "cand"): {"verdict": "fail"}})["overall"], "fail")
        self.assertEqual(run({})["baselines"]["stock"]["verdicts"]["8-16Ki"], "not_run")


if __name__ == "__main__":
    unittest.main()
