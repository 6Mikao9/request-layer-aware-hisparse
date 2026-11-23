import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.compare_runs import compare_runs, main, read_run
from benchmarks.sglang_stream_client import Generation, digest_ids, summarize


def record(alias, ids, finish, *, workload="mixed", first=0.5, tpot=0.25,
           e2e=2, coalesced=False, failed=False):
    budget = 3
    generation = Generation(alias, [10, 11], budget)
    for count in ([len(ids)] if coalesced else range(1, len(ids) + 1)):
        data = {"output_ids": ids[:count], "meta_info": {
            "completion_tokens": count, "finish_reason": finish if count == len(ids) else None,
        }}
        generation.consume(json.dumps(data), first + (count - 1) * tpot if not coalesced else first)
    result = generation.metrics(e2e, not failed)
    return result | {
        "request_alias": alias, "workload": workload, "input_tokens": 2,
        "max_new_tokens": budget, "input_ids_sha256": digest_ids([10, 11]), "successful": not failed,
    } | ({"error": "transport"} if failed else {})


def run(records, duration):
    return {
        "status": "complete", "missing_request_aliases": [], "client_concurrency": 3,
        "source_sha256": {"workload": hashlib.sha256(b"example workload").hexdigest(),
                          "client": hashlib.sha256(b"example client").hexdigest()},
        "summary": summarize(records, duration, 0.1, 0.1), "requests": records,
    }


class CompareRunsTest(unittest.TestCase):
    def setUp(self):
        self.baseline = run([
            record("alias-eos", [20, 21], "stop"),
            record("alias-length", [20, 21, 22], "length"),
            record("alias-failure", [20], None, failed=True),
            record("alias-single", [22], "stop", workload="other", first=1),
            record("alias-coalesced", [20, 21], "stop", workload="other", first=1.5, coalesced=True),
        ], 10)
        self.candidate = run([
            record("alias-coalesced", [20, 21], "stop", workload="other", first=1, coalesced=True),
            record("alias-single", [22], "stop", workload="other", first=0.5),
            record("alias-failure", [20, 21], "stop"),
            record("alias-length", [20, 21, 24], "length", tpot=0.125),
            record("alias-eos", [20, 21], "stop", tpot=0.125),
        ], 5)
        self.thresholds = [(0.5, 0.25), (0.5, 0.125)]

    def test_matches_by_alias_and_reports_observed_metrics(self):
        report = compare_runs(self.baseline, self.candidate, self.thresholds)
        self.assertEqual(report["environment"]["status"], "unverified")
        self.assertEqual(report["pairing"]["status"], "matched_reported_inputs")
        self.assertNotIn("alias-eos", json.dumps(report))
        overall = report["overall"]
        self.assertEqual(overall["output_id_hash_pairs"], {"equal": 3, "different": 1, "unavailable": 1})
        baseline = overall["baseline"]
        self.assertEqual((baseline["requests"], baseline["failures"], baseline["length_limited_requests"]),
                         (5, 1, 1))
        self.assertEqual(baseline["failure_fraction_all_requests"], 1 / 5)
        self.assertEqual(baseline["length_limited_fraction_all_requests"], 1 / 5)
        self.assertEqual(baseline["completed_output_tokens_per_s"], 8 / 10)
        self.assertEqual(baseline["all_observed_output_tokens_per_s"], 9 / 10)
        self.assertEqual(baseline["failed_requests_observed_output_tokens"], 1)
        self.assertEqual(baseline["latency"]["tpot_s"], {
            "p50": 0.25, "p95": 0.25, "p99": 0.25, "included_successes": 2, "missing_successes": 2,
            "excluded_failures": 1,
        })
        self.assertAlmostEqual(baseline["latency"]["ttft_s"]["p50"], 0.75)
        self.assertAlmostEqual(baseline["latency"]["ttft_s"]["p95"], 1.425)
        self.assertAlmostEqual(baseline["latency"]["ttft_s"]["p99"], 1.485)
        self.assertEqual(baseline["latency"]["end_to_end_s"]["p95"], 2)
        self.assertEqual(baseline["slo"][0]["natural_eos_slo_goodput_per_s"], 1 / 10)
        self.assertEqual(baseline["slo"][0]["response_slo_goodput_per_s"], 2 / 10)
        self.assertEqual(baseline["slo"][0]["response_compliant_fraction_all_requests"], 2 / 5)
        self.assertEqual(baseline["slo"][1]["natural_eos_compliant_requests"], 0)
        self.assertEqual(overall["candidate"]["slo"][1]["natural_eos_compliant_requests"], 1)
        self.assertEqual(overall["candidate_minus_baseline"]["slo"][1]["natural_eos_slo_goodput_per_s"], 1 / 5)
        self.assertLess(overall["candidate_minus_baseline"]["latency"]["ttft_s"]["p95"], 0)
        mixed = report["by_workload"]["mixed"]["baseline"]
        self.assertEqual(mixed["slo"][0]["response_compliant_fraction_all_requests"], 2 / 3)
        self.assertEqual(mixed["slo"][0]["response_slo_goodput_per_s"], 2 / 10)
        other = report["by_workload"]["other"]
        self.assertEqual(other["baseline"]["latency"]["tpot_s"]["missing_successes"], 2)
        self.assertIsNone(other["baseline"]["latency"]["tpot_s"]["p50"])
        self.assertIsNone(other["candidate_minus_baseline"]["latency"]["tpot_s"]["p95"])
        self.assertIsNone(other["candidate_minus_baseline"]["latency"]["tpot_s"]["p99"])

    def test_refuses_partial_or_inconsistent_reports(self):
        mutations = (
            ("partial status", lambda r: r.update(status="partial"), "complete report"),
            ("missing aliases", lambda r: r["missing_request_aliases"].append("missing"), "complete report"),
            ("missing row", lambda r: r["requests"].pop(), "inconsistent"),
            ("duplicate alias", lambda r: r["requests"].append(r["requests"][0]), "duplicate request alias"),
            ("nonfinite makespan", lambda r: r["summary"].update(makespan_s=float("inf")), "finite positive makespan"),
            ("nonfinite TTFT", lambda r: r["requests"][0].update(ttft_s=float("nan")), "nonnegative seconds"),
            ("missing TPOT key", lambda r: r["requests"][0].pop("tpot_s"), "tpot_s"),
            ("length below budget", lambda r: r["requests"][3].update(max_new_tokens=4), "length stop"),
            ("invalid hash", lambda r: r["requests"][0].update(output_ids_sha256="bad"), "output ID hash"),
        )
        for name, mutate, error in mutations:
            with self.subTest(name=name):
                candidate = deepcopy(self.candidate)
                mutate(candidate)
                with self.assertRaisesRegex(ValueError, error):
                    compare_runs(self.baseline, candidate, self.thresholds)

    def test_refuses_different_inputs_or_concurrency(self):
        mutations = (
            ("workload SHA", lambda r: r["source_sha256"].update(workload="1" * 64)),
            ("concurrency", lambda r: r.update(client_concurrency=2)),
            ("alias", lambda r: r["requests"][0].update(request_alias="another")),
            ("input_ids_sha256", lambda r: r["requests"][0].update(input_ids_sha256="1" * 64)),
            ("input_tokens", lambda r: r["requests"][0].update(input_tokens=3)),
            ("max_new_tokens", lambda r: r["requests"][0].update(max_new_tokens=4)),
            ("workload differs", lambda r: r["requests"][0].update(workload="another")),
        )
        for error, mutate in mutations:
            with self.subTest(error=error):
                candidate = deepcopy(self.candidate)
                mutate(candidate)
                with self.assertRaisesRegex(ValueError, error):
                    compare_runs(self.baseline, candidate, self.thresholds)

    def test_refuses_invalid_or_duplicate_slos(self):
        for invalid in ([], [(float("inf"), 1)], [(0, 1)], [(True, 1)], [(1, 1), (1, 1)]):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "SLOs"):
                compare_runs(self.baseline, self.candidate, invalid)

    def test_cli_uses_new_output_and_rejects_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as folder:
            baseline, candidate, output = (Path(folder) / name for name in ("a.json", "b.json", "out.json"))
            baseline.write_text(json.dumps(self.baseline), encoding="utf-8")
            candidate.write_text(json.dumps(self.candidate), encoding="utf-8")
            argv = ["--baseline", str(baseline), "--candidate", str(candidate), "--output", str(output),
                    "--slo", "0.5", "0.25", "--slo", "0.5", "0.125"]
            main(argv)
            original = output.read_bytes()
            self.assertEqual(json.loads(original)["overall"]["candidate"]["successes"], 5)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(argv)
            self.assertEqual(output.read_bytes(), original)
            for invalid in ("NaN", "Infinity", "1e999"):
                candidate.write_text('{"ignored": ' + invalid + '}', encoding="utf-8")
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "finite numbers"):
                    read_run(candidate)
            candidate.write_text('{"ignored": 1, "ignored": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate object key"):
                read_run(candidate)


if __name__ == "__main__":
    unittest.main()
