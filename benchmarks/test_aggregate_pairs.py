import contextlib
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.aggregate_pairs import WINDOW_METRIC, aggregate_pairs, main
from benchmarks.sglang_stream_client import Generation, digest_ids, summarize


def record(alias, count, *, first=0.5, step=0.125, skip=(), failed=False, length=False, shift=0):
    inputs = [10, ord(alias[0])]
    generation = Generation(alias, inputs, 40)
    finish = None if failed else ("length" if length else "stop")
    for end in range(1, count + 1):
        if end in skip:
            continue
        generation.consume(json.dumps({"output_ids": list(range(100 + shift, 100 + shift + end)),
                                       "meta_info": {"completion_tokens": end,
                                                     "finish_reason": finish if end == count else None}}),
                           first + (end - 1) * step)
    return generation.metrics(8, not failed) | {
        "request_alias": alias, "workload": "mixed", "input_tokens": len(inputs), "max_new_tokens": 40,
        "input_ids_sha256": digest_ids(inputs), "successful": not failed,
    } | ({"error": "transport"} if failed else {})


def report(rows, duration):
    return {"status": "complete", "missing_request_aliases": [], "client_concurrency": 1,
            "source_sha256": {"workload": "1" * 64, "client": "2" * 64},
            "requests": rows, "summary": summarize(rows, duration, 1, 0.2)}


class AggregatePairsTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.a = report([
            record("a-private-alias", 34, skip=(11,)),
            record("b-private-alias", 40, skip=(7,), length=True),
            record("c-private-alias", 2),
            record("d-private-alias", 34, failed=True),
            record("e-private-alias", 36),
        ], 20)
        self.b = report([
            record("e-private-alias", 38),
            record("d-private-alias", 34),
            record("c-private-alias", 3),
            record("b-private-alias", 40, length=True),
            record("a-private-alias", 34, first=0.25, step=0.0625, shift=5),
        ], 10)
        # A cumulative report without the full final sequence cannot supply a fixed-window observation.
        self.a["requests"][4]["token_events"].pop()
        self.plan = {"slo": [[0.5, 0.125], [0.25, 0.0625]], "token_window": [8, 32], "pairs": []}
        self.add_pair("train1", self.a, self.b)

    def save(self, name, value):
        (self.directory / name).write_text(json.dumps(value), encoding="utf-8")

    def add_pair(self, label, left, right, *, order="AB", split="train", comparison="ab"):
        pair = {"label": label, "split": split, "order": order, "comparison": comparison,
                "a": f"{label}-a.json", "b": f"{label}-b.json"}
        self.save(pair["a"], left)
        self.save(pair["b"], right)
        self.plan["pairs"].append(pair)

    def aggregate(self):
        return aggregate_pairs(self.plan, self.directory)

    def test_fixed_window_requires_observed_endpoints_and_preserves_slo_denominators(self):
        result = self.aggregate()
        block = result["blocks"][0]
        self.assertEqual(result["environment"]["status"], "unverified")
        self.assertNotIn("private-alias", json.dumps(result))
        self.assertNotIn("train1-a.json", json.dumps(result))
        self.assertEqual(block["a"]["token_window_counts"], {
            "eligible": 1, "coalesced_boundary": 1, "short_output": 1, "failed": 1, "incomplete_events": 1,
        })
        self.assertEqual(block["b"]["token_window_counts"], {"eligible": 4, "short_output": 1})
        window = block["paired_latency_b_minus_a"][WINDOW_METRIC]
        self.assertEqual((window["total_requests"], window["included"], window["excluded"]), (5, 1, 4))
        self.assertEqual((window["a_mean"], window["b_mean"], window["mean"]), (0.125, 0.0625, -0.0625))
        self.assertEqual(block["a"]["all_observed_output_tokens"], 146)
        self.assertEqual(block["a"]["completed_output_tokens"], 112)
        self.assertEqual(block["a"]["mean_observed_output_tokens_per_request"], 146 / 5)
        self.assertEqual(block["a"]["mean_completed_output_tokens_per_success"], 28)
        self.assertEqual(block["a"]["natural_eos_requests"], 3)
        self.assertEqual(block["a"]["failures"], 1)
        self.assertEqual(block["a"]["slo"][0]["natural_eos_compliant_requests"], 3)
        self.assertEqual(block["a"]["slo"][0]["response_compliant_fraction_all_requests"], 4 / 5)
        self.assertEqual(block["paired_latency_b_minus_a"]["ttft_s"]["included"], 4)
        self.assertEqual(block["paired_latency_b_minus_a"]["ttft_s"]["mean"], -0.0625)
        self.assertEqual(block["paired_latency_b_minus_a"]["end_to_end_s"]["zero"], 4)
        self.assertEqual(block["output_id_hash_pairs"], {"equal": 1, "different": 3, "unavailable": 1})

    def test_repeats_keep_ab_and_aa_separate_and_do_not_add_distinct_inputs(self):
        self.add_pair("train2", self.b, self.a, order="BA")
        self.add_pair("control1", self.b, self.b, comparison="aa", order="BA")
        self.add_pair("control2", self.a, self.b, comparison="aa")
        self.add_pair("holdout1", self.a, self.b, split="holdout")
        result = self.aggregate()
        self.assertEqual(result["distinct_input_hashes_by_split"], {"train": 5, "holdout": 5})
        self.assertEqual(result["input_hashes_shared_by_train_and_holdout"], 5)
        holdout, train = result["strata"]
        self.assertEqual(holdout["aa_noise"]["blocks"], 0)
        self.assertEqual(train["distinct_input_hashes"], 5)
        self.assertEqual(train["ab"]["order_counts"], {"AB": 1, "BA": 1})
        ab = train["ab"]["block_deltas_b_minus_a"][WINDOW_METRIC]
        self.assertEqual((ab["mean"], ab["median"], ab["negative"], ab["positive"]), (0, 0, 1, 1))
        aa = train["aa_noise"]["block_deltas_b_minus_a"][WINDOW_METRIC]
        self.assertEqual((aa["included"], aa["min"], aa["max"], aa["median"]), (2, -0.0625, 0, -0.03125))

    def test_missing_and_coalesced_boundary_excluded_without_interpolation(self):
        a = deepcopy(self.a)
        a["requests"][0].pop("token_events")
        a["requests"][4] = record("e-private-alias", 36, skip=(32,))
        self.save(self.plan["pairs"][0]["a"], a)
        block = self.aggregate()["blocks"][0]
        self.assertEqual(block["a"]["token_window_counts"]["missing_events"], 1)
        self.assertEqual(block["a"]["token_window_counts"]["coalesced_boundary"], 2)
        self.assertIsNone(block["paired_latency_b_minus_a"][WINDOW_METRIC]["mean"])

    def test_short_request_slo_latency_must_agree_with_available_events(self):
        for replacement, reason in (
            (record("c-private-alias", 2) | {"tpot_s": 0.01}, "TPOT differs from complete token events"),
            (record("c-private-alias", 2, skip=(1,)) | {"tpot_s": 0.01, "first_event_tokens": 1},
             "first event count differs"),
        ):
            with self.subTest(reason=reason):
                a = deepcopy(self.a)
                a["requests"][2] = replacement
                self.save(self.plan["pairs"][0]["a"], a)
                with self.assertRaisesRegex(ValueError, reason):
                    self.aggregate()

    def test_refuses_incompatible_inputs_and_inconsistent_event_streams(self):
        cases = (
            (lambda b: b.update(status="partial"), "complete report"),
            (lambda b: b["source_sha256"].update(client="3" * 64), "client source SHA-256"),
            (lambda b: b["source_sha256"].update(workload="3" * 64), "workload SHA-256"),
            (lambda b: b.update(client_concurrency=3), "client concurrency"),
            (lambda b: b["requests"][0].update(max_new_tokens=41), "max_new_tokens"),
            (lambda b: b["requests"][0]["token_events"][8].update(new_tokens=2), "cumulative token event counts"),
            (lambda b: b["requests"][0]["token_events"][8].update(arrival_s=0), "ordered finite event arrivals"),
        )
        for modify, expected in cases:
            with self.subTest(expected=expected):
                b = deepcopy(self.b)
                modify(b)
                self.save(self.plan["pairs"][0]["b"], b)
                with self.assertRaisesRegex(ValueError, expected):
                    self.aggregate()

    def test_refuses_reusing_report_and_conflicting_repeated_workload(self):
        a, b = deepcopy(self.a), deepcopy(self.b)
        a["requests"][0]["input_ids_sha256"] = b["requests"][-1]["input_ids_sha256"] = "a" * 64
        self.add_pair("train2", a, b)
        with self.assertRaisesRegex(ValueError, "repeated workload SHA-256"):
            self.aggregate()
        self.plan["pairs"][1]["a"] = self.plan["pairs"][0]["b"]
        with self.assertRaisesRegex(ValueError, "reused across pair slots"):
            self.aggregate()

    def test_cli_reads_relative_paths_and_refuses_overwrite_and_unregistered_window(self):
        self.save("plan.json", self.plan)
        plan, output = (self.directory / name for name in ("plan.json", "out.json"))
        argv = ["--plan", str(plan), "--output", str(output)]
        main(argv)
        original = output.read_bytes()
        self.assertEqual(json.loads(original)["plan_sha256"], sha256(plan.read_bytes()).hexdigest())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(argv)
        self.assertEqual(output.read_bytes(), original)
        self.plan["token_window"] = [8, 31]
        with self.assertRaisesRegex(ValueError, "fixed token ordinals"):
            self.aggregate()


if __name__ == "__main__":
    unittest.main()
