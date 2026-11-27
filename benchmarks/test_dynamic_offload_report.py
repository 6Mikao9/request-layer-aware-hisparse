"""The issue #11 report on a synthetic recipe output directory."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks import dynamic_offload_report as report
from benchmarks.test_nsys_layer_timeline import export


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


def client(hashes, tokens_per_s=100.0, tpot=0.03, status="complete"):
    requests = [dict(request_alias=a, successful=h is not None, output_ids_sha256=h, ttft_s=1.0, tpot_s=tpot,
                     finish_reason="stop") for a, h in hashes.items()]
    return dict(status=status, requests=requests, summary=dict(
        requests=len(requests), failures=sum(h is None for h in hashes.values()),
        completed_output_tokens_per_s=tokens_per_s, natural_eos_requests=len(requests), length_limited_requests=0,
        slo=dict(natural_eos_slo_goodput_per_s=1.0, response_slo_goodput_per_s=1.0)))


def receipt(event, rank, **fields):
    return json.dumps(dict(event=f"hisparse_dynamic_offload_{event}", rank=rank, **fields), sort_keys=True)


def decode_log(batch, lines, *extra):
    rows = [f"[TP0] Decode batch, #running-req: {batch}, #token: 9, token usage: 0.1, cuda graph: True, "
            "gen throughput (token/s): 1.00, #queue-req: 0"] * lines
    return "\n".join([*extra, *rows]) + "\n"


class Report(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = root = Path(self.tmp.name)
        same = dict(a="h1", b="h2", c="h3")
        for label in ("stock", "stock_repeat", "frozen_a", "dynamic-none"):
            write(root / "correctness" / f"{label}.b1.json", client(same))
            write(root / "correctness" / f"{label}.b3.json", client(same))
        write(root / "correctness" / "dynamic-b-to-a.b1.json", client(dict(same, b="other")))
        write(root / "correctness" / "dynamic-live.b1.json", client(dict(same, c=None)))
        write(root / "correctness-dynamic.server.log", "\n".join([
            "Capture decode CUDA graph begin.", receipt("bound", 0, logical_kv=dict(tokens=9)),
            receipt("switch", 0, status="committed", target="1", rows=dict(prepare=4, apply=8),
                    seconds=dict(prepare=0.1, apply=0.2)), receipt("switch", 0, status="refused", target="0")]))
        write(root / "correctness-dynamic.captures-after-switch.txt", "0\n")
        for repeat, (frozen, stock) in enumerate(((120.0, 100.0), (126.0, 100.0), (110.0, 100.0), (500.0, 100.0)), 1):
            write(root / f"frozen_a-r{repeat}-c16.json", client(same, frozen, 0.025))
            write(root / f"stock-r{repeat}-c16.json", client(same, stock, 0.030))
            write(root / f"frozen_a-r{repeat}.server.log", decode_log(16, 8 if repeat < 4 else 2))
            write(root / f"stock-r{repeat}.server.log", decode_log(16, 9))
        write(root / "dynamic-r1.server.log", decode_log(16, 8, receipt("switch", 3, status="committed", target="1")))
        write(root / "dynamic-r1.switch1-time.json", dict(http_seconds=0.5))
        write(root / "stock-r1.gpu-memory.csv", "timestamp, index, memory.used [MiB], memory.total [MiB]\n"
              "t, 0, 70000 MiB, 81920 MiB\nt, 1, 71000 MiB, 81920 MiB\nt, 0, 72000 MiB, 81920 MiB\n")
        write(root / "stock-r1.server-info.json", dict(internal_states=[dict(max_total_num_tokens=123)]))
        write(root / "gate.json", dict(passed=True))

    def test_correctness_flags_mismatches_failures_and_switches(self):
        out = report.correctness(self.root)
        self.assertEqual(out["passes"]["dynamic-b-to-a"]["b1"], dict(compared=3, identical=2, mismatched=["b"],
                                                                     failed=[]))
        self.assertEqual(out["passes"]["dynamic-live"]["b1"]["failed"], ["c"])
        self.assertEqual(out["passes"]["frozen_a"]["b3"]["identical"], 3)
        self.assertEqual((out["b1_identical"], out["b3_noise_mismatches"], out["no_capture_after_switch"]),
                         (False, 0, True))
        server = out["servers"]["correctness-dynamic"]
        self.assertEqual([s["status"] for s in server["switches"]], ["committed", "refused"])
        self.assertEqual(server["switches"][0]["rows"], dict(prepare=4, apply=8))

    def test_qualified_pairs_and_their_interval(self):
        out = report.serving(self.root)
        pair = next(c for c in out["comparisons"] if (c["candidate"], c["baseline"]) == ("frozen_a", "stock"))
        self.assertEqual((pair["qualified_repeats"], pair["unqualified_repeats"]), ([1, 2, 3], [4]))
        changes = [0.2, 0.26, 0.1]
        half = 4.303 * (sum((x - sum(changes) / 3) ** 2 for x in changes) / 2) ** 0.5 / 3 ** 0.5
        self.assertAlmostEqual(pair["output_tokens_per_s"]["mean"], sum(changes) / 3)
        for got, want in zip(pair["output_tokens_per_s"]["ci95"], (sum(changes) / 3 - half, sum(changes) / 3 + half)):
            self.assertAlmostEqual(got, want)
        self.assertAlmostEqual(pair["tpot_p50"]["mean"], 0.025 / 0.03 - 1)
        run = out["runs"]["frozen_a"][16][4]
        self.assertEqual((run["graph_lines_at_batch"], run["qualified"]), (2, False))

    def test_memory_switch_and_cli(self):
        memory = report.memory(self.root)
        self.assertEqual((memory["stock-r1"]["card_peak_mib"], memory["stock-r1"]["max_total_num_tokens"]),
                         ({0: 72000.0, 1: 71000.0}, 123))
        self.assertEqual(memory["correctness-dynamic"]["ranks"][0]["logical_kv"], dict(tokens=9))
        switch = report.switches(self.root)["dynamic-r1"]
        self.assertEqual((switch["http"], switch["receipts"][0]["rank"]), ([dict(http_seconds=0.5)], 3))
        self.assertIn("pass --attention-kernel", report.nsys(self.root, "deepseek_v4")["skipped"])
        export(self.root / "frozen_a-nsys-77.sqlite")
        timeline = report.nsys(self.root, "minimax_m3")["timelines"]["frozen_a"]
        self.assertIn("TP8", timeline["reasons"][0])                  # the synthetic capture has two ranks
        output = self.root / "report.json"
        self.assertEqual(report.main(["all", "--dir", str(self.root), "--output", str(output)]), 0)
        self.assertEqual(json.loads(output.read_text())["gate"], True)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            report.main(["all", "--dir", str(self.root), "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
