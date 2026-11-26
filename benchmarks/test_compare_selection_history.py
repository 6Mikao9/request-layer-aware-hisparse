"""Same-history comparison of logical selections and KV digests on synthetic records."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks import compare_selection_history as check


def record(step, layer, *, history="h", selected=(3, 1, 2, -1), kv="k", request="r0", rank=0):
    return dict(request=request, step=step, layer=layer, rank=rank, history_sha256=f"{history}{step}",
                selected=list(selected), kv_sha256=f"{kv}{step}{layer}")


def index(rows):
    return {(r["request"], r["step"], r["layer"], r["rank"]): r for r in rows}


class SameHistory(unittest.TestCase):
    def test_equal_sets_in_any_order_pass(self):
        a = index(record(step, layer) for step in range(3) for layer in range(2))
        b = index(record(step, layer, selected=(-1, 2, 3, 1)) for step in range(3) for layer in range(2))
        result = check.compare(a, b)
        self.assertEqual((result["status"], result["records"]["same_history"]), ("pass", 6))

    def test_selection_and_kv_mismatches_name_their_first_record(self):
        a = index(record(step, layer) for step in range(3) for layer in range(2))
        b = index([record(0, 0), record(0, 1, kv="x"), record(1, 0, selected=(3, 1, 9)), record(1, 1),
                   record(2, 0, history="other"), record(2, 1, history="other"), record(5, 0)])
        result = check.compare(a, b)
        self.assertEqual((result["status"], result["selection_mismatches"], result["kv_mismatches"]), ("fail", 1, 1))
        self.assertEqual(result["first_divergence"], dict(
            history=dict(request="r0", step=2, layer=0, rank=0), selection=dict(request="r0", step=1, layer=0, rank=0),
            kv=dict(request="r0", step=0, layer=1, rank=0)))
        self.assertEqual((result["records"]["same_history"], result["unpaired"]), (4, dict(a_only=0, b_only=1)))

    def test_cli_reads_jsonl_refuses_malformed_records_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in "ab":
                (tmp / f"{name}.jsonl").write_text("".join(json.dumps(record(s, 0)) + "\n" for s in range(2)))
            argv = ["--a", str(tmp / "a.jsonl"), "--b", str(tmp / "b.jsonl"), "--output", str(tmp / "out.json")]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check.main(argv), 0)
            self.assertEqual(json.loads((tmp / "out.json").read_text())["status"], "pass")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                check.main(argv)
            (tmp / "bad.jsonl").write_text(json.dumps(dict(record(0, 0), selected="1,2")) + "\n")
            with self.assertRaisesRegex(ValueError, "needs exactly"):
                check.read_records(tmp / "bad.jsonl")
        self.assertEqual(check.compare({}, {})["status"], "no_overlap")


if __name__ == "__main__":
    unittest.main()
