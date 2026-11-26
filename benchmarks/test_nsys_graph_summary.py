import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from benchmarks import nsys_graph_summary as summary


def synthetic_trace(path):
    """Minimal nsys SQLite schema: two graph launches (event, kernel, event nodes) and one plain launch."""
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, correlationId INTEGER, nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, correlationId INTEGER,
                                                 streamId INTEGER, graphNodeId INTEGER, demangledName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (start INTEGER, end INTEGER, correlationId INTEGER,
                                                 streamId INTEGER, graphNodeId INTEGER);
        INSERT INTO StringIds VALUES (1, 'cudaGraphLaunch_v10000'), (2, 'cudaLaunchKernel_v7000'),
                                     (3, 'void sglang::rlh_shared_lru::resolve_kernel<1024>'), (4, 'other');
    """)
    rows = [(100, 110, 7, 1), (500, 512, 8, 1), (900, 905, 9, 2)]
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)", rows)
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?)",
                   [(130, 430, 7, 3, 11, 3), (520, 800, 8, 3, 11, 3), (910, 950, 9, 3, None, 4)])
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (?, ?, ?, ?, ?)",
                   [(115, 125, 7, 3, 10), (431, 440, 7, 3, 12), (515, 518, 8, 3, 10), (801, 810, 8, 3, 12)])
    db.commit()
    db.close()


class GraphSummary(unittest.TestCase):
    def test_launches_nodes_order_and_launch_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.sqlite"
            synthetic_trace(trace)
            result = summary.summarize(trace)
            self.assertEqual(result["graph_launches"], 2)                 # the plain launch is not a graph
            self.assertEqual(result["nodes_per_launch"], [3])
            first = result["launches"][0]
            self.assertEqual([n["graph_node"] for n in first["nodes"]], [10, 11, 12])
            self.assertEqual(first["stream_order"], [[10, 11], [11, 12]])
            self.assertEqual((first["launch_queue_ns"], first["gpu_span_ns"], first["max_concurrent_nodes"]),
                             (5, 325, 1))
            self.assertEqual(result["launch_queue_ns"]["median"], 4)
            kernel = result["kernel_ns"]["void sglang::rlh_shared_lru::resolve_kernel<1024>"]
            self.assertEqual((kernel["count"], kernel["median"]), (2, 290))
            out = Path(tmp) / "summary.json"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(summary.main([str(trace), "--output", str(out)]), 0)
                with self.assertRaises(SystemExit):
                    summary.main([str(trace), "--output", str(out)])       # never overwrites
            self.assertEqual(len(json.loads(out.read_text())["source_sha256"]), 64)
            bad = Path(tmp) / "bad.sqlite"
            sqlite3.connect(bad).close()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(summary.main([str(bad), "--output", str(Path(tmp) / "bad.json")]), 1)

    def test_tp_launches_with_identical_correlation_ids_have_separate_timelines(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "tp#two.sqlite"
            synthetic_trace(trace)
            rank0 = (1 << 48) | (128 << 24)
            rank1 = (1 << 48) | (256 << 24)
            with contextlib.closing(sqlite3.connect(trace)) as db, db:
                db.executescript("""
                    ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME ADD COLUMN globalTid INTEGER;
                    ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN globalPid INTEGER;
                    ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN deviceId INTEGER;
                    ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN contextId INTEGER;
                    DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME;
                    DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL;
                    DELETE FROM CUPTI_ACTIVITY_KIND_MEMSET;
                """)
                db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)", [
                    (100, 110, 7, 1, rank0 | 12), (105, 115, 7, 1, rank1 | 31),
                ])
                db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                    (120, 130, 7, 3, 10, 3, rank0, 0, 1),
                    (132, 150, 7, 8, 11, 3, rank0, 0, 1),
                    (151, 165, 7, 3, 12, 3, rank0, 0, 2),
                    (170, 185, 7, 3, 13, 3, rank0, 0, 1),
                    (200, 210, 7, 3, 20, 4, rank1, 1, 1),
                ])
            first, second = summary.summarize(trace)["launches"]
            self.assertEqual(first["global_process_id"], rank0 >> 24)
            self.assertEqual(second["global_process_id"], rank1 >> 24)
            self.assertEqual([node["graph_node"] for node in first["nodes"]], [10, 11, 12, 13])
            self.assertEqual([node["graph_node"] for node in second["nodes"]], [20])
            self.assertEqual(first["stream_order"], [[10, 13]])
            self.assertEqual(second["launch_queue_ns"], 85)

    def test_missing_process_id_does_not_guess_between_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "ambiguous.sqlite"
            synthetic_trace(trace)
            with contextlib.closing(sqlite3.connect(trace)) as db, db:
                db.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME ADD COLUMN globalTid INTEGER")
                db.executemany("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid = ? WHERE correlationId = ?", [
                    ((128 << 24) | 12, 7), ((256 << 24) | 31, 8), ((128 << 24) | 12, 9),
                ])
                db.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET correlationId = 7 WHERE correlationId = 8")
            with self.assertRaisesRegex(ValueError, "missing process identifiers"):
                summary.summarize(trace)

    def test_graph_launch_without_nodes_is_not_a_successful_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace, out = Path(tmp) / "no-nodes.sqlite", Path(tmp) / "result.json"
            synthetic_trace(trace)
            with contextlib.closing(sqlite3.connect(trace)) as db, db:
                db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")
                db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_MEMSET")
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                self.assertEqual(summary.main([str(trace), "--output", str(out)]), 1)
            self.assertIn("no CUDA Graph launches matched GPU nodes", errors.getvalue())
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
