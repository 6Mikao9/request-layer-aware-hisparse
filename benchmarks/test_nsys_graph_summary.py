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
    def test_launches_nodes_order_and_exposed_wait(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.sqlite"
            synthetic_trace(trace)
            result = summary.summarize(trace)
            self.assertEqual(result["graph_launches"], 2)                 # the plain launch is not a graph
            self.assertEqual(result["nodes_per_launch"], [3])
            first = result["launches"][0]
            self.assertEqual([n["graph_node"] for n in first["nodes"]], [10, 11, 12])
            self.assertEqual(first["stream_order"], [[10, 11], [11, 12]])
            self.assertEqual((first["exposed_wait_ns"], first["gpu_span_ns"], first["max_concurrent_nodes"]),
                             (5, 325, 1))
            self.assertEqual(result["exposed_wait_ns"]["median"], 4)
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


if __name__ == "__main__":
    unittest.main()
