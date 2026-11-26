"""Nsight timeline summaries on synthetic SQLite exports of a lockstep TP decode."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from runtime import combined_timeline as timeline

US = 1000                                    # the export's timestamps are nanoseconds
COLUMNS = ("start", "end", "deviceId", "streamId", "correlationId", "globalPid", "shortName", "graphNodeId",
           "gridX", "gridY", "gridZ")


def step_kernels(device, *, layers, offload, resolver, slow):
    """(name, µs) of one decode step on one rank in stream order; ``slow`` = (device, layer, read µs)."""
    out = [("embed", 4)]
    for layer in layers:
        out.append(("qkv_gemm", 10))
        if layer in offload:
            out.append(("load_cache_to_device_buffer_kernel", slow[2] if (device, layer) == slow[:2] else 20))
        elif resolver:
            out.append(("resolve_resident_kernel", 4))
        out += [("_gqa_share_sparse_decode_kernel_impl", 30), ("_merge_topk_attn_out_kernel", 5),
                ("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", 8), ("moe_gemm", 15)]
    return out


def export(path, *, devices=2, steps=5, layers=(0, 1, 2, 3), offload=(1, 3), resolver=False, slow=(1, 3, 50),
           side_us=0, steady=None):
    """Synthetic ``nsys export --type sqlite``: per device two warmup replays of a smaller graph, the steady
    replays (``steady`` overrides their count per device) and an eager kernel after every replay. ``side_us``
    runs a side-stream graph kernel over device 0's first offloaded read."""
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    db.execute(f"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ({', '.join(COLUMNS)})")
    db.execute("CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT)")
    db.execute("INSERT INTO META_DATA_EXPORT VALUES ('EXPORT_SCHEMA_VERSION', '3.17.0')")
    ids, rows, correlation = {}, [], 0

    def name(value):
        if value not in ids:
            ids[value] = len(ids) + 1
            db.execute("INSERT INTO StringIds VALUES (?, ?)", (ids[value], value))
        return ids[value]

    for device in range(devices):
        clock, pid = 0, 4000 + device
        for grid, count in ((2, 2), (8, (steady or {}).get(device, steps))):
            for _ in range(count):
                correlation += 1
                side = side_us if device == 0 else 0
                for value, duration in step_kernels(device, layers=layers, offload=offload, resolver=resolver,
                                                    slow=slow):
                    if side and value == "load_cache_to_device_buffer_kernel":     # stream 9 beside stream 7
                        rows.append((clock, clock + side * US, device, 9, correlation, pid, name("side"), 2, grid, 1,
                                     1))
                        side = 0
                    rows.append((clock, clock + duration * US, device, 7, correlation, pid, name(value), 1, grid, 1, 1))
                    clock += duration * US
                rows.append((clock + 20 * US, clock + 23 * US, device, 7, 90000 + correlation, pid,
                             name("sampling"), None, 1, 1, 1))
                clock += 50 * US
    db.executemany(f"INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES ({', '.join('?' * len(COLUMNS))})", rows)
    db.commit()
    db.close()
    return path


class Timeline(unittest.TestCase):
    def analyze(self, layers=(0, 1, 2, 3), tp_size=2, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            return timeline.analyze(export(Path(tmp) / "report.sqlite", **kwargs), layers=layers, tp_size=tp_size)

    def test_steady_graph_whole_step_bottleneck_rank_and_its_layers(self):
        result = self.analyze()
        self.assertEqual((result["status"], result["steps"], result["export_schema"]),
                         ("pass", 5, {"EXPORT_SCHEMA_VERSION": "3.17.0"}))
        fast, slow = result["ranks"]
        self.assertEqual((fast["graph_replays"], fast["steady_replays"]), (7, 5), "warmup graph and eager kernels")
        self.assertEqual((fast["whole_step_us"]["p50"], slow["whole_step_us"]["p50"]), (316, 346))
        self.assertEqual((fast["bottleneck_steps"], slow["bottleneck_steps"]), (0, 5))
        self.assertEqual(fast["collective_us"]["p50"], 32)
        self.assertEqual(fast["read_kernels_per_step"], {"load_cache_to_device_buffer_kernel": 2.0})
        rows = [tuple(layer[f] for f in ("layer", *timeline.LAYER_FIELDS)) for layer in fast["layers"]]
        self.assertEqual(rows, [(0, 0, 0, 35, 49), (1, 20, 20, 55, 88), (2, 0, 0, 35, 68), (3, 20, 20, 55, 88)])
        self.assertEqual(result["bottleneck"]["whole_step_us"]["p50"], 346)
        self.assertEqual(result["bottleneck"]["layers"][3], dict(layer=3, read_path_us=50, exposed_wait_us=50,
                                                                 read_to_merge_us=85, layer_us=118))
        self.assertEqual({t["bottleneck_rank"] for t in result["timeline"]}, {1})

    def test_resident_resolver_and_overlapped_reads(self):
        result = self.analyze(resolver=True, side_us=6)
        fast = result["ranks"][0]
        self.assertEqual(fast["read_kernels_per_step"],
                         {"load_cache_to_device_buffer_kernel": 2.0, "resolve_resident_kernel": 2.0})
        self.assertEqual(fast["whole_step_us"]["p50"], 324)
        first, second = fast["layers"][:2]
        self.assertEqual((first["read_path_us"], first["read_to_merge_us"], first["layer_us"]), (4, 39, 53))
        self.assertEqual((second["read_path_us"], second["exposed_wait_us"]), (20, 14), "6 µs ran beside the read")

    def test_mismatches_are_reasons_and_null_fields(self):
        result = self.analyze(layers=(0, 1, 2, 3, 4), tp_size=4, steady={1: 4})
        self.assertEqual((result["status"], result["steps"]), ("fail", 4))
        self.assertEqual(len(result["reasons"]), 3, result["reasons"])
        self.assertIn("plan runs TP4", result["reasons"][0])
        self.assertIn("[4, 5] times", result["reasons"][1])
        self.assertIn("5 sparse layers", result["reasons"][2])
        self.assertIsNone(result["ranks"][0]["layers"])
        self.assertIsNone(result["bottleneck"]["layers"])

    def test_unknown_schema_and_eager_only_captures_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.sqlite"
            db = sqlite3.connect(path)
            db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
            older = ", ".join(c for c in COLUMNS if c != "graphNodeId")
            db.execute(f"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ({older})")
            db.close()
            with self.assertRaisesRegex(timeline.TimelineError, "graphNodeId"):
                timeline.analyze(path, layers=(0,), tp_size=1)
            eager = export(Path(tmp) / "eager.sqlite", devices=1, steps=0)
            db = sqlite3.connect(eager)
            db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId IS NOT NULL")
            db.commit()
            db.close()
            with self.assertRaisesRegex(timeline.TimelineError, "no CUDA Graph kernel nodes"):
                timeline.analyze(eager, layers=(0,), tp_size=1)
            (Path(tmp) / "torn.sqlite").write_bytes(b"not a database" * 100)
            with self.assertRaisesRegex(timeline.TimelineError, "unreadable"):
                timeline.analyze(Path(tmp) / "torn.sqlite", layers=(0,), tp_size=1)

    def test_kernel_kinds_follow_the_pinned_names(self):
        kinds = {name: timeline.kind_of(name) for name in (
            "load_cache_to_device_buffer_kernel", "resolve_kernel", "resolve_resident_kernel",
            "_gqa_share_sparse_decode_kernel_impl_0d1d", "_merge_topk_attn_out_kernel",
            "ncclDevKernel_AllReduce_Sum_bf16_RING_LL", "cross_device_reduce_1stage", "moe_gemm")}
        self.assertEqual(list(kinds.values()), ["read", "read", "read", "attention", "merge", "collective",
                                                "collective", None])


if __name__ == "__main__":
    unittest.main()
