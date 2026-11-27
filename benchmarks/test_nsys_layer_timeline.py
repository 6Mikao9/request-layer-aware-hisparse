"""Nsight layer timelines on synthetic SQLite exports of a lockstep TP decode (issue #3's reducer, issue #11 kinds)."""
import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from benchmarks import nsys_layer_timeline as timeline

US = 1000                                    # the export's timestamps are nanoseconds
COLUMNS = ("start", "end", "deviceId", "streamId", "correlationId", "globalPid", "shortName", "graphNodeId",
           "gridX", "gridY", "gridZ")
CORE = ("layer", "read_path_us", "exposed_wait_us", "read_to_merge_us", "layer_us")
SPLIT = ("swap_us", "attention_us", "collective_us", "other_us", "idle_us")


def step_kernels(device, *, layers, offload, slow, waits=None, missing=None, names=None):
    """(name, µs) of one decode step on one rank in stream order; ``slow`` = (device, layer, read µs). The
    all-reduce after a layer's end lasts 8 µs plus ``waits[(device, layer)]``; ``missing`` = (device, layer) drops
    it. ``names`` overrides (swap, attention, layer end)."""
    swap, attention, end = names or ("load_cache_to_device_buffer_kernel", "_gqa_share_sparse_decode_kernel_impl",
                                     "_merge_topk_attn_out_kernel")
    out = [("embed", 4)]
    for layer in layers:
        out.append(("qkv_gemm", 10))
        if layer in offload:
            out.append((swap, slow[2] if (device, layer) == slow[:2] else 20))
        out += [(attention, 30), (end, 5)]
        if (device, layer) != missing:
            out.append(("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", 8 + (waits or {}).get((device, layer), 0)))
        out.append(("moe_gemm", 15))
    return out


def export(path, *, devices=2, steps=5, layers=(0, 1, 2, 3), offload=(1, 3), slow=(1, 3, 50), side_us=0,
           steady=None, waits=None, missing=None, names=None):
    """Synthetic ``nsys export --type sqlite``: per device two warmup replays of a smaller graph, the steady
    replays (``steady`` overrides their count per device) and an eager kernel after every replay. ``side_us``
    runs a side-stream graph kernel over device 0's first offloaded read."""
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    db.execute(f"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ({', '.join(COLUMNS)})")
    db.execute("CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT)")
    db.execute("INSERT INTO META_DATA_EXPORT VALUES ('EXPORT_SCHEMA_VERSION', '3.17.0')")
    ids, rows, correlation = {}, [], 0
    swap = (names or ("load_cache_to_device_buffer_kernel",))[0]

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
                for value, duration in step_kernels(device, layers=layers, offload=offload, slow=slow, waits=waits,
                                                    missing=missing, names=names):
                    if side and value == swap:                                    # stream 9 beside stream 7
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
    def analyze(self, layers=(0, 1, 2, 3), tp_size=2, profile=None, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            return timeline.analyze(export(Path(tmp) / "report.sqlite", **kwargs), layers=layers, tp_size=tp_size,
                                    profile=profile)

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
        rows = [tuple(layer[f] for f in CORE) for layer in fast["layers"]]
        self.assertEqual(rows, [(0, 0, 0, 35, 49), (1, 20, 20, 55, 88), (2, 0, 0, 35, 68), (3, 20, 20, 55, 88)])
        split = [tuple(layer[f] for f in SPLIT) for layer in fast["layers"]]
        self.assertEqual(split, [(0, 35, 0, 14, 0), (20, 35, 8, 25, 0), (0, 35, 8, 25, 0), (20, 35, 8, 25, 0)],
                         "the layer period split exclusively")
        self.assertEqual(result["bottleneck"]["whole_step_us"]["p50"], 346)
        self.assertEqual({f: result["bottleneck"]["layers"][3][f] for f in CORE},
                         dict(layer=3, read_path_us=50, exposed_wait_us=50, read_to_merge_us=85, layer_us=118))
        self.assertEqual({t["bottleneck_rank"] for t in result["timeline"]}, {1})

    def test_tp_waits_name_the_rank_the_others_waited_for(self):
        """Rank 1 swaps layer 1 for 50 µs instead of 20; rank 0 waits 30 µs at the next all-reduce, which opens
        layer 2's period. Both whole steps are equal, as in a lockstep capture."""
        result = self.analyze(slow=(1, 1, 50), waits={(0, 1): 30})
        self.assertEqual(result["status"], "pass", result["reasons"])
        first, second = result["ranks"]
        self.assertEqual((first["whole_step_us"]["p50"], second["whole_step_us"]["p50"]), (346, 346))
        self.assertEqual((first["tp_wait_us"]["p50"], second["tp_wait_us"]["p50"]), (30, 0))
        self.assertEqual((first["critical_steps"], second["critical_steps"]), (0, 5))
        self.assertEqual([t["critical_rank"] for t in result["timeline"]], [1] * 5)
        self.assertEqual(result["timeline"][0]["tp_wait_us"], [30, 0])
        self.assertEqual([(row["collective_us"], row["tp_wait_us"]) for row in first["layers"]],
                         [(0, 0), (8, 0), (38, 30), (8, 0)])
        critical = result["critical"]
        self.assertEqual([row["read_path_us"] for row in critical["layers"]], [0, 50, 0, 20])
        self.assertEqual([row["critical_ranks"] for row in critical["layers"]], [{}, {0: 5}, {1: 5}, {0: 5}],
                         "no collective before layer 0; ties go to the lowest rank")
        for row in (*first["layers"], *second["layers"], *critical["layers"]):
            self.assertEqual(sum(row[f] for f in SPLIT), row["layer_us"])

    def test_unpaired_collectives_leave_waits_null(self):
        result = self.analyze(missing=(1, 2))
        self.assertEqual(result["status"], "fail")
        self.assertIn("different collectives", result["reasons"][0])
        self.assertIsNone(result["critical"])
        self.assertEqual({row["tp_wait_us"] for row in result["ranks"][0]["layers"]}, {None})

    def test_sweep_counts_each_instant_once(self):
        parts, idle = timeline.sweep([(0, 10, "swap"), (5, 20, "other"), (30, 40, "collective"),
                                      (35, 45, "attention"), (-5, 2, "swap")], 0, 50)
        self.assertEqual((parts, idle), (dict(swap=10, attention=10, collective=5, other=10), 15))

    def test_overlapped_reads_and_explicit_kernel_names(self):
        result = self.analyze(side_us=6)
        second = result["ranks"][0]["layers"][1]
        self.assertEqual((second["read_path_us"], second["exposed_wait_us"]), (20, 14), "6 µs ran beside the read")
        self.assertEqual((second["swap_us"], second["other_us"], second["idle_us"]), (20, 25, 0))
        names = ("load_cache_to_device_buffer_kernel", "flash_mla_decode", "c4_attn_combine")   # another model
        profile = dict(swap=names[:1], attention=names[1:2], merge=names[2:])
        other = self.analyze(names=names, profile=profile)
        self.assertEqual([tuple(row[f] for f in CORE) for row in other["ranks"][0]["layers"]],
                         [tuple(row[f] for f in CORE) for row in self.analyze()["ranks"][0]["layers"]])

    def test_mismatches_are_reasons_and_null_fields(self):
        result = self.analyze(layers=(0, 1, 2, 3, 4), tp_size=4, steady={1: 4})
        self.assertEqual((result["status"], result["steps"], len(result["reasons"])), ("fail", 4, 3))
        self.assertIn("plan runs TP4", result["reasons"][0])
        self.assertIn("[4, 5] times", result["reasons"][1])
        self.assertIn("5 such layers", result["reasons"][2])
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

    def test_cli_needs_a_model_or_every_kernel_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace, output = export(Path(tmp) / "t.sqlite"), Path(tmp) / "timeline.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                timeline.main([str(trace), "--layers", "0-3", "--tp-size", "2", "--attention-kernel", "x",
                               "--output", str(output)])
            self.assertEqual(timeline.main([str(trace), "--layers", "0-3", "--tp-size", "2", "--model", "minimax_m3",
                                            "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text())["steps"], 5)
            self.assertEqual(timeline.parse_layers("2-42:2,0"), (0, *range(2, 43, 2)))

    def test_pair_plot_data(self):
        from benchmarks import plot_layer_timeline as plot
        control, candidate = self.analyze(offload=(0, 1, 2, 3)), self.analyze(offload=(1,), slow=(9, 9, 0))
        data = plot.pair_data(control, candidate, ("Stock", "Frozen"))
        self.assertEqual((data["swapped"], data["tp_size"]), ([1], 2))
        arm = data["arms"][1]
        for j, row in enumerate(candidate["critical"]["layers"]):
            self.assertAlmostEqual(sum(arm["parts"][part][j] for part in plot.PARTS), row["layer_us"])
        shifted = json.loads(json.dumps(candidate))
        for row in shifted["critical"]["layers"]:
            row["layer"] += 1
        with self.assertRaisesRegex(ValueError, "same layers"):
            plot.pair_data(control, shifted)
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            return
        with tempfile.TemporaryDirectory() as tmp:
            plot.draw(data, Path(tmp) / "pair.png")
            self.assertGreater((Path(tmp) / "pair.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
