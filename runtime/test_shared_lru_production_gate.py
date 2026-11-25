"""CPU checks for the production-geometry correctness fixture."""
import subprocess
import sys
import unittest

from runtime import cuda_shared_lru_production_correctness as gate
from runtime import shared_lru as reference
from runtime import study_config as sc


class ProductionGate(unittest.TestCase):
    def test_selection_reaches_full_shared_domain_and_cross_plane_payload(self):
        g = gate.GEOMETRY
        self.assertEqual((g.layers, g.hot, g.blocks * g.block, g.page, g.dim, g.slots),
                         (60, 4096, 2048, 128, 128, 2))
        config = sc.parse({"schema": sc.SCHEMA, "hot_pool": gate.ARMS["shared"]})
        plan = reference.plan_arena(config, range(g.layers), g.hot, g.blocks * g.block)
        mirror = reference.KernelMirror(plan, plane_rows=g.page + g.slots * (g.hot + g.page))
        mirror.init_slot(0, 0, list(range(g.page, g.hot + 2 * g.page)))
        last = g.layers - 1
        self.assertEqual(plan.domain(last), (0, 245760))
        hot, cold, mixed = gate.selections(g)
        seq = (cold[-1] + 1) * g.block + 4
        self.assertLess(seq, g.context)
        flat, copies, counts = mirror.resolve_row(0, last, cold, g.block, seq)
        self.assertEqual(counts["misses"], 2048)
        self.assertEqual(len(copies), 2048)
        self.assertTrue(all(mirror.address(loc)[0] == 0 for loc in flat))
        _, _, counts = mirror.resolve_row(0, 0, hot, g.block, seq)
        self.assertEqual(counts["misses"], 2048)
        _, _, counts = mirror.resolve_row(0, last, cold, g.block, seq)
        self.assertEqual(counts["hits"], 2048)
        tokens = reference.selected_tokens(mixed, g.block, seq)
        self.assertIn(seq - 1, tokens)
        self.assertIn(-1, tokens)
        self.assertEqual(len(tokens), 2048)
        self.assertLess(len(set(tokens)), len(tokens))
        self.assertEqual(mirror.status, 0)

    def test_switched_mirrors_follow_the_production_sequence(self):
        """60 planes, H4096: the gate's sequence gives the same rows and tables with both switches on."""
        g = gate.GEOMETRY
        config = sc.parse({"schema": sc.SCHEMA, "hot_pool": gate.ARMS["shared"]})
        plan = reference.plan_arena(config, range(g.layers), g.hot, g.blocks * g.block)
        rows = g.page + g.slots * (g.hot + g.page)
        mirrors = [reference.KernelMirror(plan, plane_rows=rows, features=features)
                   for features in ((), ("owner_summary", "touched_bitmap"))]
        for mirror in mirrors:
            mirror.init_slot(0, 0, list(range(g.page, g.hot + 2 * g.page)))
        hot, cold, mixed = gate.selections(g)
        seq, last = (cold[-1] + 1) * g.block + 4, g.layers - 1
        for plane, blocks in ((last, cold), (0, hot), (last, cold), (last, mixed), (17, mixed)):
            results = [m.resolve_row(0, plane, blocks, g.block, seq) for m in mirrors]
            self.assertEqual(results[0], results[1])
            for table in ("owner", "order", "occ", "token"):
                self.assertEqual(mirrors[0].slots[0][table], mirrors[1].slots[0][table])
            self.assertEqual(mirrors[1].slots[0]["summary"], mirrors[1].live_summary(0))
        summary = mirrors[1].slots[0]["summary"]
        self.assertEqual(len(summary), 7680)
        self.assertGreater(sum(w >> last & 1 for w in summary), 64, "the last layer borrowed rows of plane 0")
        self.assertLess(sum(w >> 17 & 1 for w in summary), 7680 // 2, "a layer's bit stays in few blocks")

    def test_cuda_opt_in_precedes_torch_import(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c", "import sys; sys.modules['torch'] = None; "
             "from runtime.cuda_shared_lru_production_correctness import main; main([])"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --allow-cuda-test", result.stderr)


if __name__ == "__main__":
    unittest.main()
