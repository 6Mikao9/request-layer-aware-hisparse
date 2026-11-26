import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime import shared_lru as s
from runtime.benchmark_shared_lru_scan import (LAYOUTS, RING_STATES, Case, main, owner_summary_words, recency_lists,
                                               ring_rebuilds, selected_blocks)
from runtime.cuda_shared_lru_correctness import FEATURE_SETS, Geometry
from runtime.cuda_shared_lru_production_correctness import GEOMETRY

try:
    import torch
except ImportError:
    torch = None


class SelectionTests(unittest.TestCase):
    def test_production_geometry(self):
        for misses in range(GEOMETRY.blocks + 1):
            blocks = selected_blocks(GEOMETRY, misses)
            tokens = [b * 128 + j for b in blocks for j in range(128)]
            self.assertEqual(len(set(tokens)), 2048)
            self.assertEqual(sum(t >= 4096 for t in tokens), misses * 128)
            self.assertLess(max(tokens), GEOMETRY.context - 1)

    def test_bounds_and_cli_gate(self):
        for misses in (-1, 17):
            with self.assertRaises(ValueError):
                selected_blocks(GEOMETRY, misses)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            path = Path(directory) / "out.json"
            with self.assertRaises(SystemExit):
                main(["--output", str(path)])
            path.write_text("original")
            with self.assertRaises(SystemExit):
                main(["--allow-cuda-test", "--output", str(path)])
            self.assertEqual(path.read_text(), "original")


@unittest.skipIf(torch is None, "requires CPU torch")
class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.g = Geometry(layers=3, hot=16, page=4, slots=2, blocks=2, block=4, dim=8, context=32)

    def fake_launch(self, case):
        # Independent writes through the physical planes. Shared mode steals
        # from plane 0; isolated and stock select victims in the requesting plane.
        g, c = self.g, case.coordinator
        for rid in range(g.slots):
            for i, t in enumerate(case.tokens[rid].tolist()):
                plane = case.plane
                row = t
                if t >= g.hot:
                    plane = 0 if case.arm == "shared" else plane
                    row = g.hot - 1 - (i - case.hits)
                    c.req_device_buffer_tokens[plane, rid, row] = t
                    if case.runtime is not None:
                        case.runtime.row_owner[rid, plane * g.hot + row] = (1 << 16) | case.plane
                loc = int(c.req_device_buffer_token_locs[plane, rid, row])
                if t >= g.hot:
                    for target, host in zip((case.k, case.v), case.host):
                        target[plane, loc].copy_(host[rid * g.context + t])
                case.out[rid, i] = plane * case.rows + loc
        # Exercise the snapshot restoration even for an all-hit launch.
        if case.runtime is None:
            c.lru_slots.zero_()
        else:
            case.runtime.order.zero_()
            case.runtime.occupancy.zero_()

    def test_replay_restores_every_mutable_table_and_payload(self):
        for arm in ("stock", "isolated", "shared"):
            for misses in (0, 1, 2):
                case = Case(torch, arm, self.g, misses, device="cpu")
                pristine = [buf.clone() for buf in (case.k, case.v)]
                for _ in range(2):
                    self.fake_launch(case)
                    self.assertEqual(case.verify(), dict(hits=16 - 8 * misses, misses=8 * misses))
                    case.reset()
                    for buf, original in zip((case.k, case.v), pristine):
                        self.assertTrue(torch.equal(buf, original))
                    for buf, original in case.snapshots:
                        self.assertTrue(torch.equal(buf, original))
                    self.assertTrue(bool((case.out == -1).all()))

    def test_stale_cold_payload_and_wrong_slot_rejected(self):
        case = Case(torch, "shared", self.g, 1, device="cpu")
        self.fake_launch(case)
        case.v.flatten(0, 1)[int(case.out[0, -1])].fill_(-999)
        with self.assertRaisesRegex(RuntimeError, "KV bytes"):
            case.verify()
        case.out[0, -1] = case.out[1, -1]
        with self.assertRaisesRegex(RuntimeError, "wrong request"):
            case.verify()


@unittest.skipIf(torch is None, "requires CPU torch")
class SwitchProbeTests(unittest.TestCase):
    """The probe's switch comparison on CPU: the launch is KernelMirror behind the kernel ABI."""

    def setUp(self):
        self.g = Geometry(layers=4, hot=32, page=4, slots=2, blocks=2, block=4, dim=8, context=64)

    def launch_with_mirror(self, case):
        from runtime import shared_lru_device as device
        from runtime.test_shared_lru_gate import Emulator
        features = tuple(name for name, _ in case.runtime.config.features)
        jit = lambda *variant: Emulator(case.runtime.plan, variant[1], variant[2], variant[5], features)
        with patch.object(device, "_jit_module", jit):
            case.launch()

    def test_switch_sets_reproduce_the_switches_off_state_on_both_layouts(self):
        for arm in ("isolated", "shared"):
            for layout in LAYOUTS:
                for misses in (0, 1, 2):
                    states = {}
                    for name, features in FEATURE_SETS.items():
                        for ring_state in RING_STATES if "incremental_lru" in features else ("template",):
                            with self.subTest(arm=arm, layout=layout, misses=misses, features=name, ring=ring_state):
                                case = Case(torch, arm, self.g, misses, device="cpu", features=features,
                                            layout=layout, ring_state=ring_state)
                                self.launch_with_mirror(case)
                                self.assertEqual(case.verify(), dict(hits=16 - 8 * misses, misses=8 * misses))
                                states[name, ring_state] = case.logical_sha256()
                                raw = case.state_sha256()
                                self.assertEqual(ring_rebuilds(case), 2 if ring_state == "rebuild" else 0)
                                case.reset()
                                self.launch_with_mirror(case)                 # replay from the restored state
                                self.assertEqual(case.state_sha256(), raw)
                    self.assertEqual(len(set(states.values())), 1, f"{arm}/{layout}/{misses}: {states}")

    def test_rebuild_state_is_a_full_ring_with_the_same_recency_list(self):
        for arm in ("isolated", "shared"):
            plain = Case(torch, arm, self.g, 1, device="cpu", features=("incremental_lru",))
            full = Case(torch, arm, self.g, 1, device="cpu", features=("incremental_lru",), ring_state="rebuild")
            r = full.runtime
            domain = s.ring_domain(r.plan, full.plane)
            _, capacity, _, _ = s.ring_domains(r.plan)[domain]
            head, tail = r.ring_meta[0, 4 * domain:4 * domain + 2].tolist()
            self.assertEqual((head, tail + r.plan.selected), (0, capacity + 1))
            self.assertTrue(torch.equal(recency_lists(torch, plain.runtime), recency_lists(torch, r)))
            mirror = s.KernelMirror(r.plan, plane_rows=r.plane_rows, features=("incremental_lru",))
            from runtime.cuda_shared_lru_correctness import device_state
            mirror.slots[0] = dict(device_state(r, full.coordinator.req_device_buffer_tokens, 0),
                                   locs=full.coordinator.req_device_buffer_token_locs[0, 0].tolist())
            from runtime.test_shared_lru_opt import check_structures
            check_structures("rebuild state", mirror, 0)

    def test_fragmented_layout_scatters_every_layer_and_keeps_the_summary_exact(self):
        contiguous = Case(torch, "shared", self.g, 0, device="cpu", features=("owner_summary",))
        case = Case(torch, "shared", self.g, 0, device="cpu", features=("owner_summary",), layout="fragmented")
        self.assertEqual(contiguous.home.tolist(), list(range(96, 128)))
        self.assertGreater(len(set((case.home // 32).tolist())), 2, "measured layer stayed in its home blocks")
        r = case.runtime
        planes = [[None if t < 0 else t & 0xFFFF for t in r.row_owner[slot].tolist()] for slot in range(2)]
        self.assertEqual(sorted(p for p in planes[0] if p is not None), sorted(r.plan.owner_template()))
        words = [[w & (2**64 - 1) for w in r.owner_summary[slot].tolist()] for slot in range(2)]
        self.assertEqual(words, [s.owner_summary(p) for p in planes])
        self.assertTrue(any(bin(w).count("1") > 1 for w in words[0]), "blocks still hold a single layer")
        self.assertTrue(torch.equal(owner_summary_words(torch, r.row_owner, r.generation, 4), r.owner_summary))
        for slot in range(2):                             # recency positions follow the moved rows
            self.assertEqual(sorted(r.order[slot].tolist()), list(range(128)))
        with self.assertRaises(ValueError):
            Case(torch, "stock", self.g, 0, device="cpu", layout="fragmented")
        with self.assertRaises(ValueError):
            Case(torch, "stock", self.g, 0, device="cpu", features=("touched_bitmap",))


if __name__ == "__main__":
    unittest.main()
