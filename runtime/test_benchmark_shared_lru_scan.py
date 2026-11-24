import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from runtime.benchmark_shared_lru_scan import Case, main, selected_blocks
from runtime.cuda_shared_lru_correctness import Geometry
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


if __name__ == "__main__":
    unittest.main()
