"""CPU semantic fixtures; real kernel/model validation is a separate GPU gate."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import torch

from runtime.selection_oracle import SelectionOracle, reference_lru


class OracleTests(unittest.TestCase):
    def fixture(self, seq=12):
        hmax, active, width = 8, 6, 2
        kv = torch.full((32, 1, 3), -11, dtype=torch.bfloat16)
        vv = kv.clone()
        locs = torch.full((2, hmax + 2), -1, dtype=torch.int32)
        locs[0, :hmax + 1] = torch.arange(1, hmax + 2)
        tokens = torch.full_like(locs, -1)
        tokens[0, :hmax] = torch.arange(hmax)
        lru = torch.arange(hmax, dtype=torch.int16).repeat(2, 1)
        host_k = torch.arange(60, dtype=torch.bfloat16).reshape(20, 1, 3)
        host_v = -host_k - 2
        host_locs = torch.arange(20, dtype=torch.int64).repeat(2, 1)
        for index in range(min(seq, hmax)):
            kv[locs[0, index]] = host_k[index]
            vv[locs[0, index]] = host_v[index]
        if seq > hmax:
            kv[locs[0, hmax]] = torch.tensor([[[101, 102, 103]]]).reshape(1, 3)
            vv[locs[0, hmax]] = -kv[locs[0, hmax]]
        selections = [0, (seq - 1) // width, -1]
        return dict(top_k_blocks=torch.tensor([selections, [-1] * 3], dtype=torch.int32),
            device_buffer_tokens=tokens, device_buffer_locs=locs, lru_slots=lru,
            device_buffer_k=kv, device_buffer_v=vv, host_cache_k=host_k, host_cache_v=host_v,
            host_cache_locs=host_locs, top_k_device_locs=torch.full((2, 6), -7, dtype=torch.int32),
            req_pool_indices=torch.tensor([0, 999]), seq_lens=torch.tensor([seq, 0]),
            num_real_reqs=torch.tensor([1], dtype=torch.int32), hot_buffer_size=hmax,
            active_hot_buffer_size=active, sparse_block_size=width, skip_io=False)

    @staticmethod
    def correct_live(**kw):
        # Simple independent CPU implementation, only to exercise oracle gates.
        seq, hmax, active = int(kw["seq_lens"][0]), kw["hot_buffer_size"], kw["active_hot_buffer_size"]
        out = kw["top_k_device_locs"]
        out.fill_(-1)
        available = iter(range(active))
        for i, block in enumerate(kw["top_k_blocks"][0].tolist()):
            for offset in range(kw["sparse_block_size"]):
                token = block * kw["sparse_block_size"] + offset
                if block < 0 or token >= seq:
                    continue
                if seq <= hmax:
                    loc = kw["device_buffer_locs"][0, token]
                elif token == seq - 1:
                    loc = kw["device_buffer_locs"][0, hmax]
                else:
                    slot = next(available)
                    loc = kw["device_buffer_locs"][0, slot]
                    host_row = kw["host_cache_locs"][0, token]
                    kw["device_buffer_k"][loc] = kw["host_cache_k"][host_row]
                    kw["device_buffer_v"][loc] = kw["host_cache_v"][host_row]
                    kw["device_buffer_tokens"][0, slot] = token
                out[0, i * kw["sparse_block_size"] + offset] = loc
        if seq > hmax:
            # Fixture [0,1,10,latest] hits slots0/1 and replaces stale slot2.
            # Fixed expected order is independent of reference_lru().
            kw["lru_slots"][0, :active] = torch.tensor([3, 4, 5, 2, 0, 1])

    def run_case(self, kwargs, live):
        with TemporaryDirectory() as folder:
            engine = SelectionOracle(torch, folder, 0, 4, synchronize=lambda: None, allocated_bytes=lambda: 0)
            engine.check(59, kwargs, live, None, final_layer=59)
            report = json.loads((Path(folder) / "rank-0.json").read_text())
            return report

    def test_historical_host_rows_and_latest_gpu_prestate(self):
        kwargs = self.fixture()
        report = self.run_case(kwargs, self.correct_live)
        self.assertEqual(report["mismatches"], 0)
        self.assertEqual(report["layers"][0]["active_capacity"], 6)
        self.assertEqual(report["layers"][0]["long_rows"], 1)
        self.assertEqual(report["layers"][0]["token_replacement_calls"], 1)
        # The latest host row intentionally differs from the GPU reference.
        self.assertFalse(torch.equal(kwargs["host_cache_k"][11], kwargs["device_buffer_k"][9]))

    def test_short_path_and_exact_threshold(self):
        for seq in (7, 8):
            with self.subTest(seq=seq):
                report = self.run_case(self.fixture(seq), self.correct_live)
                self.assertEqual(report["mismatches"], 0)
                self.assertEqual(report["layers"][0]["short_rows"], 1)

    def test_wrong_payload_invalid_padding_and_latest_routing_are_rejected(self):
        def bad(**kw):
            self.correct_live(**kw)
            kw["device_buffer_k"][1, 0, 0] += 1
            kw["top_k_device_locs"][0, 5] = 0
            kw["top_k_device_locs"][0, 3] = 2
        report = self.run_case(self.fixture(), bad)
        self.assertGreater(report["mismatches"], 0)
        for name in ("selected_k", "selected_validity", "selected_routing"):
            self.assertEqual(report["checks"][name]["failed"], 1)

    def test_inactive_slot_hit_is_rejected_even_when_payload_matches(self):
        def bad(**kw):
            self.correct_live(**kw)
            kw["device_buffer_k"][8] = kw["device_buffer_k"][1]
            kw["device_buffer_v"][8] = kw["device_buffer_v"][1]
            kw["top_k_device_locs"][0, 0] = 8
        report = self.run_case(self.fixture(), bad)
        self.assertEqual(report["checks"]["selected_k"]["failed"], 0)
        self.assertEqual(report["checks"]["selected_routing"]["failed"], 1)
        self.assertEqual(report["checks"]["active_slot_identity"]["failed"], 1)

    def test_missing_host_mapping_fails_before_live(self):
        kwargs = self.fixture()
        kwargs["host_cache_locs"][0, 10] = -1
        calls = []
        with self.assertRaisesRegex(RuntimeError, "allocated host KV"):
            self.run_case(kwargs, lambda **kw: calls.append(True))
        self.assertEqual(calls, [])

    def test_duplicate_selection_rejected(self):
        kwargs = self.fixture()
        kwargs["top_k_blocks"][0, 1] = 0
        with self.assertRaisesRegex(RuntimeError, "unique"):
            self.run_case(kwargs, self.correct_live)

    def test_reference_transition_preserves_hit_order_and_replaces_oldest(self):
        self.assertEqual(reference_lru([0, 1, 2, 3], [3, 2, 1, 0], [1, 9, 0]),
                         ([0, 1, 2, 9], [2, 3, 1, 0], [1, 3, 0], [3]))

    def test_wrong_lru_unselected_tags_and_payload_are_rejected(self):
        def bad(**kw):
            self.correct_live(**kw)
            kw["lru_slots"][0, :6] = kw["lru_slots"][0, :6].flip(0)
            kw["device_buffer_tokens"][0, 4] = 999
            kw["device_buffer_k"][5, 0, 0] = 999
        report = self.run_case(self.fixture(), bad)
        for key in ("lru_order", "active_tags", "unchanged_tail"):
            self.assertEqual(report["checks"][key]["failed"], 1)


if __name__ == "__main__":
    unittest.main()
