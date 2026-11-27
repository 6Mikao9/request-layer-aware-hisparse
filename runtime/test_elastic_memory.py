import unittest

try:
    import torch
except ImportError:  # pragma: no cover - CPU suite without torch
    torch = None

from runtime import elastic_memory as em

KIB = 1024


def specs(layers=(0, 1, 2), always=1, row=256, page=4):
    out = [em.BufferSpec(f"dense{i}", row, em.ALWAYS, pad_rows=page, shape=(row // 2,), dtype="bfloat16")
           for i in range(always)]
    for layer in layers:
        out += [em.BufferSpec(f"{kind}{layer}", row, em.SWITCHABLE, layer=layer, pad_rows=page,
                              shape=(row // 2,), dtype="bfloat16") for kind in "kv"]
    return out


@unittest.skipIf(torch is None, "torch is not installed")
class LayoutTests(unittest.TestCase):
    def layout(self, **kw):
        return em.Layout(specs(**kw), chunk_bytes=4 * KIB, max_tokens=256, hot_rows=32, page_tokens=4)

    def test_resident_region_starts_on_a_chunk(self):
        layout = self.layout()
        spec = layout.specs[1]
        self.assertEqual(layout.hot_offset(spec) * spec.row_bytes % layout.chunk, 0)
        self.assertGreaterEqual(layout.hot_offset(spec), 4 + 32)

    def test_offloading_frees_chunks_for_more_tokens(self):
        layout = self.layout()
        chunks = layout.chunks_needed(tokens=128, hot_rows=32, offloaded=frozenset())
        resident = layout.max_tokens_for(chunks, hot_rows=32, offloaded=frozenset())
        one = layout.max_tokens_for(chunks, hot_rows=32, offloaded=frozenset({0}))
        two = layout.max_tokens_for(chunks, hot_rows=32, offloaded=frozenset({0, 1}))
        self.assertGreaterEqual(resident, 128)          # the last partial chunk holds a few more tokens
        self.assertLess(resident, one)
        self.assertLess(one, two)
        self.assertLessEqual(layout.chunks_needed(tokens=two, hot_rows=32, offloaded=frozenset({0, 1})), chunks)

    def test_unaligned_rows_share_one_boundary_chunk(self):
        odd = [em.BufferSpec("c4", 37440, em.SWITCHABLE, layer=0, tokens_per_row=4, pad_rows=1)]
        layout = em.Layout(odd, chunk_bytes=2 * KIB * KIB, max_tokens=4096, hot_rows=64, page_tokens=64)
        self.assertEqual(layout.hot_offset(odd[0]), 65)
        both = layout.chunks_needed(tokens=4096, hot_rows=64, offloaded=frozenset({0}), filling=(0,))
        hot = layout.chunks_needed(tokens=4096, hot_rows=64, offloaded=frozenset({0}))
        resident = layout.chunks_needed(tokens=4096, hot_rows=64, offloaded=frozenset())
        self.assertLess(both, hot + resident)          # the straddling chunk is counted once


@unittest.skipIf(torch is None, "torch is not installed")
class ElasticMemoryTests(unittest.TestCase):
    def memory(self, physical=48):
        layout = em.Layout(specs(), chunk_bytes=4 * KIB, max_tokens=256, hot_rows=32, page_tokens=4)
        return em.ElasticMemory(layout, em.CpuBackend(torch), physical, torch=torch)

    def test_offload_keeps_stale_rows_until_their_chunks_are_needed(self):
        mem = self.memory()
        mem.resize(tokens=64, hot_rows=32, offloaded=frozenset())
        mem.check()
        res = mem.view("k0", "resident")
        res[: 4 + 64] = torch.arange(68, dtype=torch.bfloat16)[:, None]
        dense = mem.view("dense0")
        dense[: 4 + 64] = 7
        mem.resize(tokens=64, hot_rows=32, offloaded=frozenset({0}))
        mem.check()
        self.assertEqual(mem.stale_rows(0), 80)         # 68 rows, backed through the end of their fifth chunk
        self.assertTrue(torch.equal(mem.view("k0", "resident")[: 68, 0], torch.arange(68, dtype=torch.bfloat16)))
        self.assertTrue(bool((mem.view("k0", "hot")[: 36] == 0).all()))     # fresh hot rows are zeroed
        free = mem.free_chunks
        # Growing past the free pool reclaims stale chunks from the top of layer 0's region.
        target = mem.layout.max_tokens_for(len(mem.handles), hot_rows=32, offloaded=frozenset({0}))
        mem.resize(tokens=target, hot_rows=32, offloaded=frozenset({0}))
        mem.check()
        self.assertGreater(target, 64)
        self.assertLess(mem.stale_rows(0), 68)
        self.assertTrue(bool((mem.view("dense0")[: 68] == 7).all()))        # live rows keep their data
        self.assertLessEqual(mem.free_chunks, free)

    def test_backfill_backs_both_regions_then_releases_hot(self):
        mem = self.memory()
        mem.resize(tokens=64, hot_rows=32, offloaded=frozenset({1}))
        before = mem.used_chunks
        mem.resize(tokens=64, hot_rows=32, offloaded=frozenset({1}), filling=(1,))
        mem.check()
        self.assertGreater(mem.used_chunks, before)
        mem.view("v1", "resident")[:68] = 3
        mem.resize(tokens=64, hot_rows=32, offloaded=frozenset())
        mem.check()
        self.assertTrue(bool((mem.view("v1", "resident")[:68] == 3).all()))
        self.assertTrue(bool((mem.view("v1", "hot").view(torch.uint8) == em.POISON).all()))

    def test_a_state_that_does_not_fit_is_refused(self):
        mem = self.memory(physical=6)
        with self.assertRaises(MemoryError):
            mem.resize(tokens=256, hot_rows=32, offloaded=frozenset())
        self.assertEqual(mem.used_chunks, 0)


if __name__ == "__main__":
    unittest.main()
