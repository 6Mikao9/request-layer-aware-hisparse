"""CPU evidence that the resolver switches keep hits, victims, payloads and tables unchanged.

* ``KernelMirror`` with every switch set runs in lockstep with the plain mirror
  (itself checked against ``SharedLRUReference``) on randomized traces for every
  policy, including long traces whose rows migrate between layers until the
  owner summary is fragmented.
* Thread-level statements of the CUDA index arithmetic (chunked compaction,
  bit-order append, warp walk over summary blocks) are checked against the
  sequential statements on adversarial inputs. They transcribe
  ``shared_lru_kernel.cuh`` line by line; the A100 gate still decides.
"""
from itertools import accumulate
import random
import unittest
from unittest.mock import patch

from runtime import shared_lru as s
from runtime import study_config as sc
from runtime import test_shared_lru as base
from runtime.cuda_shared_lru_correctness import FEATURE_SETS

KERNEL_MIRROR = s.KernelMirror


class Lockstep:
    """Drop-in for ``KernelMirror``: every switch set resolves the same call and must agree."""

    def __init__(self, plan, *, plane_rows):
        self.mirrors = {name: KERNEL_MIRROR(plan, plane_rows=plane_rows, features=features)
                        for name, features in FEATURE_SETS.items()}
        self.plain, self.plan = self.mirrors["none"], plan
        self.calls = self.skipped_blocks = 0

    @property
    def status(self):
        return self.plain.status

    def address(self, flat):
        return self.plain.address(flat)

    def init_slot(self, slot, generation, locs):
        for mirror in self.mirrors.values():
            mirror.init_slot(slot, generation, locs)

    def reset_slot(self, slot):
        for mirror in self.mirrors.values():
            mirror.reset_slot(slot)

    def resolve_row(self, slot, plane, blocks, block_size, seq_len):
        if seq_len > self.plan.hot_rows:  # blocks the summary walk will skip on this long-path call
            offset, size = self.plan.domain(plane)
            summary = self.mirrors["owner_summary"].slots[slot]["summary"]
            self.skipped_blocks += sum(not summary[b] >> plane & 1
                                       for b in range(offset // s.SUMMARY_ROWS, (offset + size - 1) // s.SUMMARY_ROWS + 1))
        results = {name: m.resolve_row(slot, plane, blocks, block_size, seq_len) for name, m in self.mirrors.items()}
        expected, state = results["none"], self.plain.slots[slot]
        self.calls += 1
        for name, mirror in self.mirrors.items():
            assert results[name] == expected, f"{name}: outputs, copies or counters diverged"
            for table in ("owner", "order", "occ", "token"):
                assert mirror.slots[slot][table] == state[table], f"{name}: {table} diverged"
            assert mirror.status == self.plain.status, f"{name}: status diverged"
            if "summary" in mirror.slots[slot]:
                assert mirror.slots[slot]["summary"] == mirror.live_summary(slot), f"{name}: stale summary"
        return expected


def lockstep_trace(cfg, seed, **kwargs):
    with patch.object(s, "KernelMirror", Lockstep):
        digest, rank = base.random_trace(cfg, seed, **kwargs)
    return digest, rank


def policy_configs():
    config = base.config
    return [base.ISOLATED, base.SHARED,
            config(mode="isolated", layout="arena", eviction="fifo"),
            config(mode="shared", layout="arena", eviction="fifo"),
            config(mode="isolated", layout="arena", quota="static", quota_rows={"0": 6, "1": 12, "2": 6}),
            config(mode="shared", layout="arena", quota="soft_floor", floor_rows={"0": 8, "1": 8, "2": 7}),
            config(mode="shared", layout="arena", admission="bypass_list", bypass_layers=[1]),
            config(mode="shared", layout="arena", eviction="fifo", admission="bypass_list", bypass_layers=[0]),
            config(mode="shared", layout="arena", quota="soft_floor", floor_rows={"0": 8, "1": 8},
                   admission="bypass_list", bypass_layers=[2])]


class SwitchedMirrorsMatchThePlainStatement(unittest.TestCase):
    def test_every_policy_and_switch_set_on_random_traces(self):
        for index, cfg in enumerate(policy_configs()):
            with self.subTest(pool=cfg.hot_pool):
                digest, rank = lockstep_trace(cfg, 100 + index)     # Rank.step also checks the reference
                self.assertEqual(digest, base.random_trace(cfg, 100 + index)[0])
                self.assertEqual(rank.mirror.status, 0)
                self.assertGreater(rank.mirror.calls, 100)

    def test_long_shared_traces_fragment_ownership_and_still_agree(self):
        """Rows migrate between layers for hundreds of calls; summaries must track every move."""
        fifo = base.config(mode="shared", layout="arena", eviction="fifo")
        for cfg, seed in ((base.SHARED, 5), (base.SHARED, 6), (fifo, 7)):
            with self.subTest(pool=cfg.hot_pool, seed=seed):
                _, rank = lockstep_trace(cfg, seed, layers=tuple(range(8)), H=16, block_size=2, blocks=4,
                                         steps=40, requests=3)
                summary = rank.mirror.mirrors["owner_summary"]
                fragmented = mixed = 0
                for slot, state in summary.slots.items():
                    planes = [None if t < 0 or t >> 16 != state["gen"] else t & 0xFFFF for t in state["owner"]]
                    for row, plane in enumerate(planes):
                        fragmented += plane is not None and plane != row // 16   # row lent to another layer
                    mixed += sum(bin(word).count("1") > 2 for word in state["summary"])
                self.assertGreater(fragmented, 20, "trace never moved rows between layers")
                self.assertGreater(mixed, 0, "no block ever held rows of more than its two home planes")
                self.assertGreater(rank.mirror.skipped_blocks, 100, "the summary walk never skipped a block")

    def test_unknown_feature_is_rejected(self):
        plan = s.plan_arena(base.SHARED, (0, 1), 8, 4)
        with self.assertRaises(ValueError):
            s.KernelMirror(plan, plane_rows=64, features=("hash_lookup",))
        self.assertEqual(set(s.RESOLVER_FEATURES), set(sc.FEATURES))


# -- thread-level statements of the CUDA index arithmetic ------------------------------------

def simt_compact(region, touched, offset, block_size, unroll=8):
    """touched_bitmap compaction loop of phase 9 (chunk, fast paths, ballots, warp-0 scan)."""
    region, size, warps = list(region), len(region), block_size // 32
    chunk, counts = block_size * unroll, unroll * (block_size // 32)
    segment = -(-counts // 32)
    write = 0
    for start in range(0, size, chunk):
        rows = [[region[start + k * block_size + t] if start + k * block_size + t < size else -1
                 for k in range(unroll)] for t in range(block_size)]           # all reads before the barrier
        keep = [[r >= 0 and (r - offset) not in touched for r in rows[t]] for t in range(block_size)]
        removed = any(r >= 0 and not kept for t in range(block_size) for r, kept in zip(rows[t], keep[t]))
        if not removed:                                                         # __syncthreads_or == 0
            if write != start:
                for t in range(block_size):
                    for k in range(unroll):
                        if rows[t][k] >= 0:
                            region[write + k * block_size + t] = rows[t][k]
            write += min(chunk, size - start)
            continue
        rank, count = [[0] * unroll for _ in range(block_size)], [0] * (counts + 1)
        for k in range(unroll):
            for w in range(warps):
                ballot = [keep[w * 32 + lane][k] for lane in range(32)]
                for lane in range(32):
                    rank[w * 32 + lane][k] = sum(ballot[:lane])
                count[k * warps + w] = sum(ballot)
        segments = [(min(counts, lane * segment), min(counts, min(counts, lane * segment) + segment))
                    for lane in range(32)]
        sums = [sum(count[a:b]) for a, b in segments]
        inclusive = list(accumulate(sums))
        count[counts] = inclusive[31]
        for lane, (a, b) in enumerate(segments):
            running = inclusive[lane] - sums[lane]
            for i in range(a, b):
                count[i], running = running, running + count[i]
        for t in range(block_size):
            for k in range(unroll):
                if keep[t][k]:
                    target = write + count[k * warps + t // 32] + rank[t][k]
                    assert target <= start + k * block_size + t, "compaction wrote past the entry's own index"
                    region[target] = rows[t][k]
        write += count[counts]
    return region, write


def simt_append(touched, size, offset, write, block_size):
    """Bit-order append of phase 9: contiguous words per thread, exclusive scan, set bits ascending."""
    words = -(-size // 32)
    bits = [0] * words
    for index in touched:
        bits[index >> 5] |= 1 << (index & 31)
    per, out, position = -(-words // block_size), {}, write
    for t in range(block_size):
        begin = min(words, t * per)
        end = min(words, begin + per)
        mine = position
        for w in range(begin, end):
            word = bits[w]
            while word:
                low = (word & -word).bit_length() - 1
                assert mine not in out
                out[mine] = offset + w * 32 + low
                mine += 1
                word &= word - 1
        position += sum(bin(bits[w]).count("1") for w in range(begin, end))
    return out, position - write


def simt_summary_hits(summary, owner, token, uid, tag, plane, lo, size, warps, unroll=4):
    """owner_summary walk of phase 3: warp-strided summary words, ballot, kSummaryUnroll blocks per round."""
    hi, found = lo + size, {}
    last = (hi - 1) // 32
    for warp in range(warps):
        b0 = lo // 32 + warp * 32
        while b0 <= last:
            pending = sum(1 << lane for lane in range(32)
                          if b0 + lane <= last and summary[b0 + lane] >> plane & 1)
            while pending:
                rows = []
                for _ in range(unroll):
                    lanes = [-1] * 32
                    if pending:
                        block = b0 + (pending & -pending).bit_length() - 1
                        pending &= pending - 1
                        lanes = [r if lo <= (r := block * 32 + lane) < hi else -1 for lane in range(32)]
                    rows.append(lanes)
                for lanes in rows:
                    for r in lanes:
                        if r >= 0 and owner[r] == tag and token[r] in uid:
                            assert uid[token[r]] not in found, "two rows hold one key"
                            found[uid[token[r]]] = r
            b0 += warps * 32
    return found


class ThreadLevelIndexArithmetic(unittest.TestCase):
    def touched_sets(self, rng, size):
        yield set()
        yield set(range(size))
        yield {rng.randrange(size)}
        yield set(rng.sample(range(size), size // 3))
        run = rng.randrange(size)
        yield set(range(run, min(size, run + 600)))                      # one refreshed run, as in steady LRU
        yield set(range(0, min(size, 40))) | set(range(max(0, size - 40), size))

    def test_compaction_and_append_equal_the_sequential_update(self):
        rng = random.Random(20261126)
        for block_size in (256, 1024):
            chunk = block_size * 8
            for size in (1, 31, 32, 33, chunk - 1, chunk, chunk + 1, 3 * chunk + 17):
                offset = rng.choice((0, 4, 4096))
                region = [offset + i for i in range(size)]
                rng.shuffle(region)
                for touched in self.touched_sets(rng, size):
                    with self.subTest(block=block_size, size=size, touched=len(touched)):
                        compacted, write = simt_compact(region, touched, offset, block_size)
                        kept = [row for row in region if row - offset not in touched]
                        self.assertEqual(compacted[:write], kept)
                        appended, total = simt_append(touched, size, offset, write, block_size)
                        self.assertEqual(total, len(touched))
                        self.assertEqual([appended[write + i] for i in range(total)],
                                         sorted(offset + i for i in touched))
                        self.assertEqual(kept + sorted(offset + i for i in touched),
                                         compacted[:write] + [appended[write + i] for i in range(total)])

    def test_summary_walk_finds_exactly_the_owned_selected_rows(self):
        rng = random.Random(7)
        for warps in (8, 32):
            for planes, H, lo, size in ((60, 64, 0, 3840), (8, 16, 4, 12), (8, 16, 40, 70), (3, 8, 0, 20)):
                rows = planes * H
                gen, plane = 3, rng.randrange(planes)
                tag = gen << 16 | plane
                for layout in ("home", "fragmented"):
                    owners = [r // H for r in range(rows)]
                    if layout == "fragmented":
                        rng.shuffle(owners)
                    owner = [(gen << 16 | o) if rng.random() > 0.1 else rng.choice((-1, 2 << 16 | o))
                             for o in owners]                                     # free and stale rows mixed in
                    token = list(range(rows))                                     # one row per (plane, token)
                    selected = rng.sample(range(rows), min(rows, 3 * H))
                    uid = {t: u for u, t in enumerate(selected)}
                    live = [None if t < 0 or t >> 16 != gen else t & 0xFFFF for t in owner]
                    exact = s.owner_summary(live)
                    loose = [w | (rng.getrandbits(planes) if rng.random() < 0.3 else 0) for w in exact]
                    expected = {uid[token[r]]: r for r in range(lo, lo + size)
                                if owner[r] == tag and token[r] in uid}
                    for summary in (exact, loose):
                        with self.subTest(warps=warps, rows=rows, layout=layout, exact=summary is exact):
                            self.assertEqual(simt_summary_hits(summary, owner, token, uid, tag, plane, lo, size,
                                                               warps), expected)
                    missing = list(exact)
                    if expected:                                                # a cleared bit loses the hit
                        block = next(iter(expected.values())) // 32
                        missing[block] &= ~(1 << plane)
                        self.assertNotEqual(simt_summary_hits(missing, owner, token, uid, tag, plane, lo, size,
                                                              warps), expected)


if __name__ == "__main__":
    unittest.main()
