"""CPU fixtures for the shared cross-layer hot pool (issue table "Required CPU fixtures")."""
import json
import random
import unittest

from runtime import shared_lru as s
from runtime import study_config as sc

ITEM = 8


def config(**pool):
    return sc.parse({"schema": sc.SCHEMA, "hot_pool": pool})


SHARED = config(mode="shared", layout="arena")
ISOLATED = config(mode="isolated", layout="arena")


class Rank:
    """One TP rank: reference, optional kernel mirror and direct reader on the same bytes."""

    def __init__(self, cfg, layers, H, K, *, page=1, rank=0, total_locs=None, mirror=True):
        self.plan = s.plan_arena(cfg, layers, H, K)
        self.ref = s.SharedLRUReference(self.plan, page=page, item_bytes=ITEM, rank=rank, total_locs=total_locs)
        self.page, self.H, self.rank, self.epoch = page, H, rank, 0
        self.mirror = s.KernelMirror(self.plan, plane_rows=self.ref.allocator.total_locs) if mirror else None
        self.reader = s.DirectReader(self.ref.device, rank=rank, item_bytes=ITEM, hot_rows=H, planes=len(layers))
        self.seq = {}

    def admit(self, slot, gen, prompt):
        """Staged prompt plus the first decode token, as the coordinator sees it at decode."""
        locs = self.ref.admit(slot, gen, self.rank, self.plan.layers, min(prompt, self.H), self.plan.rows,
                              seq_len=prompt, epoch=self.epoch + 1)
        if self.mirror:
            self.mirror.init_slot(slot, gen, locs)
        self.seq[slot] = (gen, prompt)
        self.advance(slot)
        return locs

    def step(self, layer, picks, *, block_size=1, defer=False, complete=True, real=None):
        """picks: {slot: blocks}. Returns {slot: Resolved}; verifies bytes and the mirror."""
        self.epoch = (self.epoch + 1) & s.EPOCH_MASK
        rows = tuple(s.Selection(slot, self.seq[slot][0], tuple(blocks), self.seq[slot][1],
                                 True if real is None else real.get(slot, True))
                     for slot, blocks in picks.items())
        out = self.ref.resolve(s.Batch(layer, block_size, rows), self.epoch, defer_gather=defer)
        result = {}
        for sel, res in zip(rows, out):
            result[sel.slot] = res
            if not sel.real:
                continue
            locs = self.ref.locs(sel.slot, sel.generation)
            tokens = s.selected_tokens(sel.blocks, block_size, sel.seq_len)
            if self.mirror:
                flat, _, counters = self.mirror.resolve_row(sel.slot, self.plan.plane(layer), sel.blocks,
                                                            block_size, sel.seq_len)
                assert [self.mirror.address(x) for x in flat] == res.addresses, "kernel mirror diverged"
                assert counters["hits"] == res.hits and counters["misses"] == res.misses
                assert counters["breaches"] == res.floor_breaches
            if not defer:
                self.reader.check(res, generation=sel.generation, layer=layer, plane=self.plan.plane(layer),
                                  tokens=tokens, seq_len=sel.seq_len, locs=locs)
            if complete and res.lease is not None:
                self.ref.complete(res.lease, "attention")
        return result

    def advance(self, slot):
        gen, seq_len = self.seq[slot]
        lease = self.ref.backup(slot, gen)
        if lease is not None:
            self.ref.complete(lease, "backup")
        self.ref.advance(slot, gen)
        self.seq[slot] = (gen, seq_len + 1)


class Native:
    def __init__(self, layers, H, *, page=1):
        self.lru = s.NativeLayerLRU(layers, H, page=page, item_bytes=ITEM)
        self.reader = s.DirectReader(self.lru.device, rank=0, item_bytes=ITEM, hot_rows=H, planes=len(layers))
        self.seq = {}

    def admit(self, slot, gen, prompt):
        self.lru.admit(slot, gen, seq_len=prompt)
        self.lru.advance(slot, gen)
        self.seq[slot] = (gen, prompt + 1)

    def step(self, layer, picks, block_size=1):
        rows = tuple(s.Selection(slot, self.seq[slot][0], tuple(b), self.seq[slot][1]) for slot, b in picks.items())
        out = self.lru.resolve(s.Batch(layer, block_size, rows))
        for sel, res in zip(rows, out):
            self.reader.check(res, generation=sel.generation, layer=layer, plane=self.lru.layers.index(layer),
                              tokens=s.selected_tokens(sel.blocks, block_size, sel.seq_len),
                              seq_len=sel.seq_len, locs=self.lru.locs(sel.slot, sel.generation))
        return dict(zip(picks, out))
class BalancedLending(unittest.TestCase):
    def test_shared_rows_serve_cycle_that_native_uniform_misses(self):
        A, B, x = 0, 1, 6
        rank, native = Rank(SHARED, (A, B), 2, 1), Native((A, B), 2)
        rank.admit(0, 1, 10)
        native.admit(0, 1, 10)
        shared_hits, native_a_misses = [], []
        for step in range(9):
            a_token = 3 + step % 3
            res = rank.step(A, {0: [a_token]})[0]
            shared_hits.append(res.hits)
            self.assertEqual(rank.step(B, {0: [x]})[0].misses, 1 if step == 0 else 0)
            native_a_misses.append(native.step(A, {0: [a_token]})[0].misses)
            native.step(B, {0: [x]})
        self.assertEqual(shared_hits[3:], [1] * 6)       # all hits after initial population
        self.assertEqual(native_a_misses, [1] * 9)       # H=2 per layer cannot hold a 3-cycle
        occupancy = rank.ref.occupancy(0, 1)
        self.assertEqual((occupancy[A], occupancy[B]), (3, 1))
        self.assertEqual(sum(occupancy.values()), rank.plan.rows)
        planes = {addr[0] for t in (3, 4, 5) for addr in rank.step(A, {0: [t]})[0].addresses}
        self.assertEqual(planes, {0, 1}, "layer A must read K/V physically lent by layer B's plane")


class PollutionQuotasFifoBypass(unittest.TestCase):
    A_TRACE = (3, 4, 4, 4, 4, 3)

    def run_trace(self, cfg):
        rank = Rank(cfg, (0, 1), 2, 1)
        rank.admit(0, 1, 20)
        last = None
        for step, a in enumerate(self.A_TRACE):
            last = rank.step(0, {0: [a]})[0]
            rank.step(1, {0: [5 + step]})
        return rank, last

    def test_fixed_partition_keeps_a_global_lru_evicts_it(self):
        self.assertEqual(self.run_trace(ISOLATED)[1].hits, 1)
        static = config(mode="isolated", layout="arena", quota="static", quota_rows={"0": 2, "1": 2})
        self.assertEqual(self.run_trace(static)[1].hits, 1)
        self.assertEqual(self.run_trace(SHARED)[1].misses, 1)

    def test_soft_floor_prefers_victims_above_floor(self):
        floor = config(mode="shared", layout="arena", quota="soft_floor", floor_rows={"0": 2})
        rank, last = self.run_trace(floor)
        self.assertEqual(last.hits, 1)
        self.assertGreaterEqual(rank.ref.occupancy(0, 1)[0], 2)
        # With every candidate at its floor the oldest safe row is still taken and the breach published.
        tight = config(mode="shared", layout="arena", quota="soft_floor", floor_rows={"0": 2, "1": 2})
        rank = Rank(tight, (0, 1), 2, 1)
        rank.admit(0, 1, 20)
        res = rank.step(0, {0: [7]})[0]
        self.assertEqual((res.misses, res.floor_breaches), (1, 1))

    def test_bypass_reads_exact_bytes_without_admission(self):
        bypass = config(mode="shared", layout="arena", admission="bypass_list", bypass_layers=[1])
        rank, last = self.run_trace(bypass)
        self.assertEqual(rank.plan.scratch, (3, 1))        # K scratch rows charged to the arena
        self.assertEqual(last.hits, 1)
        occupancy = rank.ref.occupancy(0, 1)
        self.assertEqual(occupancy[1], 1, "bypassed layer kept only its initial unscratched row")
        res = rank.step(1, {0: [17]})[0]
        self.assertEqual((res.classes, res.bypassed, res.misses), ([s.BYPASS], 1, 1))
        self.assertEqual(rank.ref.occupancy(0, 1), occupancy)

    def test_fifo_and_lru_differ_on_hit_refresh(self):
        victims = {}
        for eviction in ("lru", "fifo"):
            rank = Rank(config(mode="isolated", layout="arena", eviction=eviction), (0,), 2, 1)
            rank.admit(0, 1, 20)
            for token in (7, 8, 7):
                rank.step(0, {0: [token]})
            victims[eviction] = rank.step(0, {0: [9]})[0].victims
            again = rank.step(0, {0: [7]})[0]
            self.assertEqual(again.hits, 1 if eviction == "lru" else 0)
        self.assertNotEqual(victims["lru"], victims["fifo"])
class IdentityOrderMasks(unittest.TestCase):
    def test_same_token_is_distinct_per_layer_request_and_rank(self):
        token = 5
        ranks = [Rank(SHARED, (0, 1), 4, 2, rank=r) for r in (0, 1)]
        seen = {}
        for rank in ranks:
            rank.admit(0, 11, 30)
            rank.admit(1, 12, 30)
            for layer in (0, 1):
                for slot in (0, 1):
                    address = rank.step(layer, {slot: [token]})[slot].addresses[0]
                    seen[(rank.rank, layer, slot)] = rank.ref.device[address]
        self.assertEqual(len(set(seen.values())), len(seen), "sentinel payloads collided")

    def test_duplicates_newest_partial_invalid_and_padding(self):
        rank = Rank(SHARED, (0, 1), 16, 16, page=4)
        rank.admit(0, 3, 29)          # newest token 29 lives in block 7 (tokens 28..31)
        res = rank.step(1, {0: [4, 4, 7, -1]}, block_size=4)[0]
        classes = res.classes
        self.assertEqual(classes[:8], [s.MISS] * 8)
        self.assertEqual(res.addresses[:4], res.addresses[4:8], "duplicate block must share rows")
        self.assertEqual((res.unique, res.misses, res.bytes_requested), (5, 5, 5 * 2 * ITEM))
        self.assertEqual(classes[8:12], [s.MISS, s.LATEST, s.PAD, s.PAD])   # 28, newest 29, 30, 31
        self.assertEqual(classes[12:], [s.PAD] * 4)
        locs = rank.ref.locs(0, 3)
        self.assertEqual(res.addresses[9], (1, locs[16]))
        before = (rank.ref.occupancy(0, 3), rank.ref.order(0, 3, 0), dict(rank.ref.device))
        padded = rank.step(1, {0: [6, 6, 6, 6]}, block_size=4, real={0: False})[0]
        self.assertEqual(padded.classes, [s.PAD] * 16)
        self.assertEqual((padded.unique, padded.bytes_requested), (0, 0))
        self.assertEqual(before, (rank.ref.occupancy(0, 3), rank.ref.order(0, 3, 0), dict(rank.ref.device)))

    def test_short_to_long_transition(self):
        rank = Rank(SHARED, (0, 1), 4, 4, page=2)
        rank.admit(0, 9, 2)            # prompt 0..1, first decode token 2
        res = rank.step(0, {0: [0, 1]}, block_size=2)[0]
        self.assertEqual(res.classes, [s.DIRECT] * 3 + [s.PAD])
        rank.advance(0)                # token 3 fills the last short position
        self.assertEqual(rank.step(1, {0: [0, 1]}, block_size=2)[0].classes, [s.DIRECT] * 4)
        rank.advance(0)                # token 4 goes to the tail: the request is long now
        res = rank.step(0, {0: [1, 2]}, block_size=2)[0]
        self.assertEqual(res.classes, [s.HIT, s.HIT, s.LATEST, s.PAD])
        rank.advance(0)
        res = rank.step(0, {0: [2]}, block_size=2)[0]
        self.assertEqual(res.classes, [s.MISS, s.LATEST])  # token 4 left the tail for host history
class OverlapAndFailures(unittest.TestCase):
    def rank(self, H=2, layers=(0, 1)):
        rank = Rank(SHARED, layers, H, 1, mirror=False)
        rank.admit(0, 1, 40)
        return rank

    def test_pending_completions_hold_rows(self):
        rank = self.rank()
        pending = rank.step(0, {0: [10]}, defer=True, complete=False)[0]
        row = pending.victims[0]
        self.assertEqual(rank.ref._requests[0].rows[row].state, s.LOADING)
        # Three other rows remain; the loading row is never chosen while its gather is pending.
        for token in (11, 12, 13):
            res = rank.step(1, {0: [token]}, complete=False)[0]
            self.assertNotIn(row, res.victims)
        with self.assertRaises(s.Deferred):
            rank.step(1, {0: [14]})
        rank.ref.complete(pending.lease, "gather")
        self.assertEqual(rank.ref._requests[0].rows[row].state, s.VALID)
        with self.assertRaises(s.Deferred):   # attention readers still pin every row
            rank.step(1, {0: [14]})
        for lease in list(rank.ref._leases):
            rank.ref.complete(lease, "attention")
        self.assertEqual(rank.step(1, {0: [14]})[0].victims, (row,))

    def test_prefetch_and_backup_fences(self):
        rank = self.rank()
        rank.step(0, {0: [20]})
        lease = rank.ref.pin(0, 1, 0, [20])
        for token in (21, 22, 23):
            self.assertNotEqual(rank.step(1, {0: [token]})[0].victims, ())
        self.assertEqual(rank.step(0, {0: [20]})[0].hits, 1)
        rank.ref.complete(lease, "prefetch")
        backup = rank.ref.backup(0, 1)             # first decode token 40 heads to the host
        with self.assertRaises(s.Deferred):
            rank.ref.advance(0, 1)                 # tail token 40 is not on the host yet
        rank.ref.complete(backup, "backup")
        rank.ref.advance(0, 1)

    def test_later_hit_in_same_selection_is_not_evicted(self):
        rank = Rank(SHARED, (0,), 2, 2, mirror=False)
        rank.admit(0, 1, 40)
        rank.step(0, {0: [10, 11]})
        rank.step(0, {0: [11, 12]})               # 12 lands in row 0, the oldest (stamp, row)
        res = rank.step(0, {0: [13, 12]})[0]       # 13 misses; 12 hits and must survive
        self.assertEqual(res.classes, [s.MISS, s.HIT])
        self.assertEqual(res.hits, 1)

    def test_all_pinned_defers_without_mutation(self):
        rank = Rank(SHARED, (0,), 2, 2, mirror=False)
        rank.admit(0, 1, 40)
        first = rank.step(0, {0: [10, 11]}, complete=False)[0]
        snapshot = (rank.ref.order(0, 1, 0), rank.ref.occupancy(0, 1))
        with self.assertRaises(s.Deferred):
            rank.step(0, {0: [12, 13]})
        self.assertEqual(snapshot, (rank.ref.order(0, 1, 0), rank.ref.occupancy(0, 1)))
        rank.ref.complete(first.lease, "attention")
        self.assertEqual(rank.step(0, {0: [12, 13]})[0].misses, 2)
    def test_cancel_failure_stale_empty_finish_and_reuse(self):
        rank = self.rank()
        rank.step(0, {0: [10]})
        cancelled = rank.step(0, {0: [11]}, defer=True, complete=False)[0]
        rank.ref.cancel(cancelled.lease)          # mid-copy: old mapping and bytes survive
        self.assertEqual(rank.step(0, {0: [10]})[0].hits, 1)
        failed = rank.step(1, {0: [12]}, defer=True, complete=False)[0]
        rank.ref.fail(failed.lease)
        with self.assertRaises(s.RequestFailed):
            rank.step(0, {0: [10]})
        self.assertEqual(rank.ref.resolve(s.Batch(0, 1, ()), 999), [])
        self.assertTrue(rank.ref.finish(0, 1))
        self.assertFalse(rank.ref.finish(0, 1))   # repeated finish never double-frees
        rank.admit(0, 2, 40)                       # slot reuse with a new generation
        with self.assertRaises(s.StaleCompletion):
            rank.ref.complete(failed.lease, "attention")
        stale = rank.step(0, {0: [10]}, complete=False)[0]
        rank.ref.complete(stale.lease, "attention")
        with self.assertRaises(s.StaleCompletion):
            rank.ref.complete(stale.lease, "attention")
        self.assertEqual(rank.step(0, {0: [10]})[0].hits, 1)
        held = rank.step(0, {0: [30]}, complete=False)[0]
        with self.assertRaises(s.Deferred):        # completion fence before retirement
            rank.ref.finish(0, 2)
        rank.ref.complete(held.lease, "attention")
        self.assertTrue(rank.ref.finish(0, 2))

    def test_fully_reusable_pool_distinguishes_span_from_occupancy(self):
        H, page = 4, 2
        rank = Rank(SHARED, (0, 1), H, 2, page=page, total_locs=(H + page) * 3 + page, mirror=False)
        free_pages = rank.ref.allocator.free_pages
        for slot in range(3):
            rank.admit(slot, 1, 2)
        with self.assertRaises(MemoryError):
            rank.admit(3, 1, 2)
        allocated = rank.ref.allocator.allocated_rows
        occupied = sum(sum(rank.ref.occupancy(slot, 1).values()) for slot in range(3))
        self.assertEqual(allocated, 3 * (H + page) * 2)
        self.assertEqual(occupied, 3 * 2 * 3)      # prompt positions 0..1 plus decode token 2, per plane
        for slot in range(3):
            self.assertTrue(rank.ref.finish(slot, 1))
        self.assertEqual((rank.ref.allocator.free_pages, rank.ref.allocator.allocated_rows), (free_pages, 0))
        for slot in range(3):
            rank.admit(slot, 2, 2)
def random_trace(cfg, seed, *, layers=(0, 1, 2), H=8, block_size=2, blocks=3, steps=30, requests=3):
    """Mixed long/short requests, duplicates, invalid blocks, slot reuse; returns a digest."""
    rng = random.Random(seed)
    rank = Rank(cfg, layers, H, blocks * block_size, page=2)
    generation = {}
    for slot in range(requests):
        generation[slot] = slot + 1
        rank.admit(slot, generation[slot], rng.choice((2, 4, H + 3, 40)))
    log = []
    for step in range(steps):
        for layer in layers:
            picks = {}
            for slot in range(requests):
                seq_len = rank.seq[slot][1]
                top = (seq_len + block_size - 1) // block_size
                picks[slot] = [rng.choice((-1, rng.randrange(top), rng.randrange(top))) for _ in range(blocks)]
            for slot, res in rank.step(layer, picks, block_size=block_size).items():
                log.append((step, layer, slot, tuple(res.addresses), tuple(res.classes), res.floor_breaches))
        for slot in range(requests):
            rank.advance(slot)
        if step % 11 == 10:                        # retire one request and reuse its slot
            slot = step % requests
            rank.ref.finish(slot, generation[slot])
            rank.mirror.reset_slot(slot)
            generation[slot] += requests
            rank.admit(slot, generation[slot], rng.choice((1, H, 30)))
    device = sorted(rank.ref.device.items())
    return json.dumps([log, [list(k) + [v[0].hex(), v[1].hex()] for k, v in device]]), rank


class PhysicalScaleAndStockPath(unittest.TestCase):
    def test_kernel_mirror_matches_reference_for_every_policy(self):
        configs = [ISOLATED, SHARED,
                   config(mode="isolated", layout="arena", eviction="fifo"),
                   config(mode="shared", layout="arena", eviction="fifo"),
                   config(mode="isolated", layout="arena", quota="static", quota_rows={"0": 6, "1": 12, "2": 6}),
                   config(mode="shared", layout="arena", quota="soft_floor", floor_rows={"0": 8, "1": 8, "2": 7}),
                   config(mode="shared", layout="arena", admission="bypass_list", bypass_layers=[1]),
                   config(mode="shared", layout="arena", eviction="fifo", admission="bypass_list", bypass_layers=[0]),
                   config(mode="shared", layout="arena", quota="soft_floor", floor_rows={"0": 8, "1": 8},
                          admission="bypass_list", bypass_layers=[2])]
        for index, cfg in enumerate(configs):
            with self.subTest(config=cfg.hot_pool):
                digest, rank = random_trace(cfg, 100 + index)   # Rank.step asserts mirror == reference
                self.assertEqual(rank.mirror.status, 0)
                log = json.loads(digest)[0]
                seen = {cls for row in log for cls in row[4]}
                self.assertTrue({s.DIRECT, s.HIT, s.MISS, s.LATEST, s.PAD} <= seen, seen)
                if cfg.hot_pool.bypass_layers:
                    self.assertIn(s.BYPASS, seen)
                if cfg.hot_pool.quota == "soft_floor":
                    self.assertGreater(sum(row[5] for row in log), 0, "trace never reached a floor breach")

    def test_repeat_runs_are_byte_identical(self):
        first, _ = random_trace(SHARED, 7)
        second, _ = random_trace(SHARED, 7)
        self.assertEqual(first, second)
        self.assertNotEqual(first, random_trace(SHARED, 8)[0])

    def test_single_layer_shared_equals_isolated(self):
        one = dict(layers=(4,), H=8, steps=20)
        self.assertEqual(random_trace(SHARED, 3, **one)[0], random_trace(ISOLATED, 3, **one)[0])

    def test_wide_arena_ids_high_positions_and_rank_local_state(self):
        layers, H = tuple(range(9)), 4096           # 36,864 arena rows: beyond int16 row IDs
        ranks = [Rank(ISOLATED, layers, H, 8, page=4, rank=r, total_locs=8208, mirror=False) for r in (0, 1)]
        for rank in ranks:                          # direct-style admission: no rows on device yet
            for slot, gen, seq_len in ((0, 5, 2_000_000), (1, 6, 3_000_000)):
                rank.ref.admit(slot, gen, rank.rank, layers, 0, rank.plan.rows, seq_len=seq_len, epoch=1)
                rank.ref.advance(slot, gen)
                rank.seq[slot] = (gen, seq_len + 1)
        far = {0: [499_990, 499_991], 1: [749_000, 6]}
        near = {0: [499_980, 499_981], 1: [1, 2]}
        for rank, picks in zip(ranks, (far, near)):     # different rank-local selections
            res = rank.step(8, picks, block_size=4)
            self.assertGreater(min(res[0].victims), 32767)
            self.assertEqual(rank.step(8, picks, block_size=4)[1].hits, 8)
        keys = [set(rank.ref._requests[1].index) for rank in ranks]
        self.assertTrue(keys[0] and keys[1] and not keys[0] & keys[1])

    def test_heterogeneous_static_domains_and_default_rejection(self):
        static = config(mode="isolated", layout="arena", quota="static", quota_rows={"0": 3, "1": 5})
        plan = s.plan_arena(static, (0, 1), 4, 2)
        self.assertEqual(plan.domains, ((0, 3), (3, 5)))
        self.assertEqual(plan.owner_template(), [0, 0, 0, None, 1, 1, 1, 1])
        for bad in ({"0": 3, "1": 4}, {"0": 1, "1": 7}, {"0": 8}):
            with self.assertRaises(ValueError):
                s.plan_arena(config(mode="isolated", layout="arena", quota="static", quota_rows=bad), (0, 1), 4, 2)
        with self.assertRaises(ValueError):
            s.plan_arena(sc.StudyConfig(), (0, 1), 4, 2)    # stock layout has no arena state
        with self.assertRaises(ValueError):
            s.plan_arena(config(mode="shared", layout="arena", admission="bypass_list", bypass_layers=[9]),
                         (0, 1), 4, 2)

    def test_epochs_are_wrap_safe(self):
        rank = Rank(SHARED, (0, 1), 2, 1)
        rank.epoch = s.EPOCH_MASK - 2
        rank.admit(0, 1, 10)
        for step in range(8):                       # crosses 2**32 mid-trace
            rank.step(step % 2, {0: [3 + step % 3]})
        self.assertLess(rank.epoch, 8)
        with self.assertRaises(ValueError):
            rank.ref.resolve(s.Batch(0, 1, (s.Selection(0, 1, (3,), rank.seq[0][1]),)), rank.epoch)


if __name__ == "__main__":
    unittest.main()
