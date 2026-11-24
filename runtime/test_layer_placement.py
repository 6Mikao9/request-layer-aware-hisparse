"""CPU fixtures for the combined layout: frozen offload set S plus the per-request hot pool.

Rows follow docs/implementation/combined.md, "无 GPU 交付测试". Every resolved row is
checked by ``CombinedReader`` against synthesized bytes of (generation, rank, layer, token).
"""
import random
import unittest

from runtime import layer_placement as lp
from runtime import shared_lru as s
from runtime import shared_lru_plan as plan
from runtime import study_config as sc

ITEM = 8
SHARED = dict(mode="shared", layout="arena")
ISOLATED = dict(mode="isolated", layout="arena")


def config(offload=None, **pool):
    data = {"schema": sc.SCHEMA}
    if offload is not None:
        data["placement"] = {"mode": "fixed_layers", "offload_layers": list(offload)}
    if pool:
        data["hot_pool"] = pool
    return sc.parse(data)


class Model:
    """One TP rank of the combined layout with the second reader on every resolved row."""

    def __init__(self, cfg, layers, H, K, *, page=1, rank=0, logical_locs=None, device_locs=None):
        self.ref = lp.CombinedReference(cfg, layers, hot_rows=H, selected=K, page=page, item_bytes=ITEM, rank=rank,
                                        logical_locs=logical_locs, device_locs=device_locs)
        self.reader = lp.CombinedReader(self.ref)
        self.epoch, self.seq = 0, {}

    def admit(self, slot, gen, prompt):
        """Staged prompt plus the first decode token, as the coordinator sees it at decode."""
        self.epoch += 1
        out = self.ref.admit(slot, gen, seq_len=prompt, epoch=self.epoch)
        self.seq[slot] = (gen, prompt)
        self.advance(slot)
        return out

    def advance(self, slot):
        gen, seq_len = self.seq[slot]
        lease = self.ref.backup(slot, gen)
        if lease is not None:
            self.ref.complete(lease, "backup")
        self.ref.advance(slot, gen)
        self.seq[slot] = (gen, seq_len + 1)

    def step(self, layer, picks, *, block_size=1, defer=False, complete=True):
        """picks: {slot: blocks}. Returns {slot: Resolved}; every real row is read back by the second reader."""
        self.epoch = (self.epoch + 1) & s.EPOCH_MASK
        rows = tuple(s.Selection(slot, self.seq[slot][0], tuple(blocks), self.seq[slot][1])
                     for slot, blocks in picks.items())
        out = self.ref.resolve(s.Batch(layer, block_size, rows), self.epoch, defer_gather=defer)
        for sel, res in zip(rows, out):
            if not defer:
                self.reader.check(res, slot=sel.slot, generation=sel.generation, layer=layer,
                                  tokens=s.selected_tokens(sel.blocks, block_size, sel.seq_len), seq_len=sel.seq_len)
            if complete and res.lease is not None:
                self.ref.complete(res.lease, "attention")
        return dict(zip(picks, out))

    def snapshot(self, slot):
        """Everything another request, rank or resident layer could disturb."""
        gen = self.seq[slot][0]
        hot = {}
        if self.ref.hot is not None:
            hot = {key: value for key, value in self.ref.hot.device.items()
                   if key[1] in set(self.ref.hot.locs(slot, gen))}
        resident = {key: value for key, value in self.ref.resident.items()
                    if key[1] in set(self.ref.req_to_token(slot, gen))}
        occupancy = self.ref.occupancy(slot, gen) if self.ref.plan is not None else None
        return dict(hot=hot, resident=resident, req_to_token=self.ref.req_to_token(slot, gen), occupancy=occupancy)


def trace(seed, layers, steps, prompt=24, width=2, block_size=2):
    """Seeded selections with duplicates, invalid entries, out-of-range blocks and the newest block."""
    rng, out = random.Random(seed), []
    for step in range(steps):
        seq_len = prompt + 1 + step
        top = -(-seq_len // block_size)
        for layer in layers:
            blocks = [rng.randrange(top + 1) for _ in range(width)]     # top itself is out of range
            if rng.random() < 0.3:
                blocks[-1] = -1
            if rng.random() < 0.3:
                blocks[0] = blocks[-1]
            if rng.random() < 0.4:
                blocks[0] = top - 1
            out.append((step, layer, tuple(blocks)))
    return out


def replay(model, items, *, slot=0, block_size=2):
    """Run a trace; returns per invocation (classes, valid tokens read) after every reader check."""
    out, current = [], None
    for step, layer, blocks in items:
        if current is not None and step != current:
            model.advance(slot)
        current = step
        res = model.step(layer, {slot: blocks}, block_size=block_size)[slot]
        tokens = s.selected_tokens(blocks, block_size, model.seq[slot][1])
        out.append((res.classes, [t if a is not None else -1 for t, a in zip(tokens, res.addresses)], res.misses))
    return out


class PhysicalLendingAmongOffloadedLayers(unittest.TestCase):
    """Two offloaded layers, one resident layer, two requests, two ranks, distinct payloads."""

    def test_offloaded_layer_reuses_returned_slot_of_another_and_nothing_else_moves(self):
        layers, S = (0, 1, 2), (0, 1)
        ranks = [Model(config(S, **SHARED), layers, 4, 4, page=4, rank=r) for r in (0, 1)]
        for model in ranks:
            model.admit(0, 1, 16)      # request A
            model.admit(1, 2, 16)      # request B
        rank0, rank1 = ranks
        before = dict(other_rank=rank1.snapshot(0), request_b=rank0.snapshot(1),
                      resident_a=rank0.snapshot(0)["resident"], a_map=rank0.ref.req_to_token(0, 1))
        rank0.step(0, {0: [0, 1]}, block_size=2)                  # layer 0 refreshes its own prefill rows
        lent = rank0.step(0, {0: [4, 5]}, block_size=2)[0]        # four misses take layer 1's idle rows
        self.assertEqual((lent.misses, lent.hits), (4, 0))
        self.assertEqual({a[:2] for a in lent.addresses}, {(lp.HOT, 1)}, "K/V now live in layer 1's plane")
        self.assertEqual(rank0.ref.occupancy(0, 1), {0: 8, 1: 0})
        token8 = rank0.ref.read(lent.addresses[0])
        self.assertEqual(token8, rank0.ref.host_payload(0, 1, 0, 8))
        self.assertNotEqual(token8, rank1.ref.host_payload(0, 1, 0, 8), "rank-local payloads differ")
        self.assertEqual(rank1.snapshot(0), before["other_rank"])
        self.assertEqual(rank0.snapshot(1), before["request_b"])
        self.assertEqual(rank0.snapshot(0)["resident"], before["resident_a"])
        self.assertEqual(rank0.ref.req_to_token(0, 1), before["a_map"])
        resident = rank0.step(2, {0: [4, 5], 1: [4, 5]}, block_size=2)
        self.assertEqual({a[:2] for res in resident.values() for a in res.addresses}, {(lp.RESIDENT, 0)})
        back = rank0.step(1, {0: [0, 1]}, block_size=2)[0]         # layer 1 borrows layer 0's older rows
        self.assertEqual((back.misses, {a[:2] for a in back.addresses}), (4, {(lp.HOT, 0)}))
        self.assertEqual(rank0.ref.occupancy(0, 1), {0: 4, 1: 4})


class EqualBytesHPlusHVersusTwoH(unittest.TestCase):
    """H=2 per offloaded layer; the shared arm pools the same 2H history rows of both layers."""
    ARMS = (("shared", SHARED), ("isolated", ISOLATED), ("stock_layout", {}))

    def models(self, prompt):
        models = {name: Model(config((0, 1), **pool), (0, 1, 2), 2, 1) for name, pool in self.ARMS}
        for model in models.values():
            model.admit(0, 1, prompt)
        return models

    def test_two_h_serves_a_three_cycle_and_a_reread_after_warmup(self):
        models, hits = self.models(10), {name: [] for name, _ in self.ARMS}
        for step in range(9):
            a = 3 + step % 3
            for name, model in models.items():
                hits[name].append(model.step(0, {0: [a]})[0].hits)
                model.step(1, {0: [6]})
                model.step(2, {0: [a]})                   # resident layer: direct read, no hot-pool effect
        self.assertEqual(hits["shared"][3:], [1] * 6)
        self.assertEqual(hits["isolated"], [0] * 9)
        self.assertEqual(hits["stock_layout"], [0] * 9)
        self.assertEqual(models["shared"].ref.occupancy(0, 1), {0: 3, 1: 1})
        ledgers = {name: lp.placement_ledger(placement=m.ref.placement, hot_rows=2, page=1, item_bytes=ITEM, slots=4,
                                             device_rows=64, logical_rows=128, plan=m.ref.plan)
                   for name, m in models.items()}
        budget = {(l["hot_main_kv_bytes"], l["resident_main_kv_bytes"], l["per_request"]["history_rows"])
                  for l in ledgers.values()}
        self.assertEqual(budget, {(2 * 64 * 2 * ITEM, 128 * 2 * ITEM, 4)}, "same history byte budget in every arm")

    def test_scanning_layer_pollutes_the_shared_pool_but_not_a_partition(self):
        last = {}
        for name, pool in self.ARMS[:2]:
            model = Model(config((0, 1), **pool), (0, 1, 2), 2, 1)
            model.admit(0, 1, 20)
            for step, a in enumerate((3, 4, 4, 4, 4, 3)):
                last[name] = model.step(0, {0: [a]})[0]
                model.step(1, {0: [5 + step]})              # layer 1 scans a new token every step
                model.step(2, {0: [5 + step]})
        self.assertEqual(last["isolated"].classes, [s.HIT])
        self.assertEqual(last["shared"].classes, [s.MISS])


class QuotasSingleLayerAndBoundaries(unittest.TestCase):
    def test_soft_floor_keeps_a_layer_under_the_same_total_residency(self):
        floor = Model(config((0, 1), **SHARED, quota="soft_floor", floor_rows={"0": 2}), (0, 1, 2), 2, 1)
        floor.admit(0, 1, 20)
        breaches = 0
        for step, a in enumerate((3, 4, 4, 4, 4, 3)):
            last = floor.step(0, {0: [a]})[0]
            breaches += floor.step(1, {0: [5 + step]})[0].floor_breaches
        self.assertEqual((last.classes, breaches), ([s.HIT], 0))
        self.assertEqual(floor.ref.plan.rows, Model(config((0, 1), **SHARED), (0, 1, 2), 2, 1).ref.plan.rows)

    def test_empty_s_is_the_all_resident_layout(self):
        items = trace(1, (0, 1, 2), 6)
        empty = Model(config(()), (0, 1, 2), 4, 4, page=2)
        full = sc.parse({"schema": sc.SCHEMA, "placement": {"mode": "full_resident"}})
        resident = Model(full, (0, 1, 2), 4, 4, page=2)
        for model in (empty, resident):
            model.admit(0, 1, 24)
        self.assertIsNone(empty.ref.hot)
        self.assertEqual(replay(empty, items), replay(resident, items))
        classes = {c for cls, _, _ in replay(empty, trace(2, (0, 1, 2), 2)) for c in cls}
        self.assertLessEqual(classes, {lp.RESIDENT, s.PAD})
        self.assertEqual((empty.ref.host_ledger()["bytes"], empty.ref.device_rows()["hot"]), (0, 0))
        with self.assertRaisesRegex(ValueError, "no host backing"):
            empty.ref.host_payload(0, 1, 1, 0)

    def test_single_offloaded_layer_makes_shared_equal_isolated(self):
        items = trace(3, (0, 1, 2), 12)
        outputs = []
        for pool in (SHARED, ISOLATED):
            model = Model(config((1,), **pool), (0, 1, 2), 4, 4, page=2)
            model.admit(0, 1, 24)
            outputs.append((replay(model, items), model.ref.occupancy(0, 1)))
        self.assertEqual(outputs[0], outputs[1])

    def test_all_eligible_layers_reproduce_the_all_layer_arena_and_stock(self):
        items = trace(4, (0, 1, 2), 12)
        for pool in (SHARED, ISOLATED, {}):
            with self.subTest(pool=pool):
                fixed, native = Model(config((0, 1, 2), **pool), (0, 1, 2), 4, 4, page=2), \
                    Model(config(None, **pool), (0, 1, 2), 4, 4, page=2)
                for model in (fixed, native):
                    model.admit(0, 1, 24)
                self.assertEqual(fixed.ref.placement.resident, ())
                self.assertEqual(replay(fixed, items), replay(native, items))
                self.assertEqual(fixed.ref.hot.device, native.ref.hot.device)
        arena = s.SharedLRUReference(s.plan_arena(config(**SHARED), (0, 1, 2), 4, 4), page=2, item_bytes=ITEM)
        combined = Model(config((0, 1, 2), **SHARED), (0, 1, 2), 4, 4, page=2)
        locs = arena.admit(0, 1, 0, (0, 1, 2), 4, 12, seq_len=24, epoch=1)
        self.assertEqual(combined.admit(0, 1, 24)["device"], locs, "same spans as the all-layer reference")


class InFlightCopiesReadersAndReuse(unittest.TestCase):
    def model(self):
        model = Model(config((0, 1), **SHARED), (0, 1, 2), 2, 1)
        model.admit(0, 1, 40)
        return model

    def test_pending_gather_is_neither_evicted_nor_visible_until_its_fence(self):
        model = self.model()
        pending = model.step(0, {0: [10]}, defer=True, complete=False)[0]
        row = pending.victims[0]
        for token in (11, 12, 13):                      # the other offloaded layer misses around it
            self.assertNotIn(row, model.step(1, {0: [token]})[0].victims)
        with self.assertRaisesRegex(s.Deferred, "loading"):
            model.step(0, {0: [10]})                    # no hit before both K and V arrived
        self.assertEqual(model.step(2, {0: [10]})[0].classes, [lp.RESIDENT])   # resident reads never wait
        model.ref.complete(pending.lease, "attention")  # completion order does not publish early
        with self.assertRaises(s.Deferred):
            model.step(0, {0: [10]})
        model.ref.complete(pending.lease, "gather")
        hit = model.step(0, {0: [10]})[0]
        self.assertEqual((hit.classes, model.ref.read(hit.addresses[0])),
                         ([s.HIT], model.ref.host_payload(0, 1, 0, 10)))

    def test_backup_and_prefetch_fences_hold_rows_across_both_pools(self):
        model = self.model()
        model.step(0, {0: [20]})
        lease = model.ref.pin(0, 1, 0, [20])
        for token in (21, 22, 23):
            model.step(1, {0: [token]})
        self.assertEqual(model.step(0, {0: [20]})[0].hits, 1, "the prefetch pin kept layer 0's row")
        model.ref.complete(lease, "prefetch")
        with self.assertRaisesRegex(ValueError, "resident"):
            model.ref.pin(0, 1, 2, [20])
        before = (model.ref.req_to_token(0, 1), model.ref.device_rows())
        backup = model.ref.backup(0, 1)
        with self.assertRaises(s.Deferred):
            model.ref.advance(0, 1)                     # tail not on the host yet: neither pool appends
        self.assertEqual((model.ref.req_to_token(0, 1), model.ref.device_rows()), before)
        model.ref.complete(backup, "backup")
        model.ref.advance(0, 1)
        self.assertEqual(len(model.ref.req_to_token(0, 1)), len(before[0]) + 1)

    def test_generation_reuse_never_reads_the_previous_owner(self):
        model = self.model()
        stale = model.step(0, {0: [10]}, complete=False)[0]
        first = model.ref.req_to_token(0, 1)
        with self.assertRaises(s.Deferred):              # completion fence before retirement
            model.ref.finish(0, 1)
        model.ref.complete(stale.lease, "attention")
        self.assertTrue(model.ref.finish(0, 1))
        self.assertFalse(model.ref.finish(0, 1))
        model.seq.pop(0)
        model.admit(0, 2, 40)
        self.assertEqual(model.ref.req_to_token(0, 2)[:40], first[:40], "second owner reuses the logical pages")
        with self.assertRaises(s.StaleCompletion):
            model.ref.complete(stale.lease, "attention")
        for layer in (0, 1, 2):
            res = model.step(layer, {0: [10]})[0]        # reader checks generation-2 bytes
            self.assertNotEqual(model.ref.read(res.addresses[0])[0], s.payload(1, 0, layer, 10, "k", ITEM))


class RoutingShortToLongAndTail(unittest.TestCase):
    def test_new_and_stock_paths_read_the_same_tokens_and_padding(self):
        items = trace(5, (0, 1, 2, 3), 10)
        arms = [Model(config(None), (0, 1, 2, 3), 4, 4, page=2),
                Model(config((1, 3)), (0, 1, 2, 3), 4, 4, page=2),
                Model(config((1, 3), **SHARED), (0, 1, 2, 3), 4, 4, page=2)]
        results, raw = [], []
        for model in arms:
            model.admit(0, 1, 24)
            raw.append(replay(model, items))
            results.append([(read, [c == s.PAD for c in cls]) for cls, read, _ in raw[-1]])
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])
        for (_, layer, _), (classes, _, misses) in zip(items, raw[1]):
            if layer in (0, 2):
                self.assertTrue(set(classes) <= {lp.RESIDENT, s.PAD} and misses == 0, "resident layers read directly")
            else:
                self.assertNotIn(lp.RESIDENT, classes)
        self.assertGreater(sum(m for (_, layer, _), (_, _, m) in zip(items, raw[1]) if layer in (1, 3)), 0)

    def test_only_offloaded_layers_hold_host_backing(self):
        model = Model(config((1, 3), **SHARED), (0, 1, 2, 3), 4, 4, page=2)
        model.admit(0, 1, 24)
        ledger = model.ref.host_ledger()
        self.assertEqual((ledger["layers"], ledger["rows"]), ([1, 3], 25 * 2))
        self.assertEqual(model.ref.host_payload(0, 1, 3, 24), (s.payload(1, 0, 3, 24, "k", ITEM),
                                                                s.payload(1, 0, 3, 24, "v", ITEM)))
        for layer in (0, 2):
            with self.assertRaisesRegex(ValueError, "no host backing"):
                model.ref.host_payload(0, 1, layer, 3)

    def test_short_capacity_follows_real_state_and_newest_rows(self):
        model = Model(config((1,), **SHARED), (0, 1, 2), 16, 4, page=4)
        model.admit(0, 1, 5)                            # seq_len 6 after the first decode token
        self.assertEqual(model.ref.device_rows(), dict(hot=8, resident=8 * 2))
        short = model.step(1, {0: [0, 5]}, block_size=2)[0]
        self.assertEqual(short.classes, [s.DIRECT, s.DIRECT, s.PAD, s.PAD])
        newest = model.step(2, {0: [2]}, block_size=2)[0]
        self.assertEqual(newest.classes, [lp.RESIDENT, lp.RESIDENT])
        for _ in range(11):
            model.advance(0)                            # seq_len 17 > H: long path, tail slot in use
        self.assertEqual(model.ref.device_rows(), dict(hot=16 + 4, resident=20 * 2))
        long = model.step(1, {0: [1, 8]}, block_size=2)[0]
        self.assertEqual(long.classes, [s.HIT, s.HIT, s.LATEST, s.PAD])
        self.assertEqual(model.step(2, {0: [8]}, block_size=2)[0].classes, [lp.RESIDENT, s.PAD])

    def test_padded_graph_rows_change_nothing(self):
        model = Model(config((0,), **SHARED), (0, 1), 2, 2)
        model.admit(0, 1, 12)
        before = model.snapshot(0)
        for layer in (0, 1):
            pad = s.Selection(0, 1, (3, 4), model.seq[0][1], False)
            res = model.ref.resolve(s.Batch(layer, 1, (pad,)), 77)[0]
            self.assertEqual((res.classes, res.addresses, res.unique, res.lease), ([s.PAD] * 2, [None] * 2, 0, None))
        self.assertEqual(model.snapshot(0), before)


class UserSwitchesAndRejections(unittest.TestCase):
    LAYERS = (0, 1, 2, 3)

    def test_no_config_and_all_off_equal_stock(self):
        items = trace(6, self.LAYERS, 8)
        stock = s.NativeLayerLRU(self.LAYERS, 4, page=2, item_bytes=ITEM)
        stock.admit(0, 1, seq_len=24)
        stock.advance(0, 1)
        expected, seq = [], 25
        for step, layer, blocks in items:
            if step and layer == self.LAYERS[0]:
                stock.backup(0, 1)
                stock.advance(0, 1)
                seq += 1
            res = stock.resolve(s.Batch(layer, 2, (s.Selection(0, 1, blocks, seq),)))[0]
            expected.append([None if a is None else (lp.HOT, *a) for a in res.addresses])
        for cfg in (sc.StudyConfig(), config(None, mode="isolated", layout="native")):
            model = Model(cfg, self.LAYERS, 4, 4, page=2)
            model.admit(0, 1, 24)
            got = []
            for step, layer, blocks in items:
                if step and layer == self.LAYERS[0]:
                    model.advance(0)
                got.append(model.step(layer, {0: blocks}, block_size=2)[0].addresses)
            self.assertTrue(cfg.is_native)
            self.assertEqual(got, expected)

    def test_each_switch_alone_and_both_change_only_their_fields(self):
        stock = sc.StudyConfig()
        placement, sharing, both = config((1, 3)), config(None, **SHARED), config((1, 3), **SHARED)
        self.assertEqual(plan.changed_fields(stock, placement), ("placement.mode", "placement.offload_layers"))
        self.assertEqual(plan.changed_fields(stock, sharing), ("hot_pool.layout", "hot_pool.mode"))
        self.assertEqual(plan.changed_fields(placement, both), ("hot_pool.layout", "hot_pool.mode"))
        self.assertEqual(plan.changed_fields(sharing, both), ("placement.mode", "placement.offload_layers"))
        for cfg in (placement, sharing, both):
            model = Model(cfg, self.LAYERS, 4, 4, page=2)
            model.admit(0, 1, 24)
            self.assertTrue(replay(model, trace(7, self.LAYERS, 4)))

    def test_illegal_layers_layouts_and_selection_width_fail_loudly(self):
        cases = [
            (config((1, 9)), 4, 4, "not eligible"),                                  # layer outside the model
            (config((1, 3), **SHARED), 4, 8, "selected width"),                     # K above H
            (config((1, 3), **ISOLATED, quota="static", quota_rows={"1": 2, "3": 6}), 4, 4, "complete selection"),
            (config((1, 3), **ISOLATED, quota="static", quota_rows={"1": 4, "2": 4}), 4, 4, "non-eligible"),
            (config((1, 3), **SHARED, admission="bypass_list", bypass_layers=[2]), 4, 4, "non-eligible"),
            (config((1, 3), **SHARED, quota="soft_floor", floor_rows={"0": 1}), 4, 4, "non-eligible"),
        ]
        for cfg, H, K, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                Model(cfg, self.LAYERS, H, K)
        for raw in ({"mode": "fixed_layers"}, {"mode": "fixed_layers", "offload_layers": [1, 1]},
                    {"mode": "fixed_layers", "offload_layers": [1], "typo": 1}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                sc.parse({"schema": sc.SCHEMA, "placement": raw})
        with self.assertRaisesRegex(ValueError, "requires hot_pool.layout=arena"):
            config((1,), mode="shared")


class ByteBudget(unittest.TestCase):
    """MiniMax TP8 example units; worker receipts replace every estimate with allocations."""

    def test_resident_layers_cost_logical_rows_and_shrink_the_hot_budget(self):
        row, ratio, index = 512, 2, 256 * 60
        native = lp.cell_bytes(sparse_layers=60, offload_layers=60, dense_layers=0, row_bytes=row, ratio=ratio,
                               index_bytes=index)
        fixed = lp.cell_bytes(sparse_layers=60, offload_layers=40, dense_layers=0, row_bytes=row, ratio=ratio,
                              index_bytes=index)
        self.assertEqual(fixed - native, 20 * (ratio - 1) * row)
        budget = 396_544 * native                       # stock auto-sized device tokens at TP8
        self.assertLess(budget // fixed, 396_544)
        with self.assertRaises(ValueError):
            lp.cell_bytes(sparse_layers=60, offload_layers=61, dense_layers=0, row_bytes=row, ratio=2, index_bytes=0)
        placement = lp.plan_placement(config(tuple(range(20, 60))), tuple(range(60)))
        size = budget // fixed // 128 * 128
        device_rows, logical_rows = size + 128, size * ratio + 128
        for pool, plan_ in (({}, None), (SHARED, s.plan_arena(config(range(20, 60), **SHARED), placement.offload,
                                                              4096, 2048))):
            ledger = lp.placement_ledger(placement=placement, hot_rows=4096, page=128, item_bytes=256, slots=128,
                                         device_rows=device_rows, logical_rows=logical_rows, plan=plan_)
            self.assertEqual(ledger["resident_main_kv_bytes"], 20 * logical_rows * 512)
            self.assertEqual(ledger["hot_main_kv_bytes"], 40 * device_rows * 512)
            self.assertEqual(ledger["host_pinned_bytes"], 40 * (logical_rows - 128) * 512)
            self.assertEqual(ledger["per_request"]["history_payload_bytes"], 40 * 4096 * 512)   # 80 MiB vs 120 MiB
            self.assertEqual(ledger["arena_metadata_bytes"] > 0, plan_ is not None)
        empty = lp.placement_ledger(placement=lp.plan_placement(config(()), (0, 1)), hot_rows=4, page=2,
                                    item_bytes=ITEM, slots=2, device_rows=10, logical_rows=18)
        self.assertEqual((empty["hot_main_kv_bytes"], empty["host_pinned_bytes"]), (0, 0))

    def test_high_positions_and_logical_exhaustion_without_partial_state(self):
        model = Model(config((1,), **SHARED), (0, 1), 4, 4, page=64, logical_locs=64 * 627)
        model.admit(0, 1, 40_000)                       # 626 pages including the first decode token
        locs = model.ref.req_to_token(0, 1)
        self.assertGreater(max(locs), 32_767)
        self.assertEqual(model.step(0, {0: [9_000]}, block_size=4)[0].classes, [lp.RESIDENT] * 4)
        self.assertEqual(model.step(1, {0: [9_000]}, block_size=4)[0].misses, 4)
        with self.assertRaises(MemoryError):
            model.ref.admit(1, 2, seq_len=64)
        self.assertNotIn(1, model.ref.logical.req_to_token)
        self.assertEqual(model.ref.hot.allocator.allocated_rows, 2 * 64, "only request A's H + page span")


if __name__ == "__main__":
    unittest.main()