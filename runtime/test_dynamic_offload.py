"""CPU tests: slab layout, a decode-semantics oracle across switches, and the TP switch protocol."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest

from . import dynamic_offload as d
from .selection_oracle import reference_lru

PAGE, H = 4, 8
GEOMETRY = d.Geometry(layers=(2, 4, 6, 8, 10), hot_rows=64, resident_rows=128, hot_slots=H, slots=5, page=PAGE,
                      page_bytes=PAGE * 256)
LAYOUT = d.SlabLayout(GEOMETRY, 3)
LISTS = {"A": (4, 6, 10), "B": (2, 6, 8), "C": (2, 4, 8)}


class Rank:
    """One rank's switchable layers as a CPU model. Rows hold token identities (uid, layer, token) instead of
    K/V bytes, so a read that returns the selected token's identity is what a bit-equal read needs.

    ``indirect`` is the slab arm: tail store, eager flush and the route-aware swap. Otherwise every layer
    follows the stock path of a frozen S: the newest token goes to the hot buffer's reserved slot (offloaded)
    or to its logical row (resident), and the native backup copies it to host one step later.
    """

    def __init__(self, offload, *, indirect=True, host_shadow=False):
        self.indirect, self.host_shadow = indirect, host_shadow
        self.routes = d.initial_routes(LAYOUT, offload)
        self.slab, self.host, self.tables, self.reqs = {}, {}, {}, {}
        self.free_pages = list(range(1, GEOMETRY.hot_rows // PAGE))      # page 0 is the padding sink
        self.free_logical = list(range(PAGE, GEOMETRY.resident_rows - PAGE))
        self.free_host = list(range(4096))
        self.misses = 0

    def _route(self, layer):
        return self.routes.route(layer)[0], self.routes.base(LAYOUT, layer)

    def _pages(self, count):
        pages = [self.free_pages.pop(0) for _ in range(count)]
        return [page * PAGE + i for page in pages for i in range(PAGE)]

    def admit(self, slot, uid, prompt):
        """Prefill, staging and alloc_device_buffer of the pinned coordinator."""
        device = self._pages(-(-prompt // PAGE))
        r = dict(uid=uid, seq=prompt, skip_first=True, host=[self.free_host.pop(0) for _ in range(prompt)],
                 logical=[self.free_logical.pop(0) for _ in range(prompt)])
        for layer in GEOMETRY.layers:
            mode, base = self._route(layer)
            for t in range(prompt):
                self.slab[base + (r["logical"][t] if mode == d.RESIDENT else device[t])] = (uid, layer, t)
                if mode == d.OFFLOADED or self.host_shadow:
                    self.host[(layer, r["host"][t])] = (uid, layer, t)
        alloc = min(-(-prompt // PAGE) * PAGE, H)
        alloc += PAGE if alloc == H else 0
        self.free_pages += sorted({row // PAGE for row in device[alloc:]})
        r["buffer"] = device[:alloc] + self._pages((alloc - len(device[:alloc])) // PAGE)
        self.reqs[slot] = r
        for layer in GEOMETRY.layers:
            base = self._route(layer)[1]
            locs = [base + row for row in r["buffer"]]
            self.tables[(layer, slot)] = (list(range(H)) + [-1] * PAGE, locs + [-1] * (H + PAGE - len(locs)),
                                          list(range(H)))

    def finish(self, slot):
        r = self.reqs.pop(slot)
        self.free_pages += sorted({row // PAGE for row in r["buffer"]})
        self.free_logical += r["logical"]
        self.free_host += r["host"]
        for layer in GEOMETRY.layers:
            self.tables.pop((layer, slot))

    def step(self, slots, selections, padded=0):
        """One decode step: eager flush and buffer growth, then per layer the store and swap of every row."""
        for slot in slots:
            r = self.reqs[slot]
            r["seq"] += 1
            r["logical"].append(self.free_logical.pop(0))
            if r["skip_first"]:
                r["skip_first"] = False
            else:
                self._flush(slot, r["seq"] - 2)
            self._grow(slot)
        reads = {}
        for layer in GEOMETRY.layers:
            for slot in [*slots, *[0] * padded]:
                self._store(layer, slot)
            for slot in slots:
                reads[(slot, layer)] = self._swap(layer, slot, selections[(slot, layer)])
        return reads

    def _flush(self, slot, p):
        r = self.reqs[slot]
        r["host"].append(self.free_host.pop(0))      # _eager_backup_previous_token allocates token p's host slot
        for layer in GEOMETRY.layers:
            mode, base = self._route(layer)
            if self.indirect:
                ident = self.slab[LAYOUT.tail_row(layer, slot)]
                if ident != (r["uid"], layer, p):
                    raise AssertionError(f"tail of layer {layer} slot {slot} holds {ident}, expected token {p}")
                if mode == d.RESIDENT:
                    self.slab[base + r["logical"][p]] = ident
                elif p < H:                              # short-request slot p, read by the short path
                    self.slab[base + r["buffer"][p]] = ident
            else:
                ident = self.slab[base + (r["logical"][p] if mode == d.RESIDENT else r["buffer"][min(p, H)])]
            if mode == d.OFFLOADED or (self.indirect and self.host_shadow):
                self.host[(layer, r["host"][p])] = ident

    def _grow(self, slot):
        """_grow_device_buffers: short requests take pages up to H, then the reserved page."""
        r = self.reqs[slot]
        seq, cap = r["seq"], len(r["buffer"])
        if seq <= H and seq > cap:
            new = min(-(-seq // PAGE) * PAGE, H)
            new += PAGE if new == H else 0
            r["buffer"] += self._pages((new - cap) // PAGE)
            for layer in GEOMETRY.layers:
                base = self._route(layer)[1]
                self.tables[(layer, slot)][1][cap:new] = [base + row for row in r["buffer"][cap:new]]

    def _store(self, layer, slot):
        if slot == 0:                                    # padded graph row: request slot 0, location 0
            self.slab[LAYOUT.tail_row(layer, 0) if self.indirect else self._route(layer)[1]] = ("padding", layer)
            return
        r = self.reqs[slot]
        newest, (mode, base) = r["seq"] - 1, self._route(layer)
        if self.indirect:
            row = LAYOUT.tail_row(layer, slot)
        else:
            row = base + (r["logical"][newest] if mode == d.RESIDENT else r["buffer"][min(newest, H)])
        self.slab[row] = (r["uid"], layer, newest)

    def _swap(self, layer, slot, selection):
        """Rows the swap kernel returns for one request; checks every row holds the selected token."""
        r = self.reqs[slot]
        seq, (mode, base) = r["seq"], self._route(layer)
        newest, valid = seq - 1, [t for t in selection if 0 <= t < seq]
        tail = LAYOUT.tail_row(layer, slot) if self.indirect else None
        if mode == d.RESIDENT:
            rows = {t: tail if self.indirect and t == newest else base + r["logical"][t] for t in valid}
        elif seq <= H:
            rows = {t: tail if self.indirect and t == newest else self.tables[(layer, slot)][1][t] for t in valid}
        else:
            tokens, locs, lru = self.tables[(layer, slot)]
            older = [t for t in valid if t != newest]
            present = set(tokens[:H])
            tags, order, picked, victims = reference_lru(tokens[:H], lru, older)
            for t, victim in zip([t for t in older if t not in present], victims):
                self.slab[locs[victim]] = self.host[(layer, r["host"][t])]
                self.misses += 1
            tokens[:H], lru[:] = tags, order
            rows = dict(zip(older, (locs[s] for s in picked)))
            if newest in valid:
                rows[newest] = tail if self.indirect else locs[H]
        idents = [self.slab.get(rows[t]) for t in valid]
        if idents != [(r["uid"], layer, t) for t in valid]:
            raise AssertionError(f"layer {layer} slot {slot} read {idents} for {valid}")
        return idents

    # Switch executor over the model: the device side of SwitchProtocol.
    def requests(self):
        return [d.Request(slot, r["seq"] if r["skip_first"] else r["seq"] - 1, tuple(r["logical"]),
                          tuple(r["host"]), tuple(r["buffer"])) for slot, r in sorted(self.reqs.items())]

    def quiesce(self):
        state = [(slot, r["uid"], r["seq"], r["skip_first"]) for slot, r in sorted(self.reqs.items())]
        return hashlib.sha256(json.dumps(state).encode()).hexdigest()

    def prepare(self, plan):
        ops = d.copies(LAYOUT, plan, self.requests(), host_shadow=self.host_shadow)
        for (_, row), (_, layer, host_row) in ops["phase1"]:
            self.host[(layer, host_row)] = self.slab[row]
        return len(ops["phase1"])

    def apply(self, plan, fail_after=None):
        """Phase 2, tables and descriptor; ``fail_after`` stops after that many row copies (copy failure)."""
        ops = d.copies(LAYOUT, plan, self.requests(), host_shadow=True)
        for done, ((_, layer, host_row), (_, row)) in enumerate(ops["phase2"]):
            if done == fail_after:
                raise RuntimeError("injected copy failure")
            self.slab[row] = self.host[(layer, host_row)]
        for layer, slot, tokens, locs, lru in ops["tables"]:
            self.tables[(layer, slot)] = (tokens, locs, lru)
        self.routes = plan.target
        return len(ops["phase2"])

    def switch(self, name):
        plan = d.plan_switch(LAYOUT, self.routes, LISTS[name])
        self.prepare(plan)
        self.apply(plan)
        return plan


def selections(rng, model, slots, width=6):
    """Distinct token positions per (request, layer); the newest token half the time, sometimes invalid."""
    chosen = {}
    for slot in slots:
        seq = model.reqs[slot]["seq"] + 1                # positions after this step's growth
        for layer in GEOMETRY.layers:
            pick = rng.sample(range(seq - 1), min(width, seq - 1)) if seq > 1 else []
            pick += [seq - 1] if rng.random() < 0.5 else []
            chosen[(slot, layer)] = pick + ([-1] if rng.random() < 0.2 else [])
    return chosen


def drive(models, events, seed=7):
    """Run one request history on every model; switches apply to the slab arm only. Returns per-step reads."""
    rng, log = random.Random(seed), []
    for kind, *args in events:
        if kind == "step":
            chosen = selections(rng, models[0], args[0])
            log.append([model.step(args[0], chosen, padded=args[1]) for model in models])
        elif kind == "switch":
            for model in models:
                if model.indirect:
                    model.switch(args[0])
        else:
            for model in models:
                getattr(model, kind)(*args)
    return log


def steps(slots, count, padded=0):
    return [("step", slots, padded)] * count


# Short (3), page-crossing (6) and long (11, 14) prompts; slot reuse, padding, admission next to switches.
HISTORY = [("admit", 1, "a", 3), ("admit", 2, "b", 11), *steps((1, 2), 4, padded=2), ("admit", 3, "c", 6),
           *steps((1, 2, 3), 5, padded=1), ("finish", 1), ("admit", 1, "d", 14), *steps((1, 2, 3), 6)]
SWITCHED = [("admit", 1, "a", 3), ("admit", 2, "b", 11), *steps((1, 2), 4, padded=2), ("switch", "B"),
            *steps((1, 2), 3), ("admit", 3, "c", 6), ("switch", "A"), *steps((1, 2, 3), 4, padded=1),
            ("finish", 1), ("switch", "C"), ("admit", 1, "d", 14), ("switch", "B"), *steps((1, 2, 3), 6),
            ("switch", "C"), ("switch", "A"), *steps((1, 2, 3), 3, padded=3)]


class LayoutTests(unittest.TestCase):
    def test_planes_tails_and_descriptor(self):
        self.assertEqual((LAYOUT.tail_base, LAYOUT.tail_stride, LAYOUT.rows), (3 * 64 + 2 * 128, 8, 488))
        self.assertTrue(all(LAYOUT.tail_view(layer) % PAGE == 0 for layer in GEOMETRY.layers))
        self.assertEqual(LAYOUT.tail_row(10, 4), LAYOUT.tail_base + 4 * 8 + 4)
        routes = d.initial_routes(LAYOUT, LISTS["A"])
        self.assertEqual(routes.descriptor(LAYOUT), [(1, 192, 448, 0), (0, 0, 456, 0), (0, 64, 464, 0),
                                                     (1, 320, 472, 0), (0, 128, 480, 0)])
        for bad in [(4, 6), (4, 6, 6), (4, 6, 12), (1, 4, 6)]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                d.initial_routes(LAYOUT, bad)
        with self.assertRaises(ValueError):
            LAYOUT.tail_row(2, 5)

    def test_plan_pairs_planes_and_rollback_is_the_inverse(self):
        routes = d.initial_routes(LAYOUT, LISTS["A"])
        plan = d.plan_switch(LAYOUT, routes, LISTS["B"])
        # 2 and 8 enter S on the hot planes of 4 and 10; 4 and 10 take the resident planes of 2 and 8.
        self.assertEqual(plan.entering, ((2, 0, 0), (8, 1, 2)))
        self.assertEqual(plan.leaving, ((4, 0, 0), (10, 2, 1)))
        self.assertEqual((plan.target.offload, plan.target.epoch), ((2, 6, 8), 1))
        back = d.rollback_plan(plan)
        self.assertEqual((back.target, back.entering, back.leaving), (routes, ((4, 0, 0), (10, 1, 2)),
                                                                     ((2, 0, 0), (8, 2, 1))))
        self.assertEqual(d.plan_switch(LAYOUT, routes, LISTS["A"]).entering, ())
        with self.assertRaises(d.SwitchError):
            d.plan_switch(LAYOUT, routes, (2, 4))
        planes = {(mode, plane) for _, mode, plane in plan.target.entries}
        self.assertEqual(len(planes), len(GEOMETRY.layers))       # one layer per plane, no borrowing

    def test_copy_plan_rows(self):
        model = Rank(LISTS["A"])
        drive([model], [("admit", 1, "a", 3), ("admit", 2, "b", 11), *steps((1, 2), 2)])
        plan = d.plan_switch(LAYOUT, model.routes, LISTS["B"])
        requests = model.requests()
        self.assertEqual([r.home for r in requests], [4, 12])
        ops = d.copies(LAYOUT, plan, requests, host_shadow=False)
        # Phase 1: every home token of both entering layers; phase 2: hot init up to H, then full history.
        self.assertEqual(len(ops["phase1"]), 2 * (4 + 12))
        self.assertEqual(len(ops["phase2"]), 2 * (4 + H) + 2 * (4 + 12))
        self.assertEqual(d.copies(LAYOUT, plan, requests, host_shadow=True)["phase1"], [])
        writes = {row for _, (_, row) in ops["phase2"]}
        moved = [(LAYOUT.hot_base(0), 64), (LAYOUT.hot_base(2), 64), (LAYOUT.resident_base(0), 128),
                 (LAYOUT.resident_base(1), 128)]
        self.assertTrue(all(any(base <= row < base + size for base, size in moved) for row in writes))
        self.assertEqual(len(ops["tables"]), 4)

    def test_ledger_counts_tails_tables_and_host_of_the_union(self):
        ledger = d.ledger(LAYOUT)
        dynamic, frozen = ledger["device"]["dynamic"], ledger["device"]["frozen"]
        row = PAGE * 256 * 2 // PAGE
        self.assertEqual((dynamic["hot"], dynamic["resident"]), (3 * 64 * row, 2 * 128 * row))
        self.assertEqual((frozen["hot"], frozen["resident"]), (dynamic["hot"], dynamic["resident"]))
        self.assertEqual(dynamic["tails"], 5 * 8 * row)
        self.assertEqual(dynamic["tables"] - frozen["tables"], 2 * d.table_bytes(GEOMETRY))
        self.assertEqual(ledger["host"], dict(dynamic=5 * 128 * row, frozen=3 * 128 * row))


class OracleTests(unittest.TestCase):
    """Every read is checked against the true token inside the model; these tests add the cross-arm checks."""

    def test_zero_switches_match_the_frozen_path_hit_for_hit(self):
        for shadow in (False, True):
            with self.subTest(host_shadow=shadow):
                slab, frozen = Rank(LISTS["A"], host_shadow=shadow), Rank(LISTS["A"], indirect=False)
                log = drive([slab, frozen], HISTORY)
                self.assertTrue(all(a == b for a, b in log))
                self.assertGreater(frozen.misses, 0)
                self.assertEqual(slab.misses, frozen.misses)

    def test_switch_sequences_read_what_the_frozen_target_reads(self):
        for shadow in (False, True):
            with self.subTest(host_shadow=shadow):
                slab = Rank(LISTS["A"], host_shadow=shadow)
                frozen = {name: Rank(offload, indirect=False) for name, offload in LISTS.items()}
                log = drive([slab, *frozen.values()], SWITCHED)
                self.assertEqual(len(log), 20)
                self.assertTrue(all(row[0] == other for row in log for other in row[1:]))
                self.assertEqual(slab.routes.offload, LISTS["A"])
                self.assertEqual(slab.routes.epoch, 6)

    def test_host_shadow_keeps_resident_history_on_host(self):
        model = Rank(LISTS["A"], host_shadow=True)
        drive([model], SWITCHED[:7])
        plan = d.plan_switch(LAYOUT, model.routes, LISTS["C"])
        self.assertEqual(model.prepare(plan), 0)
        model.apply(plan)
        drive([model], steps((1, 2), 3))

    def test_partial_rebuild_rolls_back_from_host(self):
        for fail_after in (0, 5, 30):
            with self.subTest(fail_after=fail_after):
                model, reference = Rank(LISTS["A"]), Rank(LISTS["A"], indirect=False)
                drive([model, reference], SWITCHED[:7])
                source = model.routes
                plan = d.plan_switch(LAYOUT, source, LISTS["C"])
                model.prepare(plan)
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    model.apply(plan, fail_after=fail_after)
                model.apply(d.rollback_plan(plan))
                self.assertEqual(model.routes, source)
                log = drive([model, reference], [*steps((1, 2), 4), ("admit", 3, "c", 9), *steps((1, 2, 3), 4)])
                self.assertTrue(all(a == b for a, b in log))

    def test_switch_before_the_first_decode_step(self):
        model = Rank(LISTS["B"])
        drive([model], [("admit", 1, "a", 13), ("switch", "A"), ("admit", 2, "b", 5), ("switch", "C"),
                        *steps((1, 2), 6, padded=1), ("finish", 2), ("switch", "B"), ("admit", 2, "e", 2),
                        *steps((1, 2), 5)])
        self.assertEqual(model.routes.offload, LISTS["B"])


class Exchange:
    def __init__(self, world, failed_phase=None):
        self.records, self.failed_phase = [None] * world, failed_phase
        self.barrier = Barrier(world, timeout=10)


class Transport:
    """Thread stand-in for GlooDemotionTransport; ``failed_phase`` times out after the exchange completes."""

    def __init__(self, rank, exchange):
        self.rank, self.exchange, self.world_size = rank, exchange, len(exchange.records)

    def all_gather(self, value):
        self.exchange.records[self.rank] = value
        self.exchange.barrier.wait()
        result = tuple(self.exchange.records)
        self.exchange.barrier.wait()
        if value[0] == self.exchange.failed_phase:
            raise TimeoutError("control group timeout")
        return result


class Executor:
    """A rank model as the protocol's executor, with at most one injected fault."""

    def __init__(self, model, fault=None):
        self.model, self.fault, self.applied, self.published = model, fault, 0, None

    def quiesce(self):
        if self.fault == "in_flight":
            raise d.SwitchError("a prefill chunk is in flight")
        return self.model.quiesce() + ("-one-step-behind" if self.fault == "skew" else "")

    def prepare(self, plan):
        if self.fault == "d2h":
            raise RuntimeError("device-to-host copy failed")
        return self.model.prepare(plan)

    def apply(self, plan):
        self.applied += 1
        if self.fault == "copy" and self.applied == 1:
            return self.model.apply(plan, fail_after=3)
        if self.fault == "interrupt" and self.applied == 1:
            raise KeyboardInterrupt
        if self.fault == "rebuild":
            raise RuntimeError("rebuild failed")
        return self.model.apply(plan)

    def publish(self, routes):
        if self.fault == "publish":
            raise RuntimeError("publish failed")
        self.published = routes


PREFIX = [("admit", 1, "a", 3), ("admit", 2, "b", 11), *steps((1, 2), 4, padded=1)]


def ranks(world, faults=None, failed_phase=None):
    """``world`` TP ranks with identical request histories."""
    exchange, group = Exchange(world, failed_phase), []
    for rank in range(world):
        model = Rank(LISTS["A"])
        drive([model], PREFIX)
        protocol = d.SwitchProtocol(LAYOUT, LISTS, Executor(model, (faults or {}).get(rank)),
                                    Transport(rank, exchange), active="A", config_digest="config-v1")
        group.append((protocol, model))
    return group


def switch_all(group, names):
    def run(index):
        try:
            return group[index][0].switch(names[index])
        except (d.SwitchError, d.SwitchProtocolError) as exc:
            return exc

    with ThreadPoolExecutor(len(group)) as pool:
        return list(pool.map(run, range(len(group))))


def gloo_worker(rank, address, output):
    import torch.distributed as dist
    from .layer_eviction_tp import GlooDemotionTransport
    dist.init_process_group("gloo", init_method=address, rank=rank, world_size=2, timeout=timedelta(seconds=25))
    try:
        group = dist.new_group(backend="gloo", timeout=timedelta(seconds=10))
        model = Rank(LISTS["A"])
        drive([model], PREFIX)
        protocol = d.SwitchProtocol(LAYOUT, LISTS, Executor(model), GlooDemotionTransport(group), active="A",
                                    config_digest="config-v1")
        receipt = protocol.switch("B")
        drive([model], steps((1, 2), 2), seed=9)
        output.put((rank, receipt["receipt_sha256"], protocol.routes.offload))
    finally:
        dist.destroy_process_group()


try:
    import torch.distributed as _dist
    GLOO = _dist.is_available() and _dist.is_gloo_available()
except ImportError:
    GLOO = False


class ProtocolTests(unittest.TestCase):
    def test_tp8_commits_one_receipt_and_keeps_decoding(self):
        group = ranks(8)
        results = switch_all(group, ["B"] * 8)
        self.assertEqual(len({result["receipt_sha256"] for result in results}), 1)
        first = results[0]
        self.assertEqual([first[key] for key in ("status", "epoch", "source", "target")], ["committed", 1, "A", "B"])
        self.assertEqual(first["rows"], dict(prepare=40, apply=68))
        for protocol, model in group:
            self.assertEqual(model.routes, protocol.routes)
            self.assertEqual((protocol.routes.offload, protocol.active), (LISTS["B"], "B"))
            self.assertIs(protocol.executor.published, protocol.routes)
        logs = [drive([model], steps((1, 2), 3), seed=3) for _, model in group]
        self.assertTrue(all(log == logs[0] for log in logs))
        self.assertEqual({result["epoch"] for result in switch_all(group, ["A"] * 8)}, {2})
        self.assertEqual({result["status"] for result in switch_all(group, ["A"] * 8)}, {"unchanged"})

    def test_refusals_keep_the_source_routes(self):
        cases = [({3: "in_flight"}, ["B"] * 8, "in flight"), ({5: "skew"}, ["B"] * 8, "rank skew"),
                 ({2: "d2h"}, ["B"] * 8, "device-to-host"), ({}, ["B"] * 7 + ["C"], "different"),
                 ({}, ["B"] * 7 + ["Z"], "unregistered")]
        for faults, names, pattern in cases:
            with self.subTest(pattern=pattern):
                group = ranks(8, faults)
                for (protocol, model), result in zip(group, switch_all(group, names)):
                    self.assertIsInstance(result, d.SwitchError)
                    self.assertRegex(str(result), pattern)
                    self.assertFalse(protocol.poisoned)
                    self.assertEqual((model.routes.offload, protocol.routes.epoch), (LISTS["A"], 0))
                for protocol, _ in group:
                    protocol.executor.fault = None
                self.assertEqual({r["status"] for r in switch_all(group, ["B"] * 8)}, {"committed"})

    def test_copy_failure_and_interruption_roll_back_every_rank(self):
        for fault in ("copy", "interrupt"):
            with self.subTest(fault=fault):
                group = ranks(8, {6: fault})
                for (protocol, model), result in zip(group, switch_all(group, ["B"] * 8)):
                    self.assertIsInstance(result, d.SwitchError)
                    self.assertIn("rolled back", str(result))
                    self.assertEqual((model.routes.offload, protocol.receipts[-1]["status"]),
                                     (LISTS["A"], "rolled_back"))
                    self.assertFalse(protocol.poisoned)
                logs = [drive([model], steps((1, 2), 3), seed=5) for _, model in group]
                self.assertTrue(all(log == logs[0] for log in logs))
                self.assertEqual({r["status"] for r in switch_all(group, ["B"] * 8)}, {"committed"})

    def test_failed_rollback_publication_divergence_and_timeouts_poison_the_group(self):
        cases = [dict(faults={1: "rebuild"}), dict(faults={3: "publish"}),
                 *(dict(failed_phase=phase) for phase in ("propose", "prepare", "apply", "commit"))]
        for case in cases:
            with self.subTest(**case):
                group = ranks(4, **case)
                results = switch_all(group, ["B"] * 4)
                self.assertTrue(all(isinstance(result, d.SwitchProtocolError) for result in results))
                self.assertTrue(all(protocol.poisoned for protocol, _ in group))
                with self.assertRaisesRegex(d.SwitchProtocolError, "poisoned"):
                    group[0][0].switch("C")
        for skew in ("epoch", "config"):
            with self.subTest(skew=skew):
                group = ranks(4)
                protocol = group[2][0]
                if skew == "epoch":
                    protocol.routes = d.Routes(protocol.routes.entries, epoch=3)
                else:
                    protocol._config = ("config-v2", *protocol._config[1:])
                results = switch_all(group, ["B"] * 4)
                self.assertTrue(all(isinstance(result, d.SwitchProtocolError) for result in results))
                self.assertTrue(all(model.routes.offload == LISTS["A"] for _, model in group))

    def test_concurrent_entry_poisons_and_configuration_is_checked(self):
        protocol = ranks(1)[0][0]
        protocol._lock.acquire()
        with self.assertRaisesRegex(d.SwitchProtocolError, "concurrent"):
            protocol.switch("B")
        protocol._lock.release()
        self.assertTrue(protocol.poisoned)
        executor, transport = Executor(Rank(LISTS["A"])), Transport(0, Exchange(1))
        cases = [(LISTS, "Z", "x", "active"), ({"A": (4, 6, 10), "D": (10, 6, 4)}, "A", "x", "same offload list"),
                 ({"A": (4, 6, 10), "B": (4, 6, 8)}, "A", "x", "union"), (LISTS, "A", "", "digest"),
                 ({"A": (4, 6)}, "A", "x", "distinct switchable")]
        for lists, active, digest, pattern in cases:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                d.SwitchProtocol(LAYOUT, lists, executor, transport, active=active, config_digest=digest)

    @unittest.skipUnless(GLOO, "CPU Gloo unavailable")
    def test_gloo_process_group(self):
        ctx = multiprocessing.get_context("spawn")
        with TemporaryDirectory() as directory:
            address = (Path(directory) / "rendezvous").as_uri()
            output = ctx.Queue()
            processes = [ctx.Process(target=gloo_worker, args=(i, address, output)) for i in range(2)]
            try:
                for process in processes:
                    process.start()
                results = sorted(output.get(timeout=60) for _ in processes)
                self.assertEqual([(rank, offload) for rank, _, offload in results], [(0, LISTS["B"]), (1, LISTS["B"])])
                self.assertEqual(results[0][1], results[1][1])
                for process in processes:
                    process.join(timeout=10)
                    self.assertEqual(process.exitcode, 0)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                output.close()


if __name__ == "__main__":
    unittest.main()
