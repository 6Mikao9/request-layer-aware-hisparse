"""CPU tests of the layer offload adapter: torch slab state against the pure model, page-padded rows, dry runs."""
import collections
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest.mock
import unittest

from . import dynamic_offload as do
from . import dynamic_offload_device as dev
from . import study_config as sc
from .test_dynamic_offload import GEOMETRY, H, LAYOUT, LISTS, PAGE, SWITCHED, HISTORY, Rank, drive, steps

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).parent
SRC = os.environ.get("SGLANG_SRC")


def encode(uid, layer, token):
    return [float(uid), float(layer), float(token), 1.0]


class DeviceRank:
    """The torch SlabState and route statement driven like ``test_dynamic_offload.Rank``; the native parts of the
    coordinator (allocator, admission, growth, request_finished) are restated on plain tensors. The A100 gate
    runs the same history on CUDA with the route kernel as ``swap`` (runtime.cuda_dynamic_offload_correctness)."""
    indirect = True

    def __init__(self, offload, *, host_shadow=False, device="cpu", swap=None):
        t, g, slots, ctx, d = torch, GEOMETRY, GEOMETRY.slots, 64, torch.device(device)
        self.uids, self.swap, width = {}, swap or dev.emulate_route, H + PAGE
        self.c = c = types.SimpleNamespace(
            req_to_token_pool=types.SimpleNamespace(req_to_token=t.zeros((slots, ctx), dtype=t.int32, device=d)),
            req_to_device_buffer=t.zeros((slots, width), dtype=t.int64, device=d),
            req_device_buffer_size=t.zeros(slots, dtype=t.int64), req_to_host_pool=t.full((slots, ctx), -1, device=d),
            req_device_buffer_tokens=t.full((len(g.layers), slots, width), -1, dtype=t.int32, device=d),
            req_device_buffer_token_locs=t.full((len(g.layers), slots, width), -1, dtype=t.int32, device=d),
            lru_slots=t.arange(H, dtype=t.int16, device=d).repeat(len(g.layers), slots, 1),
            top_k_device_locs_buffer=t.full((slots, H), -1, dtype=t.int32, device=d),
            num_real_reqs=t.zeros(1, dtype=t.int32, device=d), decode_backup_stream=None, decode_producer_stream=None,
            item_size_bytes=16, swap_in_block_size=256, device_buffer_size=H)
        pinned = d.type == "cuda"
        host = types.SimpleNamespace(k_buffer=[t.zeros((4096, 1, 4), pin_memory=pinned) for _ in g.layers],
                                     v_buffer=[t.zeros((4096, 1, 4), pin_memory=pinned) for _ in g.layers])
        self.k, self.v = t.zeros((LAYOUT.rows, 1, 4), device=d), t.zeros((LAYOUT.rows, 1, 4), device=d)
        lists = {name: tuple(layers) for name, layers in LISTS.items()}
        self.s = dev.SlabState(LAYOUT, lists, dev.MHARows(t, self.k, self.v, host), active="A", host_shadow=host_shadow,
                               ratio=1, torch=t, device=d)
        self.s.routes = do.initial_routes(LAYOUT, offload)
        self.s.route.copy_(t.tensor(self.s.routes.descriptor(LAYOUT), dtype=t.int32))
        self.s.coordinator, self.mapping = c, t.zeros(GEOMETRY.resident_rows, dtype=t.int64, device=d)
        self.free_pages = list(range(1, g.hot_rows // PAGE))
        self.free_logical, self.free_host = list(range(PAGE, g.resident_rows - PAGE)), list(range(4096))
        self.reqs, self.skip = {}, {}

    @property
    def routes(self):
        return self.s.routes

    def _write(self, rows, uid, layer, token):
        value = torch.tensor(encode(self.uids[uid], layer, token), device=self.k.device)
        self.k[rows], self.v[rows] = value, -value

    def _pages(self, count):
        pages = [self.free_pages.pop(0) for _ in range(count)]
        return [page * PAGE + i for page in pages for i in range(PAGE)]

    def admit(self, slot, uid, prompt):
        self.uids.setdefault(uid, len(self.uids) + 1)
        c, s, t = self.c, self.s, torch
        device = self._pages(-(-prompt // PAGE))
        logical = [self.free_logical.pop(0) for _ in range(prompt)]
        c.req_to_token_pool.req_to_token[slot, :prompt] = s._long(logical).to(t.int32)
        self.mapping[s._long(logical)] = s._long(device[:prompt])
        for layer in GEOMETRY.layers:                       # prefill stores through the routes
            base, _ = s.plane(layer)
            for token in range(prompt):
                self._write(base + (device[token] if s.offloaded(layer) else logical[token]), uid, layer, token)
        host = [self.free_host.pop(0) for _ in range(prompt)]
        c.req_to_host_pool[slot, :prompt] = s._long(host)
        s.stage(s._long(logical), self.mapping[s._long(logical)], s._long(host))
        alloc = min(-(-prompt // PAGE) * PAGE, H)
        alloc += PAGE if alloc == H else 0
        self.free_pages += sorted({row // PAGE for row in device[alloc:]})
        buffer = device[:alloc] + self._pages((alloc - len(device[:alloc])) // PAGE)
        c.req_to_device_buffer[slot, :alloc] = s._long(buffer)
        c.req_device_buffer_size[slot] = alloc
        c.req_device_buffer_tokens[:, slot, :H] = s._long(range(H)).to(t.int32)
        c.req_device_buffer_token_locs[:, slot, :alloc] = s._long(buffer).to(t.int32)
        s.fix_locs(slot, 0, alloc)
        s.home[slot] = prompt
        self.reqs[slot], self.skip[slot] = dict(uid=uid, seq=prompt), True

    def _grow(self, slot):
        c, s, t, r = self.c, self.s, torch, self.reqs[slot]
        seq, cap = r["seq"], int(c.req_device_buffer_size[slot])
        if cap < seq <= H:
            new = min(-(-seq // PAGE) * PAGE, H)
            new += PAGE if new == H else 0
            rows = s._long(self._pages((new - cap) // PAGE))
            c.req_to_device_buffer[slot, cap:new] = rows
            c.req_device_buffer_token_locs[:, slot, cap:new] = rows.to(t.int32)
            c.req_device_buffer_size[slot] = new
            s.fix_locs(slot, cap, new)

    def step(self, slots, selections, padded=0):
        c, s, t = self.c, self.s, torch
        flushed, records = [], []
        for slot in slots:
            r = self.reqs[slot]
            r["seq"] += 1
            c.req_to_token_pool.req_to_token[slot, r["seq"] - 1] = self.free_logical.pop(0)
            if self.skip[slot]:
                self.skip[slot] = False
            else:
                flushed.append(slot)
                records.append(r["seq"] - 2)
        if flushed:
            host = [self.free_host.pop(0) for _ in flushed]
            for slot, record, row in zip(flushed, records, host):
                c.req_to_host_pool[slot, record] = row
            s.flush(flushed, records, s._long(host))
        for slot in slots:
            self._grow(slot)
        rows = [*slots, *[0] * padded]
        rpi, seq_lens = s._long(rows), s._long([self.reqs[x]["seq"] if x else 1 for x in rows])
        c.num_real_reqs[0], reads = len(slots), {}
        names = {index: uid for uid, index in self.uids.items()}
        for layer in GEOMETRY.layers:
            for slot in rows:                                # decode store: the tail row of each graph row's slot
                if slot:
                    self._write(LAYOUT.tail_row(layer, slot), self.reqs[slot]["uid"], layer, self.reqs[slot]["seq"] - 1)
                else:
                    self.k[LAYOUT.tail_row(layer, 0)] = -7.0
            chosen = [selections[(slot, layer)] for slot in slots] + [[] for _ in range(padded)]
            out = self.swap(s, c, rpi, seq_lens, chosen, layer, c.top_k_device_locs_buffer[:len(rows), :H])
            for b, slot in enumerate(slots):
                got = [int(out[b, i]) for i, p in enumerate(chosen[b]) if 0 <= p < self.reqs[slot]["seq"]]
                if not torch.equal(self.v[got], -self.k[got]):
                    raise AssertionError("K and V rows diverged")
                reads[(slot, layer)] = [(names[int(v[0])], int(v[1]), int(v[2])) for v in self.k[got, 0].tolist()]
            if padded and not bool((out[len(slots):] == -1).all()):
                raise AssertionError("padded graph rows must resolve to -1")
        return reads

    def finish(self, slot):
        c, s, t = self.c, self.s, torch
        s.restore_locs(slot)
        cap, r = int(c.req_device_buffer_size[slot]), self.reqs.pop(slot)
        self.free_pages += sorted({int(row) // PAGE for row in c.req_to_device_buffer[slot, :cap]})
        self.free_logical += c.req_to_token_pool.req_to_token[slot, :r["seq"]].tolist()
        self.free_host += [x for x in c.req_to_host_pool[slot].tolist() if x >= 0]
        for table in (c.req_device_buffer_tokens, c.req_device_buffer_token_locs):
            table[:, slot] = -1
        c.req_to_device_buffer[slot], c.req_device_buffer_size[slot], c.req_to_host_pool[slot] = 0, 0, -1
        c.lru_slots[:, slot] = s._long(range(H)).to(t.int16)
        s.home.pop(slot)

    def switch(self, name):
        plan = do.plan_switch(LAYOUT, self.s.routes, LISTS[name])
        self.s.prepare(plan)
        self.s.apply(plan)
        self.s.publish(plan.target)
        return plan


@unittest.skipIf(torch is None, "CPU torch unavailable")
class DeviceStateTests(unittest.TestCase):
    def test_zero_switches_match_the_frozen_path(self):
        device, frozen = DeviceRank(LISTS["A"]), Rank(LISTS["A"], indirect=False)
        log = drive([device, frozen], HISTORY)
        self.assertTrue(all(a == b for a, b in log))
        self.assertEqual(sorted(device.s.home), [1, 2, 3])

    def test_switch_sequences_match_the_model_and_every_frozen_list(self):
        for shadow in (False, True):
            with self.subTest(host_shadow=shadow):
                device, model = DeviceRank(LISTS["A"], host_shadow=shadow), Rank(LISTS["A"], host_shadow=shadow)
                frozen = [Rank(offload, indirect=False) for offload in LISTS.values()]
                log = drive([device, model, *frozen], SWITCHED)
                self.assertTrue(all(row[0] == other for row in log for other in row[1:]))
                self.assertEqual((device.routes, device.s.route.tolist()),
                                 (model.routes, [list(r) for r in model.routes.descriptor(LAYOUT)]))
                locs = device.c.req_device_buffer_token_locs
                for index, layer in enumerate(GEOMETRY.layers):   # offloaded tables hold absolute rows of their plane
                    base, rows = device.s.plane(layer)
                    if device.s.offloaded(layer):
                        live = locs[index, 1, :int(device.c.req_device_buffer_size[1])]
                        self.assertTrue(bool(((live >= base) & (live < base + rows)).all()))

    def test_failed_rebuild_rolls_back_from_host(self):
        device, frozen = DeviceRank(LISTS["A"]), Rank(LISTS["A"], indirect=False)
        drive([device, frozen], SWITCHED[:7])
        plan, source, h2d = do.plan_switch(LAYOUT, device.routes, LISTS["C"]), device.routes, device.s.rows.h2d
        calls = []

        def failing(*args):
            calls.append(args)
            if len(calls) == 2:
                raise RuntimeError("injected copy failure")
            return h2d(*args)

        device.s.prepare(plan)
        device.s.rows.h2d = failing
        with self.assertRaisesRegex(RuntimeError, "injected"):
            device.s.apply(plan)
        device.s.rows.h2d = h2d
        device.s.apply(do.rollback_plan(plan))
        self.assertEqual(device.s.route.tolist(), [list(r) for r in source.descriptor(LAYOUT)])
        log = drive([device, frozen], [*steps((1, 2), 3), ("admit", 3, "c", 9), *steps((1, 2, 3), 4)])
        self.assertTrue(all(a == b for a, b in log))

    def test_paged_rows_follow_the_c4_page_layout(self):
        pages, item = 4, 37440
        slab = torch.zeros((pages, item), dtype=torch.uint8)
        host = types.SimpleNamespace(kv_buffer=[torch.zeros((2, item), dtype=torch.uint8)])
        rows = dev.PagedRows(torch, slab, host)

        def record(buffer, row):
            flat = buffer.view(-1)
            page, slot = divmod(row, 64)
            value = flat[page * item + slot * 576: page * item + slot * 576 + 576]
            scale = flat[page * item + 36864 + slot * 8: page * item + 36864 + slot * 8 + 8]
            return value, scale

        for row in (3, 64, 200):
            value, scale = record(slab, row)
            value.copy_(torch.randint(0, 255, (576,), dtype=torch.uint8))
            scale.copy_(torch.randint(0, 255, (8,), dtype=torch.uint8))
        rows.d2d(torch.tensor([3, 64]), torch.tensor([130, 255]))
        rows.d2h(0, torch.tensor([200]), torch.tensor([65]))
        rows.h2d(0, torch.tensor([65]), torch.tensor([1]))
        for src, dst, target in ((3, 130, slab), (64, 255, slab), (200, 65, host.kv_buffer[0]), (200, 1, slab)):
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(record(slab, src), record(target, dst))))
        untouched = record(slab, 2)
        self.assertTrue(all(int(part.abs().sum()) == 0 for part in untouched))

    def test_direct_resident_c4_slots(self):
        req_to_token = torch.arange(4 * 64, dtype=torch.int32).view(4, 64) + 1000
        c = types.SimpleNamespace(req_to_token_pool=types.SimpleNamespace(req_to_token=req_to_token),
                                  top_k_device_locs_buffer=torch.zeros((4, 3), dtype=torch.int32),
                                  num_real_reqs=torch.tensor([2], dtype=torch.int32))
        top_k = torch.tensor([[0, 2, -1], [5, 1, 9], [0, 0, 0]], dtype=torch.int32)
        out = dev.resident_slots(c, torch.tensor([1, 3, 0]), torch.tensor([3, 6, 4]), top_k, torch)
        expect = [[(1064 + 3) // 4, (1064 + 11) // 4, -1], [(1192 + 23) // 4, (1192 + 7) // 4, -1], [-1, -1, -1]]
        self.assertEqual(out.tolist(), expect)

    def test_flush_work_is_fixed_and_sync_free(self):
        """The decode flush issues the same index ops for 5 and 40 switchable layers, never a boolean-mask
        selection (a host sync on CUDA), one index upload, and the pinned backup's host copies: per layer for
        MiniMax rows, one launch for C4 pages."""
        from torch.profiler import ProfilerActivity, profile

        def flush(paged, layers):
            page = 64 if paged else PAGE
            g = do.Geometry(layers=tuple(range(layers)), hot_rows=page * 8, resident_rows=page * 16, hot_slots=8,
                            slots=5, page=page, page_bytes=37440 if paged else page * 16, buffers=1 if paged else 2)
            layout = do.SlabLayout(g, layers // 2)
            host = types.SimpleNamespace(kv_buffer=[torch.zeros((2, 37440), dtype=torch.uint8)] * layers,
                                         k_buffer=[torch.zeros((64, 1, 4))] * layers,
                                         v_buffer=[torch.zeros((64, 1, 4))] * layers)
            rows = dev.PagedRows(torch, torch.zeros((layout.rows // 64, 37440), dtype=torch.uint8), host) if paged \
                else dev.MHARows(torch, torch.zeros((layout.rows, 1, 4)), torch.zeros((layout.rows, 1, 4)), host)
            s = dev.SlabState(layout, {"0": tuple(range(layers // 2))}, rows, active="0", host_shadow=False,
                              ratio=4 if paged else 1, torch=torch, device=torch.device("cpu"))
            s.coordinator = types.SimpleNamespace(
                req_to_token_pool=types.SimpleNamespace(req_to_token=torch.arange(5 * 64, dtype=torch.int32)
                                                        .view(5, 64) % (page * 15)),
                req_to_device_buffer=torch.arange(5 * 12).view(5, 12) % (page * 7),
                decode_backup_stream=None, decode_producer_stream=None)
            launches = collections.Counter()
            rows.d2d = lambda *a: launches.update(["d2d"])
            if paged:                                          # one transfer_cache_dsv4_mla launch per plan
                rows.d2h_plan = lambda plan, *a: launches.update(["host"] if plan[0] else [])
            else:                                              # one transfer_kv_per_layer launch per layer
                rows._host = lambda *a: launches.update(["host"])
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                s.flush([1, 2, 3, 4], [0, 3, 7, 9], torch.tensor([0, 1, 0, 1]))
            views = {"aten::select", "aten::slice", "aten::view", "aten::unsqueeze", "aten::as_strided",
                     "aten::expand", "aten::flatten", "aten::reshape", "aten::alias"}     # no CUDA launch
            top = [e.name for e in prof.events() if e.name.startswith("aten::") and e.name not in views
                   and (e.cpu_parent is None or not e.cpu_parent.name.startswith("aten::"))]
            masks = sum(e.name in ("aten::nonzero", "aten::masked_select") for e in prof.events())
            return len(top), masks, dict(launches), s

        for paged in (False, True):
            with self.subTest(paged=paged):
                (few, few_masks, _, _), (many, many_masks, launches, s) = flush(paged, 5), flush(paged, 40)
                self.assertEqual((few, few_masks, many_masks), (many, 0, 0))
                self.assertEqual(launches, dict(d2d=1, host=1 if paged else 20))
        indices, bases, src, _ = s.copy_plans().flush         # C4 pointers start at the tail view's page
        self.assertEqual(src.tolist(), [s.rows.k[base // 64].data_ptr() for base in bases])
        with self.assertRaisesRegex(ValueError, "page"):
            s.rows.plan([0], [3])

    def test_a100_gate_harness_on_cpu(self):
        """The gate's own drivers with the statement in place of the kernel: history with a rollback, and both
        models' production geometry through a switch (the A100 run then only swaps in the kernel)."""
        from . import cuda_dynamic_offload_correctness as gate

        class Torch:
            cuda = types.SimpleNamespace(synchronize=lambda: None)

            def __getattr__(self, name):
                return getattr(torch, name)

        def statement(model):
            block = gate.SPECS[model]["block"]

            def run(s, c, rpi, seq, chosen, layer, out):
                expanded = [[None if b < 0 else b * block + o for b in picks for o in range(block)] if block > 1
                            else picks for picks in chosen]
                return dev.emulate_route(s, c, rpi, seq, expanded, layer, out)
            return run

        previous, gate.DEVICE = gate.DEVICE, "cpu"
        original, gate.cuda_rank = gate.cuda_rank, lambda t, offload, swap, shadow: DeviceRank(offload,
                                                                                            host_shadow=shadow)
        try:
            result = gate.history(Torch(), graph=False, host_shadow=False)
            self.assertEqual((result["passed"], result["events"]), (True, 37), result["problems"])
            for model in gate.SPECS:
                result = gate.production(Torch(), model, kernel=statement(model))
                self.assertTrue(result["passed"], result["problems"])
                self.assertGreater(result["compared_rows"], 8000)
                self.assertEqual(result["final_routes"][0][0], do.RESIDENT)    # layer 2 left S in the switch
                if model == "minimax_m3":                   # consumer over routed rows vs FP32 and frozen reader
                    self.assertEqual(result["attention_checks"], 2 * 2 * 4)
                    self.assertLess(result["fp32_max_abs_error"], 0.002)
                else:
                    self.assertIsNone(result["fp32_max_abs_error"])
        finally:
            gate.DEVICE, gate.cuda_rank = previous, original


class AdapterTests(unittest.TestCase):
    def test_layers_split_and_geometry(self):
        self.assertEqual(dev.parse_layers("2-42:2"), tuple(range(2, 43, 2)))
        self.assertEqual(dev.parse_layers("7,3,3-4"), (3, 4, 7))
        config = sc.parse({"schema": sc.SCHEMA, "placement": {"mode": "fixed_layers", "offload_layers": [4, 6],
                                                               "alternate_offload_layers": [[6, 8]]},
                           "features": {"route_indirect": True}})
        self.assertEqual(dev.split_layers(config, (2, 4, 6, 8, 10)), ({"0": (4, 6), "1": (6, 8)}, (4, 6, 8), (2, 10)))
        with self.assertRaisesRegex(RuntimeError, "not eligible"):
            dev.split_layers(config, (2, 4, 6))
        geo = dict(dev.PRESETS["minimax_m3"], eligible=dev.parse_layers("0-59"))
        mm = sc.parse(json.loads((ROOT / "fixtures" / "dynamic_offload_arms" / "minimax_dynamic.json").read_text()))
        lists, slab, resident, layout = dev.plan_geometry(mm, "minimax_m3", **geo)
        self.assertEqual((len(lists["0"]), len(slab), len(resident), layout.k), (40, 51, 9, 40))
        self.assertEqual(layout.rows, 40 * 396_672 + 11 * (2 * 396_544 + 128) + 51 * 256)
        ledger = do.ledger(layout)["device"]
        self.assertEqual(ledger["dynamic"]["hot"] + ledger["dynamic"]["resident"],
                         ledger["frozen"]["hot"] + ledger["frozen"]["resident"])

    def test_install_is_inert_without_a_fixed_offload_set(self):
        self.assertFalse(dev.install(sc.parse({"schema": sc.SCHEMA}), "0" * 64))
        arena = sc.parse({"schema": sc.SCHEMA, "hot_pool": {"mode": "shared", "layout": "arena"}})
        self.assertFalse(dev.install(arena, "0" * 64))
        self.assertIsNone(dev._STATE)
        with self.assertRaisesRegex(RuntimeError, "full_resident"):
            dev.install(sc.parse({"schema": sc.SCHEMA, "placement": {"mode": "full_resident"}}), "0" * 64)
        self.assertFalse(dev.uninstall())

    def test_route_header_is_pinned(self):
        manifest = json.loads((ROOT / "PROVENANCE.json").read_text())
        self.assertEqual(dev.route_sha256(), manifest["dynamic_offload_generated_sha256"]["hisparse_route.cuh"])
        text = dev.ROUTE_HEADER.read_text()
        self.assertEqual(text.count("RESIDENT_RATIO,\n"), 5)       # four launch instantiations and the kernel
        self.assertEqual(text.count("newest_row"), 5)       # defined once, bound in four paths
        self.assertNotIn("req_device_buffer_locs[newest_slot]", text)
        if SRC:
            from .build_route_overlay import SOURCE, derive
            root = Path(SRC) / "python" / "sglang" if (Path(SRC) / "python" / "sglang").is_dir() else Path(SRC)
            self.assertEqual(derive((root / SOURCE).read_text()), text)

    @unittest.skipUnless(SRC, "set SGLANG_SRC to a pinned checkout")
    def test_pinned_checkout_resolves_every_arm(self):
        arms = sorted((ROOT / "fixtures" / "dynamic_offload_arms").glob("*.json"))
        self.assertGreaterEqual(len(arms), 6)
        for path in arms:
            model = "deepseek_v4" if path.name.startswith("dsv4") else "minimax_m3"
            with self.subTest(arm=path.name), contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(dev.main(["dry-run", "--sglang-src", SRC, "--config", str(path), "--model", model]), 0)
                report = json.loads(out.getvalue())
                self.assertEqual(report["status"], "pass")
                self.assertEqual(report["route_kernel"] is None, not report["resolved_config"]["features"].get(
                    "route_indirect", False))


class LauncherTests(unittest.TestCase):
    BASE = ["--enable-hisparse", "--tp-size", "8", "--hisparse-config", '{"top_k":2048,"device_buffer_size":4096}',
            "--disable-overlap-schedule", "--disable-radix-cache"]

    def resolve(self, arm, *extra):
        from . import launch
        with contextlib.redirect_stderr(io.StringIO()):
            return launch.resolve(["--hisparse-study-config", str(ROOT / "fixtures" / "dynamic_offload_arms" / arm),
                                   "--", *self.BASE, *extra])

    def test_arms_resolve_with_their_execution_rules(self):
        self.assertIsNotNone(self.resolve("minimax_frozen_a.json")[0].study_config_sha256)
        for arm in ("minimax_indirect.json", "dsv4_dynamic.json"):
            with self.subTest(arm=arm):
                with self.assertRaises(SystemExit):           # route_indirect needs eager prefill
                    self.resolve(arm)
                self.resolve(arm, "--disable-prefill-cuda-graph")
                self.resolve(arm, "--cuda-graph-config", '{"decode": {"backend": "full"}, "prefill": '
                                                         '{"backend": "disabled"}}')
                with self.assertRaises(SystemExit):
                    self.resolve(arm, "--disable-prefill-cuda-graph", "--enable-two-batch-overlap")

    def test_worker_installs_the_offload_adapter(self):
        from . import launch
        calls = []
        plan, _, _ = self.resolve("minimax_dynamic.json", "--disable-prefill-cuda-graph")
        with unittest.mock.patch("runtime.dynamic_offload_device.install",
                                 lambda config, digest: calls.append((config.digest, digest))):
            launch.install_for_process(plan)
        self.assertEqual(calls, [(sc.read(plan.study_config_path)[0].digest, plan.study_config_sha256)])


if __name__ == "__main__":
    unittest.main()
