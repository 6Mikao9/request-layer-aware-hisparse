"""CPU tests for the SGLang adapter: fake pinned modules plus CPU torch, no GPU or model."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from runtime import shared_lru as s
from runtime import shared_lru_device as device
from runtime import study_config as sc

try:
    import torch
except ImportError:  # pure-Python environments still run the other suites
    torch = None

SHARED = sc.parse({"schema": sc.SCHEMA, "hot_pool": {"mode": "shared", "layout": "arena"}})
STATIC = sc.parse({"schema": sc.SCHEMA, "hot_pool": {"mode": "isolated", "layout": "arena", "quota": "static",
                                                    "quota_rows": {"0": 4, "1": 12, "2": 8}}})
BYPASS = sc.parse({"schema": sc.SCHEMA, "hot_pool": {"mode": "shared", "layout": "arena",
                                                    "admission": "bypass_list", "bypass_layers": [1]}})
FLOOR = sc.parse({"schema": sc.SCHEMA, "hot_pool": {"mode": "shared", "layout": "arena", "quota": "soft_floor",
                                                   "floor_rows": {"2": 3}}, "diagnostics": {"mode": "counters"}})


def fake_constants():
    sglang = types.ModuleType("sglang")
    srt = types.ModuleType("sglang.srt")
    constants = types.ModuleType("sglang.srt.constants")
    constants.GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
    return {"sglang": sglang, "sglang.srt": srt, "sglang.srt.constants": constants}


def fake_pool(layers=3, plane_rows=64, head_dim=4):
    shapes = ((plane_rows, 1, head_dim), (plane_rows, 1, head_dim))
    return types.SimpleNamespace(
        start_layer=0, layer_num=layers, end_layer=layers, model_layer_ids=tuple(range(layers)), page_size=2,
        head_num=1, head_dim=head_dim, dtype=torch.bfloat16, store_dtype=torch.bfloat16,
        is_quantized_kv_cache=False, post_capture_active=False, kv_cache_layout="nhd", use_hnd=False, device="cpu",
        memory_saver_adapter=types.SimpleNamespace(region=lambda _: contextlib.nullcontext()),
        enable_custom_mem_pool=False, custom_mem_pool=None, _kv_buffer_shapes=lambda: shapes)


def fake_coordinator(pool, *, H=8, slots=4, top_k=4):
    L, i32 = pool.layer_num, torch.int32
    padded = H + pool.page_size
    return types.SimpleNamespace(
        mem_pool_device=pool, is_m3_hisparse=True, is_dsv4_hisparse=False, skip_io=False,
        spec_swap=types.SimpleNamespace(enabled=False), enable_prefetch=False, device_buffer_size=H,
        padded_buffer_size=padded, top_k=top_k, item_size_bytes=pool.head_dim * 2,
        req_device_buffer_tokens=torch.full((L, slots, padded), -1, dtype=i32),
        req_device_buffer_token_locs=torch.full((L, slots, padded), -1, dtype=i32),
        lru_slots=torch.zeros((L, slots, H), dtype=torch.int16),
        top_k_device_locs_buffer=torch.full((slots, H), -1, dtype=i32), num_real_reqs=torch.zeros(1, dtype=i32),
        req_to_host_pool=torch.full((slots, 64), -1, dtype=torch.int64),
        mem_pool_host=types.SimpleNamespace(k_buffer=[torch.zeros((128, 1, pool.head_dim)) for _ in range(L)],
                                            v_buffer=[torch.zeros((128, 1, pool.head_dim)) for _ in range(L)]),
        tp_world_size=1, tp_group=None)


def fake_context(**overrides):
    values = dict(disable_overlap_schedule=True, tbo=False, sbo=False, radix=True, spec=None, disagg="null",
                  pdmux=False, pp=1, decode="full")
    values.update(overrides)
    graph = types.SimpleNamespace(cuda_graph_config=types.SimpleNamespace(
        decode=types.SimpleNamespace(backend=values["decode"])))
    overlap = types.SimpleNamespace(enable_two_batch_overlap=values["tbo"], enable_single_batch_overlap=values["sbo"])
    return types.SimpleNamespace(
        get_schedule=lambda: types.SimpleNamespace(disable_overlap_schedule=values["disable_overlap_schedule"]),
        get_exec=lambda: types.SimpleNamespace(overlap=overlap, graph=graph),
        get_memory=lambda: types.SimpleNamespace(disable_radix_cache=values["radix"]),
        get_spec=lambda: types.SimpleNamespace(speculative_algorithm=values["spec"]),
        get_disagg=lambda: types.SimpleNamespace(disaggregation_mode=values["disagg"], enable_pdmux=values["pdmux"]),
        get_parallel=lambda: types.SimpleNamespace(pp_size=values["pp"]))
@unittest.skipUnless(torch is not None, "CPU torch required")
class PlanesAndTables(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, fake_constants())
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.pool = fake_pool()
        device.contiguous_create_buffers(self.pool)

    def test_contiguous_planes_are_verified_views(self):
        k_all, v_all = device.verify_planes(self.pool)
        self.assertEqual(tuple(k_all.shape), (3, 64, 1, 4))
        self.pool.k_buffer[1][5].fill_(3.0)
        self.assertTrue(torch.equal(k_all[1, 5], torch.full((1, 4), 3.0, dtype=torch.bfloat16)))
        stride = k_all.stride(0) * k_all.element_size()
        self.assertEqual([p.data_ptr() - k_all.data_ptr() for p in self.pool.k_buffer], [i * stride for i in range(3)])
        self.pool.k_buffer[1] = self.pool.k_buffer[1].clone()   # an unverified second base
        with self.assertRaisesRegex(RuntimeError, "plane 1"):
            device.verify_planes(self.pool)
        hnd = fake_pool()
        hnd.kv_cache_layout = "hnd"
        with self.assertRaises(RuntimeError):
            device.contiguous_create_buffers(hnd)
        with self.assertRaisesRegex(RuntimeError, "before install"):
            device.verify_planes(fake_pool())

    def test_runtime_tables_match_kernel_mirror(self):
        for config in (SHARED, STATIC, BYPASS, FLOOR):
            with self.subTest(pool=config.hot_pool):
                coordinator = fake_coordinator(self.pool)
                runtime = device.ArenaRuntime(coordinator, config, "0" * 64, torch=torch)
                mirror = s.KernelMirror(runtime.plan, plane_rows=runtime.plane_rows)
                for allocation in range(2):
                    self.assertEqual(runtime.init_slot(2), s.device_generation(allocation))
                    mirror.init_slot(2, allocation, list(range(10)))
                    state = mirror.slots[2]
                    self.assertEqual(runtime.row_owner[2].tolist(), state["owner"])
                    self.assertEqual(runtime.order[2].tolist(), state["order"])
                    self.assertEqual(runtime.occupancy[2].tolist(), state["occ"])
                    self.assertEqual(int(runtime.generation[2]), state["gen"])
                    runtime.reset_slot(2)
                    self.assertTrue(torch.all(runtime.row_owner[2] == s.FREE_TAG))
                self.assertTrue(torch.all(runtime.row_owner[0] == s.FREE_TAG), "other slots untouched")
                self.assertEqual(tuple(runtime.counters.shape), (3, 8) if config is FLOOR else (0, 0))
                self.assertEqual(tuple(runtime.owner_summary.shape), (0, 0))

    def test_owner_summary_tables_follow_the_mirror_and_the_ledger(self):
        for pool in ({"mode": "shared", "layout": "arena"},
                     {"mode": "isolated", "layout": "arena", "quota": "static", "quota_rows": {"0": 4, "1": 12, "2": 8}},
                     {"mode": "shared", "layout": "arena", "admission": "bypass_list", "bypass_layers": [1]}):
            config = sc.parse({"schema": sc.SCHEMA, "hot_pool": pool, "features": {"owner_summary": True}})
            with self.subTest(pool=pool):
                coordinator = fake_coordinator(self.pool)
                runtime = device.ArenaRuntime(coordinator, config, "0" * 64, torch=torch)
                mirror = s.KernelMirror(runtime.plan, plane_rows=runtime.plane_rows, features=("owner_summary",))
                self.assertEqual(tuple(runtime.owner_summary.shape), (4, 1))   # 24 rows -> one 32-row block
                runtime.init_slot(1)
                mirror.init_slot(1, 0, list(range(10)))
                words = [w & (2**64 - 1) for w in runtime.owner_summary[1].tolist()]
                self.assertEqual(words, mirror.slots[1]["summary"])
                self.assertEqual(words, mirror.live_summary(1))
                self.assertFalse(runtime.owner_summary[0].any(), "other slots untouched")
                runtime.reset_slot(1)
                mirror.reset_slot(1)
                self.assertFalse(runtime.owner_summary[1].any())
                self.assertEqual(mirror.slots[1]["summary"], [0])
                receipt = runtime.receipt(coordinator, 0, 1)
                ledger = s.byte_ledger(layers=runtime.layers, hot_rows=8, page=2, item_bytes=8, slots=4, pool_rows=64,
                                       plan=runtime.plan, features=("owner_summary",))
                self.assertEqual(ledger["arena_metadata_bytes"], sum(receipt["bytes"]["arena_tables"].values()))
                self.assertEqual(receipt["bytes"]["arena_tables"]["owner_summary"], 4 * 8)
        high = sc.parse({"schema": sc.SCHEMA, "hot_pool": {"mode": "shared", "layout": "arena"},
                         "features": {"owner_summary": True}})
        wide = fake_pool(layers=64)                                      # plane 63 is the int64 sign bit
        device.contiguous_create_buffers(wide)
        runtime = device.ArenaRuntime(fake_coordinator(wide), high, "0" * 64, torch=torch)
        self.assertEqual(tuple(runtime.owner_summary.shape), (4, 16))     # 64 planes x 8 rows = 512 rows
        runtime.init_slot(0)
        words = [w & (2**64 - 1) for w in runtime.owner_summary[0].tolist()]
        self.assertEqual(words, s.owner_summary(runtime.plan.owner_template()))
        self.assertEqual(words[-1], (1 << 63) | (1 << 62) | (1 << 61) | (1 << 60))

    def test_consumer_views_accept_only_arena_planes(self):
        runtime = device.ArenaRuntime(fake_coordinator(self.pool), SHARED, "0" * 64, torch=torch)
        flat_k, flat_v = runtime.consumer_views(self.pool.k_buffer[2], self.pool.v_buffer[2])
        self.assertEqual(tuple(flat_k.shape), (3 * 64, 1, 4))
        self.assertEqual(flat_k.data_ptr(), runtime.k_all.data_ptr())
        self.assertEqual(flat_v.untyped_storage().data_ptr(), runtime.v_all.untyped_storage().data_ptr())
        for k, v in ((torch.zeros((64, 1, 4), dtype=torch.bfloat16), self.pool.v_buffer[0]),
                     (self.pool.k_buffer[0], self.pool.v_buffer[0][1:]),
                     (self.pool.k_buffer[0][:, :, :2], self.pool.v_buffer[0])):
            with self.assertRaises(RuntimeError):
                runtime.consumer_views(k, v)
    def test_resolve_assembles_one_graph_safe_launch(self):
        calls = []

        class Module:
            def resolve(self, *args):
                calls.append(args)

        variants = []

        def jit(*variant):
            variants.append(variant)
            return Module()

        coordinator = fake_coordinator(self.pool)
        for config, plane, domain, bypass in ((STATIC, 1, (4, 12), False), (BYPASS, 1, (0, 20), True),
                                              (SHARED, 2, (0, 24), False)):
            with self.subTest(pool=config.hot_pool), patch.object(device, "_jit_module", jit):
                runtime = device.ArenaRuntime(coordinator, config, "0" * 64, torch=torch)
                rpi, seq = torch.tensor([1, 3], dtype=torch.int64), torch.tensor([20, 30], dtype=torch.int32)
                blocks = torch.tensor([[2, -1], [5, 1]], dtype=torch.int32)
                out = runtime.resolve(coordinator, rpi, seq, blocks, plane, 2)
                args = calls[-1]
                self.assertEqual(variants[-1], (device.BLOCK_SIZE, 2, 2, False, False, bypass, False) + (False,) * 5)
                self.assertEqual(len(args), len(device.RESOLVE_TENSORS) + len(device.RESOLVE_INTS))
                self.assertIs(args[0], blocks)
                self.assertIs(args[3], coordinator.num_real_reqs)
                self.assertEqual(args[10].data_ptr(), coordinator.req_device_buffer_token_locs[plane].data_ptr())
                self.assertIs(args[12], coordinator.mem_pool_host.k_buffer[plane])
                self.assertIs(args[14], runtime.k_all)
                self.assertIs(args[19], runtime.owner_summary)
                self.assertEqual(tuple(args[19].shape), (0, 0))
                self.assertEqual([tuple(t.shape) for t in args[20:26]], [(0, 0), (0,), (0, 0), (0, 0), (0, 0), (0, 0)])
                self.assertEqual(args[26:], (plane, 3, 8, 64, *domain, *runtime.plan.scratch, 8, 0, 0,
                                             0, 0, 0, 0, device.INT32_MAX))
                # the kernel requires the recency table to reach the domain end; bypass scratch has no entries
                self.assertEqual(tuple(args[6].shape), (4, sum(size for _, size in runtime.plan.domains)))
                self.assertGreaterEqual(args[6].shape[1], domain[0] + domain[1])
                self.assertEqual((tuple(out.shape), out.data_ptr()),
                                 ((2, 4), coordinator.top_k_device_locs_buffer.data_ptr()))
                with self.assertRaises(ValueError):
                    runtime.resolve(coordinator, rpi, seq, blocks[:, :1], plane, 2)       # width 2 != K
                with self.assertRaises(ValueError):
                    runtime.resolve(coordinator, rpi, seq, blocks.long(), plane, 2)
                with self.assertRaises(ValueError):
                    runtime.resolve(coordinator, rpi, seq, blocks, 3, 2)

    def test_receipt_reports_allocations(self):
        coordinator = fake_coordinator(self.pool)
        runtime = device.ArenaRuntime(coordinator, FLOOR, "a" * 64, torch=torch)
        receipt = json.loads(json.dumps(runtime.receipt(coordinator, 3, 8)))
        self.assertEqual((receipt["rank"], receipt["tp_size"], receipt["config_file_sha256"]), (3, 8, "a" * 64))
        self.assertEqual(receipt["bytes"]["main_kv_k"], 3 * 64 * 4 * 2)
        self.assertEqual(receipt["bytes"]["arena_tables"]["row_owner"], 4 * 24 * 4)
        self.assertEqual(receipt["signature"]["plan_rows"], 24)
        ledger = s.byte_ledger(layers=runtime.layers, hot_rows=8, page=2, item_bytes=8,
                               slots=4, pool_rows=64, plan=runtime.plan, counters=True)
        self.assertEqual(ledger["arena_metadata_bytes"], sum(receipt["bytes"]["arena_tables"].values()))

    def test_index_ring_and_compact_tables_follow_the_mirror_ledger_and_launch(self):
        """Issue #6 switches: templates equal the mirror's, every tensor equals its ledger formula, and each
        plane's launch passes its own ring region and rebuild threshold."""
        from runtime.cuda_shared_lru_correctness import slot_differences
        calls = []

        class Module:
            def resolve(self, *args):
                calls.append(args)

        for base in (SHARED, STATIC, BYPASS, FLOOR):
            for features in (("hit_index",), ("incremental_lru",), ("compact_owner",),
                             ("compact_owner", "hit_index", "incremental_lru")):
                config = sc.parse(dict(base.as_dict(), features=dict.fromkeys(features, True)))
                with self.subTest(pool=base.hot_pool, features=features):
                    coordinator = fake_coordinator(self.pool)
                    coordinator.req_device_buffer_tokens[:, 1, :8] = torch.arange(8, dtype=torch.int32)
                    runtime = device.ArenaRuntime(coordinator, config, "0" * 64, torch=torch)
                    mirror = s.KernelMirror(runtime.plan, plane_rows=runtime.plane_rows, features=features)
                    self.assertEqual(runtime.row_owner.dtype, torch.uint8 if "compact_owner" in features else torch.int32)
                    for allocation in range(2):
                        runtime.init_slot(1)
                        mirror.init_slot(1, allocation, list(range(10)))
                        self.assertEqual(slot_differences(runtime, coordinator.req_device_buffer_tokens, mirror, 1), [])
                        if "hit_index" in features:   # the template copy is deterministic: compare exactly
                            self.assertEqual(runtime.index[1].tolist(), mirror.index_entries(1))
                        runtime.reset_slot(1)
                        mirror.reset_slot(1)
                        self.assertEqual(runtime.row_owner[1].tolist(), mirror.slots[1]["owner"])
                        if "hit_index" in features:
                            self.assertEqual(runtime.index[1].tolist(), mirror.index_entries(1))
                    tables = {k: v for k, v in runtime.receipt(coordinator, 0, 1)["bytes"]["arena_tables"].items() if v}
                    ledger = s.byte_ledger(layers=runtime.layers, hot_rows=8, page=2, item_bytes=8, slots=4,
                                           pool_rows=64, plan=runtime.plan, counters=config.diagnostics == "counters",
                                           features=features)
                    self.assertEqual(tables, ledger["arena_metadata_breakdown"])
                    self.assertEqual(ledger["arena_metadata_bytes"],
                                     4 * ledger["arena_metadata_per_slot_bytes"] + ledger["arena_metadata_fixed_bytes"])
                    with patch.object(device, "_jit_module", lambda *variant: Module()):
                        for plane in range(3):
                            runtime.resolve(coordinator, torch.tensor([1]), torch.tensor([20], dtype=torch.int32),
                                            torch.tensor([[2, 1]], dtype=torch.int32), plane, 2)
                            args = dict(zip(device.RESOLVE_TENSORS + device.RESOLVE_INTS, calls[-1]))
                            for name in ("row_owner", "order", "index", "index_flags", "ring", "ring_dead", "ring_pos",
                                         "ring_meta"):
                                self.assertIs(args[name], getattr(runtime, name))
                            ring = device.ring_launch_args(runtime.plan, plane, "incremental_lru" in features)
                            self.assertEqual(tuple(args[n] for n in device.RESOLVE_INTS[-5:]), ring)
                            if "incremental_lru" in features:
                                first = s.ring_domains(runtime.plan)[s.ring_domain(runtime.plan, plane)][3]
                                self.assertEqual(ring[4] < device.INT32_MAX, plane == first)
                            self.assertEqual((args["index_half"], args["row_bits"]),
                                             s.index_geometry(runtime.plan) if "hit_index" in features else (0, 0))


@unittest.skipUnless(torch is not None, "CPU torch required")
class CompatibilityGate(unittest.TestCase):
    def setUp(self):
        with patch.dict(sys.modules, fake_constants()):
            self.pool = fake_pool()
            device.contiguous_create_buffers(self.pool)

    def check(self, coordinator=None, capability=(8, 0), **context):
        coordinator = coordinator or fake_coordinator(self.pool)
        with patch.object(torch.cuda, "get_device_capability", lambda *a: capability):
            return device.check_coordinator(coordinator, torch=torch, context=fake_context(**context))

    def test_accepts_pinned_single_stream_minimax(self):
        self.assertEqual(self.check(), (0, 1, 2))
        self.assertEqual(self.check(decode="disabled"), (0, 1, 2))

    def test_rejects_every_unsupported_mode_before_allocation(self):
        cases = [dict(disable_overlap_schedule=False), dict(tbo=True), dict(sbo=True), dict(radix=False),
                 dict(spec="EAGLE"), dict(disagg="decode"), dict(pdmux=True), dict(pp=2), dict(decode="breakable"),
                 dict(capability=(8, 6))]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(RuntimeError):
                self.check(**case)
        for mutate in (lambda c: setattr(c, "is_m3_hisparse", False), lambda c: setattr(c, "skip_io", True),
                       lambda c: setattr(c.spec_swap, "enabled", True), lambda c: setattr(c, "enable_prefetch", True),
                       lambda c: setattr(c.mem_pool_device, "head_num", 2),
                       lambda c: setattr(c.mem_pool_device, "dtype", torch.float16),
                       lambda c: setattr(c.mem_pool_device, "model_layer_ids", (0, 2, 3)),
                       lambda c: setattr(c, "padded_buffer_size", 11), lambda c: setattr(c, "top_k", 9)):
            coordinator = fake_coordinator(fake_pool())
            mutate(coordinator)
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                self.check(coordinator)
def fake_sglang(tmp):
    """Pinned symbols as plain classes; the hook only relies on names and call contracts."""
    modules = fake_constants()
    modules["sglang"].__file__ = str(Path(tmp) / "__init__.py")
    log = dict(consumer=[], alloc=[], finished=[])

    class HiSparseCoordinator:
        def __init__(self, pool, **_):
            vars(self).update(vars(fake_coordinator(pool)))

        def alloc_device_buffer(self, req):
            log["alloc"].append(req.kv.req_pool_idx)
            self.req_device_buffer_tokens[:, req.kv.req_pool_idx, :self.device_buffer_size] = torch.arange(8)

        def request_finished(self, req):
            log["finished"].append(req.kv.req_pool_idx)

        def admit_request_direct(self, req):
            return "stock direct"

        def swap_in_selected_blocks(self, *args):
            return "stock swap"

    class HiSparseMHAMainPool:
        pass

    def flash_decode_with_gqa_share_sparse(**kwargs):
        log["consumer"].append((kwargs["k_cache"], kwargs["v_cache"]))
        return "attention"

    flash_decode_with_gqa_share_sparse.__module__ = "sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse"
    for name in ("sglang.srt.managers.hisparse_coordinator", "sglang.srt.mem_cache.hisparse_memory_pool",
                 device.CONSUMER_MODULE, "sglang.srt.runtime_context"):
        modules[name] = types.ModuleType(name)
    modules["sglang.srt.managers.hisparse_coordinator"].HiSparseCoordinator = HiSparseCoordinator
    modules["sglang.srt.mem_cache.hisparse_memory_pool"].HiSparseMHAMainPool = HiSparseMHAMainPool
    setattr(modules[device.CONSUMER_MODULE], device.CONSUMER, flash_decode_with_gqa_share_sparse)
    vars(modules["sglang.srt.runtime_context"]).update(vars(fake_context()))
    return modules, HiSparseCoordinator, HiSparseMHAMainPool, log


@unittest.skipUnless(torch is not None, "CPU torch required")
class InstallHooks(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.modules, self.Coordinator, self.Pool, self.log = fake_sglang(tmp.name)
        for context in (patch.dict(sys.modules, self.modules), patch.object(device, "_STATE", None),
                        patch.object(device, "_CREATED_POOLS", 0),
                        patch.object(device, "verify_digests", lambda root: {}),
                        patch.object(torch.cuda, "get_device_capability", lambda *a: (8, 0)),
                        patch.object(torch.distributed, "get_rank", lambda *a: 0, create=True),
                        patch.object(torch.distributed, "all_gather_object",
                                     lambda out, value, group=None: out.__setitem__(slice(None), [value] * len(out)),
                                     create=True)):
            context.start()
            self.addCleanup(context.stop)
        self.originals = {name: getattr(self.Coordinator, name) for name in device._PATCHED}

    def test_stock_config_installs_nothing_and_unsupported_fails_closed(self):
        self.assertFalse(device.install(sc.StudyConfig(), "0" * 64))
        self.assertEqual({n: getattr(self.Coordinator, n) for n in device._PATCHED}, self.originals)
        self.assertNotIn("_create_buffers_normal", vars(self.Pool))
        counters = sc.parse({"schema": sc.SCHEMA, "diagnostics": {"mode": "counters"}})
        with self.assertRaises(RuntimeError):
            device.install(counters, "0" * 64)
        self.assertFalse(device.uninstall())
    def test_bound_worker_routes_swap_consumer_and_lifecycle(self):
        self.assertTrue(device.install(BYPASS, "b" * 64))
        self.assertTrue(device.install(BYPASS, "b" * 64))          # idempotent for the same frozen config
        with self.assertRaises(RuntimeError):
            device.install(SHARED, "c" * 64)
        pool = fake_pool()
        self.Pool._create_buffers_normal(pool)                    # what MHATokenToKVPool.__init__ calls
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            coordinator = self.Coordinator(pool)
        receipt = json.loads(stdout.getvalue())
        self.assertEqual((receipt["event"], receipt["config_sha256"]), ("hisparse_shared_lru_bound", BYPASS.digest))
        runtime = coordinator._request_layer_hisparse_arena
        req = types.SimpleNamespace(kv=types.SimpleNamespace(req_pool_idx=1))
        coordinator.alloc_device_buffer(req)
        self.assertEqual(self.log["alloc"], [1])
        self.assertEqual(int(runtime.generation[1]), 1)
        with patch.object(runtime, "resolve", lambda *args: ("arena", args[1:])):
            routed = coordinator.swap_in_selected_blocks("rpi", "seq", "blocks", 2, 128)
        self.assertEqual(routed, ("arena", ("rpi", "seq", "blocks", 2, 128)))
        consumer = getattr(sys.modules[device.CONSUMER_MODULE], device.CONSUMER)
        self.assertEqual(consumer(k_cache=pool.k_buffer[1], v_cache=pool.v_buffer[1], hisparse_slots="slots"),
                         "attention")
        self.assertIs(self.log["consumer"][-1][0], runtime.flat_k)
        consumer(k_cache=pool.k_buffer[1], v_cache=pool.v_buffer[1], hisparse_slots=None)
        self.assertIs(self.log["consumer"][-1][0], pool.k_buffer[1])   # dense/other paths unchanged
        with self.assertRaises(RuntimeError):
            consumer(k_cache=torch.zeros_like(pool.k_buffer[1]), v_cache=pool.v_buffer[1], hisparse_slots="s")
        with self.assertRaises(RuntimeError):
            coordinator.admit_request_direct(req)
        coordinator.request_finished(req)
        self.assertEqual(self.log["finished"], [1])
        self.assertTrue(torch.all(runtime.row_owner[1] == s.FREE_TAG))
        with self.assertRaises(RuntimeError):                     # one coordinator per worker
            with contextlib.redirect_stdout(io.StringIO()):
                self.Coordinator(pool)
        with self.assertRaisesRegex(RuntimeError, "new process"):
            device.uninstall()

    def test_uninstall_before_any_pool_restores_pinned_symbols(self):
        self.assertTrue(device.install(SHARED, "0" * 64))
        self.assertIsNot(self.Coordinator.swap_in_selected_blocks, self.originals["swap_in_selected_blocks"])
        self.assertTrue(device.uninstall())
        self.assertEqual({n: getattr(self.Coordinator, n) for n in device._PATCHED}, self.originals)
        self.assertNotIn("_create_buffers_normal", vars(self.Pool))

    def test_stacked_hooks_are_refused(self):
        self.Coordinator.swap_in_selected_blocks = lambda *a: "other hook"
        self.Coordinator.swap_in_selected_blocks.__module__ = "elsewhere"
        with self.assertRaisesRegex(RuntimeError, "refusing"):
            device.install(SHARED, "0" * 64)
SYNTHETIC_TREE = {
    device.COORDINATOR: """
class HiSparseCoordinator:
    def __init__(self, req_to_token_pool, allocator): pass
    def alloc_device_buffer(self, req): pass
    def admit_request_direct(self, req): pass
    def admit_request_into_staging(self, req): pass
    def request_finished(self, req): pass
    def map_last_loc_to_buffer(self, seq_lens, out_cache_loc): pass
    def _eager_backup_previous_token(self, seq_lens): pass
    def wait_for_pending_backup(self): pass
    def swap_in_selected_blocks(self, req_pool_indices, seq_lens, top_k_blocks, layer_id, sparse_block_size): pass
""",
    "srt/mem_cache/hisparse_memory_pool.py": "class HiSparseMHAMainPool:\n    def __init__(self, size): pass\n",
    "srt/mem_cache/memory_pool.py": ("class MHATokenToKVPool:\n    def _create_buffers_normal(self): pass\n"
                                     "    def _kv_buffer_shapes(self): pass\n"),
    "srt/layers/attention/minimax_sparse_backend.py": (
        "class MiniMaxSparseAttnBackend:\n"
        "    def _hisparse_swap_in_blocks(self, forward_batch, topk_idx, layer_id): pass\n"),
    "srt/layers/attention/minimax_sparse_ops/minimax_sparse.py": (
        "from x import flash_decode_with_gqa_share_sparse\n"
        "def minimax_sparse_decode(q):\n"
        "    return flash_decode_with_gqa_share_sparse(q=q, k_cache=1, v_cache=2, hisparse_slots=None)\n"),
    "kernels/ops/attention/minimax_sparse/decode/topk_sparse.py": (
        "def flash_decode_with_gqa_share_sparse(q, sink, k_cache, v_cache, hisparse_slots=None): pass\n"),
    "kernels/ops/kvcache/hisparse.py": "def load_blocks_to_device_buffer_mha(top_k_blocks): pass\n",
}


class DryRun(unittest.TestCase):
    def write_tree(self, tmp, **overrides):
        for rel, text in dict(SYNTHETIC_TREE, **overrides).items():
            path = Path(tmp) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def test_symbol_resolution_flags_signature_and_consumer_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_tree(tmp)
            self.assertTrue(all(r["status"] == "ok" for r in device.resolve_symbols(tmp)))
            drift = SYNTHETIC_TREE[device.COORDINATOR].replace("layer_id, sparse", "sparse, layer_id")
            positional = SYNTHETIC_TREE["srt/layers/attention/minimax_sparse_ops/minimax_sparse.py"].replace(
                "q=q, k_cache=1", "q, 1")
            self.write_tree(tmp, **{device.COORDINATOR: drift,
                                    "srt/layers/attention/minimax_sparse_ops/minimax_sparse.py": positional})
            failed = {r["symbol"] for r in device.resolve_symbols(tmp) if r["status"] != "ok"}
            self.assertEqual(failed, {"HiSparseCoordinator.swap_in_selected_blocks",
                                      f"{device.CONSUMER} call sites"})
            status = device.verify_digests(tmp)
            self.assertEqual(status[device.COORDINATOR], "mismatch")
            self.assertEqual(status["srt/model_executor/model_runner.py"], "missing")

    def test_sm80_geometry_rejects_unsupported_plans(self):
        geometry = dict(tp_size=8, hot_rows=4096, page=128, block_size=128, head_dim=128,
                        dtype_bytes=2, kv_heads_per_rank=1, slots=128, plane_rows=396_672)
        for planes, blocks, error in ((65, 16, "planes"), (60, 12, "selection width"),
                                       (60, 32, "selection width")):
            geo = device.describe_geometry(layers=range(planes), top_k_blocks=blocks, **geometry)
            with self.subTest(planes=planes, blocks=blocks), self.assertRaisesRegex(ValueError, error):
                device.dry_run(Path("."), SHARED, geo)

    def test_geometry_contracts_and_budget_units(self):
        for tp in (4, 8):
            geo = device.describe_geometry(tp_size=tp, layers=range(60), hot_rows=4096, page=128, top_k_blocks=16,
                                           block_size=128, head_dim=128, dtype_bytes=2, kv_heads_per_rank=1,
                                           slots=128, plane_rows=396_672)
            self.assertEqual((geo["selected"], geo["kv_row_bytes"]), (2048, 512))
        for bad in (dict(tp_size=2), dict(kv_heads_per_rank=2)):
            with self.assertRaises(ValueError):
                device.describe_geometry(**dict(dict(tp_size=8, layers=range(60), hot_rows=4096, page=128,
                                                     top_k_blocks=16, block_size=128, head_dim=128, dtype_bytes=2,
                                                     kv_heads_per_rank=1, slots=128, plane_rows=396_672), **bad))
        with tempfile.TemporaryDirectory() as tmp:
            self.write_tree(tmp)
            report = device.dry_run(tmp, BYPASS, geo)
        self.assertEqual(report["status"], "fail")                 # synthetic tree: digests cannot match
        per_request = report["budget_units"]["per_request"]
        self.assertEqual(per_request["history_payload_bytes"], 120 * 2**20)   # 60 x 4096 x 512 B
        self.assertEqual(per_request["tail_payload_bytes"], int(3.75 * 2**20))
        self.assertEqual(per_request["bypass_scratch_rows"], 2048)
        self.assertEqual(report["implementation"]["template_args"],
                         [[1024, 16, 128, False, False, False, False] + [False] * 5,
                          [1024, 16, 128, False, False, True, False] + [False] * 5])
        self.assertEqual(report["implementation"]["max_shared_memory_bytes_per_cta"], 100_904)
        self.assertIn("plane * 396672 + loc", report["tensor_contracts"]["output"])
        launch = report["launch"]
        self.assertEqual((launch["block"], launch["dynamic_shared_memory_bytes"]), ([1024, 1, 1], 100_904))
        self.assertEqual(len(launch["planes"]), 60)
        self.assertEqual([p["template_args"][5] for p in launch["planes"]], [False] + [True] + [False] * 58)
        self.assertEqual(launch["planes"][7]["int_args"]["order_size"], 245_760 - 2048)

    def test_resolver_switches_change_variants_ledger_and_shared_memory(self):
        geo = device.describe_geometry(tp_size=8, layers=range(60), hot_rows=4096, page=128, top_k_blocks=16,
                                       block_size=128, head_dim=128, dtype_bytes=2, kv_heads_per_rank=1,
                                       slots=128, plane_rows=396_672)
        both = sc.parse({"schema": sc.SCHEMA, "hot_pool": {"mode": "shared", "layout": "arena"},
                         "features": {"owner_summary": True, "touched_bitmap": True}})
        indexed = sc.parse({"schema": sc.SCHEMA, "hot_pool": {"mode": "shared", "layout": "arena"},
                            "features": {"hit_index": True, "incremental_lru": True, "compact_owner": True}})
        with tempfile.TemporaryDirectory() as tmp:
            self.write_tree(tmp)
            plain, report = device.dry_run(tmp, SHARED, geo), device.dry_run(tmp, both, geo)
            issue6 = device.dry_run(tmp, indexed, geo)
        self.assertEqual(report["implementation"]["template_args"],
                         [[1024, 16, 128, False, False, False, False, True, True, False, False, False]])
        # 7,680 owner-summary words per slot plus the template; the bitmap uses shared memory only
        extra = report["budget_units"]["arena_metadata_bytes"] - plain["budget_units"]["arena_metadata_bytes"]
        self.assertEqual(extra, 129 * 7680 * 8)
        self.assertEqual(report["implementation"]["max_shared_memory_bytes_per_cta"], 100_912 + 7680 * 4)
        self.assertEqual(report["capabilities"]["features"], {"owner_summary": True, "touched_bitmap": True,
                                                              "hit_index": False, "incremental_lru": False,
                                                              "compact_owner": False})
        # issue #6: index 2 x 15,360 buckets x 16 entries, ring D + 61 K, positions, one byte per owner tag
        self.assertEqual(issue6["implementation"]["template_args"],
                         [[1024, 16, 128, False, False, False, False, False, False, True, True, True]])
        self.assertEqual(issue6["implementation"]["max_shared_memory_bytes_per_cta"], 100_904)
        breakdown = issue6["budget_units"]["arena_metadata_breakdown"]
        self.assertEqual((breakdown["index"], breakdown["ring"], breakdown["ring_pos"], breakdown["row_owner"]),
                         (128 * 491_520 * 4, 128 * 370_688 * 4, 128 * 245_760 * 4, 128 * 245_760))
        planes = issue6["launch"]["planes"]
        self.assertEqual(planes[0]["int_args"]["rebuild_tail"], 245_760 + 2048)
        self.assertTrue(all(p["int_args"]["rebuild_tail"] == device.INT32_MAX for p in planes[1:]))
        self.assertEqual({(p["int_args"]["index_half"], p["int_args"]["row_bits"]) for p in planes}, {(15_360, 18)})
        self.assertIn("uint8 plane", issue6["tensor_contracts"]["row_owner"])
        bitmap = sc.parse({"schema": sc.SCHEMA, "hot_pool": {"mode": "shared", "layout": "arena"},
                           "features": {"touched_bitmap": True}})
        wide = s.plan_arena(bitmap, tuple(range(64)), 16384, 2048)   # 1,048,576-row shared domain
        with self.assertRaisesRegex(ValueError, "shared memory"):
            device.check_shared_memory(bitmap, wide)
        self.assertEqual(device.check_shared_memory(SHARED, wide), 100_904)
        with self.assertRaisesRegex(ValueError, "fingerprint"):         # > 2**23 rows leave too few fingerprint bits
            s.index_geometry(s.plan_arena(indexed, tuple(range(64)), (1 << 17) + 1, 2048))
        self.assertEqual(s.index_geometry(s.plan_arena(indexed, tuple(range(64)), 1 << 17, 2048))[1], 23)

    @unittest.skipUnless("SGLANG_SRC" in __import__("os").environ, "set SGLANG_SRC to a pinned checkout")
    def test_pinned_checkout_resolves(self):
        root = Path(__import__("os").environ["SGLANG_SRC"])
        args = ["dry-run", "--sglang-src", str(root), "--config",
                str(Path(__file__).parent / "fixtures" / "shared_lru_arms" / "shared_arena.json")]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(device.main(args), 0)
        self.assertEqual(json.loads(out.getvalue())["status"], "pass")


if __name__ == "__main__":
    unittest.main()
