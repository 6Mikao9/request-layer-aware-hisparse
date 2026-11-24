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
                self.assertEqual(variants[-1], (device.BLOCK_SIZE, 2, 2, False, False, bypass, False))
                self.assertIs(args[0], blocks)
                self.assertIs(args[3], coordinator.num_real_reqs)
                self.assertEqual(args[10].data_ptr(), coordinator.req_device_buffer_token_locs[plane].data_ptr())
                self.assertIs(args[12], coordinator.mem_pool_host.k_buffer[plane])
                self.assertIs(args[14], runtime.k_all)
                self.assertEqual(args[19:], (plane, 3, 8, 64, *domain, *runtime.plan.scratch, 8))
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
                         [[1024, 16, 128, False, False, False, False], [1024, 16, 128, False, False, True, False]])
        self.assertIn("plane * 396672 + loc", report["tensor_contracts"]["output"])

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
