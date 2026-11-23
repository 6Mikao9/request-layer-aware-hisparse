"""CPU dataflow tests over a frozen real Hybrid method; not GPU acceptance."""
import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch
from abi_probe import make_bf16_method, make_indexer_adapters

ROOT = Path(__file__).resolve().parent
VLLM_SRC = os.environ.get("VLLM_SRC")
BACKENDS = (Path(VLLM_SRC) if VLLM_SRC else ROOT / "hybrid") / "vllm/v1/attention/backends/mla"
SOURCE = BACKENDS / "flashmla_sparse.py"


def method(name):
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "7ecdfef953fdb2d0867b6e958c2b2d287d70f5b390ce18156dd54cdd84485a38"
    cls = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef)
               and n.name == "FlashMLASparseImpl")
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(93)
        self.calls = []
        self.kv = torch.randn(12, 1, 576).bfloat16()
        self.q = torch.randn(2, 4, 576).bfloat16()
        self.indices = torch.full((2, 2048), -1, dtype=torch.int32)
        self.indices[:, 0] = 7
        self.indices[:, 3] = 2  # interior padding, NOT count-sized valid prefix

    def kernel(self, q, kv, indices, scale):
        self.calls.append((q.clone(), kv, indices.clone()))
        result = []
        for i in range(q.shape[0]):
            selected = indices[i, 0]
            selected = selected[(selected >= 0) & (selected < len(kv))].long()
            rows = kv[selected, 0].float()
            result.append(((q[i].float() @ rows.T * scale).softmax(-1)
                           @ rows[:, :512]).bfloat16())
        return torch.stack(result)

    def test_physical_rows_holes_and_padded_heads(self):
        owner = NS(softmax_scale=0.02, need_to_return_lse_for_decode=False)
        out, lse = make_bf16_method(self.kernel, cpu_fixture=True)(
            owner, self.q, self.kv, self.indices, torch.tensor([2, 2], dtype=torch.int32), 2)
        self.assertEqual(out.shape, (2, 2, 512))
        self.assertIsNone(lse)
        self.assertEqual(self.calls[0][2][0, 0, 3], 2)
        expected = self.kernel(self.q[:, :2], self.kv,
                               self.indices[:, None], 0.02)
        torch.testing.assert_close(out, expected)

    def test_real_hybrid_decode_then_prefill_dataflow(self):
        if VLLM_SRC is None and not SOURCE.is_file():
            self.skipTest("Set VLLM_SRC to the pinned Hybrid source to check the original call graph")
        events = []
        kv = self.kv
        indices = self.indices
        class Group:
            def cache(self, i): return object()
            def convert_decode_logical_to_physical_topk(self, i, logical, meta, **kw):
                events.append("resolve_gather")
                return indices[:1], torch.tensor([2], dtype=torch.int32)
            def physical_kv_cache(self, i): return kv
            def stage_prefill_rows(self, i, original, meta):
                events.append("stage_prefill")
                return kv.view(3, 4, 576), meta.block_table, meta.req_id_per_token[1:]
        def convert(*args, **kw):
            events.append("prefill_physical_map")
            return indices[1:], torch.tensor([2], dtype=torch.int32)
        flat_source = (BACKENDS / "sparse_utils.py").read_bytes()
        self.assertEqual(hashlib.sha256(flat_source).hexdigest(),
                         "49001615238477d5d8c314b2193b771d82204ca0b7931cedf8bbfb1506a33415")
        flat_node = next(n for n in ast.parse(flat_source).body
                         if isinstance(n, ast.FunctionDef) and n.name == "flat_kv_row_view")
        ns = dict(torch=torch, HiSparseMLAIndexGroup=Group,
                  triton_convert_req_index_to_global_index=convert)
        nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                 flat_node, method("_forward_bf16_kv")]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     str(SOURCE), "exec"), ns)
        def consumer(*args):
            events.append("attention")
            return self.kernel(*args)
        owner = NS(index_group=Group(), index_group_index=0, softmax_scale=0.02)
        owner._bf16_flash_mla_kernel = make_bf16_method(consumer, cpu_fixture=True).__get__(owner)
        meta = NS(block_table=torch.tensor([[1]], dtype=torch.int32),
                  req_id_per_token=torch.tensor([0, 0], dtype=torch.int32),
                  num_decode_tokens=1, num_actual_tokens=2, block_size=4)
        out, lse = ns["_forward_bf16_kv"](owner, self.q, self.kv, self.indices, meta, 2)
        self.assertEqual(events, ["resolve_gather", "attention", "stage_prefill",
                                  "prefill_physical_map", "attention"])
        self.assertEqual(out.shape, (2, 2, 512))
        self.assertIsNone(lse)

    def test_reject_lse_fp8_and_inconsistent_counts(self):
        fn = make_bf16_method(self.kernel, cpu_fixture=True)
        with self.assertRaisesRegex(ValueError, "LSE"):
            fn(NS(need_to_return_lse_for_decode=True), self.q, self.kv, self.indices)
        with self.assertRaisesRegex(ValueError, "BF16"):
            fn(NS(), self.q.float(), self.kv, self.indices)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            fn(NS(), self.q, self.kv, self.indices, torch.tensor([1, 1], dtype=torch.int32))

    def test_default_cannot_be_mistaken_for_cpu_backend(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            make_bf16_method(self.kernel)(NS(), self.q, self.kv, self.indices)

    def test_tuple_indexer_and_2d_lens_adapter(self):
        calls = []
        def kernel(*args, **kwargs):
            calls.append((args, kwargs)); return args[0]
        prefill, decode = make_indexer_adapters(kernel, kernel)
        q = torch.zeros(2, 1, 4, 128, dtype=torch.float8_e4m3fn)
        kv = torch.zeros(3, 64, 1, 132, dtype=torch.uint8)
        lens = torch.tensor([[61], [117]], dtype=torch.int32)
        result = decode((q, None), kv, None, lens, None, None, max_model_len=128)
        self.assertIs(result, q)
        self.assertEqual(calls[-1][0][3].tolist(), [61, 117])
        with self.assertRaisesRegex(ValueError, "FP4"):
            prefill((q, torch.ones(1)), kv, None, None, None)
        with self.assertRaisesRegex(ValueError, "speculative"):
            decode((q, None), kv, None, lens, None, None,
                   max_model_len=128, indices=torch.zeros(1, dtype=torch.int32))


if __name__ == "__main__":
    unittest.main()
