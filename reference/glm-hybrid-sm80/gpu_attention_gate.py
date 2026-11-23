"""No-model SM80 Hybrid correctness gate; only runs CUDA on explicit invocation.

Uses real HiSparseRuntime/CacheHandle, prefill staging and inherited BF16 route.
Not a full model/startup acceptance or a throughput benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path


def run(output, batches):
    import math
    import torch
    import vllm
    from types import SimpleNamespace as NS
    from vllm.v1.attention.backends.mla.hybrid_triton_sm80 import HybridTritonSM80Impl
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
    from vllm.v1.hisparse.runtime import (
        HiSparseRuntime, HiSparseCacheHandle, ResolvedHiSparseConfig,
        build_hisparse_prefill_staging_plan,
    )
    if output.exists(): raise ValueError("Use a new output receipt")
    here = Path(__file__).resolve().parent
    package_root = Path(vllm.__file__).resolve().parent.parent
    pins = json.loads((here / "attention-source-pins.json").read_text(encoding="utf-8"))
    checked = {}
    # Registry is the deliberately changed file; the complete original runtime,
    # common builder, mapping and model-routing sources must still be pinned.
    for rel, expected in pins.items():
        if rel.endswith("backends/registry.py"): continue
        digest = hashlib.sha256((package_root / rel).read_bytes()).hexdigest()
        if digest != expected: raise ValueError(f"Installed source mismatch: {rel}")
        checked[rel] = digest
    if torch.cuda.get_device_capability() != (8, 0) or "A100" not in torch.cuda.get_device_name():
        raise ValueError("This gate requires actual A100 SM80")
    assert AttentionBackendEnum.HYBRID_TRITON_SM80.get_class().get_impl_cls() is HybridTritonSM80Impl
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(32021)
    records = []
    K, H, L, P, W = 2048, 4096, 8192, 64, 576

    def reference(q, rows, logical, reqs):
        result = []
        for i in range(len(q)):
            idx = logical[i]
            idx = idx[idx >= 0].long()
            selected = rows[int(reqs[i]), idx].to(device).float()
            logits = q[i].float() @ selected.T / math.sqrt(W)
            result.append((logits.softmax(-1) @ selected[:, :512]).bfloat16())
        return torch.stack(result)

    for b in batches:
        for mode in ("host_hot", "resident", "mixed_pages"):
            runtime = HiSparseRuntime(
                ResolvedHiSparseConfig(top_k=K, device_buffer_size=H, host_pool_gib=1),
                max_num_reqs=b, row_width=W, kv_dtype=torch.bfloat16, device=device)
            hot_blocks = b * (H // P)
            total_blocks = hot_blocks + b * (L // P)
            raw = torch.zeros(total_blocks * P * W * 2, dtype=torch.uint8, device=device)
            hot_table = torch.arange(hot_blocks, dtype=torch.int32, device=device).view(b, -1)
            runtime.bind_hot_cache(raw, byte_offset=0, block_stride=P*W*2,
                num_blocks=total_blocks, block_size=P, block_table=hot_table)
            runtime.request_state_indices = torch.arange(b, dtype=torch.int32, device=device)
            host = torch.randn(b * (L // P), P, W).bfloat16().pin_memory()
            runtime.bind_source_cache(host)
            host_rows = host.view(b, L, W)
            cache = HiSparseCacheHandle(runtime)
            resident_table = torch.arange(hot_blocks, total_blocks, dtype=torch.int32, device=device).view(b, -1)
            runtime.hot.cache[hot_blocks:].copy_(host.to(device))
            if mode == "host_hot": resident_table.fill_(-1)
            if mode == "mixed_pages": resident_table[:, ::2] = -1
            cache.bind_cache(raw, byte_offset=0, block_stride=P*W*2, num_blocks=total_blocks,
                block_size=P, block_table=resident_table,
                slot_mapping=torch.full((b,), -1, dtype=torch.int64, device=device))
            cache.all_context_pages_resident = mode == "resident"
            source_table = torch.arange(b * (L // P), dtype=torch.int32, device=device).view(b, -1)
            cache.source_block_table = source_table
            group = object.__new__(HiSparseMLAIndexGroup)
            group.caches = [cache]
            group.request_ids = torch.arange(b, dtype=torch.int32, device=device)
            group.physical_topk_indices = torch.empty(b, K, dtype=torch.int32, device=device)
            # Constructor/model initialization is a separate full-model gate.
            impl = object.__new__(HybridTritonSM80Impl)
            impl.index_group = group
            impl.index_group_index = 0
            impl.softmax_scale = 1 / math.sqrt(W)
            impl.need_to_return_lse_for_decode = False
            reqs = torch.arange(b, dtype=torch.int32, device=device)
            q = torch.randn(b, 8, W, device=device).bfloat16() * 0.25
            meta = NS(block_table=source_table, req_id_per_token=reqs,
                num_decode_tokens=b, num_actual_tokens=b, num_decodes=b,
                decode_max_query_len=1, block_size=P)
            for step in range(4):
                logical = torch.stack([(torch.arange(K, device=device, dtype=torch.int32)
                           + (step % 3) * K + r * 7) % L for r in range(b)])
                logical[:, 11] = -1
                logical[:, -1] = -1
                runtime.begin_forward()
                actual, _ = impl._forward_bf16_kv(q, host, logical, meta, 8)
                expected = reference(q, host_rows, logical, reqs)
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)
                records.append(dict(batch=b, mode=mode, stage="decode", step=step,
                    max_abs_error=float((actual.float()-expected.float()).abs().max()),
                    finite=bool(torch.isfinite(actual).all())))
            # Two causal prefill query rows per request. Short valid set + holes
            # is intentional: verifies negative padding and per-query mask.
            reqs = torch.arange(b, dtype=torch.int32, device=device).repeat_interleave(2)
            logical = torch.full((2*b, K), -1, dtype=torch.int32, device=device)
            for i in range(2*b):
                length = 31 + i % 2
                logical[i, :length] = torch.arange(length, device=device, dtype=torch.int32)
                logical[i, 7] = -1
            q = torch.randn(2*b, 8, W, device=device).bfloat16() * 0.25
            plan = build_hisparse_prefill_staging_plan(source_table,
                torch.full((b,), L, dtype=torch.int32, device=device), P, b*(L//P))
            meta = NS(block_table=source_table, req_id_per_token=reqs,
                num_decode_tokens=0, num_actual_tokens=2*b, num_decodes=0,
                decode_max_query_len=0, block_size=P,
                prefill=NS(host_staging_plan=plan))
            runtime.begin_forward()
            actual, _ = impl._forward_bf16_kv(q, host, logical, meta, 8)
            expected = reference(q, host_rows, logical, reqs)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)
            records.append(dict(batch=b, mode=mode, stage="prefill",
                max_abs_error=float((actual.float()-expected.float()).abs().max()),
                finite=bool(torch.isfinite(actual).all())))
    receipt = dict(schema="sm80-hybrid-attention-gate-v1", status="passed",
        scope="real runtime resolve/gather/stage plus Triton consumer; no model load",
        source_hashes=checked,
        device=torch.cuda.get_device_name(), torch=torch.__version__,
        context=L, topk=K, hot_rows=H, width=W, batches=batches, records=records,
        model_verified=False, performance_claim=False)
    with output.open("x", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    a = p.parse_args()
    if any(b not in (1, 2, 4, 8) for b in a.batches): p.error("B must be 1/2/4/8")
    run(a.output, a.batches)
