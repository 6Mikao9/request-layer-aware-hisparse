"""Fused decode route of elastic per-layer HiSparse (issue #13): the CUDA kernels of ``elastic_route.cuh``, built by
the pinned SGLang JIT, for ``ElasticKV.resident_page_table``, ``smooth_select``, ``merge``, ``store_rows`` and
``gate``.

The torch statements in ``elastic_kv`` are the reference; each is a handful of small graph nodes per managed layer
and decode step. Each kernel computes the same output in one launch, reading the layer's route row and stale limits
from the device, so one captured graph serves every offload level. ``self_check`` compares both on random inputs;
a worker binds the kernels only after it passes.
"""
import functools
from pathlib import Path

SOURCE = Path(__file__).with_name("elastic_route.cuh")
JIT_NAME = "rlh_elastic_route"
ENTRIES = ("resident", "select", "merge", "store_rows", "gate")
RESIDENT = 1


@functools.cache
def _module():
    from sglang.kernels.jit.utils import load_jit
    return load_jit(JIT_NAME, cuda_files=[str(SOURCE)],
                    cuda_wrappers=[(name, f"rlh_elastic_route::{name}") for name in ENTRIES])


def resident(kv, req_pool_indices, top_k, seq_lens, offset=0):
    """``ElasticKV.resident_page_table`` with ``seq_lens``."""
    table = kv.torch.empty(top_k.shape, dtype=kv.torch.int32, device=top_k.device)
    _module().resident(top_k, req_pool_indices, seq_lens, kv.c.req_to_token_pool.req_to_token, table, kv.ratio,
                       int(offset))
    return table


def select(kv, i, req_pool_indices, seq_lens, top_k, block=None):
    """``ElasticKV.smooth_select``; ``reads`` is int8 (nonzero = read the combined row)."""
    t = kv.torch
    n, k = top_k.shape
    width = k if block is None else k * block
    masked = t.empty((n, k), dtype=top_k.dtype, device=top_k.device)
    rows = t.empty((n, width), dtype=t.int64, device=top_k.device)
    reads = t.empty((n, width), dtype=t.int8, device=top_k.device)
    _module().select(top_k, req_pool_indices, seq_lens, kv.c.req_to_token_pool.req_to_token, kv.stale_limit[i],
                     kv.route[i], masked, rows, reads, kv.ratio, block or 0)
    return masked, rows, reads


def merge(torch, swapped, rows, reads):
    out = torch.empty(rows.shape, dtype=swapped.dtype, device=swapped.device)
    _module().merge(swapped, rows, reads, out)
    return out


def store_rows(kv, i, loc):
    """``ElasticKV.store_rows`` (int64 rows of the shape of ``loc``)."""
    flat = loc.reshape(-1)
    out = kv.torch.empty(flat.shape, dtype=kv.torch.int64, device=loc.device)
    _module().store_rows(flat, kv.mapping, kv.route[i], out)
    return out.view(loc.shape)


def gate(kv, i, num_real_reqs):
    """``ElasticKV.gate``."""
    out = kv.torch.empty(1, dtype=kv.torch.int32, device=num_real_reqs.device)
    _module().gate(num_real_reqs, kv.route[i], out)
    return out


def self_check(torch, device, *, block=None, ratio=1, seed=0):
    """Kernels and torch statements agree on random tables in both modes (resident, and offloaded with stale
    rows): selection, merge, store rows, the planned-copy gate and, for record selection, resident page tables."""
    import types
    from .elastic_kv import ElasticKV
    t = torch
    g = t.Generator(device="cpu").manual_seed(seed)
    slots, ctx, n, k = 6, 256, 4, (8 if block is None else 2)
    r2t = t.randperm(slots * ctx, generator=g).view(slots, ctx).to(t.int32).to(device)
    probe = types.SimpleNamespace(torch=t, ratio=ratio, fused=None, c=types.SimpleNamespace(
        req_to_token_pool=types.SimpleNamespace(req_to_token=r2t)),
        stale_limit=t.randint(0, ctx // ratio, (2, slots), generator=g).to(t.int32).to(device),
        route=t.tensor([[0, 1000, slots * ctx // 2, 0], [RESIDENT, 1000, 0, 0]], dtype=t.int32).to(device),
        mapping=t.randint(1, 1 << 20, (slots * ctx + 1,), generator=g).to(device))
    rid = t.randint(1, slots, (n,), generator=g).to(device)
    seq = t.randint(1, ctx // ratio, (n,), generator=g).to(device)
    top = (ctx // ratio) if block is None else (ctx // ratio) // block
    top_k = t.randint(-1, top, (n, k), generator=g).to(t.int32).to(device)
    swapped = t.randint(0, 1 << 20, (n, k if block is None else k * block), generator=g).to(t.int32).to(device)
    loc = t.randint(-1, slots * ctx, (3 * n,), generator=g).to(device)
    num_real = t.tensor([n], dtype=t.int32, device=device)
    for i in (0, 1):
        want = ElasticKV.reference_select(probe, i, rid, seq, top_k, block=block)
        got = select(probe, i, rid, seq, top_k, block=block)
        hit = want[2]
        checks = [t.equal(want[0], got[0]), t.equal(hit, got[2].bool()),
                  t.equal(t.where(hit, want[1], 0), t.where(hit, got[1], 0)),
                  t.equal(ElasticKV.reference_merge(probe, swapped, want[1], hit), merge(t, swapped, got[1], got[2])),
                  t.equal(ElasticKV.store_rows(probe, i, loc), store_rows(probe, i, loc)),
                  t.equal(ElasticKV.gate(probe, i, num_real), gate(probe, i, num_real))]
        if block is None:
            checks.append(t.equal(ElasticKV.resident_page_table(probe, rid, top_k, seq, offset=1000),
                                  resident(probe, rid, top_k, seq, offset=1000)))
        if not all(checks):
            return False
    return True
