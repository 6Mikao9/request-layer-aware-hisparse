"""Fused smooth-mode route of elastic per-layer HiSparse (issue #13): Triton kernels for ``ElasticKV.smooth_select``
and ``ElasticKV.merge``.

The torch statement in ``elastic_kv`` is the reference: about twenty small graph nodes per managed layer and
decode step. These kernels compute the same three outputs in one launch per layer (plus one launch for the
merge), reading the layer's route row and stale limits from the device, so one captured graph serves every
offload level. ``self_check`` compares both on random inputs; a worker binds the fused path only after it passes.
"""
import triton
import triton.language as tl

RESIDENT = 1


@triton.jit
def _select_tokens(top_k, rid, seq, r2t, stale, route, masked, rows, reads, k, r2t_stride, stale_stride, ratio,
                   BLOCK: tl.constexpr):
    req = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = offs < k
    pos = tl.load(top_k + req * k + offs, mask=inside, other=-1).to(tl.int64)
    slot = tl.load(rid + req).to(tl.int64)
    length = tl.load(seq + req).to(tl.int64)
    mode, base, backed = tl.load(route), tl.load(route + 1).to(tl.int64), tl.load(route + 2).to(tl.int64)
    limit = tl.load(stale + slot * stale_stride).to(tl.int64)
    in_range = (pos >= 0) & (pos < length)
    loc = tl.load(r2t + slot * r2t_stride + (tl.maximum(pos, 0) + 1) * ratio - 1, mask=inside & in_range,
                  other=0).to(tl.int64)
    row = loc // ratio
    hit = in_range & ((mode == RESIDENT) | ((pos < limit) & (row < backed)))
    tl.store(masked + req * k + offs, tl.where(hit, -1, pos).to(masked.dtype.element_ty), mask=inside)
    tl.store(rows + req * k + offs, row + base, mask=inside)
    tl.store(reads + req * k + offs, hit.to(tl.int8), mask=inside)


@triton.jit
def _select_blocks(top_k, rid, seq, r2t, stale, route, masked, rows, reads, k, r2t_stride, stale_stride, ratio,
                   BLOCK: tl.constexpr):
    """MiniMax: one program per selected block; a block skips the swap-in only when all its in-range records
    are readable from the resident region."""
    req, j = tl.program_id(0), tl.program_id(1)
    block = tl.load(top_k + req * k + j).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    pos = tl.where(block >= 0, block * BLOCK + offs, -1)
    slot = tl.load(rid + req).to(tl.int64)
    length = tl.load(seq + req).to(tl.int64)
    mode, base, backed = tl.load(route), tl.load(route + 1).to(tl.int64), tl.load(route + 2).to(tl.int64)
    limit = tl.load(stale + slot * stale_stride).to(tl.int64)
    in_range = (pos >= 0) & (pos < length)
    loc = tl.load(r2t + slot * r2t_stride + (tl.maximum(pos, 0) + 1) * ratio - 1, mask=in_range,
                  other=0).to(tl.int64)
    row = loc // ratio
    hit = in_range & ((mode == RESIDENT) | ((pos < limit) & (row < backed)))
    whole = tl.min((hit | ~in_range).to(tl.int32), axis=0)
    tl.store(masked + req * k + j, tl.where(whole == 1, -1, block).to(masked.dtype.element_ty))
    out = req * k * BLOCK + j * BLOCK + offs
    tl.store(rows + out, row + base)
    tl.store(reads + out, ((whole == 1) & in_range).to(tl.int8))


@triton.jit
def _merge(swapped, rows, reads, out, width, swapped_stride, BLOCK: tl.constexpr):
    req = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = offs < width
    s = tl.load(swapped + req * swapped_stride + offs, mask=inside, other=-1)
    w = tl.load(rows + req * width + offs, mask=inside, other=0)
    r = tl.load(reads + req * width + offs, mask=inside, other=0)
    tl.store(out + req * width + offs, tl.where(r != 0, w.to(s.dtype), s), mask=inside)


def select(kv, i, req_pool_indices, seq_lens, top_k, block=None):
    """Same contract as ``ElasticKV.smooth_select``; ``reads`` is int8 (nonzero = read the combined row)."""
    t = kv.torch
    n, k = top_k.shape
    width = k if block is None else k * block
    masked = t.empty_like(top_k)
    rows = t.empty((n, width), dtype=t.int64, device=top_k.device)
    reads = t.empty((n, width), dtype=t.int8, device=top_k.device)
    r2t = kv.c.req_to_token_pool.req_to_token
    stale = kv.stale_limit[i]
    args = (top_k.contiguous(), req_pool_indices, seq_lens, r2t, stale, kv.route[i], masked, rows, reads, k,
            r2t.stride(0), stale.stride(0), kv.ratio)
    if block is None:
        _select_tokens[(n, triton.cdiv(k, 256))](*args, BLOCK=256)
    else:
        _select_blocks[(n, k)](*args, BLOCK=block)
    return masked, rows, reads


def merge(torch, swapped, rows, reads):
    n, width = rows.shape
    out = torch.empty((n, width), dtype=swapped.dtype, device=swapped.device)
    _merge[(n, triton.cdiv(width, 256))](swapped, rows, reads, out, width, swapped.stride(0), BLOCK=256)
    return out


def self_check(torch, device, *, block=None, ratio=1, seed=0):
    """Fused and torch outputs agree on random tables in both modes (resident, and offloaded with stale rows)."""
    import types
    from .elastic_kv import ElasticKV
    t = torch
    g = t.Generator(device="cpu").manual_seed(seed)
    slots, ctx, n, k = 6, 256, 4, (8 if block is None else 2)
    r2t = t.randperm(slots * ctx, generator=g).view(slots, ctx).to(t.int32).to(device)
    probe = types.SimpleNamespace(torch=t, ratio=ratio, c=types.SimpleNamespace(
        req_to_token_pool=types.SimpleNamespace(req_to_token=r2t)),
        stale_limit=t.randint(0, ctx // ratio, (2, slots), generator=g).to(t.int32).to(device),
        route=t.tensor([[0, 1000, slots * ctx // 2, 0], [RESIDENT, 1000, 0, 0]], dtype=t.int32).to(device))
    rid = t.randint(1, slots, (n,), generator=g).to(device)
    seq = t.randint(1, ctx // ratio, (n,), generator=g).to(device)
    top = (ctx // ratio) if block is None else (ctx // ratio) // block
    top_k = t.randint(-1, top, (n, k), generator=g).to(t.int32).to(device)
    swapped = t.randint(0, 1 << 20, (n, k if block is None else k * block), generator=g).to(t.int32).to(device)
    for i in (0, 1):
        want = ElasticKV.reference_select(probe, i, rid, seq, top_k, block=block)
        got = select(probe, i, rid, seq, top_k, block=block)
        hit = want[2]
        if not (t.equal(want[0], got[0]) and t.equal(hit, got[2].bool())
                and t.equal(t.where(hit, want[1], 0), t.where(hit, got[1], 0))
                and t.equal(t.where(hit, want[1].to(swapped.dtype), swapped), merge(t, swapped, got[1], got[2]))):
            return False
    return True
