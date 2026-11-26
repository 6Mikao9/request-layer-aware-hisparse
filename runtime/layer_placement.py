"""CPU reference for a frozen whole-layer placement S joined with the per-request hot pool.

Standard library only; importing this module never imports torch or SGLang.

Layers outside S keep their complete main KV on the GPU at the logical token
location the stock allocator records in ``req_to_token``. The model's own
selection reads those rows directly and never reaches the hot pool. Layers in S
keep complete history in pinned host memory and serve decode through a hot pool
whose eligible layers are exactly S: the stock per-layer LRU (``NativeLayerLRU``,
layout native) or the arena (``SharedLRUReference``, isolated or shared).
Borrowing can therefore move rows only between layers of S, of one request, on
one TP rank. S equal to every eligible layer is the all-layer arena of
``shared_lru``; an empty S is the all-resident layout.

Addresses are ``(pool, plane, loc)``: ``("resident", r, logical loc)`` in the
r-th resident plane or ``("hot", p, device loc)`` in the p-th offloaded plane.
``features.resident_direct`` picks how a resident layer reaches the consumer:
directly through ``req_to_token`` or through the resident resolver, whose
output table ``resident_slot_table`` mirrors the CUDA kernel. Both read the
same rows; only the extra resolver launch differs.
"""
from dataclasses import dataclass, replace

from . import shared_lru as s

RESIDENT, HOT = "resident", "hot"


def resident_slot_table(top_k, req_pool_indices, seq_lens, req_to_token, *, num_real, block_size):
    """CPU mirror of ``rlh_layer_placement::resolve_resident`` (``features.resident_direct=false``).

    Row b lists, per flattened selected position, the ``req_to_token`` location the
    stock consumer reads, or -1 for invalid blocks, positions at or past ``seq_len``
    (or past the table width) and padded graph rows ``b >= num_real``: the output
    conventions of the native swap kernel.
    """
    out = []
    for b, blocks in enumerate(top_k):
        if b >= num_real:
            out.append([-1] * (len(blocks) * block_size))
            continue
        locs = req_to_token[req_pool_indices[b]]
        out.append([locs[t] if 0 <= t < len(locs) else -1
                    for t in s.selected_tokens(blocks, block_size, seq_lens[b])])
    return out


@dataclass(frozen=True)
class PlacementPlan:
    layers: tuple     # eligible sparse physical KV layers, ascending
    offload: tuple    # S in plane order: host history plus hot pool
    resident: tuple   # every other eligible layer: complete GPU main KV

    def route(self, layer):
        """(pool, plane index) serving ``layer``."""
        if layer in self.offload:
            return HOT, self.offload.index(layer)
        if layer in self.resident:
            return RESIDENT, self.resident.index(layer)
        raise ValueError(f"Layer {layer} is not an eligible sparse layer {list(self.layers)}")


def plan_placement(config, layers):
    """Resolve S against the model's eligible layers, which the JSON parser cannot know."""
    layers = tuple(layers)
    if not layers or any(type(x) is not int or x < 0 for x in layers) or list(layers) != sorted(set(layers)):
        raise ValueError("Eligible layers must be sorted unique absolute physical layer IDs")
    mode = config.placement.mode
    if mode == "native":
        offload = layers
    elif mode == "full_resident":
        offload = ()
    else:
        unknown = sorted(set(config.placement.offload_layers) - set(layers))
        if unknown:
            raise ValueError(f"placement.offload_layers names {unknown}, which are not eligible sparse layers "
                             f"{list(layers)}")
        offload = config.placement.offload_layers
    return PlacementPlan(layers, tuple(offload), tuple(x for x in layers if x not in offload))


def native_device_rows(seq_len, hot_rows, page):
    """Rows one offloaded plane holds for a request (``alloc_device_buffer``, ``_grow_device_buffers``)."""
    cap = min(-(-seq_len // page) * page, hot_rows)
    return hot_rows + page if cap == hot_rows else cap


class LogicalAllocator:
    """Stock page-granular logical locations (``req_to_token`` rows); page 0 is the padding sink."""

    def __init__(self, total_locs, page):
        if any(type(x) is not int or x <= 0 for x in (total_locs, page)) or total_locs % page:
            raise ValueError("Require positive integers with total_locs divisible by page")
        self.page, self.total_locs = page, total_locs
        self._free = list(range(page, total_locs, page))
        self._pages, self.req_to_token = {}, {}

    def can_extend(self, slot, count):
        length = len(self.req_to_token.get(slot, ()))
        return -(-(length + count) // self.page) - -(-length // self.page) <= len(self._free)

    def extend(self, slot, count):
        """Append ``count`` tokens; new pages come from the top of the free list (non-identity locations)."""
        if not self.can_extend(slot, count):
            raise MemoryError("Logical pool exhausted; nothing was allocated")
        locs, pages = self.req_to_token.setdefault(slot, []), self._pages.setdefault(slot, [])
        for _ in range(count):
            if len(locs) % self.page == 0:
                pages.append(self._free.pop())
            locs.append(pages[-1] + len(locs) % self.page)
        return locs

    def free(self, slot):
        pages = self._pages.pop(slot, None)
        self.req_to_token.pop(slot, None)
        if pages is None:
            return False
        self._free = sorted(self._free + pages)
        return True

    @property
    def allocated_rows(self):
        return sum(len(p) for p in self._pages.values()) * self.page


class CombinedReference:
    """One TP rank: resident planes, the hot pool of S, logical locations and host backing."""

    def __init__(self, config, layers, *, hot_rows, selected, page, item_bytes, rank=0, logical_locs=None,
                 device_locs=None):
        self.config, self.placement = config, plan_placement(config, layers)
        if type(selected) is not int or type(hot_rows) is not int or not 0 < selected <= hot_rows:
            raise ValueError("Require 0 < selected width K <= H: every layer must hold one complete selection")
        self.hot_rows, self.selected, self.page, self.item_bytes, self.rank = hot_rows, selected, page, item_bytes, rank
        offload, self.plan, self.hot = self.placement.offload, None, None
        if offload and config.uses_arena:
            self.plan = s.plan_arena(config, offload, hot_rows, selected)
            self.hot = s.SharedLRUReference(self.plan, page=page, item_bytes=item_bytes, rank=rank,
                                            total_locs=device_locs)
        elif offload:
            self.hot = s.NativeLayerLRU(offload, hot_rows, page=page, item_bytes=item_bytes, rank=rank,
                                        total_locs=device_locs)
        self.logical = LogicalAllocator(logical_locs or page * 4096, page)
        self.resident = {}      # (resident plane, logical loc) -> (K bytes, V bytes)
        self._requests = {}     # slot -> [generation, seq_len]
        self.resident_direct = config.uses_resident_direct
        self.resolver_launches = 0   # resident resolver invocations (zero on the direct path)

    def _live(self, slot, generation):
        req = self._requests.get(slot)
        if req is None or req[0] != generation:
            raise s.StaleCompletion(f"No live request generation {generation} in slot {slot}")
        return req

    def _arena(self):
        if self.plan is None:
            raise ValueError("Leases exist only for arena layouts; the stock layer LRU completes in-kernel")
        return self.hot

    def _write_resident(self, slot, generation, token):
        loc = self.logical.req_to_token[slot][token]
        for plane, layer in enumerate(self.placement.resident):
            self.resident[(plane, loc)] = (s.payload(generation, self.rank, layer, token, "k", self.item_bytes),
                                           s.payload(generation, self.rank, layer, token, "v", self.item_bytes))

    def admit(self, slot, generation, *, seq_len, epoch=0):
        """Prefill: resident rows are written at their logical locations, S is staged to the host."""
        if slot in self._requests:
            raise ValueError("Request slot is still live")
        if type(seq_len) is not int or seq_len < 1:
            raise ValueError("A request needs at least one prefill token")
        locs = list(self.logical.extend(slot, seq_len))
        try:
            device = None
            if self.plan is not None:
                device = self.hot.admit(slot, generation, self.rank, self.plan.layers, min(seq_len, self.hot_rows),
                                        self.plan.rows, seq_len=seq_len, epoch=epoch)
            elif self.hot is not None:
                device = self.hot.admit(slot, generation, seq_len=seq_len)
        except BaseException:
            self.logical.free(slot)
            raise
        self._requests[slot] = [generation, seq_len]
        for token in range(seq_len):
            self._write_resident(slot, generation, token)
        return dict(logical=locs, device=device)

    def advance(self, slot, generation):
        """Append one decoded token: hot-pool tail rules first, then the resident write at its new location."""
        req = self._live(slot, generation)
        if not self.logical.can_extend(slot, 1):
            raise MemoryError("Logical pool exhausted; the token was not appended")
        if self.hot is not None:
            self.hot.advance(slot, generation)        # Deferred while the old tail lacks its host backup
        self.logical.extend(slot, 1)
        token, req[1] = req[1], req[1] + 1
        self._write_resident(slot, generation, token)
        return token

    def backup(self, slot, generation):
        """Previous-token D2H backup of the offloaded planes; a lease on arena layouts, else immediate."""
        self._live(slot, generation)
        return None if self.hot is None else self.hot.backup(slot, generation)

    def complete(self, lease_id, kind):
        self._arena().complete(lease_id, kind)

    def cancel(self, lease_id):
        self._arena().cancel(lease_id)

    def fail(self, lease_id):
        self._arena().fail(lease_id)

    def pin(self, slot, generation, layer, tokens, kind="prefetch"):
        if self.placement.route(layer)[0] != HOT:
            raise ValueError(f"Layer {layer} is resident; its rows are never evicted")
        return self._arena().pin(slot, generation, layer, tokens, kind)

    def resolve(self, batch, epoch, *, defer_gather=False):
        """One layer invocation in original selection order; resident layers never reach the hot pool."""
        pool, plane = self.placement.route(batch.layer)
        if pool == HOT:
            out = (self.hot.resolve(batch, epoch, defer_gather=defer_gather) if self.plan is not None
                   else self.hot.resolve(batch, epoch))
            return [replace(r, addresses=[None if a is None else (HOT, *a) for a in r.addresses]) for r in out]
        real = [sel.slot for sel in batch.rows if sel.real]
        if len(set(real)) != len(real):
            raise ValueError("A request appears twice in one layer invocation")
        resolved = [self._resolve_resident(batch, plane, sel) for sel in batch.rows]
        if self.resident_direct:
            return resolved
        self.resolver_launches += 1        # one kernel per resident layer invocation; padded rows stay -1
        rows = [sel for sel in batch.rows if sel.real] + [sel for sel in batch.rows if not sel.real]
        if rows != list(batch.rows):
            raise ValueError("Padded graph rows must follow the real rows")
        table = resident_slot_table([sel.blocks for sel in rows], [sel.slot for sel in rows],
                                    [sel.seq_len for sel in rows], self.logical.req_to_token,
                                    num_real=len(real), block_size=batch.block_size)
        return [replace(r, addresses=[None if loc < 0 else (RESIDENT, plane, loc) for loc in row])
                for r, row in zip(resolved, table)]

    def _resolve_resident(self, batch, plane, sel):
        width = len(sel.blocks) * batch.block_size
        if not sel.real:
            return s.Resolved([None] * width, [s.PAD] * width)
        req = self._live(sel.slot, sel.generation)
        if sel.seq_len != req[1]:
            raise ValueError("Selection sequence length differs from the request state")
        if width > self.selected:
            raise ValueError("Selection wider than the planned width K")
        tokens = s.selected_tokens(sel.blocks, batch.block_size, sel.seq_len)
        locs = self.logical.req_to_token[sel.slot]
        return s.Resolved([(RESIDENT, plane, locs[t]) if t >= 0 else None for t in tokens],
                          [RESIDENT if t >= 0 else s.PAD for t in tokens])

    def finish(self, slot, generation):
        """Retire after every completion fence; logical pages return to the stock allocator. Idempotent."""
        req = self._requests.get(slot)
        if req is None or req[0] != generation:
            return False
        if self.hot is not None and not self.hot.finish(slot, generation):   # Deferred with leases outstanding
            raise AssertionError("The hot pool lost a live request")
        self.logical.free(slot)
        del self._requests[slot]
        return True

    # -- inspection -------------------------------------------------------

    def read(self, address):
        pool, plane, loc = address
        return (self.resident if pool == RESIDENT else self.hot.device).get((plane, loc))

    def req_to_token(self, slot, generation):
        self._live(slot, generation)
        return list(self.logical.req_to_token[slot])

    def occupancy(self, slot, generation):
        """Hot rows owned per offloaded layer (arena layouts)."""
        return self._arena().occupancy(slot, generation)

    def host_payload(self, slot, generation, layer, token):
        """Bytes the pinned host history holds; resident layers have no host backing."""
        if self.placement.route(layer)[0] != HOT:
            raise ValueError(f"Layer {layer} is resident and has no host backing")
        generation, seq_len = self._live(slot, generation)
        if not 0 <= token < seq_len:
            raise ValueError(f"Token {token} is outside the request history")
        return (s.payload(generation, self.rank, layer, token, "k", self.item_bytes),
                s.payload(generation, self.rank, layer, token, "v", self.item_bytes))

    def host_ledger(self):
        """Pinned host rows of live requests: every token of every offloaded layer, none of a resident one."""
        tokens = sum(seq_len for _, seq_len in self._requests.values())
        rows = tokens * len(self.placement.offload)
        return dict(layers=list(self.placement.offload), tokens=tokens, rows=rows, bytes=rows * 2 * self.item_bytes)

    def device_rows(self):
        """Live physical rows by pool under the stock sizing rules (short requests own only their pages)."""
        hot = sum(native_device_rows(seq_len, self.hot_rows, self.page) for _, seq_len in self._requests.values())
        return dict(hot=hot * len(self.placement.offload),
                    resident=self.logical.allocated_rows * len(self.placement.resident))


class CombinedReader:
    """Second reader: checks each address, its pool and its bytes without reading policy state."""

    def __init__(self, ref):
        self.ref = ref
        self.hot_reader = None if ref.hot is None else s.DirectReader(
            ref.hot.device, rank=ref.rank, item_bytes=ref.item_bytes, hot_rows=ref.hot_rows,
            planes=len(ref.placement.offload))

    def check(self, resolved, *, slot, generation, layer, tokens, seq_len):
        pool, plane = self.ref.placement.route(layer)
        if pool == HOT:
            if any(a is not None and a[0] != HOT for a in resolved.addresses):
                raise AssertionError(f"layer {layer}: an offloaded layer resolved outside its hot planes")
            bare = replace(resolved, addresses=[None if a is None else a[1:] for a in resolved.addresses])
            return self.hot_reader.check(bare, generation=generation, layer=layer, plane=plane, tokens=tokens,
                                         seq_len=seq_len, locs=self.ref.hot.locs(slot, generation))
        locs = self.ref.logical.req_to_token[slot]
        for position, (token, address, cls) in enumerate(zip(tokens, resolved.addresses, resolved.classes)):
            where = f"resident layer {layer} position {position} token {token}"
            if token < 0:
                if address is not None or cls != s.PAD:
                    raise AssertionError(f"{where}: invalid position must stay padding")
                continue
            if address != (RESIDENT, plane, locs[token]) or cls != RESIDENT:
                raise AssertionError(f"{where}: must read its own logical location {locs[token]}, got {address}")
            expected = (s.payload(generation, self.ref.rank, layer, token, "k", self.ref.item_bytes),
                        s.payload(generation, self.ref.rank, layer, token, "v", self.ref.item_bytes))
            if self.ref.resident.get(address[1:]) != expected:
                raise AssertionError(f"{where}: bytes at {address} are not this layer/token's K/V")
        return sum(t >= 0 for t in tokens)


def cell_bytes(*, sparse_layers, offload_layers, dense_layers, row_bytes, ratio, index_bytes):
    """Device-token cost the pool configurator charges for MiniMax sparse attention with HiSparse.

    Stock: ``(dense * ratio + sparse) * row + index * ratio``. A resident sparse layer is
    addressed by logical location, so it costs ``ratio`` rows instead of one hot row.
    """
    resident = sparse_layers - offload_layers
    if not 0 <= offload_layers <= sparse_layers:
        raise ValueError("Offloaded layers must be a subset of the sparse layers")
    return (dense_layers * ratio + offload_layers + resident * ratio) * row_bytes + index_bytes * ratio


def placement_ledger(*, placement, hot_rows, page, item_bytes, slots, device_rows, logical_rows, plan=None,
                     counters=False):
    """Estimated per-rank bytes of one combined-layout arm; GPU receipts replace them with allocations.

    ``device_rows`` is one offloaded plane (stock ``size + page``) and ``logical_rows`` one
    resident plane (``size * ratio + page``). Host history covers the logical capacity of S only.
    """
    S, R = placement.offload, placement.resident
    hot = (s.byte_ledger(layers=S, hot_rows=hot_rows, page=page, item_bytes=item_bytes, slots=slots,
                         pool_rows=device_rows, plan=plan, counters=counters) if S else None)
    resident = len(R) * logical_rows * 2 * item_bytes
    empty = dict(history_rows=0, history_payload_bytes=0, tail_rows=0, tail_payload_bytes=0,
                 bypass_scratch_rows=0, cacheable_history_rows=0)
    unavailable = dict(hot["unavailable"]) if hot else {}
    return dict(
        estimated=True, rank_local=True, offload_layers=list(S), resident_layers=list(R),
        main_kv_bytes=(hot["main_kv_bytes"] if hot else 0) + resident,
        hot_main_kv_bytes=hot["main_kv_bytes"] if hot else 0,
        resident_main_kv_bytes=resident,
        main_kv_layout=dict(offloaded=None if not S else ("contiguous planes of S" if plan is not None
                                                           else "one allocation per offloaded layer"),
                            resident="one plane per resident layer, addressed by logical location"),
        native_metadata_bytes=hot["native_metadata_bytes"] if hot else 0,
        arena_metadata_bytes=hot["arena_metadata_bytes"] if hot else 0,
        host_pinned_bytes=(logical_rows - page) * len(S) * 2 * item_bytes,
        per_request=dict(hot["per_request"]) if hot else empty,
        overlaps=["per_request history and tails are rows of hot_main_kv_bytes",
                  "live resident rows are rows of resident_main_kv_bytes"],
        unavailable=unavailable | {"index_kv": "identical in every arm; measured by the GPU receipt"},
    )