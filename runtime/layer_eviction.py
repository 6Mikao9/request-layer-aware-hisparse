"""Eager, rank-local request/layer demotion with real K/V and reusable pages.

This is an opt-in adapter, not installed in the SGLang scheduler. CUDA uses
the existing fused swap; CPU uses an explicit reference implementation. The
caller must serialize forwards and transitions and coordinate TP commits.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
import hashlib
import struct

import torch

from .layer_hybrid_kv import LayerHybridKV, _native_fused_swap
from .policy import validate_capacity


@dataclass(frozen=True, eq=False)
class DemotionTicket:
    lease: object
    layer: int
    length: int
    hot_capacity: int
    reclaimable_bytes: int


@dataclass
class _History:
    logical: list[int] = field(default_factory=list)
    hot: object = None
    logical_digest: object = field(default_factory=lambda: hashlib.sha256(b"layer-history-v1"))


@dataclass
class _Hot:
    capacity: int
    host_k: torch.Tensor
    host_v: torch.Tensor
    host_locations: torch.Tensor
    tokens: torch.Tensor
    locations: torch.Tensor
    lru: torch.Tensor

    @property
    def host_bytes(self):
        return (self.host_k.numel() + self.host_v.numel()) * self.host_k.element_size()


class RequestLayerEvictionKV:
    """One independent page domain per layer, heterogeneous modes per request.

    Main backing is fixed; retirement increases pool capacity, not free HBM
    in NVML. Metadata and pinned host history are allocated at prepare time.
    No prefix sharing, graph capture, concurrent caller, or TP commit protocol
    is implemented here. Those must not be silently enabled by a launcher.
    """

    def __init__(self, layout, logical_slots, head_dim, device, *, max_context,
                 host_budget_bytes, pool_factory=None):
        if any(group.mode != "resident" or len(group.layers) != 1 for group in layout.groups):
            raise ValueError("Dynamic eviction requires singleton resident layer groups")
        if type(max_context) is not int or not 1 <= max_context < 2**31:
            raise ValueError("Expected positive int32 context capacity")
        if type(host_budget_bytes) is not int or host_budget_bytes < 0:
            raise ValueError("Host history budget must be nonnegative integer bytes")
        self._kv = LayerHybridKV(layout, logical_slots, head_dim, device,
                                 pool_factory=pool_factory)
        self._max_context = max_context
        self._host_budget = host_budget_bytes
        self._host_reserved = 0
        self._history = {}
        self._pending = {}
        # Logical ownership survives demotion even after device bindings retire.
        self._logical_owners = {layer: {} for layer in range(layout.layer_count)}

    @property
    def layout(self):
        return self._kv.layout

    def _state(self, lease, layer, *, allow_pending=False):
        self._kv._require_eager_stream()
        group = self._kv._pages.route(layer)
        self._kv._pages.page_ids(lease, group.name)
        if not allow_pending and any(t.lease is lease and t.layer == layer for t in self._pending):
            raise RuntimeError("Layer has a prepared transition; commit or cancel it first")
        return self._history[lease, layer]

    def open_request(self, row):
        lease = self._kv.open_request(row)
        for layer in range(self.layout.layer_count):
            self._history[lease, layer] = _History()
        return lease

    def mode(self, lease, layer):
        return "hot" if self._state(lease, layer, allow_pending=True).hot else "resident"

    def get_kv_buffer(self, layer):
        """Read-only export; scheduler must quiesce users before any transition."""
        return self._kv.get_kv_buffer(layer)

    def accounting(self):
        self._kv._require_eager_stream()
        page_bytes = self.layout.page_tokens * self.layout.kv_bytes_per_token_per_layer
        return {
            "main_backing_bytes": self.layout.backing_bytes,
            "host_reserved_bytes": self._host_reserved,
            "free_bytes_by_layer": {
                group.layers[0]: self._kv._pages.free_pages(group.name) * page_bytes
                for group in self.layout.groups
            },
            "pending_transitions": len(self._pending),
        }

    def append(self, lease, layer, logical_locations, cache_k, cache_v):
        """Append completed K/V on one layer; hot decode accepts one token.

        The hot latest token is mirrored to pinned CPU history on every write.
        This deliberately synchronized reference path is not a performance
        optimization; event-based backup can replace it after GPU validation.
        """
        state = self._state(lease, layer)
        logical = list(logical_locations)
        if (not logical or len(set(logical)) != len(logical)
                or any(type(i) is not int or not self.layout.page_tokens <= i < self._kv._logical_slots
                       for i in logical)):
            raise ValueError("Append requires distinct valid logical data slots")
        if any(i in self._logical_owners[layer] for i in logical):
            raise ValueError("Logical slot is already owned on this layer")
        end = len(state.logical) + len(logical)
        if end > self._max_context:
            raise ValueError("Append exceeds host context capacity")
        if any(t.shape != (len(logical), 1, self._kv._head_dim)
               or t.dtype != torch.bfloat16 or t.device != self._kv._device
               for t in (cache_k, cache_v)):
            raise ValueError("Expected matching BF16 NHD K/V on the owning device")
        if state.hot:
            if len(logical) != 1:
                raise ValueError("Hot decode accepts exactly one latest token")
            hot = state.hot
            slot = int(hot.locations[0, hot.capacity].item())
            k, v = self.get_kv_buffer(layer)
            with self._kv._poison_on_error("hot latest write/backup"):
                k[slot:slot + 1].copy_(cache_k)
                v[slot:slot + 1].copy_(cache_v)
                hot.host_k[end - 1:end].copy_(cache_k, non_blocking=False)
                hot.host_v[end - 1:end].copy_(cache_v, non_blocking=False)
                hot.tokens[0, hot.capacity] = end - 1
                self._kv._fence_before_reuse()
        else:
            group = self._kv._pages.route(layer)
            count = (end + self.layout.page_tokens - 1) // self.layout.page_tokens
            indices = torch.tensor(logical, dtype=torch.int64, device=self._kv._device)
            self._kv.grow(lease, {group.name: count})
            # After the first binding mutation, even a pre-store failure must
            # not leave a healthy adapter with orphaned mappings in a partial page.
            with self._kv._poison_on_error("resident append"):
                self._kv.bind_logical_slots(lease, layer, dict(zip(
                    logical, range(len(state.logical), end))))
                self._kv.set_kv_buffer(SimpleNamespace(layer_id=layer), indices, cache_k, cache_v)
        with self._kv._poison_on_error("append ownership publication"):
            state.logical.extend(logical)
            for index in logical:
                state.logical_digest.update(struct.pack("<Q", index))
            self._logical_owners[layer].update({i: lease for i in logical})

    def _demotion_geometry(self, lease, layer, hot_capacity):
        state = self._state(lease, layer)
        if state.hot:
            raise ValueError("Layer is already hot")
        validate_capacity(hot_capacity, hot_capacity, self.layout.topk_blocks, self.layout.block_tokens)
        if hot_capacity % self.layout.page_tokens:
            raise ValueError("Hot capacity must be page aligned")
        if len(state.logical) <= hot_capacity:
            raise ValueError("Demotion requires context longer than the hot capacity")
        group = self._kv._pages.route(layer)
        pages = self._kv._pages.page_ids(lease, group.name)
        # Prefix sharing is deliberately unsupported. Sealed-but-single-owner
        # pages also cannot become writable LRU slots.
        owner = self._kv._pages._groups[group.name]
        if any(p in owner.sealed or owner.refcounts[p] != 1 for p in pages):
            raise ValueError("Demotion requires private unsealed pages")
        keep = hot_capacity // self.layout.page_tokens + 1
        reclaim = (len(pages) - keep) * self.layout.page_tokens * self.layout.kv_bytes_per_token_per_layer
        if reclaim <= 0:
            raise ValueError("Demotion would release no complete device pages")
        return state, keep, reclaim

    def reclaimable_bytes(self, lease, layer, hot_capacity):
        """Actual private full pages minus retained H and latest-token page."""
        return self._demotion_geometry(lease, layer, hot_capacity)[2]

    def _allocate_hot(self, lease, layer, capacity, length):
        shape = (self._max_context, 1, self._kv._head_dim)
        pinned = self._kv._device.type == "cuda"
        host_k = torch.empty(shape, dtype=torch.bfloat16, pin_memory=pinned)
        host_v = torch.empty(shape, dtype=torch.bfloat16, pin_memory=pinned)
        device = self._kv._device
        tokens = torch.arange(capacity + 1, dtype=torch.int32, device=device).unsqueeze(0)
        tokens[0, capacity] = length - 1
        locations = torch.tensor([[self._kv._pages.location(lease, layer, i)
                                   for i in range(capacity + 1)]], dtype=torch.int32, device=device)
        return _Hot(capacity, host_k, host_v,
                    torch.arange(self._max_context, dtype=torch.int64, device=device).unsqueeze(0),
                    tokens, locations,
                    torch.arange(capacity, dtype=torch.int16, device=device).unsqueeze(0))

    def _backup_history(self, lease, layer, hot, length):
        # Bound transient device gathering to one allocator page, not N KV rows.
        k, v = self.get_kv_buffer(layer)
        for start in range(0, length, self.layout.page_tokens):
            end = min(start + self.layout.page_tokens, length)
            indices = torch.tensor([self._kv._pages.location(lease, layer, i)
                                    for i in range(start, end)], dtype=torch.int64,
                                   device=self._kv._device)
            hot.host_k[start:end].copy_(k.index_select(0, indices), non_blocking=False)
            hot.host_v[start:end].copy_(v.index_select(0, indices), non_blocking=False)

    def prepare_demote(self, lease, layer, hot_capacity):
        """Back up without changing resident data; freeze this layer until decision.

        For TP, every rank must prepare the same request generation/layer/H and
        report success before any rank commits. A failed preparation can cancel
        healthy ranks. A failure after commit starts requires worker-group exit.
        """
        state, _, reclaim = self._demotion_geometry(lease, layer, hot_capacity)
        charge = self._max_context * self.layout.kv_bytes_per_token_per_layer
        if self._host_reserved + charge > self._host_budget:
            raise MemoryError("Pinned host history budget exhausted")
        # Prepare bookkeeping before charging or allocating backing storage.
        ticket = DemotionTicket(lease, layer, len(state.logical), hot_capacity, reclaim)
        self._host_reserved += charge
        try:
            hot = self._allocate_hot(lease, layer, hot_capacity, len(state.logical))
        except BaseException:
            # Allocation is not a resident mutation. Ordinary host/device OOM
            # is retryable; it must not consume the host reservation permanently.
            self._host_reserved -= charge
            raise
        try:
            self._kv._fence_before_reuse()
            self._backup_history(lease, layer, hot, len(state.logical))
            self._kv._fence_before_reuse()
        except BaseException:
            if self._kv._device.type == "cuda":
                self._kv._poisoned_at = "demotion preparation"
            else:
                self._host_reserved -= charge
            raise
        try:
            self._pending[ticket] = hot
        except BaseException:
            # All unpublished copies were fenced. Metadata allocation failure
            # must not leak host budget or leave an invisible prepared state.
            self._pending.pop(ticket, None)
            self._host_reserved -= charge
            raise
        return ticket

    def _ticket(self, ticket):
        self._kv._require_eager_stream()
        if ticket not in self._pending:
            raise ValueError("Unknown, stale, or already consumed demotion ticket")
        self._state(ticket.lease, ticket.layer, allow_pending=True)
        return self._pending[ticket]

    def cancel_demote(self, ticket):
        hot = self._ticket(ticket)
        # Preparation completed its fence; these unpublished buffers have no
        # readers, and this layer stayed frozen since preparation.
        del self._pending[ticket]
        self._host_reserved -= hot.host_bytes

    def commit_demote(self, ticket):
        """Publish hot mode and release tail only after copies and readers finish."""
        hot = self._ticket(ticket)
        lease, layer = ticket.lease, ticket.layer
        state = self._history[lease, layer]
        group = self._kv._pages.route(layer)
        before = self._kv._pages.free_pages(group.name)
        keep = hot.capacity // self.layout.page_tokens + 1
        k, v = self.get_kv_buffer(layer)
        slot = int(hot.locations[0, hot.capacity].item())
        with self._kv._poison_on_error("demotion commit"):
            self._kv._fence_before_reuse()
            k[slot:slot + 1].copy_(hot.host_k[ticket.length - 1:ticket.length], non_blocking=False)
            v[slot:slot + 1].copy_(hot.host_v[ticket.length - 1:ticket.length], non_blocking=False)
            self._kv._unbind(group.name, state.logical)
            self._kv._fence_before_reuse()
            state.hot = hot
            self._kv._pages.release_tail(lease, group.name, keep)
            del self._pending[ticket]
        released = self._kv._pages.free_pages(group.name) - before
        return released * self.layout.page_tokens * self.layout.kv_bytes_per_token_per_layer

    def resolve_decode(self, layer, topk_blocks, req_rows, seq_lens, req_to_token):
        """Mixed resident/hot rows, original order, fixed model Top-K semantics."""
        self._kv._require_eager_stream()
        self._kv._pages.route(layer)
        for tensor, dims in ((topk_blocks, 2), (req_rows, 1), (seq_lens, 1), (req_to_token, 2)):
            self._kv._index_tensor(tensor, dims)
        if (not req_rows.numel() or topk_blocks.dtype != torch.int32
                or topk_blocks.shape != (req_rows.numel(), self.layout.topk_blocks)
                or seq_lens.shape != req_rows.shape or req_to_token.size(0) != self.layout.max_requests):
            raise ValueError("Invalid decode batch geometry")
        rows = req_rows.tolist()
        if len(rows) != len(set(rows)):
            raise ValueError("Decode requires distinct real request rows")
        if torch.any(topk_blocks.to(torch.int64) >= 2**31 // self.layout.block_tokens).item():
            raise ValueError("Top-K block offset overflows int32")
        prepared = []
        # Validate the entire batch before any hot row mutates its cache.
        for i, row in enumerate(rows):
            lease = self._kv._pages.active_lease(row)
            state = self._state(lease, layer)
            n = len(state.logical)
            if not n or int(seq_lens[i].item()) != n or req_to_token.size(1) < n:
                raise ValueError("Decode length does not match this layer's completed history")
            if req_to_token[row, :n].tolist() != state.logical:
                raise ValueError("Request token table differs from its owned history")
            offsets, valid = self._kv._decode_offsets(topk_blocks[i:i+1], seq_lens[i:i+1])
            selected = offsets[valid].tolist()
            historical = [token for token in selected if token < n - 1]
            if state.hot and len(historical) != len(set(historical)):
                raise ValueError("Duplicate historical Top-K selections are unsupported by the fused kernel")
            prepared.append((lease, state))
        output = []
        for i, (lease, state) in enumerate(prepared):
            if state.hot:
                with self._kv._poison_on_error("dynamic hot swap"):
                    result = self._resolve_hot(layer, state.hot, topk_blocks[i:i+1], seq_lens[i:i+1])
            else:
                result = self._kv.resolve_decode(layer, topk_blocks[i:i+1], req_rows[i:i+1],
                                                 seq_lens[i:i+1], req_to_token)
            output.append(result)
        return torch.cat(output, dim=0)

    def _resolve_hot(self, layer, hot, blocks, lengths):
        k, v = self.get_kv_buffer(layer)
        result = torch.full((1, self.layout.topk_blocks * self.layout.block_tokens), -1,
                            dtype=torch.int32, device=self._kv._device)
        if self._kv._device.type == "cpu":
            self._reference_swap(hot, blocks, lengths, k, v, result)
        else:
            _native_fused_swap(
                top_k_blocks=blocks.contiguous(), device_buffer_tokens=hot.tokens,
                host_cache_locs=hot.host_locations, device_buffer_locs=hot.locations,
                host_cache_k=hot.host_k, host_cache_v=hot.host_v,
                device_buffer_k=k, device_buffer_v=v, top_k_device_locs=result,
                req_pool_indices=torch.zeros(1, dtype=torch.int64, device=self._kv._device),
                seq_lens=lengths.contiguous(), lru_slots=hot.lru,
                item_size_bytes=2*self._kv._head_dim, hot_buffer_size=hot.capacity,
                sparse_block_size=self.layout.block_tokens, page_size=1,
                block_size=960, skip_io=False,
            )
        return result

    def _reference_swap(self, hot, blocks, lengths, k, v, result):
        """CPU oracle, never reported as native-kernel performance or traffic."""
        offsets, valid = self._kv._decode_offsets(blocks, lengths)
        length = int(lengths[0].item())
        selected = offsets[valid].tolist()
        old_lru = hot.lru[0].tolist()
        lookup = {token: slot for slot, token in enumerate(hot.tokens[0, :hot.capacity].tolist())
                  if token >= 0}
        hits = {lookup[token] for token in selected if token < length-1 and token in lookup}
        misses = [token for token in selected if token < length-1 and token not in lookup]
        victims = [slot for slot in old_lru if slot not in hits][:len(misses)]
        if len(victims) != len(misses):
            raise RuntimeError("Insufficient unselected hot slots")
        for token, slot in zip(misses, victims):
            location = int(hot.locations[0, slot].item())
            k[location].copy_(hot.host_k[token])
            v[location].copy_(hot.host_v[token])
            hot.tokens[0, slot] = token
            lookup[token] = slot
        lru = [slot for slot in old_lru if slot not in hits and slot not in victims]
        lru += victims + [slot for slot in old_lru if slot in hits]
        hot.lru[0].copy_(torch.tensor(lru, dtype=torch.int16))
        for column, (token, ok) in enumerate(zip(offsets[0].tolist(), valid[0].tolist())):
            if ok:
                slot = hot.capacity if token == length-1 else lookup[token]
                result[0, column] = hot.locations[0, slot]

    def finish(self, lease):
        for layer in range(self.layout.layer_count):
            self._state(lease, layer)  # A pending ticket must be explicitly cancelled.
        self._kv.finish(lease)
        for layer in range(self.layout.layer_count):
            state = self._history.pop((lease, layer))
            for logical in state.logical:
                del self._logical_owners[layer][logical]
            if state.hot:
                self._host_reserved -= state.hot.host_bytes
