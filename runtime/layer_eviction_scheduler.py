"""Explicit eager scheduler/writer boundary for request-layer demotion.

Not auto-installed: the pinned MiniMax scheduler must pass its actual request
table, leases and logical output locations through this boundary. Index K/V
and model Top-K stay on the native path.
"""

from dataclasses import dataclass

import torch

from .layer_hybrid_pool import PageBudgetExceeded


@dataclass(frozen=True, eq=False)
class ForwardTicket:
    sequence: int
    mode: str
    leases: tuple
    logical_chunks: tuple


@dataclass(frozen=True)
class ForwardDemand:
    """Read-only main-KV requirements, not a reservation or admission result."""
    mode: str
    leases: tuple
    logical_chunks: tuple
    groups: tuple
    needed_pages: tuple
    free_pages: tuple
    page_bytes: int
    reservations: tuple

    @property
    def fits_main_pool(self):
        return all(need <= free for need, free in zip(self.needed_pages, self.free_pages))


class LayerEvictionSchedulerBridge:
    """Serial, real-row, eager only. A failed partial forward exits the worker.

    Caller supplies a TP controller for distributed demotion. Owner-stream
    fences are intentionally retained until live event integration is tested.
    """

    def __init__(self, adapter, req_to_token, *, tp_controller=None):
        adapter._kv._index_tensor(req_to_token, 2)
        if req_to_token.shape[0] != adapter.layout.max_requests:
            raise ValueError("Use the native request table including padding row")
        if tp_controller is not None and tp_controller.kv is not adapter:
            raise ValueError("TP controller must own the same adapter")
        self.adapter = adapter
        self.req_to_token = req_to_token
        self.tp = tp_controller
        self._active = None
        self._old_pages = None
        self._written = set()
        self._resolved = set()
        self._sequence = 0
        self._submitted = False
        self._leases = {}
        self._request_ids = {}

    def _ready(self):
        self.adapter._kv._require_eager_stream()
        if self.tp is not None:
            self.tp.require_decode_ready(forward_ticket=self._active)

    def _idle(self):
        self._ready()
        if self._active is not None:
            raise RuntimeError("Forward is active; migration/admission/retirement must wait")

    def _lease(self, lease):
        if self._leases.get(lease.row) is not lease:
            raise ValueError("Unknown or stale scheduler lease")

    def open_request(self, row, *, request_id=None):
        self._idle()
        if request_id is not None and (type(request_id) is not str or not request_id
                                      or request_id in self._request_ids.values()):
            raise ValueError("Expected a distinct nonempty scheduler request ID")
        lease = self.adapter.open_request(row)
        self._leases[row] = lease
        self._request_ids[row] = request_id
        return lease

    def finish(self, lease):
        self._idle()
        self._lease(lease)
        self.adapter.finish(lease)
        del self._leases[lease.row]
        del self._request_ids[lease.row]

    def inspect_forward(self, leases, logical_chunks, *, mode):
        """Inspect actual incremental page requirements without changing ownership.

        Native scheduler has already assigned logical slots and updated its
        request table. The caller retains those slots for a prepared-batch retry,
        or must fully undo native preparation before releasing them. Never free
        only pending tail tokens from a page still holding history. This does not cover
        native index/activation/metadata memory and may become stale immediately.
        """
        self._idle()
        leases, chunks = tuple(leases), tuple(tuple(c) for c in logical_chunks)
        if mode not in ("prefill", "decode") or not leases or len(leases) != len(chunks):
            raise ValueError("Invalid forward mode or batch geometry")
        if len({lease.row for lease in leases}) != len(leases):
            raise ValueError("Only distinct real request rows are supported")
        flat = [loc for chunk in chunks for loc in chunk]
        if (any(not chunk for chunk in chunks) or len(set(flat)) != len(flat)
                or any(type(i) is not int or not self.adapter.layout.page_tokens <= i < self.adapter._kv._logical_slots
                       for i in flat)):
            raise ValueError("Expected distinct non-padding logical write locations")
        if mode == "decode" and any(len(c) != 1 for c in chunks):
            raise ValueError("Decode writes exactly one token per real request")
        reservations = []
        needed = {group.name: 0 for group in self.adapter.layout.groups}
        for lease, chunk in zip(leases, chunks):
            self._lease(lease)
            reference = self.adapter._state(lease, 0).logical
            end = len(reference) + len(chunk)
            if end > min(self.req_to_token.shape[1], self.adapter._max_context):
                raise ValueError("Forward exceeds context capacity")
            if self.req_to_token[lease.row, :end].tolist() != reference + list(chunk):
                raise ValueError("Native request table does not match completed and pending tokens")
            for layer in range(self.adapter.layout.layer_count):
                state = self.adapter._state(lease, layer)
                if state.logical != reference:
                    raise ValueError("All layers must agree on completed request history")
                if any(loc in self.adapter._logical_owners[layer] for loc in chunk):
                    raise ValueError("Pending logical location already belongs to a request")
                if state.hot is not None:
                    if mode != "decode":
                        raise ValueError("Hot requests cannot enter prefill/extend")
                    continue
                name = self.adapter._kv._pages.route(layer).name
                old = len(self.adapter._kv._pages.page_ids(lease, name))
                target = (end + self.adapter.layout.page_tokens - 1) // self.adapter.layout.page_tokens
                if target < old:
                    raise ValueError("Unexpected surplus resident reservation")
                needed[name] += target - old
                reservations.append((lease, name, old, target))
        groups = tuple(needed)
        return ForwardDemand(mode, leases, chunks, groups,
            tuple(needed[name] for name in groups),
            tuple(self.adapter._kv._pages.free_pages(name) for name in groups),
            self.adapter.layout.page_tokens * self.adapter.layout.kv_bytes_per_token_per_layer,
            tuple(reservations))

    def begin_forward(self, leases, logical_chunks, *, mode):
        """Reinspect and reserve every layer; never trust a planning snapshot."""
        demand = self.inspect_forward(leases, logical_chunks, mode=mode)
        leases, chunks, reservations = demand.leases, demand.logical_chunks, demand.reservations
        for name, count, free in zip(demand.groups, demand.needed_pages, demand.free_pages):
            if count > free:
                raise PageBudgetExceeded(f"Batch needs {count} pages from {name}")
        # Allocate publication metadata before reserving pages. A MemoryError
        # constructing the ticket must not leak an unowned batch reservation.
        sequence = self._sequence + 1
        ticket = ForwardTicket(sequence, mode, leases, chunks)
        written, resolved = set(), set()
        # No data has been submitted yet. Reservation bookkeeping is reversible.
        try:
            for lease, name, old, target in reservations:
                self.adapter._kv._pages.grow(lease, {name: target})
        except BaseException:
            with self.adapter._kv._poison_on_error("batch reservation rollback"):
                for lease, name, old, _ in reversed(reservations):
                    self.adapter._kv._pages.release_tail(lease, name, old)
            raise
        self._sequence = sequence
        self._active, self._old_pages = ticket, reservations
        self._written, self._resolved = written, resolved
        self._submitted = False
        return ticket

    def _ticket(self, ticket):
        self._ready()
        if ticket is not self._active or ticket is None:
            raise ValueError("Stale or foreign forward ticket")

    def set_kv_buffer(self, ticket, layer, logical_locations, cache_k, cache_v):
        """Route a native flattened write batch into each request's layer pool."""
        self._ticket(ticket)
        if not self._submitted:
            raise RuntimeError("Call begin_model before submitting native model work")
        self.adapter._kv._pages.route(layer.layer_id)
        if layer.layer_id in self._written:
            raise ValueError("Layer writer was already submitted in this forward")
        self.adapter._kv._index_tensor(logical_locations, 1)
        expected = [loc for chunk in ticket.logical_chunks for loc in chunk]
        if logical_locations.tolist() != expected:
            raise ValueError("Writer order/locations differ from admitted forward")
        shape = (len(expected), 1, self.adapter._kv._head_dim)
        if any(t.shape != shape or t.dtype != torch.bfloat16 or t.device != self.adapter._kv._device
               for t in (cache_k, cache_v)):
            raise ValueError("Writer requires native BF16 NHD K/V geometry")
        with self.adapter._kv._poison_on_error("batched layer write"):
            offset = 0
            for lease, chunk in zip(ticket.leases, ticket.logical_chunks):
                end = offset + len(chunk)
                self.adapter.append(lease, layer.layer_id, chunk, cache_k[offset:end], cache_v[offset:end])
                offset = end
            self._written.add(layer.layer_id)

    def begin_model(self, ticket):
        """Must precede *any* native work, including index K/V and projections."""
        self._ticket(ticket)
        if self._submitted:
            raise RuntimeError("Model work was already submitted")
        self._submitted = True

    def get_kv_buffer(self, layer):
        self._ready()
        return self.adapter.get_kv_buffer(layer)

    def get_prefill_mapping(self, ticket, layer):
        self._ticket(ticket)
        if ticket.mode != "prefill" or layer not in self._written:
            raise RuntimeError("Prefill must use its completed layer writer")
        return self.adapter._kv.get_prefill_mapping(layer)

    def swap_in_selected_blocks(self, ticket, req_pool_indices, seq_lens,
                                top_k_blocks, layer_id, sparse_block_size):
        """Native coordinator callback arguments plus an explicit forward ticket."""
        self._ticket(ticket)
        if ticket.mode != "decode" or layer_id not in self._written or layer_id in self._resolved:
            raise RuntimeError("Decode requires one completed writer and one resolver per layer")
        if sparse_block_size != self.adapter.layout.block_tokens:
            raise ValueError("Cannot change the model sparse block geometry")
        if req_pool_indices.tolist() != [lease.row for lease in ticket.leases]:
            raise ValueError("Resolver request order differs from admitted forward")
        result = self.adapter.resolve_decode(layer_id, top_k_blocks, req_pool_indices, seq_lens,
                                             self.req_to_token)
        self._resolved.add(layer_id)
        return result

    def end_forward(self, ticket):
        self._ticket(ticket)
        expected = set(range(self.adapter.layout.layer_count))
        if self._written != expected or (ticket.mode == "decode" and self._resolved != expected):
            raise RuntimeError("Cannot complete a forward with missing layers")
        # Caller invokes only after model.forward has submitted attention too.
        with self.adapter._kv._poison_on_error("forward completion fence"):
            self.adapter._kv._fence_before_reuse()
        self._clear_forward()

    def abort_forward(self, ticket):
        self._ticket(ticket)
        if self._submitted:
            self.adapter._kv._poisoned_at = "partial forward abort"
            raise RuntimeError("Cannot roll back a submitted model forward; exit worker group")
        with self.adapter._kv._poison_on_error("unwritten forward rollback"):
            for lease, name, old, _ in reversed(self._old_pages):
                self.adapter._kv._pages.release_tail(lease, name, old)
        self._clear_forward()

    def _clear_forward(self):
        self._active = self._old_pages = None
        self._submitted = False
        self._written, self._resolved = set(), set()

    def demote(self, epoch, lease, layer, hot_capacity):
        if self.tp is None:
            raise RuntimeError("Demotion requires an explicit TP controller, even for one rank")
        # Do not let a local preflight error skip the proposal collective and
        # strand healthy peers waiting in their demote call.
        ready = True
        try:
            self._idle()
            self._lease(lease)
        except Exception:
            ready = False
        return self.tp.demote(epoch, lease, layer, hot_capacity, local_ready=ready)
