"""Eager MiniMax K/V backing and resolver primitives for a grouped coordinator."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import torch

from .layer_hybrid_pool import LayerPagePool
from .policy import validate_metadata_shapes, validate_metadata_strides


_INDEX_DTYPES = (torch.int32, torch.int64)


class LayerHybridPoisoned(RuntimeError):
    """A mutation or fence failed; the worker must discard this backing."""


@dataclass(frozen=True)
class HotLayerState:
    """Group-local state supplied by a coordinator after host backup is ready."""

    tokens: torch.Tensor
    locations: torch.Tensor
    lru: torch.Tensor
    host_locations: torch.Tensor
    host_k: torch.Tensor
    host_v: torch.Tensor
    wait_for_backup: Callable[[], None]


@dataclass(frozen=True)
class HostPrefillTarget:
    """Group-local native host pool and reserved, unpublished host slots."""

    pool: object
    locations: torch.Tensor


@dataclass
class _Prefill:
    logical: list[int] = field(default_factory=list)
    written: dict[int, int] = field(default_factory=dict)


def _native_mha_pool(**kwargs):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    return MHATokenToKVPool(**kwargs)


def _native_fused_swap(**kwargs):
    from sglang.kernels.ops.kvcache.hisparse import load_blocks_to_device_buffer_mha

    load_blocks_to_device_buffer_mha(**kwargs)


class LayerHybridKV:
    """Own fixed BF16 NHD storage plus eager logical-write routing.

    Callers serialize scheduler admission and forward work and retire a row
    only after stopping new submissions for that row on every stream.
    """

    def __init__(self, layout, logical_slots, head_dim, device, *, pool_factory=None):
        if layout.max_requests < 2:
            raise ValueError("Native request table must include its padding row zero")
        if type(logical_slots) is not int or logical_slots <= layout.page_tokens:
            raise ValueError("Logical address space must include padding and data")
        if type(head_dim) is not int or head_dim <= 0:
            raise ValueError("Head dimension must be positive")
        if layout.kv_bytes_per_token_per_layer != 4 * head_dim:
            raise ValueError("Backing budget must charge BF16 K and V for one head per layer")
        self._pages = LayerPagePool(layout)
        self._device = torch.device(device)
        if self._device.type not in ("cpu", "cuda"):
            raise ValueError("Expected CPU testing or CUDA eager execution")
        self._stream = None
        self._poisoned_at = None
        if self._device.type == "cuda":
            if self._device.index is None:
                self._device = torch.device("cuda", torch.cuda.current_device())
            self._stream = torch.cuda.current_stream(self._device).cuda_stream
            self._require_eager_stream()
        self._logical_slots = logical_slots
        self._head_dim = head_dim
        self._pools = {}
        self._local_layers = {}
        self._maps = {}
        self._bindings = {}
        # One HOST bit per physical slot, shared by the layers of a group.
        # This indexes active logical WRITERS only. An empty bit says nothing
        # about page ownership: hot caches retain pages after unbinding writers.
        self._physical_writers = {}
        self._prefill = {}
        pool_factory = _native_mha_pool if pool_factory is None else pool_factory
        for group in layout.groups:
            self._local_layers.update({layer: local for local, layer in enumerate(group.layers)})
            size = group.allocatable_pages * layout.page_tokens
            if size + layout.page_tokens > 2**31:
                raise ValueError("Physical slots must fit the native int32 decode result")
            pool = pool_factory(
                size=size, page_size=layout.page_tokens, dtype=torch.bfloat16,
                head_num=1, head_dim=head_dim, layer_num=len(group.layers),
                device=str(self._device), enable_memory_saver=False,
                start_layer=0, end_layer=len(group.layers), enable_alt_stream=False,
                kv_cache_layout="nhd", post_capture_active=False,
            )
            for local in range(len(group.layers)):
                for tensor in pool.get_kv_buffer(local):
                    self._validate_backing(tensor, size + layout.page_tokens)
            self._pools[group.name] = pool
            mapping = torch.full((logical_slots,), -1, dtype=torch.int64, device=self._device)
            mapping[:layout.page_tokens] = 0
            self._maps[group.name] = mapping
            self._bindings[group.name] = {}
            self._physical_writers[group.name] = bytearray((size + layout.page_tokens + 7) // 8)
        self._block_offsets = torch.arange(layout.block_tokens, device=self._device)

    @property
    def layout(self):
        return self._pages.layout

    @property
    def writer_index_bytes(self):
        """Fixed CPU bitmap payload; excludes Python headers and other metadata."""
        return sum(len(bits) for bits in self._physical_writers.values())

    def _require_eager_stream(self):
        if self._poisoned_at is not None:
            raise LayerHybridPoisoned(f"LayerHybridKV is poisoned after {self._poisoned_at}; discard this worker")
        if self._stream is not None:
            with torch.cuda.device(self._device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("LayerHybridKV has no CUDA Graph integration")
                if torch.cuda.current_stream(self._device).cuda_stream != self._stream:
                    raise RuntimeError("LayerHybridKV requires its owning CUDA stream")

    @contextmanager
    def _poison_on_error(self, stage):
        try:
            yield
        except BaseException:
            if self._poisoned_at is None:
                self._poisoned_at = stage
            raise

    def _validate_backing(self, tensor, rows):
        if (tensor.shape != (rows, 1, self._head_dim) or tensor.device != self._device
                or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous()):
            raise ValueError("Expected contiguous BF16 NHD K/V storage with the charged row count")

    def _index_tensor(self, tensor, dims):
        if tensor.device != self._device or tensor.dtype not in _INDEX_DTYPES or tensor.ndim != dims:
            raise ValueError("Invalid index tensor device, dtype, or rank")

    @staticmethod
    def _check_range(indices, low, high, description):
        if torch.any((indices < low) | (indices >= high)).item():
            raise ValueError(f"{description} is out of bounds")

    def open_request(self, row):
        self._require_eager_stream()
        if row == 0:
            raise ValueError("Native request row zero is reserved for padding")
        return self._pages.open_request(row)

    def open_prefill(self, row):
        lease = self.open_request(row)
        self._prefill[lease] = _Prefill()
        return lease

    def _require_decode_lease(self, lease):
        if lease in self._prefill:
            raise ValueError("Prefill lease must transition before decode mutations")

    def grow(self, lease, targets):
        """Reserve decode pages for one request before publishing logical locations."""
        self._require_eager_stream()
        self._require_decode_lease(lease)
        self._pages.grow(lease, targets)

    def append_prefill(self, lease, logical_locations):
        """Reserve and bind a chunk of full prefill in all groups at once."""
        self._require_eager_stream()
        state = self._prefill.get(lease)
        if state is None:
            raise ValueError("Expected an active prefill lease")
        logical = tuple(logical_locations)
        if not logical or any(type(index) is not int or not
                              self.layout.page_tokens <= index < self._logical_slots for index in logical):
            raise ValueError("Expected nonempty valid prefill logical data slots")
        if len(set(logical)) != len(logical) or any(
            index in bindings for bindings in self._bindings.values() for index in logical
        ):
            raise ValueError("Prefill logical slots must be new and distinct")
        start = len(state.logical)
        end = start + len(logical)
        pages = (end + self.layout.page_tokens - 1) // self.layout.page_tokens
        self._pages.grow_prefill(lease, {group.name: pages for group in self.layout.groups})
        with self._poison_on_error("prefill binding"):
            positions = dict(zip(logical, range(start, end)))
            for group in self.layout.groups:
                self._bind_logical_slots(lease, group.layers[0], positions)
            state.logical.extend(logical)

    def _prefill_keep_pages(self, group, prefix_pages):
        hot_pages = self.layout.hot_page_limit(group)
        return hot_pages if prefix_pages >= hot_pages - 1 else prefix_pages

    def _prefill_backup(self, lease, group, target, logical):
        if type(target) is not HostPrefillTarget:
            raise ValueError("Expected native host pool and reserved locations")
        self._index_tensor(target.locations, 1)
        if (target.locations.dtype != torch.int64 or not target.locations.is_contiguous()
                or target.locations.numel() != len(logical)
                or torch.unique(target.locations).numel() != len(logical)):
            raise ValueError("Host backup requires one distinct contiguous int64 index per prefill token")
        host_pool = target.pool
        if self._stream is not None:
            from sglang.srt.mem_cache.pool_host.mha import HiSparseMHATokenToKVPoolHost

            if (type(host_pool) is not HiSparseMHATokenToKVPoolHost
                    or host_pool.token_stride_size != 2 * self._head_dim
                    or host_pool.page_size != self.layout.page_tokens):
                raise ValueError("Expected the native MiniMax host transfer pool for this layout")
        if (len(host_pool.k_buffer) != len(group.layers)
                or len(host_pool.v_buffer) != len(group.layers)):
            raise ValueError("Host backup layer count differs from its group")
        for tensor in (*host_pool.k_buffer, *host_pool.v_buffer):
            if (tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16
                    or tensor.ndim != 3 or tensor.shape[1:] != (1, self._head_dim)
                    or not tensor.is_contiguous()
                    or (self._stream is not None and not tensor.is_pinned())):
                raise ValueError("Host backup requires matching BF16 NHD storage")
            self._check_range(target.locations, 0, tensor.size(0), "Reserved host slot")
        source = torch.tensor(
            [self._pages.location(lease, group.layers[0], pos) for pos in range(len(logical))],
            dtype=torch.int64, device=self._device,
        )
        return host_pool, target.locations, source

    def commit_prefill(self, lease, host_targets):
        """Back up full hot history, then retain only each group's decode pages.

        Host targets remain caller-owned and unpublished until this succeeds.
        A CPU backup failure can leave partial host data but keeps prefill
        retryable. A native CUDA backup or fence failure requires worker exit.
        """
        self._require_eager_stream()
        state = self._prefill.get(lease)
        if state is None or not state.logical:
            raise ValueError("Expected a populated prefill lease")
        count = len(state.logical)
        if any(state.written.get(layer, 0) != count for layer in range(self.layout.layer_count)):
            raise ValueError("Every layer must write the entire prefill before backup")
        hot_groups = tuple(group for group in self.layout.groups if group.mode == "hot")
        if set(host_targets) != {group.name for group in hot_groups}:
            raise ValueError("Exactly one host backup target is required for each hot group")
        backups = {
            group.name: self._prefill_backup(lease, group, host_targets[group.name], state.logical)
            for group in hot_groups
        }
        host_ptrs = [
            tensor.data_ptr() for pool, _, _ in backups.values()
            for tensor in (*pool.k_buffer, *pool.v_buffer)
        ]
        if len(host_ptrs) != len(set(host_ptrs)):
            raise ValueError("Host backup groups and layer buffers must not alias")
        current = {group.name: len(self._pages.page_ids(lease, group.name)) for group in hot_groups}
        keep = {
            group.name: self._prefill_keep_pages(group, current[group.name])
            for group in hot_groups
        }
        self._pages.grow_prefill(lease, {name: max(current[name], keep[name]) for name in keep})
        try:
            for name, (pool, dst, src) in backups.items():
                pool.backup_from_device_all_layer(self._pools[name], dst, src, io_backend="kernel")
        except BaseException:
            if self._stream is not None:
                self._poisoned_at = "native prefill backup"
                raise
            with self._poison_on_error("prefill backup rollback"):
                self._fence_before_reuse()
                for name, pages in current.items():
                    self._pages.release_tail(lease, name, pages)
            raise
        with self._poison_on_error("prefill commit"):
            self._fence_before_reuse()
            for name in keep:
                self._unbind(name, state.logical)
            if keep:
                self._fence_before_reuse()
            for name, pages in keep.items():
                self._pages.release_tail(lease, name, pages)
            del self._prefill[lease]
            return {
                group.name: tuple(
                    self._pages.location(lease, group.layers[0], pos)
                    for pos in range(keep[group.name] * self.layout.page_tokens)
                )
                for group in hot_groups
            }

    def bind_logical_slots(self, lease, layer, positions):
        """Publish a set of logical address to private group-slot bindings.

        Call on the owning stream before the native K/V write and complete the
        write before scheduling any consumer. Bind once per group, not per layer.
        """
        self._require_eager_stream()
        self._require_decode_lease(lease)
        self._bind_logical_slots(lease, layer, positions)

    def _bind_logical_slots(self, lease, layer, positions):
        group = self._pages.route(layer).name
        bindings = self._bindings[group]
        # Validate the same live lease/group without copying all its page IDs.
        self._pages._request_pages(lease)[group]
        writers = self._physical_writers[group]
        pending = {}
        for logical, slot in positions.items():
            if type(logical) is not int or not self.layout.page_tokens <= logical < self._logical_slots:
                raise ValueError("Logical data slot is out of bounds or reserved for padding")
            if logical in bindings:
                raise ValueError("Logical slot is already bound in this group")
            pending[logical] = self._pages.location(lease, layer, slot, write=True)
        physical = set(pending.values())
        if len(physical) != len(pending) or any(
            writers[location >> 3] & (1 << (location & 7)) for location in physical
        ):
            raise ValueError("Multiple logical writers cannot own the same physical slot")
        if pending:
            with self._poison_on_error("logical binding"):
                mapping = self._maps[group]
                mapping[list(pending)] = mapping.new_tensor(list(pending.values()))
                bindings.update({logical: (lease, loc) for logical, loc in pending.items()})
                for location in physical:
                    writers[location >> 3] |= 1 << (location & 7)

    def _unbind(self, group, indices):
        # All callers own a poison boundary and fence before reusing pages.
        # Resolve through the authoritative forward bindings; a bit is not an
        # owner or a generation. Check the entire removal before publication.
        if indices:
            bindings = self._bindings[group]
            writers = self._physical_writers[group]
            locations = [bindings[logical][1] for logical in indices]
            if len(set(locations)) != len(locations) or any(
                not writers[location >> 3] & (1 << (location & 7)) for location in locations
            ):
                raise RuntimeError("Physical writer index disagrees with logical bindings")
            self._maps[group][indices] = -1
            for logical, location in zip(indices, locations):
                del bindings[logical]
                writers[location >> 3] &= ~(1 << (location & 7))

    def unbind_logical_slots(self, lease, layer, indices):
        """Retire writers before reusing slots such as the reserved latest position."""
        self._require_eager_stream()
        self._require_decode_lease(lease)
        group = self._pages.route(layer).name
        self._pages.page_ids(lease, group)
        indices = list(set(indices))
        if any(index not in self._bindings[group] or self._bindings[group][index][0] is not lease
               for index in indices):
            raise ValueError("Cannot unbind a foreign or unbound logical slot")
        with self._poison_on_error("logical retirement"):
            self._fence_before_reuse()
            self._unbind(group, indices)
            self._fence_before_reuse()

    def _fence_before_reuse(self):
        if self._stream is not None:
            torch.cuda.synchronize(self._device)

    def release_tail(self, lease, layer, keep_pages):
        """Retire one group's tail after the scheduler has stopped new consumers."""
        self._require_eager_stream()
        self._require_decode_lease(lease)
        group = self._pages.route(layer).name
        page_ids = self._pages.page_ids(lease, group)
        if type(keep_pages) is not int or not 0 <= keep_pages <= len(page_ids):
            raise ValueError("Retained page count is outside the allocated range")
        dropped = set(page_ids[keep_pages:])
        bindings = self._bindings[group]
        with self._poison_on_error("tail release"):
            self._fence_before_reuse()
            self._unbind(group, [
                logical for logical, (owner, slot) in bindings.items()
                if owner is lease and slot // self.layout.page_tokens in dropped
            ])
            self._fence_before_reuse()
            self._pages.release_tail(lease, group, keep_pages)

    def finish(self, lease):
        self._require_eager_stream()
        self._pages.page_ids(lease, self.layout.groups[0].name)
        with self._poison_on_error("request retirement"):
            self._fence_before_reuse()
            for group, bindings in self._bindings.items():
                self._unbind(group, [logical for logical, (owner, _) in bindings.items() if owner is lease])
            self._fence_before_reuse()
            self._pages.finish(lease)
            self._prefill.pop(lease, None)

    def get_kv_buffer(self, layer):
        """Main K/V getter. Keep the native index K/V getters on the wrapper."""
        self._require_eager_stream()
        group = self._pages.route(layer).name
        return self._pools[group].get_kv_buffer(self._local_layers[layer])

    def get_prefill_mapping(self, layer):
        """Read-only by contract. Prefill integration must bind its whole history."""
        self._require_eager_stream()
        return self._maps[self._pages.route(layer).name]

    def set_kv_buffer(self, layer, logical_locations, cache_k, cache_v):
        """Native BF16 main-K/V store given unwrapped logical write locations."""
        self._require_eager_stream()
        self._index_tensor(logical_locations, 1)
        self._check_range(logical_locations, 0, self._logical_slots, "Logical write slot")
        group = self._pages.route(layer.layer_id).name
        physical = self._maps[group][logical_locations]
        if torch.any(physical < 0).item():
            raise ValueError("Missing group-local logical write binding")
        data = physical[physical != 0]
        if torch.unique(data).numel() != data.numel():
            raise ValueError("A native write batch cannot target the same data slot twice")
        if any(tensor.shape != (physical.numel(), 1, self._head_dim)
               or tensor.dtype != torch.bfloat16 or tensor.device != self._device
               for tensor in (cache_k, cache_v)):
            raise ValueError("Main K/V write requires one BF16 head per token")
        layer_id = layer.layer_id
        progress = self._prefill_write_progress(layer_id, group, logical_locations)
        with self._poison_on_error("native K/V store"):
            self._pools[group].set_kv_buffer(
                layer, physical, cache_k, cache_v, layer_id_override=self._local_layers[layer_id]
            )
            for lease, count in progress.items():
                self._prefill[lease].written[layer_id] = count

    def _prefill_write_progress(self, layer, group, logical_locations):
        progress = {}
        if self._prefill:
            for logical in logical_locations.tolist():
                if logical < self.layout.page_tokens:
                    continue
                lease = self._bindings[group][logical][0]
                state = self._prefill.get(lease)
                if state is None:
                    continue
                count = progress.get(lease, state.written.get(layer, 0))
                if count >= len(state.logical) or logical != state.logical[count]:
                    raise ValueError("Prefill writes must follow the leased token order on every layer")
                progress[lease] = count + 1
        return progress

    def resolve_decode(self, layer, topk_blocks, req_rows, seq_lens, req_to_token, *, hot_state=None):
        """Return the native swap callback's flattened int32 positions with -1 padding."""
        self._require_eager_stream()
        for tensor, dims in ((topk_blocks, 2), (req_rows, 1), (seq_lens, 1), (req_to_token, 2)):
            self._index_tensor(tensor, dims)
        if (req_to_token.size(0) != self.layout.max_requests
                or req_to_token.size(1) < 1
                or req_rows.numel() < 1
                or topk_blocks.shape != (req_rows.numel(), self.layout.topk_blocks)
                or topk_blocks.dtype != torch.int32
                or seq_lens.shape != req_rows.shape):
            raise ValueError("Invalid decode batch geometry")
        self._check_range(req_rows, 0, self.layout.max_requests, "Request row")
        self._check_range(seq_lens, 1, req_to_token.size(1) + 1, "Sequence length")
        if torch.any(topk_blocks.to(torch.int64) >= 2**31 // self.layout.block_tokens).item():
            raise ValueError("Top-K block offset must fit native int32 token arithmetic")
        rows = req_rows.tolist()
        if len(rows) != len(set(rows)):
            raise ValueError("Eager decode requires distinct real request rows")
        leases = [self._pages.active_lease(row) for row in rows]
        for lease in leases:
            self._require_decode_lease(lease)
        group = self._pages.route(layer)
        if group.mode == "hot":
            if hot_state is None:
                raise RuntimeError("Hot layer requires group-local host and fused-swap state")
            return self._swap_hot(layer, group, topk_blocks, req_rows, seq_lens, leases, hot_state)
        if hot_state is not None:
            raise ValueError("Resident layer cannot accept host-backed state")
        offsets, valid = self._decode_offsets(topk_blocks, seq_lens)
        safe_offsets = torch.where(valid, offsets, 0)
        logical = req_to_token[req_rows[:, None], safe_offsets]
        logical = torch.where(valid, logical, 0)
        self._check_range(logical, 0, self._logical_slots, "Selected logical location")
        bindings = self._bindings[group.name]
        for lease, indices in zip(leases, logical.tolist()):
            if any(index and (index not in bindings or bindings[index][0] is not lease)
                   for index in indices):
                raise ValueError("A resident selection lacks this request's group-local binding")
        if torch.any(valid & (logical < self.layout.page_tokens)).item():
            raise ValueError("A valid resident selection points to padding")
        return torch.where(valid, self._maps[group.name][logical], -1).to(torch.int32)

    def _decode_offsets(self, topk_blocks, seq_lens):
        offsets = topk_blocks.to(torch.int64).unsqueeze(-1) * self.layout.block_tokens + self._block_offsets
        offsets = offsets.flatten(1)
        return offsets, (offsets >= 0) & (offsets < seq_lens[:, None])

    def _swap_hot(self, layer, group, topk_blocks, req_rows, seq_lens, leases, state):
        if self._device.type != "cuda":
            raise RuntimeError("Native fused HiSparse swap requires CUDA")
        if type(state) is not HotLayerState or not callable(state.wait_for_backup):
            raise ValueError("Expected explicit hot-layer state and host backup fence")
        with self._poison_on_error("hot backup fence"):
            state.wait_for_backup()
        self._require_eager_stream()
        h = group.hot_capacity
        validate_metadata_shapes(state.lru.shape, state.tokens.shape, state.locations.shape, h)
        validate_metadata_strides(state.lru.stride(), state.tokens.stride(), state.locations.stride())
        for tensor, dtype in (
            (state.tokens, torch.int32), (state.locations, torch.int32), (state.lru, torch.int16),
            (state.host_locations, torch.int64),
        ):
            if tensor.device != self._device or tensor.dtype != dtype or tensor.ndim != 2:
                raise ValueError("Invalid hot-layer metadata tensor")
            if tensor.size(0) != self.layout.max_requests or tensor.stride(1) != 1:
                raise ValueError("Invalid hot-layer metadata row dimensions")
        if torch.any(seq_lens > state.host_locations.size(1)).item():
            raise ValueError("Hot host-location row is shorter than the decode context")
        for tensor in (state.host_k, state.host_v):
            if (tensor.device.type != "cpu" or not tensor.is_pinned() or tensor.dtype != torch.bfloat16
                    or tensor.ndim != 3 or tensor.shape[1:] != (1, self._head_dim)
                    or not tensor.is_contiguous()):
                raise ValueError("Expected contiguous, pinned BF16 host K/V")
        if state.host_k.shape != state.host_v.shape:
            raise ValueError("Hot host K/V dimensions differ")
        self._validate_hot_indices(layer, group, topk_blocks, req_rows, seq_lens, leases, state)
        device_k, device_v = self.get_kv_buffer(layer)
        selected = torch.full(
            (req_rows.numel(), self.layout.topk_blocks * self.layout.block_tokens),
            -1, dtype=torch.int32, device=self._device,
        )
        with self._poison_on_error("native fused swap"):
            _native_fused_swap(
                top_k_blocks=topk_blocks.contiguous(),
                device_buffer_tokens=state.tokens, host_cache_locs=state.host_locations,
                device_buffer_locs=state.locations, host_cache_k=state.host_k, host_cache_v=state.host_v,
                device_buffer_k=device_k, device_buffer_v=device_v, top_k_device_locs=selected,
                req_pool_indices=req_rows.contiguous(), seq_lens=seq_lens.contiguous(),
                lru_slots=state.lru, item_size_bytes=2 * self._head_dim, hot_buffer_size=h,
                sparse_block_size=self.layout.block_tokens, page_size=1,
                block_size=960, skip_io=False,
            )
        return selected

    def _validate_hot_indices(self, layer, group, topk_blocks, req_rows, seq_lens, leases, state):
        h = group.hot_capacity
        expected_locations = [
            [self._pages.location(lease, layer, slot) for slot in range(min(length, h + 1))]
            for lease, length in zip(leases, seq_lens.tolist())
        ]
        for expected, actual in zip(expected_locations, state.locations[req_rows, :h + 1].tolist()):
            if actual[:len(expected)] != expected:
                raise ValueError("Hot location row does not match its group page owner")
        long_rows = seq_lens > h
        lru = state.lru[req_rows[long_rows]].sort(dim=1).values
        if torch.any(lru != torch.arange(h, device=self._device)).item():
            raise ValueError("Hot LRU must permute all active slots")
        tags = state.tokens[req_rows[long_rows], :h].sort(dim=1).values
        if torch.any(tags >= seq_lens[long_rows, None] - 1).item():
            raise ValueError("Hot LRU tags must reference history before the latest token")
        if torch.any((tags[:, 1:] >= 0) & (tags[:, 1:] == tags[:, :-1])).item():
            raise ValueError("Hot LRU cannot contain the same historical token twice")
        offsets, valid = self._decode_offsets(topk_blocks, seq_lens)
        historical = valid & long_rows[:, None] & (offsets < seq_lens[:, None] - 1)
        rows, selections = historical.nonzero(as_tuple=True)
        host_locations = state.host_locations[req_rows[rows], offsets[rows, selections]]
        self._check_range(host_locations, 0, state.host_k.size(0), "Selected historical host slot")
