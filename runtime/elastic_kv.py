"""Device state and transitions of elastic per-layer HiSparse (issue #13); torch only, no SGLang import.

Each pool layer ``i`` owns one combined buffer per K/V tensor from ``elastic_memory``: rows ``[0, hot_offset)`` are
the stock HiSparse hot rows (row r is device row r) and row ``hot_offset + L`` holds logical location L of the
resident region. One pointer and one row index therefore address either mode, which is what lets a single CUDA
Graph (smooth mode) switch a layer by rewriting a device route table, and lets the steady mode hand stock kernels
the hot view or the resident view of the same reservation.

Invariants kept here:

* Write-through: every managed layer's host history is complete up to the previous decode token, whichever mode
  wrote it (``written`` is the mode of the last forward), so offloading a layer moves no bytes to host. A
  transition first flushes the newest record (the one write-through has not reached) from the rows that hold it.
* Offload: the layer's hot rows get exactly what its swap table lists (positions 0..H-1 of a request as stock
  staging leaves them, or the LRU state of an earlier offloaded period), copied device-to-device from the
  resident rows. A DSA shared-index follower uses its anchor's table, which the anchor's kernel keeps current.
* Backfill: the resident rows are refilled from host (paced in smooth mode), skipping rows still valid in the
  layer's stale resident region, then the newest token, then the mode flips.
* Stale rows (smooth): after an offload, position p of a live request may still be read from row hot_offset + L
  while p < stale_limit[layer, slot] and L < the backed rows reported by ``ElasticMemory.stale_rows``.
"""
from contextlib import nullcontext

OFFLOADED, RESIDENT, FLUSHED = 0, 1, 2      # FLUSHED: the newest record already went to host at a transition


class MHALayers:
    """MiniMax: per-layer combined K and V buffers [rows, heads, dim]; host is the HiSparse MHA host pool."""
    kind = "mha"

    def __init__(self, torch, k, v, host):
        self.torch, self.k, self.v, self.host = torch, list(k), list(v), host
        # per-token bytes of one layer's K (= V) row, as the stock host pool passes them to the kernels
        self.item = getattr(host, "token_stride_size", None) or self.k[0].stride(0) * self.k[0].element_size()

    def d2d(self, i, src, dst):
        self.k[i][dst] = self.k[i][src]
        self.v[i][dst] = self.v[i][src]

    def _host(self, i, src, dst, to_host):
        hk, hv = self.host.k_buffer[i], self.host.v_buffer[i]
        pairs = ((self.k[i], hk), (self.v[i], hv)) if to_host else ((hk, self.k[i]), (hv, self.v[i]))
        if self.k[i].is_cuda:
            from sgl_kernel.kvcacheio import transfer_kv_per_layer
            (sk, dk), (sv, dv) = pairs
            transfer_kv_per_layer(src_k=sk, dst_k=dk, src_v=sv, dst_v=dv, src_indices=src, dst_indices=dst,
                                  item_size=self.item)
            return
        for source, target in pairs:
            target[dst] = source[src]

    def d2h(self, i, src, host_rows):
        self._host(i, src, host_rows, True)

    def h2d(self, i, host_rows, dst):
        self._host(i, host_rows, dst, False)


class MLALayers:
    """DSA (GLM-5.x, DeepSeek-V3.2): per-layer combined latent buffers [rows, 1, kv_cache_dim]; host is the MLA
    host pool in layer-first layout."""
    kind = "mla"

    def __init__(self, torch, kv, host):
        self.torch, self.kv, self.host = torch, list(kv), host
        self.item = getattr(host, "token_stride_size", None) or self.kv[0].stride(0) * self.kv[0].element_size()

    def d2d(self, i, src, dst):
        self.kv[i][dst] = self.kv[i][src]

    def _host(self, i, src, dst, to_host):
        pair = (self.kv[i], self.host.kv_buffer[i]) if to_host else (self.host.kv_buffer[i], self.kv[i])
        if self.kv[i].is_cuda:
            from sgl_kernel.kvcacheio import transfer_kv_per_layer_mla
            transfer_kv_per_layer_mla(src=pair[0], dst=pair[1], src_indices=src, dst_indices=dst,
                                      item_size=self.item)
            return
        pair[1][dst] = pair[0][src]

    def d2h(self, i, src, host_rows):
        self._host(i, src, host_rows, True)

    def h2d(self, i, host_rows, dst):
        self._host(i, host_rows, dst, False)


class C4Layers:
    """DeepSeek-V4 C4: per-layer combined page buffers [pages, 37440] uint8 (64 records of 576 B value + 8 B
    scale); row r is record r % 64 of page r // 64. Host is the C4 paged host pool."""
    kind = "c4"
    PAGE, VALUE, SCALE = 64, 576, 8

    def __init__(self, torch, pages, host):
        self.torch, self.pages, self.host = torch, list(pages), host
        self.ptrs = {}                       # data pointer -> one-element device tensor for the copy kernel

    def _parts(self, buffer):
        pages = buffer.view(-1, buffer.shape[-1])
        cut = self.PAGE * self.VALUE
        return (pages[:, :cut].unflatten(1, (self.PAGE, self.VALUE)),
                pages[:, cut:cut + self.PAGE * self.SCALE].unflatten(1, (self.PAGE, self.SCALE)))

    def _ptr(self, tensor, device, table=None, index=None):
        """One-element pointer tensor of a buffer for the copy kernel: the host pool's kernel pointer table entry
        when it has one (pinned host memory), else the raw data pointer."""
        if table is not None:
            return table[index:index + 1]
        key = tensor.data_ptr()
        if key not in self.ptrs:
            self.ptrs[key] = self.torch.tensor([key], dtype=self.torch.uint64, device=device)
        return self.ptrs[key]

    def _copy(self, source, src, target, dst, src_table=None, dst_table=None, i=None):
        if source.is_cuda or target.is_cuda:
            from sglang.kernels.ops.kvcache.hisparse import transfer_cache_dsv4_mla
            device = src.device
            transfer_cache_dsv4_mla(self._ptr(source, device, src_table, i), self._ptr(target, device, dst_table, i),
                                    src.long(), dst.long())
            return
        for part_s, part_t in zip(self._parts(source), self._parts(target)):
            part_t[dst // self.PAGE, dst % self.PAGE] = part_s[src // self.PAGE, src % self.PAGE]

    def d2d(self, i, src, dst):
        self._copy(self.pages[i], src, self.pages[i], dst)

    def d2h(self, i, src, host_rows):
        self._copy(self.pages[i], src, self.host.kv_buffer[i], host_rows,
                   dst_table=getattr(self.host, "data_ptrs", None), i=i)

    def h2d(self, i, host_rows, dst):
        self._copy(self.host.kv_buffer[i], host_rows, self.pages[i], dst,
                   src_table=getattr(self.host, "data_ptrs", None), i=i)


class ElasticKV:
    """Modes, route and stale tables, live requests and the device side of every transition for one rank.

    ``coordinator`` is the stock HiSparseCoordinator (or a test double with the same tables). ``hot_offset[i]``
    is the first resident row of layer i's combined buffer, ``managed`` the model layers the config may offload,
    ``owner`` maps a pool index to the pool index whose swap table it uses (DSA shared-index followers ->
    anchor). ``through`` is ``features.write_through``: without it resident layers skip the per-step host copy
    and an offload flushes their whole history instead. ``mapping`` is the allocator's logical-to-hot location
    table (smooth stores of offloaded layers).
    """

    def __init__(self, rows, coordinator, *, layers, managed, hot_offset, offloaded, torch, device, ratio=1,
                 owner=None, through=True, mapping=None):
        self.rows, self.c, self.torch, self.device, self.ratio = rows, coordinator, torch, device, ratio
        self.layers = tuple(layers)                           # model layer id of each pool index
        self.index = {layer: i for i, layer in enumerate(self.layers)}
        self.managed = frozenset(self.index[layer] for layer in managed)
        self.hot_offset = list(hot_offset)
        self.owner = dict(owner or {})
        self.through, self.mapping = through, mapping
        self.fused = None                    # runtime.elastic_route once a worker bound the route kernels
        self.fills = []                      # PD admissions still copying host -> resident rows: (event, req)
        n, slots = len(self.layers), coordinator.req_to_token_pool.req_to_token.shape[0]
        self.mode = [OFFLOADED if layer in offloaded else RESIDENT for layer in self.layers]
        self.written = list(self.mode)
        self.route = torch.tensor([[m, h, 0, 0] for m, h in zip(self.mode, self.hot_offset)], dtype=torch.int32,
                                  device=device)
        self.stale_limit = torch.zeros((n, slots), dtype=torch.int32, device=device)
        self.live = {}                       # slot -> records on the device, the newest decode record included
        self.hosted = {}                     # slot -> records whose host copy is complete (write-through lags 1)
        self.filled = {}                     # pool index -> {slot: records refilled} while a backfill runs

    # ---- queries -----------------------------------------------------------------------------------------
    def _long(self, values):
        return self.torch.as_tensor(values, dtype=self.torch.int64, device=self.device)

    def logical_rows(self, slot, start, stop):
        """Logical rows of records start..stop-1 of a slot (a C4 record's row is its closing token's loc / 4)."""
        r = self.ratio
        tokens = self.c.req_to_token_pool.req_to_token[slot, start * r:stop * r]
        return tokens[r - 1::r].long() // r

    def rows_of(self, slot, positions):
        """Logical rows of arbitrary record positions of one slot."""
        r = self.ratio
        return self.c.req_to_token_pool.req_to_token[slot, (positions + 1) * r - 1].long() // r

    def resident_rows(self, i, slot, start, stop):
        return self.logical_rows(slot, start, stop) + self.hot_offset[i]

    def _stream(self, stream):
        return self.torch.cuda.stream(stream) if self.device.type == "cuda" and stream is not None else nullcontext()

    # ---- request lifecycle -------------------------------------------------------------------------------
    def admitted(self, slot, records):
        """A request whose ``records`` prompt records are on the device and complete in host."""
        self.live[slot] = self.hosted[slot] = records
        self.stale_limit[:, slot] = 0

    def decode_begins(self, slots, records):
        """The coming forward leaves ``records[i]`` records of ``slots[i]`` on the device."""
        for slot, count in zip(slots, records):
            self.live[slot] = count

    def finished(self, slot):
        self.live.pop(slot, None)
        self.hosted.pop(slot, None)
        self.stale_limit[:, slot] = 0
        for progress in self.filled.values():
            progress.pop(slot, None)

    # ---- write-through -----------------------------------------------------------------------------------
    def write_through(self, slots, records, host_rows):
        """Back the previous decode record of ``slots`` up to host for every managed layer, from the rows the
        last forward wrote: the request's device-buffer slot (offloaded) or its logical row (resident). Runs
        on the decode backup stream after the forward that produced the records, like the stock backup.
        Layers a transition already flushed skip this record; without ``through`` resident layers skip it."""
        c, t = self.c, self.torch
        cuda = self.device.type == "cuda"
        copy = [i for i in sorted(self.managed)
                if self.written[i] == OFFLOADED or (self.written[i] == RESIDENT and self.through)]
        slots_t, records_t = self._long(slots), self._long(records)
        host_t = host_rows.to(self.device).long()
        hot = c.req_to_device_buffer[slots_t, records_t.clamp(max=c.device_buffer_size)].long()
        logical = c.req_to_token_pool.req_to_token[slots_t, (records_t + 1) * self.ratio - 1].long() // self.ratio
        if cuda:
            c.wait_for_pending_backup()
            schedule = t.cuda.current_stream()
        with self._stream(c.decode_backup_stream):
            if cuda:
                c.decode_backup_stream.wait_stream(schedule)
                if c.decode_producer_stream is not None:
                    c.decode_backup_stream.wait_stream(c.decode_producer_stream)
            for i in copy:
                src = hot if self.written[i] == OFFLOADED else logical + self.hot_offset[i]
                self.rows.d2h(i, src, host_t)
            if cuda:
                c._backup_done_event.record()
                for tensor in (slots_t, records_t, host_t, hot, logical):
                    tensor.record_stream(c.decode_backup_stream)
        if cuda:
            c._has_pending_backup = True
        for slot, record in zip(slots, records):
            self.hosted[slot] = max(self.hosted.get(slot, 0), record + 1)

    def forward_begins(self):
        """The coming forward writes with the current modes."""
        self.written = list(self.mode)

    def stage(self, slot, host_rows, hot_rows):
        """Prefill backup of every managed layer of one request from the rows its prefill wrote (resident layers
        too: write-through starts from a complete host copy)."""
        n = len(hot_rows)
        logical = self.logical_rows(slot, 0, n)
        for i in sorted(self.managed):
            self.rows.d2h(i, hot_rows if self.mode[i] == OFFLOADED else logical + self.hot_offset[i], host_rows)

    def fill_request(self, slot, records, host_rows, *, stream=None, req=None):
        """PD decode admission: copy a request's records from host into every resident layer's logical rows (KV
        lands in host for all HiSparse layers, as in the stock direct admission). On CUDA the copies run on
        ``stream`` and the request waits in ``fills`` until ``ready`` reports them done."""
        t = self.torch
        cuda = self.device.type == "cuda" and stream is not None
        if records:
            logical = self.logical_rows(slot, 0, records)
            host = host_rows[:records].long()
            if cuda:
                start = t.cuda.Event()
                start.record()
            with self._stream(stream if cuda else None):
                if cuda:
                    stream.wait_event(start)
                for i, m in enumerate(self.mode):
                    if m == RESIDENT:
                        self.rows.h2d(i, host, logical + self.hot_offset[i])
                if cuda:
                    done = t.cuda.Event()
                    done.record(stream)
                    for tensor in (logical, host):
                        tensor.record_stream(stream)
            if cuda:
                self.fills.append((done, req))
        self.admitted(slot, records)
        return records

    def ready(self):
        """How many PD fills at the head of ``fills`` finished (the caller agrees on a count across ranks)."""
        done = 0
        for event, _ in self.fills:
            if not event.query():
                break
            done += 1
        return done

    def preload(self, host_rows, device_rows):
        """PD admission of a request no longer than H: its records into the hot rows of every offloaded layer
        (the stock kernel's short fast path reads them there)."""
        for i, m in enumerate(self.mode):
            if m == OFFLOADED:
                self.rows.h2d(i, host_rows.long(), device_rows.long())

    def resident_page_table(self, req_pool_indices, top_k, seq_lens=None, offset=0):
        """Decode page table of a resident layer (graph-safe): the logical rows of the selected records (+
        ``offset``, the resident region start in the combined buffer), -1 for padded selections and, when
        ``seq_lens`` (records) is given, for selections past a request's length. One kernel of ``fused`` when
        bound."""
        if self.fused is not None and seq_lens is not None:
            return self.fused.resident(self, req_pool_indices, top_k, seq_lens, offset)
        t, r = self.torch, self.ratio
        req_to_token = self.c.req_to_token_pool.req_to_token
        safe = top_k.long().clamp(min=0, max=req_to_token.shape[1] // r - 1)
        rows = req_to_token[req_pool_indices.long()[:, None], (safe + 1) * r - 1].long() // r
        valid = top_k >= 0
        if seq_lens is not None:
            valid &= top_k.long() < seq_lens.long()[:, None]
        return t.where(valid, rows + offset, -1).to(t.int32)

    # ---- transitions -------------------------------------------------------------------------------------
    def _requests(self):
        c = self.c
        for slot in sorted(self.live):
            yield slot, self.live[slot], int(c.req_device_buffer_size[slot])

    def warm(self, indices, source="device", long_hits=True):
        """Hot rows of newly offloaded layers get what their swap table lists: every hot slot holding position
        p < n of a live request receives record p, and the reserved slot H of a request longer than H its newest
        record (the swap-in kernel reads it there until the next record closes, up to 3 steps later for C4).
        Copies come from the resident rows (``source="device"``) or from host (when those rows already lost
        their chunks; the caller flushed the newest records first). Tables stay as they are: alloc_device_buffer
        leaves positions 0..H-1 (all-miss for PD requests longer than H) and an earlier offloaded period leaves
        its LRU state, both consistent with the rows copied here. With ``long_hits`` off, a layer owning its table
        restarts requests longer than H all-miss (only the newest record is copied)."""
        c, t, moved = self.c, self.torch, 0
        hot_slots = c.device_buffer_size
        for slot, n, _ in self._requests():
            for i in indices:
                o = self.owner.get(i, i)
                if not long_hits and o == i and n > hot_slots:
                    c.req_device_buffer_tokens[i, slot, :hot_slots] = -1
                    c.lru_slots[i, slot] = t.arange(hot_slots, dtype=c.lru_slots.dtype, device=self.device)
                pos = t.cat((c.req_device_buffer_tokens[o, slot, :hot_slots].long(),
                             self._long([n - 1 if n > hot_slots else -1])))
                dst = c.req_device_buffer_token_locs[o, slot, :hot_slots + 1].long()
                keep = (pos >= 0) & (pos < n) & (dst > 0)
                pos, dst = pos[keep], dst[keep]
                if not pos.numel():
                    continue
                if source == "device":
                    self.rows.d2d(i, self.rows_of(slot, pos) + self.hot_offset[i], dst)
                else:
                    self.rows.h2d(i, c.req_to_host_pool[slot, pos].long(), dst)
                moved += int(pos.numel())
        return moved

    def begin_fill(self, indices):
        for i in indices:
            self.filled.setdefault(i, {})

    def fill(self, indices, budget=None, stale_rows=None):
        """Copy host history into the resident rows of backfilling layers, at most ``budget`` records per layer;
        records still valid in a stale region (below ``stale_limit`` and ``stale_rows[i]``) are skipped. Returns
        the records still to copy (0 when every request's host history is in place; ``catch_up`` then copies
        the newest decode record, which is not in host yet unless ``flush_newest`` put it there)."""
        c, left = self.c, 0
        for i in indices:
            progress, spent = self.filled.setdefault(i, {}), 0
            limit_row = (stale_rows or {}).get(i, 0)
            for slot in sorted(self.live):
                n = self.live[slot] if self.written[i] == FLUSHED else self.hosted.get(slot, 0)
                done = progress.get(slot, 0)
                stop = n if budget is None else min(n, done + max(0, budget - spent))
                if stop > done:
                    host = c.req_to_host_pool[slot, done:stop].long()
                    dst = self.resident_rows(i, slot, done, stop)
                    keep = int(self.stale_limit[i, slot])
                    if keep > done and limit_row:
                        valid = (self.torch.arange(done, stop, device=self.device) < keep) & \
                                (dst - self.hot_offset[i] < limit_row)
                        host, dst = host[~valid], dst[~valid]
                    if dst.numel():
                        self.rows.h2d(i, host, dst)
                    spent += stop - done
                    progress[slot] = stop
                left += n - progress.get(slot, 0)
        return left

    def catch_up(self, indices):
        """Device-to-device copy of the records the last forward wrote to the hot rows of backfilling layers but
        write-through has not backed up yet; afterwards they read as resident (``written``)."""
        c = self.c
        for slot in sorted(self.live):
            start, stop = self.hosted.get(slot, 0), self.live[slot]
            if stop <= start:
                continue
            records = self._long(list(range(start, stop)))
            hot = c.req_to_device_buffer[slot, records.clamp(max=c.device_buffer_size)].long()
            for i in indices:
                self.rows.d2d(i, hot, self.resident_rows(i, slot, start, stop))
        for i in indices:
            self.written[i] = RESIDENT

    def flush_newest(self, indices):
        """Back up now, for layers about to change mode, the records host lacks, from the rows the last forward
        wrote: the newest record (write-through lags one step), or the whole history of a resident layer that
        does not write through. Their host copy is then complete before any of their rows lose chunks, and
        they skip the next write-through (``FLUSHED``)."""
        c = self.c
        for slot in sorted(self.live):
            stop = self.live[slot]
            for i in indices:
                if self.written[i] == FLUSHED:
                    continue
                start = 0 if self.written[i] == RESIDENT and not self.through else self.hosted.get(slot, 0)
                if stop <= start:
                    continue
                host = c.mem_pool_host.alloc_paged_token_slots(c.req_to_host_pool, c.req_to_host_pool_allocated_len,
                                                               slot, start, stop - start).to(self.device).long()
                if self.written[i] == OFFLOADED:
                    records = self._long(list(range(start, stop)))
                    src = c.req_to_device_buffer[slot, records.clamp(max=c.device_buffer_size)].long()
                else:
                    src = self.resident_rows(i, slot, start, stop)
                self.rows.d2h(i, src, host)
        for i in indices:
            self.written[i] = FLUSHED

    def set_modes(self, offloaded):
        """Publish a new offload set: modes, route table and stale limits (newly offloaded layers may read the
        records live requests had written; newly resident ones have no stale region)."""
        t = self.torch
        newly = []
        for i, layer in enumerate(self.layers):
            m = OFFLOADED if layer in offloaded else RESIDENT
            if m == OFFLOADED and self.mode[i] == RESIDENT:
                newly.append(i)
            if m == RESIDENT:
                self.stale_limit[i].zero_()
                self.filled.pop(i, None)
            self.mode[i] = m
        for i in newly:
            self.stale_limit[i].zero_()
            for slot, n, _ in self._requests():
                self.stale_limit[i, slot] = n
        self.route[:, 0] = t.tensor(self.mode, dtype=t.int32, device=self.device)
        return newly

    def publish_stale(self, rows_by_index):
        """Backed rows of each offloaded layer's stale region (from ElasticMemory.stale_rows)."""
        for i, m in enumerate(self.mode):
            self.route[i, 2] = int(rows_by_index.get(i, 0)) if m == OFFLOADED else 0

    def store_rows(self, i, loc):
        """Smooth mode decode store rows of layer i (graph-safe): the device-buffer row (``mapping``) when
        offloaded, else hot_offset + logical location; a negative location (a C4 token that closes no record)
        stays negative. One kernel of ``fused`` when bound."""
        if self.fused is not None:
            return self.fused.store_rows(self, i, loc)
        t = self.torch
        rows = t.where(self.route[i, 0] == OFFLOADED, self.mapping[loc], loc + self.route[i, 1])
        return t.where(loc < 0, loc, rows)

    def gate(self, i, num_real_reqs):
        """Request count a planned copy into layer i replays (graph-safe): ``num_real_reqs`` while the layer is
        offloaded, else 0. One kernel of ``fused`` when bound."""
        if self.fused is not None:
            return self.fused.gate(self, i, num_real_reqs)
        return num_real_reqs * (self.route[i, 0:1] == OFFLOADED).to(num_real_reqs.dtype)

    def record_bytes(self):
        """Bytes one record of one layer moves (pacing of smooth backfill)."""
        return dict(mha=2 * getattr(self.rows, "item", 0), mla=getattr(self.rows, "item", 0), c4=584)[self.rows.kind]

    def smooth_select(self, i, req_pool_indices, seq_lens, top_k, block=None):
        """Smooth mode, before the stock swap-in of layer i (graph-safe, no host sync): (masked selection, rows,
        reads). Selected records the layer reads from its resident region -- all of them in resident mode,
        the still-valid stale ones in offloaded mode -- become -1 for the stock kernel and resolve to
        hot_offset + logical row instead. ``block`` is MiniMax's sparse block size (selection by block ids).
        Runs the kernels of ``elastic_route`` once a worker bound them (``fused``)."""
        if self.fused is not None:
            return self.fused.select(self, i, req_pool_indices, seq_lens, top_k, block=block)
        return self.reference_select(i, req_pool_indices, seq_lens, top_k, block=block)

    def merge(self, swapped, rows, reads):
        """Combined-buffer rows of every selected record: resident or stale reads, else the stock swap result."""
        if self.fused is not None:
            return self.fused.merge(self.torch, swapped, rows, reads)
        return self.reference_merge(swapped, rows, reads)

    def reference_select(self, i, req_pool_indices, seq_lens, top_k, block=None):
        """The torch statement of ``smooth_select`` (CPU, and the reference of the fused kernels)."""
        t, r = self.torch, self.ratio
        req_to_token = self.c.req_to_token_pool.req_to_token
        rid = req_pool_indices.long()[:, None]
        if block is None:
            pos = top_k.long()
        else:
            pos = (top_k.long()[:, :, None] * block + t.arange(block, device=top_k.device)).flatten(1)
            pos = t.where((top_k < 0).repeat_interleave(block, 1), -1, pos)
        in_range = (pos >= 0) & (pos < seq_lens.long()[:, None])
        safe = pos.clamp(min=0, max=req_to_token.shape[1] // r - 1)
        rows = req_to_token[rid, (safe + 1) * r - 1].long() // r
        resident = self.route[i, 0] == RESIDENT
        stale = (pos < self.stale_limit[i, req_pool_indices.long()].long()[:, None]) & (rows < self.route[i, 2])
        reads = in_range & (resident | stale)
        if block is not None:
            whole = (reads | ~in_range).view(reads.shape[0], -1, block).all(-1)
            reads = whole.repeat_interleave(block, 1) & in_range
            masked = t.where(whole, -1, top_k)
        else:
            masked = t.where(reads, -1, top_k)
        return masked, rows + self.route[i, 1], reads

    def reference_merge(self, swapped, rows, reads):
        return self.torch.where(reads.bool(), rows.to(swapped.dtype), swapped)

    # ---- compaction --------------------------------------------------------------------------------------
    def compact(self, moves, page, buffers, mapping=None, movers=()):
        """Move live logical pages ``moves`` [(old, new)] below a shrink ceiling: copy their rows in every backed
        logical buffer (``buffers``: (tensor, first resident row) pairs; ``movers``: stock relocations
        ``move(dst_locs, src_locs)`` of the pools this module does not own, e.g. indexer caches), rewrite
        req_to_token and the logical-to-hot mapping. Host rows are indexed per request, so they do not move."""
        if not moves:
            return 0
        t = self.torch
        old = self._long([o for o, _ in moves])
        new = self._long([n for _, n in moves])
        offsets = t.arange(page, device=self.device)
        src = (old[:, None] * page + offsets).reshape(-1)
        dst = (new[:, None] * page + offsets).reshape(-1)
        for tensor, base in buffers:
            tensor[dst + base] = tensor[src + base]
        for move in movers:
            move(dst, src)
        req_to_token = self.c.req_to_token_pool.req_to_token
        remap = t.arange(int(old.max()) + 1, device=self.device)
        remap[old] = new
        pages = req_to_token.long() // page
        hit = pages <= int(old.max())
        moved = t.where(hit, remap[pages.clamp(max=int(old.max()))] * page + req_to_token.long() % page,
                        req_to_token.long())
        req_to_token.copy_(moved.to(req_to_token.dtype))
        if mapping is not None:
            mapping[dst] = mapping[src]
            mapping[src] = 0
        return len(moves)
