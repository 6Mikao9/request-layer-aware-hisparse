"""CPU tests of elastic per-layer HiSparse on the device side: ElasticKV and the Runtime transitions over a CPU
ElasticMemory, with the DSA coordinator hooks installed on a restated stock coordinator (tables, LRU swap-in
kernel, shared-index miss plans). Every attention read is checked against the value each (request, layer,
position) was written with, across offloads, backfills, compaction, PD admission and both modes."""
import contextlib
import io
import os
import random
import types
import unittest
import unittest.mock
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from . import elastic_config as ec
from . import elastic_memory as em
if torch is not None:
    from . import elastic_device as dev
    from . import elastic_kv as ekv

PAGE, H, SLOTS, CTX, SIZE, RATIO, CHUNK, DIM, TOP_K = 4, 8, 6, 160, 64, 4, 64, 4, 6
LAYERS = (0, 1, 2, 3)
SRC = os.environ.get("SGLANG_SRC")
FIXTURES = Path(__file__).parent / "fixtures" / "elastic"


def value(uid, layer, pos):
    return [float(uid), float(layer), float(pos), 1.0]


class Paged:
    """The stock paged allocator's free-list fields (pages 1..num_pages; page 0 is padding)."""

    def __init__(self, size, page):
        self.page_size, self.num_pages = page, size // page
        self.clear()

    def clear(self):
        self.free_pages = torch.arange(1, self.num_pages + 1, dtype=torch.int64)
        self.staged_pages, self.num_staged_pages = [], 0

    def available_size(self):
        return (len(self.free_pages) + self.num_staged_pages) * self.page_size

    def take(self, pages):
        if pages > len(self.free_pages) and self.staged_pages:
            self.free_pages = torch.sort(torch.cat((self.free_pages, *self.staged_pages)))[0]
            self.staged_pages, self.num_staged_pages = [], 0
        if pages > len(self.free_pages):
            raise MemoryError("allocator exhausted")
        out, self.free_pages = self.free_pages[:pages], self.free_pages[pages:]
        return out

    def release(self, pages):
        if len(pages):
            self.staged_pages.append(pages.clone())
            self.num_staged_pages += len(pages)


class Host:
    """Layer-first host pool: per-layer rows and the stock page-granular per-request slot allocation."""

    def __init__(self, rows):
        self.kv_buffer = [torch.zeros((rows, 1, DIM)) for _ in LAYERS]
        self.next = 0

    def alloc_paged_token_slots(self, req_to_host_pool, allocated_len, slot, start, count):
        end, have = start + count, int(allocated_len[slot])
        assert start <= have
        page_end = -(-end // PAGE) * PAGE
        if page_end > have:
            req_to_host_pool[slot, have:page_end] = torch.arange(self.next, self.next + page_end - have)
            self.next += page_end - have
            allocated_len[slot] = page_end
        return req_to_host_pool[slot, start:end]


def planned_copy_mla(*, miss_src, miss_dst, miss_count, num_real_reqs, host_cache, device_buffer, skip_io=False,
                     **_):
    """copy_cache_planned_mla: replay a recorded miss plan for the first num_real_reqs[0] requests."""
    for r in range(int(num_real_reqs[0])):
        for m in range(int(miss_count[r])):
            if not skip_io:
                device_buffer[int(miss_dst[r, m])] = host_cache[int(miss_src[r, m])]


def coordinator_class():
    """A fresh restatement of the stock DSA HiSparseCoordinator methods the hooks wrap (one class per test so
    patches never leak): shared-index prefetch groups, the swap-in kernel with LRU, skip_io and plan record."""

    class Coordinator:
        def swap_in_selected_pages(self, req_pool_indices, compressed_seq_lens, top_k_result, layer_id):
            if not self.enable_prefetch:
                return self._run_swap_in_kernel(req_pool_indices, compressed_seq_lens, top_k_result, layer_id)
            n = req_pool_indices.size(0)
            if self._is_shared_index_layer[layer_id]:
                return self.top_k_device_locs_buffer[:n, :TOP_K]
            group = self._prefetch_groups.get(layer_id)
            out = self._run_swap_in_kernel(req_pool_indices, compressed_seq_lens, top_k_result, layer_id,
                                           record_plan=group is not None)
            for skip in group or ():
                self._run_copy_only_kernel(n, skip)
            return out

        def _run_copy_only_kernel(self, num_reqs, skip_layer):
            planned_copy_mla(miss_src=self._miss_src[:num_reqs], miss_dst=self._miss_dst[:num_reqs],
                             miss_count=self._miss_count[:num_reqs], num_real_reqs=self.num_real_reqs,
                             host_cache=self.mem_pool_host.kv_buffer[skip_layer],
                             device_buffer=self.mem_pool_device.kv_buffer[skip_layer], skip_io=self.skip_io)

        def _run_swap_in_kernel(self, req_pool_indices, compressed_seq_lens, top_k_result, layer_id,
                                record_plan=False):
            """Hits come from the slot table; misses take the least recently used slot no selected record holds
            and load from host. The newest record of a request longer than H sits in the reserved slot H; a
            request no longer than H reads its records position-indexed (the short fast path)."""
            n = req_pool_indices.size(0)
            out = self.top_k_device_locs_buffer[:n, :TOP_K]
            hot, host = self.mem_pool_device.kv_buffer[layer_id], self.mem_pool_host.kv_buffer[layer_id]
            for r, (slot, seq) in enumerate(zip(req_pool_indices.tolist(), compressed_seq_lens.tolist())):
                tokens = self.req_device_buffer_tokens[layer_id, slot]
                locs = self.req_device_buffer_token_locs[layer_id, slot]
                lru, picked, misses = self.lru[layer_id][slot], set(top_k_result[r].tolist()), 0
                for k, p in enumerate(top_k_result[r].tolist()):
                    if p < 0 or seq <= H or p == seq - 1:
                        out[r, k] = -1 if p < 0 else locs[min(p, H)] if seq > H else locs[p]
                        continue
                    hit = (tokens[:H] == p).nonzero()
                    s = int(hit[0]) if len(hit) else next(s for s in lru if int(tokens[s]) not in picked)
                    if not len(hit):
                        tokens[s] = p
                        src, row = int(self.req_to_host_pool[slot, p]), int(locs[s])
                        self.misses += 1
                        if not self.skip_io:
                            hot[row] = host[src]
                        if record_plan:
                            self._miss_src[r, misses], self._miss_dst[r, misses] = src, row
                            misses += 1
                    lru.remove(s)
                    lru.append(s)
                    out[r, k] = locs[s]
                if record_plan:
                    self._miss_count[r] = misses
            return out

        def wait_for_pending_backup(self):
            pass

        def has_ongoing_staging(self):
            return False

    Coordinator.__module__ = __name__
    for name in ("swap_in_selected_pages", "_run_copy_only_kernel", "_run_swap_in_kernel"):
        getattr(Coordinator, name).__module__ = __name__
    return Coordinator


def make_config(mode, groups, always=(), **sections):
    data = {"schema": ec.SCHEMA, "model": "dsa", "mode": mode,
            "layers": {"eviction_order": [list(g) for g in groups], "always_offload": list(always)},
            "controller": {"interval_steps": 1, "dwell_steps": 1, "horizon_steps": 1}}
    for key, extra in sections.items():
        data[key] = {**data.get(key, {}), **extra}
    return ec.parse(data)


class Rig:
    """One rank: the elastic pool (CPU VMM backend), a restated coordinator with the DSA hooks installed, and
    request bookkeeping of the scheduler. ``expected`` holds the value of every (uid, layer, position).
    ``size`` is the stock HiSparse device pool (hot rows; the chunk pool holds exactly its bytes)."""

    def __init__(self, case, mode="steady", groups=((0, 1), (2, 3)), always=(), shared=None, size=SIZE,
                 **sections):
        self.case, self.mode, self.rng = case, mode, random.Random(7)
        cfg = make_config(mode, groups, always, **sections)
        followers = sorted({f for g in (shared or {}).values() for f in g})
        state = dict(backend=em.CpuBackend(torch), patched=[], runtime=None)
        factory = dev.Factory(state, torch, size=size, ratio=RATIO, page=PAGE, layers=LAYERS, chunk_bytes=CHUNK)
        hot = [factory.buffer("kv", layer, (1, DIM), torch.float32) for layer in LAYERS]
        chunks = factory.seal()
        self.c = c = coordinator_class()()
        width, slots = H + PAGE, SLOTS
        c.req_to_token_pool = types.SimpleNamespace(req_to_token=torch.zeros((slots, CTX), dtype=torch.int32))
        c.req_to_device_buffer = torch.zeros((slots, width), dtype=torch.int64)
        c.req_device_buffer_size = torch.zeros(slots, dtype=torch.int64)
        c.req_to_host_pool = torch.full((slots, CTX + PAGE), -1, dtype=torch.int64)
        c.req_to_host_pool_allocated_len = torch.zeros(slots, dtype=torch.int64)
        c.req_device_buffer_tokens = torch.full((len(LAYERS), slots, width), -1, dtype=torch.int32)
        c.req_device_buffer_token_locs = torch.full((len(LAYERS), slots, width), -1, dtype=torch.int32)
        c.lru_slots = torch.arange(H, dtype=torch.int16).repeat(len(LAYERS), slots, 1)
        c.lru = [[list(range(H)) for _ in range(slots)] for _ in LAYERS]
        c.top_k_device_locs_buffer = torch.full((slots, TOP_K), -1, dtype=torch.int32)
        c._miss_src = torch.zeros((slots, TOP_K), dtype=torch.int64)
        c._miss_dst = torch.zeros((slots, TOP_K), dtype=torch.int32)
        c._miss_count = torch.zeros(slots, dtype=torch.int32)
        c.num_real_reqs = torch.zeros(1, dtype=torch.int32)
        c.device_buffer_size, c.padded_buffer_size, c.compress_ratio = H, width, 1
        c.decode_backup_stream = c.decode_producer_stream = None
        c.skip_io, c.is_dsv4_hisparse, c.item_size_bytes, c._prefetch_copy_blocks = False, False, DIM * 4, 4
        c.misses = 0
        c.enable_prefetch = bool(shared)
        c._prefetch_groups = {a: list(g) for a, g in (shared or {}).items()}
        c._is_shared_index_layer = [layer in followers for layer in LAYERS]
        c.mem_pool_host = Host(4096)
        c.mem_pool_device = types.SimpleNamespace(
            kv_buffer=hot, full_to_hisparse_device_index_mapping=torch.zeros(size * RATIO + 2 * PAGE,
                                                                             dtype=torch.int64))
        c.token_to_kv_pool_allocator = types.SimpleNamespace(logical_attn_allocator=Paged(size * RATIO, PAGE),
                                                             hisparse_attn_allocator=Paged(size, PAGE))
        c.tp_group = None
        self.cfg = cfg.resolve(LAYERS, followers=followers)
        self.state, self.views = state, factory.views
        self._bind(state, factory, chunks)
        self.expected, self.seq, self.uid, self.skip, self.stats = {}, {}, {}, set(), dict(stale=0, resident=0)

    def _bind(self, state, factory, chunks):
        """What ``elastic_device._bind`` does once the SGLang gates passed."""
        c, cfg, views = self.c, self.cfg, factory.views
        offsets = [views[(layer, "kv")]["hot_offset"] for layer in LAYERS]
        owner = {skip: anchor for anchor, group in c._prefetch_groups.items() for skip in group}

        def make_kv(rt):
            rows = ekv.MLALayers(torch, [views[(layer, "kv")]["full"] for layer in LAYERS], c.mem_pool_host)
            return ekv.ElasticKV(rows, c, layers=LAYERS, managed=cfg.managed, hot_offset=offsets,
                                 offloaded=cfg.offloaded(rt.level), torch=torch, device=torch.device("cpu"),
                                 owner=owner, through=cfg.feature("write_through"))

        def compaction_buffers(rt):
            inactive = set(rt.cfg.offloaded(rt.level)) - set(rt._filling())
            return [(v["full"], v["hot_offset"]) for (layer, _), v in views.items() if layer not in inactive], []

        state.update(config=cfg, memory=factory.memory, views=views, eligible=LAYERS, max_running=SLOTS - 1,
                     repo_root=dev.REPO_ROOT, physical_chunks=chunks, make_kv=make_kv,
                     compaction_buffers=compaction_buffers)
        dev._STATE = state
        dist = types.SimpleNamespace(get_rank=lambda group=None: 0, get_world_size=lambda group=None: 1)
        state["runtime"] = self.rt = dev.Runtime(state, c, torch=torch, dist=dist)
        dev._dsa_coordinator(state, torch, types.SimpleNamespace(HiSparseCoordinator=type(c),
                                                                 copy_cache_planned_mla=planned_copy_mla))

    # ---- requests ------------------------------------------------------------------------------------------
    @property
    def mapping(self):
        return self.c.mem_pool_device.full_to_hisparse_device_index_mapping

    def _locs(self, slot, start, stop):
        """Logical locations of positions start..stop-1 (a new page whenever one fills)."""
        a, r2t = self.c.token_to_kv_pool_allocator.logical_attn_allocator, self.c.req_to_token_pool.req_to_token
        for p in range(start, stop):
            r2t[slot, p] = int(a.take(1)[0]) * PAGE if p % PAGE == 0 else int(r2t[slot, p - 1]) + 1
        return r2t[slot, start:stop].long()

    def _rows(self, uid, layer, positions):
        for p in positions:
            self.expected[(uid, layer, p)] = value(uid, layer, p)
        return torch.tensor([value(uid, layer, p) for p in positions]).view(-1, 1, DIM)

    def _store(self, layer, slot, positions, locs):
        """What the set_mla_kv_buffer hook does: resident or route rows, else the stock mapping to hot rows."""
        rows = self._rows(self.uid[slot], layer, positions)
        target = dev._store_target(layer, locs, self.mapping)
        if target is None:
            self.views[(layer, "kv")]["hot"][self.mapping[locs]] = rows
        else:
            dst, view = target
            self.views[(layer, "kv")][view][dst] = rows

    def _hot_rows(self, count):
        pages = self.c.token_to_kv_pool_allocator.hisparse_attn_allocator.take(-(-count // PAGE))
        return (pages[:, None] * PAGE + torch.arange(PAGE)).reshape(-1)

    def admit(self, uid, n, direct=False):
        """Staged admission (prefill into each layer's current rows, staging to host, the stock device buffer of
        the first prompt rows) or PD direct admission (KV arrives in host for every layer)."""
        c, kv, width = self.c, self.rt.kv, H + PAGE
        slot = next(s for s in range(1, SLOTS) if s not in self.seq)
        self.uid[slot] = uid
        locs = self._locs(slot, 0, n)
        host = c.mem_pool_host.alloc_paged_token_slots(c.req_to_host_pool, c.req_to_host_pool_allocated_len, slot,
                                                       0, n)
        if direct:
            for layer in LAYERS:
                c.mem_pool_host.kv_buffer[layer][host] = self._rows(uid, layer, range(n))
            buffer = self._hot_rows(width)
        else:
            prefill = self._hot_rows(n)
            self.mapping[locs] = prefill[:n]
            for layer in LAYERS:
                self._store(layer, slot, range(n), locs)
            kv.stage(slot, host, prefill[:n])
            if len(prefill) < width:
                prefill = torch.cat((prefill, self._hot_rows(width - len(prefill))))
            c.token_to_kv_pool_allocator.hisparse_attn_allocator.release(torch.unique(prefill[width:] // PAGE))
            buffer = prefill[:width]
            self.mapping[locs] = 0
        c.req_to_device_buffer[slot, :width] = buffer
        c.req_device_buffer_size[slot] = width
        c.req_device_buffer_tokens[:, slot, :H] = torch.arange(H, dtype=torch.int32)
        c.req_device_buffer_token_locs[:, slot, :width] = buffer.to(torch.int32)
        for lru in c.lru:
            lru[slot] = list(range(H))
        kv.admitted(slot, n)
        if direct:
            if n <= H:
                kv.preload(c.req_to_host_pool[slot, :n], c.req_to_device_buffer[slot, :n])
            else:
                c.req_device_buffer_tokens[:, slot, :H] = -1
            kv.fill_request(slot, n, c.req_to_host_pool[slot, :n])
        self.seq[slot] = n
        self.skip.add(slot)
        return slot

    def finish(self, slot):
        c, n = self.c, self.seq.pop(slot)
        r2t = c.req_to_token_pool.req_to_token[slot, :n].long()
        c.token_to_kv_pool_allocator.logical_attn_allocator.release(torch.unique(r2t // PAGE))
        buffer = c.req_to_device_buffer[slot, :H + PAGE]
        c.token_to_kv_pool_allocator.hisparse_attn_allocator.release(torch.unique(buffer[buffer > 0] // PAGE))
        self.mapping[r2t] = 0
        c.req_device_buffer_tokens[:, slot] = -1
        c.req_device_buffer_token_locs[:, slot] = -1
        c.req_to_device_buffer[slot] = 0
        c.req_device_buffer_size[slot] = 0
        c.req_to_host_pool[slot] = -1
        c.req_to_host_pool_allocated_len[slot] = 0
        self.rt.kv.finished(slot)
        self.skip.discard(slot)

    def step(self):
        """One decode batch over every live request: the paced backfill tick, map_last_loc_to_buffer (write-through
        of the previous record, reserved slot of the new one), then per layer the store and a checked read."""
        c, rt = self.c, self.rt
        if rt.fill_job is not None:
            rt._advance_fill(None)
        kv, slots = rt.kv, sorted(self.seq)
        new = [int(self._locs(s, self.seq[s], self.seq[s] + 1)[0]) for s in slots]
        backup = [(s, self.seq[s] - 1) for s in slots if s not in self.skip]
        self.skip -= set(slots)
        if backup:
            host = torch.cat([c.mem_pool_host.alloc_paged_token_slots(
                c.req_to_host_pool, c.req_to_host_pool_allocated_len, s, record, 1) for s, record in backup])
            kv.write_through([s for s, _ in backup], [r for _, r in backup], host)
        kv.forward_begins()
        kv.decode_begins(slots, [self.seq[s] + 1 for s in slots])
        for s, loc in zip(slots, new):
            reserved = c.req_to_device_buffer[s, min(self.seq[s], H)]
            c.req_device_buffer_token_locs[:, s, H] = reserved
            self.mapping[loc] = reserved
            self.seq[s] += 1
        c.num_real_reqs.fill_(len(slots))
        rid, lens = torch.tensor(slots), torch.tensor([self.seq[s] for s in slots])
        with rt.decode_context() if self.mode == "smooth" else contextlib.nullcontext():
            for layer in LAYERS:
                with rt.layer_context(layer):
                    for s, loc in zip(slots, new):
                        self._store(layer, s, [self.seq[s] - 1], torch.tensor([loc]))
                    if not c._is_shared_index_layer[layer]:     # followers reuse their anchor's selection
                        top_k = self._select(slots)
                    rows = c.swap_in_selected_pages(rid, lens, top_k, layer)
                    self._check(layer, slots, top_k, rows, dev._view(layer, "kv"))

    def _select(self, slots):
        out = torch.full((len(slots), TOP_K), -1, dtype=torch.int32)
        for r, s in enumerate(slots):
            n = self.seq[s]
            picks = [n - 1] + self.rng.sample(range(n - 1), min(TOP_K - 1, n - 1))
            out[r, :len(picks)] = torch.tensor(picks, dtype=torch.int32)
        return out

    def _check(self, layer, slots, top_k, rows, cache):
        i = self.rt.kv.index[layer]
        for r, s in enumerate(slots):
            for k, p in enumerate(top_k[r].tolist()):
                if p < 0:
                    continue
                row = int(rows[r, k])
                got = cache[row].view(-1).tolist()
                self.case.assertEqual(got, self.expected[(self.uid[s], layer, p)],
                                      f"layer {layer} uid {self.uid[s]} position {p} row {row} mode {self.mode}")
                if self.mode == "smooth" and row >= self.rt.kv.hot_offset[i]:
                    self.stats["stale" if self.rt.kv.mode[i] == ekv.OFFLOADED else "resident"] += 1

    def move(self, level, hot=None, tokens=None):
        """Apply a transition the way the controller would: hot rows for the live buffers, logical tokens kept
        (so the leaving layers' rows stay stale) unless the level backs fewer."""
        rt = self.rt
        if hot is None:
            hot = max(rt.hot.peak_tokens(), len(self.seq) * (H + PAGE) + PAGE)
        if tokens is None:      # keep the ceiling plus a few pages of growth per request (the controller's grow)
            room = max(rt.logical.tokens(), rt.logical.used_tokens() + (2 * len(self.seq) + 1) * PAGE)
            cap = rt.controller.capacity(level, hot) if rt.fill_job is None else \
                rt._room(hot, rt.cfg.offloaded(level), rt._filling())      # a paced copy backs both regions
            tokens = min(cap, room)
        self.case.assertGreaterEqual(tokens, rt.logical.used_tokens(), "test asked for an infeasible level")
        rt.apply(dict(level=level, tokens=tokens, hot_rows=hot, admission=None, reason="test", step=rt.steps),
                 types.SimpleNamespace(max_running_requests=SLOTS - 1))
        rt.memory.check()
        return rt.receipts[-1] if rt.receipts else None


@unittest.skipIf(torch is None, "torch is not installed")
class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.emit = unittest.mock.patch.object(dev, "emit", lambda *args: None)
        self.emit.start()

    def tearDown(self):
        self.emit.stop()
        dev._STATE = None

    def run_path(self, rig, lengths, path, steps=3, finish_before=None):
        """Admit ``lengths`` (some longer than H), then decode ``steps`` batches before and after each level of
        ``path`` (finishing the requests ``finish_before[i]`` names before the i-th move); ``step`` checks
        every read."""
        slots = [rig.admit(uid, n) for uid, n in enumerate(lengths, 1)]
        for i, level in enumerate(path):
            for _ in range(steps):
                rig.step()
            for index in (finish_before or {}).get(i, ()):
                rig.finish(slots[index])
            rig.move(level)
            self.assertEqual(rig.rt.fill_job[0] if rig.rt.fill_job else rig.rt.level, level)
        for _ in range(steps):
            rig.step()
        return slots

    def test_steady_offload_and_backfill_keep_every_read(self):
        rig = Rig(self)
        self.run_path(rig, [5, 14, 20], [1, 2, 1, 0], finish_before={3: [2]})
        self.assertEqual(rig.rt.level, 0)
        self.assertEqual([r["status"] for r in rig.rt.receipts], ["committed"] * 4)
        self.assertGreater(rig.c.misses, 0)             # offloaded layers did load from host
        warm = [r["detail"] for r in rig.rt.receipts if "warmed_rows" in (r["detail"] or {})]
        self.assertTrue(warm and all(d["warmed_rows"] > 0 for d in warm))

    def test_smooth_paced_backfill_reads_stale_rows(self):
        # A few records per layer and step; the paced copy backs both regions of the returning group.
        rig = Rig(self, mode="smooth", size=128, controller={"backfill_gbps": 5e-6})
        self.run_path(rig, [5, 14, 20], [1, 2, 1], steps=4, finish_before={2: [2]})
        self.assertIsNotNone(rig.rt.fill_job)                                   # the paced copy is still running
        while rig.rt.fill_job is not None:
            rig.step()
            rig.move(rig.rt.level)                      # same-level grows go on during the copy
        self.assertEqual(rig.rt.level, 1)
        self.assertGreater(rig.stats["stale"], 0)       # offloaded layers read rows written before the offload
        self.assertGreater(rig.stats["resident"], 0)
        self.assertEqual(rig.rt.receipts[-1]["decision"]["reason"], "backfill complete")

    def test_backfill_skips_records_still_valid_in_stale_rows(self):
        rig = Rig(self)
        copies = []
        original = ekv.MLALayers.h2d
        with unittest.mock.patch.object(ekv.MLALayers, "h2d", lambda s, i, src, dst: (copies.append(len(dst)),
                                                                                       original(s, i, src, dst))):
            slots = [rig.admit(uid, n) for uid, n in enumerate([6, 10], 1)]
            rig.move(1)
            rig.step()
            copies.clear()
            rig.move(0)
            rig.step()
        live = sum(rig.seq[s] - 1 for s in slots)   # records before the last step, of both backfilled layers
        self.assertLess(sum(copies), 2 * live)      # the pre-offload records came from the stale rows

    def test_write_through_off_flushes_history_when_offloading(self):
        rig = Rig(self, features={"write_through": False})
        self.run_path(rig, [5, 14, 20], [1, 2])
        self.assertFalse(rig.rt.kv.through)

    def test_shared_index_followers_stay_offloaded_while_anchors_switch(self):
        for mode in ("steady", "smooth"):
            with self.subTest(mode=mode):
                rig = Rig(self, mode=mode, groups=((0,), (2,)), always=(1, 3), shared={0: [1], 2: [3]},
                          size=128 if mode == "smooth" else SIZE)
                self.run_path(rig, [5, 14, 20], [1, 2, 1, 0], finish_before={3: [2]})
                self.assertEqual(rig.rt.kv.owner, {1: 0, 3: 2})

    def test_pd_direct_admission_fills_resident_layers_from_host(self):
        rig = Rig(self)
        rig.admit(1, 6, direct=True)                # short: offloaded layers preloaded into the buffer
        rig.move(1, hot=3 * (H + PAGE) + PAGE)
        rig.admit(2, 18, direct=True)               # long: offloaded layers start all-miss
        for _ in range(4):
            rig.step()
        rig.move(2)
        rig.step()

    def apply(self, rig, level, tokens, hot):
        rig.rt.apply(dict(level=level, tokens=tokens, hot_rows=hot, admission=None, reason="test", step=0),
                     types.SimpleNamespace(max_running_requests=SLOTS - 1))
        rig.rt.memory.check()
        return rig.rt.receipts[-1]

    def test_warm_source_follows_the_free_chunks(self):
        tight = Rig(self)                           # level 0 backs every chunk: leaving rows are all there is
        for uid, n in enumerate([5, 14], 1):
            tight.admit(uid, n)
        tight.step()
        self.assertEqual(tight.move(1)["detail"]["warm_source"], "host")
        tight.step()
        roomy = Rig(self)
        self.apply(roomy, 0, 40, roomy.rt.hot.tokens())          # free chunks back the new hot regions
        for uid, n in enumerate([5, 14], 1):
            roomy.admit(uid, n)
        roomy.step()
        self.assertEqual(roomy.move(1)["detail"]["warm_source"], "device")
        roomy.step()

    def test_compaction_moves_live_pages_under_a_lower_ceiling(self):
        rig = Rig(self, size=128)
        rig.move(1, hot=64, tokens=160)
        slots = [rig.admit(uid, n) for uid, n in enumerate([20, 20, 20], 1)]
        rig.step()
        rig.finish(slots[0])
        rig.finish(slots[1])
        receipt = self.apply(rig, 1, 48, rig.rt.hot.tokens())
        self.assertGreater(receipt["detail"]["compacted_pages"], 0)
        pages = rig.c.req_to_token_pool.req_to_token[slots[2], :rig.seq[slots[2]]].long() // PAGE
        self.assertLessEqual(int(pages.max()), 48 // PAGE)      # every live page now sits under the ceiling
        for _ in range(3):
            rig.step()                              # resident rows moved with req_to_token; hot rows remapped

    def test_infeasible_transition_is_deferred_without_changes(self):
        rig = Rig(self)
        rig.move(1, hot=48, tokens=80)              # 48 hot rows: the 34-token prompt prefills into hot rows
        for uid, n in enumerate([30, 34], 1):
            rig.admit(uid, n)
        rig.step()
        before = (rig.rt.level, rig.rt.logical.tokens(), rig.rt.hot.tokens())
        self.assertGreater(rig.rt.logical.used_tokens(), rig.rt.controller.capacity(0, 48))
        receipt = self.apply(rig, 0, rig.rt.controller.capacity(0, 48), 48)
        self.assertEqual(receipt["status"], "deferred")
        self.assertEqual((rig.rt.level, rig.rt.logical.tokens(), rig.rt.hot.tokens()), before)
        rig.step()


def scheduler_of(rig, waiting=()):
    """The scheduler fields ``Runtime.tick`` and ``observe`` read."""
    reqs = [types.SimpleNamespace(origin_input_ids=[0] * n, output_ids=[]) for n in rig.seq.values()]
    batch = types.SimpleNamespace(is_empty=lambda: not reqs, reqs=reqs,
                                  forward_mode=types.SimpleNamespace(is_decode=lambda: True))
    return types.SimpleNamespace(max_running_requests=SLOTS - 1, running_batch=batch, last_batch=batch,
                                 waiting_queue=[types.SimpleNamespace(origin_input_ids=[0] * n) for n in waiting],
                                 chunked_req=None, chunked_prefill_size=0)


@unittest.skipIf(torch is None, "torch is not installed")
class ControlLoopTests(unittest.TestCase):
    def setUp(self):
        self.emit = unittest.mock.patch.object(dev, "emit", lambda *args: None)
        self.emit.start()

    def tearDown(self):
        self.emit.stop()
        dev._STATE = None

    def test_pinned_level_then_controller_backfills_under_low_demand(self):
        rig = Rig(self, controller={"dwell_steps": 2})
        rig.admit(1, 6)
        rig.rt.pinned_level = 1
        rig.rt.tick(scheduler_of(rig))
        self.assertEqual(rig.rt.level, 1)
        self.assertEqual(rig.rt.receipts[-1]["decision"]["reason"], "pinned to level 1")
        rig.step()
        rig.rt.pinned_level = None
        for _ in range(8):
            rig.rt.tick(scheduler_of(rig))
            rig.step()
        self.assertEqual(rig.rt.level, 0)
        self.assertTrue(any(r["decision"]["reason"] == "backfill to level 0" for r in rig.rt.receipts))

    def test_queue_pressure_offloads_through_the_controller(self):
        rig = Rig(self)
        for uid, n in enumerate([20, 20], 1):
            rig.admit(uid, n)
        rig.rt.tick(scheduler_of(rig, waiting=[30, 30]))
        self.assertGreater(rig.rt.level, 0)
        self.assertTrue(rig.rt.receipts[-1]["decision"]["reason"].startswith("offload to level"))
        rig.step()

    def test_steady_precapture_captures_every_level_and_restores_the_backing(self):
        rig = Rig(self)
        seen = []

        def capture(runner):
            rt = rig.rt
            seen.append((rt.capture_level, rt.graph_key("bs8"), rt.memory.offloaded))
        rig.state["stock_capture"] = capture
        start = (rig.rt.memory.tokens, rig.rt.memory.hot_rows, rig.rt.memory.offloaded)
        rig.rt.precapture(object())
        self.assertEqual([s[0] for s in seen], [0, 1, 2])
        self.assertEqual([s[1] for s in seen], [("hisparse-elastic", level, "bs8") for level in (0, 1, 2)])
        self.assertEqual([s[2] for s in seen], [rig.cfg.offloaded(level) for level in (0, 1, 2)])
        self.assertEqual((rig.rt.memory.tokens, rig.rt.memory.hot_rows, rig.rt.memory.offloaded), start)
        self.assertEqual(rig.rt.captured, {0, 1, 2})
        self.assertEqual(rig.rt.graph_key("bs8"), ("hisparse-elastic", 0, "bs8"))
        rig.admit(1, 6)
        rig.step()


@unittest.skipIf(torch is None, "torch is not installed")
class PartsTests(unittest.TestCase):
    def test_ceiling_hands_out_pages_below_it_and_survives_clear(self):
        a = Paged(64, PAGE)
        ceiling = dev.Ceiling(a, torch)
        ceiling.set(32)
        self.assertEqual(a.available_size(), 32)
        taken = a.take(3)
        a.release(taken[1:2])                                   # page 2 free again, pages 1 and 3 live
        self.assertEqual(ceiling.peak_tokens(), 4 * PAGE)
        self.assertEqual(ceiling.live_pages_above(2), [3])
        moves = ceiling.plan_moves(2)
        self.assertEqual(moves, [(3, 2)])
        with self.assertRaises(RuntimeError):
            ceiling.set(8)                                       # live page 3 above the new ceiling
        ceiling.set(8, moves=moves)                              # compaction moved page 3 into page 2
        self.assertEqual(a.free_pages.tolist(), [])              # pages 1 and 2 hold the live data
        a.clear()
        self.assertEqual(a.available_size(), 8)                 # clear keeps the ceiling

    def test_mha_rows_and_block_selection(self):
        k = [torch.zeros((32, 1, DIM)), torch.zeros((32, 1, DIM))]
        v = [torch.zeros((32, 1, DIM)), torch.zeros((32, 1, DIM))]
        host = types.SimpleNamespace(k_buffer=[torch.arange(64.).view(16, 1, DIM)] * 2,
                                     v_buffer=[-torch.arange(64.).view(16, 1, DIM)] * 2)
        rows = ekv.MHALayers(torch, k, v, host)
        rows.h2d(1, torch.tensor([2, 3]), torch.tensor([20, 21]))
        self.assertEqual(k[1][20].view(-1).tolist(), host.k_buffer[1][2].view(-1).tolist())
        rows.d2d(1, torch.tensor([20]), torch.tensor([5]))
        rows.d2h(1, torch.tensor([5]), torch.tensor([9]))
        self.assertEqual(host.v_buffer[1][9].view(-1).tolist(), v[1][20].view(-1).tolist())
        self.assertEqual(rows.item, DIM * 4)
        c = types.SimpleNamespace(req_to_token_pool=types.SimpleNamespace(
            req_to_token=torch.arange(64, dtype=torch.int32).view(2, 32)))
        kv = ekv.ElasticKV(rows, c, layers=(0, 1), managed=(0, 1), hot_offset=[8, 8], offloaded=(1,),
                           torch=torch, device=torch.device("cpu"))
        kv.stale_limit[1, 1] = 3                                 # records 0..2 of slot 1 still in the stale rows
        kv.route[1, 2] = 64
        blocks = torch.tensor([[0, 1, 2]])                       # blocks of 2 records: 0-1, 2-3, 4-5
        masked, rows_out, reads = kv.smooth_select(1, torch.tensor([1]), torch.tensor([6]), blocks, block=2)
        self.assertEqual(masked.tolist(), [[-1, 1, 2]])          # only whole stale blocks skip the swap-in
        self.assertEqual(reads.tolist(), [[True, True, False, False, False, False]])
        self.assertEqual(rows_out[0, :2].tolist(), [32 + 8, 33 + 8])


class AdapterTests(unittest.TestCase):
    BASE = ["--enable-hisparse", "--tp-size", "8", "--hisparse-config", '{"top_k":2048,"device_buffer_size":4096}',
            "--disable-overlap-schedule", "--disable-radix-cache"]

    def resolve(self, name, *extra):
        from . import launch
        with contextlib.redirect_stderr(io.StringIO()):
            return launch.resolve(["--hisparse-elastic-config", str(FIXTURES / name), "--", *self.BASE, *extra])

    def test_launcher_enforces_the_elastic_execution_rules(self):
        with self.assertRaises(SystemExit):                       # prefill must run eagerly
            self.resolve("minimax_m3_steady.json")
        plan = self.resolve("minimax_m3_steady.json", "--disable-prefill-cuda-graph")[0]
        self.assertEqual(plan.elastic_config_sha256, ec.read(plan.elastic_config_path)[1])
        self.resolve("glm52_smooth.json", "--disable-prefill-cuda-graph", "--disaggregation-mode", "decode")
        for extra in (("--disaggregation-mode", "prefill"), ("--kv-cache-dtype", "fp8_e4m3"),
                      ("--enable-mixed-chunk",), ("--speculative-algorithm", "EAGLE")):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.resolve("minimax_m3_steady.json", "--disable-prefill-cuda-graph", *extra)

    def test_worker_installs_the_elastic_adapter(self):
        from . import launch
        calls = []
        plan = self.resolve("dsv32_steady.json", "--disable-prefill-cuda-graph")[0]
        with unittest.mock.patch("runtime.elastic_device.install",
                                 lambda config, digest: calls.append((config.digest, digest))):
            launch.install_for_process(plan)
        self.assertEqual(calls, [(ec.read(plan.elastic_config_path)[0].digest, plan.elastic_config_sha256)])

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_deepseek_v4_is_refused_until_its_c4_pools_are_wired(self):
        config, digest = ec.read(FIXTURES / "dsv4_flash_steady.json")
        with self.assertRaises(NotImplementedError):
            dev.install(config, digest)
        self.assertIsNone(dev._STATE)

    @unittest.skipUnless(SRC and torch is not None, "set SGLANG_SRC to the pinned checkout")
    def test_pinned_checkout_resolves_every_fixture(self):
        root = Path(SRC) / "python" / "sglang" if (Path(SRC) / "python" / "sglang").is_dir() else Path(SRC)
        for name in ("minimax_m3_steady.json", "minimax_m3_smooth_slo.json", "glm52_smooth.json",
                     "dsv32_steady.json"):
            with self.subTest(fixture=name):
                config, _ = ec.read(FIXTURES / name)
                geo = dict(dev.PRESETS[config.model], **({"eligible": "0-60"} if name.startswith("dsv32") else {}))
                report = dev.dry_run(root, config, config.model, geo)
                self.assertEqual(report["status"], "pass", [s for s in report["symbols"] if s["status"] != "ok"])
                tokens = [level["tokens"] for level in report["levels"]]
                self.assertEqual(tokens, sorted(tokens))           # every group offloaded backs more tokens


if __name__ == "__main__":
    unittest.main()
