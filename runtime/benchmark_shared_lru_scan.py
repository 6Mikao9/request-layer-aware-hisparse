"""Measure one complete MHA swap-in on synthetic A100 inputs under CUDA Graph replay.

python -m runtime.benchmark_shared_lru_scan --allow-cuda-test --output scan.json

Arena arms also run every requested resolver switch set (``--features``; the
issue #6 set is ``indexed`` = hit_index + incremental_lru + compact_owner) from
the same initial state; each set must reproduce the switches-off output
indices, owner planes, recency order, occupancy, reverse tokens and status
(``logical_sha256``) before its timing is kept. ``--layouts fragmented``
scatters the rows of every layer over the whole domain after admission.
``--ring-states rebuild`` starts incremental_lru sets from a ring whose next
append would overflow, so the timed call includes the stable rebuild.
"""
import argparse
import gc
import hashlib
import json
import random
import statistics
from dataclasses import replace, asdict
from pathlib import Path
from types import SimpleNamespace

from runtime import shared_lru as reference
from runtime import study_config as sc
from runtime.cuda_shared_lru_correctness import ARMS, FEATURE_SETS
from runtime.cuda_shared_lru_production_correctness import GEOMETRY
from runtime.shared_lru_device import ArenaRuntime, BLOCK_SIZE, KERNEL_SOURCE, verify_digests

LAYOUTS = ("contiguous", "fragmented")
RING_STATES = ("template", "rebuild")


def selected_blocks(g, miss_blocks):
    if not 0 <= miss_blocks <= g.blocks:
        raise ValueError("miss-blocks must be between zero and the top-k block count")
    cold = g.hot // g.block
    if g.hot % g.block or cold < g.blocks or (cold + miss_blocks) * g.block >= g.context:
        raise ValueError("selection overlaps the tail or exceeds the benchmark geometry")
    return list(range(g.blocks - miss_blocks)) + list(range(cold, cold + miss_blocks))


def make_history(torch, rows, dim, pin):
    """Chunked setup. Three base-127 digits identify a host row exactly in BF16."""
    shape = (rows, 1, dim)
    k = torch.empty(shape, dtype=torch.bfloat16, pin_memory=pin)
    v = torch.empty_like(k, pin_memory=pin)
    columns = torch.arange(dim)
    for start in range(0, rows, 4096):
        stop = min(start + 4096, rows)
        ids = torch.arange(start, stop).unsqueeze(1)
        values = (ids * 7 + columns * 13) % 251 - 125
        for col, divisor in enumerate((1, 127, 127 * 127)):
            values[:, col] = (ids[:, 0] // divisor) % 127
        k[start:stop, 0] = values
        v[start:stop, 0] = -values - 1
    return k, v


class Case:
    """Only the requesting plane needs host backing. All arena rows remain addressable.

    ``home[t]`` is the arena row that holds the requesting layer's token ``t`` before the
    launch. The contiguous layout keeps admission's row ``plane * H + t``. The fragmented
    layout applies one seeded permutation of the measured domain to every slot: row
    ``pi(r)`` takes the owner tag, reverse token, recency position and K/V of row ``r``.
    """

    def __init__(self, torch, arm, g, miss_blocks, device="cuda", features=(), layout="contiguous", seed=20261126,
                 ring_state="template"):
        if layout not in LAYOUTS or ring_state not in RING_STATES or (
                arm == "stock" and (features or layout != "contiguous" or ring_state != "template")):
            raise ValueError("stock runs the native kernel on its own contiguous layout without switches")
        self.torch, self.arm, self.g, self.features, self.layout = torch, arm, g, tuple(features), layout
        self.hits = (g.blocks - miss_blocks) * g.block
        self.selected = g.blocks * g.block
        planes = 1 if arm == "stock" else g.layers
        p = self.plane = planes - 1
        cols = g.hot + g.page
        self.rows = rows = g.page + g.slots * cols
        self.host = make_history(torch, g.slots * g.context, g.dim, device == "cuda")
        shape = (planes, rows, 1, g.dim)
        self.k = torch.full(shape, -999, dtype=torch.bfloat16, device=device)
        self.v = torch.full_like(self.k, -999)
        pool = SimpleNamespace(start_layer=0, layer_num=planes, page_size=g.page,
                               k_buffer=self.k.unbind(), v_buffer=self.v.unbind(),
                               _rlh_contiguous_kv=(self.k, self.v))
        i32 = torch.int32
        def arange(n, dtype):
            return torch.arange(n, dtype=dtype, device=device)
        self.locs = locs = arange(g.slots * cols, i32).reshape(g.slots, cols) + g.page
        tokens = torch.full((planes, g.slots, cols), -1, dtype=i32, device=device)
        tokens[:, :, :g.hot] = arange(g.hot, i32)
        tokens[:, :, g.hot] = g.context - 1
        self.out = torch.full((g.slots, self.selected), -1, dtype=i32, device=device)
        host_locs = arange(g.slots * g.context, torch.int64).reshape(g.slots, g.context)
        c = self.coordinator = SimpleNamespace(
            mem_pool_device=pool, device_buffer_size=g.hot, top_k=self.selected, item_size_bytes=g.dim * 2,
            req_device_buffer_tokens=tokens, req_device_buffer_token_locs=locs.repeat(planes, 1, 1),
            lru_slots=arange(g.hot, torch.int16).repeat(planes, g.slots, 1),
            top_k_device_locs_buffer=self.out, num_real_reqs=torch.tensor([g.slots], dtype=i32, device=device),
            req_to_host_pool=host_locs, mem_pool_host=SimpleNamespace(
                k_buffer=[None] * p + [self.host[0]], v_buffer=[None] * p + [self.host[1]]))
        self.req = arange(g.slots, torch.int64)
        self.seq = torch.full((g.slots,), g.context, dtype=i32, device=device)
        blocks = selected_blocks(g, miss_blocks)
        self.top = torch.tensor(blocks, dtype=i32, device=device).repeat(g.slots, 1)
        self.tokens = (self.top.long().unsqueeze(2) * g.block + arange(g.block, torch.int64)).flatten(1)
        self.host_rows = host_locs.gather(1, self.tokens).cpu()
        mutable = [tokens]
        self.runtime = None
        self.home = arange(g.hot, torch.int64) + p * g.hot
        if arm == "stock":
            mutable.append(c.lru_slots)
        else:
            cfg = sc.parse({"schema": sc.SCHEMA, "hot_pool": ARMS[arm], "features": dict.fromkeys(features, True)})
            r = self.runtime = ArenaRuntime(c, cfg, cfg.digest, torch=torch)
            for rid in range(g.slots):
                r.init_slot(rid)
            mutable.extend(t for t in (r.row_owner, r.order, r.occupancy, r.status, r.owner_summary, r.index,
                                       r.index_flags, r.ring, r.ring_dead, r.ring_pos, r.ring_meta) if t.numel())
            if layout == "fragmented":
                self.fragment(random.Random(seed))
            if ring_state == "rebuild" and r.ring.numel():
                self.fill_ring()
        self.ring_state = ring_state if self.runtime is not None and self.runtime.ring.numel() else "template"
        home = self.home[self.tokens[:, :self.hits]]                       # rows of the hit tokens
        hit_planes, hit_locs = home // g.hot, locs.gather(1, home % g.hot)
        for buf, history in zip((self.k, self.v), self.host):
            buf[hit_planes.flatten(), hit_locs.long().flatten()] = history[self.host_rows[:, :self.hits].flatten()].to(device)
        self.snapshots = [(t, t.clone()) for t in mutable]

    def fragment(self, rng):
        torch, g, r = self.torch, self.g, self.runtime
        offset, size = r.plan.domain(self.plane)
        perm = list(range(size))
        rng.shuffle(perm)
        target = torch.tensor(perm, dtype=torch.int64, device=self.out.device) + offset    # pi(offset + i)
        planes = r.plan.rows // g.hot
        tokens = self.coordinator.req_device_buffer_tokens
        flat = tokens[:, :, :g.hot].permute(1, 0, 2).reshape(g.slots, planes * g.hot).clone()
        for table in (r.row_owner, flat) + ((r.ring_pos,) if r.ring.numel() else ()):  # ring positions move too
            table[:, target] = table[:, offset:offset + size].clone()
        tokens[:, :, :g.hot] = flat.reshape(g.slots, planes, g.hot).permute(1, 0, 2)
        if r.ring.numel():    # template ring: the domain's live entries are its first ``size`` entries
            start = reference.ring_domains(r.plan)[reference.ring_domain(r.plan, self.plane)][0]
            region = r.ring[:, start:start + size]
        else:
            region = r.order[:, offset:offset + size]
        region.copy_(target[(region - offset).long()].to(region.dtype))
        mapping = torch.arange(r.plan.rows, dtype=torch.int64, device=self.out.device)
        mapping[offset:offset + size] = target
        self.home = mapping[self.home]
        if r.owner_summary.numel():
            r.owner_summary.copy_(owner_summary_words(torch, r.row_owner, r.generation, planes))
        if r.index.numel():   # every slot holds the same permuted tables: index slot 0's keys once, copy
            owners = owner_planes(torch, r.row_owner[:1], r.generation[:1])[0].tolist()
            index = reference.index_from_keys(r.plan, {row: (plane, token) for row, (plane, token)
                                                       in enumerate(zip(owners, flat[0].tolist())) if plane >= 0})
            r.index.copy_(torch.tensor(index, dtype=torch.int32, device=self.out.device).expand_as(r.index))

    def fill_ring(self):
        """Ring state whose next append overflows: G dead entries, then the live list, tail = capacity - K + 1."""
        torch, r = self.torch, self.runtime
        domain = reference.ring_domain(r.plan, self.plane)
        start, capacity, words, _ = reference.ring_domains(r.plan)[domain]
        size = r.plan.domains[domain][1]
        gap = capacity - size - r.plan.selected + 1
        live = r.ring[:, start:start + size].clone()
        filler = live[:, :gap] if gap <= size else live[:, torch.arange(gap, device=self.out.device) % size]
        r.ring[:, start:start + gap] = filler
        r.ring[:, start + gap:start + gap + size] = live
        offset = r.plan.domains[domain][0]
        rows = torch.arange(offset, offset + size, device=self.out.device)
        r.ring_pos[:, rows] += gap
        bits = torch.zeros(-(-capacity // 32) * 32, dtype=torch.int64, device=self.out.device)
        bits[:gap] = 1
        packed = (bits.view(-1, 32) << torch.arange(32, device=self.out.device)).sum(1)
        r.ring_dead[:, words:words + packed.numel()] = (packed - (packed >= 2**31).long() * 2**32).to(torch.int32)
        r.ring_meta[:, 4 * domain:4 * domain + 2] = torch.tensor([0, gap + size], dtype=torch.int32,
                                                                device=self.out.device)

    def launch(self):
        g, c, p = self.g, self.coordinator, self.plane
        if self.runtime is not None:
            return self.runtime.resolve(c, self.req, self.seq, self.top, p, g.block)
        from sglang.kernels.ops.kvcache.hisparse import load_blocks_to_device_buffer_mha
        load_blocks_to_device_buffer_mha(
            top_k_blocks=self.top, device_buffer_tokens=c.req_device_buffer_tokens[p],
            host_cache_locs=c.req_to_host_pool, device_buffer_locs=c.req_device_buffer_token_locs[p],
            host_cache_k=self.host[0], host_cache_v=self.host[1], device_buffer_k=self.k[p],
            device_buffer_v=self.v[p], top_k_device_locs=self.out, req_pool_indices=self.req,
            seq_lens=self.seq, lru_slots=c.lru_slots[p], item_size_bytes=c.item_size_bytes,
            hot_buffer_size=g.hot, sparse_block_size=g.block, page_size=1, block_size=960,
            num_real_reqs=c.num_real_reqs, skip_io=False)

    def reset(self):
        # All non-hit rows initially contain the sentinel. Only the destinations
        # of this launch's cold suffix could have been written by swap-in.
        if self.hits < self.selected:
            idx = self.out[:, self.hits:].reshape(-1).long()
            for buf in (self.k, self.v):
                buf.flatten(0, 1).index_fill_(0, idx, -999)
        for current, initial in self.snapshots:
            current.copy_(initial)
        self.out.fill_(-1)

    def verify(self):
        torch, g = self.torch, self.g
        idx = self.out.long()
        planes = idx // self.rows
        local = (idx % self.rows) - g.page
        slots = local // (g.hot + g.page)
        row = local % (g.hot + g.page)
        def require(ok, message):
            if not bool(ok):
                raise RuntimeError(message)
        require(((planes >= 0) & (planes < self.k.size(0)) &
                 (slots == self.req[:, None]) & (row < g.hot)).all(), "wrong request or invalid output row")
        # With this admitted layout a returned location was a hit exactly when it is the home
        # row of the requested token (its original owner and token matched the requested key).
        was_hit = (self.tokens < g.hot) & (planes * g.hot + row == self.home[self.tokens.clamp(max=g.hot - 1)])
        require(was_hit[:, :self.hits].all() & ~was_hit[:, self.hits:].any(), "unexpected hit/miss pattern")
        require((self.coordinator.req_device_buffer_tokens[planes, slots, row] == self.tokens).all(),
                "reverse token disagrees with output")
        if self.runtime is not None:
            r = self.runtime
            owners = owner_planes(torch, r.row_owner, r.generation)
            require((owners[slots, planes * g.hot + row] == self.plane).all(), "wrong owner tag")
            require(r.status.item() == 0, "resolver status bits are set")
            if r.owner_summary.numel():
                live = owner_summary_words(torch, r.row_owner, r.generation, r.plan.rows // g.hot)
                require(torch.equal(r.owner_summary, live), "owner summary differs from the owner tags")
            if r.index.numel():
                require(not bool(r.index_flags.any()), "a hit index overflowed")
        for buf, history in zip((self.k, self.v), self.host):
            require(torch.equal(buf.flatten(0, 1)[idx].cpu(), history[self.host_rows]), "returned KV bytes differ")
        hits = int(was_hit.sum())
        return dict(hits=hits, misses=idx.numel() - hits)

    def state_sha256(self):
        """Digest of every output and table this switch set writes (owner summary excluded): equal replays."""
        digest = hashlib.sha256()
        tables = [self.out, self.coordinator.req_device_buffer_tokens]
        if self.runtime is not None:
            r = self.runtime
            tables += [t for t in (r.row_owner, r.order, r.occupancy, r.status, r.ring, r.ring_dead, r.ring_pos,
                                   r.ring_meta) if t.numel()]
        else:
            tables.append(self.coordinator.lru_slots)
        for table in tables:
            digest.update(bytes(table.cpu().clone().untyped_storage()))   # clone: exactly this view's bytes
        return digest.hexdigest()

    def logical_sha256(self):
        """Digest of what every switch set must agree on: outputs, reverse tokens, owner planes, the recency
        list of every domain (decoded from the ring under incremental_lru), occupancy and status."""
        torch, digest = self.torch, hashlib.sha256()
        tables = [self.out, self.coordinator.req_device_buffer_tokens]
        if self.runtime is not None:
            r = self.runtime
            tables += [owner_planes(torch, r.row_owner, r.generation), recency_lists(torch, r), r.occupancy, r.status]
        else:
            tables.append(self.coordinator.lru_slots)
        for table in tables:
            digest.update(bytes(table.cpu().to(torch.int64).clone().untyped_storage()))
        return digest.hexdigest()


def owner_planes(torch, row_owner, generation):
    """Plane owning each row of the slot's generation, else -1 (int32 tags or compact_owner bytes)."""
    if row_owner.dtype == torch.uint8:
        tags = row_owner.long()
        return torch.where(tags < 64, tags, -1)
    live = (row_owner >= 0) & ((row_owner >> 16) == generation.view(-1, 1))
    return torch.where(live, row_owner & 0xFFFF, -1).long()


def recency_lists(torch, runtime):
    """[slots, sum of domain sizes] recency lists in plain ``order`` form, decoded from the ring if present."""
    if not runtime.ring.numel():
        return runtime.order
    ring, dead, meta = runtime.ring.cpu(), runtime.ring_dead.cpu().long() & 0xFFFFFFFF, runtime.ring_meta.cpu()
    lists = []
    for slot in range(ring.shape[0]):
        parts = []
        for d, (start, capacity, words, _) in enumerate(reference.ring_domains(runtime.plan)):
            head, tail = int(meta[slot, 4 * d]), int(meta[slot, 4 * d + 1])
            at = torch.arange(head, tail)
            alive = ((dead[slot, words + at // 32] >> (at % 32)) & 1) == 0
            parts.append(ring[slot, start + at[alive]])
        lists.append(torch.cat(parts))
    return torch.stack(lists)


def owner_summary_words(torch, row_owner, generation, planes):
    """Owner-plane bits per 32-row block from each slot's live tags (int64, plane 63 is the sign bit)."""
    slots, rows = row_owner.shape
    blocks = reference.summary_blocks(rows)
    owners = torch.nn.functional.pad(owner_planes(torch, row_owner, generation), (0, blocks * reference.SUMMARY_ROWS - rows),
                                     value=-1)
    plane = owners.view(slots, blocks, reference.SUMMARY_ROWS)
    words = torch.zeros((slots, blocks), dtype=torch.int64, device=row_owner.device)
    for p in range(planes):
        words |= (plane == p).any(-1).to(torch.int64) << p
    return words


def run_case(torch, arm, g, miss_blocks, repeats, warmup, *, features=(), layout="contiguous", seed=20261126,
             ring_state="template"):
    torch.cuda.reset_peak_memory_stats()
    case = Case(torch, arm, g, miss_blocks, features=features, layout=layout, seed=seed, ring_state=ring_state)
    # Build all specializations before capture. This eager launch also validates
    # output locations before the first sparse payload reset.
    case.launch()
    counts = case.verify()
    state, logical = case.state_sha256(), case.logical_sha256()
    rebuilds = ring_rebuilds(case)
    case.reset()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    with torch.cuda.graph(graph):
        start.record()
        case.launch()
        end.record()
    timings = []
    for step in range(warmup + repeats):
        if step:
            case.reset()
        graph.replay()
        end.synchronize()
        if step >= warmup:
            timings.append(start.elapsed_time(end) * 1000)
        if step in (0, warmup, warmup + repeats - 1):
            if case.verify() != counts or case.state_sha256() != state:
                raise RuntimeError("Graph replay changed the hit/miss counts or the resolved tables")
    runtime = case.runtime
    domain = runtime.plan.domain(case.plane)[1] if runtime else g.hot
    arena_tables = sum(t.numel() * t.element_size() for t in runtime.tables().values()) if runtime else 0
    return dict(arm=arm, features=list(features), layout=layout, ring_state=case.ring_state,
                ring_rebuilds_per_launch=rebuilds, geometry=asdict(g), measured_plane=case.plane,
                device_planes=case.k.size(0), blocks_per_grid=g.slots,
                threads_per_cta=960 if arm == "stock" else BLOCK_SIZE,
                shared_memory_bytes_per_cta=runtime.shared_memory if runtime else None,
                arena_table_bytes=arena_tables, domain_rows_per_request=domain,
                selected_rows_per_request=case.selected, miss_blocks=miss_blocks,
                actual_counts_per_launch=counts, miss_kv_bytes_per_launch=counts["misses"] * 4 * g.dim,
                state_sha256=state, logical_sha256=logical,
                host_pinned_bytes=sum(t.numel() * t.element_size() for t in case.host),
                device_kv_bytes=case.k.numel() * 4, torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                kernel_us=dict(median=statistics.median(timings), samples=timings))


def ring_rebuilds(case):
    """Ring rebuilds the last launch performed, summed over slots (0 without incremental_lru)."""
    r = case.runtime
    if r is None or not r.ring.numel():
        return 0
    before = dict(case.snapshots)
    return int((r.ring_meta.view(r.ring_meta.shape[0], -1, 4)[:, :, 2]
                - before[r.ring_meta].view(r.ring_meta.shape[0], -1, 4)[:, :, 2]).sum())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cuda-test", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=("stock", "isolated", "shared"),
                        default=["stock", "isolated", "shared"])
    parser.add_argument("--features", nargs="+", choices=tuple(FEATURE_SETS), default=list(FEATURE_SETS),
                        help="resolver switch sets for arena arms; 'none' always runs first as the reference")
    parser.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=["contiguous"],
                        help="fragmented permutes the measured domain's rows (arena arms only)")
    parser.add_argument("--ring-states", nargs="+", choices=RING_STATES, default=["template"],
                        help="rebuild starts incremental_lru sets from a ring the timed call must compact")
    parser.add_argument("--no-reference", action="store_true",
                        help="profiling only: run just the listed sets, without the switches-off reference "
                             "case and its state comparison (the clean timing run performs it)")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 8, 60])
    parser.add_argument("--miss-blocks", nargs="+", type=int, default=[0, 4])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261126, help="row permutation of the fragmented layout")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.allow_cuda_test or args.output.exists():
        parser.error("Use --allow-cuda-test and a fresh --output path")
    if (min(args.batches) < 1 or min(args.layers) < 1 or max(args.layers) > 64 or
            min(args.miss_blocks) < 0 or max(args.miss_blocks) > GEOMETRY.blocks or
            args.warmup < 0 or args.repeats < 1):
        parser.error("invalid batch, layer, miss block, warmup, or repeat count")
    import torch
    import sglang
    if "A100" not in torch.cuda.get_device_name() or torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("A100 (sm_80) required")
    bad = {k: v for k, v in verify_digests(Path(sglang.__file__).resolve().parent).items() if v != "ok"}
    if bad:
        raise RuntimeError(f"Pinned SGLang source mismatch: {bad}")
    switch_sets = list(args.features) if args.no_reference else (
        ["none"] + [name for name in args.features if name != "none"])
    results = []
    for batch in args.batches:
        for layers in args.layers:
            for misses in args.miss_blocks:
                for layout in args.layouts:
                    for arm in args.arms:
                        if arm == "stock" and layout != "contiguous":
                            continue                                  # no arena rows to permute
                        reference_state = None
                        for name in ["none"] if arm == "stock" else switch_sets:
                            ring = "incremental_lru" in FEATURE_SETS[name]
                            for ring_state in [s for s in args.ring_states if ring or s == "template"] or ["template"]:
                                result = run_case(torch, arm, replace(GEOMETRY, slots=batch, layers=layers), misses,
                                                  args.repeats, args.warmup, features=FEATURE_SETS[name],
                                                  layout=layout, seed=args.seed, ring_state=ring_state)
                                if not args.no_reference:
                                    reference_state = reference_state or result["logical_sha256"]
                                    if result["logical_sha256"] != reference_state:
                                        raise RuntimeError(f"{arm} {name}: hits, victims or tables differ from "
                                                           "switches off")
                                result["matches_switches_off"] = None if args.no_reference else True
                                results.append(result)
                                print(json.dumps(result), flush=True)
                                gc.collect()
                                torch.cuda.empty_cache()
    summary = dict(event="hisparse_resolver_scan_microbench", device=torch.cuda.get_device_name(),
                   torch=str(torch.__version__), cuda=torch.version.cuda, warmup=args.warmup,
                   repeats=args.repeats, dtype="bfloat16", kv_heads_per_rank=1, diagnostics="none",
                   feature_sets={name: list(FEATURE_SETS[name]) for name in switch_sets}, layouts=args.layouts,
                   ring_states=args.ring_states, fragment_seed=args.seed,
                   arena_kernel_sha256=hashlib.sha256(KERNEL_SOURCE.read_bytes()).hexdigest(),
                   timing="external CUDA events inside one graph, one complete resolver, reset before replay",
                   cache_condition="synthetic locations, metadata touched by device reset before each sample",
                   results=results)
    with args.output.open("x") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
