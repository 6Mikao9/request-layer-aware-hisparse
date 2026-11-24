"""Measure one complete MHA swap-in on synthetic A100 inputs under CUDA Graph replay.

python -m runtime.benchmark_shared_lru_scan --allow-cuda-test --output scan.json
"""
import argparse
import gc
import hashlib
import json
import statistics
from dataclasses import replace, asdict
from pathlib import Path
from types import SimpleNamespace

from runtime import study_config as sc
from runtime.cuda_shared_lru_correctness import ARMS
from runtime.cuda_shared_lru_production_correctness import GEOMETRY
from runtime.shared_lru_device import ArenaRuntime, BLOCK_SIZE, KERNEL_SOURCE, verify_digests


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
    """Only the requesting plane needs host backing. All arena rows remain addressable."""

    def __init__(self, torch, arm, g, miss_blocks, device="cuda"):
        self.torch, self.arm, self.g = torch, arm, g
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
        locs = arange(g.slots * cols, i32).reshape(g.slots, cols) + g.page
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
        hit_locs = locs.gather(1, self.tokens[:, :self.hits]).long().flatten()
        for buf, history in zip((self.k, self.v), self.host):
            buf[p, hit_locs] = history[self.host_rows[:, :self.hits].flatten()].to(device)
        mutable = [tokens]
        self.runtime = None
        if arm == "stock":
            mutable.append(c.lru_slots)
        else:
            cfg = sc.parse({"schema": sc.SCHEMA, "hot_pool": ARMS[arm]})
            r = self.runtime = ArenaRuntime(c, cfg, cfg.digest, torch=torch)
            for rid in range(g.slots):
                r.init_slot(rid)
            mutable.extend((r.row_owner, r.order, r.occupancy, r.status))
        self.snapshots = [(t, t.clone()) for t in mutable]

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
        # With this admitted layout a returned location was a hit exactly when
        # its original owner and token already matched the requested key.
        was_hit = (planes == self.plane) & (row == self.tokens)
        require(was_hit[:, :self.hits].all() & ~was_hit[:, self.hits:].any(), "unexpected hit/miss pattern")
        require((self.coordinator.req_device_buffer_tokens[planes, slots, row] == self.tokens).all(),
                "reverse token disagrees with output")
        if self.runtime is not None:
            tag = (self.runtime.generation[slots] << 16) | self.plane
            require((self.runtime.row_owner[slots, planes * g.hot + row] == tag).all(), "wrong owner tag")
            require(self.runtime.status.item() == 0, "resolver status bits are set")
        for buf, history in zip((self.k, self.v), self.host):
            require(torch.equal(buf.flatten(0, 1)[idx].cpu(), history[self.host_rows]), "returned KV bytes differ")
        hits = int(was_hit.sum())
        return dict(hits=hits, misses=idx.numel() - hits)


def run_case(torch, arm, g, miss_blocks, repeats, warmup):
    torch.cuda.reset_peak_memory_stats()
    case = Case(torch, arm, g, miss_blocks)
    # Build all specializations before capture. This eager launch also validates
    # output locations before the first sparse payload reset.
    case.launch()
    counts = case.verify()
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
            if case.verify() != counts:
                raise RuntimeError("Graph replay changed the hit/miss counts")
    domain = case.runtime.plan.domain(case.plane)[1] if case.runtime else g.hot
    return dict(arm=arm, geometry=asdict(g), measured_plane=case.plane, device_planes=case.k.size(0),
                blocks_per_grid=g.slots, threads_per_cta=960 if arm == "stock" else BLOCK_SIZE,
                domain_rows_per_request=domain, selected_rows_per_request=case.selected, miss_blocks=miss_blocks,
                actual_counts_per_launch=counts, miss_kv_bytes_per_launch=counts["misses"] * 4 * g.dim,
                host_pinned_bytes=sum(t.numel() * t.element_size() for t in case.host),
                device_kv_bytes=case.k.numel() * 4, torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                kernel_us=dict(median=statistics.median(timings), samples=timings))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cuda-test", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=("stock", "isolated", "shared"),
                        default=["stock", "isolated", "shared"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 8, 60])
    parser.add_argument("--miss-blocks", nargs="+", type=int, default=[0, 4])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
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
    results = []
    for batch in args.batches:
        for layers in args.layers:
            for misses in args.miss_blocks:
                for arm in args.arms:
                    result = run_case(torch, arm, replace(GEOMETRY, slots=batch, layers=layers),
                                      misses, args.repeats, args.warmup)
                    results.append(result)
                    print(json.dumps(result), flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
    summary = dict(event="hisparse_resolver_scan_microbench", device=torch.cuda.get_device_name(),
                   torch=str(torch.__version__), cuda=torch.version.cuda, warmup=args.warmup,
                   repeats=args.repeats, dtype="bfloat16", kv_heads_per_rank=1, diagnostics="none",
                   arena_kernel_sha256=hashlib.sha256(KERNEL_SOURCE.read_bytes()).hexdigest(),
                   timing="external CUDA events inside one graph, one complete resolver, reset before replay",
                   cache_condition="synthetic contiguous locations, metadata touched by device reset before each sample",
                   results=results)
    with args.output.open("x") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
