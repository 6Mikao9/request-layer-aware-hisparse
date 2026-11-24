"""CUDA correctness gate for the combined layout (fixed offload set S + arena). Explicit invocation only.

Requires the pinned SGLang image, an idle A100 (sm_80) and the repository root on
PYTHONPATH; never run it on an active service. No model is loaded. One synthetic
rank holds resident planes (complete main KV at page-granular logical locations)
and an arena over the offloaded planes only. Each decode step runs, per layer in
model order, the path the adapter dispatches: offloaded layers resolve through
``rlh_shared_lru::resolve`` (checked against ``KernelMirror`` like the arena gate)
and attend with HiSparse slots over the arena view; resident layers attend through
``req_to_token`` with no slots. The pinned consumer ``flash_decode_with_gqa_share_sparse``
must give bit-identical outputs over a dense logical twin of each offloaded layer,
the paged-contiguous resident read must equal its gather read, and every output
must match an FP32 softmax over the host-history K/V of the selected tokens.
Eager and CUDA Graph replay (one bucket, changing real rows) are both covered.

    python -m runtime.cuda_combined_correctness --allow-cuda-test --output /tmp/combined-gate.json
"""
import argparse
import json
from pathlib import Path
import random

from runtime import cuda_shared_lru_correctness as gate
from runtime import shared_lru as reference
from runtime import study_config as sc

LAYERS, OFFLOAD = (0, 1, 2, 3), (1, 3)
GEOMETRY = gate.Geometry(layers=len(OFFLOAD), hot=128, page=16, slots=4, blocks=4, block=16, dim=64, context=1024)
Q_HEADS, BUCKET, ATOL, RTOL = 4, 4, 1.0, 1e-2
ARMS = {
    "isolated": {"mode": "isolated", "layout": "arena"},
    "shared": {"mode": "shared", "layout": "arena"},
    "shared_fifo": {"mode": "shared", "layout": "arena", "eviction": "fifo"},
    "isolated_static": {"mode": "isolated", "layout": "arena", "quota": "static", "quota_rows": {"1": 96, "3": 160}},
    "shared_soft_floor": {"mode": "shared", "layout": "arena", "quota": "soft_floor",
                          "floor_rows": {"1": 100, "3": 100}},
    "shared_bypass": {"mode": "shared", "layout": "arena", "admission": "bypass_list", "bypass_layers": [3]},
}


def config_for(pool):
    return sc.parse({"schema": sc.SCHEMA, "placement": {"mode": "fixed_layers", "offload_layers": list(OFFLOAD)},
                     "hot_pool": pool})


def load_consumer():
    from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import flash_decode_with_gqa_share_sparse
    return flash_decode_with_gqa_share_sparse


class CombinedBench:
    """Arena bench over the offloaded planes, resident planes, logical twins and the decode consumer."""

    def __init__(self, torch, config, seed, consumer, device="cuda", geometry=GEOMETRY):
        from runtime.shared_lru_device import ArenaRuntime
        g, bf16 = geometry, torch.bfloat16
        self.torch, self.consumer, self.device, self.g = torch, consumer, device, geometry
        plain = sc.parse({"schema": sc.SCHEMA, "hot_pool": {"mode": "isolated", "layout": "arena"}})
        self.arena = gate.Bench(torch, plain, seed, device, geometry)
        pool = self.arena.coordinator.mem_pool_device        # bind the arena to absolute layers S, like the adapter
        pool._rlh_offload_layers = pool.model_layer_ids = OFFLOAD
        pool.end_layer = len(OFFLOAD)
        self.arena.runtime = ArenaRuntime(self.arena.coordinator, config, "0" * 64, torch=torch)
        self.arena.mirror = reference.KernelMirror(self.arena.runtime.plan, plane_rows=self.arena.plane_rows)
        rng = random.Random(seed ^ 0x5EED)
        per = g.context // g.page
        pages = list(range(1, 1 + g.slots * per))           # page 0 is the padding sink
        rng.shuffle(pages)
        r2t = [[pages[s * per + t // g.page] * g.page + t % g.page for t in range(g.context)] for s in range(g.slots)]
        self.req_to_token = torch.tensor(r2t, dtype=torch.int32, device=device)
        rows, logical_rows = g.slots * g.context, (1 + g.slots * per) * g.page
        locations = self.req_to_token.reshape(-1).long()
        self.history, self.planes = {}, {}
        for layer in LAYERS:
            if layer in OFFLOAD:                              # the arena bench already holds this host history
                plane = OFFLOAD.index(layer)
                k, v = self.arena.host_k[plane], self.arena.host_v[plane]
            else:
                k, v = (self._history(torch, layer, salt, rows) for salt in (0, 101))
            self.history[layer] = (k, v)
            planes = tuple(torch.full((logical_rows, 1, g.dim), -999.0, dtype=bf16, device=device) for _ in "kv")
            for plane, source in zip(planes, (k, v)):
                plane.index_copy_(0, locations, source.to(device))
            self.planes[layer] = planes                       # resident plane, or the logical twin of an S layer
        self.q_rng = random.Random(seed ^ 0xC0FFEE)

    def _history(self, torch, layer, salt, rows):
        row = torch.arange(rows, dtype=torch.int64).unsqueeze(1)
        element = torch.arange(self.g.dim, dtype=torch.int64).unsqueeze(0)
        mix = (row * 11 + element * 5 + 37 * layer + salt) % 251
        values = torch.where(element == 0, row % 251, torch.where(element == 1, (row // 251) % 251, mix))
        return (values - 125).reshape(rows, 1, self.g.dim).to(torch.bfloat16)

    def queries(self, bucket):
        torch = self.torch
        values = [self.q_rng.gauss(0.0, 0.01) for _ in range(bucket * Q_HEADS * self.g.dim)]
        return torch.tensor(values, dtype=torch.float32).reshape(bucket, Q_HEADS, self.g.dim).to(
            torch.bfloat16).to(self.device)

    def forward(self, layer, rpi, seq, blocks, q):
        """The dispatched decode of one layer plus its comparison read; returns (main, twin) outputs."""
        g, consumer, topk = self.g, self.consumer, blocks.unsqueeze(0)
        common = dict(q=q, sink=None, req_to_token=self.req_to_token, seq_lens=seq, slot_ids=rpi,
                      block_size=g.block, topk_idx=topk)
        twin_k, twin_v = self.planes[layer]
        if layer in OFFLOAD:
            runtime = self.arena.runtime
            slots = runtime.resolve(self.arena.coordinator, rpi, seq, blocks, layer, g.block)
            main = consumer(k_cache=runtime.flat_k, v_cache=runtime.flat_v, page_size=g.page,
                            hisparse_slots=slots.unsqueeze(0), **common)
            return main, consumer(k_cache=twin_k, v_cache=twin_v, page_size=g.page, hisparse_slots=None, **common)
        main = consumer(k_cache=twin_k, v_cache=twin_v, page_size=g.page, hisparse_slots=None, **common)
        return main, consumer(k_cache=twin_k, v_cache=twin_v, page_size=0, hisparse_slots=None, **common)

    def reference(self, layer, slot, blocks, seq_len, q):
        """FP32 softmax over the host-history K/V of every selected position, in selection order."""
        torch, g = self.torch, self.g
        k, v = self.history[layer]
        tokens = [t for t in reference.selected_tokens(blocks, g.block, seq_len) if t >= 0]
        rows = torch.tensor([slot * g.context + t for t in tokens], dtype=torch.long)
        keys, values = k.index_select(0, rows)[:, 0].float(), v.index_select(0, rows)[:, 0].float()
        scores = (q.float().cpu() @ keys.T) * g.dim ** -0.5
        return torch.softmax(scores, dim=-1) @ values

    def check(self, layer, rows, q, outputs):
        torch = self.torch
        main, twin = outputs
        if self.device == "cuda":
            torch.cuda.synchronize()
        for b, (slot, blocks) in enumerate(rows):
            got = main[b].float().cpu()
            assert torch.equal(main[b], twin[b]), f"layer {layer} row {b}: dispatched read differs from its twin"
            expected = self.reference(layer, slot, blocks, self.arena.seq[slot], q[b])
            error = (got - expected).abs().max().item()
            assert torch.allclose(got, expected, atol=ATOL, rtol=RTOL), f"layer {layer} row {b}: max error {error}"


def run_case(torch, pool, *, seq_dtype, rpi_dtype, graph, seed, steps, consumer, device="cuda"):
    bench = CombinedBench(torch, config_for(pool), seed, consumer, device)
    arena, H = bench.arena, GEOMETRY.hot
    for slot, prompt in enumerate((H + 5, H + 200, H - 3)):   # long, longer, short -> long transition
        arena.admit(slot, prompt)
    graphs, outputs, static = {}, {}, None
    if graph:
        static = arena.tensors([(slot, [0] * GEOMETRY.blocks) for slot in range(3)], seq_dtype, rpi_dtype, BUCKET)
        static = (*static, bench.queries(BUCKET))
        arena.coordinator.num_real_reqs.fill_(0)            # warmup and capture mutate no arena state
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for layer in LAYERS:
                bench.forward(layer, *static)
        torch.cuda.current_stream().wait_stream(side)
        for layer in LAYERS:
            graphs[layer] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[layer]):
                outputs[layer] = bench.forward(layer, *static)
        torch.cuda.synchronize()
    launches = 0
    for step in range(steps):
        active = [(step + k) % 3 for k in range((1, 3, 2, 3)[step % 4])]   # real rows change per replay
        for layer in LAYERS:
            rows = [(slot, arena.selection(slot)) for slot in active]
            expected = ([arena.expect(slot, OFFLOAD.index(layer), blocks) for slot, blocks in rows]
                        if layer in OFFLOAD else None)
            bucket = BUCKET if graph else len(rows)
            tensors = (*arena.tensors(rows, seq_dtype, rpi_dtype, bucket), bench.queries(bucket))
            arena.coordinator.num_real_reqs.fill_(len(rows))
            if graph:
                for target, source in zip(static, tensors):
                    target.copy_(source)
                graphs[layer].replay()
                result = outputs[layer]
            else:
                result = bench.forward(layer, *tensors)
            if expected is not None:
                arena.check(OFFLOAD.index(layer), rows, expected, bucket)
            bench.check(layer, rows, tensors[3], result)
            launches += 1
        for slot in active:
            arena.advance(slot)
    return launches


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--allow-cuda-test", action="store_true")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--output", type=Path, help="optional JSON result; never overwritten")
    args = parser.parse_args()
    if not args.allow_cuda_test:
        parser.error("Run only in an approved idle test process with --allow-cuda-test")
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    import torch
    import sglang
    from runtime.layer_placement_device import verify_digests
    bad = {k: v for k, v in verify_digests(Path(sglang.__file__).resolve().parent).items() if v != "ok"}
    if bad:
        raise RuntimeError(f"Pinned SGLang source mismatch: {bad}")
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("A100 (sm_80) required")
    consumer, results = load_consumer(), []
    for index, (name, pool) in enumerate(ARMS.items()):
        for seq_dtype, rpi_dtype in ((torch.int32, torch.int64), (torch.int64, torch.int32)):
            for graph in (False, True):
                launches = run_case(torch, pool, seq_dtype=seq_dtype, rpi_dtype=rpi_dtype, graph=graph,
                                    seed=args.seed + index, steps=args.steps, consumer=consumer)
                results.append(dict(arm=name, seq_dtype=str(seq_dtype), rpi_dtype=str(rpi_dtype), graph=graph,
                                    launches=launches, status="pass"))
    summary = dict(event="hisparse_combined_cuda_gate", status="pass", layers=list(LAYERS), offload=list(OFFLOAD),
                   cases=len(results), launches=sum(r["launches"] for r in results),
                   device=torch.cuda.get_device_name(), torch=torch.__version__, results=results)
    if args.output is not None:
        with args.output.open("x") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"PASS {summary['cases']} cases / {summary['launches']} layer launches: arena tables and bytes on S, "
          "resident direct reads, twin-exact consumer outputs; eager and graph")


if __name__ == "__main__":
    main()