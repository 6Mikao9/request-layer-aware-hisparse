"""CUDA correctness gate for ``rlh_shared_lru::resolve``. Explicit invocation only.

Requires the pinned SGLang image, an idle A100 (sm_80) and the repository root on
PYTHONPATH; never run it on an active service. No model is loaded. Every launch
is compared with ``runtime.shared_lru.KernelMirror`` on the same synthetic
tables: flat consumer indices, owner tags, recency lists, occupancy, reverse
tokens, counters, status bits and the K/V bytes behind every returned index.
Eager launches and CUDA Graph replay (one bucket, changing real rows, padded
rows) are both covered. This is a correctness check, not a benchmark.

    python -m runtime.cuda_shared_lru_correctness --allow-cuda-test --output /tmp/gate.json
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import types

from runtime import shared_lru as reference
from runtime import study_config as sc

ARMS = {
    "isolated": {"mode": "isolated", "layout": "arena"},
    "shared": {"mode": "shared", "layout": "arena"},
    "shared_fifo": {"mode": "shared", "layout": "arena", "eviction": "fifo"},
    "isolated_static": {"mode": "isolated", "layout": "arena", "quota": "static",
                        "quota_rows": {"0": 32, "1": 96, "2": 64}},
    "shared_soft_floor": {"mode": "shared", "layout": "arena", "quota": "soft_floor",
                          "floor_rows": {"0": 60, "1": 60, "2": 60}},
    "shared_bypass": {"mode": "shared", "layout": "arena", "admission": "bypass_list", "bypass_layers": [1]},
}
L, H, PAGE, SLOTS, BLOCKS, BLOCK, DIM, BUCKET = 3, 64, 16, 4, 4, 8, 8, 4
CONTEXT = 4096


@dataclass(frozen=True)
class Geometry:
    layers: int = L
    hot: int = H
    page: int = PAGE
    slots: int = SLOTS
    blocks: int = BLOCKS
    block: int = BLOCK
    dim: int = DIM
    context: int = CONTEXT


class Bench:
    """Synthetic coordinator, contiguous planes and pinned host history on one A100."""

    def __init__(self, torch, config, seed, device="cuda", geometry=Geometry()):
        from runtime.shared_lru_device import ArenaRuntime
        self.torch, self.rng, self.device = torch, random.Random(seed), device
        self.geometry = geometry
        cuda, bf16 = torch.device(device), torch.bfloat16
        pin = device == "cuda"
        plane_rows = geometry.page + geometry.slots * (geometry.hot + geometry.page)
        k_all = torch.full((geometry.layers, plane_rows, 1, geometry.dim), -999.0, dtype=bf16, device=cuda)
        v_all = torch.full_like(k_all, -999.0)
        pool = types.SimpleNamespace(start_layer=0, layer_num=geometry.layers, page_size=geometry.page, k_buffer=list(k_all.unbind(0)),
                                     v_buffer=list(v_all.unbind(0)), _rlh_contiguous_kv=(k_all, v_all))
        rows = geometry.slots * geometry.context
        # Exact BF16 integers: elements 0/1 encode the host row, the rest mix plane and K/V salt.
        row = torch.arange(rows, dtype=torch.int64).unsqueeze(1)
        element = torch.arange(geometry.dim, dtype=torch.int64).unsqueeze(0)

        def history(plane, salt):
            mix = (row * 7 + element * 13 + 31 * plane + salt) % 251
            values = torch.where(element == 0, row % 251, torch.where(element == 1, (row // 251) % 251, mix))
            values = (values - 125).reshape(rows, 1, geometry.dim).to(bf16)
            return values.pin_memory() if pin else values

        self.host_k = [history(p, 0) for p in range(geometry.layers)]
        self.host_v = [history(p, 101) for p in range(geometry.layers)]
        locs = list(range(geometry.page, plane_rows))
        self.rng.shuffle(locs)
        self.locs = [locs[s * (geometry.hot + geometry.page):(s + 1) * (geometry.hot + geometry.page)] for s in range(geometry.slots)]
        i32 = torch.int32
        self.coordinator = types.SimpleNamespace(
            mem_pool_device=pool, device_buffer_size=geometry.hot, top_k=geometry.blocks * geometry.block, item_size_bytes=geometry.dim * 2,
            req_device_buffer_tokens=torch.full((geometry.layers, geometry.slots, geometry.hot + geometry.page), -1, dtype=i32, device=cuda),
            req_device_buffer_token_locs=torch.tensor([self.locs] * geometry.layers, dtype=i32, device=cuda),
            lru_slots=torch.zeros((geometry.layers, geometry.slots, geometry.hot), dtype=torch.int16, device=cuda),
            top_k_device_locs_buffer=torch.full((geometry.slots, geometry.hot), -7, dtype=i32, device=cuda),
            num_real_reqs=torch.zeros(1, dtype=i32, device=cuda),
            req_to_host_pool=(torch.arange(geometry.slots * geometry.context, dtype=torch.int64, device=cuda).reshape(geometry.slots, geometry.context)),
            mem_pool_host=types.SimpleNamespace(k_buffer=self.host_k, v_buffer=self.host_v))
        self.runtime = ArenaRuntime(self.coordinator, config, "0" * 64, torch=torch)
        self.mirror = reference.KernelMirror(self.runtime.plan, plane_rows=plane_rows)
        self.k_all, self.v_all, self.plane_rows = k_all, v_all, plane_rows
        self.seq = {}
        self.counters = {p: dict.fromkeys(reference.COUNTER_FIELDS, 0) for p in range(geometry.layers)}
    def _write(self, slot, index, token):
        """Stock KV write of one token into every plane's slot ``index`` (position or tail)."""
        host_row = slot * self.geometry.context + token
        loc = self.locs[slot][index]
        for p in range(self.geometry.layers):
            self.k_all[p, loc].copy_(self.host_k[p][host_row])
            self.v_all[p, loc].copy_(self.host_v[p][host_row])

    def admit(self, slot, prompt):
        """Staged prompt (native alloc_device_buffer tables) plus the first decode token."""
        torch = self.torch
        self.coordinator.req_device_buffer_tokens[:, slot, :self.geometry.hot] = torch.arange(self.geometry.hot, dtype=torch.int32)
        allocation = self.runtime._allocations[slot]
        self.runtime.init_slot(slot)
        self.mirror.init_slot(slot, allocation, self.locs[slot])
        count = min(prompt, self.geometry.hot)
        indices = torch.tensor(self.locs[slot][:count], dtype=torch.long, device=self.device)
        start = slot * self.geometry.context
        for plane in range(self.geometry.layers):
            for buffer, history in ((self.k_all, self.host_k), (self.v_all, self.host_v)):
                rows = history[plane][start:start + count].to(self.device)
                buffer[plane].index_copy_(0, indices, rows)
        self.seq[slot] = prompt
        self.advance(slot)

    def advance(self, slot):
        token = self.seq[slot]
        self._write(slot, min(token, self.geometry.hot), token)
        self.seq[slot] = token + 1

    def selection(self, slot):
        top = (self.seq[slot] + self.geometry.block - 1) // self.geometry.block
        blocks = [self.rng.randrange(top) for _ in range(self.geometry.blocks)]
        if self.rng.random() < 0.5:
            blocks[-1] = top - 1
        if self.rng.random() < 0.2:
            blocks[-1] = -1
        if self.rng.random() < 0.2:
            blocks[1] = blocks[0]
        return blocks

    def expect(self, slot, plane, blocks):
        flat, _, counters = self.mirror.resolve_row(slot, plane, blocks, self.geometry.block, self.seq[slot])
        if self.seq[slot] > self.geometry.hot:  # the kernel counts only on the long path
            tokens = reference.selected_tokens(blocks, self.geometry.block, self.seq[slot])
            totals = self.counters[plane]
            for name in ("unique", "hits", "misses", "bypassed", "breaches"):
                totals[name] += counters[name]
            totals["bytes"] += 2 * self.geometry.dim * 2 * counters["misses"]
            totals["latest"] += tokens.count(self.seq[slot] - 1)
            totals["invalid"] += tokens.count(-1)
        return flat
    def check(self, plane, rows, expected, padded):
        """Compare one launch with the mirror: indices, tables, bytes and untouched padded slots."""
        torch = self.torch
        if self.device == "cuda":
            torch.cuda.synchronize()
        out = self.coordinator.top_k_device_locs_buffer
        width = self.geometry.blocks * self.geometry.block
        flat_k, flat_v = self.runtime.flat_k, self.runtime.flat_v
        for b, (slot, blocks) in enumerate(rows):
            got = out[b, :width].tolist()
            assert got == expected[b], f"plane {plane} slot {slot}: indices {got} != mirror {expected[b]}"
            state = self.mirror.slots[slot]
            assert self.runtime.row_owner[slot].tolist() == state["owner"], "owner tags diverged"
            assert self.runtime.order[slot].tolist() == state["order"], "recency list diverged"
            assert self.runtime.occupancy[slot].tolist() == state["occ"], "occupancy diverged"
            tokens = self.coordinator.req_device_buffer_tokens[:, slot, :self.geometry.hot].reshape(-1).tolist()
            assert tokens == state["token"], "reverse token table diverged"
            selected = reference.selected_tokens(blocks, self.geometry.block, self.seq[slot])
            valid = [(got[i], slot * self.geometry.context + t) for i, t in enumerate(selected) if t >= 0]
            if valid:
                locations, host_rows = zip(*valid)
                locations = torch.tensor(locations, dtype=torch.long, device=self.device)
                host_rows = torch.tensor(host_rows, dtype=torch.long)
                for label, flat, history in (("K", flat_k, self.host_k), ("V", flat_v, self.host_v)):
                    actual = flat.index_select(0, locations).cpu()
                    assert torch.equal(actual, history[plane].index_select(0, host_rows)), f"{label} bytes differ"
        for b in range(len(rows), padded):
            assert out[b, :width].tolist() == [-1] * width, "padded graph row must stay invalid"
        assert int(self.runtime.status.item()) == 0, f"status bits {int(self.runtime.status.item())}"

    def tensors(self, rows, seq_dtype, rpi_dtype, bucket):
        torch = self.torch
        slots = [slot for slot, _ in rows] + [self.geometry.slots - 1] * (bucket - len(rows))
        seqs = [self.seq[slot] for slot, _ in rows] + [1] * (bucket - len(rows))
        blocks = [list(b) for _, b in rows] + [[0] * self.geometry.blocks] * (bucket - len(rows))
        return (torch.tensor(slots, dtype=rpi_dtype, device=self.device),
                torch.tensor(seqs, dtype=seq_dtype, device=self.device),
                torch.tensor(blocks, dtype=torch.int32, device=self.device))

    def check_counters(self):
        if self.runtime.counters.numel():
            got = self.runtime.counter_snapshot()["layers"]
            for plane in range(self.geometry.layers):
                assert got[str(plane)] == self.counters[plane], f"counters plane {plane}: {got[str(plane)]}"
def run_case(torch, pool, *, counters, seq_dtype, rpi_dtype, graph, seed, steps, device="cuda"):
    config = sc.parse({"schema": sc.SCHEMA, "hot_pool": pool,
                       "diagnostics": {"mode": "counters" if counters else "none"}})
    bench = Bench(torch, config, seed, device)
    for slot, prompt in enumerate((H + 5, H + 200, H - 3)):   # long, longer, short -> long transition
        bench.admit(slot, prompt)
    graphs, static = {}, None
    if graph:
        static = bench.tensors([(slot, [0] * BLOCKS) for slot in range(3)], seq_dtype, rpi_dtype, BUCKET)
        bench.coordinator.num_real_reqs.fill_(0)            # warmup and capture mutate nothing
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for plane in range(L):
                bench.runtime.resolve(bench.coordinator, *static, plane, BLOCK)
        torch.cuda.current_stream().wait_stream(side)
        for plane in range(L):
            graphs[plane] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[plane]):
                bench.runtime.resolve(bench.coordinator, *static, plane, BLOCK)
        torch.cuda.synchronize()
    launches = 0
    for step in range(steps):
        active = [(step + k) % 3 for k in range((1, 3, 2, 3)[step % 4])]   # real rows change per replay
        for plane in range(L):
            rows = [(slot, bench.selection(slot)) for slot in active]
            expected = [bench.expect(slot, plane, blocks) for slot, blocks in rows]
            bucket = BUCKET if graph else len(rows)
            tensors = bench.tensors(rows, seq_dtype, rpi_dtype, bucket)
            bench.coordinator.num_real_reqs.fill_(len(rows))
            if graph:
                for target, source in zip(static, tensors):
                    target.copy_(source)
                graphs[plane].replay()
            else:
                bench.runtime.resolve(bench.coordinator, *tensors, plane, BLOCK)
            bench.check(plane, rows, expected, bucket)
            launches += 1
        for slot in active:
            bench.advance(slot)
    bench.check_counters()
    return launches


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--allow-cuda-test", action="store_true")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--output", type=Path, help="optional JSON result; never overwritten")
    args = parser.parse_args()
    if not args.allow_cuda_test:
        parser.error("Run only in an approved idle test process with --allow-cuda-test")
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    import torch
    import sglang
    from runtime.shared_lru_device import verify_digests
    bad = {k: v for k, v in verify_digests(Path(sglang.__file__).resolve().parent).items() if v != "ok"}
    if bad:
        raise RuntimeError(f"Pinned SGLang source mismatch: {bad}")
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("A100 (sm_80) required")
    results = []
    for index, (name, pool) in enumerate(ARMS.items()):
        for seq_dtype in (torch.int32, torch.int64):
            for rpi_dtype in (torch.int32, torch.int64):
                for graph in (False, True):
                    launches = run_case(torch, pool, counters=name == "shared_soft_floor", seq_dtype=seq_dtype,
                                        rpi_dtype=rpi_dtype, graph=graph, seed=args.seed + index, steps=args.steps)
                    results.append(dict(arm=name, seq_dtype=str(seq_dtype), rpi_dtype=str(rpi_dtype),
                                        graph=graph, launches=launches, status="pass"))
    summary = dict(event="hisparse_shared_lru_cuda_gate", status="pass", cases=len(results),
                   launches=sum(r["launches"] for r in results), device=torch.cuda.get_device_name(),
                   torch=torch.__version__, results=results)
    if args.output is not None:
        with args.output.open("x") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"PASS {summary['cases']} cases / {summary['launches']} launches: indices, tables, bytes; eager and graph")


if __name__ == "__main__":
    main()
