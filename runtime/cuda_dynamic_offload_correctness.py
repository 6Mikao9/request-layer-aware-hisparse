"""CUDA correctness gate of the dynamic offload route kernel and slab executor (issue #11). Explicit invocation only.

Requires the pinned SGLang image, an idle A100 (sm_80) and the repository root on PYTHONPATH; no model is loaded.

* ``history``: the CPU test history (short, page-crossing and long requests, buffer growth, slot reuse, padded graph
  rows, admission next to a switch and the switch sequence A->B->A->C->B->C->A) on CUDA. The route kernel runs
  eagerly in one pass and from CUDA Graphs captured once per (layer, bucket) before the first switch in another;
  both are compared after every event with the same history through ``emulate_route``: every read returns the
  selected token, and outputs, native tables, slab and host match bit for bit. A rebuild failure injected during a
  switch is rolled back from host and decoding continues.
* ``production``: MiniMax-M3 (H 4096, 16 blocks of 128 tokens, BF16 rows) and DeepSeek-V4 C4 (H 1024, top-512,
  page-padded 584-byte records) per-request geometry with short and long requests. Each step runs the kernel and the
  statement on identical copies; a switch between steps moves both layers through the real executor and the next
  step replays the graph captured before it.

    python -m runtime.cuda_dynamic_offload_correctness --allow-cuda-test --output /tmp/dynamic-gate.json
"""
import argparse
import copy
import json
from pathlib import Path
import random
import types

from runtime import dynamic_offload as do
from runtime import dynamic_offload_device as dev

DEVICE = "cuda"          # the CPU tests run the harness itself on "cpu" with the statement in place of the kernel
SPECS = {
    "minimax_m3": dict(hot=4096, page=128, width=16, block=128, ratio=1, records=(3000, 6000, 9000, 12000)),
    "deepseek_v4": dict(hot=1024, page=64, width=512, block=1, ratio=4, records=(700, 1500, 2600, 4100)),
}


def coordinator(torch, layers, slots, width, ctx, item, block_size, hot, page):
    d, cuda = torch.device(DEVICE), DEVICE == "cuda"
    c = types.SimpleNamespace(
        req_to_token_pool=types.SimpleNamespace(req_to_token=torch.zeros((slots, ctx), dtype=torch.int32, device=d)),
        req_to_device_buffer=torch.zeros((slots, hot + page), dtype=torch.int64, device=d),
        req_device_buffer_size=torch.zeros(slots, dtype=torch.int64), req_to_host_pool=torch.full((slots, ctx), -1,
                                                                                                   device=d),
        req_device_buffer_tokens=torch.full((layers, slots, hot + page), -1, dtype=torch.int32, device=d),
        req_device_buffer_token_locs=torch.full((layers, slots, hot + page), -1, dtype=torch.int32, device=d),
        lru_slots=torch.arange(hot, dtype=torch.int16, device=d).repeat(layers, slots, 1),
        top_k_device_locs_buffer=torch.full((slots, width), -1, dtype=torch.int32, device=d),
        num_real_reqs=torch.zeros(1, dtype=torch.int32, device=d), decode_backup_stream=torch.cuda.Stream() if cuda
        else None, decode_producer_stream=None, item_size_bytes=item, swap_in_block_size=block_size,
        device_buffer_size=hot, _backup_done_event=torch.cuda.Event() if cuda else None, _has_pending_backup=False)

    def wait_for_pending_backup():
        if c._has_pending_backup:
            c._backup_done_event.wait(torch.cuda.current_stream())
            c._has_pending_backup = False

    c.wait_for_pending_backup = wait_for_pending_backup
    return c


class GraphSwap:
    """``route_swap`` from CUDA Graphs: the first call per (layer, bucket) runs eagerly on static inputs and then
    captures; later calls copy the step's inputs into those buffers and replay. Never re-captured."""

    def __init__(self, torch, width, block=1):
        self.torch, self.width, self.block, self.graphs, self.replays = torch, width, block, {}, 0

    def __call__(self, s, c, rpi, seq_lens, chosen, layer, out):
        t, key = self.torch, (layer, len(chosen))
        top_k = t.full((len(chosen), self.width), -1, dtype=t.int32)
        for b, picks in enumerate(chosen):
            top_k[b, :len(picks)] = t.tensor(picks, dtype=t.int32)
        if key not in self.graphs:
            static = [rpi.clone(), seq_lens.clone(), top_k.cuda()]
            result = dev.route_swap(s, c, *static[:2], static[2], layer, self.block)
            t.cuda.synchronize()
            graph = t.cuda.CUDAGraph()
            with t.cuda.graph(graph):
                dev.route_swap(s, c, *static[:2], static[2], layer, self.block)
            self.graphs[key] = (graph, static)
            return result
        graph, static = self.graphs[key]
        for target, value in zip(static, (rpi, seq_lens, top_k)):
            target.copy_(value)
        graph.replay()
        self.replays += 1
        return out


def eager_swap(torch, width):
    def swap(s, c, rpi, seq_lens, chosen, layer, out):
        top_k = torch.full((len(chosen), width), -1, dtype=torch.int32)
        for b, picks in enumerate(chosen):
            top_k[b, :len(picks)] = torch.tensor(picks, dtype=torch.int32)
        return dev.route_swap(s, c, rpi, seq_lens, top_k.cuda(), layer, 1)
    return swap
def cuda_rank(torch, offload, swap, host_shadow):
    from runtime.test_dynamic_offload_device import DeviceRank
    rank = DeviceRank(offload, host_shadow=host_shadow, device="cuda", swap=swap)
    stream_state = coordinator(torch, 1, 1, 1, 1, 16, 256, 8, 4)
    for name in ("decode_backup_stream", "_backup_done_event", "_has_pending_backup", "wait_for_pending_backup"):
        setattr(rank.c, name, getattr(stream_state, name))
    return rank


def same_state(torch, a, b):
    """Names of the device state that differs between two ranks (empty when bit-equal)."""
    pairs = dict(slab_k=(a.k, b.k), slab_v=(a.v, b.v), route=(a.s.route, b.s.route),
                 **{name: (getattr(a.c, name), getattr(b.c, name)) for name in (
                     "req_device_buffer_tokens", "req_device_buffer_token_locs", "lru_slots", "req_to_host_pool")})
    pairs.update({f"host_{kind}{i}": (x, y) for kind in ("k", "v") for i, (x, y) in enumerate(zip(
        getattr(a.s.rows.host, f"{kind}_buffer"), getattr(b.s.rows.host, f"{kind}_buffer")))})
    return [name for name, (x, y) in pairs.items() if not torch.equal(x.cpu(), y.cpu())]


def history(torch, *, graph, host_shadow, rollback_at=7):
    """The CPU test history on CUDA: kernel (eager or graph) against emulate_route after every event."""
    from runtime.test_dynamic_offload import LISTS, SWITCHED, Rank, selections, steps
    swap = GraphSwap(torch, 8) if graph else eager_swap(torch, 8)
    kernel, statement = cuda_rank(torch, LISTS["A"], swap, host_shadow), cuda_rank(torch, LISTS["A"], None,
                                                                                  host_shadow)
    model = Rank(LISTS["A"], host_shadow=host_shadow)
    events = [*SWITCHED[:rollback_at], ("rollback", "C"), *SWITCHED[rollback_at:],
              ("admit", 4, "z", 17), *steps((1, 2, 3, 4), 4, padded=0)]
    rng, problems, reads = random.Random(11), [], 0
    for index, (kind, *args) in enumerate(events):
        if kind == "step":
            chosen = selections(rng, model, args[0])
            logs = [rank.step(args[0], chosen, padded=args[1]) for rank in (kernel, statement, model)]
            reads += sum(len(v) for v in logs[0].values())
            if not logs[0] == logs[1] == logs[2]:
                problems.append(f"event {index}: reads differ")
        elif kind == "rollback":                  # a rebuild failure in the middle of a switch, then the inverse
            for rank in (kernel, statement):
                plan, h2d, calls = do.plan_switch(rank.s.layout, rank.s.routes, LISTS[args[0]]), rank.s.rows.h2d, []

                def failing(*a, h2d=h2d, calls=calls):
                    calls.append(a)
                    if len(calls) == 2:
                        raise RuntimeError("injected copy failure")
                    return h2d(*a)

                rank.s.prepare(plan)
                rank.s.rows.h2d = failing
                try:
                    rank.s.apply(plan)
                    problems.append("injected failure did not stop the rebuild")
                except RuntimeError:
                    pass
                rank.s.rows.h2d = h2d
                rank.s.apply(do.rollback_plan(plan))
        else:
            for rank in (kernel, statement, model):
                getattr(rank, kind)(*args)
        torch.cuda.synchronize()
        problems += [f"event {index} ({kind}): {name}" for name in same_state(torch, kernel, statement)]
    return dict(graph=graph, host_shadow=host_shadow, events=len(events), reads=reads,
                graph_replays=getattr(swap, "replays", None), graphs=len(getattr(swap, "graphs", ())),
                final_routes=kernel.s.route.tolist(), problems=problems[:20], passed=not problems)
def build(torch, model, seed):
    """One rank's slab for two switchable layers (2 offloaded, 4 resident) and four requests at production per-request
    geometry, filled from host truth with plain torch indexing. Same seed, same bytes."""
    spec, t, rng = SPECS[model], torch, random.Random(seed)
    gen = t.Generator().manual_seed(seed)
    mla, H, page, ratio, records = model == "deepseek_v4", spec["hot"], spec["page"], spec["ratio"], spec["records"]
    layers, slots = (2, 4), len(records) + 1
    spans = [-(-(n + 2) // page) * page for n in records]
    geometry = do.Geometry(layers=layers, hot_rows=page + len(records) * (H + page), resident_rows=page + sum(spans),
                           hot_slots=H, slots=slots, page=page, page_bytes=37440 if mla else page * 256,
                           buffers=1 if mla else 2)
    layout = do.SlabLayout(geometry, 1)
    total, width = sum(n + 2 for n in records), spec["width"] * spec["block"]
    host_rows = -(-total // page) * page + page
    c = coordinator(t, 2, slots, width if mla else H, (max(records) + 2) * ratio, 584 if mla else 256, 960, H, page)
    pin = (lambda x: x.pin_memory()) if DEVICE == "cuda" else (lambda x: x)
    if mla:
        slab = t.randint(0, 255, (layout.rows // page, 37440), dtype=t.uint8, generator=gen).to(DEVICE)
        hosts = [pin(t.randint(0, 255, (host_rows // page, 37440), dtype=t.uint8, generator=gen)) for _ in layers]
        host = types.SimpleNamespace(kv_buffer=hosts, data_ptrs=t.tensor([h.data_ptr() for h in hosts],
                                                                          dtype=t.uint64, device=DEVICE))
        rows = dev.PagedRows(t, slab, host)
    else:
        k, v = (t.randn((layout.rows, 1, 128), generator=gen).to(t.bfloat16).to(DEVICE) for _ in range(2))
        host = types.SimpleNamespace(**{f"{kind}_buffer": [pin(t.randn((host_rows, 1, 128), generator=gen).to(
            t.bfloat16)) for _ in layers] for kind in ("k", "v")})
        rows = dev.MHARows(t, k, v, host)
    s = dev.SlabState(layout, {"0": (2,), "1": (4,)}, rows, active="0", host_shadow=False, ratio=ratio, torch=t,
                      device=t.device(DEVICE))
    s.coordinator = c
    perm, offset, logical = list(range(total)), 0, page
    rng.shuffle(perm)
    truth = {}
    for i, n in enumerate(records):
        slot = i + 1
        c.req_to_token_pool.req_to_token[slot, :(n + 2) * ratio] = t.arange((n + 2) * ratio, dtype=t.int32) \
            + logical * ratio
        h = perm[offset:offset + n + 2]
        c.req_to_host_pool[slot, :n + 2] = t.tensor(h)
        cap = H + page if n > H else min(-(-n // page) * page, H)
        cap += page if cap == H else 0
        buffer = [page + i * (H + page) + j for j in range(cap)]
        c.req_to_device_buffer[slot, :cap] = t.tensor(buffer)
        c.req_device_buffer_size[slot] = cap
        c.req_device_buffer_tokens[:, slot, :H] = t.arange(H, dtype=t.int32)
        s.fix_locs(slot, 0, cap)
        s.home[slot] = n - 1
        truth[slot] = dict(host=h, logical=logical, n=n, buffer=buffer)
        older = list(range(n - 1))
        cached = rng.sample(older, H) if n > H else older
        if n > H:
            order = list(range(H))
            rng.shuffle(order)
            c.req_device_buffer_tokens[0, slot, :H] = t.tensor(cached, dtype=t.int32)
            c.lru_slots[0, slot] = t.tensor(order, dtype=t.int16)
        put(s, 0, [h[x] for x in cached], [buffer[j] for j in range(len(cached))])
        put(s, 1, [h[x] for x in older], [layout.resident_base(0) + logical + x for x in older])
        for index, layer in enumerate(layers):
            put(s, index, [h[n - 1]], [layout.tail_row(layer, slot)])
        offset, logical = offset + n + 2, logical + spans[i]
    return s, c, truth


def put(s, index, host_rows, slab_rows):
    """slab rows <- host rows of one layer, with plain indexing (independent of the transfer kernels)."""
    t, rows = s.torch, s.rows
    src, dst = t.tensor(host_rows, dtype=t.int64), t.tensor(slab_rows, dtype=t.int64, device=DEVICE)
    if rows.buffers == 1:
        for host_part, slab_part in zip(rows._parts(rows.host.kv_buffer[index]), rows._parts(rows.k)):
            slab_part[dst // 64, dst % 64] = host_part[src // 64, src % 64].to(DEVICE)
    else:
        rows.k[dst] = rows.host.k_buffer[index][src].to(DEVICE)
        rows.v[dst] = rows.host.v_buffer[index][src].to(DEVICE)


def holds(s, index, slab_rows, host_rows):
    """True when every slab row holds the bytes of its host row (K and V, or C4 value and scale)."""
    t, rows = s.torch, s.rows
    src, dst = t.tensor(host_rows, dtype=t.int64), t.tensor(slab_rows, dtype=t.int64, device=DEVICE)
    if rows.buffers == 1:
        return all(t.equal(slab_part[dst // 64, dst % 64].cpu(), host_part[src // 64, src % 64])
                   for host_part, slab_part in zip(rows._parts(rows.host.kv_buffer[index]), rows._parts(rows.k)))
    return t.equal(rows.k[dst].cpu(), rows.host.k_buffer[index][src]) and \
        t.equal(rows.v[dst].cpu(), rows.host.v_buffer[index][src])


def choose(rng, model, seq):
    """(kernel selection, expanded positions or None) of one request."""
    spec = SPECS[model]
    if model == "deepseek_v4":
        picks = rng.sample(range(seq - 1), spec["width"] - 1) + [seq - 1]
        rng.shuffle(picks)
        return picks, picks
    count = -(-seq // spec["block"])
    blocks = sorted({0, count - 1} | set(rng.sample(range(count), min(count, spec["width"]) - 2)))[:spec["width"]]
    blocks += [-1] * (spec["width"] - len(blocks))
    if rng.random() < 0.5:
        blocks[rng.randrange(1, spec["width"] - 1)] = -1
    return blocks, [None if b < 0 else b * spec["block"] + o for b in blocks for o in range(spec["block"])]
def production(torch, model, seed=5, kernel=None):
    """Kernel from one captured graph per layer against the statement, before and after a switch of both layers."""
    spec = SPECS[model]
    (sa, ca, truth), (sb, cb, _) = build(torch, model, seed), build(torch, model, seed)
    graph = kernel or GraphSwap(torch, spec["width"], spec["block"])
    rng, problems, compared = random.Random(seed + 1), [], 0
    slots = sorted(truth)
    for step in range(2):
        if step:                                   # switch, then this step's flush and decode store
            for s in (sa, sb):
                plan = do.plan_switch(s.layout, s.routes, (4,))
                s.prepare(plan)
                s.apply(plan)
                s.publish(plan.target)
            for s in (sa, sb):
                s.flush(slots, [truth[x]["n"] - 1 for x in slots], s._long([truth[x]["host"][truth[x]["n"] - 1]
                                                                            for x in slots]))
                for index, layer in enumerate(s.geometry.layers):
                    put(s, index, [truth[x]["host"][truth[x]["n"]] for x in slots],
                        [s.layout.tail_row(layer, x) for x in slots])
            for x in slots:
                truth[x]["n"] += 1
            torch.cuda.synchronize()
        rows = [*slots, 0]
        rpi = sa._long(rows)
        seq = sa._long([truth[x]["n"] for x in slots] + [1])
        for c in (ca, cb):
            c.num_real_reqs[0] = len(slots)
        picks = [choose(rng, model, truth[x]["n"]) for x in slots]
        for index, layer in enumerate(sa.geometry.layers):
            width = spec["width"] * spec["block"]
            got = graph(sa, ca, rpi, seq, [p for p, _ in picks] + [[]], layer, ca.top_k_device_locs_buffer[:5, :width])
            want = dev.emulate_route(sb, cb, rpi, seq, [e for _, e in picks] + [[]], layer,
                                     cb.top_k_device_locs_buffer[:5, :width])
            torch.cuda.synchronize()
            if not torch.equal(got.cpu(), want.cpu()):
                problems.append(f"step {step} layer {layer}: kernel slots differ from the statement")
            for name in ("req_device_buffer_tokens", "lru_slots", "req_device_buffer_token_locs"):
                if not torch.equal(getattr(ca, name).cpu(), getattr(cb, name).cpu()):
                    problems.append(f"step {step} layer {layer}: {name} differs")
            if not all(torch.equal(x.cpu(), y.cpu()) for x, y in ((sa.rows.k, sb.rows.k),
                                                                   *(((sa.rows.v, sb.rows.v),) if spec["block"] > 1
                                                                     else ()))):
                problems.append(f"step {step} layer {layer}: slab bytes differ")
            for b, (x, (_, expanded)) in enumerate(zip(slots, picks)):
                valid = [(i, p) for i, p in enumerate(expanded) if p is not None and p < truth[x]["n"]]
                out = got[b].tolist()
                compared += len(valid)
                if not holds(sa, index, [out[i] for i, _ in valid], [truth[x]["host"][p] for _, p in valid]):
                    problems.append(f"step {step} layer {layer} slot {x}: a slot does not hold its token")
            if got[len(slots):].ne(-1).any():
                problems.append(f"step {step} layer {layer}: padded row is not -1")
    return dict(model=model, records=list(spec["records"]), hot_slots=spec["hot"], compared_rows=compared,
                graph_replays=getattr(graph, "replays", None), graphs=len(getattr(graph, "graphs", ())),
                final_routes=sa.route.tolist(),
                problems=problems[:20], passed=not problems)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--allow-cuda-test", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=("history", "production"), default=("history", "production"))
    parser.add_argument("--output", type=Path, help="optional JSON result; never overwritten")
    args = parser.parse_args()
    if not args.allow_cuda_test:
        parser.error("Run only in an approved idle test process with --allow-cuda-test")
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    import torch
    if torch.cuda.get_device_capability() != (8, 0):
        parser.error("The route kernel gate runs on sm_80 (A100)")
    result = dict(schema="hisparse-dynamic-offload-gate-v1", route_header_sha256=dev.route_sha256(),
                  device=torch.cuda.get_device_name(), torch=torch.__version__, cases=list(args.cases),
                  history=[history(torch, graph=graph, host_shadow=shadow) for graph in (False, True)
                           for shadow in (False, True)] if "history" in args.cases else [],
                  production=[production(torch, model) for model in SPECS] if "production" in args.cases else [])
    result["passed"] = all(case["passed"] for case in (*result["history"], *result["production"]))
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output is not None:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
