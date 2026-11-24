"""Production-geometry shared LRU correctness gate for one A100, with no model.

    python -m runtime.cuda_shared_lru_production_correctness --allow-cuda-test --output gate.json

The default exercises the two boundary planes. --all-planes adds a checked
pass over every plane in both eager and graph modes. Always allocates all
60 planes. The synthetic input is only for byte and table comparisons.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from runtime import study_config as sc
from runtime.cuda_shared_lru_correctness import ARMS, Bench, Geometry

GEOMETRY = Geometry(layers=60, hot=4096, page=128, slots=2, blocks=16, block=128, dim=128, context=6400)


def selections(g):
    first_cold = g.hot // g.block
    hot = list(range(g.blocks))
    cold = list(range(first_cold, first_cold + g.blocks))
    mixed = [hot[0], hot[0], -1, cold[-1] + 1] + cold[:g.blocks - 4]
    return hot, cold, mixed


def run(torch, *, graph, all_planes, seed):
    g = GEOMETRY
    bench = Bench(torch, sc.parse({"schema": sc.SCHEMA, "hot_pool": ARMS["shared"],
                                   "diagnostics": {"mode": "counters"}}), seed, geometry=g)
    slot, bucket = 0, g.slots
    hot, cold, mixed = selections(g)
    prompt = (cold[-1] + 1) * g.block + 3
    bench.admit(slot, prompt)
    last = g.layers - 1
    steps = [(last, cold), (0, hot), (last, cold), (last, mixed)]
    if all_planes:
        steps.extend((plane, mixed) for plane in range(g.layers))
    static = bench.tensors([(slot, hot)], torch.int32, torch.int64, bucket)
    graphs = {}
    if graph:
        bench.coordinator.num_real_reqs.zero_()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            bench.runtime.resolve(bench.coordinator, *static, last, g.block)
        torch.cuda.current_stream().wait_stream(side)
        for plane in sorted({p for p, _ in steps}):
            graphs[plane] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[plane]):
                bench.runtime.resolve(bench.coordinator, *static, plane, g.block)
        torch.cuda.synchronize()

    def launch(plane, blocks):
        rows = [(slot, blocks)]
        expected = [bench.expect(slot, plane, blocks)]
        tensors = bench.tensors(rows, torch.int32, torch.int64, bucket)
        bench.coordinator.num_real_reqs.fill_(len(rows))
        if graph:
            for target, source in zip(static, tensors):
                target.copy_(source)
            graphs[plane].replay()
        else:
            bench.runtime.resolve(bench.coordinator, *tensors, plane, g.block)
        bench.check(plane, rows, expected, bucket)

    for plane, blocks in steps:
        launch(plane, blocks)
    # Distinguish the new request at the same slot and host addresses while
    # keeping the pinned host pointers captured in the original graph valid.
    bench.runtime.reset_slot(slot)
    bench.mirror.reset_slot(slot)
    assert bench.runtime.row_owner[slot].tolist() == bench.mirror.slots[slot]["owner"]
    first_gen = bench.mirror.slots[slot]["gen"]
    for history in (*bench.host_k, *bench.host_v):
        history[:g.context].neg_()
    bench.admit(slot, prompt)
    assert int(bench.runtime.generation[slot]) == bench.mirror.slots[slot]["gen"] != first_gen
    for blocks in (cold, mixed):
        launch(last, blocks)
    bench.check_counters()
    padded_slot = bucket - 1
    assert int(bench.runtime.generation[padded_slot]) == 0
    assert bool(torch.all(bench.runtime.row_owner[padded_slot] == -1))
    assert torch.equal(bench.runtime.order[padded_slot], bench.runtime._order_template)
    assert not bool(torch.any(bench.runtime.occupancy[padded_slot]))
    assert bool(torch.all(bench.coordinator.req_device_buffer_tokens[:, padded_slot] == -1))
    return dict(mode="graph" if graph else "eager", planes=sorted({p for p, _ in steps}),
                launches=len(steps) + 2, generations=2, status="pass")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cuda-test", action="store_true")
    parser.add_argument("--all-planes", action="store_true")
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.allow_cuda_test:
        parser.error("The CUDA gate requires --allow-cuda-test")
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    if not __debug__:
        parser.error("Run without -O so all correctness comparisons are enabled")
    import torch
    import sglang
    from runtime.shared_lru_device import verify_digests
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("A100 (sm_80) required")
    bad = {k: v for k, v in verify_digests(Path(sglang.__file__).resolve().parent).items() if v != "ok"}
    if bad:
        raise RuntimeError(f"Pinned SGLang source mismatch: {bad}")
    results = [run(torch, graph=graph, all_planes=args.all_planes, seed=args.seed) for graph in (False, True)]
    summary = dict(event="hisparse_shared_lru_production_cuda_gate", geometry=asdict(GEOMETRY),
                   seed=args.seed, status="pass", torch=str(torch.__version__),
                   device=torch.cuda.get_device_name(), results=results)
    if args.output is not None:
        with args.output.open("x") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
