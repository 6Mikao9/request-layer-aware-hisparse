"""Compare pinned MiniMax attention readers at the two measured TP8 batch shapes."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import median


POINTS = {8: 45056, 32: 24576}
ARMS = ("stock_hot", "resident_direct")
PAGE, HOT, BLOCKS, DIM, HEADS = 128, 4096, 16, 128, 8


def fixture(batch):
    import torch
    from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import flash_decode_with_gqa_share_sparse

    sequence = POINTS[batch]
    rng = torch.Generator(device="cuda").manual_seed(20261125 + batch)
    selected = random.Random(20261125 + batch)

    def values(rows):
        return torch.randn((rows, 1, DIM), dtype=torch.bfloat16, device="cuda", generator=rng)

    resident_rows = batch * sequence + PAGE
    keys, vals = values(resident_rows), values(resident_rows)
    page_ids = torch.randperm(batch * sequence // PAGE, device="cuda", generator=rng).reshape(batch, -1) + 1
    request_table = ((page_ids[..., None] * PAGE + torch.arange(PAGE, device="cuda"))
                     .reshape(batch, sequence).to(torch.int32))
    indices = torch.tensor([selected.sample(range(sequence // PAGE), BLOCKS) for _ in range(batch)],
                           device="cuda", dtype=torch.int32)
    positions = (indices[..., None] * PAGE + torch.arange(PAGE, device="cuda")).reshape(batch, -1).long()
    actual = request_table.gather(1, positions).long()
    count = BLOCKS * PAGE
    hot_rows = batch * (HOT + PAGE) + PAGE
    slots = (torch.randperm(hot_rows - PAGE, device="cuda", generator=rng)[:batch * count] + PAGE)
    hot_keys = torch.empty((hot_rows, 1, DIM), device="cuda", dtype=torch.bfloat16)
    hot_vals = torch.empty_like(hot_keys)
    hot_keys.index_copy_(0, slots, keys.index_select(0, actual.flatten()))
    hot_vals.index_copy_(0, slots, vals.index_select(0, actual.flatten()))
    slots = slots.reshape(1, batch, count).to(torch.int32)
    query = torch.randn((batch, HEADS, DIM), dtype=torch.bfloat16, device="cuda", generator=rng)
    opts = dict(q=query, sink=None, req_to_token=request_table,
                seq_lens=torch.full((batch,), sequence, dtype=torch.int32, device="cuda"),
                slot_ids=torch.arange(batch, dtype=torch.int64, device="cuda"),
                block_size=PAGE, topk_idx=indices.unsqueeze(0), page_size=PAGE)

    def invoke(arm):
        if arm == "stock_hot":
            return flash_decode_with_gqa_share_sparse(k_cache=hot_keys, v_cache=hot_vals,
                                                      hisparse_slots=slots, **opts)
        return flash_decode_with_gqa_share_sparse(k_cache=keys, v_cache=vals, hisparse_slots=None, **opts)

    reference_keys = keys.index_select(0, actual.flatten()).reshape(batch, count, DIM).float()
    reference_vals = vals.index_select(0, actual.flatten()).reshape(batch, count, DIM).float()
    reference = ((query.float() @ reference_keys.transpose(1, 2) * (DIM ** -0.5)).softmax(-1) @ reference_vals)
    return invoke, reference


def measure(batch, probe):
    import sglang
    import torch
    from runtime.layer_placement_device import verify_digests

    source = verify_digests(Path(sglang.__file__).resolve().parent)
    if torch.cuda.get_device_capability() != (8, 0) or set(source.values()) != {"ok"}:
        raise RuntimeError("Use the pinned SM80 runtime")
    invoke, reference = fixture(batch)
    graphs, outputs, errors = {}, {}, {}
    for arm in ARMS:
        invoke(arm)  # JIT outside graph capture
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs[arm] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[arm], stream=stream):
            outputs[arm] = invoke(arm)
        graphs[arm].replay()
        torch.cuda.synchronize()
        expected = reference.to(outputs[arm].dtype)
        torch.testing.assert_close(outputs[arm], expected, atol=0.002, rtol=0.02)
        errors[arm] = float((outputs[arm].float() - reference).abs().max().item())
    difference = float((outputs["stock_hot"].float() - outputs["resident_direct"].float()).abs().max().item())
    timings = defaultdict(list)
    if probe:
        torch.cuda.nvtx.range_push("issue3-attention")
        try:
            graphs[probe].replay()
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        for repeat in range(24):
            for arm in ARMS if repeat % 2 == 0 else ARMS[::-1]:
                start, finish = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                graphs[arm].replay()
                finish.record()
                finish.synchronize()
                timings[arm].append(start.elapsed_time(finish) * 1000)
    return dict(
        batch=batch, context=POINTS[batch], head_dim=DIM, local_query_heads=HEADS, local_kv_heads=1,
        page_rows=PAGE, hot_rows_per_request=HOT, selected_per_request=BLOCKS * PAGE, probe=probe,
        max_abs_error_fp32=errors, max_abs_difference_readers=difference, kv_dtype="bfloat16",
        timing_us={arm: dict(median=median(values), samples=values) for arm, values in timings.items()},
        scope="One A100, synthetic equal payload, sparse attention Graph with prepopulated K/V, no swap.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cuda-test", action="store_true")
    parser.add_argument("--batch", type=int, required=True, choices=tuple(POINTS))
    parser.add_argument("--probe", choices=ARMS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_cuda_test or args.output.exists():
        parser.error("Use explicit GPU authorization and a fresh output path")
    result = measure(args.batch, args.probe)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
