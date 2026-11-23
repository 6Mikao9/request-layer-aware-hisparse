"""Model-free eager CUDA gate for request-type capacity dispatch on pinned A100.

Run in an authorized idle process:
python -m runtime.cuda_request_type_correctness --allow-cuda-test
"""

import argparse
from pathlib import Path


HMAX = 4096
CAPACITIES = (2048, 2560, 3072, HMAX)
BLOCK = 128
WIDTH = 16
HEAD_DIM = 128
REQUEST_SLOTS = 4
BATCH_ROWS = 4
MAX_SEQ = HMAX + 1024


def selection(seq_len, phase, include_latest):
    block_count = (seq_len + BLOCK - 1) // BLOCK
    if block_count < WIDTH:
        return list(range(block_count)) + [-1] * (WIDTH - block_count)
    last = block_count - 1
    count = WIDTH - int(include_latest)
    blocks = [(phase * 11 + index * 3) % last for index in range(count)]
    # Small fixtures and some lengths may repeat under modular stepping.
    blocks = list(dict.fromkeys(blocks))
    for block in range(last):
        if len(blocks) == count:
            break
        if block not in blocks:
            blocks.append(block)
    if include_latest:
        blocks.append(last)
    return blocks + [-1] * (WIDTH - len(blocks))


def make_layer(torch, layer, lengths):
    device = "cuda"
    physical = (2 * torch.arange(REQUEST_SLOTS * (HMAX + 1), device=device, dtype=torch.int32)
                + 3).reshape(REQUEST_SLOTS, HMAX + 1)
    count = REQUEST_SLOTS * MAX_SEQ
    rows = torch.arange(count)[:, None] + layer * count
    columns = torch.arange(HEAD_DIM)[None, :]
    host_k = ((rows * 17 + columns * 3) % 251 - 125).to(torch.bfloat16)
    host_v = ((rows * 23 + columns * 5) % 251 - 125).to(torch.bfloat16)
    for host in (host_k, host_v):
        host[:, 0] = (rows[:, 0] % 251).to(torch.bfloat16)
        host[:, 1] = (rows[:, 0] // 251).to(torch.bfloat16)
    host_k, host_v = host_k.pin_memory(), host_v.pin_memory()

    loc_storage = torch.full((REQUEST_SLOTS, HMAX + 131), -999, dtype=torch.int32, device=device)
    token_storage = torch.full_like(loc_storage, -999)
    lru_storage = torch.full((REQUEST_SLOTS, HMAX + 5), -999, dtype=torch.int16, device=device)
    loc_storage[:, :HMAX + 1].copy_(physical)
    token_storage[:, :HMAX + 128].fill_(-1)
    lru_storage[:, :HMAX].copy_(torch.arange(HMAX, dtype=torch.int16, device=device))
    device_k = torch.full((2 * REQUEST_SLOTS * (HMAX + 1) + 7, HEAD_DIM), -777,
                          dtype=torch.bfloat16, device=device)
    device_v = device_k.clone()
    host_locs = torch.arange(count, dtype=torch.int64, device=device).reshape(REQUEST_SLOTS, MAX_SEQ)
    state = dict(tokens=token_storage, locations=loc_storage, lru=lru_storage,
                 key=device_k, value=device_v)
    oracle = {name: tensor.clone() for name, tensor in state.items()}

    def append(lengths_before, lengths_after):
        for slot, (previous, length) in enumerate(zip(lengths_before, lengths_after)):
            offset = slot * MAX_SEQ
            for buffers in (state, oracle):
                start, stop = min(previous, HMAX), min(length, HMAX)
                if start < stop:
                    dest = physical[slot, start:stop].long()
                    buffers["key"][dest] = host_k[offset + start:offset + stop].cuda()
                    buffers["value"][dest] = host_v[offset + start:offset + stop].cuda()
                    buffers["tokens"][slot, start:stop] = torch.arange(start, stop, device=device)
                # Existing producer writes the newest position at the physical
                # reserve slot once the sequence is longer than Hmax.
                newest = min(length - 1, HMAX)
                dest = physical[slot, newest].long()
                buffers["key"][dest] = host_k[offset + length - 1].cuda()
                buffers["value"][dest] = host_v[offset + length - 1].cuda()

    append((0,) * REQUEST_SLOTS, lengths)
    return state, oracle, host_locs, host_k, host_v, physical, append


def exercise(torch, grouped, single, request_dtype, length_dtype):
    device = "cuda"
    lengths = (HMAX, HMAX + 257, HMAX + 129, HMAX + 385)
    layers = [make_layer(torch, layer, lengths) for layer in range(2)]
    output = torch.full((BATCH_ROWS, WIDTH * BLOCK + 3), -999, dtype=torch.int32, device=device)
    expected = output.clone()
    selection_storage = torch.full((BATCH_ROWS, WIDTH + 2), -999, dtype=torch.int32, device=device)
    selections = selection_storage[:, :WIDTH]
    rounds = (
        ((2, 0, 1), lengths),
        ((1, 2, 0), tuple(length + 1 for length in lengths)),
        ((2, 1), tuple(length + 2 for length in lengths)),
    )
    for phase, (slots, new_lengths) in enumerate(rounds):
        slot_rows = (*slots, *((999,) * (BATCH_ROWS - len(slots))))
        requests = torch.tensor(slot_rows, dtype=request_dtype, device=device)
        seqs = torch.tensor([new_lengths[slot] if slot != 999 else -999 for slot in slot_rows],
                            dtype=length_dtype, device=device)
        real = torch.tensor([len(slots)], dtype=torch.int32, device=device)
        for layer, (state, oracle, host_locs, host_k, host_v, physical, append) in enumerate(layers):
            if phase:
                append(lengths, new_lengths)
            assigned = (2048, 3072, HMAX, 2560) if layer == 0 else (3072, HMAX, 2048, 2560)
            assigned_tensor = torch.tensor(assigned, dtype=torch.int32, device=device)
            blocks = [selection(new_lengths[slot], phase + layer + slot,
                                include_latest=(slot + phase + layer) % 2 == 0)
                      if slot != 999 else [999] * WIDTH for slot in slot_rows]
            selections.copy_(torch.tensor(blocks, dtype=torch.int32, device=device))

            def arguments(buffers):
                return dict(device_buffer_tokens=buffers["tokens"][:, :HMAX + 128],
                            device_buffer_locs=buffers["locations"][:, :HMAX + 128],
                            lru_slots=buffers["lru"][:, :HMAX], host_cache_locs=host_locs,
                            host_cache_k=host_k, host_cache_v=host_v,
                            device_buffer_k=buffers["key"], device_buffer_v=buffers["value"],
                            item_size_bytes=HEAD_DIM * 2, hot_buffer_size=HMAX,
                            sparse_block_size=BLOCK, page_size=1, block_size=960)

            grouped(assigned_capacities=assigned_tensor, capacities=CAPACITIES,
                    top_k_blocks=selections, top_k_device_locs=output[:, :WIDTH * BLOCK],
                    req_pool_indices=requests, seq_lens=seqs, num_real_reqs=real, **arguments(state))
            expected[:, :WIDTH * BLOCK].fill_(-1)
            oracle_args = arguments(oracle)
            for row, slot in enumerate(slots):
                single(top_k_blocks=selections[row:row + 1],
                       top_k_device_locs=expected[row:row + 1, :WIDTH * BLOCK],
                       req_pool_indices=requests[row:row + 1], seq_lens=seqs[row:row + 1],
                       active_hot_buffer_size=assigned[slot], **oracle_args)
            torch.cuda.synchronize()
            context = f"{request_dtype}/{length_dtype} phase={phase} layer={layer}"
            assert torch.equal(output, expected), f"output or row padding mismatch: {context}"
            for name, tensor in state.items():
                assert torch.equal(tensor, oracle[name]), f"{name} mismatch: {context}"
            for row, slot in enumerate(slots):
                length = new_lengths[slot]
                logical = [block * BLOCK + index for block in blocks[row] for index in range(BLOCK)]
                columns = [index for index, position in enumerate(logical) if 0 <= position < length]
                pos = [logical[index] for index in columns]
                locations = output[row, columns].long()
                assert torch.all(locations >= 0), f"missing valid output: {context}"
                host_rows = [slot * MAX_SEQ + position for position in pos]
                assert torch.equal(state["key"][locations], host_k[host_rows].cuda()), context
                assert torch.equal(state["value"][locations], host_v[host_rows].cuda()), context
                if length - 1 in pos:
                    newest = min(length - 1, HMAX)
                    assert locations[pos.index(length - 1)] == physical[slot, newest], context
        lengths = new_lengths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cuda-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_cuda_test:
        parser.error("pass --allow-cuda-test in an authorized idle A100 process")
    if not __debug__:
        raise RuntimeError("correctness checks require Python assertions")

    import sglang
    import torch
    from runtime.hook import verify_sources

    verify_sources(Path(sglang.__file__).resolve().parent)
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("an A100 device is required")
    from runtime.generated.ops import load_blocks_to_device_buffer_mha as single
    from runtime.request_dispatch import (
        load_grouped_blocks_to_device_buffer_mha as grouped, verify_grouped_sources,
    )
    verify_grouped_sources()

    for request_dtype in (torch.int32, torch.int64):
        for length_dtype in (torch.int32, torch.int64):
            exercise(torch, grouped, single, request_dtype, length_dtype)
    print("Eager grouped dispatch matched serial fused and logical K/V oracles on A100.")


if __name__ == "__main__":
    main()
