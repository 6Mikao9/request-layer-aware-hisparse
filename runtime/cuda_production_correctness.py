"""Model-free CUDA gate using the inspected MiniMax TP4 MHA specialization.

Explicit, isolated execution only. This measures correctness, not performance.
The pinned model uses one BF16 KV head/rank, head_dim=128, sparse_block=128,
16 selected blocks (2048 positions), and swap_in_block_size=960.
"""

import argparse
import json
from pathlib import Path


def selected_blocks(seq, phase, *, width=16, block=128):
    """Unique logical blocks, with the latest block and optional -1 padding."""
    count = (seq + block - 1) // block
    if count <= width:
        return list(range(count)) + [-1] * (width - count)
    candidates = [(phase * 17 + i) % (count - 1) for i in range(width - 1)]
    return candidates + [count - 1]


def main():
    if not __debug__:
        raise RuntimeError('Correctness checks require Python assertions; disable -O/PYTHONOPTIMIZE')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-cuda-test', action='store_true')
    parser.add_argument('--hmax', type=int, default=8192, choices=(4096, 8192))
    parser.add_argument('--active', type=int, nargs='+', default=[2048, 3072, 4096, 5120])
    args = parser.parse_args()
    if not args.allow_cuda_test:
        parser.error('Use --allow-cuda-test only in an approved idle A100 process')
    if len(set(args.active)) != len(args.active) or any(not 2048 <= h <= args.hmax for h in args.active):
        parser.error('Active capacities must be distinct and between 2048 and Hmax')

    import torch
    import sglang
    from runtime.hook import verify_sources
    verify_sources(Path(sglang.__file__).resolve().parent)
    if 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('A100 required')
    from sglang.kernels.ops.kvcache.hisparse import load_blocks_to_device_buffer_mha as stock
    from runtime.generated.ops import load_blocks_to_device_buffer_mha as patched

    hmax, block, width, dim, batch = args.hmax, 128, 16, 128, 3
    maxseq = 2 * hmax + block + 1
    real = 2
    # Exactly representable BF16 values. First two columns jointly identify the
    # entire host row, unlike a large float arange rounded to BF16.
    row_ids = torch.arange(batch * maxseq, dtype=torch.int64)[:, None]
    columns = torch.arange(dim, dtype=torch.int64)[None, :]
    host_k = ((row_ids * 17 + columns * 3) % 251 - 125).to(torch.bfloat16)
    host_k[:, 0] = (row_ids[:, 0] % 251).to(torch.bfloat16)
    host_k[:, 1] = (row_ids[:, 0] // 251).to(torch.bfloat16)
    host_k = host_k.pin_memory()
    host_v = (-host_k - 7).pin_memory()

    def case(start_seq, active, fn, graph):
        slots = batch * (hmax + 1)
        kv = torch.full((2 * slots + 7, dim), -777, device='cuda', dtype=torch.bfloat16)
        vv = kv.clone()
        loc_storage = torch.full((batch, hmax + 131), -999, device='cuda', dtype=torch.int32)
        token_storage = torch.full_like(loc_storage, -999)
        locs, tokens = loc_storage[:, :hmax + 128], token_storage[:, :hmax + 128]
        physical = (2 * torch.arange(slots - 1, -1, -1, device='cuda', dtype=torch.int32) + 3).reshape(batch, hmax + 1)
        host_locs = torch.arange(batch * maxseq, device='cuda', dtype=torch.int64).reshape(batch, maxseq)
        lru_storage = torch.full((batch, hmax + 5), -999, device='cuda', dtype=torch.int16)
        lru = lru_storage[:, :hmax]
        selection_storage = torch.full((batch, width + 2), -999, device='cuda', dtype=torch.int32)
        selections = selection_storage[:, :width]
        output_storage = torch.full((batch, width * block + 3), -999, device='cuda', dtype=torch.int32)
        output = output_storage[:, :width * block]
        req = torch.tensor([1, 0, 999], device='cuda', dtype=torch.int64)
        seqs = torch.zeros(batch, device='cuda', dtype=torch.int64)
        nreal = torch.tensor([real], device='cuda', dtype=torch.int32)
        seq = start_seq

        def reset(new_seq):
            """Reinitialize a reused row without replacing any graph tensor."""
            nonlocal seq
            seq = new_seq
            kv.fill_(-777)
            vv.fill_(-777)
            loc_storage.fill_(-999)
            token_storage.fill_(-999)
            lru_storage.fill_(-999)
            locs[:, :hmax + 1].copy_(physical)
            tokens[:, :hmax + 1].copy_(torch.arange(hmax + 1, device='cuda', dtype=torch.int32).repeat(batch, 1))
            lru.copy_(torch.arange(hmax, device='cuda', dtype=torch.int16).repeat(batch, 1))
            for rid in range(batch):
                count = min(seq, hmax)
                dest = physical[rid, :count].long()
                kv[dest] = host_k[rid * maxseq:rid * maxseq + count].cuda()
                vv[dest] = host_v[rid * maxseq:rid * maxseq + count].cuda()
                latest = min(seq - 1, hmax)
                dest_latest = physical[rid, latest].long()
                kv[dest_latest] = host_k[rid * maxseq + seq - 1].cuda()
                vv[dest_latest] = host_v[rid * maxseq + seq - 1].cuda()
                locs[rid, hmax] = physical[rid, latest]
                tokens[rid, hmax] = seq - 1
            seqs[:real].fill_(seq)

        kwargs = dict(top_k_blocks=selections, device_buffer_tokens=tokens,
                      host_cache_locs=host_locs, device_buffer_locs=locs,
                      host_cache_k=host_k, host_cache_v=host_v,
                      device_buffer_k=kv, device_buffer_v=vv, top_k_device_locs=output,
                      req_pool_indices=req, seq_lens=seqs, lru_slots=lru,
                      item_size_bytes=dim * 2, hot_buffer_size=hmax,
                      sparse_block_size=block, page_size=1, block_size=960,
                      num_real_reqs=nreal)
        if fn is patched:
            kwargs['active_hot_buffer_size'] = active

        def run():
            fn(**kwargs)

        def check(blocks, before):
            torch.cuda.synchronize()
            assert torch.equal(output[2], torch.full_like(output[2], -1)), 'inactive output'
            assert torch.equal(tokens[2], before[0][2]), 'inactive token row changed'
            assert torch.equal(locs, before[1]), 'physical map changed'
            assert torch.equal(lru[2], before[2][2]), 'inactive request LRU changed'
            assert torch.equal(lru[:, active:], before[2][:, active:]), 'inactive LRU tail changed'
            assert torch.equal(lru[:2, :active].sort().values, before[2][:2, :active].sort().values), 'LRU permutation'
            unchanged = torch.cat((physical[:2, active:hmax].reshape(-1), physical[2])).long()
            assert torch.equal(kv[unchanged], before[3][unchanged]), 'inactive K changed'
            assert torch.equal(vv[unchanged], before[4][unchanged]), 'inactive V changed'
            if seq <= hmax:
                assert torch.equal(tokens, before[0]) and torch.equal(lru, before[2]), 'short state changed'
                assert torch.equal(kv, before[3]) and torch.equal(vv, before[4]), 'short path wrote KV'
            else:
                assert torch.equal(tokens[:, active:], before[0][:, active:]), 'inactive token metadata changed'
            logical = [b * block + i if b >= 0 and b * block + i < seq else -1 for b in blocks for i in range(block)]
            mask = torch.tensor([t >= 0 for t in logical], device='cuda')
            for out_row, rid in enumerate([1, 0]):
                assert torch.equal(output[out_row] >= 0, mask), 'selected index validity'
                valid = [t for t in logical if t >= 0]
                logical_tensor = torch.tensor(valid, device='cuda', dtype=torch.int64)
                got_locs = output[out_row][mask].long()
                if seq <= hmax:
                    assert torch.equal(got_locs, locs[rid, logical_tensor].long()), 'short physical routing'
                else:
                    older = logical_tensor != seq - 1
                    # Invert this fixture's unique descending/strided physical
                    # map. Correct payload alone cannot exclude an inactive-tail hit.
                    offsets = physical[rid, 0].long() - got_locs[older]
                    assert torch.all(offsets % 2 == 0), 'physical index outside the fixture map'
                    active_slots = offsets // 2
                    assert torch.all((active_slots >= 0) & (active_slots < active)), 'returned inactive hot slot'
                    assert torch.equal(tokens[rid, active_slots].long(), logical_tensor[older]), 'active slot identity'
                ids = [rid * maxseq + t for t in valid]
                expected_k, expected_v = host_k[ids].cuda(), host_v[ids].cuda()
                got_k, got_v = kv[output[out_row][mask].long()], vv[output[out_row][mask].long()]
                assert torch.equal(got_k, expected_k) and torch.equal(got_v, expected_v), 'selected K/V oracle'
                for index, t in enumerate(logical):
                    if t == seq - 1:
                        assert output[out_row, index] == locs[rid, hmax], 'latest alias'
                q = torch.linspace(-0.001, 0.001, dim, device='cuda')
                result = torch.softmax(got_k.float() @ q, 0) @ got_v.float()
                reference = torch.softmax(expected_k.float() @ q, 0) @ expected_v.float()
                torch.testing.assert_close(result, reference, rtol=0, atol=0)
            assert torch.all(loc_storage[:, hmax + 128:] == -999), 'location padding'
            assert torch.all(token_storage[:, hmax + 128:] == -999), 'token padding'
            assert torch.all(lru_storage[:, hmax:] == -999), 'LRU padding'
            assert torch.all(output_storage[:, width * block:] == -999), 'output padding'

        reset(seq)
        selections.copy_(torch.tensor([selected_blocks(seq, 0)] * batch, device='cuda', dtype=torch.int32))
        run()  # Compile before capture.
        g = None
        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run()
        states = []
        # Fresh state and request-row reuse with existing captured graph buffers.
        # This synthesizes reset/preload; it does not test the coordinator's free path.
        for epoch_seq in (start_seq, 2 * hmax + block):
            reset(epoch_seq)
            for phase in (0, 2, 4, 0):
                blocks = selected_blocks(seq, phase)
                selections.copy_(torch.tensor([blocks] * batch, device='cuda', dtype=torch.int32))
                before = (tokens.clone(), locs.clone(), lru.clone(), kv.clone(), vv.clone())
                run() if g is None else g.replay()
                check(blocks, before)
            states.append(tuple(x.cpu() for x in (output, tokens, lru, kv, vv)))
        return states

    count = 0
    for seq in (hmax - 1, hmax, hmax + 1, 2 * hmax + block):
        for graph in (False, True):
            baseline = case(seq, hmax, stock, graph)
            identity = case(seq, hmax, patched, graph)
            for baseline_epoch, identity_epoch in zip(baseline, identity):
                for before, after in zip(baseline_epoch, identity_epoch):
                    assert torch.equal(before, after), 'stock/identity state mismatch'
            for active in args.active:
                case(seq, active, patched, graph)
            count += 1
            print(json.dumps(dict(case=count, seq=seq, graph=graph, hmax=hmax,
                                  active=args.active, status='passed')), flush=True)
    print(json.dumps(dict(status='passed', cases=count, specialization=dict(
        top_k=2048, sparse_block_size=128, swap_in_block_size=960, kv_dtype='bfloat16',
        head_dim=128, item_size_bytes=256), scope='model-free correctness only')), flush=True)


if __name__ == '__main__':
    main()
