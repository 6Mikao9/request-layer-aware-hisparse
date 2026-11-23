"""Full-state split/fused equivalence on the pinned MHA production layout.

No model weights, server, performance claim or modifications to SGLang. Covers
short/latest/boundary routing, active capacities, padded rows, reused requests,
graph replay with changing real counts, and the actual coordinator adapter.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    if not __debug__:
        raise RuntimeError('Assertions are required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-cuda-test', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_cuda_test or args.output.exists():
        parser.error('Use explicit --allow-cuda-test and a fresh output path')
    import torch
    import sglang
    from runtime.hook import verify_sources
    from runtime.policy import StaticPolicy
    from runtime.split_copy import SplitCopyConfig, SplitCopyState, swap_in_selected_blocks as split
    from runtime.generated.coordinator_method import swap_in_selected_blocks as fused
    from runtime.cuda_production_correctness import selected_blocks
    verify_sources(Path(sglang.__file__).resolve().parent)
    if 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('A100 required')
    torch.set_num_threads(1)
    hmax, dim, meta_rows, maxseq = 8192, 128, 3, 16641
    ids = torch.arange(meta_rows * maxseq, dtype=torch.int64)[:, None]
    cols = torch.arange(dim, dtype=torch.int64)[None, :]
    hk = ((ids * 17 + cols * 3) % 251 - 125).to(torch.bfloat16)
    hk[:, 0], hk[:, 1] = (ids[:, 0] % 251).to(torch.bfloat16), (ids[:, 0] // 251).to(torch.bfloat16)
    hk, hv = hk.pin_memory(), (-hk - 7).pin_memory()
    hostlocs = torch.arange(meta_rows * maxseq, device='cuda', dtype=torch.int64).reshape(meta_rows, maxseq)
    slots = meta_rows * (hmax + 1)
    physical = (2 * torch.arange(slots - 1, -1, -1, device='cuda', dtype=torch.int32) + 3).reshape(meta_rows, hmax + 1)
    config = SplitCopyConfig((5,))
    cases = []
    report = dict(schema='hisparse-mha-split-correctness-v1', complete=False, cases=cases,
        scope='Model-free same-state split/fused exact equality; not inference accuracy or speed',
        specialization=dict(hmax=hmax, top_k=2048, kv_dim=128, item_bytes=256, planner_threads=960))
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')

    def fixture(active, batch):
        # Both objects have identical storage layout and input, independent mutable state.
        k = torch.empty((2 * slots + 7, dim), device='cuda', dtype=torch.bfloat16)
        v = torch.empty_like(k)
        token_store = torch.empty((meta_rows, hmax + 131), device='cuda', dtype=torch.int32)
        loc_store = torch.empty_like(token_store)
        lru_store = torch.empty((meta_rows, hmax + 5), device='cuda', dtype=torch.int16)
        out_store = torch.empty((batch, 2051), device='cuda', dtype=torch.int32)
        selection_store = torch.empty((batch, 18), device='cuda', dtype=torch.int32)
        req = torch.empty(batch, device='cuda', dtype=torch.int64)
        lens = torch.empty_like(req)
        nreal = torch.empty(1, device='cuda', dtype=torch.int32)
        policy = StaticPolicy(hmax, (active,) * 60)
        coord = SimpleNamespace(is_m3_hisparse=True, device_buffer_size=hmax,
            _request_layer_hisparse_policy=policy, _request_layer_hisparse_split=SplitCopyState(config, 'cuda'),
            req_device_buffer_tokens=token_store[:, :hmax + 128].unsqueeze(0),
            req_device_buffer_token_locs=loc_store[:, :hmax + 128].unsqueeze(0),
            lru_slots=lru_store[:, :hmax].unsqueeze(0), top_k_device_locs_buffer=out_store[:, :2048],
            mem_pool_host=SimpleNamespace(k_buffer=[hk], v_buffer=[hv]),
            mem_pool_device=SimpleNamespace(start_layer=5, get_key_buffer=lambda _: k, get_value_buffer=lambda _: v),
            req_to_host_pool=hostlocs, item_size_bytes=256, swap_in_block_size=960, skip_io=False, num_real_reqs=nreal)
        arrays = [k, v, token_store, loc_store, lru_store, out_store, selection_store, req, lens, nreal]
        current_seqs = [0] * batch

        def reset(seq, real, mixed=False):
            nonlocal current_seqs
            for x in arrays:
                x.fill_(-999 if x.dtype != torch.bfloat16 else -777)
            loc_store[:, :hmax + 1].copy_(physical)
            token_store[:, :hmax].copy_(torch.arange(hmax, device='cuda', dtype=torch.int32).expand(meta_rows, -1))
            lru_store[:, :hmax].copy_(torch.arange(hmax, device='cuda', dtype=torch.int16).expand(meta_rows, -1))
            row_order = [1, 0, 2][:real]
            logical_seqs = [seq] * meta_rows
            if mixed:
                logical_seqs[row_order[0]] = 127
            for rid in range(meta_rows):
                row_seq = logical_seqs[rid]
                token_store[rid, hmax] = row_seq - 1
                count = min(row_seq, hmax)
                dest = physical[rid, :count].long()
                k[dest] = hk[rid * maxseq:rid * maxseq + count].cuda()
                v[dest] = hv[rid * maxseq:rid * maxseq + count].cuda()
                latest = min(row_seq - 1, hmax)
                dest_last = physical[rid, latest].long()
                k[dest_last] = hk[rid * maxseq + row_seq - 1].cuda()
                v[dest_last] = hv[rid * maxseq + row_seq - 1].cuda()
                loc_store[rid, hmax] = dest_last
            req.fill_(999999)
            req[:real].copy_(torch.tensor(row_order, device='cuda', dtype=torch.int64))
            current_seqs = [logical_seqs[rid] for rid in row_order] + [0] * (batch - real)
            lens.copy_(torch.tensor(current_seqs, device='cuda', dtype=torch.int64))
            nreal.fill_(real)

        def select(seq, phase):
            selection_store[:, :16].copy_(torch.tensor([selected_blocks(s, phase) for s in current_seqs], device='cuda', dtype=torch.int32))

        def call(use_split, layer):
            # For the excluded-layer route, expose the same fixture as layer 6.
            coord.mem_pool_device.start_layer = layer
            return (split if use_split else fused)(coord, req, lens, selection_store[:, :16], layer, 128)
        return arrays, reset, select, call, coord._request_layer_hisparse_split

    for active in (2048, 3072, 3584, 4096, 5120, 8192):
        for batch, layer in ((1, 5), (2, 5), (3, 5), (2, 6)):
            for graph_mode in (False, True):
                reference, reset_a, select_a, call_a, _ = fixture(active, batch)
                candidate, reset_b, select_b, call_b, plan = fixture(active, batch)
                reset_a(16512, min(batch, 2)); reset_b(16512, min(batch, 2))
                select_a(16512, 0); select_b(16512, 0)
                call_a(False, layer); call_b(True, layer)  # all JIT before capture
                graphs = []
                if graph_mode:
                    for call, use_split in ((call_a, False), (call_b, True)):
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            call(use_split, layer)
                        graphs.append(g)
                checks = 0
                positive_plan_calls = 0
                for seq in (127, hmax - 1, hmax, hmax + 1, 16512):
                    for real in range(1, batch + 1):
                        for mixed in ((False, True) if real > 1 and seq > hmax else (False,)):
                            reset_a(seq, real, mixed); reset_b(seq, real, mixed)
                            for phase in (0, 2, 4, 0):
                                select_a(seq, phase); select_b(seq, phase)
                                for tensor in (plan.src, plan.dst, plan.counts):
                                    tensor.fill_(-991)
                                if graph_mode:
                                    graphs[0].replay(); graphs[1].replay()
                                else:
                                    call_a(False, layer); call_b(True, layer)
                                torch.cuda.synchronize()
                                for index, (a, b) in enumerate(zip(reference, candidate)):
                                    assert torch.equal(a, b), (active, batch, layer, graph_mode, seq, real, phase, index)
                                assert torch.all(candidate[5][real:, :2048] == -1), 'padded output'
                                if config.uses_split(layer, batch):
                                    counts = plan.counts[:real].tolist()
                                    assert all(0 <= n <= 2048 for n in counts), 'planner did not run'
                                    positive_plan_calls += int(any(counts))
                                    for row, count in enumerate(counts):
                                        if candidate[8][row] <= hmax:
                                            assert count == 0, 'short path must emit zero misses'
                                        assert torch.all(plan.src[row, :count] >= 0) and torch.all(plan.dst[row, :count] >= 0), 'unwritten plan'
                                        assert torch.all(plan.src[row, count:] == -991) and torch.all(plan.dst[row, count:] == -991), 'plan tail changed'
                                    assert torch.all(plan.counts[real:] == -991), 'padding plan count changed'
                                    assert torch.all(plan.src[real:] == -991) and torch.all(plan.dst[real:] == -991), 'padding plan changed'
                                else:
                                    assert all(torch.all(t == -991) for t in (plan.src, plan.dst, plan.counts)), 'fallback touched plan'
                                checks += 1
                if config.uses_split(layer, batch):
                    assert positive_plan_calls > 0, 'no cold misses exercised'
                cases.append(dict(active=active, captured_batch=batch, layer=layer,
                    execution='graph' if graph_mode else 'eager', route='split' if config.uses_split(layer, batch) else 'fused_fallback',
                    checked_calls=checks, positive_plan_calls=positive_plan_calls, status='full_state_exact'))
                args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
                print(json.dumps(cases[-1]), flush=True)
                graphs.clear()
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
