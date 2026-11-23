"""CUDA correctness gate. Explicit invocation only, never run on an active service.

Requires pinned SGLang image, A100, and runtime/ on PYTHONPATH. No model load.
Checks stock Hmax vs overlay Hmax and overlay reduced-H against dense host KV.
Not a performance benchmark. Do not interpret correctness as speedup.
"""
import argparse
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--allow-cuda-test',action='store_true')
    args=parser.parse_args()
    if not args.allow_cuda_test:
        parser.error('Run only in an approved idle test process with --allow-cuda-test')
    import torch
    import sglang
    from runtime.hook import verify_sources
    verify_sources(Path(sglang.__file__).resolve().parent)
    if 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('A100 required')
    from sglang.kernels.ops.kvcache.hisparse import load_blocks_to_device_buffer_mha as stock
    from runtime.generated.ops import load_blocks_to_device_buffer_mha as patched
    hmax, width, block, dim, maxseq = 64, 4, 4, 64, 128
    batch, real = 3, 2

    def case(seq, active, fn, seq_dtype, req_dtype, graph):
        # Distinct pinned host rows provide an independent token->K/V oracle.
        host_k=torch.arange(batch*maxseq*dim,dtype=torch.float32).reshape(batch*maxseq,dim).pin_memory()
        host_v=(-host_k-7).pin_memory()
        kv=torch.full((2*batch*(hmax+1)+7,dim),-777.,device='cuda')
        vv=kv.clone()
        loc_storage=torch.full((batch,hmax+128+3),-999,dtype=torch.int32,device='cuda')
        locs=loc_storage[:,:hmax+128]
        locs[:,:hmax+1].copy_((2*torch.arange(batch*(hmax+1)-1,-1,-1,dtype=torch.int32,device='cuda')+3).reshape(batch,hmax+1))
        physical_locs=locs.clone()
        token_storage=torch.full((batch,hmax+128+3),-999,dtype=torch.int32,device='cuda')
        tokens=token_storage[:,:hmax+128]
        tokens[:,:hmax+1].copy_(torch.arange(hmax+1,dtype=torch.int32,device='cuda').repeat(batch,1))
        host_locs=torch.arange(batch*maxseq,dtype=torch.int64,device='cuda').reshape(batch,maxseq)
        lru_storage=torch.full((batch,hmax+5),-999,dtype=torch.int16,device='cuda')
        lru=lru_storage[:,:hmax]
        lru.copy_(torch.arange(hmax,dtype=torch.int16,device='cuda').repeat(batch,1))
        for rid in range(batch):
            for t in range(min(seq,hmax)):
                kv[physical_locs[rid,t].long()].copy_(host_k[rid*maxseq+t])
                vv[physical_locs[rid,t].long()].copy_(host_v[rid*maxseq+t])
            latest=min(seq-1,hmax)
            kv[physical_locs[rid,latest].long()].copy_(host_k[rid*maxseq+seq-1])
            vv[physical_locs[rid,latest].long()].copy_(host_v[rid*maxseq+seq-1])
            # Physical short latest can be < Hmax; metadata alias stays at Hmax.
            locs[rid,hmax]=physical_locs[rid,latest]
            tokens[rid,hmax]=seq-1
        selected=list(dict.fromkeys([0,8,16,(seq-1)//block]))
        selected += [-1]*(width-len(selected))
        selection_storage=torch.full((batch,width+2),-999,dtype=torch.int32,device='cuda')
        selections=selection_storage[:,:width]
        selections.copy_(torch.tensor([selected]*batch,dtype=torch.int32,device='cuda'))
        output_storage=torch.full((batch,width*block+3),-999,dtype=torch.int32,device='cuda')
        output=output_storage[:,:width*block]
        req=torch.tensor([1,0,999],dtype=req_dtype,device='cuda')
        seqs=torch.tensor([seq,seq,0],dtype=seq_dtype,device='cuda')
        nreal=torch.tensor([real],dtype=torch.int32,device='cuda')
        kwargs=dict(top_k_blocks=selections,device_buffer_tokens=tokens,host_cache_locs=host_locs,
            device_buffer_locs=locs,host_cache_k=host_k,host_cache_v=host_v,device_buffer_k=kv,
            device_buffer_v=vv,top_k_device_locs=output,req_pool_indices=req,seq_lens=seqs,lru_slots=lru,
            item_size_bytes=dim*4,hot_buffer_size=hmax,sparse_block_size=block,num_real_reqs=nreal)
        if fn is patched:
            kwargs['active_hot_buffer_size']=active
        before=(tokens.clone(),locs.clone(),lru.clone(),kv.clone(),vv.clone())
        def run():
            fn(**kwargs)
        def check():
            torch.cuda.synchronize()
            assert torch.equal(output[2],torch.full_like(output[2],-1)), 'inactive graph row must be invalid'
            assert torch.equal(tokens[2],before[0][2]), 'inactive request state changed'
            assert torch.equal(locs,before[1]), 'physical map rebound/mutated'
            assert torch.equal(lru[:,active:],before[2][:,active:]), 'inactive LRU tail changed'
            untouched=physical_locs[:2,active:hmax].reshape(-1).long()
            inactive=physical_locs[2,:hmax+1].long()
            assert torch.equal(kv[untouched],before[3][untouched]) and torch.equal(vv[untouched],before[4][untouched]), 'inactive physical tail changed'
            assert torch.equal(kv[inactive],before[3][inactive]) and torch.equal(vv[inactive],before[4][inactive]), 'inactive request KV changed'
            assert torch.equal(lru[:2,:active].sort().values, before[2][:2,:active].sort().values), 'active LRU not a permutation'
            if seq <= hmax:
                assert torch.equal(tokens,before[0]) and torch.equal(lru,before[2]), 'short path mutated state'
            for row,rid in enumerate([1,0]):
                expected_tokens=[b*block+i if b>=0 and b*block+i<seq else -1 for b in selected for i in range(block)]
                mask=torch.tensor([t>=0 for t in expected_tokens],device='cuda')
                assert torch.equal(output[row]>=0,mask), 'flattened attention index validity differs'
                valid_tokens=[t for t in expected_tokens if t>=0]
                expected_k=host_k[[rid*maxseq+t for t in valid_tokens]].cuda()
                expected_v=host_v[[rid*maxseq+t for t in valid_tokens]].cuda()
                got_k=kv[output[row][mask].long()]
                got_v=vv[output[row][mask].long()]
                assert torch.equal(got_k,expected_k) and torch.equal(got_v,expected_v), 'complete selected K/V mismatch'
                for col,t in enumerate(expected_tokens):
                    if t==seq-1:
                        assert output[row,col]==locs[rid,hmax], 'latest alias mismatch'
                # Attention consumption sanity: no missing/reordered selected inputs.
                q=torch.linspace(-0.001,0.001,dim,device='cuda')
                result=torch.softmax(got_k@q,dim=0)@got_v
                reference=torch.softmax(expected_k@q,dim=0)@expected_v
                torch.testing.assert_close(result,reference,rtol=0,atol=0)
            assert torch.all(loc_storage[:,hmax+128:]==-999) and torch.all(token_storage[:,hmax+128:]==-999), "metadata row padding changed"
            assert torch.all(lru_storage[:,hmax:]==-999) and torch.all(output_storage[:,width*block:]==-999), "LRU/output row padding changed"
        run()  # JIT/prewarm outside capture
        check()
        g=None
        if graph:
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run()
            check()
        # Mutate graph inputs in-place: all misses, hits, then replacement/revisit.
        # A short -> Hmax+1 transition synthesizes stock preload/latest metadata.
        for next_seq, blocks in [(seq,[0,8,16,23]), (seq,[4,12,20,22]), (seq,[2,10,18,21]),
                                  (seq,[0,8,16,23]), (max(seq,hmax+1),[0,8,16,23])]:
            if next_seq != seq:
                for rid in [0,1]:
                    for t in range(seq,min(next_seq,hmax)):
                        kv[physical_locs[rid,t].long()].copy_(host_k[rid*maxseq+t])
                        vv[physical_locs[rid,t].long()].copy_(host_v[rid*maxseq+t])
                        tokens[rid,t]=t
                    latest=min(next_seq-1,hmax)
                    locs[rid,hmax]=physical_locs[rid,latest]
                    kv[physical_locs[rid,latest].long()].copy_(host_k[rid*maxseq+next_seq-1])
                    vv[physical_locs[rid,latest].long()].copy_(host_v[rid*maxseq+next_seq-1])
                    tokens[rid,hmax]=next_seq-1
                seq=next_seq
                seqs[:real].fill_(seq)
            selected=blocks
            selections.copy_(torch.tensor([selected]*batch,dtype=torch.int32,device='cuda'))
            before=(tokens.clone(),locs.clone(),lru.clone(),kv.clone(),vv.clone())
            if g is None:
                run()
            else:
                g.replay()
            check()
        return tuple(x.cpu() for x in (output,tokens,lru,kv,vv))

    count=0
    for seq in [1,31,32,33,63,64,65,96]:
        for seq_dtype in [torch.int32,torch.int64]:
            for req_dtype in [torch.int32,torch.int64]:
                for graph in [False,True]:
                    baseline=case(seq,hmax,stock,seq_dtype,req_dtype,graph)
                    identity=case(seq,hmax,patched,seq_dtype,req_dtype,graph)
                    for a,b in zip(baseline,identity):
                        assert torch.equal(a,b), 'stock/overlay uniform-Hmax mismatch'
                    case(seq,32,patched,seq_dtype,req_dtype,graph)
                    count+=1
    print(f'PASS {count} cases: stock/identity/reduced H; eager/graph; indices and complete selected K/V')

if __name__=='__main__':
    main()
