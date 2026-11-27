"""A100 gate of the elastic per-layer HiSparse device pieces (issue #13); run before any serving benchmark.

    python -m runtime.cuda_elastic_gate --output elastic_gate.json

1. CUDA VMM: an ElasticMemory over the pinned SGLang VMM helpers moves chunks between buffers through offload,
   grow, backfill and shrink states; rows that stay mapped keep their bytes, newly mapped rows read zero.
2. The fused smooth route (Triton) against its torch statement, token and block selection, both modes.
3. Host transfers of ElasticKV (sgl_kernel transfer_kv_per_layer / _mla) against torch indexing.
Prints one JSON receipt and exits non-zero when a check fails.
"""
import argparse
import json
from pathlib import Path
import sys


def vmm_check(torch):
    from . import elastic_memory as em
    chunk = 2 << 20
    rows = chunk // 512                         # 512-byte rows: 4096 rows per chunk
    specs = [em.BufferSpec(f"kv{layer}", 512, em.SWITCHABLE, layer=layer, pad_rows=64, shape=(256,),
                           dtype="bfloat16") for layer in range(4)]
    layout = em.Layout(specs, chunk_bytes=chunk, max_tokens=4 * rows, hot_rows=rows, page_tokens=64)
    memory = em.ElasticMemory(layout, em.CudaVmmBackend(torch), physical_chunks=12, torch=torch)
    states = [dict(tokens=2 * rows, hot_rows=rows // 2, offloaded=frozenset()),
              dict(tokens=3 * rows, hot_rows=rows // 2, offloaded=frozenset({0, 1})),
              dict(tokens=2 * rows, hot_rows=rows // 2, offloaded=frozenset({0, 1}), filling=frozenset({1})),
              dict(tokens=2 * rows, hot_rows=rows // 2, offloaded=frozenset({0}))]
    marks, ok = {}, True        # (buffer, region) -> (physical chunk of the region's first chunk, bytes written)
    for step, state in enumerate(states, 1):
        memory.resize(**state)
        memory.check()
        for name, buf in memory.buffers.items():
            for region in ("hot", "resident"):
                first = buf.hot_offset * buf.spec.row_bytes // chunk if region == "resident" else 0
                physical = buf.chunks[first]
                if physical is None:
                    marks.pop((name, region), None)
                    continue
                probe = memory.view(name, region)[64:128]
                previous = marks.get((name, region))
                if previous is not None and previous[0] == physical:
                    ok &= torch.equal(probe, previous[1])       # stayed mapped: bytes kept
                else:
                    ok &= bool((probe == 0).all())              # newly mapped: zero-filled
                value = torch.full_like(probe, float(step))
                probe.copy_(value)
                marks[(name, region)] = (physical, value)
    torch.cuda.synchronize()
    return dict(ok=bool(ok), chunks=len(memory.handles), free=memory.free_chunks, states=len(states))


def route_check(torch):
    from . import elastic_route
    device = torch.device("cuda")
    return dict(tokens=elastic_route.self_check(torch, device), tokens_c4=elastic_route.self_check(
        torch, device, ratio=4), blocks=elastic_route.self_check(torch, device, block=128))


def transfer_check(torch):
    from . import elastic_kv as ekv
    d, ok = torch.device("cuda"), True
    host_k = [torch.randn(256, 1, 128, dtype=torch.bfloat16).pin_memory() for _ in range(2)]
    host_v = [torch.randn(256, 1, 128, dtype=torch.bfloat16).pin_memory() for _ in range(2)]
    host = type("Host", (), dict(k_buffer=host_k, v_buffer=host_v, kv_buffer=host_k))()
    k = [torch.zeros(512, 1, 128, dtype=torch.bfloat16, device=d) for _ in range(2)]
    v = [torch.zeros(512, 1, 128, dtype=torch.bfloat16, device=d) for _ in range(2)]
    src, dst = torch.tensor([3, 7, 9], device=d), torch.tensor([100, 5, 300], device=d)
    for rows in (ekv.MHALayers(torch, k, v, host), ekv.MLALayers(torch, k, host)):
        rows.h2d(1, src, dst)
        rows.d2d(1, dst, dst + 1)
        rows.d2h(1, dst + 1, src + 100)
        torch.cuda.synchronize()
        ok &= torch.equal(k[1][dst + 1].cpu(), host_k[1][src.cpu()])
        ok &= torch.equal(host_k[1][(src + 100).cpu()], host_k[1][src.cpu()])
    return dict(ok=bool(ok))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    import torch
    if options.output.exists():
        parser.error(f"refusing to overwrite {options.output}")
    receipt = dict(event="hisparse_elastic_gate", device=torch.cuda.get_device_name(),
                   capability=list(torch.cuda.get_device_capability()), torch=torch.__version__,
                   vmm=vmm_check(torch), route=route_check(torch), transfer=transfer_check(torch))
    passed = receipt["vmm"]["ok"] and all(receipt["route"].values()) and receipt["transfer"]["ok"]
    receipt["status"] = "pass" if passed else "fail"
    options.output.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps(receipt, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
