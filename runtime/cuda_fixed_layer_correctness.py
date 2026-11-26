"""CUDA correctness gate for a fixed offload set on the stock hot-pool layout. Explicit invocation only.

Requires the pinned SGLang image, an idle A100 (sm_80) and the repository root on
PYTHONPATH; never run it on an active service. No model is loaded. Each arm is one
synthetic TP rank, and every decode step runs all layers in model order on the same
selections, queries and pinned host history in every arm:

* ``stock``: every layer host-backed. The pinned ``HiSparseCoordinator.swap_in_selected_blocks``
  fills HiSparse slots and the pinned ``flash_decode_with_gqa_share_sparse`` reads them.
* ``direct``, ``resolver``, ``split``: S host-backed and every other layer resident at its
  logical rows. Offloaded layers call the pinned method with the plane that
  ``PlacementBinding.plane`` maps them to, or ``PlacementBinding.split_copy`` with
  ``features.split_copy``. Resident layers read ``req_to_token`` without slots
  (``features.resident_direct``) or through ``rlh_layer_placement::resolve_resident``.

Every fixed-arm output equals the stock output of its layer bit for bit, and the stock output
matches an FP32 softmax over the true history of the selected tokens. Host rows are written
like the pinned staging and decode backup: the prompt at admission and each decoded token one
step later, so the newest token exists only in the tail row (offloaded) or at its logical row
(resident). Returned slots hold the selected tokens' bytes, invalid positions and padded rows
stay -1, the tables of an S plane equal those of the same layer in the stock arm, and resolver
slots equal ``layer_placement.resident_slot_table``. Mid-run, requests finish and their slots
take new requests with new history and new logical pages. Eager decode and CUDA Graph replay of
whole steps (one bucket, changing real rows, padded rows, tables and pages rewritten between
replays) are both covered: at a small geometry, and with ``--production`` at H4096, top-k
16 x 128 and head dim 128 over 60 layers with the calibrated T40 list.

    python -m runtime.cuda_fixed_layer_correctness --allow-cuda-test [--production] --output /tmp/fixed-gate.json
"""
import argparse
import dataclasses
import json
from pathlib import Path
import random
import types

from runtime import cuda_combined_correctness as combined
from runtime import cuda_shared_lru_correctness as arena_gate
from runtime import layer_placement as lp
from runtime import layer_placement_device as adapter
from runtime import shared_lru as reference
from runtime import study_config as sc

SMALL = arena_gate.Geometry(layers=4, hot=128, page=16, slots=4, blocks=4, block=16, dim=64, context=1024)
PRODUCTION = arena_gate.Geometry(layers=60, hot=4096, page=128, slots=4, blocks=16, block=128, dim=128,
                                 context=12288)
SPEC = Path(__file__).with_name("fixtures") / "fixed_layer_plan_spec.example.json"
ARMS = {"direct": {"resident_direct": True}, "resolver": {}, "split": {"resident_direct": True, "split_copy": True}}
REQUESTS, BUCKET = 6, 4         # slots 0-2 serve requests; slot 3 only pads graph rows


def calibrated_t40(spec=SPEC):
    lists = json.loads(Path(spec).read_text())["offload_lists"]
    return tuple(next(item["offload_layers"] for item in lists if item["name"] == "calibrated_t40"))


def load_kernels():
    """The pinned swap method and decode consumer; split-copy modules load inside ``PlacementBinding``."""
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    return dict(swap=HiSparseCoordinator.swap_in_selected_blocks, consumer=combined.load_consumer())


class History:
    """True K/V of every request and layer, the pinned host planes backed up from it, pages and lengths."""

    def __init__(self, torch, g, device):
        self.torch, self.g = torch, g
        rows, per = REQUESTS * g.context, g.context // g.page
        row = torch.arange(rows, dtype=torch.int64).unsqueeze(1)
        element = torch.arange(g.dim, dtype=torch.int64).unsqueeze(0)

        def values(layer, salt):        # exact BF16 integers; elements 0-2 encode the host row
            code = torch.where(element == 0, row % 251, torch.where(element == 1, (row // 251) % 251,
                                                                    (row // 63001) % 251))
            mixed = torch.where(element < 3, code, (row * 11 + element * 5 + 37 * layer + salt) % 251)
            return (mixed - 125).reshape(rows, 1, g.dim).to(torch.bfloat16)

        self.true = {layer: (values(layer, 0), values(layer, 101)) for layer in range(g.layers)}
        self.host = {layer: tuple(torch.full_like(t, -999.0) for t in kv) for layer, kv in self.true.items()}
        if device == "cuda":
            self.host = {layer: tuple(t.pin_memory() for t in kv) for layer, kv in self.host.items()}
        self.logical_rows = (1 + REQUESTS * per) * g.page             # page 0 is the padding sink
        rng = random.Random(g.context)
        self.pages = [rng.sample(range(1 + r * per, 1 + (r + 1) * per), per) for r in range(REQUESTS)]
        self.req_to_token = torch.zeros((g.slots, g.context), dtype=torch.int32, device=device)
        self.request, self.seq = {}, {}

    def host_row(self, slot, token):
        return self.request[slot] * self.g.context + token

    def backup(self, slot, tokens):
        """Pinned staging (the whole prompt) and decode backup (one token, a step after its decode)."""
        index = self.torch.tensor([self.host_row(slot, t) for t in tokens], dtype=self.torch.long)
        for layer, planes in self.host.items():
            for host, true in zip(planes, self.true[layer]):
                host.index_copy_(0, index, true.index_select(0, index))

    def admit(self, slot, request, prompt):
        g = self.g
        self.request[slot], self.seq[slot] = request, prompt
        self.req_to_token[slot] = self.torch.tensor(
            [self.pages[request][t // g.page] * g.page + t % g.page for t in range(g.context)], dtype=self.torch.int32)
        self.backup(slot, range(prompt))

    def selection(self, rng, slot):
        """Distinct selector blocks, half the time with the newest token's block, sometimes padded with -1."""
        g, top = self.g, (self.seq[slot] + self.g.block - 1) // self.g.block
        blocks = rng.sample(range(top), min(g.blocks, top))
        if rng.random() < 0.5 and top - 1 not in blocks:
            blocks[-1] = top - 1
        blocks += [-1] * (g.blocks - len(blocks))
        if rng.random() < 0.2:
            blocks[-1] = -1
        return blocks

    def reference(self, layer, slot, blocks, q):
        """FP32 softmax over the true K/V of every selected position, in selection order."""
        torch, k, v = self.torch, *self.true[layer]
        tokens = [t for t in reference.selected_tokens(blocks, self.g.block, self.seq[slot]) if t >= 0]
        rows = torch.tensor([self.host_row(slot, t) for t in tokens], dtype=torch.long)
        keys, values = k.index_select(0, rows)[:, 0].float(), v.index_select(0, rows)[:, 0].float()
        return torch.softmax((q.float().cpu() @ keys.T) * self.g.dim ** -0.5, dim=-1) @ values


class Rank:
    """Native tables, S planes and resident planes of one arm on one synthetic rank."""

    def __init__(self, torch, g, history, offload, config, seed, device):
        self.torch, self.g, self.history, self.device = torch, g, history, device
        self.offload = tuple(offload)
        self.resident = tuple(layer for layer in range(g.layers) if layer not in self.offload)
        cuda, i32, H, page, S = torch.device(device), torch.int32, g.hot, g.page, len(self.offload)
        rows = page + g.slots * (H + page)
        self.k = torch.full((S, rows, 1, g.dim), -999.0, dtype=torch.bfloat16, device=cuda)
        self.v = torch.full_like(self.k, -999.0)
        locs = list(range(page, rows))
        random.Random(seed).shuffle(locs)
        self.locs = [locs[s * (H + page):(s + 1) * (H + page)] for s in range(g.slots)]
        self.resident_kv = {layer: tuple(torch.full((history.logical_rows, 1, g.dim), -999.0, dtype=torch.bfloat16,
                                                    device=cuda) for _ in "kv") for layer in self.resident}
        main = types.SimpleNamespace(start_layer=0, page_size=page, _rlh_offload_layers=self.offload,
                                     get_key_buffer=self.k.__getitem__, get_value_buffer=self.v.__getitem__)
        self.c = types.SimpleNamespace(
            is_m3_hisparse=True, skip_io=False, mem_pool_device=main, device_buffer_size=H, swap_in_block_size=960,
            item_size_bytes=g.dim * 2, num_real_reqs=torch.zeros(1, dtype=i32, device=cuda),
            req_device_buffer_tokens=torch.full((S, g.slots, H + page), -1, dtype=i32, device=cuda),
            req_device_buffer_token_locs=torch.full((S, g.slots, H + page), -1, dtype=i32, device=cuda),
            lru_slots=torch.arange(H, dtype=torch.int16, device=cuda).repeat(S, g.slots, 1),
            top_k_device_locs_buffer=torch.full((g.slots, H), -1, dtype=i32, device=cuda),
            req_to_host_pool=torch.full((g.slots, g.context), -1, dtype=torch.int64, device=cuda),
            mem_pool_host=types.SimpleNamespace(k_buffer=[history.host[layer][0] for layer in self.offload],
                                                v_buffer=[history.host[layer][1] for layer in self.offload]),
            token_to_kv_pool_allocator=types.SimpleNamespace(size_full=history.logical_rows))
        self.binding = None             # stock: the pinned method gets the absolute layer, as without the adapter
        if config is not None:
            kv = types.SimpleNamespace(main_pool=main, _rlh_resident_pool=types.SimpleNamespace(
                model_layer_ids=self.resident))
            self.binding = adapter.PlacementBinding(self.c, kv, config, "0" * 64, None, torch=torch)

    def admit(self, slot, request, prompt):
        """Tables of the pinned ``request_finished`` and ``alloc_device_buffer``, then the staged prompt rows."""
        torch, g, c, H = self.torch, self.g, self.c, self.g.hot
        c.req_device_buffer_tokens[:, slot] = -1
        c.req_device_buffer_tokens[:, slot, :H] = torch.arange(H, dtype=torch.int32)
        c.req_device_buffer_token_locs[:, slot] = torch.tensor(self.locs[slot], dtype=torch.int32)
        c.lru_slots[:, slot] = torch.arange(H, dtype=torch.int16)
        c.req_to_host_pool[slot] = torch.arange(request * g.context, (request + 1) * g.context)
        prompt_rows = slice(request * g.context, request * g.context + prompt)
        staged = torch.tensor(self.locs[slot][:min(prompt, H)], dtype=torch.long, device=self.device)
        for plane, layer in enumerate(self.offload):
            for buffer, true in zip((self.k, self.v), self.history.true[layer]):
                buffer[plane].index_copy_(0, staged, true[prompt_rows][:len(staged)].to(self.device))
        logical = self.history.req_to_token[slot, :prompt].long()
        for layer, planes in self.resident_kv.items():
            for buffer, true in zip(planes, self.history.true[layer]):
                buffer.index_copy_(0, logical, true[prompt_rows].to(self.device))

    def write(self, slot, token):
        """Stock KV write of one decoded token: tail row (its own row while short) or logical row."""
        history, row = self.history, self.history.host_row(slot, token)
        loc, logical = self.locs[slot][min(token, self.g.hot)], int(history.req_to_token[slot, token])
        for plane, layer in enumerate(self.offload):
            for buffer, true in zip((self.k, self.v), history.true[layer]):
                buffer[plane, loc] = true[row].to(self.device)
        for layer, planes in self.resident_kv.items():
            for buffer, true in zip(planes, history.true[layer]):
                buffer[logical] = true[row].to(self.device)
    def host_backed(self, layer):
        return self.binding is None or layer in self.binding.plane_of

    def decode(self, layer, rpi, seq, blocks, q, kernels):
        """The adapter's decode of one layer: attention output and a copy of the slots it read, or None."""
        g, binding, consumer = self.g, self.binding, kernels["consumer"]
        common = dict(q=q, sink=None, req_to_token=self.history.req_to_token, seq_lens=seq, slot_ids=rpi,
                      block_size=g.block, topk_idx=blocks.unsqueeze(0), page_size=g.page)
        if self.host_backed(layer):
            plane = layer if binding is None else binding.plane(layer)
            if binding is not None and binding.split is not None:
                slots = binding.split_copy(self.c, rpi, seq, blocks, layer, g.block)
            else:
                slots = kernels["swap"](self.c, rpi, seq, blocks, plane, g.block)
            slots = slots.clone()                   # the next layer reuses the native output buffer
            return consumer(k_cache=self.k[plane], v_cache=self.v[plane], hisparse_slots=slots.unsqueeze(0),
                            **common), slots
        k, v = self.resident_kv[layer]
        if binding.resident_direct:
            return consumer(k_cache=k, v_cache=v, hisparse_slots=None, **common), None
        slots = binding.resolve_resident(self.c, rpi, seq, blocks, self.history.req_to_token, g.block).clone()
        return consumer(k_cache=k, v_cache=v, hisparse_slots=slots.unsqueeze(0), **common), slots

    def check_slots(self, name, layer, rows, slots):
        """Bytes behind every slot of a host-backed layer, read at the layer's own plane; -1 elsewhere."""
        torch, g, history = self.torch, self.g, self.history
        plane, width = self.offload.index(layer), g.blocks * g.block
        for b, (slot, blocks) in enumerate(rows):
            tokens, got = reference.selected_tokens(blocks, g.block, history.seq[slot]), slots[b, :width].tolist()
            assert all((x == -1) == (t < 0) for x, t in zip(got, tokens)), f"{name} layer {layer} row {b}: validity"
            valid = [(x, history.host_row(slot, t)) for x, t in zip(got, tokens) if t >= 0]
            locations = torch.tensor([x for x, _ in valid], dtype=torch.long, device=self.device)
            host_rows = torch.tensor([r for _, r in valid], dtype=torch.long)
            for label, buffer, true in zip("KV", (self.k, self.v), history.true[layer]):
                assert torch.equal(buffer[plane].index_select(0, locations).cpu(), true.index_select(0, host_rows)), \
                    f"{name} layer {layer} row {b}: {label} bytes differ"
        assert all(x == -1 for row in slots[len(rows):, :width].tolist() for x in row), \
            f"{name} layer {layer}: padded graph rows must stay -1"

    def check_resolver(self, layer, rows, slots):
        history = self.history
        table = lp.resident_slot_table([blocks for _, blocks in rows], [slot for slot, _ in rows],
                                       [history.seq[slot] for slot, _ in rows], history.req_to_token.tolist(),
                                       num_real=len(rows), block_size=self.g.block)
        got = slots.tolist()
        assert got[:len(rows)] == table and all(x == -1 for row in got[len(rows):] for x in row), \
            f"layer {layer}: resident resolver slots differ from resident_slot_table"
def tensors(torch, g, history, rows, rng, seq_dtype, rpi_dtype, device):
    """Bucket inputs of one step: slots, lengths, then selections and queries per layer; slot 3 pads."""
    active = [slot for slot, _ in rows[0]]
    pad = BUCKET - len(active)
    rpi = torch.tensor(active + [g.slots - 1] * pad, dtype=rpi_dtype, device=device)
    seq = torch.tensor([history.seq[slot] for slot in active] + [1] * pad, dtype=seq_dtype, device=device)
    blocks = torch.tensor([[blocks for _, blocks in layer] + [[0] * g.blocks] * pad for layer in rows],
                          dtype=torch.int32, device=device)
    q = torch.tensor([rng.gauss(0.0, 0.01) for _ in range(len(rows) * BUCKET * combined.Q_HEADS * g.dim)])
    return rpi, seq, blocks, q.reshape(len(rows), BUCKET, combined.Q_HEADS, g.dim).to(torch.bfloat16).to(device)


def check_step(ranks, history, rows, q, outputs):
    torch, stock = history.torch, ranks["stock"]
    for layer, layer_rows in enumerate(rows):
        expected = outputs["stock"][layer][0]
        for b, (slot, blocks) in enumerate(layer_rows):
            got, want = expected[b].float().cpu(), history.reference(layer, slot, blocks, q[layer, b])
            assert torch.allclose(got, want, atol=combined.ATOL, rtol=combined.RTOL), \
                f"stock layer {layer} row {b}: max error {(got - want).abs().max().item()}"
        for name, rank in ranks.items():
            out, slots = outputs[name][layer]
            for b in range(len(layer_rows)):
                assert torch.equal(out[b], expected[b]), f"{name} layer {layer} row {b}: output differs from stock"
            if rank.host_backed(layer):
                rank.check_slots(name, layer, layer_rows, slots)
            elif slots is not None:
                rank.check_resolver(layer, layer_rows, slots)
    active = [slot for slot, _ in rows[0]]
    for name, rank in ranks.items():
        for plane, layer in enumerate(rank.offload if rank is not stock else ()):
            for table in ("req_device_buffer_tokens", "lru_slots"):
                ours, theirs = getattr(rank.c, table)[plane], getattr(stock.c, table)[layer]
                assert torch.equal(ours[active], theirs[active]), f"{name} layer {layer}: {table} differ from stock"


def run_case(torch, *, geometry, offload, graph, seq_dtype, rpi_dtype, seed, steps, kernels, device="cuda"):
    """One eager or graph case of every arm; returns the number of layer decodes checked."""
    g, H, rng = geometry, geometry.hot, random.Random(seed)
    history = History(torch, g, device)
    ranks = {"stock": Rank(torch, g, history, range(g.layers), None, seed, device)}
    for index, (name, features) in enumerate(ARMS.items()):
        config = sc.parse({"schema": sc.SCHEMA, "placement": {"mode": "fixed_layers", "offload_layers": list(offload)},
                           "features": features})
        ranks[name] = Rank(torch, g, history, offload, config, seed + 1 + index, device)

    def advance(slot, admitted=False):
        token = history.seq[slot]
        if not admitted:
            history.backup(slot, [token - 1])       # the previous newest token reaches the host one step later
        for rank in ranks.values():
            rank.write(slot, token)
        history.seq[slot] = token + 1

    def admit(slot, request, prompt):
        history.admit(slot, request, prompt)
        for rank in ranks.values():
            rank.admit(slot, request, prompt)
        advance(slot, admitted=True)

    for slot, prompt in enumerate((H + 5, H + 200, H - 3)):      # long, longer, short -> long
        admit(slot, slot, prompt)
    static, graphs, captured = None, {}, {}
    if graph:
        static = tensors(torch, g, history, [[(slot, [0] * g.blocks) for slot in range(3)]] * g.layers, rng,
                         seq_dtype, rpi_dtype, device)
        for rank in ranks.values():
            rank.c.num_real_reqs.fill_(0)           # warmup and capture leave every table and row untouched
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for rank in ranks.values():
                for layer in range(g.layers):
                    rank.decode(layer, static[0], static[1], static[2][layer], static[3][layer], kernels)
        torch.cuda.current_stream().wait_stream(side)
        for name, rank in ranks.items():
            graphs[name] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[name]):
                captured[name] = [rank.decode(layer, static[0], static[1], static[2][layer], static[3][layer],
                                              kernels) for layer in range(g.layers)]
        torch.cuda.synchronize()
    launches = 0
    for step in range(steps):
        if step == steps // 2:                  # requests finish; their slots take new requests, history and pages
            for slot, request, prompt in ((0, 3, H + 60), (2, 4, H - 1)):
                admit(slot, request, prompt)
        active = [(step + k) % 3 for k in range((1, 3, 2, 3)[step % 4])]       # real rows change per replay
        rows = [[(slot, history.selection(rng, slot)) for slot in active] for _ in range(g.layers)]
        inputs = tensors(torch, g, history, rows, rng, seq_dtype, rpi_dtype, device)
        for rank in ranks.values():
            rank.c.num_real_reqs.fill_(len(active))
        if graph:
            for target, source in zip(static, inputs):
                target.copy_(source)
            for replay in graphs.values():
                replay.replay()
            outputs = captured
        else:
            outputs = {name: [rank.decode(layer, inputs[0], inputs[1], inputs[2][layer], inputs[3][layer], kernels)
                              for layer in range(g.layers)] for name, rank in ranks.items()}
        if device == "cuda":
            torch.cuda.synchronize()
        check_step(ranks, history, rows, inputs[3], outputs)
        launches += g.layers * len(ranks)
        for slot in active:
            advance(slot)
    return launches


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--allow-cuda-test", action="store_true")
    parser.add_argument("--production", action="store_true", help="H4096 x 60 layers with the calibrated T40 list")
    parser.add_argument("--steps", type=int, help="decode steps per case (default 16, production 6)")
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--output", type=Path, help="optional JSON result; never overwritten")
    args = parser.parse_args()
    if not args.allow_cuda_test:
        parser.error("Run only in an approved idle test process with --allow-cuda-test")
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    import torch
    import sglang
    bad = {k: v for k, v in adapter.verify_digests(Path(sglang.__file__).resolve().parent).items() if v != "ok"}
    if bad:
        raise RuntimeError(f"Pinned SGLang source mismatch: {bad}")
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("A100 (sm_80) required")
    geometry, offload = (PRODUCTION, calibrated_t40()) if args.production else (SMALL, (1, 3))
    steps, kernels, results = args.steps or (6 if args.production else 16), load_kernels(), []
    dtypes = [(torch.int32, torch.int64)] + ([] if args.production else [(torch.int64, torch.int32)])
    for seq_dtype, rpi_dtype in dtypes:
        for graph in (False, True):
            launches = run_case(torch, geometry=geometry, offload=offload, graph=graph, seq_dtype=seq_dtype,
                                rpi_dtype=rpi_dtype, seed=args.seed, steps=steps, kernels=kernels)
            results.append(dict(seq_dtype=str(seq_dtype), rpi_dtype=str(rpi_dtype), graph=graph, launches=launches,
                                status="pass"))
    summary = dict(event="hisparse_fixed_layer_cuda_gate", status="pass", geometry=dataclasses.asdict(geometry),
                   offload=list(offload), arms=["stock", *ARMS], resident_kernel_sha256=adapter.kernel_sha256(),
                   split_copy=dict(ctas=adapter.SPLIT_COPY_CTAS, threads=adapter.SPLIT_COPY_THREADS),
                   cases=len(results), launches=sum(r["launches"] for r in results),
                   device=torch.cuda.get_device_name(), torch=torch.__version__, results=results)
    if args.output is not None:
        with args.output.open("x") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"PASS {summary['cases']} cases / {summary['launches']} layer decodes: fused and split swaps on S planes, "
          "resident direct and resolver reads, bit-equal to stock; eager and graph")


if __name__ == "__main__":
    main()
