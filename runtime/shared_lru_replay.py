"""CPU replay of a tracked synthetic fixture through one study-config arm.

    python -B -m runtime.shared_lru_replay --fixture runtime/fixtures/shared_lru_fixture.json \
        --config runtime/fixtures/shared_lru_arms/shared_arena.json --output /tmp/shared.json

Stock layout replays ``NativeLayerLRU``; arena layouts replay ``SharedLRUReference``
with ``KernelMirror`` in lockstep. ``DirectReader`` checks every resolved
position against synthesized bytes. Selections depend only on the fixture, so
every arm sees identical inputs (``selection_sha256``). Exit status is nonzero
for invalid or unsupported configs and for any correctness mismatch. Results
are miss counts on synthetic traces, not service measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

from . import shared_lru as s
from . import study_config as sc

FIXTURE_SCHEMA = "shared-lru-fixture-v1"
RESULT_SCHEMA = "shared-lru-replay-result-v1"
PROFILES = ("cycle", "uniform", "skewed")
PENDING_HARDWARE = ("sm80_build", "cuda_graph_capture_replay", "physical_memory_receipt",
                    "serving_tpot_throughput")


def _int(value, where, low=0):
    if type(value) is not int or value < low:
        raise ValueError(f"{where} must be an integer >= {low}")
    return value


def read_fixture(path):
    raw = Path(path).read_bytes()
    data = sc.loads(raw)
    keys = {"schema", "seed", "layers", "hot_rows", "page", "block_size", "blocks", "item_bytes", "ranks",
            "steps", "working_set_blocks", "requests", "reuse"}
    if set(data) != keys or data["schema"] != FIXTURE_SCHEMA:
        raise ValueError(f"Fixture must be {FIXTURE_SCHEMA} with fields {sorted(keys)}")
    for name in ("seed", "hot_rows", "page", "block_size", "blocks", "item_bytes", "steps"):
        _int(data[name], name, 0 if name == "seed" else 1)
    layers = data["layers"]
    if type(layers) is not list or not layers or layers != sorted(set(layers)):
        raise ValueError("layers must be sorted unique absolute IDs")
    if (type(data["working_set_blocks"]) is not list or len(data["working_set_blocks"]) != len(layers)
            or type(data["ranks"]) is not list or not data["ranks"]):
        raise ValueError("working_set_blocks needs one entry per layer; ranks must be non-empty")
    slots = set()
    for item in data["requests"] + data["reuse"]:
        if set(item) - {"slot", "generation", "prompt", "profile", "after_step"} or item["profile"] not in PROFILES:
            raise ValueError(f"Invalid request entry {item}")
        _int(item["prompt"], "prompt", 1)
    for item in data["requests"]:
        if item["slot"] in slots or "after_step" in item:
            raise ValueError("Initial requests need distinct slots and no after_step")
        slots.add(item["slot"])
    return data, hashlib.sha256(raw).hexdigest()
def selection(fixture, rank, request, generation, plane, step, seq_len):
    """Ordered selector blocks for one request/layer/step; identical for every arm."""
    width, block_size = fixture["blocks"], fixture["block_size"]
    top = (seq_len + block_size - 1) // block_size     # the last block may be partial
    key = f"{fixture['seed']}/{rank}/{request['slot']}/{generation}/{plane}/{step}".encode()
    rng = random.Random(int(hashlib.sha256(key).hexdigest()[:16], 16))
    if request["profile"] == "cycle":                  # per-layer working sets of different sizes
        start = 1 + request["slot"] % 3
        pool = [b for b in range(start, start + fixture["working_set_blocks"][plane]) if b < top - 1]
        pool = pool or list(range(top))
        blocks = [pool[(step * width + j) % len(pool)] for j in range(width)]
    elif request["profile"] == "uniform":
        blocks = [rng.randrange(top) for _ in range(width)]
    else:                                              # skewed: a small hot set plus a random tail
        hot = list(range(min(top, 3)))
        blocks = [rng.choice(hot) if rng.random() < 0.75 else rng.randrange(top) for _ in range(width)]
    if rng.random() < 0.5:
        blocks[-1] = top - 1                           # local block holding the newest token
    if rng.random() < 0.2:
        blocks[-1] = -1                                # trailing invalid selector entry
    if width > 1 and rng.random() < 0.2:
        blocks[1] = blocks[0]                          # duplicate block
    return blocks


class _Arm:
    """One TP rank of one arm on the CPU physical model."""

    def __init__(self, config, fixture, rank):
        layers, H, page = tuple(fixture["layers"]), fixture["hot_rows"], fixture["page"]
        slots = len(fixture["requests"]) + 1
        self.total_locs = -(-((H + page) * slots + page) // page) * page
        self.fixture, self.rank, self.H, self.mirror = fixture, rank, H, None
        self.plan = None
        if config.uses_arena:
            self.plan = s.plan_arena(config, layers, H, fixture["blocks"] * fixture["block_size"])
            self.engine = s.SharedLRUReference(self.plan, page=page, item_bytes=fixture["item_bytes"],
                                               rank=rank, total_locs=self.total_locs)
            self.mirror = s.KernelMirror(self.plan, plane_rows=self.total_locs)
        else:
            self.engine = s.NativeLayerLRU(layers, H, page=page, item_bytes=fixture["item_bytes"], rank=rank,
                                           total_locs=self.total_locs)
        self.reader = s.DirectReader(self.engine.device, rank=rank, item_bytes=fixture["item_bytes"],
                                     hot_rows=H, planes=len(layers))
        self.seq = {}
        self.peak_allocated = self.peak_logical = 0

    def admit(self, request, generation, epoch):
        slot, prompt, layers = request["slot"], request["prompt"], tuple(self.fixture["layers"])
        if self.plan is None:
            locs = self.engine.admit(slot, generation, seq_len=prompt)
        else:
            locs = self.engine.admit(slot, generation, self.rank, layers, min(prompt, self.H), self.plan.rows,
                                     seq_len=prompt, epoch=epoch)
            self.mirror.init_slot(slot, generation, locs)
        self.seq[slot] = (generation, prompt)
        self.advance(slot)                             # first decode token, as at the first decode step

    def advance(self, slot):
        generation, seq_len = self.seq[slot]
        lease = self.engine.backup(slot, generation)
        if lease is not None:
            self.engine.complete(lease, "backup")
        self.engine.advance(slot, generation)
        self.seq[slot] = (generation, seq_len + 1)

    def finish(self, slot):
        if not self.engine.finish(slot, self.seq.pop(slot)[0]):
            raise AssertionError("Live request did not finish")
        if self.mirror:
            self.mirror.reset_slot(slot)
    def resolve(self, layer, plane, rows, epoch):
        block_size = self.fixture["block_size"]
        batch = s.Batch(layer, block_size, tuple(rows))
        results = self.engine.resolve(batch, epoch)
        checked = 0
        for sel, res in zip(rows, results):
            if not sel.real:
                if res.classes != [s.PAD] * len(res.classes) or res.unique:
                    raise AssertionError("Padded graph row produced state or counters")
                continue
            tokens = s.selected_tokens(sel.blocks, block_size, sel.seq_len)
            locs = self.engine.locs(sel.slot, sel.generation)
            if self.mirror is not None:
                flat, _, counters = self.mirror.resolve_row(sel.slot, plane, sel.blocks, block_size, sel.seq_len)
                if [self.mirror.address(x) for x in flat] != res.addresses or counters["misses"] != res.misses:
                    raise AssertionError(f"Kernel mirror diverged at layer {layer} slot {sel.slot}")
            checked += self.reader.check(res, generation=sel.generation, layer=layer, plane=plane,
                                         tokens=tokens, seq_len=sel.seq_len, locs=locs)
            if res.lease is not None:
                self.engine.complete(res.lease, "attention")
        self.peak_allocated = max(self.peak_allocated, self.engine.allocator.allocated_rows)
        if self.plan is not None:
            logical = sum(sum(self.engine.occupancy(slot, gen).values()) for slot, (gen, _) in self.seq.items())
            self.peak_logical = max(self.peak_logical, logical)
        return results, checked


def _accumulate(total, res):
    total["unique"] += res.unique
    total["hits"] += res.hits
    total["misses"] += res.misses
    total["bypassed"] += res.bypassed
    total["bytes"] += res.bytes_requested
    total["breaches"] += res.floor_breaches
    total["latest"] += res.classes.count(s.LATEST)
    total["invalid"] += res.classes.count(s.PAD)
    total["direct"] += res.classes.count(s.DIRECT)


def replay(fixture, config):
    fields = s.COUNTER_FIELDS + ("direct",)
    selections, outputs, device = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    per_rank, checked, ledgers = {}, 0, {}
    for rank in fixture["ranks"]:
        arm, epoch, live = _Arm(config, fixture, rank), 0, {}
        layers = {layer: dict.fromkeys(fields, 0) for layer in fixture["layers"]}
        for request in fixture["requests"]:
            epoch += 1
            arm.admit(request, request["generation"], epoch)
            live[request["slot"]] = request
        for step in range(fixture["steps"]):
            for event in fixture["reuse"]:
                if event["after_step"] == step:
                    arm.finish(event["slot"])
                    epoch += 1
                    arm.admit(event, event["generation"], epoch)
                    live[event["slot"]] = event
            for plane, layer in enumerate(fixture["layers"]):
                epoch = (epoch + 1) & s.EPOCH_MASK
                rows = []
                for slot in sorted(live):
                    generation, seq_len = arm.seq[slot]
                    blocks = selection(fixture, rank, live[slot], generation, plane, step, seq_len)
                    rows.append(s.Selection(slot, generation, tuple(blocks), seq_len))
                rows.append(s.Selection(rows[0].slot, rows[0].generation, rows[0].blocks, rows[0].seq_len, False))
                results, count = arm.resolve(layer, plane, rows, epoch)
                checked += count
                selections.update(json.dumps([rank, layer, [list(r.blocks) + [r.seq_len] for r in rows]]).encode())
                outputs.update(json.dumps([[r.addresses, r.classes] for r in results]).encode())
                for res in results:
                    _accumulate(layers[layer], res)
            for slot in sorted(live):
                arm.advance(slot)
        for key, value in sorted(arm.engine.device.items()):
            device.update(json.dumps([rank, list(key), value[0].hex(), value[1].hex()]).encode())
        if arm.mirror is not None and arm.mirror.status:
            raise AssertionError(f"Kernel mirror status {arm.mirror.status}")
        per_rank[str(rank)] = {str(layer): counts for layer, counts in layers.items()}
        ledger = s.byte_ledger(layers=fixture["layers"], hot_rows=fixture["hot_rows"], page=fixture["page"],
                               item_bytes=fixture["item_bytes"], slots=len(fixture["requests"]) + 1,
                               pool_rows=arm.total_locs, plan=arm.plan, counters=config.diagnostics == "counters")
        ledger["fake_allocator"] = dict(peak_physical_rows_allocated=arm.peak_allocated,
                                        peak_logical_rows_valid=arm.peak_logical if arm.plan is not None else None)
        ledgers[str(rank)] = ledger
    total = dict.fromkeys(fields, 0)
    for layers in per_rank.values():
        for counts in layers.values():
            for name in fields:
                total[name] += counts[name]
    return dict(selection_sha256=selections.hexdigest(), output_sha256=outputs.hexdigest(),
                device_bytes_sha256=device.hexdigest(), counters=dict(total=total, per_rank=per_rank),
                history_hit_rate=total["hits"] / total["unique"] if total["unique"] else None,
                checks=dict(direct_reader_positions=checked,
                            kernel_mirror="lockstep" if config.uses_arena else "not applicable"),
                ledger=ledgers)
def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU replay of a tracked fixture through one study arm.")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="hisparse-study-config-v1 arm file")
    parser.add_argument("--output", type=Path, required=True, help="result JSON; never overwritten")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError(f"Refusing to overwrite existing output {args.output}")
        fixture, fixture_digest = read_fixture(args.fixture)
        config, config_digest = sc.read(args.config)
        reasons = sc.unsupported_reasons(config)
        if reasons:
            raise ValueError("; ".join(reasons))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    try:
        result = replay(fixture, config)
    except (AssertionError, ValueError, s.Deferred, s.StaleCompletion, s.RequestFailed) as error:
        print(json.dumps(dict(status="fail", error=str(error))), file=sys.stderr)
        return 1
    result = dict(schema=RESULT_SCHEMA, status="pass", fixture_sha256=fixture_digest,
                  config_file_sha256=config_digest, config_sha256=config.digest,
                  resolved_config=config.as_dict(), capabilities=sc.capabilities(config),
                  engine="SharedLRUReference+KernelMirror" if config.uses_arena else "NativeLayerLRU",
                  pending_hardware=list(PENDING_HARDWARE), **result)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(dict(status="pass", output=str(args.output), engine=result["engine"],
                          history_hit_rate=result["history_hit_rate"],
                          selection_sha256=result["selection_sha256"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
