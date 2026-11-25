"""Freeze offload lists and static quotas from training calibration counts; CPU only, standard library.

    python -B -m runtime.combined_calibration --source data/a100-minimax-20261122.json \
        --training-bucket 8-16 --training-bucket 64-128 --heldout-bucket 16-32 \
        --offload-layers 40 --hot-rows 4096 --selected 2048 --output /tmp/combined-calibration.json \
        [--spec runtime/fixtures/combined_plan_spec.example.json --spec-output /tmp/spec.json]

The source is an anonymous native receipt whose ``calibration`` entries hold, per
input-length bucket, the selected and missed history rows of every layer on each
TP rank. Only training buckets are read; held-out buckets are named so the split
is checked to be disjoint, and their counts are never parsed. A layer's cost is
its slowest rank's miss rate pooled over the training buckets (sum of max-rank
misses over sum of selected rows), because a TP decode step waits for the slowest
rank.

* ``calibrated``: the T lowest-cost layers are offloaded (their swap-in is cheapest).
* ``reverse``: the T highest-cost layers, the contract's high-cost mechanism check.
* ``random``: ``random.Random(seed).sample`` of T eligible layers per preregistered seed.
* Static quota of each list (arm D): a layer's estimated per-request working set is
  ``H + K * cost`` (the rows its slowest rank misses per step at H); the ``T * H``
  rows are shared in proportion to it by largest remainder (ties to the lower layer
  ID) with at least K rows each, so a skewed demand gets skewed rows at equal bytes.

Each list records ``training_sha256`` (the source bytes) and ``list_sha256``
(canonical JSON of its sorted layers), the values ``placement.calibration``
freezes. With ``--spec`` the lists, quotas and the training digest are written
into a copy of a ``combined-plan-spec-v1`` file; everything else is kept.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import sys

from . import study_config as sc

SCHEMA = "combined-calibration-v1"
RANDOM_SEEDS = (13, 37, 71)     # preregistered in docs/implementation/layer-eviction.md


def list_digest(layers):
    return hashlib.sha256(json.dumps(sorted(layers), separators=(",", ":")).encode()).hexdigest()


def _counts(bucket, where):
    rows = bucket.get("counts_on_layers")
    if type(rows) is not list or not rows:
        raise ValueError(f"{where}: counts_on_layers must be a non-empty list")
    out, ranks = {}, None
    for row in rows:
        if type(row) is not dict or set(row) != {"layer", "selected", "misses"}:
            raise ValueError(f"{where}: each layer row needs exactly layer, selected and misses")
        layer, selected, misses = row["layer"], row["selected"], row["misses"]
        if type(layer) is not int or layer < 0 or layer in out:
            raise ValueError(f"{where}: layer IDs must be unique non-negative integers")
        if (type(selected) is not list or type(misses) is not list or not selected or len(selected) != len(misses)
                or any(type(x) is not int or x < 0 for x in selected + misses)
                or any(m > n for m, n in zip(misses, selected)) or not all(selected)):
            raise ValueError(f"{where} layer {layer}: per-rank selected > 0 and 0 <= misses <= selected")
        if ranks is not None and len(selected) != ranks:
            raise ValueError(f"{where}: every layer must report the same TP ranks")
        ranks = len(selected)
        out[layer] = (selected, misses)
    return out


def read_training(raw, training, heldout):
    """Per-layer (max-rank misses, selected rows) pooled over the training buckets only."""
    if not training or len(set(training)) != len(training) or len(set(heldout)) != len(heldout):
        raise ValueError("Name each training and held-out bucket once")
    if set(training) & set(heldout):
        raise ValueError(f"Training and held-out buckets overlap: {sorted(set(training) & set(heldout))}")
    data = sc.loads(raw)
    buckets = data.get("calibration")
    if type(buckets) is not list:
        raise ValueError("Source needs a calibration list of length buckets")
    labels = [b.get("input_ki_tokens") if type(b) is dict else None for b in buckets]
    if len(set(labels)) != len(labels) or any(type(x) is not str for x in labels):
        raise ValueError("Calibration buckets need unique input_ki_tokens labels")
    missing = sorted(set(training) - set(labels))
    if missing:
        raise ValueError(f"Training bucket(s) {missing} are not in the source")
    pooled, layers = {}, None
    for label, bucket in zip(labels, buckets):
        if label not in training:
            continue                                    # held-out and unused buckets are never parsed
        counts = _counts(bucket, f"bucket {label}")
        if layers is not None and set(counts) != layers:
            raise ValueError(f"bucket {label}: layers differ from the other training buckets")
        layers = set(counts)
        for layer, (selected, misses) in counts.items():
            m, n = pooled.get(layer, (0, 0))
            pooled[layer] = (m + max(misses), n + max(selected))
    return {layer: pooled[layer] for layer in sorted(pooled)}, data


def static_quota(layers, cost, *, hot_rows, selected):
    """Rows per layer of one list: shares of T * H in proportion to H + K * cost, at least K each."""
    weights = {layer: hot_rows + selected * cost[layer] for layer in layers}
    rows, free, total = {}, list(layers), len(layers) * hot_rows
    while True:
        weight = sum(weights[layer] for layer in free)
        ideal = {layer: total * weights[layer] / weight for layer in free}
        low = [layer for layer in free if ideal[layer] < selected]
        if not low:
            break
        for layer in low:                       # pinned at one complete selection; the rest share what remains
            rows[layer] = selected
        free = [layer for layer in free if layer not in low]
        total -= selected * len(low)
    for layer in free:
        rows[layer] = int(ideal[layer])
    spare = total - sum(rows[layer] for layer in free)
    for layer in sorted(free, key=lambda x: (-(ideal[x] - int(ideal[x])), x))[:spare]:
        rows[layer] += 1
    return {layer: rows[layer] for layer in layers}


def calibrate(raw, *, source_name, training, heldout, sizes, hot_rows, selected, seeds=RANDOM_SEEDS):
    """Lists and quotas for every T in ``sizes`` from the training buckets of one source."""
    if type(hot_rows) is not int or type(selected) is not int or not 0 < selected <= hot_rows:
        raise ValueError("Require 0 < selected width K <= H")
    if not seeds or len(set(seeds)) != len(seeds) or len(seeds) < 3:
        raise ValueError("Use at least three distinct preregistered random seeds")
    pooled, data = read_training(raw, tuple(training), tuple(heldout))
    layers, digest = tuple(pooled), hashlib.sha256(raw).hexdigest()
    cost = {layer: Fraction(m, n) for layer, (m, n) in pooled.items()}
    lists, quotas = [], {}
    for T in sizes:
        if type(T) is not int or not 0 < T < len(layers):
            raise ValueError(f"T={T}: S must leave a resident layer; all {len(layers)} layers are the all-layer arena")
        chosen = [(f"calibrated_t{T}", sorted(sorted(layers, key=lambda x: (cost[x], x))[:T]), None),
                  (f"reverse_t{T}", sorted(sorted(layers, key=lambda x: (-cost[x], x))[:T]), None)]
        chosen += [(f"random_t{T}_s{seed}", sorted(random.Random(seed).sample(layers, T)), seed) for seed in seeds]
        for name, offload, seed in chosen:
            item = dict(name=name, selection="calibrated" if seed is None else "random", offload_layers=offload)
            if seed is None:
                item["calibration"] = dict(training_sha256=digest, list_sha256=list_digest(offload))
            else:
                item["seed"] = seed
            lists.append(item)
            rows = static_quota(offload, cost, hot_rows=hot_rows, selected=selected)
            quotas[name] = {str(layer): count for layer, count in rows.items()}
    return dict(
        schema=SCHEMA,
        source=dict(name=source_name, sha256=digest, model=data.get("model"),
                    tensor_parallel=data.get("tensor_parallel")),
        split=dict(training_buckets=list(training), heldout_buckets=list(heldout), heldout_counts_read=False),
        geometry=dict(hot_rows=hot_rows, selected=selected, eligible_layers=list(layers)),
        method=dict(
            cost="slowest-rank miss rate pooled over training buckets: sum(max_rank misses) / sum(max_rank selected)",
            calibrated="the T lowest-cost layers are offloaded; ties to the lower layer ID",
            reverse="the T highest-cost layers are offloaded (high-cost mechanism check)",
            random=f"sorted(random.Random(seed).sample(eligible, T)) for seeds {list(seeds)}",
            static_quota="T*H rows shared in proportion to H + K*cost by largest remainder, at least K per layer"),
        layer_cost={str(layer): dict(max_rank_misses=m, selected=n, miss_rate=round(float(cost[layer]), 6))
                    for layer, (m, n) in pooled.items()},
        offload_lists=lists, static_quota_rows=quotas)


def update_spec(raw, calibration):
    """A ``combined-plan-spec-v1`` with the frozen lists, their quotas and the training digest replaced."""
    spec = sc.loads(raw)
    if spec.get("schema") != "combined-plan-spec-v1" or type(spec.get("split")) is not dict:
        raise ValueError("--spec must be a combined-plan-spec-v1 file")
    spec["split"] = dict(spec["split"], training_sha256=calibration["source"]["sha256"])
    spec["offload_lists"] = calibration["offload_lists"]
    spec["static_quota_rows"] = calibration["static_quota_rows"]
    return spec


def dumps(value, indent=0, width=116, lead=0):
    """JSON with one key per line; arrays and objects of scalars are packed into wrapped lines.

    ``lead`` is the column where ``value`` starts, so the first packed line respects ``width`` too.
    """
    pad, inner = " " * indent, " " * (indent + 2)
    if isinstance(value, dict) and value:
        if all(not isinstance(v, (dict, list)) for v in value.values()):
            return _pack("{", [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in value.items()], "}", inner, width,
                         lead)
        items = [(f"{inner}{json.dumps(k)}: ", v) for k, v in value.items()]
        return "{\n" + ",\n".join(key + dumps(v, indent + 2, width, len(key)) for key, v in items) + "\n" + pad + "}"
    if isinstance(value, list) and value:
        if all(not isinstance(v, (dict, list)) for v in value):
            return _pack("[", [json.dumps(v) for v in value], "]", inner, width, lead)
        return ("[\n" + ",\n".join(inner + dumps(v, indent + 2, width, len(inner)) for v in value) + "\n" + pad
                + "]")
    return json.dumps(value)


def _pack(opening, tokens, closing, inner, width, lead):
    lines, line, start = [], opening, lead
    for index, token in enumerate(tokens):
        piece = token + ("," if index < len(tokens) - 1 else closing)
        if line not in ("", opening) and start + len(line) + 1 + len(piece) > width:
            lines.append(line)
            line, start = inner + piece, 0
        else:
            line += ("" if line == opening else " ") + piece
    return "\n".join(lines + [line])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Freeze offload lists and static quotas from training counts.")
    parser.add_argument("--source", type=Path, required=True, help="anonymous receipt with calibration buckets")
    parser.add_argument("--training-bucket", action="append", required=True, help="repeat; the only buckets read")
    parser.add_argument("--heldout-bucket", action="append", default=[], help="repeat; checked disjoint, never read")
    parser.add_argument("--offload-layers", type=int, action="append", required=True, metavar="T",
                        help="number of offloaded layers per list; repeat for several T")
    parser.add_argument("--hot-rows", type=int, default=4096, help="native H per offloaded layer")
    parser.add_argument("--selected", type=int, default=2048, help="flattened selection width K")
    parser.add_argument("--seed", type=int, action="append", help=f"random-list seeds (default {list(RANDOM_SEEDS)})")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec", type=Path, help="combined-plan-spec-v1 to receive the frozen lists")
    parser.add_argument("--spec-output", type=Path)
    args = parser.parse_args(argv)
    if (args.spec is None) != (args.spec_output is None):
        parser.error("--spec and --spec-output go together")
    if any(path is not None and path.exists() for path in (args.output, args.spec_output)):
        parser.error("Refusing to overwrite an existing output")
    try:
        raw = args.source.read_bytes()
        result = calibrate(raw, source_name=args.source.as_posix(), training=args.training_bucket,
                           heldout=args.heldout_bucket, sizes=args.offload_layers, hot_rows=args.hot_rows,
                           selected=args.selected, seeds=tuple(args.seed or RANDOM_SEEDS))
        spec = update_spec(args.spec.read_bytes(), result) if args.spec is not None else None
    except (OSError, ValueError) as error:
        parser.error(str(error))
    with args.output.open("x") as handle:
        handle.write(dumps(result) + "\n")
    if spec is not None:
        with args.spec_output.open("x") as handle:
            handle.write(dumps(spec) + "\n")
    print(json.dumps(dict(status="pass", output=str(args.output), training_sha256=result["source"]["sha256"],
                          lists={item["name"]: item["offload_layers"] for item in result["offload_lists"]})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
