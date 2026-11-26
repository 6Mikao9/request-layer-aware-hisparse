"""Same-history check of two arms' logical selections and KV before any performance comparison.

    python -m benchmarks.compare_selection_history --a arm-a.jsonl --b arm-b.jsonl --output check.json

Each arm writes one JSONL record per (request, decode step, layer, rank) with:

* ``request`` (alias), ``step``, ``layer``, ``rank``: the decode position of the record;
* ``history_sha256``: SHA-256 of the request's token IDs before this step (prompt and outputs);
* ``selected``: the logical token positions the selector chose, in any order (-1 for invalid slots);
* ``kv_sha256``: SHA-256 of the selected rows' K then V bytes, rows in ascending token position.

Records pair on (request, step, layer, rank). Only pairs with equal histories are comparable; there
the selected position sets and the KV digests must agree. The SGLang fixed-layer gate feeds its arms
the same selections and compares outputs and bytes (runtime.cuda_fixed_layer_correctness); this check
is framework-neutral, for arms whose selections come from their own indexers, such as a vLLM control
once its harness writes these records.
"""
import argparse
import json
from pathlib import Path

SCHEMA = "same-history-comparison-v1"
KEYS = ("request", "step", "layer", "rank")
FIELDS = {*KEYS, "history_sha256", "selected", "kv_sha256"}


def read_records(path):
    records = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (type(row) is not dict or set(row) != FIELDS or type(row["selected"]) is not list
                or any(type(row[key]) is not int for key in KEYS[1:]) or type(row["request"]) is not str
                or any(type(x) is not int for x in row["selected"])):
            raise ValueError(f"{path}:{number}: a record needs exactly {sorted(FIELDS)}")
        key = tuple(row[key] for key in KEYS)
        if key in records:
            raise ValueError(f"{path}:{number}: duplicate record {key}")
        records[key] = row
    return records


def compare(a, b):
    """Summary over paired records; the first divergence of each kind names its record key."""
    paired = sorted(set(a) & set(b))
    same = [key for key in paired if a[key]["history_sha256"] == b[key]["history_sha256"]]
    first = dict(history=next((key for key in paired if a[key]["history_sha256"] != b[key]["history_sha256"]),
                              None), selection=None, kv=None)
    counts = dict(selection=0, kv=0)
    for key in same:
        if {x for x in a[key]["selected"] if x >= 0} != {x for x in b[key]["selected"] if x >= 0}:
            counts["selection"] += 1
            first["selection"] = first["selection"] or key
        elif a[key]["kv_sha256"] != b[key]["kv_sha256"]:
            counts["kv"] += 1
            first["kv"] = first["kv"] or key
    status = "no_overlap" if not same else "fail" if counts["selection"] or counts["kv"] else "pass"
    return dict(schema=SCHEMA, status=status, records=dict(a=len(a), b=len(b), paired=len(paired),
                                                           same_history=len(same)),
                unpaired=dict(a_only=len(set(a) - set(b)), b_only=len(set(b) - set(a))),
                selection_mismatches=counts["selection"], kv_mismatches=counts["kv"],
                first_divergence={kind: None if key is None else dict(zip(KEYS, key)) for kind, key in first.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="JSON summary; never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    result = compare(read_records(args.a), read_records(args.b))
    if args.output is not None:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(dict(status=result["status"], **result["records"]), sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
