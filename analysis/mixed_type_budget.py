"""Train-only exact allocation for an explicitly fixed concurrent request mix.

This is offline capacity replay. Logical active slots are not measured backing
bytes. It neither changes a live cache nor predicts an end-to-end speedup.
"""

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re


CAPACITIES = (2048, 3072, 4096, 6144, 8192)
MODES = ("uniform", "layer", "type", "joint")
PARTITION_SCHEMA = "hisparse-fixed-mix-anonymous-partition-v1"
CURVE_SCHEMA = "hisparse-fixed-mix-curve-receipt-v1"
EVALUATION_ROLES = {"historical_evaluation", "independent_evaluation"}
ROW_SCOPES = {"post_warmup", "strict_long"}


def validate(curves, mix, capacities, baseline):
    if (not mix or any(type(n) is not int or n <= 0 for n in mix.values())
            or set(curves) != set(mix)):
        raise ValueError("Supply exactly the positive active request types in the fixed mix")
    if (baseline not in capacities or len(set(capacities)) != len(capacities)
            or any(type(c) is not int or c < 2048 or c > 32767 or c % 128 for c in capacities)):
        raise ValueError("Keep Top-K=2048 and a valid page-aligned capacity grid including the baseline")
    layers = sorted(next(iter(curves.values())))
    if not layers or any(set(table) != set(layers) for table in curves.values()):
        raise ValueError("Every active type must cover the same nonempty layer set")
    for table in curves.values():
        for row in table.values():
            if (set(row) != {"selected", *capacities} or type(row["selected"]) is not int
                    or row["selected"] <= 0
                    or any(type(row[c]) is not int or not 0 <= row[c] <= row["selected"] for c in capacities)):
                raise ValueError("Incomplete integer replay counts or zero exposure")
    return layers


def exact_choices(options, budget):
    """Exact multiple-choice knapsack; costs are integer slots, losses rational.

    Ties minimize displacement from uniform, then use sorted capacity choices.
    No monotone or concave miss-curve assumption is necessary.
    """
    low = sum(min(option[0] for option in group) for group in options)
    high = sum(max(option[0] for option in group) for group in options)
    states = {0: (Fraction(0), 0, ())}
    for group in options:
        low -= min(option[0] for option in group)
        high -= max(option[0] for option in group)
        next_states = {}
        for used, (loss, change, choices) in states.items():
            for cost, value, displacement, capacity in group:
                total = used + cost
                if not budget - high <= total <= budget - low:
                    continue
                candidate = loss + value, change + displacement, choices + (capacity,)
                if total not in next_states or candidate < next_states[total]:
                    next_states[total] = candidate
        states = next_states
    if budget not in states:
        raise ValueError("Capacity grid cannot realize the exact mixed-request budget")
    return states[budget][2]


def allocation(curves, mix, mode, capacities=CAPACITIES, baseline=4096):
    layers = validate(curves, mix, capacities, baseline)
    if mode not in MODES:
        raise ValueError("Unknown allocation dimension")
    cells = [(kind, layer) for kind in sorted(mix) for layer in layers]
    if mode == "uniform":
        return dict.fromkeys(cells, baseline)
    if mode == "layer":
        groups = [[(kind, layer) for kind in sorted(mix)] for layer in layers]
    elif mode == "type":
        groups = [[(kind, layer) for layer in layers] for kind in sorted(mix)]
    else:
        groups = [[cell] for cell in cells]
    options = []
    for group in groups:
        multiplicity = sum(mix[kind] for kind, _ in group)
        options.append([
            (capacity * multiplicity,
             sum(mix[kind] * Fraction(curves[kind][layer][capacity], curves[kind][layer]["selected"])
                 for kind, layer in group),
             abs(capacity - baseline) * multiplicity, capacity)
            for capacity in sorted(capacities)
        ])
    budget = baseline * len(layers) * sum(mix.values())
    choices = exact_choices(options, budget)
    table = {cell: capacity for group, capacity in zip(groups, choices) for cell in group}
    if sum(mix[kind] * value for (kind, _), value in table.items()) != budget:
        raise AssertionError("Mixed-request budget was not conserved")
    return table


def score(curves, mix, table):
    layers = sorted(next(iter(curves.values())))
    per_type = {
        kind: sum(Fraction(curves[kind][layer][table[kind, layer]], curves[kind][layer]["selected"])
                  for layer in layers) / len(layers)
        for kind in mix
    }
    weighted = sum(mix[kind] * value for kind, value in per_type.items()) / sum(mix.values())
    return {"fixed_mix_mean_layer_miss_rate": float(weighted),
            "per_type_mean_layer_miss_rate": {kind: float(value) for kind, value in per_type.items()}}


def compare(train, test, mix, capacities=CAPACITIES, baseline=4096):
    layers = validate(train, mix, capacities, baseline)
    if validate(test, mix, capacities, baseline) != layers:
        raise ValueError("Train/test layer domains differ")
    policies = {}
    for mode in MODES:
        table = allocation(train, mix, mode, capacities, baseline)
        policies[mode] = {
            "capacities": {kind: {layer: table[kind, layer] for layer in layers} for kind in sorted(mix)},
            "active_slots_per_request_by_type": {
                kind: sum(table[kind, layer] for layer in layers) for kind in sorted(mix)},
            "train": score(train, mix, table), "evaluation": score(test, mix, table),
        }
    return {
        "schema": "hisparse-fixed-mix-global-budget-replay-v1", "mix": mix,
        "layers": layers, "capacity_grid": list(capacities),
        "logical_active_slots_per_rank": baseline * len(layers) * sum(mix.values()),
        "budget_scope": "one fixed concurrent request mix; requests may have unequal total capacities",
        "objective": "mix-weighted mean per-layer training replay miss probability; evaluation never fits",
        "evaluation_provenance": "caller_supplied_unverified",
        "policies": policies,
        "limitations": [
            "Offline fixed-mix replay, not an online mixed-request admission or capacity-resize implementation.",
            "Logical active slots do not prove physical HBM reclamation or capacity gains.",
            "Historical selected-token curves do not model changed outputs, queueing, or runtime scan cost.",
            "Changing the live request mix requires a new feasible allocation and safe cache migration.",
            "Cold-LRU short-context simulations are not the current native full-resident short path.",
            "Physical TP ranks may replicate KV heads and are not independent statistical samples.",
        ],
    }


def read_curves(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    curves = {kind: {int(layer): {int(k) if k != "selected" else k: v for k, v in row.items()}
                     for layer, row in table.items()} for kind, table in data.items()}
    return curves, hashlib.sha256(raw).hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def origin_set(values):
    require(isinstance(values, list) and bool(values) and all(map(is_sha, values))
            and len(set(values)) == len(values), "Invalid stable origin hash collection")
    return set(values)


def partition_roles(partition):
    require(partition.get("schema") == PARTITION_SCHEMA and is_sha(partition.get("source_partition_sha256")),
            "Invalid anonymous fixed partition")
    roles = partition["roles"]
    other = set(roles) - {"train"}
    require("train" in roles and len(other) == 1 and other <= EVALUATION_ROLES,
            "Partition needs train and one explicit evaluation role")
    evaluation = next(iter(other))
    train, test = roles["train"], roles[evaluation]
    require(
        is_sha(train.get("dataset_sha256")) and is_sha(test.get("dataset_sha256"))
        and train["dataset_sha256"] != test["dataset_sha256"],
        "Partition requires distinct fixed dataset hashes",
    )
    require(not origin_set(train["origin_sha256"]) & origin_set(test["origin_sha256"]),
            "Train and evaluation origins overlap")
    return roles, evaluation


def checked_scope(scope):
    require(
        set(scope) == {"row_scope", "top_k", "common_hmax", "layers", "ranks", "capacity_grid", "warmup_steps"}
        and scope["row_scope"] in ROW_SCOPES and type(scope["top_k"]) is int and scope["top_k"] == 2048
        and type(scope["common_hmax"]) is int and scope["common_hmax"] >= 4096
        and scope["common_hmax"] % 128 == 0
        and type(scope["warmup_steps"]) is int and scope["warmup_steps"] == 16,
        "Invalid curve row scope, Top-K, common Hmax, or warmup",
    )
    for field, limit in (("layers", 60), ("ranks", 8)):
        values = scope[field]
        require(
            isinstance(values, list) and values and all(type(v) is int and 0 <= v < limit for v in values)
            and sorted(set(values)) == values,
            f"Invalid {field} domain",
        )
    grid = scope["capacity_grid"]
    require(
        isinstance(grid, list) and 4096 in grid
        and all(type(c) is int and 2048 <= c <= min(32767, scope["common_hmax"]) and c % 128 == 0 for c in grid)
        and sorted(set(grid)) == grid,
        "Invalid common-Hmax capacity grid",
    )
    return scope


def verify_curve_receipts(partition, partition_sha, artifacts):
    """Bind both supplied curve files to a pre-authorized anonymous partition."""
    require(is_sha(partition_sha), "Invalid partition digest")
    roles, evaluation = partition_roles(partition)
    require(set(artifacts) == set(roles), "Missing or mislabelled fixed-partition arm")
    scopes, seals = {}, {}
    for role, (curves, curves_sha, receipt) in artifacts.items():
        expected = roles[role]
        require(
            receipt.get("schema") == CURVE_SCHEMA and receipt.get("role") == role
            and receipt.get("partition_sha256") == partition_sha
            and receipt.get("source_partition_sha256") == partition["source_partition_sha256"]
            and receipt.get("dataset_sha256") == expected["dataset_sha256"]
            and is_sha(curves_sha) and receipt.get("curves_sha256") == curves_sha
            and is_sha(receipt.get("capture_seal_sha256")) and is_sha(receipt.get("replay_summary_sha256")),
            f"Curve receipt does not bind the {role} inputs",
        )
        require(origin_set(receipt["origin_sha256"]) == origin_set(expected["origin_sha256"]),
                f"Stable origins differ for {role}")
        scopes[role] = checked_scope(receipt["scope"])
        seals[role] = receipt["capture_seal_sha256"]
        require(
            curves and all(sorted(table) == scopes[role]["layers"] for table in curves.values()),
            f"Curve layers differ from the {role} receipt",
        )
    require(scopes["train"] == scopes[evaluation], "Train/evaluation row or geometry scope differs")
    require(seals["train"] != seals[evaluation], "Train/evaluation must use distinct capture seals")
    return evaluation, scopes["train"]


def encode_json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def export_verified_tp8(
    summary_bytes, summary_sha, partition_bytes, partition_sha, *, role, alias_origin_sha256, scope,
    capture_seal_sha256
):
    """Create anonymous curves/receipt after the private capture and split were verified.

    The upstream verifier must pin the exact replay summary bytes and the sealed
    capture's results.json, and project the true source hashes onto its aliases.
    Local replay aliases are never used as globally unique origin identities.
    """
    summary = verify_json_bytes(summary_bytes, summary_sha)
    partition = verify_json_bytes(partition_bytes, partition_sha)
    roles, _ = partition_roles(partition)
    scope = checked_scope(scope)
    require(
        role in roles and is_sha(partition_sha) and is_sha(summary_sha) and is_sha(capture_seal_sha256)
        and summary.get("coverage") == "complete" and summary.get("warmup_steps") == scope["warmup_steps"]
        and scope["layers"] == list(range(60)) and scope["ranks"] == list(range(8))
        and scope["common_hmax"] == 8192
        and summary.get("dataset_sha256") == roles[role]["dataset_sha256"]
        and summary.get("capture_evidence", {}).get("results_sha256") == capture_seal_sha256,
        "TP8 replay does not match the verified capture or fixed-partition role",
    )
    requests = {row["request"]: row for row in summary["requests"]}
    require(len(requests) == len(summary["requests"]) and set(alias_origin_sha256) == set(requests)
            and set(alias_origin_sha256.values()) == origin_set(roles[role]["origin_sha256"]),
            "Anonymous source projection does not match the fixed partition")
    projected = {(row["origin_group"], alias_origin_sha256[alias]) for alias, row in requests.items()}
    outcomes = Counter(row["finish_reason"] for row in requests.values())
    require(
        len(projected) == len({group for group, _ in projected}) == len({origin for _, origin in projected})
        and outcomes.keys() <= {"stop", "length"}
        and outcomes == summary["capture_evidence"]["finish_reason_counts"]
        and all(type(row["output_tokens"]) is int and row["output_tokens"] >= 1 for row in requests.values()),
        "Invalid origin projection or outcome receipt",
    )
    grid = scope["capacity_grid"]
    prefix = "strict_long_" if scope["row_scope"] == "strict_long" else ""
    curves, seen = {}, set()
    for row in summary["rank_request_layer_after_first_16"]:
        alias, layer, rank = row["request"], row["layer"], row["rank"]
        request = requests.get(alias)
        key = alias, layer, rank
        require(
            request is not None and request["output_tokens"] > 1
            and type(layer) is int and layer in scope["layers"] and type(rank) is int and rank in scope["ranks"]
            and row["workload"] == request["workload"] and key not in seen,
            "Invalid, missing, or duplicate TP8 replay stream",
        )
        seen.add(key)
        if request["finish_reason"] != "stop":
            continue
        selected = row[prefix + "selected"]
        misses = {cap: row[f"{prefix}simulated_misses_{cap}"] for cap in grid}
        require(
            type(selected) is int and selected >= 0
            and all(type(count) is int and 0 <= count <= selected for count in misses.values()),
            "Invalid TP8 replay counts",
        )
        table = curves.setdefault(request["workload"], {})
        curve = table.setdefault(str(layer), {"selected": 0, **{str(cap): 0 for cap in grid}})
        curve["selected"] += selected
        for cap, count in misses.items():
            curve[str(cap)] += count
    expected_count = len(scope["layers"]) * len(scope["ranks"])
    require(len(seen) == expected_count * sum(r["output_tokens"] > 1 for r in requests.values()),
            "Incomplete TP8 request/layer/rank replay")
    payload = encode_json(curves)
    receipt = {
        "schema": CURVE_SCHEMA, "role": role, "partition_sha256": partition_sha,
        "source_partition_sha256": partition["source_partition_sha256"],
        "dataset_sha256": summary["dataset_sha256"], "origin_sha256": sorted(set(alias_origin_sha256.values())),
        "curves_sha256": hashlib.sha256(payload).hexdigest(), "scope": scope,
        "capture_seal_sha256": capture_seal_sha256, "replay_summary_sha256": summary_sha,
        "finish_reason_counts": outcomes,
    }
    return payload, encode_json(receipt)


def verify_json_bytes(raw, expected_sha):
    require(is_sha(expected_sha) and hashlib.sha256(raw).hexdigest() == expected_sha,
            "Metadata receipt SHA does not match its trusted pin")
    return json.loads(raw)


def trusted_json(path, expected_sha):
    return verify_json_bytes(path.read_bytes(), expected_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mix", required=True, help='JSON positive active counts, e.g. {"chatbot":2,"agent":1}')
    for name in ("partition", "train-receipt", "evaluation-receipt"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    args = parser.parse_args()
    train, train_sha = read_curves(args.train)
    evaluation, evaluation_sha = read_curves(args.evaluation)
    partition_sha, train_receipt_sha, evaluation_receipt_sha = (
        args.partition_sha256, args.train_receipt_sha256, args.evaluation_receipt_sha256
    )
    partition = trusted_json(args.partition, partition_sha)
    train_receipt = trusted_json(args.train_receipt, train_receipt_sha)
    evaluation_receipt = trusted_json(args.evaluation_receipt, evaluation_receipt_sha)
    role = evaluation_receipt["role"]
    require(role in EVALUATION_ROLES, "Evaluation receipt must declare its evaluation role")
    role, scope = verify_curve_receipts(partition, partition_sha, {
        "train": (train, train_sha, train_receipt),
        role: (evaluation, evaluation_sha, evaluation_receipt),
    })
    mix = json.loads(args.mix)
    require(isinstance(mix, dict) and set(mix) <= set(train) and set(mix) <= set(evaluation),
            "Selected type missing from a verified arm")
    result = compare({kind: train[kind] for kind in mix}, {kind: evaluation[kind] for kind in mix},
                     mix, tuple(scope["capacity_grid"]))
    result.update(training_sha256=train_sha, evaluation_sha256=evaluation_sha,
                  evaluation_provenance=role, row_scope=scope["row_scope"], scope=scope, gpu_measured=False,
                  partition_sha256=partition_sha, train_receipt_sha256=train_receipt_sha,
                  evaluation_receipt_sha256=evaluation_receipt_sha,
                  capture_seal_sha256={"train": train_receipt["capture_seal_sha256"],
                                       role: evaluation_receipt["capture_seal_sha256"]})
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
