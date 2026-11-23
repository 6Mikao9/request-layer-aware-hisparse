"""Freeze a capacity-only A/B contract; never launch a model or mutate an input."""

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "hisparse-controlled-layer-comparison-v1"
LAYERS = 60
SAMPLE = (5, 15, 25, 35, 45, 55)
SLOTS = LAYERS * 4096
REQUIRED_FIXED = {
    "image_id", "model_config_sha256", "runtime_sources_sha256", "workload_sha256",
    "warmup_workload_sha256", "device_uuids", "tp_size", "precision", "top_k",
    "physical_hmax", "pool_capacity", "host_ratio", "context_length",
    "client_concurrency", "max_running_requests", "graph_config", "overlap_schedule",
    "radix_cache", "generation_settings", "cache_seed_sha256", "kernel_variant",
    "split_copy_config", "speculative_decode", "layer_policy_adapter",
}


def digest_object(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_fixed(value):
    if not isinstance(value, dict) or set(value) != REQUIRED_FIXED:
        raise ValueError("Fixed contract fields must be exact; missing/extra fields cannot be silently ignored")
    if value["tp_size"] not in (4, 8) or type(value["tp_size"]) is not int:
        raise ValueError("Expected verified TP4/TP8 configuration")
    if (len(value["device_uuids"]) != value["tp_size"]
            or len(set(value["device_uuids"])) != value["tp_size"]):
        raise ValueError("Device identities must be unique and cover all TP ranks")
    if value["kernel_variant"] != "fused" or value["split_copy_config"] is not None:
        raise ValueError("Capacity-only comparison excludes split-copy and unrelated kernel changes")
    if value["top_k"] != 2048 or value["speculative_decode"] is not False:
        raise ValueError("Keep the native Top-K=2048 and disable speculative decoding")
    if value["layer_policy_adapter"] != "static-active-prefix":
        raise ValueError("Both arms must use the same active-prefix policy adapter")
    if not 1 <= value["client_concurrency"] <= value["max_running_requests"]:
        raise ValueError("Invalid client concurrency envelope")
    if not 2048 <= value["physical_hmax"] <= 32767:
        raise ValueError("Invalid common Hmax")
    for key in ("model_config_sha256", "runtime_sources_sha256", "workload_sha256",
                "warmup_workload_sha256", "cache_seed_sha256"):
        sha = value[key]
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("Missing content binding: " + key)
    return value


def validate_pair(a, b, *, scope):
    if scope not in ("six_layers", "all_layers"):
        raise ValueError("Declare six_layers or all_layers before evaluating results")
    for arm in (a, b):
        if set(arm) != {"fixed", "active_capacity_by_layer"}:
            raise ValueError("An arm contains an unreviewed control field")
        validate_fixed(arm["fixed"])
        cap = arm["active_capacity_by_layer"]
        if (not isinstance(cap, list) or len(cap) != LAYERS
                or any(type(c) is not int or c % 128 or not 2048 <= c <= arm["fixed"]["physical_hmax"] for c in cap)
                or sum(cap) != SLOTS):
            raise ValueError("Each arm must retain exactly 60x4096 active slots per request and TP rank")
    if a["fixed"] != b["fixed"]:
        differences = sorted(k for k in REQUIRED_FIXED if a["fixed"][k] != b["fixed"][k])
        raise ValueError("Confounded comparison: " + ", ".join(differences))
    if a["active_capacity_by_layer"] != [4096] * LAYERS:
        raise ValueError("Reference must be uniform-active4096 under the same adapter")
    changed = [i for i in range(LAYERS) if a["active_capacity_by_layer"][i] != b["active_capacity_by_layer"][i]]
    if not changed or (scope == "six_layers" and any(i not in SAMPLE for i in changed)):
        raise ValueError("Capacity changes must match the prespecified layer scope")
    return {
        "schema": SCHEMA, "scope": scope, "fixed_runtime_sha256": digest_object(a["fixed"]),
        "baseline_sha256": digest_object(a), "candidate_sha256": digest_object(b),
        "changed_layers": changed, "active_slots_per_request_per_rank": SLOTS,
        "kernel_variant": "fused", "timing_measured": False, "physical_reclamation_measured": False,
        "primary_scope": "Natural EOS full-request latency; additionally report actual context > common Hmax",
        "profile_scope": "Independent diagnostic capture; paired step-state timing requires an explicit trajectory binding",
        "limitations": [
            "A frozen launch contract is not evidence that runtime configuration or GPU kernels matched it.",
            "Active capacity equality does not prove physical HBM savings or higher admission capacity.",
            "Free generation can diverge; preserve original budgets, EOS/length stops and actual batch distributions.",
            "Use interleaved repeated A/B and B/A runs; profile-on timings do not replace uninstrumented latency.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scope", choices=("six_layers", "all_layers"), required=True)
    args = parser.parse_args(argv)
    result = validate_pair(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()), scope=args.scope)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
