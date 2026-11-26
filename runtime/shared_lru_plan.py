"""Compile per-arm study configs and a paired measurement plan; never launches anything.

    python -B -m runtime.shared_lru_plan --spec runtime/fixtures/shared_lru_plan_spec.json \
        --output-dir /tmp/shared-lru-plan [--replay-results stock.json shared.json ...]

Each arm is one hisparse-study-config-v1 file. Every pair must differ from its
control in exactly the declared fields. The plan records seed, split digests,
geometry, the real-B requirement, the equal-byte ledger and the hardware stages
still pending. Exit status is nonzero for invalid specs or configs, unexpected
config differences, an unbalanced byte ledger, or replay results whose
selection digests differ. A compiled plan is not a resource grant or a result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from . import shared_lru as reference
from . import shared_lru_device as device
from . import study_config as sc

SPEC_SCHEMA = "shared-lru-plan-spec-v1"
PLAN_SCHEMA = "shared-lru-paired-plan-v1"
SHARED = {"mode": "shared", "layout": "arena"}
SWITCHES = {"owner_summary": True, "touched_bitmap": True}
INDEXED = {"hit_index": True, "incremental_lru": True, "compact_owner": True}   # issue #6
# (arm, hot_pool fields, features)
ARMS = (
    ("stock", {}, {}),
    ("isolated_arena", {"layout": "arena"}, {}),
    ("isolated_static", {"layout": "arena", "quota": "static", "quota_rows": "static_quota_rows"}, {}),
    ("shared_lru", SHARED, {}),
    ("shared_fifo", dict(SHARED, eviction="fifo"), {}),
    ("shared_soft_floor", dict(SHARED, quota="soft_floor", floor_rows="floor_rows"), {}),
    ("shared_bypass", dict(SHARED, admission="bypass_list", bypass_layers="bypass_layers"), {}),
    ("shared_lru_owner_summary", SHARED, {"owner_summary": True}),
    ("shared_lru_touched_bitmap", SHARED, {"touched_bitmap": True}),
    ("shared_lru_optimized", SHARED, SWITCHES),
    ("isolated_arena_optimized", {"layout": "arena"}, SWITCHES),
    ("shared_lru_hit_index", SHARED, {"hit_index": True}),
    ("shared_lru_incremental_lru", SHARED, {"incremental_lru": True}),
    ("shared_lru_compact_owner", SHARED, {"compact_owner": True}),
    ("shared_lru_indexed", SHARED, INDEXED),
    ("isolated_arena_indexed", {"layout": "arena"}, INDEXED),
)
BOTH = ("features.owner_summary", "features.touched_bitmap")
ALL_THREE = ("features.compact_owner", "features.hit_index", "features.incremental_lru")
# (control, candidate, fields expected to differ). Each resolver switch is measured alone and then
# combined at a fixed algorithm; algorithms are then compared at the same engineering level.
PAIRS = (
    ("stock", "isolated_arena", ("hot_pool.layout",)),
    ("isolated_arena", "isolated_static", ("hot_pool.quota", "hot_pool.quota_rows")),
    ("isolated_arena", "shared_lru", ("hot_pool.mode",)),
    ("shared_lru", "shared_fifo", ("hot_pool.eviction",)),
    ("shared_lru", "shared_soft_floor", ("hot_pool.floor_rows", "hot_pool.quota")),
    ("shared_lru", "shared_bypass", ("hot_pool.admission", "hot_pool.bypass_layers")),
    ("shared_lru", "shared_lru_owner_summary", ("features.owner_summary",)),
    ("shared_lru", "shared_lru_touched_bitmap", ("features.touched_bitmap",)),
    ("shared_lru", "shared_lru_optimized", BOTH),
    ("isolated_arena", "isolated_arena_optimized", BOTH),
    ("isolated_arena_optimized", "shared_lru_optimized", ("hot_pool.mode",)),
    ("stock", "shared_lru_optimized", BOTH + ("hot_pool.layout", "hot_pool.mode")),
    ("shared_lru", "shared_lru_hit_index", ("features.hit_index",)),
    ("shared_lru", "shared_lru_incremental_lru", ("features.incremental_lru",)),
    ("shared_lru", "shared_lru_compact_owner", ("features.compact_owner",)),
    ("shared_lru", "shared_lru_indexed", ALL_THREE),
    ("isolated_arena", "isolated_arena_indexed", ALL_THREE),
    ("isolated_arena_indexed", "shared_lru_indexed", ("hot_pool.mode",)),
    ("stock", "shared_lru_indexed", ALL_THREE + ("hot_pool.layout", "hot_pool.mode")),
)
SPEC_KEYS = {"schema", "seed", "split", "geometry", "static_quota_rows", "floor_rows", "bypass_layers",
             "pressure_points", "repeats", "upstream_args", "sm80_bf16_projections"}
PENDING = ("sm80_jit_build", "cuda_graph_capture_replay", "real_batch_from_server_info", "physical_memory_receipt",
           "serving_tpot_throughput")


def _fields(config):
    out = {}
    for section, value in config.as_dict().items():
        if type(value) is dict:
            out.update({f"{section}.{field}": item for field, item in value.items()})
        else:
            out[section] = value
    return out


def changed_fields(control, candidate):
    a, b = _fields(control), _fields(candidate)
    return tuple(sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k)))


def read_spec(path):
    raw = Path(path).read_bytes()
    spec = sc.loads(raw)
    if set(spec) != SPEC_KEYS or spec["schema"] != SPEC_SCHEMA:
        raise ValueError(f"Spec must be {SPEC_SCHEMA} with fields {sorted(SPEC_KEYS)}")
    if type(spec["seed"]) is not int or type(spec["repeats"]) is not int or spec["repeats"] < 3:
        raise ValueError("Spec needs an integer seed and at least three AB/BA repeats per pressure point")
    split = spec["split"]
    if type(split) is not dict or set(split) != {"training_sha256", "heldout_sha256"}:
        raise ValueError("split needs training_sha256 and heldout_sha256 digests of the private manifests")
    for value in split.values():
        sc._digest(value, "split digest")
    if split["training_sha256"] == split["heldout_sha256"] != "0" * 64:
        raise ValueError("Training and held-out manifests must differ")
    points = spec["pressure_points"]
    if type(points) is not list or not points or any(
            type(p) is not dict or set(p) != {"name", "input_tokens", "real_batch"} or type(p["real_batch"]) is not int
            or p["real_batch"] < 2 for p in points):
        raise ValueError("pressure_points need name, input_tokens and an integer real_batch >= 2")
    if type(spec["upstream_args"]) is not list or any(type(a) is not str for a in spec["upstream_args"]):
        raise ValueError("upstream_args must be the SGLang argv shared by every arm")
    return spec, hashlib.sha256(raw).hexdigest()


def arm_configs(spec):
    """Study config per arm; arms whose frozen inputs are absent are skipped with a reason."""
    configs, skipped = {}, {}
    for name, fields, features in ARMS:
        pool = {}
        for key, value in fields.items():
            if value in ("static_quota_rows", "floor_rows", "bypass_layers"):
                if spec[value] is None:
                    skipped[name] = f"spec.{value} is null (freeze it from training sources first)"
                    break
                value = spec[value]
            pool[key] = value
        else:
            data = {"schema": sc.SCHEMA}
            if pool:
                data["hot_pool"] = pool
            if features:
                data["features"] = dict(features)
            configs[name] = sc.parse(data)
    return configs, skipped
class PlanError(ValueError):
    pass


def _check_launch(argv):
    from . import launch
    try:
        launch.resolve(argv)
    except SystemExit as error:  # argparse reports through SystemExit
        raise PlanError([f"launcher rejected {argv[1]}: exit {error.code}"]) from None


def compile_plan(spec, spec_digest, output_dir, replay_results=()):
    g = spec["geometry"]
    geo = device.describe_geometry(
        tp_size=g["tp_size"], layers=range(g["first_layer"], g["first_layer"] + g["layers"]), hot_rows=g["hot_rows"],
        page=g["page"], top_k_blocks=g["top_k_blocks"], block_size=g["block_size"], head_dim=g["head_dim"],
        dtype_bytes=g["dtype_bytes"], kv_heads_per_rank=g["kv_heads_per_rank"], slots=g["slots"],
        plane_rows=g["plane_rows"])
    configs, skipped = arm_configs(spec)
    arms, errors = {}, []
    for name, _, _ in ARMS:
        if name not in configs:
            continue
        config = configs[name]
        plan = (reference.plan_arena(config, tuple(geo["layers"]), geo["hot_rows"], geo["selected"])
                if config.uses_arena else None)
        ledger = reference.byte_ledger(layers=geo["layers"], hot_rows=geo["hot_rows"], page=geo["page"],
                                       item_bytes=geo["item_size_bytes"], slots=geo["slots"],
                                       pool_rows=geo["plane_rows"], plan=plan,
                                       features=[feature for feature, _ in config.features])
        if plan is not None:
            device.check_shared_memory(config, plan)
        arms[name] = dict(config=config, plan=plan, ledger=ledger)
    stock = arms["stock"]["ledger"]
    for name, arm in arms.items():  # equal-byte contract: same main KV, same history rows per request
        if (arm["ledger"]["main_kv_bytes"], arm["ledger"]["per_request"]["history_rows"]) != (
                stock["main_kv_bytes"], stock["per_request"]["history_rows"]):
            errors.append(f"{name}: main-KV bytes or history rows differ from stock")
        plan = arm["plan"]
        if plan is not None and sum(size for _, size in plan.domains) + plan.scratch[1] != plan.rows:
            errors.append(f"{name}: domains and scratch do not partition the arena")
    pairs = []
    for control, candidate, expected in PAIRS:
        if candidate not in arms:
            continue                     # the arm itself is listed in skipped with its reason
        if control not in arms:
            skipped[f"{control}->{candidate}"] = f"control {control} was skipped"
            continue
        actual = changed_fields(arms[control]["config"], arms[candidate]["config"])
        if actual != tuple(sorted(expected)):
            errors.append(f"{candidate} vs {control}: changed {list(actual)}, expected {sorted(expected)}")
        pairs.append(dict(control=control, candidate=candidate, changed_fields=list(actual)))
    digests = {json.loads(Path(p).read_text())["selection_sha256"] for p in replay_results}
    if len(digests) > 1:
        errors.append(f"replay results used different selections: {sorted(digests)}")
    if errors:
        raise PlanError(errors)
    per_request = stock["per_request"]["history_payload_bytes"] + stock["per_request"]["tail_payload_bytes"]
    padded = geo["hot_rows"] + geo["page"]
    pool_requests = (geo["plane_rows"] - geo["page"]) // padded      # padded buffers per rank, prefill aside
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "plan.json"
    targets = [output_dir / f"{name}.json" for name in arms] + [plan_path]
    existing = [str(p) for p in targets if p.exists()]
    if existing:
        raise PlanError([f"refusing to overwrite {existing}"])
    launches, summary = {}, []
    for name, arm in arms.items():
        path = output_dir / f"{name}.json"
        path.write_text(json.dumps(arm["config"].as_dict(), indent=2, sort_keys=True) + "\n")
        argv = ["--hisparse-study-config", str(path)] + (["--sm80-bf16-projections"]
                                                          if spec["sm80_bf16_projections"] else [])
        argv += ["--", *spec["upstream_args"]]
        _check_launch(argv)
        launches[name] = ["python", "-m", "runtime.launch", *argv]
        extra = arm["ledger"]["arena_metadata_bytes"]
        lost = -(-extra // per_request) if extra else 0
        summary.append(dict(
            arm=name, config=path.name, config_sha256=arm["config"].digest,
            file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            capabilities=sc.capabilities(arm["config"]), ledger=arm["ledger"],
            capacity=dict(stock_padded_buffers_per_rank=pool_requests, arena_metadata_bytes=extra,
                          requests_charged_for_metadata=lost, requests_per_rank_after_charge=pool_requests - lost),
            feasible={p["name"]: p["real_batch"] <= pool_requests - lost for p in spec["pressure_points"]}))
    schedule = []
    for point in spec["pressure_points"]:
        for pair in pairs:
            for repeat in range(spec["repeats"]):
                order = [pair["control"], pair["candidate"]][::1 if repeat % 2 == 0 else -1]
                schedule.append(dict(point=point["name"], pair=f"{pair['control']}->{pair['candidate']}",
                                     repeat=repeat, order=order, real_batch=point["real_batch"],
                                     input_tokens=point["input_tokens"]))
    placeholder = "0" * 64 in spec["split"].values()
    plan = dict(schema=PLAN_SCHEMA, spec_sha256=spec_digest, seed=spec["seed"], split=spec["split"], geometry=geo,
                split_placeholder=placeholder,
                arms=summary, skipped=skipped, pairs=pairs, schedule=schedule, launch=launches,
                real_batch_requirement="each run must report the replayed decode batch from server info and Graph "
                                       "receipts; a run below the planned real_batch is rejected, not rescaled",
                metrics=["tpot_ms_p50", "tpot_ms_p95", "tpot_ms_p99", "decode_tokens_per_s", "slo_goodput",
                         "native_repeat_noise"],
                cuda_graph_verification="pending_hardware",
                pending_hardware=list(PENDING) + (["replace placeholder split digests"] if placeholder else []),
                replay_selection_sha256=sorted(digests))
    with plan_path.open("x") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compile shared-arena arms and a paired measurement plan.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-results", type=Path, nargs="*", default=(),
                        help="CPU replay outputs whose selection digests must agree")
    args = parser.parse_args(argv)
    try:
        spec, digest = read_spec(args.spec)
        plan = compile_plan(spec, digest, args.output_dir, args.replay_results)
    except PlanError as error:
        print(json.dumps(dict(status="fail", errors=error.args[0])), file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(status="fail", errors=[str(error)])), file=sys.stderr)
        return 1
    print(json.dumps(dict(status="pass", arms=[a["arm"] for a in plan["arms"]], pairs=len(plan["pairs"]),
                          runs=len(plan["schedule"]), skipped=plan["skipped"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
