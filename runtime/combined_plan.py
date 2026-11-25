"""Compile arms A–E of the combined study and their paired, prioritized plan; never launches anything.

    python -B -m runtime.combined_plan --spec runtime/fixtures/combined_plan_spec.example.json \
        --output-dir /tmp/combined-plan

Arms (docs/implementation/combined.md, "固定物理预算的稳态比较"):

* A ``stock``: native all-layer HiSparse.
* B ``all_isolated_arena`` / ``all_shared_arena``: the same addressing with isolated
  quotas, then all-layer sharing.
* Per frozen offload list L (three or more random seeds, the training-calibrated list and
  its high-cost reverse, frozen by ``runtime.combined_calibration``), all with
  ``features.resident_direct``: C ``L_stock_layout`` (per-layer offload alone, stock
  kernel) and ``L_isolated_arena`` (same addressing as E, borrowing off); D
  ``L_static_quota`` (training-frozen static quota); E ``L_shared_arena`` (borrowing among
  S only). Lists in ``resident_direct_lists`` add ``L_shared_resident_resolver``: E with
  the resident resolver instead of the direct read.

Pairs change exactly their declared fields, except acceptance pairs, which compare the
preregistered candidate with each declared baseline at the one HBM budget. E–C and E–D
separate the sharing gain; B1→C2 and B2→E isolate the placement effect with borrowing
off and on, so the four arena arms form the 2x2 placement x borrowing interaction;
stock→stock records native repeat noise. Capacity per arm is re-derived at the stock
HBM budget (resident planes cost logical rows); a pair runs at a pressure point only
when both arms reach its real batch. The schedule lists AB/BA rounds in priority order
(noise, acceptance, mechanisms, selection, resident path); round r of every pair reads
input batch r. Placeholders are reported, not filled. A compiled plan is not a resource
grant or a result; ``runtime.combined_run`` executes it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from . import layer_placement as lp
from . import layer_placement_device as placement_device
from . import shared_lru as s
from . import shared_lru_device as arena
from . import study_config as sc
from .launch import flag_value
from .shared_lru_plan import PlanError, _check_launch, changed_fields

SPEC_SCHEMA = "combined-plan-spec-v1"
PLAN_SCHEMA = "combined-paired-plan-v1"
SPEC_KEYS = {"schema", "seed", "split", "geometry", "offload_lists", "static_quota_rows", "resident_direct_lists",
             "pressure_points", "repeats", "measurement", "acceptance", "upstream_args", "sm80_bf16_projections"}
GEOMETRY_KEYS = {"tp_size", "first_layer", "layers", "dense_layers", "hot_rows", "page", "top_k_blocks",
                 "block_size", "head_dim", "dtype_bytes", "kv_heads_per_rank", "slots", "plane_rows",
                 "host_to_device_ratio", "index_bytes_per_token"}
MEASUREMENT_KEYS = {"max_new_tokens", "warmup_requests", "warmup_steps", "min_window_steps", "min_decode_log_samples",
                    "functional_requests", "functional_max_new_tokens", "ttft_slo_s", "tpot_slo_s",
                    "server_start_timeout_s", "request_timeout_s"}
ACCEPTANCE_KEYS = {"candidate", "baselines", "min_gain", "tpot_p95_regression_max", "ci_level", "min_pressure_points",
                   "min_pairs", "quality_mismatch_over_stock_max"}
LIST_KEYS = {"name", "selection", "offload_layers", "seed", "calibration"}
ISOLATED = {"mode": "isolated", "layout": "arena"}
SHARED = {"mode": "shared", "layout": "arena"}
DIRECT = {"resident_direct": True}
PRIORITY = ("aa_noise", "acceptance", "sharing", "sharing_vs_static", "static_quota", "placement", "addressing",
            "placement_borrowing_off", "placement_borrowing_on", "selection_isolated_arena", "selection_shared_arena",
            "resident_direct")
PENDING = ("sm80_jit_build", "cuda_graph_capture_replay", "split_pool_receipt_per_rank",
           "real_batch_from_server_info", "physical_memory_receipt", "serving_tpot_throughput", "nsys_rank_timeline")


def _int(value, where, low=0):
    if type(value) is not int or value < low:
        raise ValueError(f"{where} must be an integer >= {low}")
    return value


def _number(value, where, low=0.0):
    if type(value) not in (int, float) or not value > low:
        raise ValueError(f"{where} must be a number > {low}")
    return value


def _measurement(m, upstream):
    """Preregistered window, warmup, probe and SLO settings shared by every run."""
    if type(m) is not dict or set(m) != MEASUREMENT_KEYS:
        raise ValueError(f"measurement needs exactly {sorted(MEASUREMENT_KEYS)}")
    for key in ("max_new_tokens", "warmup_steps", "min_window_steps", "min_decode_log_samples", "functional_requests",
                "functional_max_new_tokens"):
        _int(m[key], f"measurement.{key}", 1)
    _int(m["warmup_requests"], "measurement.warmup_requests")
    for key in ("ttft_slo_s", "tpot_slo_s", "server_start_timeout_s", "request_timeout_s"):
        _number(m[key], f"measurement.{key}")
    if m["max_new_tokens"] <= m["warmup_steps"] + m["min_window_steps"]:
        raise ValueError("measurement.max_new_tokens must exceed warmup_steps + min_window_steps")
    interval = flag_value(upstream, "--decode-log-interval")
    if (interval is None or not interval.isdigit() or int(interval) < 1
            or int(interval) * m["min_decode_log_samples"] > m["min_window_steps"]):
        raise ValueError("upstream_args need --decode-log-interval N with N * min_decode_log_samples <= "
                         "min_window_steps: the decode log is the real-batch and Graph evidence of every window")


def _acceptance(a, points, repeats):
    """Primary claim: the candidate against every declared baseline (names checked once arms exist)."""
    if type(a) is not dict or set(a) != ACCEPTANCE_KEYS:
        raise ValueError(f"acceptance needs exactly {sorted(ACCEPTANCE_KEYS)}")
    baselines = a["baselines"]
    if (type(a["candidate"]) is not str or type(baselines) is not list or not baselines
            or any(type(b) is not str for b in baselines) or len(set(baselines)) != len(baselines)
            or a["candidate"] in baselines):
        raise ValueError("acceptance names one candidate arm and distinct baseline arms")
    _number(a["min_gain"], "acceptance.min_gain")
    if type(a["tpot_p95_regression_max"]) not in (int, float) or a["tpot_p95_regression_max"] < 0:
        raise ValueError("acceptance.tpot_p95_regression_max must be a number >= 0")
    if a["ci_level"] != 0.95:
        raise ValueError("acceptance.ci_level: this build computes two-sided 95% t intervals only")
    if not 2 <= _int(a["min_pressure_points"], "acceptance.min_pressure_points", 2) <= len(points):
        raise ValueError("acceptance.min_pressure_points must be at least two and at most the pressure points")
    if not 3 <= _int(a["min_pairs"], "acceptance.min_pairs", 3) <= repeats:
        raise ValueError("acceptance.min_pairs must be at least three and at most repeats")
    quality = a["quality_mismatch_over_stock_max"]
    if type(quality) not in (int, float) or not 0 <= quality <= 1:
        raise ValueError("acceptance.quality_mismatch_over_stock_max must be within [0, 1]")


def read_spec(path):
    raw = Path(path).read_bytes()
    spec = sc.loads(raw)
    if set(spec) != SPEC_KEYS or spec["schema"] != SPEC_SCHEMA:
        raise ValueError(f"Spec must be {SPEC_SCHEMA} with fields {sorted(SPEC_KEYS)}")
    _int(spec["seed"], "seed")
    _int(spec["repeats"], "repeats (AB/BA pairs per pressure point)", 3)
    split = spec["split"]
    if type(split) is not dict or set(split) != {"training_sha256", "heldout_sha256"}:
        raise ValueError("split needs training_sha256 and heldout_sha256 digests of the private manifests")
    for value in split.values():
        sc._digest(value, "split digest")
    if split["training_sha256"] == split["heldout_sha256"] != "0" * 64:
        raise ValueError("Training and held-out manifests must differ")
    geometry = spec["geometry"]
    if type(geometry) is not dict or set(geometry) != GEOMETRY_KEYS:
        raise ValueError(f"geometry needs exactly {sorted(GEOMETRY_KEYS)}")
    for key, value in geometry.items():
        _int(value, f"geometry.{key}", 0 if key in ("first_layer", "dense_layers", "index_bytes_per_token") else 1)
    lists, names = spec["offload_lists"], set()
    if type(lists) is not list or not lists:
        raise ValueError("offload_lists must name at least one frozen offload set")
    for item in lists:
        if (type(item) is not dict or not {"name", "selection", "offload_layers"} <= set(item) <= LIST_KEYS
                or type(item["name"]) is not str or not item["name"].isidentifier() or item["name"] in names):
            raise ValueError(f"Offload list {item!r} needs a unique identifier name, selection and offload_layers")
        names.add(item["name"])
        sc.parse({"schema": sc.SCHEMA, "placement": dict({k: v for k, v in item.items() if k != "name"},
                                                         mode="fixed_layers")})
        if not item["offload_layers"]:
            raise ValueError(f"Offload list {item['name']} is empty; the all-resident control is not a HiSparse arm")
    calibrated = [i for i in lists if i["selection"] == "calibrated"]
    for item in calibrated:
        T = len(item["offload_layers"])
        if sum(i["selection"] == "random" and len(i["offload_layers"]) == T for i in lists) < 3:
            raise ValueError(f"Calibrated list {item['name']} needs at least three random lists of the same T={T}")
        if item["calibration"]["training_sha256"] != split["training_sha256"]:
            raise ValueError(f"Calibrated list {item['name']} was frozen from another source than "
                             "split.training_sha256; calibration reads the training split only")
    quotas = spec["static_quota_rows"]
    if quotas is not None and (type(quotas) is not dict or set(quotas) - names):
        raise ValueError("static_quota_rows must be null or keyed by offload list name")
    direct = spec["resident_direct_lists"]
    if type(direct) is not list or set(direct) - names or len(set(direct)) != len(direct):
        raise ValueError("resident_direct_lists must name offload lists, each once")
    points = spec["pressure_points"]
    if type(points) is not list or len(points) < 2 or any(
            type(p) is not dict or set(p) != {"name", "input_tokens", "real_batch"} or type(p["real_batch"]) is not int
            or p["real_batch"] < 2 or type(p["input_tokens"]) is not list or len(p["input_tokens"]) != 2
            for p in points) or len({p["name"] for p in points}) != len(points):
        raise ValueError("Two or more pressure_points need unique names, input_tokens [low, high] and an integer "
                         "real_batch >= 2")
    if type(spec["upstream_args"]) is not list or any(type(a) is not str for a in spec["upstream_args"]):
        raise ValueError("upstream_args must be the SGLang argv shared by every arm")
    _measurement(spec["measurement"], spec["upstream_args"])
    _acceptance(spec["acceptance"], points, spec["repeats"])
    return spec, hashlib.sha256(raw).hexdigest()


def _placed_fields(item):
    fields = {"placement.mode", "placement.offload_layers", "features.resident_direct"}
    if item["selection"] != "fixed":
        fields.add("placement.selection")
    fields |= {f"placement.{key}" for key in ("seed", "calibration") if key in item}
    return tuple(sorted(fields))


def arm_configs(spec):
    """Arms with their primary control and expected changed fields, cross pairs, and skipped arms."""
    arms, pairs, skipped = {}, [], {}

    def add(name, placement, pool, control, expected, kind, features=None):
        data = {"schema": sc.SCHEMA}
        if placement:
            data["placement"] = placement
        if pool:
            data["hot_pool"] = pool
        if features:
            data["features"] = features
        arms[name] = sc.parse(data)
        if control is not None:
            pairs.append(dict(control=control, candidate=name, expected=tuple(sorted(expected)), kind=kind))

    add("stock", None, None, None, (), None)
    pairs.append(dict(control="stock", candidate="stock", expected=(), kind="aa_noise"))
    add("all_isolated_arena", None, ISOLATED, "stock", ("hot_pool.layout",), "addressing")
    add("all_shared_arena", None, SHARED, "all_isolated_arena", ("hot_pool.mode",), "sharing")
    for item in spec["offload_lists"]:
        name, placed = item["name"], _placed_fields(item)
        placement = dict({k: v for k, v in item.items() if k != "name"}, mode="fixed_layers")
        add(f"{name}_stock_layout", placement, None, "stock", placed, "placement", DIRECT)
        add(f"{name}_isolated_arena", placement, ISOLATED, f"{name}_stock_layout", ("hot_pool.layout",), "addressing",
            DIRECT)
        add(f"{name}_shared_arena", placement, SHARED, f"{name}_isolated_arena", ("hot_pool.mode",), "sharing", DIRECT)
        pairs.append(dict(control="all_isolated_arena", candidate=f"{name}_isolated_arena", expected=placed,
                          kind="placement_borrowing_off"))
        pairs.append(dict(control="all_shared_arena", candidate=f"{name}_shared_arena", expected=placed,
                          kind="placement_borrowing_on"))
        if name in spec["resident_direct_lists"]:
            add(f"{name}_shared_resident_resolver", placement, SHARED, None, (), None)
            pairs.append(dict(control=f"{name}_shared_resident_resolver", candidate=f"{name}_shared_arena",
                              expected=("features.resident_direct",), kind="resident_direct"))
        quota = (spec["static_quota_rows"] or {}).get(name)
        if quota is None:
            skipped[f"{name}_static_quota"] = "no frozen static quota for this list (freeze it from training first)"
            continue
        add(f"{name}_static_quota", placement, dict(ISOLATED, quota="static", quota_rows=quota),
            f"{name}_isolated_arena", ("hot_pool.quota", "hot_pool.quota_rows"), "static_quota", DIRECT)
        pairs.append(dict(control=f"{name}_static_quota", candidate=f"{name}_shared_arena",
                          expected=("hot_pool.mode", "hot_pool.quota", "hot_pool.quota_rows"),
                          kind="sharing_vs_static"))
    by_name = {item["name"]: item for item in spec["offload_lists"]}
    for item in spec["offload_lists"]:
        if item["selection"] != "calibrated":
            continue
        for other in by_name.values():
            if other["selection"] == "random" and len(other["offload_layers"]) == len(item["offload_layers"]):
                fields = tuple(sorted((set(_placed_fields(item)) | set(_placed_fields(other)))
                                      - {"placement.mode", "features.resident_direct"}))
                for pool in ("isolated_arena", "shared_arena"):
                    pairs.append(dict(control=f"{other['name']}_{pool}", candidate=f"{item['name']}_{pool}",
                                      expected=fields, kind=f"selection_{pool}"))
    if not any(item["selection"] == "calibrated" for item in spec["offload_lists"]):
        skipped["calibrated_selection"] = "no calibrated list: add the training-frozen list and its digests"
    return arms, pairs, skipped


def _arm_budget(config, geo, g):
    """Arena plan, byte ledger and capacity of one arm at the stock HBM budget."""
    H, page, ratio = geo["hot_rows"], geo["page"], g["host_to_device_ratio"]
    if config.uses_fixed_layers:
        placement, cap = placement_device.capacity(config, geo, dense_layers=g["dense_layers"], ratio=ratio,
                                                   index_bytes=g["index_bytes_per_token"])
        layers = placement.offload
    else:
        size = geo["plane_rows"] - page
        layers = tuple(geo["layers"])
        cap = dict(estimated=True, device_tokens=size, device_rows=geo["plane_rows"], logical_tokens=size * ratio,
                   long_requests_upper_bound=size // (H + page))
        placement = lp.plan_placement(config, layers)
    plan = s.plan_arena(config, layers, H, geo["selected"]) if config.uses_arena else None
    if plan is not None:
        arena.validate_device_geometry(len(plan.layers), plan.selected)
    common = dict(hot_rows=H, page=page, item_bytes=geo["item_size_bytes"], slots=geo["slots"], plan=plan)
    if config.uses_fixed_layers:
        ledger = lp.placement_ledger(placement=placement, device_rows=cap["device_rows"],
                                     logical_rows=cap["logical_rows"], **common)
    else:
        ledger = s.byte_ledger(layers=layers, pool_rows=geo["plane_rows"], **common)
        ledger["hot_main_kv_bytes"], ledger["resident_main_kv_bytes"] = ledger["main_kv_bytes"], 0
    span = len(layers) * (H + page) * geo["kv_row_bytes"]       # one request's device span across hot planes
    lost = -(-ledger["arena_metadata_bytes"] // span) if ledger["arena_metadata_bytes"] else 0
    cap = dict(cap, requests_charged_for_metadata=lost, requests_upper_bound=cap["long_requests_upper_bound"] - lost)
    return dict(config=config, plan=plan, ledger=ledger, capacity=cap, placement=placement)


def _feasible(arm, point):
    cap = arm["capacity"]
    return point["real_batch"] <= cap["requests_upper_bound"] and \
        point["real_batch"] * point["input_tokens"][0] <= cap["logical_tokens"]


def compile_plan(spec, spec_digest, output_dir):
    g = spec["geometry"]
    geo = arena.describe_geometry(
        tp_size=g["tp_size"], layers=range(g["first_layer"], g["first_layer"] + g["layers"]), hot_rows=g["hot_rows"],
        page=g["page"], top_k_blocks=g["top_k_blocks"], block_size=g["block_size"], head_dim=g["head_dim"],
        dtype_bytes=g["dtype_bytes"], kv_heads_per_rank=g["kv_heads_per_rank"], slots=g["slots"],
        plane_rows=g["plane_rows"])
    configs, pairs, skipped = arm_configs(spec)
    arms, errors = {}, []
    for name, config in configs.items():
        try:
            arms[name] = _arm_budget(config, geo, g)
        except ValueError as error:
            errors.append(f"{name}: {error}")
    accept = spec["acceptance"]
    unknown = [name for name in [accept["candidate"], *accept["baselines"]] if name not in configs]
    if unknown:
        errors.append(f"acceptance names arms this spec does not compile: {unknown}")
    known = {(p["control"], p["candidate"]): p for p in pairs}
    for baseline in [] if unknown else accept["baselines"]:
        pair = known.get((baseline, accept["candidate"]))
        if pair is None:                      # cross-mechanism comparison at one HBM budget
            pair = dict(control=baseline, candidate=accept["candidate"], expected=None, kind="acceptance")
            pairs.append(pair)
        pair["acceptance"] = True
    for pair in pairs:
        pair.setdefault("acceptance", False)
        pair["priority"] = PRIORITY.index("acceptance" if pair["acceptance"] else pair["kind"])
        control, candidate = arms.get(pair["control"]), arms.get(pair["candidate"])
        if control is None or candidate is None:
            continue
        actual = changed_fields(control["config"], candidate["config"])
        pair["changed_fields"] = list(actual)
        if pair["expected"] is not None and actual != pair["expected"]:
            errors.append(f"{pair['candidate']} vs {pair['control']}: changed {list(actual)}, "
                          f"expected {list(pair['expected'])}")
        if pair["expected"] is not None and not pair["kind"].startswith("placement"):   # same S: one budget
            a, b = control["ledger"], candidate["ledger"]
            if (a["main_kv_bytes"], a["per_request"]["history_rows"]) != (b["main_kv_bytes"],
                                                                          b["per_request"]["history_rows"]):
                errors.append(f"{pair['candidate']} vs {pair['control']}: main-KV bytes or history rows differ")
    for name, arm in arms.items():
        plan = arm["plan"]
        if plan is not None and sum(size for _, size in plan.domains) + plan.scratch[1] != plan.rows:
            errors.append(f"{name}: domains and scratch do not partition the arena")
    if errors:
        raise PlanError(errors)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "plan.json"
    existing = [str(p) for p in [output_dir / f"{name}.json" for name in arms] + [plan_path] if p.exists()]
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
        summary.append(dict(arm=name, config=path.name, config_sha256=arm["config"].digest,
                            file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            capabilities=sc.capabilities(arm["config"]), offload_layers=list(arm["placement"].offload),
                            ledger=arm["ledger"], capacity=arm["capacity"],
                            estimated_feasible={p["name"]: _feasible(arm, p) for p in spec["pressure_points"]}))
    schedule, infeasible = [], []
    ordered = sorted(pairs, key=lambda p: (p["priority"], p["control"], p["candidate"]))
    for pair in ordered:
        names = (pair["control"], pair["candidate"])
        for point in spec["pressure_points"]:
            if not all(_feasible(arms[n], point) for n in names):
                infeasible.append(dict(point=point["name"], pair=f"{names[0]}->{names[1]}",
                                       reason="estimated capacity below the real batch for at least one arm"))
                continue
            for repeat in range(spec["repeats"]):
                schedule.append(dict(point=point["name"], pair=f"{names[0]}->{names[1]}", control=names[0],
                                     candidate=names[1], kind=pair["kind"], acceptance=pair["acceptance"],
                                     priority=pair["priority"], repeat=repeat, input_batch=repeat,
                                     order=list(names[::1 if repeat % 2 == 0 else -1]),
                                     real_batch=point["real_batch"], input_tokens=point["input_tokens"]))
    lists = [item["name"] for item in spec["offload_lists"]]
    placeholder = "0" * 64 in spec["split"].values()
    plan = dict(
        schema=PLAN_SCHEMA, spec_sha256=spec_digest, seed=spec["seed"], split=spec["split"], geometry=dict(geo, **g),
        split_placeholder=placeholder, arms=summary, skipped=skipped,
        pairs=[dict(control=p["control"], candidate=p["candidate"], kind=p["kind"], changed_fields=p["changed_fields"],
                    acceptance=p["acceptance"], priority=p["priority"]) for p in ordered],
        interaction={name: dict(borrowing_off=["all_isolated_arena", f"{name}_isolated_arena"],
                                borrowing_on=["all_shared_arena", f"{name}_shared_arena"],
                                estimate="(E - C_arena) - (B_shared - B_isolated) on matched real B")
                     for name in lists},
        pressure_points=spec["pressure_points"], repeats=spec["repeats"], measurement=spec["measurement"],
        acceptance=accept, resident_direct_lists=spec["resident_direct_lists"], priority=list(PRIORITY),
        infeasible=infeasible, schedule=schedule, launch=launches,
        real_batch_requirement="every window's decode log samples show the planned B on a CUDA graph, and every "
                               "request of the batch stays in the window; otherwise the run is rejected, not rescaled",
        capacity_note="estimates at the stock KV budget; workers auto-size pools (no --max-total-tokens) and their "
                      "receipts plus /server_info replace every estimate in the runner's capacity stage",
        metrics=["tpot_ms_p50", "tpot_ms_p95", "tpot_ms_p99", "tokens_per_s", "ttft_s", "slo_goodput",
                 "native_repeat_noise", "real_batch", "graph_replay"],
        cuda_graph_verification="pending_hardware",
        pending_hardware=list(PENDING) + (["replace placeholder split digests"] if placeholder else []))
    with plan_path.open("x") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compile combined-study arms A–E and a paired measurement plan.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        spec, digest = read_spec(args.spec)
        plan = compile_plan(spec, digest, args.output_dir)
    except PlanError as error:
        print(json.dumps(dict(status="fail", errors=error.args[0])), file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(status="fail", errors=[str(error)])), file=sys.stderr)
        return 1
    print(json.dumps(dict(status="pass", arms=[a["arm"] for a in plan["arms"]], pairs=len(plan["pairs"]),
                          runs=len(plan["schedule"]), infeasible=len(plan["infeasible"]),
                          skipped=plan["skipped"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())