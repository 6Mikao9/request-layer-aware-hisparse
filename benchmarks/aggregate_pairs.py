"""Aggregate a prespecified list of SGLang run pairs offline."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean, median

from benchmarks.compare_runs import (
    INPUT_FIELDS, LATENCY_FIELDS, compare_runs, finite_float, is_count, is_number, json_object, read_run, require,
)


WINDOW = (8, 32)
WINDOW_METRIC = "token_window_s_per_token"
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def validate_plan(plan):
    require(isinstance(plan, dict) and set(plan) == {"pairs", "slo", "token_window"},
            "plan", "expected pairs, slo and token_window")
    require(plan["token_window"] == list(WINDOW)
            and all(type(value) is int for value in plan["token_window"]),
            "plan.token_window", "expected fixed token ordinals [8, 32]")
    thresholds = plan["slo"]
    require(isinstance(thresholds, list) and bool(thresholds)
            and all(isinstance(pair, list) and len(pair) == 2
                    and all(is_number(value) and value > 0 for value in pair) for pair in thresholds),
            "plan.slo", "expected one or more finite positive [TTFT, TPOT] threshold pairs")
    require(len({tuple(pair) for pair in thresholds}) == len(thresholds), "plan.slo", "duplicate threshold pair")
    require(isinstance(plan["pairs"], list) and bool(plan["pairs"]), "plan.pairs", "expected at least one pair")
    labels = set()
    for index, pair in enumerate(plan["pairs"]):
        context = f"plan.pairs[{index}]"
        require(isinstance(pair, dict) and set(pair) == {"label", "split", "order", "comparison", "a", "b"},
                context, "expected label, split, order, comparison, a and b")
        label = pair["label"]
        require(isinstance(label, str) and LABEL.fullmatch(label) and label not in labels,
                context, "expected a unique anonymous label")
        labels.add(label)
        require(pair["split"] in ("train", "holdout"), context, "split must be train or holdout")
        require(pair["order"] in ("AB", "BA"), context, "order must be AB or BA")
        require(pair["comparison"] in ("ab", "aa"), context, "comparison must be ab or aa")
        require(all(isinstance(pair[arm], str) and pair[arm].strip() for arm in ("a", "b")),
                context, "expected nonempty paths for a and b")


def distribution(values):
    return {
        "included": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "negative": sum(value < 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "positive": sum(value > 0 for value in values),
    }


def fixed_window(record, makespan, context):
    if not record["successful"]:
        return None, "failed"
    short = record["output_tokens"] < WINDOW[1]
    events = record.get("token_events")
    if events is None or events == []:
        return None, "short_output" if short else "missing_events"
    require(isinstance(events, list), context, "expected a token event list")
    previous_count = previous_arrival = 0
    boundaries = {}
    for index, event in enumerate(events):
        location = f"{context}.token_events[{index}]"
        require(isinstance(event, dict), location, "expected a token event object")
        count, delta, arrival = (event.get(field) for field in ("completion_tokens", "new_tokens", "arrival_s"))
        require(is_count(delta, 1) and is_count(count, 1) and count == previous_count + delta
                and count <= record["output_tokens"], location, "inconsistent cumulative token event counts")
        require(is_number(arrival) and previous_arrival <= arrival <= makespan
                and (record["end_to_end_s"] is None or arrival <= record["end_to_end_s"] + 1e-9),
                location, "expected ordered finite event arrivals within the request and run")
        if index == 0:
            require(record["ttft_s"] is not None and abs(arrival - record["ttft_s"]) <= 1e-9,
                    location, "first arrival differs from TTFT")
            first_count = record.get("first_event_tokens")
            require(type(first_count) is int and first_count == delta,
                    location, "first event count differs from first_event_tokens")
        if count in WINDOW and delta == 1:
            boundaries[count] = arrival
        previous_count, previous_arrival = count, arrival
    if previous_count != record["output_tokens"]:
        return None, "short_output" if short else "incomplete_events"
    # The same complete event trace must support the reported SLO latency,
    # even when a request is too short to contribute to the fixed window.
    expected_tpot = (previous_arrival - record["ttft_s"]) / (previous_count - 1) if (
        first_count == 1 and previous_count > 1) else None
    tpot = record["tpot_s"]
    require((expected_tpot is None and tpot is None)
            or (expected_tpot is not None and tpot is not None and abs(tpot - expected_tpot) <= 1e-9),
            context, "TPOT differs from complete token events")
    if short:
        return None, "short_output"
    if len(boundaries) != 2:
        return None, "coalesced_boundary"
    return (boundaries[WINDOW[1]] - boundaries[WINDOW[0]]) / (WINDOW[1] - WINDOW[0]), "eligible"


def block_metrics(left, right, overall, context):
    records = [{row["request_alias"]: row for row in run["requests"]} for run in (left, right)]
    summaries = (overall["baseline"], overall["candidate"])
    windows = [Counter(), Counter()]
    measurements = {field: [] for field in (*LATENCY_FIELDS, WINDOW_METRIC)}
    for index, (alias, a) in enumerate(records[0].items()):
        b = records[1][alias]
        for field in LATENCY_FIELDS:
            if a["successful"] and b["successful"] and a[field] is not None and b[field] is not None:
                measurements[field].append((a[field], b[field]))
        timings = []
        for arm, row in enumerate((a, b)):
            timing, reason = fixed_window(row, summaries[arm]["run_makespan_s"], f"{context}.arm[{arm}].request[{index}]")
            windows[arm][reason] += 1
            timings.append(timing)
        if all(value is not None for value in timings):
            measurements[WINDOW_METRIC].append(timings)

    arms = []
    for summary, window_counts in zip(summaries, windows):
        tokens = summary["completed_output_tokens"] + summary["failed_requests_observed_output_tokens"]
        arms.append(summary | {
            "all_observed_output_tokens": tokens,
            "mean_observed_output_tokens_per_request": tokens / summary["requests"],
            "mean_completed_output_tokens_per_success": (
                summary["completed_output_tokens"] / summary["successes"] if summary["successes"] else None),
            "token_window_counts": dict(window_counts),
        })
    total = len(records[0])
    paired = {
        field: distribution([b - a for a, b in values]) | {
            "total_requests": total, "excluded": total - len(values),
            "a_mean": mean(a for a, _ in values) if values else None,
            "b_mean": mean(b for _, b in values) if values else None,
        }
        for field, values in measurements.items()
    }
    return arms, paired


def summarize_blocks(blocks):
    a_b_metrics = ("mean_observed_output_tokens_per_request", "all_observed_output_tokens_per_s",
                   "failure_fraction_all_requests", "length_limited_fraction_all_requests")
    measurements = {field: [block["b"][field] - block["a"][field] for block in blocks]
                    for field in a_b_metrics}
    for field in (*LATENCY_FIELDS, WINDOW_METRIC):
        values = [block["paired_latency_b_minus_a"][field]["mean"] for block in blocks]
        measurements[field] = [value for value in values if value is not None]
    slo = []
    if blocks:
        for index, thresholds in enumerate(blocks[0]["a"]["slo"]):
            slo.append({field: thresholds[field] for field in ("ttft_s", "tpot_s")} | {
                field: distribution([block["b"]["slo"][index][field] - block["a"]["slo"][index][field]
                                     for block in blocks])
                for field in ("natural_eos_slo_goodput_per_s", "response_slo_goodput_per_s")
            })
    return {"blocks": len(blocks), "order_counts": dict(Counter(block["order"] for block in blocks)),
            "block_deltas_b_minus_a": {field: distribution(values) for field, values in measurements.items()},
            "slo_deltas_b_minus_a": slo}


def input_signature(run):
    return {row["request_alias"]: {field: row[field].lower() if field == "input_ids_sha256" else row[field]
                                    for field in INPUT_FIELDS} for row in run["requests"]}


def aggregate_pairs(plan, base_directory):
    validate_plan(plan)
    used_files = set()
    inputs_by_workload = {}
    inputs_by_split = {split: set() for split in ("train", "holdout")}
    groups = defaultdict(list)
    blocks = []
    for pair in plan["pairs"]:
        context = pair["label"]
        reports = []
        for arm in ("a", "b"):
            path = (base_directory / pair[arm]).resolve()
            require(path not in used_files, context, "a run report is reused across pair slots")
            used_files.add(path)
            reports.append(read_run(path))
        left, right = reports
        comparison = compare_runs(left, right, plan["slo"])
        pairing = comparison["pairing"]
        require(pairing["same_client_source_sha256"], context, "client source SHA-256 differs")
        workload = pairing["workload_sha256"]
        signature = input_signature(left)
        require(inputs_by_workload.setdefault(workload, signature) == signature,
                context, "repeated workload SHA-256 has different request records")
        inputs_by_split[pair["split"]].update(row["input_ids_sha256"] for row in signature.values())
        (a, b), paired = block_metrics(left, right, comparison["overall"], context)
        block = {field: pair[field] for field in ("label", "split", "order", "comparison")} | {
            "pairing": pairing, "a": a, "b": b, "paired_latency_b_minus_a": paired,
            "output_id_hash_pairs": comparison["overall"]["output_id_hash_pairs"],
        }
        blocks.append(block)
        groups[pair["split"], workload, pairing["baseline_client_sha256"], pairing["client_concurrency"]].append(block)
    strata = []
    for (split, workload, client, concurrency), group in sorted(groups.items()):
        strata.append({
            "split": split, "workload_sha256": workload, "client_sha256": client, "client_concurrency": concurrency,
            "distinct_input_hashes": len({row["input_ids_sha256"] for row in inputs_by_workload[workload].values()}),
            "ab": summarize_blocks([block for block in group if block["comparison"] == "ab"]),
            "aa_noise": summarize_blocks([block for block in group if block["comparison"] == "aa"]),
        })
    return {
        "schema": "hisparse-repeated-run-pairs-v1",
        "environment": comparison["environment"],
        "slo": plan["slo"], "token_window": list(WINDOW), "blocks": blocks, "strata": strata,
        "distinct_input_hashes_by_split": {split: len(inputs) for split, inputs in inputs_by_split.items()},
        "input_hashes_shared_by_train_and_holdout": len(inputs_by_split["train"] & inputs_by_split["holdout"]),
        "interpretation": {
            "differences": "B minus A regardless of order. Negative latency and positive goodput favor B.",
            "repeats": "Descriptive summaries weight blocks equally within each stratum. Repeated inputs share "
                       "sources. Distinct hashes do not establish independent samples.",
            "aa_noise": "Min and max describe observed same-mode block deltas within the stratum. "
                        "Order and same-mode declarations are supplied by the plan. Verify with the controller.",
            "token_window": "Successful requests with observed single-token boundaries at ordinals 8 and 32 "
                            "on both arms. Elapsed seconds divided by 24. This compares arrival timing at fixed "
                            "positions while output token content and trajectories can differ.",
            "slo": "All planned requests remain in the SLO denominator, including failed and short responses.",
            "output_ids": "Output hash counts describe available response IDs. Verify cache correctness independently.",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="Prespecified JSON design and run report paths")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path")
    args = parser.parse_args(argv)
    try:
        plan_bytes = args.plan.read_bytes()
        plan = json.loads(plan_bytes.decode("utf-8"), object_pairs_hook=json_object,
                          parse_float=finite_float, parse_constant=finite_float)
        report = aggregate_pairs(plan, args.plan.parent)
        report["plan_sha256"] = sha256(plan_bytes).hexdigest()
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(serialized)
    except (ValueError, OSError, OverflowError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
