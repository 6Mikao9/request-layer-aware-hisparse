"""Compare two complete native SGLang stream-client reports offline."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re


INPUT_FIELDS = ("workload", "input_ids_sha256", "input_tokens", "max_new_tokens")
LATENCY_FIELDS = ("ttft_s", "tpot_s", "end_to_end_s")
SHA256 = re.compile(r"[a-fA-F0-9]{64}")
ENVIRONMENT_EVIDENCE = (
    "Model revision, weights, tokenizer, precision, effective sampling and server/runtime revisions",
    "Effective server_info/config: graph, overlap, Radix, TP, attention backend and KV/HiSparse settings",
    "Measured GPU residency and budgets: per-device NVML plus server allocation and pool settings",
    "GPU identity/topology, competing activity, clocks, warmup, client timeouts and measurement protocol",
)


def require(condition, context, expected):
    if not condition:
        raise ValueError(f"{context}: {expected}")


def is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def is_count(value, minimum=0):
    return type(value) is int and value >= minimum


def is_sha(value):
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def validate_record(record, context, makespan):
    require(isinstance(record, dict), context, "expected an object")
    for field in ("request_alias", "workload"):
        require(isinstance(record.get(field), str) and bool(record[field]),
                f"{context}.{field}", "expected a nonempty string")
    require(is_sha(record.get("input_ids_sha256")), context, "expected an input SHA-256")
    for field in ("input_tokens", "max_new_tokens"):
        require(is_count(record.get(field), 1), f"{context}.{field}", "expected a positive integer")
    count = record.get("output_tokens")
    require(is_count(count) and count <= record["max_new_tokens"], context,
            "expected an output token count within the request budget")
    successful = record.get("successful")
    require(type(successful) is bool, context, "expected a Boolean success flag")
    reason = record.get("finish_reason")
    require("finish_reason" in record and reason in (None, "stop", "length", "abort", "error"),
            context, "expected a valid finish reason or null")
    if successful:
        require(reason in ("stop", "length") and "error" not in record,
                context, "successful requests must finish with stop or length")
        require(reason != "length" or count == record["max_new_tokens"], context,
                "successful length stop must reach the request budget")
    else:
        require(isinstance(record.get("error"), str) and bool(record["error"]),
                context, "failed requests must include an error category")
    for field in LATENCY_FIELDS:
        value = record.get(field)
        require(field in record and (value is None or (is_number(value) and 0 <= value <= makespan)),
                f"{context}.{field}", "expected null or finite nonnegative seconds within the run makespan")
    first, tpot, e2e = (record[field] for field in LATENCY_FIELDS)
    require(first is None or count >= 1, context, "TTFT requires an observed output token")
    first_count = record.get("first_event_tokens")
    require(tpot is None or (first is not None and count >= 2 and type(first_count) is int and first_count == 1),
            context, "TPOT requires a single-token first event and multiple output tokens")
    if e2e is not None and first is not None:
        last = first if tpot is None else first + tpot * (count - 1)
        require(last <= e2e + 1e-9, context, "token arrival timings exceed end-to-end latency")
    digest = record.get("output_ids_sha256")
    status = record.get("output_ids_status")
    require((status == "complete" and successful and is_sha(digest))
            or (status == "unavailable" and "output_ids_sha256" in record and digest is None),
            context, "inconsistent output ID hash and availability")


def validate_run(run, context):
    require(isinstance(run, dict), context, "expected an object")
    require(run.get("status") == "complete" and run.get("missing_request_aliases") == [],
            context, "expected a complete report with no missing aliases")
    source = run.get("source_sha256")
    require(isinstance(source, dict) and all(is_sha(source.get(field)) for field in ("workload", "client")),
            context, "expected workload and client SHA-256 provenance")
    require(is_count(run.get("client_concurrency"), 1), context, "expected positive client concurrency")
    summary, records = run.get("summary"), run.get("requests")
    require(isinstance(summary, dict), context, "expected a run summary")
    require(isinstance(records, list) and bool(records), context, "expected a nonempty request list")
    makespan = summary.get("makespan_s")
    require(is_number(makespan) and makespan > 0, context, "expected a finite positive makespan")
    indexed = {}
    for number, record in enumerate(records, 1):
        location = f"{context}.requests[{number - 1}]"
        validate_record(record, location, makespan)
        alias = record["request_alias"]
        require(alias not in indexed, location, "duplicate request alias")
        indexed[alias] = record
    successes = sum(record["successful"] for record in records)
    expected_counts = {"requests": len(records), "unreturned_requests": 0,
                       "successes": successes, "failures": len(records) - successes}
    for field, expected in expected_counts.items():
        value = summary.get(field)
        require(is_count(value) and value == expected, f"{context}.summary.{field}",
                "count is inconsistent with complete request records")
    return indexed


def quantile(sorted_values, fraction):
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    low, high = sorted_values[lower], sorted_values[math.ceil(position)]
    return low + (position - lower) * (high - low)


def latency_percentiles(records):
    successful = [record for record in records if record["successful"]]
    failures = len(records) - len(successful)
    latencies = {}
    for field in LATENCY_FIELDS:
        values = sorted(record[field] for record in successful if record[field] is not None)
        latencies[field] = {
            "p50": quantile(values, 0.50), "p95": quantile(values, 0.95),
            "p99": quantile(values, 0.99),
            "included_successes": len(values), "missing_successes": len(successful) - len(values),
            "excluded_failures": failures,
        }
    return latencies


def summarize(records, makespan, thresholds):
    successful = [record for record in records if record["successful"]]
    failures = len(records) - len(successful)
    finishes = Counter(record["finish_reason"] for record in successful)
    tokens = sum(record["output_tokens"] for record in successful)
    failed_tokens = sum(record["output_tokens"] for record in records if not record["successful"])
    slo_results = []
    for ttft, tpot in thresholds:
        response = [record for record in successful
                    if record["ttft_s"] is not None and record["tpot_s"] is not None
                    and record["ttft_s"] <= ttft and record["tpot_s"] <= tpot]
        natural = sum(record["finish_reason"] == "stop" for record in response)
        slo_results.append({
            "ttft_s": ttft, "tpot_s": tpot,
            "natural_eos_compliant_requests": natural,
            "natural_eos_compliant_fraction_all_requests": natural / len(records),
            "natural_eos_slo_goodput_per_s": natural / makespan,
            "response_compliant_requests": len(response),
            "response_compliant_fraction_all_requests": len(response) / len(records),
            "response_slo_goodput_per_s": len(response) / makespan,
        })
    return {
        "run_makespan_s": makespan, "requests": len(records), "successes": len(successful),
        "failures": failures, "failure_fraction_all_requests": failures / len(records),
        "natural_eos_requests": finishes["stop"], "length_limited_requests": finishes["length"],
        "length_limited_fraction_all_requests": finishes["length"] / len(records),
        "completed_output_tokens": tokens, "completed_output_tokens_per_s": tokens / makespan,
        "failed_requests_observed_output_tokens": failed_tokens,
        "all_observed_output_tokens_per_s": (tokens + failed_tokens) / makespan,
        "latency": latency_percentiles(records), "slo": slo_results,
    }


def difference(baseline, candidate):
    return None if baseline is None or candidate is None else candidate - baseline


def compare_group(pairs, baseline_duration, candidate_duration, thresholds):
    baseline_rows, candidate_rows = zip(*pairs)
    baseline = summarize(baseline_rows, baseline_duration, thresholds)
    candidate = summarize(candidate_rows, candidate_duration, thresholds)
    ids = {"equal": 0, "different": 0, "unavailable": 0}
    for left, right in pairs:
        left_hash, right_hash = left["output_ids_sha256"], right["output_ids_sha256"]
        if left_hash is None or right_hash is None:
            ids["unavailable"] += 1
        else:
            ids["equal" if left_hash.lower() == right_hash.lower() else "different"] += 1

    changes = {field: candidate[field] - baseline[field] for field in (
        "run_makespan_s", "completed_output_tokens_per_s", "all_observed_output_tokens_per_s",
        "failure_fraction_all_requests", "length_limited_fraction_all_requests",
    )}
    changes["latency"] = {
        field: {pct: difference(baseline["latency"][field][pct], candidate["latency"][field][pct])
                for pct in ("p50", "p95", "p99")}
        for field in LATENCY_FIELDS
    }
    changes["slo"] = [
        {"ttft_s": left["ttft_s"], "tpot_s": left["tpot_s"],
         "natural_eos_slo_goodput_per_s": right["natural_eos_slo_goodput_per_s"]
         - left["natural_eos_slo_goodput_per_s"],
         "response_slo_goodput_per_s": right["response_slo_goodput_per_s"]
         - left["response_slo_goodput_per_s"]}
        for left, right in zip(baseline["slo"], candidate["slo"])
    ]
    return {"baseline": baseline, "candidate": candidate,
            "candidate_minus_baseline": changes, "output_id_hash_pairs": ids}


def compare_runs(baseline, candidate, thresholds):
    thresholds = [tuple(pair) for pair in thresholds]
    require(bool(thresholds) and all(len(pair) == 2 and all(is_number(value) and value > 0 for value in pair)
                                     for pair in thresholds),
            "SLOs", "provide one or more finite positive TTFT/TPOT threshold pairs")
    require(len(set(thresholds)) == len(thresholds), "SLOs", "duplicate threshold pair")
    left, right = validate_run(baseline, "baseline"), validate_run(candidate, "candidate")
    left_source, right_source = baseline["source_sha256"], candidate["source_sha256"]
    require(left_source["workload"].lower() == right_source["workload"].lower(), "pair", "workload SHA-256 differs")
    require(baseline["client_concurrency"] == candidate["client_concurrency"], "pair", "client concurrency differs")
    require(left.keys() == right.keys(), "pair", "request alias sets differ")
    pairs = []
    workloads = defaultdict(list)
    for number, (alias, left_record) in enumerate(left.items(), 1):
        right_record = right[alias]
        for field in INPUT_FIELDS:
            left_value, right_value = left_record[field], right_record[field]
            if field == "input_ids_sha256":
                left_value, right_value = left_value.lower(), right_value.lower()
            require(left_value == right_value, f"pair record {number}", f"{field} differs")
        pair = (left_record, right_record)
        pairs.append(pair)
        workloads[left_record["workload"]].append(pair)
    durations = baseline["summary"]["makespan_s"], candidate["summary"]["makespan_s"]
    return {
        "pairing": {"status": "matched_reported_inputs", "workload_sha256": left_source["workload"].lower(),
                    "client_concurrency": baseline["client_concurrency"],
                    "baseline_client_sha256": left_source["client"].lower(),
                    "candidate_client_sha256": right_source["client"].lower(),
                    "same_client_source_sha256": left_source["client"].lower() == right_source["client"].lower()},
        "environment": {"status": "unverified", "evidence_needed": list(ENVIRONMENT_EVIDENCE)},
        "interpretation": {
            "metric_selection": "Prespecify the performance metric for the load regime. Report throughput, "
                                "latency distributions, failures and SLO goodput for the same requests.",
            "slo_thresholds": "Externally supplied inclusive limits. Missing metrics and failures are misses.",
            "rates": "All rates use the full run makespan. Workload rows are contributions to mixed-run rates.",
            "latency": "Linear interpolation among successful requests with the metric available.",
            "differences": "Candidate minus baseline. Positive goodput/throughput and negative latency/failure "
                           "deltas indicate favorable observations. Length-stop deltas are descriptive.",
            "output_ids": "Equal hashes provide evidence of matching streamed output IDs only. Kernel bytes "
                          "and model accuracy remain unverified.",
            "scope": "Observed differences from two reports. Effective configuration, equal GPU budgets and "
                     "causal acceleration remain unverified.",
        },
        "overall": compare_group(pairs, *durations, thresholds),
        "by_workload": {label: compare_group(group, *durations, thresholds)
                        for label, group in sorted(workloads.items())},
    }


def json_object(items):
    result = {}
    for key, value in items:
        require(key not in result, "JSON", "duplicate object key")
        result[key] = value
    return result


def finite_float(value):
    parsed = float(value)
    require(math.isfinite(parsed), "JSON", "expected finite numbers")
    return parsed


def read_run(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source, object_pairs_hook=json_object, parse_float=finite_float,
                         parse_constant=finite_float)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--slo", type=float, nargs=2, action="append", required=True,
                        metavar=("TTFT_S", "TPOT_S"), help="Repeat for each prespecified SLO pair")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path")
    args = parser.parse_args(argv)
    try:
        report = compare_runs(read_run(args.baseline), read_run(args.candidate), args.slo)
        output = json.dumps(report, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(output)
    except (ValueError, OSError, OverflowError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
