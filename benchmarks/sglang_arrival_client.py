"""Replay a frozen open-loop arrival schedule against an already running service.

Arrivals are independent of completions. max_workers limits HTTP workers, not
the finite executor queue or the GPU batch. Queueing and request construction
count toward arrival TTFT; transport token intervals keep their original origin.
This is a load generator, not an admission controller or a GPU launcher.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import math
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event, Lock
import time

from . import sglang_stream_client as stream
from . import compare_runs as comparison


SCHEMA = "hisparse-arrival-schedule-v1"
REPORT_SCHEMA = "hisparse-open-loop-result-v1"


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("Nonfinite JSON value")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def _keys(value, names):
    if type(value) is not dict or set(value) != set(names.split()):
        raise ValueError("Unexpected arrival schedule fields")


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Expected a finite nonnegative arrival time")
    return value


def _label(value):
    if (type(value) is not str or not 1 <= len(value) <= 96 or not value[0].isalnum()
            or any(not (c.isascii() and (c.isalnum() or c in "_.-")) for c in value)):
        raise ValueError("Use an opaque ASCII arrival label")
    return value


def read_schedule(raw, workload, workload_sha256):
    """Bind every workload alias exactly once to a contiguous phase timeline."""
    value = _json(raw)
    _keys(value, "schema workload_sha256 phases arrivals")
    if value["schema"] != SCHEMA or value["workload_sha256"] != workload_sha256:
        raise ValueError("Arrival schema/workload hash mismatch")
    if (type(workload_sha256) is not str or len(workload_sha256) != 64
            or any(c not in "0123456789abcdef" for c in workload_sha256)):
        raise ValueError("Expected a workload SHA256")
    rows = {r["request_alias"]: r for r in workload}
    if not rows or len(rows) != len(workload):
        raise ValueError("Expected unique nonempty workload aliases")
    phases, end = {}, 0
    if type(value["phases"]) is not list or not value["phases"]:
        raise ValueError("Expected nonempty phases")
    for phase in value["phases"]:
        _keys(phase, "name start_s end_s")
        name = _label(phase["name"])
        start, stop = _number(phase["start_s"]), _number(phase["end_s"])
        if name in phases or start != end or stop <= start:
            raise ValueError("Phases must be unique, contiguous and increasing from zero")
        phases[name], end = phase, stop
    if type(value["arrivals"]) is not list:
        raise ValueError("Expected an arrival array")
    seen, previous = set(), 0
    for event in value["arrivals"]:
        _keys(event, "request_alias at_s phase")
        alias, phase = _label(event["request_alias"]), _label(event["phase"])
        at = _number(event["at_s"])
        p = phases.get(phase)
        if (alias in seen or alias not in rows or p is None or at < previous
                or not p["start_s"] <= at < p["end_s"]):
            raise ValueError("Invalid alias, order or phase in arrivals")
        seen.add(alias)
        previous = at
    if seen != set(rows):
        raise ValueError("Every workload request must have exactly one arrival")
    return value


def read_workload(raw):
    for line in raw.splitlines():
        if line.strip():
            row = _json(line)
            _keys(row, "request_alias workload input_ids max_new_tokens")
            _label(row["request_alias"])
            _label(row["workload"])
    return stream.read_workload(raw)


def arrival_record(record, event, timing):
    """Keep transport metrics intact and derive arrival metrics from one clock."""
    scheduled, submitted = event["at_s"], timing["submitted_s"]
    worker, api = timing["worker_started_s"], timing.get("transport_started_s")
    if not scheduled <= submitted <= worker or (api is not None and api < worker):
        raise ValueError("Nonmonotonic client timing")
    first = None if api is None or record["ttft_s"] is None else api + record["ttft_s"]
    end = timing["worker_finished_s"] if api is None else api + record["end_to_end_s"]
    return record | {"arrival_timing": dict(
        phase=event["phase"], scheduled_s=scheduled, submitted_s=submitted,
        worker_started_s=worker, transport_started_s=api,
        first_token_s=first, completed_s=end,
        dispatch_lateness_s=submitted-scheduled, executor_queue_s=worker-submitted,
        request_preparation_s=None if api is None else api-worker,
        arrival_ttft_s=None if first is None else first-scheduled,
        arrival_end_to_end_s=end-scheduled,
    )}


def summarize(records, schedule, duration, ttft_slo, tpot_slo):
    """Goodput uses the full offer-plus-drain duration, never only the busy phase."""
    transport = stream.summarize(records, duration, ttft_slo, tpot_slo, len(schedule["arrivals"]))
    arrival = []
    for record in records:
        copy = dict(record, ttft_s=record["arrival_timing"]["arrival_ttft_s"],
                    end_to_end_s=record["arrival_timing"]["arrival_end_to_end_s"])
        arrival.append(copy)
    measured = stream.summarize(arrival, duration, ttft_slo, tpot_slo, len(schedule["arrivals"]))
    for original, scored in zip(records, arrival):
        original["arrival_meets_response_slo"] = scored["meets_response_slo"]
        original["arrival_meets_natural_eos_slo"] = scored["meets_natural_eos_slo"]
    cohorts = []
    for phase in schedule["phases"]:
        count = sum(e["phase"] == phase["name"] for e in schedule["arrivals"])
        cohort = [r for r in arrival if r["arrival_timing"]["phase"] == phase["name"]]
        cohorts.append(phase | dict(requests=count, returned=len(cohort),
            offered_requests_per_s=count/(phase["end_s"]-phase["start_s"]),
            arrival_latency=comparison.latency_percentiles(cohort),
            natural_eos_compliant_requests=sum(r["meets_natural_eos_slo"] for r in cohort),
            finish_reason_counts=dict(Counter(r["finish_reason"] or "unavailable" for r in cohort))))
    return dict(transport=transport, arrival_slo=measured["slo"], phase_arrival_cohorts=cohorts,
                arrival_latency=comparison.latency_percentiles(arrival),
                configured_offer_duration_s=schedule["phases"][-1]["end_s"],
                observed_makespan_s=duration,
                drain_s=max(0, duration-schedule["phases"][-1]["end_s"]))


def run(endpoint, workload, schedule, max_workers, timeout, ttft_slo, tpot_slo, on_result=None):
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("Expected positive HTTP worker limit")
    for value in (timeout, ttft_slo, tpot_slo):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Expected positive finite timeout/SLO")
    # Revalidate public API input; identity binding was checked by read_schedule.
    schedule = read_schedule(json.dumps(schedule).encode(), workload, schedule["workload_sha256"])
    rows = {r["request_alias"]: r for r in workload}
    arrivals = schedule["arrivals"]
    horizon = schedule["phases"][-1]["end_s"]
    pool = ThreadPoolExecutor(max_workers=max_workers)
    ready, stopped, lock = SimpleQueue(), Event(), Lock()
    futures, timing, completed = {}, {}, {}
    index, status = 0, "complete"
    origin = time.perf_counter()

    def worker(event, state):
        with lock:
            state["worker_started_s"] = time.perf_counter()-origin
            if stopped.is_set():
                state["not_sent"] = True
                return None
        def start(value):
            with lock:
                if stopped.is_set():
                    state["not_sent"] = True
                    raise InterruptedError("Client stopped before transport")
                state["transport_started_s"] = value-origin
        try:
            record = stream.generate(endpoint, rows[event["request_alias"]], timeout, on_start=start)
        except Exception as error:
            if state.get("not_sent"):
                return None
            # Never log exception text: third-party errors can embed input text.
            record = dict(request_alias=event["request_alias"], workload=rows[event["request_alias"]]["workload"],
                successful=False, error="client_exception", error_type=type(error).__name__,
                output_tokens=0, finish_reason=None, ttft_s=None, tpot_s=None, end_to_end_s=0)
            if "transport_started_s" in state:
                record["end_to_end_s"] = time.perf_counter()-origin-state["transport_started_s"]
        with lock:
            state["worker_finished_s"] = time.perf_counter()-origin
            snapshot = dict(state)
        return arrival_record(record, event, snapshot)

    def collect(future):
        idx = futures[future]
        record = future.result()
        if record is not None:
            completed[idx] = record
            if on_result is not None:
                on_result(record)

    pending = set()
    try:
        while True:
            now = time.perf_counter()-origin
            while index < len(arrivals) and arrivals[index]["at_s"] <= now:
                state = {"submitted_s": time.perf_counter()-origin}
                timing[index] = state
                future = pool.submit(worker, arrivals[index], state)
                futures[future] = index
                pending.add(future)
                with lock:
                    state["submission_confirmed"] = True
                future.add_done_callback(ready.put)
                index += 1
                now = time.perf_counter()-origin
            if index == len(arrivals) and not pending and now >= horizon:
                break
            deadline = (arrivals[index]["at_s"] if index < len(arrivals)
                        else horizon if now < horizon else now+.25)
            try:
                future = ready.get(timeout=max(0, min(.25, deadline-now)))
            except Empty:
                continue
            pending.remove(future)
            collect(future)
    except KeyboardInterrupt:
        status = "partial"
    finally:
        stopped.set()
        pool.shutdown(wait=False, cancel_futures=True)
    # Completed HTTP responses are not timestamped at collection or journal time.
    for future in tuple(pending):
        if future.done() and not future.cancelled():
            collect(future)
    client_loop_elapsed = time.perf_counter()-origin
    with lock:
        states = {idx: dict(value) for idx, value in timing.items()}
    lifecycle = []
    for idx, event in enumerate(arrivals):
        state = states.get(idx, {})
        if idx in completed:
            kind = "completed"
        elif idx not in states:
            kind = "not_submitted"
        elif state.get("not_sent"):
            kind = "not_sent_cancelled"
        elif not state.get("submission_confirmed"):
            kind = "in_flight_or_unobserved"
        elif "worker_started_s" not in state:
            kind = "not_sent_cancelled"
        else:
            kind = "in_flight_or_unobserved"
        lifecycle.append(dict(request_alias=event["request_alias"], state=kind, timing=state))
    records = [completed[idx] for idx in sorted(completed)]
    # Do not put completion collection / journal flushing into EOF or makespan.
    duration = (max(horizon, max((r["arrival_timing"]["completed_s"] for r in records), default=0))
                if status == "complete" else client_loop_elapsed)
    return dict(schema=REPORT_SCHEMA, status=status, http_worker_limit=max_workers,
        client_loop_elapsed_s=client_loop_elapsed,
        request_lifecycle=lifecycle, requests=records,
        summary=summarize(records, schedule, duration, ttft_slo, tpot_slo),
        measurement_scope=dict(
            arrivals="Frozen open-loop schedule; late submissions do not shift later deadlines. Queue is bounded only by the finite workload.",
            latency="Arrival TTFT includes dispatch lateness, executor queue and request preparation. Transport API start is not wire send or server admission. TPOT is unchanged.",
            congestion="Client queue/dispatch delays are not server waiting. Validate client capacity before attributing arrival-SLO changes to the mechanism.",
            throughput="Complete-run cohort goodput uses max(configured offer end, last response EOF), excluding collection/journal tail; not steady-state throughput. Partial-run rates are descriptive only. Phase labels describe arrivals, not completion intervals.",
            partial="All scheduled requests remain in denominator. Missing requests are not server rejections. Running HTTP workers can outlive return/process shutdown until completion or socket inactivity timeout; no server cancellation or retry.",
            correctness="Independent token correctness and actual GPU batch/physical-page reuse require separate service evidence."))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-results", type=Path)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--timeout", type=stream.positive_float, default=600)
    parser.add_argument("--ttft-slo-s", type=stream.positive_float, required=True)
    parser.add_argument("--tpot-slo-s", type=stream.positive_float, required=True)
    args = parser.parse_args(argv)
    # Delayed import avoids the planner -> schedule parser dependency cycle.
    from .artifact_paths import _named_path
    raw = _named_path(args.workload).read_bytes()
    schedule_raw = _named_path(args.schedule).read_bytes()
    workload = read_workload(raw)
    schedule = read_schedule(schedule_raw, workload, sha256(raw).hexdigest())
    output_path = _named_path(args.output, writing=True)
    journal_path = _named_path(args.request_results or args.output.with_suffix(".request-results.jsonl"), writing=True)
    if args.max_workers < 1 or output_path == journal_path:
        parser.error("Use positive workers and distinct fresh output paths")
    sources = dict(workload=sha256(raw).hexdigest(), schedule=sha256(schedule_raw).hexdigest(),
                   client=sha256(Path(__file__).read_bytes()).hexdigest(),
                   latency_metrics=sha256(Path(comparison.__file__).read_bytes()).hexdigest(),
                   stream_client=sha256(Path(stream.__file__).read_bytes()).hexdigest())
    with journal_path.open("x", encoding="utf-8") as journal, output_path.open("x", encoding="utf-8") as output:
        json.dump(dict(schema=REPORT_SCHEMA, status="partial", source_sha256=sources,
                       scheduled_request_aliases=[e["request_alias"] for e in schedule["arrivals"]]), output)
        output.write("\n")
        output.flush()
        def append(record):
            journal.write(json.dumps(record, allow_nan=False)+"\n")
            journal.flush()
        result = run(args.url, workload, schedule, args.max_workers, args.timeout,
                     args.ttft_slo_s, args.tpot_slo_s, on_result=append)
        result["source_sha256"] = sources
        output.seek(0)
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
        output.truncate()
    return 130 if result["status"] == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
