"""Decode windows, real-batch evidence and paired verdicts of the combined study; standard library only.

``runtime.combined_run`` uses these on the maintainer's host and the CPU tests use them
on synthetic traces. Nothing here launches a process or reads private inputs.

* ``parse_decode_line``: SGLang's decode log line (``scheduler_components/metrics_reporter``):
  running batch, CUDA graph flag, queue. With ``--decode-log-interval N`` one line per N steps.
* ``steady_window``: the full-batch decode window of one run from the client's token arrivals.
  It starts once every request has made ``warmup_steps`` decode tokens after its first token
  (so prefill, the prefill barrier and warmup lie before it) and ends at the first request's
  last token, after which the batch is no longer full. Per-request TPOT is the mean interval
  between in-window tokens; throughput is all in-window tokens over the window's wall time.
  The fixed workload (``ignore_eos``, every request runs to its budget) is measured this way.
* ``natural_window``: the natural-load decode phase, from the prefill barrier to the last token.
* ``real_batch``: decode log samples inside the window must all show the planned B on a graph;
  under natural load they must reach B and never exceed it.
* ``paired``: per block (one AB/BA round with its own input batch, the independent unit)
  the relative output-throughput gain and p95-TPOT change; a two-sided t interval over
  blocks; ``pass`` when the interval excludes zero, the mean gain reaches the preregistered
  target and the tail and quality checks hold, ``fail`` when the interval excludes the
  target or quality fails, else ``inconclusive``.
* ``acceptance``: the primary claim against every declared baseline over pressure points.
"""
import math
import re
from statistics import mean, stdev

from benchmarks.compare_runs import quantile

DECODE = re.compile(r"Decode batch(?: \[\d+\])?, #running-req: (?P<running>\d+), .*?"
                    r"(?:cuda|cpu|npu|musa) graph: (?P<graph>True|False), "
                    r"gen throughput \(token/s\): (?P<throughput>[0-9.]+), #queue-req: (?P<queue>\d+)")
# Two-sided 97.5% Student t quantiles for 1..30 degrees of freedom; 1.96 beyond.
T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)
VERDICTS = ("pass", "fail", "inconclusive")


def parse_decode_line(line):
    match = DECODE.search(line)
    if match is None:
        return None
    return dict(running=int(match["running"]), graph=match["graph"] == "True", queue=int(match["queue"]),
                gen_throughput=float(match["throughput"]))


def _percentiles(values, scale=1.0):
    ordered = sorted(v * scale for v in values)
    return dict(p50=quantile(ordered, 0.50), p95=quantile(ordered, 0.95), p99=quantile(ordered, 0.99),
                included=len(ordered))


def _token_time(events, ordinal):
    """Arrival time of the ``ordinal``-th output token (1-based) from (time, new tokens) events."""
    count = 0
    for time, tokens in events:
        count += tokens
        if count >= ordinal:
            return time
    return None


def steady_window(requests, *, batch, warmup_steps, min_steps):
    """Full-batch decode window of one run; returns (window or None, rejection reasons).

    ``requests`` hold ``start`` and ``events`` [(time, new tokens)] on one monotonic clock,
    ``successful`` and ``finish_reason``.
    """
    good = [r for r in requests if r["successful"] and r["events"]]
    if len(requests) != batch or len(good) != batch:
        return None, [f"{batch - len(good)} of {batch} planned requests failed, are missing or made no token"]
    starts = [_token_time(r["events"], 1 + warmup_steps) for r in good]
    if any(t is None for t in starts):
        return None, [f"a request ended within the {warmup_steps} warmup decode steps"]
    origin = min(r["start"] for r in good)
    start, end = max(starts), min(r["events"][-1][0] for r in good)
    inside = [[(t, n) for t, n in r["events"] if start <= t <= end] for r in good]
    after = [sum(n for _, n in events[1:]) for events in inside]
    steps = min(after) if all(inside) else 0
    if end <= start or steps < min_steps:
        return None, [f"window holds {steps} full-batch decode steps; the plan needs {min_steps}"]
    tpot = [(events[-1][0] - events[0][0]) / count for events, count in zip(inside, after)]
    delivered = sum(n for r in good for t, n in r["events"] if start < t <= end)
    finishes = {}
    for r in good:
        finishes[r["finish_reason"]] = finishes.get(r["finish_reason"], 0) + 1
    return dict(start_s=start - origin, end_s=end - origin, seconds=end - start,
                barrier_s=max(r["events"][0][0] for r in good) - origin, warmup_steps=warmup_steps, steps=steps,
                requests=batch, delivered_tokens=delivered, tokens_per_s=delivered / (end - start),
                tpot_ms=_percentiles(tpot, 1000.0), ttft_s=_percentiles([r["events"][0][0] - r["start"] for r in good]),
                finish_reasons=finishes), []


def natural_window(requests, *, batch):
    """Natural-load decode phase of one run (EOS ends requests); returns (window or None, rejection reasons).

    It runs from the prefill barrier (every request has its first token) to the last token. The batch
    shrinks as requests stop, so throughput is every token after the barrier over that time and
    per-request TPOT is the mean interval after its first token; ``steps`` counts the full-batch steps
    before the first request ends.
    """
    good = [r for r in requests if r["successful"] and r["events"]]
    if len(requests) != batch or len(good) != batch:
        return None, [f"{batch - len(good)} of {batch} planned requests failed, are missing or made no token"]
    origin = min(r["start"] for r in good)
    start, end = max(r["events"][0][0] for r in good), max(r["events"][-1][0] for r in good)
    first_end = min(r["events"][-1][0] for r in good)
    if end <= start:
        return None, ["no request decoded after the prefill barrier"]
    tokens = [sum(n for _, n in r["events"]) for r in good]
    tpot = [(r["events"][-1][0] - r["events"][0][0]) / (count - 1) for r, count in zip(good, tokens) if count > 1]
    delivered = sum(n for r in good for t, n in r["events"] if t > start)
    finishes = {}
    for r in good:
        finishes[r["finish_reason"]] = finishes.get(r["finish_reason"], 0) + 1
    return dict(start_s=start - origin, end_s=end - origin, full_batch_end_s=first_end - origin, seconds=end - start,
                barrier_s=start - origin, steps=min(sum(n for t, n in r["events"] if start < t <= first_end)
                                                    for r in good),
                requests=batch, delivered_tokens=delivered, tokens_per_s=delivered / (end - start),
                tpot_ms=_percentiles(tpot, 1000.0), ttft_s=_percentiles([r["events"][0][0] - r["start"] for r in good]),
                output_tokens=_percentiles(tokens), finish_reasons=finishes,
                natural_eos_fraction=finishes.get("stop", 0) / batch), []


def real_batch(samples, *, batch, start, end, graph=True, min_samples=3, shrinking=False):
    """Decode log samples [(time, parsed line)] inside [start, end]: all at B, on a CUDA graph. With
    ``shrinking`` (natural load, where EOS shrinks the batch) they must reach B and never exceed it."""
    inside = [line for time, line in samples if start <= time <= end]
    running = sorted({line["running"] for line in inside})
    reasons = []
    if len(inside) < min_samples:
        reasons.append(f"{len(inside)} decode log samples in the window; need {min_samples} "
                       "(lower --decode-log-interval or lengthen the window)")
    if running and (running[-1] != batch if shrinking else running != [batch]):
        reasons.append(f"decode log shows running batch {running}; the plan needs {batch}")
    if graph and not all(line["graph"] for line in inside):
        reasons.append("some window steps ran without CUDA graph replay")
    return dict(samples=len(inside), running=running, graph=all(line["graph"] for line in inside) if inside else None,
                queue_max=max((line["queue"] for line in inside), default=None)), reasons


def output_mismatch(control, candidate):
    """Fraction of shared request aliases whose complete output-ID hashes differ (None hashes count as differ)."""
    shared = sorted(set(control) & set(candidate))
    if not shared or set(control) != set(candidate):
        return 1.0
    return sum(control[a] is None or control[a] != candidate[a] for a in shared) / len(shared)


def t_interval(values):
    """Two-sided 95% Student t interval of the mean; None below two values."""
    if len(values) < 2:
        return None
    half = (T975[len(values) - 2] if len(values) <= 31 else 1.96) * stdev(values) / math.sqrt(len(values))
    return (mean(values) - half, mean(values) + half)


def paired(blocks, *, min_gain, tpot_p95_regression_max, min_pairs):
    """Verdict of one (pressure point, control -> candidate) pair over its complete AB/BA blocks.

    A block is ``dict(round, order, control, candidate, quality_ok)`` with the two run windows.
    """
    gains = [b["candidate"]["tokens_per_s"] / b["control"]["tokens_per_s"] - 1 for b in blocks]
    tails = [b["candidate"]["tpot_ms"]["p95"] / b["control"]["tpot_ms"]["p95"] - 1 for b in blocks]
    medians = [b["candidate"]["tpot_ms"]["p50"] / b["control"]["tpot_ms"]["p50"] - 1 for b in blocks]
    interval, reasons = t_interval(gains), []
    gain, tail = (mean(gains) if gains else None), (mean(tails) if tails else None)
    if not all(b["quality_ok"] for b in blocks):
        verdict = "fail"
        reasons.append("outputs differ from the control beyond the preregistered quality tolerance")
    elif len(blocks) < min_pairs or interval is None:
        verdict = "inconclusive"
        reasons.append(f"{len(blocks)} complete blocks; the plan needs {min_pairs}")
    elif interval[1] < min_gain:
        verdict = "fail"
        reasons.append("negative result: the interval lies below zero" if interval[1] < 0
                       else "the interval excludes the target gain")
    elif tail > tpot_p95_regression_max:
        verdict = "fail"
        reasons.append(f"p95 TPOT changes by {tail:+.2%}, beyond the {tpot_p95_regression_max:.2%} tolerance")
    elif interval[0] > 0 and gain >= min_gain:
        verdict = "pass"
    else:
        verdict = "inconclusive"
        reasons += (["the interval includes zero"] if interval[0] <= 0 else []) + (
            ["the mean gain is below the target"] if gain < min_gain else [])
    return dict(verdict=verdict, reasons=reasons, blocks=len(blocks),
                orders=sorted({b["order"] for b in blocks}),
                throughput_gain=dict(mean=gain, interval95=None if interval is None else list(interval),
                                     per_block=gains, target=min_gain),
                tpot_p95_change=dict(mean=tail, per_block=tails, tolerance=tpot_p95_regression_max),
                tpot_p50_change=dict(mean=mean(medians) if medians else None, per_block=medians),
                quality_ok=all(b["quality_ok"] for b in blocks))


def acceptance(results, *, candidate, baselines, points, min_points):
    """Primary claim: the candidate passes against every declared baseline at ``min_points`` points.

    ``fail`` when any baseline-point verdict fails; ``inconclusive`` while evidence is short.
    """
    rows, overall = {}, "pass"
    for baseline in baselines:
        verdicts = {point: results.get((point, baseline, candidate), {}).get("verdict", "not_run")
                    for point in points}
        passed = sum(verdict == "pass" for verdict in verdicts.values())
        failed = [point for point, verdict in verdicts.items() if verdict == "fail"]
        rows[baseline] = dict(verdicts=verdicts, passed_points=passed, failed_points=failed)
        if failed:
            overall = "fail"
        elif passed < min_points and overall == "pass":
            overall = "inconclusive"
    return dict(candidate=candidate, overall=overall, min_pressure_points=min_points, baselines=rows)
