"""Measure fixed tokenized workloads against SGLang's native /generate endpoint.

Each JSONL row supplies request_alias, workload, input_ids and max_new_tokens.
Keep the source JSONL with raw IDs inside the private container. Results
contain aliases, workload labels, hashes, timings and counts.
The endpoint must already be running. Record its effective execution settings
alongside the measurement and hold them fixed within each comparison group.
Inputs containing token IDs belong in the private container workspace. Output
records contain aliases, counts, timing events and hashes, not original text.

Interrupts preserve completed records but do not abort server-side requests.
An already running HTTP worker may continue until completion or a socket-read
timeout; that timeout is not a total generation deadline. The experiment's
outer lifecycle controller must handle cancellation of its own requests.

SSE framing and cumulative/delta ID handling are based on
sparse-kv-lab/vmm_elastic_pool/e2e/benchmarks/client.py.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
from queue import Empty, SimpleQueue
import time
from urllib.error import HTTPError, URLError
from http.client import HTTPException
from urllib.request import Request, urlopen


class StreamError(ValueError):
    """An invalid or unsuccessful generation response."""


def valid_ids(value):
    return isinstance(value, list) and all(type(token) is int and token >= 0 for token in value)


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def read_workload(raw):
    requests = []
    aliases = set()
    with StringIO(raw.decode("utf-8")) as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            alias, workload = row["request_alias"], row["workload"]
            budget, ids = row["max_new_tokens"], row["input_ids"]
            if (not isinstance(alias, str) or not alias or alias in aliases
                    or not isinstance(workload, str) or not workload
                    or type(budget) is not int or budget < 1 or not valid_ids(ids) or not ids):
                raise ValueError(f"Invalid workload record at line {number}")
            aliases.add(alias)
            requests.append(row)
    if not requests:
        raise ValueError("Expected at least one workload record")
    return requests


def sse_data(response):
    pending = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if pending:
                yield "\n".join(pending)
                pending.clear()
        elif line.startswith("data:"):
            pending.append(line[5:].removeprefix(" "))
    if pending:
        raise StreamError("EOF inside an SSE event")


class Generation:
    def __init__(self, alias, prompt_ids, budget):
        self.alias = alias
        self.prompt_ids = prompt_ids
        self.budget = budget
        self.count = 0
        self.finish = None
        self.events = []
        self.ids = []
        self.ids_contiguous = True
        self.text = None
        self.done = False

    def consume(self, wire, elapsed):
        if self.done:
            raise StreamError("Unexpected data after DONE")
        if wire == "[DONE]":
            self.done = True
            return
        data = json.loads(wire)
        if not isinstance(data, dict) or data.get("error") is not None:
            raise StreamError("Server returned an error response")
        meta = data.get("meta_info")
        if not isinstance(meta, dict):
            raise StreamError("Missing SSE metadata")
        count = meta.get("completion_tokens")
        if type(count) is not int or not self.count <= count <= self.budget:
            raise StreamError("Completion count regressed or exceeded the token budget")
        if "id" in meta and meta["id"] != self.alias:
            raise StreamError("Server request ID differs from the alias")
        if "prompt_tokens" in meta:
            prompt_count = meta["prompt_tokens"]
            if type(prompt_count) is not int or prompt_count != len(self.prompt_ids):
                raise StreamError("Server prompt length differs from input")
        delta = count - self.count
        if delta:
            if self.finish is not None:
                raise StreamError("New tokens arrived after the finish reason")
            self.events.append({"arrival_s": elapsed, "completion_tokens": count, "new_tokens": delta})
        self.merge_ids(data.get("output_ids"), count, delta)
        self.count = count
        text = data.get("text")
        if text is not None:
            if not isinstance(text, str):
                raise StreamError("Invalid cumulative output text")
            self.text = text
        reason = meta.get("finish_reason")
        if reason is not None:
            kind = reason.get("type") if isinstance(reason, dict) else reason
            if kind not in ("stop", "length", "abort", "error"):
                raise StreamError("Unknown finish reason")
            if self.finish is not None and self.finish != kind:
                raise StreamError("Finish reason changed during generation")
            self.finish = kind
            if kind in ("abort", "error"):
                raise StreamError("Server aborted generation")
            if kind == "length" and count != self.budget:
                raise StreamError("Length finish count differs from the requested token budget")
            if kind == "length" and isinstance(reason, dict) and "length" in reason:
                length = reason["length"]
                if type(length) is not int or length != count:
                    raise StreamError("Reported finish length differs from the completion count")

    def merge_ids(self, incoming, count, delta):
        if not valid_ids(incoming):
            if delta:
                self.ids_contiguous = False
            return
        # The legacy first SSE update can include the prompt's detokenization suffix.
        if len(self.events) == 1 and self.count == 0 and delta:
            suffix = min(5, len(self.prompt_ids))
            if len(incoming) == count + suffix and incoming[:suffix] == self.prompt_ids[-suffix:]:
                incoming = incoming[suffix:]
        if len(incoming) == count:
            if incoming[:len(self.ids)] != self.ids:
                raise StreamError("Stream output ID prefix changed")
            self.ids = incoming
            self.ids_contiguous = True
        elif len(incoming) == delta:
            if self.ids_contiguous:
                self.ids.extend(incoming)
        elif delta:
            self.ids_contiguous = False

    def metrics(self, elapsed, successful):
        first = self.events[0]["arrival_s"] if self.events else None
        last = self.events[-1]["arrival_s"] if self.events else None
        first_count = self.events[0]["new_tokens"] if self.events else None
        ids_complete = successful and self.ids_contiguous and len(self.ids) == self.count
        return {
            "output_tokens": self.count,
            "finish_reason": self.finish,
            "end_to_end_s": elapsed,
            "ttft_s": first,
            "last_token_event_s": last,
            "finish_tail_s": elapsed - last if last is not None else None,
            "tpot_s": (last - first) / (self.count - 1) if first_count == 1 and self.count > 1 else None,
            "post_first_event_s_per_token": ((last - first) / (self.count - first_count)
                                             if first_count is not None and self.count > first_count else None),
            "first_event_tokens": first_count,
            "coalesced_events": sum(event["new_tokens"] > 1 for event in self.events),
            "observed_itl_s": [
                right["arrival_s"] - left["arrival_s"]
                for left, right in zip(self.events, self.events[1:])
                if left["new_tokens"] == right["new_tokens"] == 1
            ],
            "token_events": self.events,
            "output_ids_status": "complete" if ids_complete else "unavailable",
            "output_ids_sha256": digest_ids(self.ids) if ids_complete else None,
            "output_text_sha256": (hashlib.sha256(self.text.encode()).hexdigest()
                                   if self.text is not None else None),
        }


def generate(endpoint, row, timeout, clock=time.perf_counter, *, on_start=None):
    """Measure transport from immediately before urlopen; optionally expose its origin.

    on_start receives the same monotonic sample used by token/EOF timing. It is
    for a lightweight local timing observer, not logging or server telemetry.
    Request construction is outside this interval and must be counted by an
    arrival-based caller. The original closed-loop record schema is unchanged.
    """
    ids, budget = row["input_ids"], row["max_new_tokens"]
    generation = Generation(row["request_alias"], ids, budget)
    payload = {
        "rid": row["request_alias"], "input_ids": ids, "stream": True, "return_logprob": False,
        "sampling_params": {
            "temperature": 0, "max_new_tokens": budget, "ignore_eos": False, "stream_interval": 1,
        },
    }
    request = Request(endpoint, data=json.dumps(payload).encode(), headers={
        "Content-Type": "application/json", "Accept": "text/event-stream",
    })
    record = {
        "request_alias": row["request_alias"], "workload": row["workload"],
        "input_tokens": len(ids), "input_ids_sha256": digest_ids(ids), "max_new_tokens": budget,
    }
    started = clock()
    if on_start is not None:
        on_start(started)
    try:
        with urlopen(request, timeout=timeout) as response:
            for wire in sse_data(response):
                generation.consume(wire, clock() - started)
        if generation.finish is None:
            raise StreamError("Stream ended without a finish reason")
    except HTTPError as error:
        error.close()
        record.update(error="http", http_status=error.code)
    except (URLError, OSError, HTTPException):
        record.update(error="transport")
    except (json.JSONDecodeError, UnicodeDecodeError):
        record.update(error="parse")
    except StreamError as error:
        record.update(error="protocol", error_detail=str(error))
    successful = "error" not in record
    return record | {"successful": successful} | generation.metrics(clock() - started, successful)


def summarize(records, duration, ttft_slo, tpot_slo, total_requests=None):
    total_requests = len(records) if total_requests is None else total_requests
    successful = [record for record in records if record["successful"]]
    tokens = sum(record["output_tokens"] for record in successful)
    for record in records:
        ttft, tpot = record["ttft_s"], record["tpot_s"]
        record["meets_response_slo"] = (record["successful"] and ttft is not None and tpot is not None
                                        and ttft <= ttft_slo and tpot <= tpot_slo)
        record["meets_natural_eos_slo"] = record["meets_response_slo"] and record["finish_reason"] == "stop"
    response_compliant = sum(record["meets_response_slo"] for record in records)
    natural_compliant = sum(record["meets_natural_eos_slo"] for record in records)
    finishes = Counter(record["finish_reason"] or "unavailable" for record in records)
    successful_finishes = Counter(record["finish_reason"] for record in successful)
    return {
        "makespan_s": duration, "requests": total_requests, "successes": len(successful),
        "unreturned_requests": total_requests - len(records),
        "failures": len(records) - len(successful),
        "finish_reason_counts": dict(finishes),
        "natural_eos_requests": successful_finishes["stop"],
        "length_limited_requests": successful_finishes["length"],
        "completed_output_tokens": tokens, "completed_output_tokens_per_s": tokens / duration,
        "failed_requests_observed_output_tokens": sum(record["output_tokens"] for record in records
                                                      if not record["successful"]),
        "slo": {
            "ttft_s": ttft_slo, "tpot_s": tpot_slo,
            "natural_eos_compliant_requests": natural_compliant,
            "natural_eos_compliant_fraction_all_requests": natural_compliant / total_requests,
            "natural_eos_slo_goodput_per_s": natural_compliant / duration,
            "response_compliant_requests": response_compliant,
            "response_compliant_fraction_all_requests": response_compliant / total_requests,
            "response_slo_goodput_per_s": response_compliant / duration,
        },
    }


def run(endpoint, workload, concurrency, timeout, ttft_slo, tpot_slo, on_result=None):
    if not workload or concurrency < 1:
        raise ValueError("Expected a nonempty workload and positive client concurrency")
    pool = ThreadPoolExecutor(max_workers=concurrency)
    completed = {}
    status = "complete"
    started = time.perf_counter()
    try:
        pending = {}
        ready = SimpleQueue()
        remaining = iter(enumerate(workload))

        def fill():
            while len(pending) < concurrency:
                entry = next(remaining, None)
                if entry is None:
                    break
                index, row = entry
                future = pool.submit(generate, endpoint, row, timeout)
                pending[future] = index
                future.add_done_callback(ready.put)

        fill()
        while pending:
            try:
                future = ready.get(timeout=0.25)
            except Empty:
                continue
            index = pending.pop(future)
            record = future.result()
            if on_result is not None:
                on_result(record)
            completed[index] = record
            fill()
    except KeyboardInterrupt:
        status = "partial"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    duration = time.perf_counter() - started
    records = [completed[index] for index in sorted(completed)]
    return {
        "status": status,
        "missing_request_aliases": [row["request_alias"] for index, row in enumerate(workload)
                                    if index not in completed],
        "client_concurrency": concurrency,
        "measurement_scope": {
            "concurrency": "Closed-loop in-flight client requests. GPU batch size requires server telemetry.",
            "latency": ("E2E includes connection setup and EOF. Observed TPOT spans first to last positive "
                        "token events and requires a single-token first event and multiple output tokens."),
            "itl": "Intervals between adjacent single-token SSE updates. Coalesced tokens have no observed ITL.",
            "slo": ("Natural EOS goodput is primary. Response goodput also includes successful length stops. "
                    "Both limits are inclusive. Missing metrics and failures are SLO misses."),
            "correctness": "Hashes describe the streamed response. Independent token correctness is unverified.",
            "server": "Record effective graph, overlap, radix and other server settings separately.",
        },
        "summary": summarize(records, duration, ttft_slo, tpot_slo, len(workload)),
        "requests": records,
    }


def positive_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:31080/generate")
    parser.add_argument("--workload", type=Path, required=True, help="JSONL with fixed token IDs and aliases")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-results", type=Path,
                        help="Completion journal. Defaults to OUTPUT with suffix .request-results.jsonl")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--timeout", type=positive_float, default=600)
    parser.add_argument("--ttft-slo-s", type=positive_float, required=True)
    parser.add_argument("--tpot-slo-s", type=positive_float, required=True)
    args = parser.parse_args(argv)
    journal_path = args.request_results or args.output.with_suffix(".request-results.jsonl")
    if args.concurrency < 1 or args.output.resolve() == journal_path.resolve():
        parser.error("Use positive concurrency and distinct output paths")
    if args.output.exists() or journal_path.exists():
        parser.error("Use fresh output and request-results paths")
    workload_source = args.workload.read_bytes()
    source_sha256 = {
        "workload": hashlib.sha256(workload_source).hexdigest(),
        "client": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    workload = read_workload(workload_source)
    try:
        with journal_path.open("x", encoding="utf-8") as journal:
            with args.output.open("x", encoding="utf-8") as output:
                declaration = {"status": "partial", "missing_request_aliases": [
                    row["request_alias"] for row in workload
                ], "source_sha256": source_sha256}
                json.dump(declaration, output, indent=2)
                output.write("\n")
                output.flush()

                def append(record):
                    journal.write(json.dumps(record) + "\n")
                    journal.flush()

                result = run(args.url, workload, args.concurrency, args.timeout,
                             args.ttft_slo_s, args.tpot_slo_s, on_result=append)
                result["source_sha256"] = source_sha256
                output.seek(0)
                json.dump(result, output, indent=2)
                output.write("\n")
                output.truncate()
    except FileExistsError:
        parser.error("Use fresh output and request-results paths")
    if result.get("status") == "partial":
        raise SystemExit(130)


if __name__ == "__main__":
    main()
