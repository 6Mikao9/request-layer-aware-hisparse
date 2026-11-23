"""Hash full non-streamed responses using the audited MiniMax logprob request."""

import argparse
from contextlib import nullcontext
from hashlib import sha256
from http.client import HTTPException
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from benchmarks.sglang_stream_client import digest_ids, positive_float, read_workload, valid_ids


def response_ids(result, row):
    if not isinstance(result, dict) or result.get("error") is not None:
        raise ValueError("Server returned an error response")
    info = result.get("meta_info")
    if not isinstance(info, dict):
        raise ValueError("Missing response metadata")
    if (info.get("id") != row["request_alias"] or type(info.get("prompt_tokens")) is not int
            or info["prompt_tokens"] != len(row["input_ids"])):
        raise ValueError("Response request identity or prompt count differs")

    direct = result.get("output_ids")
    logprobs = info.get("output_token_logprobs")
    fallback = None
    if logprobs is not None:
        if not isinstance(logprobs, list) or any(not isinstance(entry, list) or len(entry) < 2 for entry in logprobs):
            raise ValueError("Invalid output token logprob entries")
        fallback = [entry[1] for entry in logprobs]
        if not valid_ids(fallback):
            raise ValueError("Invalid output token IDs in logprobs")
    ids = direct if direct is not None else fallback
    if not valid_ids(ids):
        raise ValueError("Missing or invalid exact output token IDs")
    if direct is not None and fallback is not None and direct != fallback:
        raise ValueError("Direct and logprob output token IDs disagree")
    budget = row["max_new_tokens"]
    if type(info.get("completion_tokens")) is not int or len(ids) != info["completion_tokens"] or len(ids) > budget:
        raise ValueError("Output token count differs from metadata or exceeds the request budget")

    reason = info.get("finish_reason")
    kind = reason.get("type") if isinstance(reason, dict) else reason
    if kind not in ("stop", "length"):
        raise ValueError("Missing or unsuccessful finish reason")
    if kind == "length" and (len(ids) != budget or not isinstance(reason, dict)
                             or type(reason.get("length")) is not int or reason["length"] != len(ids)):
        raise ValueError("Length finish differs from the token count or request budget")
    return ids, kind, "output_ids" if direct is not None else "output_token_logprobs"


def submit(opener, url, row, timeout, private_ledger=None):
    body = {"rid": row["request_alias"], "input_ids": row["input_ids"], "stream": False,
            "return_logprob": True, "logprob_start_len": -1, "top_logprobs_num": 0,
            "sampling_params": {"temperature": 0, "max_new_tokens": row["max_new_tokens"], "ignore_eos": False}}
    request = Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    record = {"request_alias": row["request_alias"], "workload": row["workload"],
              "input_tokens": len(row["input_ids"]), "input_ids_sha256": digest_ids(row["input_ids"]),
              "max_new_tokens": row["max_new_tokens"]}
    try:
        with opener.open(request, timeout=timeout) as response:
            result = json.load(response)
        ids, finish, source = response_ids(result, row)
    except HTTPError as error:
        error.close()
        detail = {"error": "http", "http_status": error.code}
    except (URLError, OSError, HTTPException):
        detail = {"error": "transport"}
    except (json.JSONDecodeError, UnicodeDecodeError):
        detail = {"error": "parse"}
    except ValueError as error:
        detail = {"error": "protocol", "error_detail": str(error)}
    else:
        if private_ledger is not None:
            private_ledger.write(json.dumps({"request_alias": record["request_alias"],
                                             "input_ids_sha256": record["input_ids_sha256"],
                                             "output_ids": ids, "finish_reason": finish}) + "\n")
            private_ledger.flush()
        return record | {"successful": True, "output_ids_sha256": digest_ids(ids), "output_tokens": len(ids),
                         "finish_reason": finish, "output_ids_source": source}
    return record | {"successful": False} | detail


def private_create(path, flags):
    return os.open(path, flags, 0o600)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:31080/generate")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-output-ids", type=Path, help="Fresh private JSONL for exact output token IDs")
    parser.add_argument("--timeout", type=positive_float, default=1800)
    args = parser.parse_args(argv)
    raw = args.workload.read_bytes()
    rows = read_workload(raw)
    source = {"workload": sha256(raw).hexdigest(), "client": sha256(Path(__file__).read_bytes()).hexdigest()}
    journal = args.output.with_suffix(".request-results.jsonl")
    destinations = [args.output, journal]
    if args.private_output_ids is not None:
        destinations.append(args.private_output_ids)
    if len({path.resolve() for path in destinations}) != len(destinations) or any(path.exists() for path in destinations):
        parser.error("Use fresh and distinct output, journal and private evidence paths")
    opener = build_opener(ProxyHandler({}))
    records = []
    status = "partial"
    private = (open(args.private_output_ids, "x", encoding="utf-8", opener=private_create)
               if args.private_output_ids is not None else nullcontext())
    with (private as private_ledger,
          open(journal, "x", encoding="utf-8", buffering=1, opener=private_create) as ledger,
          open(args.output, "x", encoding="utf-8", opener=private_create) as output):
        json.dump({"status": "partial", "source_sha256": source}, output)
        output.flush()
        try:
            for row in rows:
                record = submit(opener, args.url, row, args.timeout, private_ledger)
                ledger.write(json.dumps(record) + "\n")
                records.append(record)
            status = "complete" if all(record["successful"] for record in records) else "failed"
        finally:
            output.seek(0)
            json.dump({"status": status, "source_sha256": source, "requests": records,
                       "missing_request_aliases": [row["request_alias"] for row in rows[len(records):]],
                       "summary": {"requests": len(rows), "failures": sum(not row["successful"] for row in records),
                                   "unreturned_requests": len(rows) - len(records)}}, output, indent=2)
            output.write("\n")
            output.truncate()
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
