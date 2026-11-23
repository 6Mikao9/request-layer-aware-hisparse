import io
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import tempfile
from pathlib import Path
from queue import SimpleQueue
import time
import unittest
from unittest.mock import patch

from benchmarks import sglang_stream_client as client
from benchmarks.sglang_stream_client import digest_ids, generate, run, summarize


def event(count, reason=None, ids=None, text="", **metadata):
    data = {"text": text, "meta_info": {"completion_tokens": count, "finish_reason": reason, **metadata}}
    if ids is not None:
        data["output_ids"] = ids
    return ("data: " + json.dumps(data) + "\n\n").encode()


DONE = b"data: [DONE]\n\n"


def workload(alias, budget=4):
    return {"request_alias": alias, "workload": "long", "input_ids": [9, 8], "max_new_tokens": budget}


class StreamClientTest(unittest.TestCase):
    def test_closed_loop_all_rows_and_slo_denominator(self):
        lock = threading.Lock()
        active = peak = 0
        submitted = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                nonlocal active, peak
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with lock:
                    submitted.append((self.path, body))
                    active += 1
                    peak = max(peak, active)
                try:
                    time.sleep(0.025)
                    alias = body["rid"]
                    if alias == "http":
                        self.send_error(503)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    if alias == "parse":
                        self.wfile.write(b"data: {bad\n\n")
                    else:
                        self.wfile.write(event(1, ids=[11], text="a"))
                        self.wfile.write(event(2, {"type": "stop"}, [12], "ab"))
                        self.wfile.write(DONE)
                finally:
                    with lock:
                        active -= 1

            def log_message(self, *_args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                aliases = ("one", "two", "http", "parse", "five")
                result = run(f"http://127.0.0.1:{server.server_port}/generate",
                             [workload(alias) for alias in aliases], 2, 3, 10, 10)
            finally:
                server.shutdown()
                thread.join()

        self.assertEqual(peak, 2)
        self.assertEqual([record["request_alias"] for record in result["requests"]], list(aliases))
        self.assertEqual({body["rid"] for _, body in submitted}, set(aliases))
        for path, body in submitted:
            self.assertEqual(path, "/generate")
            self.assertEqual(body["input_ids"], [9, 8])
            self.assertFalse(body["return_logprob"])
            self.assertFalse(body["sampling_params"]["ignore_eos"])
        summary = result["summary"]
        self.assertEqual((summary["successes"], summary["failures"], summary["completed_output_tokens"]),
                         (3, 2, 6))
        self.assertEqual(summary["slo"]["natural_eos_compliant_fraction_all_requests"], 3 / 5)
        self.assertEqual(summary["slo"]["natural_eos_slo_goodput_per_s"], 3 / summary["makespan_s"])
        self.assertEqual(summary["completed_output_tokens_per_s"], 6 / summary["makespan_s"])
        first, _, http, parse, _ = result["requests"]
        self.assertEqual(first["output_ids_sha256"], digest_ids([11, 12]))
        self.assertEqual(first["input_ids_sha256"], digest_ids([9, 8]))
        self.assertEqual(first["output_text_sha256"], hashlib.sha256(b"ab").hexdigest())
        self.assertEqual(first["finish_reason"], "stop")
        self.assertEqual((http["error"], http["http_status"], parse["error"]), ("http", 503, "parse"))

    def from_wire(self, chunks, times, budget=4):
        with patch("benchmarks.sglang_stream_client.urlopen", return_value=io.BytesIO(b"".join(chunks))):
            return generate("http://localhost/generate", workload("R01", budget), 3,
                            clock=iter(times).__next__)

    def test_finish_tail_is_not_in_tpot_and_length_requires_budget(self):
        frames = [event(1, ids=[11]), event(2, ids=[11, 12]), event(2, {"type": "length"}), DONE]
        record = self.from_wire(frames, [0, 1, 3, 5, 6, 8], budget=2)
        self.assertTrue(record["successful"])
        self.assertEqual(record["tpot_s"], 2)
        self.assertEqual(record["observed_itl_s"], [2])
        self.assertEqual(record["finish_tail_s"], 5)
        self.assertEqual(record["end_to_end_s"], 8)
        mismatch = self.from_wire(frames, [0, 1, 3, 5, 6], budget=3)
        self.assertFalse(mismatch["successful"])
        self.assertEqual(mismatch["error"], "protocol")
        self.assertEqual(mismatch["output_tokens"], 2)

    def test_zero_and_one_token_have_no_tpot_or_itl(self):
        for count in (0, 1):
            with self.subTest(count=count):
                record = self.from_wire([event(count, {"type": "stop"}), DONE], [0, 1, 2, 3])
                self.assertTrue(record["successful"])
                self.assertEqual(record["output_tokens"], count)
                self.assertEqual(record["ttft_s"], 1 if count else None)
                self.assertIsNone(record["tpot_s"])
                self.assertEqual(record["observed_itl_s"], [])
                self.assertEqual(summarize([record], 3, 10, 10)["slo"]["response_compliant_requests"], 0)
                if count:
                    self.assertEqual(record["output_ids_status"], "unavailable")

    def test_coalesced_events_never_fabricate_tpot_or_itl(self):
        record = self.from_wire([
            event(2, ids=[12]), event(3, ids=[13]), event(4, {"type": "stop"}, [14]), DONE,
        ], [0, 1, 2, 4, 5, 7])
        self.assertTrue(record["successful"])
        self.assertEqual(record["first_event_tokens"], 2)
        self.assertEqual(record["coalesced_events"], 1)
        self.assertIsNone(record["tpot_s"])
        self.assertEqual(record["post_first_event_s_per_token"], 1.5)
        self.assertEqual(record["observed_itl_s"], [2])
        self.assertEqual(record["output_ids_status"], "unavailable")
        all_at_once = self.from_wire([event(4, {"type": "length"}, [11, 12, 13, 14]), DONE],
                                     [0, 1, 2, 3])
        self.assertIsNone(all_at_once["tpot_s"])
        self.assertIsNone(all_at_once["post_first_event_s_per_token"])
        self.assertEqual(all_at_once["output_ids_status"], "complete")
        self.assertEqual(summarize([all_at_once], 3, 10, 10)["slo"]["response_compliant_requests"], 0)

    def test_contradictory_ids_fail_even_if_a_later_event_would_restore_ids(self):
        for intermediate in ([], [event(2)]):
            with self.subTest(gap=bool(intermediate)):
                count = 3 if intermediate else 2
                frames = [event(1, ids=[11]), *intermediate,
                          event(count, ids=[99, 12, 13][:count]),
                          event(4, {"type": "stop"}, [11, 12, 13, 14]), DONE]
                record = self.from_wire(frames, range(10))
                self.assertEqual(record["error"], "protocol")
                self.assertIn("prefix changed", record["error_detail"])
                self.assertEqual(record["output_ids_status"], "unavailable")

    def test_optional_response_identity_and_length_fields(self):
        start = event(1, ids=[11], id="R01", prompt_tokens=2)
        natural = self.from_wire([start, event(2, {"type": "stop"}, [12]), DONE], range(6), budget=2)
        length = self.from_wire([start, event(2, {"type": "length", "length": 2}, [12]), DONE],
                                range(6), budget=2)
        failures = [
            event(2, {"type": "stop"}, id="another"),
            event(2, {"type": "stop"}, prompt_tokens=2.0),
            event(2, {"type": "stop"}, prompt_tokens=3),
            event(2, {"type": "length", "length": 2.0}),
            event(2, {"type": "length", "length": 1}),
        ]
        for frame in failures:
            with self.subTest(frame=frame):
                record = self.from_wire([start, frame, DONE], range(6), budget=2)
                self.assertEqual(record["error"], "protocol")
        summary = summarize([natural, length], 5, 10, 10)
        self.assertEqual(summary["finish_reason_counts"], {"stop": 1, "length": 1})
        self.assertEqual((summary["natural_eos_requests"], summary["length_limited_requests"]), (1, 1))
        self.assertEqual(summary["completed_output_tokens"], 4)
        self.assertEqual(summary["slo"]["natural_eos_slo_goodput_per_s"], 1 / 5)
        self.assertEqual(summary["slo"]["response_slo_goodput_per_s"], 2 / 5)

    def test_cli_hashes_exact_workload_bytes_and_client_source(self):
        raw = (json.dumps(workload("R01")) + "\r\n").encode()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "workload.jsonl"
            destination = Path(folder) / "result.json"
            source.write_bytes(raw)
            with patch.object(client, "run", return_value={}) as measured:
                client.main(["--workload", str(source), "--output", str(destination),
                             "--concurrency", "2", "--ttft-slo-s", "1", "--tpot-slo-s", "0.1"])
            self.assertEqual(measured.call_args.args[1], [workload("R01")])
            report = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(report["source_sha256"], {
            "workload": hashlib.sha256(raw).hexdigest(),
            "client": hashlib.sha256(Path(client.__file__).read_bytes()).hexdigest(),
        })

    def test_journal_flushes_later_input_first_and_report_restores_order(self):
        first_can_finish = threading.Event()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "workload.jsonl"
            destination = Path(folder) / "result.json"
            journal = destination.with_suffix(".request-results.jsonl")
            source.write_text("".join(json.dumps(workload(alias)) + "\n"
                                      for alias in ("slow", "fast")), encoding="utf-8")

            def request(_endpoint, row, _timeout):
                if row["request_alias"] == "slow":
                    self.assertTrue(first_can_finish.wait(3))
                return self.from_wire([event(1, {"type": "stop"}), DONE], range(5)) | {
                    "request_alias": row["request_alias"],
                }

            def watch():
                for _ in range(100):
                    if journal.exists() and journal.stat().st_size:
                        first_can_finish.set()
                        return
                    time.sleep(0.02)

            observer = threading.Thread(target=watch)
            observer.start()
            try:
                with patch.object(client, "generate", side_effect=request):
                    client.main(["--workload", str(source), "--output", str(destination),
                                 "--concurrency", "2", "--ttft-slo-s", "1", "--tpot-slo-s", "0.1"])
            finally:
                observer.join()
            records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
            report = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual([row["request_alias"] for row in records], ["fast", "slow"])
        self.assertEqual([row["request_alias"] for row in report["requests"]], ["slow", "fast"])

    def test_interrupt_preserves_journal_and_marks_missing_requests(self):
        class InterruptAfterFirst(SimpleQueue):
            calls = 0

            def get(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise KeyboardInterrupt
                return super().get(*args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "workload.jsonl"
            destination = Path(folder) / "result.json"
            journal = Path(folder) / "journal.jsonl"
            source.write_text("".join(json.dumps(workload(alias)) + "\n"
                                      for alias in ("first", "missing")), encoding="utf-8")
            record = self.from_wire([event(1, {"type": "stop"}), DONE], range(5))
            with patch.object(client, "SimpleQueue", InterruptAfterFirst), patch.object(
                client, "generate", side_effect=lambda _endpoint, row, _timeout: (
                    record | {"request_alias": row["request_alias"]}
                )
            ), self.assertRaises(SystemExit) as stopped:
                client.main(["--workload", str(source), "--output", str(destination),
                             "--request-results", str(journal), "--concurrency", "1",
                             "--ttft-slo-s", "1", "--tpot-slo-s", "0.1"])
            report = json.loads(destination.read_text(encoding="utf-8"))
            records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(stopped.exception.code, 130)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["missing_request_aliases"], ["missing"])
        self.assertEqual(report["summary"]["unreturned_requests"], 1)
        self.assertEqual(report["summary"]["successes"], 1)
        self.assertEqual([row["request_alias"] for row in records], ["first"])


if __name__ == "__main__":
    unittest.main()
