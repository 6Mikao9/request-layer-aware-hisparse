from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import io
import time
from pathlib import Path
from queue import Empty, SimpleQueue
from tempfile import TemporaryDirectory
from threading import Event, Thread
import unittest
from unittest.mock import patch

from . import sglang_arrival_client as client
from . import test_sglang_stream_client as fixtures


@contextmanager
def endpoint(release, entered, finished=None):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            alias = body["rid"]
            if alias == "first":
                entered.set()
                release.wait(3)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                finish = "length" if alias == "second" else "stop"
                self.wfile.write(fixtures.event(1, ids=[7777], text="private-answer"))
                self.wfile.write(fixtures.event(2, {"type": finish}, [8888], "private-answer"))
                self.wfile.write(fixtures.DONE)
            finally:
                if alias == "first" and finished is not None:
                    finished.set()

        def log_message(self, *_):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/generate"
        finally:
            release.set()
            server.shutdown()
            thread.join(timeout=3)


class ArrivalClientTests(unittest.TestCase):
    def setUp(self):
        self.rows = [fixtures.workload(alias, budget=2) for alias in ("first", "second")]
        self.raw = ("\n".join(json.dumps(row) for row in self.rows) + "\n").encode()
        self.schedule = dict(schema=client.SCHEMA, workload_sha256=sha256(self.raw).hexdigest(),
            phases=[dict(name="low", start_s=0, end_s=.01),
                    dict(name="high", start_s=.01, end_s=.03)],
            arrivals=[dict(request_alias="first", at_s=0, phase="low"),
                      dict(request_alias="second", at_s=.02, phase="high")])

    def run_client(self, url):
        return client.run(url, self.rows, self.schedule, 1, 3, 3, 3)

    def test_fixed_arrival_is_submitted_while_first_http_worker_still_busy(self):
        entered, release, finished = Event(), Event(), Event()
        submitted_while_in_flight = []

        class ObservedExecutor(ThreadPoolExecutor):
            def submit(self, fn, event, state):
                if event["request_alias"] == "second":
                    submitted_while_in_flight.append(entered.wait(1) and not finished.is_set())
                    result = super().submit(fn, event, state)
                    release.set()
                    return result
                return super().submit(fn, event, state)

        with endpoint(release, entered, finished) as url, patch.object(client, "ThreadPoolExecutor", ObservedExecutor):
            report = self.run_client(url)
        self.assertEqual(submitted_while_in_flight, [True])
        first, second = report["requests"]
        self.assertGreater(second["arrival_timing"]["executor_queue_s"], 0)
        self.assertLess(second["arrival_timing"]["submitted_s"], first["arrival_timing"]["completed_s"])
        self.assertEqual([request["finish_reason"] for request in report["requests"]], ["stop", "length"])
        self.assertEqual(report["summary"]["arrival_slo"]["natural_eos_compliant_fraction_all_requests"], .5)
        self.assertEqual([cohort["requests"] for cohort in report["summary"]["phase_arrival_cohorts"]], [1, 1])
        high_latency = report["summary"]["phase_arrival_cohorts"][1]["arrival_latency"]
        self.assertEqual(high_latency["ttft_s"]["p99"], second["arrival_timing"]["arrival_ttft_s"])
        self.assertEqual(high_latency["tpot_s"]["p50"], second["tpot_s"])
        self.assertEqual(report["summary"]["arrival_latency"]["ttft_s"]["included_successes"], 2)

    def test_clock_splits_arrival_latency_and_leaves_transport_tpot_intact(self):
        row = dict(ttft_s=3, tpot_s=.5, end_to_end_s=6)
        output = client.arrival_record(row, dict(phase="low", at_s=1),
            dict(submitted_s=2, worker_started_s=4, transport_started_s=7, worker_finished_s=15))
        timing = output["arrival_timing"]
        self.assertEqual((timing["dispatch_lateness_s"], timing["executor_queue_s"],
                          timing["request_preparation_s"]), (1, 2, 3))
        self.assertEqual((timing["first_token_s"], timing["completed_s"]), (10, 13))
        self.assertEqual((timing["arrival_ttft_s"], timing["arrival_end_to_end_s"]), (9, 12))
        self.assertEqual(output["tpot_s"], .5)
        self.assertEqual(output["ttft_s"], 3)

    def test_observer_uses_original_transport_clock_sample(self):
        starts = []
        data = fixtures.event(1, ids=[11])+fixtures.event(2, {"type": "stop"}, [12])+fixtures.DONE
        with patch.object(client.stream, "urlopen", return_value=io.BytesIO(data)):
            record = client.stream.generate("http://localhost/generate", self.rows[0], 3,
                clock=iter([100, 101, 102, 103, 104]).__next__, on_start=starts.append)
        self.assertEqual(starts, [100])
        self.assertEqual((record["ttft_s"], record["tpot_s"], record["end_to_end_s"]), (1, 1, 4))

    def test_completed_makespan_excludes_journal_collection_tail(self):
        self.rows = self.rows[:1]
        self.schedule["arrivals"] = self.schedule["arrivals"][:1]
        release, entered = Event(), Event()
        release.set()
        with endpoint(release, entered) as url:
            result = client.run(url, self.rows, self.schedule, 1, 3, 3, 3,
                                on_result=lambda _: time.sleep(.1))
        expected = max(.03, result["requests"][0]["arrival_timing"]["completed_s"])
        self.assertEqual(result["summary"]["observed_makespan_s"], expected)
        self.assertGreater(result["client_loop_elapsed_s"]-expected, .06)

    def test_interrupt_during_submit_does_not_call_started_http_unsubmitted(self):
        entered, release, finished = Event(), Event(), Event()
        class InterruptingExecutor(ThreadPoolExecutor):
            def submit(self, *args, **kwargs):
                super().submit(*args, **kwargs)
                if not entered.wait(1):
                    raise AssertionError("HTTP request did not start")
                raise KeyboardInterrupt
        with endpoint(release, entered, finished) as url, patch.object(client, "ThreadPoolExecutor", InterruptingExecutor):
            result = self.run_client(url)
            self.assertEqual(result["status"], "partial")
            self.assertEqual([r["state"] for r in result["request_lifecycle"]],
                             ["in_flight_or_unobserved", "not_submitted"])
            release.set()
            self.assertTrue(finished.wait(2))

    def test_horizon_wait_has_no_extra_quarter_second_poll_tail(self):
        self.rows = self.rows[:1]
        self.schedule["arrivals"] = self.schedule["arrivals"][:1]
        now = [0.0]
        class ImmediatePool:
            def __init__(self, **_):
                pass
            def submit(self, fn, *args):
                future = Future()
                future.set_result(fn(*args))
                return future
            def shutdown(self, **_):
                pass
        class TimedQueue(SimpleQueue):
            def get(self, *, timeout):
                try:
                    return super().get_nowait()
                except Empty:
                    now[0] += timeout
                    raise
        def fake_generate(_endpoint, row, _timeout, *, on_start):
            on_start(now[0])
            return dict(request_alias=row["request_alias"], successful=True, finish_reason="stop",
                        output_tokens=2, ttft_s=0, tpot_s=0, end_to_end_s=0)
        with patch.object(client, "ThreadPoolExecutor", ImmediatePool), patch.object(client, "SimpleQueue", TimedQueue), \
                patch.object(client.time, "perf_counter", side_effect=lambda: now[0]), \
                patch.object(client.stream, "generate", side_effect=fake_generate):
            result = self.run_client("unused")
        self.assertEqual(result["client_loop_elapsed_s"], .03)
        self.assertEqual(result["summary"]["observed_makespan_s"], .03)

    def test_interrupt_before_api_start_is_not_a_returned_server_failure(self):
        self.rows = self.rows[:1]
        self.schedule["arrivals"] = self.schedule["arrivals"][:1]
        prepared, release, sent = Event(), Event(), Event()
        class WaitForWorkerPool(ThreadPoolExecutor):
            def shutdown(self, **_):
                release.set()
                return super().shutdown(wait=True, cancel_futures=True)
        class InterruptQueue(SimpleQueue):
            def get(self, *, timeout):
                if not prepared.wait(1):
                    raise AssertionError("Worker did not prepare request")
                raise KeyboardInterrupt
        def fake_generate(_endpoint, _row, _timeout, *, on_start):
            prepared.set()
            if not release.wait(2):
                raise AssertionError("Shutdown did not release worker")
            on_start(time.perf_counter())
            sent.set()
            raise AssertionError("Cancelled worker must not send")
        with patch.object(client, "ThreadPoolExecutor", WaitForWorkerPool), \
                patch.object(client, "SimpleQueue", InterruptQueue), \
                patch.object(client.stream, "generate", side_effect=fake_generate):
            result = self.run_client("unused")
        self.assertFalse(sent.is_set())
        self.assertEqual(result["requests"], [])
        self.assertEqual(result["request_lifecycle"][0]["state"], "not_sent_cancelled")

    def test_schedule_pins_hash_phase_and_each_unique_request(self):
        digest = self.schedule["workload_sha256"]
        self.assertEqual(client.read_schedule(json.dumps(self.schedule), self.rows, digest), self.schedule)
        mutations = (
            lambda s: s.update(workload_sha256="0" * 64),
            lambda s: s["arrivals"].pop(),
            lambda s: s["arrivals"][1].update(request_alias="first"),
            lambda s: s["phases"][1].update(start_s=.015),
            lambda s: s["arrivals"][1].update(at_s=.03),
            lambda s: s["arrivals"].reverse(),
        )
        for mutate in mutations:
            data = deepcopy(self.schedule)
            mutate(data)
            with self.assertRaises(ValueError):
                client.read_schedule(json.dumps(data), self.rows, digest)

    def test_cli_writes_fresh_report_and_journal_without_raw_payload(self):
        release, entered = Event(), Event()
        release.set()
        with TemporaryDirectory() as directory, endpoint(release, entered) as url:
            root = Path(directory).resolve()
            workload, schedule = root / "input.jsonl", root / "schedule.json"
            workload.write_bytes(self.raw)
            schedule.write_text(json.dumps(self.schedule), encoding="utf-8")
            output = root / "report.json"
            args = ["--url", url, "--workload", str(workload), "--schedule", str(schedule),
                    "--output", str(output), "--max-workers", "1", "--timeout", "3",
                    "--ttft-slo-s", "3", "--tpot-slo-s", "3"]
            self.assertEqual(client.main(args), 0)
            report = output.read_text(encoding="utf-8")
            journal = output.with_suffix(".request-results.jsonl").read_text(encoding="utf-8")
            self.assertEqual(len(journal.splitlines()), 2)
            self.assertEqual(json.loads(report)["source_sha256"]["workload"], sha256(self.raw).hexdigest())
            for secret in ('"input_ids":', '"output_ids":', "private-answer", "[7777]", "[8888]"):
                self.assertNotIn(secret, report + journal)
            with self.assertRaises(FileExistsError):
                client.main(args)
            self.assertEqual(output.read_text(encoding="utf-8"), report)

    def test_interrupt_classifies_running_queued_and_unsubmitted_requests(self):
        self.rows.append(fixtures.workload("late"))
        self.schedule["phases"][0]["end_s"] = 1
        self.schedule["phases"][1].update(start_s=1, end_s=3)
        self.schedule["arrivals"][1].update(at_s=0, phase="low")
        self.schedule["arrivals"].append(dict(request_alias="late", at_s=2, phase="high"))
        entered, release, finished = Event(), Event(), Event()

        class InterruptAfterDispatch:
            def put(self, _):
                pass

            def get(self, *, timeout):
                if not entered.wait(1):
                    raise AssertionError("HTTP worker did not start")
                raise KeyboardInterrupt

        with endpoint(release, entered, finished) as url, patch.object(client, "SimpleQueue", InterruptAfterDispatch):
            report = self.run_client(url)
            self.assertEqual(report["status"], "partial")
            self.assertFalse(finished.is_set())
            states = [row["state"] for row in report["request_lifecycle"]]
            self.assertEqual(states, ["in_flight_or_unobserved", "not_sent_cancelled", "not_submitted"])
            self.assertEqual(report["summary"]["arrival_slo"]["natural_eos_compliant_fraction_all_requests"], 0)
            self.assertEqual(sum(phase["requests"] for phase in report["summary"]["phase_arrival_cohorts"]), 3)
            release.set()
            self.assertTrue(finished.wait(2))
