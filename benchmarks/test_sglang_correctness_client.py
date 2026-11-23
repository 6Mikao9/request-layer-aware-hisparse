from contextlib import redirect_stderr
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
import json
import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

from benchmarks import sglang_correctness_client as client
from benchmarks.sglang_stream_client import digest_ids


def row(alias):
    return {"request_alias": alias, "workload": "example", "input_ids": [10, 11], "max_new_tokens": 2}


class CorrectnessClientTest(unittest.TestCase):
    def test_fake_http_correctness_payload_and_anonymous_results(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append((self.path, body))
                alias = body["rid"]
                if alias == "http-error":
                    self.send_error(503)
                    return
                meta = {"id": alias, "prompt_tokens": 2, "completion_tokens": 2,
                        "finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.1, 42], [-0.2, 43]]}
                reply = {"meta_info": meta, "text": "private response text"}
                if alias == "both":
                    reply["output_ids"] = [42, 43]
                elif alias == "disagree":
                    reply["output_ids"] = [42, 44]
                elif alias == "clipped":
                    meta.update(completion_tokens=1, finish_reason={"type": "length", "length": 1},
                                output_token_logprobs=[[-0.1, 42]])
                elif alias == "budget-stop":
                    meta["finish_reason"] = {"type": "length", "length": 2}
                elif alias == "abort":
                    meta["finish_reason"] = {"type": "abort"}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(reply).encode())

            def log_message(self, *_args):
                pass

        with TemporaryDirectory() as directory, ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            source = Path(directory) / "workload.jsonl"
            output = Path(directory) / "report.json"
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            aliases = ["fallback", "both", "budget-stop", "disagree", "clipped", "abort", "http-error"]
            try:
                raw = "".join(json.dumps(row(alias)) + "\n" for alias in aliases).encode()
                source.write_bytes(raw)
                args = ["--workload", str(source), "--output", str(output), "--timeout", "3",
                        "--url", f"http://127.0.0.1:{server.server_port}/generate"]
                self.assertEqual(client.main(args), 1)
                report_text = output.read_text(encoding="utf-8")
                report = json.loads(report_text)
                records = report["requests"]
                journal = output.with_suffix(".request-results.jsonl").read_text(encoding="utf-8")
                self.assertEqual([json.loads(line) for line in journal.splitlines()], records)
                self.assertNotIn("private-output-ids.jsonl", {path.name for path in Path(directory).iterdir()})
                private = Path(directory) / "private-output-ids.jsonl"
                opt_in_output = Path(directory) / "opt-in.json"
                args[3] = str(opt_in_output)
                self.assertEqual(client.main(args + ["--private-output-ids", str(private)]), 1)
                opt_in_records = json.loads(opt_in_output.read_text(encoding="utf-8"))["requests"]
                self.assertEqual(opt_in_records, records)
                exact_bytes = private.read_bytes()
                private_records = [json.loads(line) for line in exact_bytes.splitlines()]
                self.assertEqual(private_records, [
                    {"request_alias": entry["request_alias"], "input_ids_sha256": digest_ids([10, 11]),
                     "output_ids": [42, 43], "finish_reason": entry["finish_reason"]}
                    for entry in records[:3]
                ])
                if os.name == "posix":
                    self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o600)
                args[3] = str(Path(directory) / "refused.json")
                for path in (private, Path(args[3])):
                    with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                        client.main(args + ["--private-output-ids", str(path)])
                self.assertEqual(private.read_bytes(), exact_bytes)
                self.assertFalse(Path(args[3]).exists())
                source.write_bytes(b"".join(raw.splitlines(keepends=True)[:3]))
                args[3] = str(Path(directory) / "all-ok.json")
                self.assertEqual(client.main(args), 0)
                self.assertEqual(json.loads(Path(args[3]).read_text(encoding="utf-8"))["status"], "complete")
            finally:
                server.shutdown()
                thread.join()

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["source_sha256"]["workload"], sha256(raw).hexdigest())
        self.assertEqual(report["summary"], {"requests": 7, "failures": 4, "unreturned_requests": 0})
        self.assertNotIn("private response text", report_text + journal)
        self.assertTrue(all("output_ids" not in record for record in records))
        self.assertEqual([record["request_alias"] for record in records], aliases)
        self.assertEqual([record["output_ids_source"] for record in records[:3]],
                         ["output_token_logprobs", "output_ids", "output_token_logprobs"])
        for record in records[:3]:
            self.assertEqual(record["output_ids_sha256"], digest_ids([42, 43]))
            self.assertEqual(record["output_tokens"], 2)
        self.assertEqual([record["finish_reason"] for record in records[:3]], ["stop", "stop", "length"])
        self.assertIn("disagree", records[3]["error_detail"])
        self.assertIn("budget", records[4]["error_detail"])
        self.assertIn("finish reason", records[5]["error_detail"])
        self.assertEqual(records[6]["http_status"], 503)
        self.assertEqual([body["rid"] for _, body in received], aliases + aliases + aliases[:3])
        for path, body in received:
            self.assertEqual(path, "/generate")
            self.assertEqual(body["input_ids"], [10, 11])
            self.assertFalse(body["stream"])
            self.assertTrue(body["return_logprob"])
            self.assertEqual((body["logprob_start_len"], body["top_logprobs_num"]), (-1, 0))
            self.assertEqual(body["sampling_params"], {"temperature": 0, "max_new_tokens": 2, "ignore_eos": False})

    def test_rejects_invalid_or_missing_id_evidence(self):
        info = {"id": "R01", "prompt_tokens": 2, "completion_tokens": 1, "finish_reason": {"type": "stop"}}
        for mutation, message in (({}, "Missing or invalid exact"), ({"id": "R02"}, "identity"),
                                  ({"output_token_logprobs": [[-0.1, True]]}, "Invalid output token IDs"),
                                  ({"completion_tokens": 2, "output_token_logprobs": [[-0.1, 42]]}, "token count")):
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, message):
                client.response_ids({"meta_info": info | mutation}, row("R01"))


if __name__ == "__main__":
    unittest.main()
