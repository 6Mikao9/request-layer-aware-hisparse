"""CPU end-to-end test of the combined-study runner against an emulated SGLang server.

The emulator keeps the contracts the runner reads: /health, /server_info, streamed and
non-streamed /generate in SGLang's response format, lockstep decode with SGLang's decode
log line, and per-rank receipt files of the experimental arms. Arms differ only in step
time, a running-batch cap or corrupted outputs, so every runner decision is exercised.
"""
import contextlib
import copy
import csv
import datetime
import gc
import hashlib
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from runtime import combined_measure as measure
from runtime import combined_plan as planner
from runtime import combined_run as runner
from runtime import study_config as sc
from runtime import test_combined_timeline as nsys_export
from runtime.test_combined_plan import SMALL

TP = SMALL["geometry"]["tp_size"]
LOG = ("[2026-11-25 10:00:00 TP0] Decode batch, #running-req: {n}, #token: 0, token usage: 0.00, "
       "cuda graph: {graph}, gen throughput (token/s): 1.00, #queue-req: 0")


class Engine:
    """Lockstep decode: each step every admitted request gets one token; a decode line every interval.

    A measured batch is prefilled together once all ``batch`` requests have arrived, so its
    requests decode and finish on the same steps however late the client threads start.
    """

    def __init__(self, step_s, *, batch=1, cap=None, corrupt=False, interval=4, prefill_s=0.004):
        self.step_s, self.cap, self.corrupt, self.interval, self.prefill_s = step_s, cap, corrupt, interval, prefill_s
        self.batch = batch
        self.cond, self.requests, self.samples, self.steps, self.stopped = threading.Condition(), [], [], 0, False
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def submit(self, rid, ids, budget):
        state = dict(ids=ids, budget=budget, tokens=[], ready=None)
        with self.cond:
            self.requests.append(state)
            held = [r for r in self.requests if r["ready"] is None]
            if rid.startswith("warmup-") or len(held) >= self.batch:
                for r in held:
                    r["ready"] = time.perf_counter() + self.prefill_s
        return state

    def loop(self):
        tick = time.perf_counter()
        while not self.stopped:
            tick += self.step_s                  # fixed-rate steps: a scheduling hiccup is caught up, not carried
            time.sleep(max(0.0, tick - time.perf_counter()))
            with self.cond:
                now = time.perf_counter()
                running = [r for r in self.requests if len(r["tokens"]) < r["budget"]
                           and r["ready"] is not None and now >= r["ready"]]
                running = running[:self.cap] if self.cap else running
                for r in running:
                    r["tokens"].append((sum(r["ids"]) * 31 + len(r["tokens"]) * 7 + self.corrupt) % 997)
                if running:
                    self.steps += 1
                    if self.steps % self.interval == 0:
                        self.samples.append((now, measure.parse_decode_line(LOG.format(n=len(running), graph=True))))
                self.cond.notify_all()


def handler(engine, info):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, body, kind="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(b"ok" if self.path == "/health" else json.dumps(info).encode())

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            ids, budget = request["input_ids"], request["sampling_params"]["max_new_tokens"]
            natural = request["stream"] and not request["sampling_params"]["ignore_eos"]
            stop = max(1, budget - sum(ids) % 5) if natural else budget      # emulated EOS of streamed requests
            state = engine.submit(request["rid"], ids, stop)

            def meta(count):
                return dict(id=request["rid"], prompt_tokens=len(ids), completion_tokens=count,
                            finish_reason=None if count != stop else dict(type="length", length=count)
                            if stop == budget else dict(type="stop", matched=2))

            if not request["stream"]:
                with engine.cond:
                    engine.cond.wait_for(lambda: len(state["tokens"]) == budget or engine.stopped)
                tokens = list(state["tokens"])
                logprobs = [[-0.5, token, None] for token in tokens]
                self.reply(json.dumps(dict(output_ids=tokens, meta_info=dict(
                    meta(budget), output_token_logprobs=logprobs))).encode())
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            sent = 0
            while sent < stop and not engine.stopped:
                with engine.cond:
                    engine.cond.wait_for(lambda: len(state["tokens"]) > sent or engine.stopped)
                    fresh = state["tokens"][sent:]
                for token in fresh:
                    sent += 1
                    event = dict(output_ids=[token], text="", meta_info=meta(sent))
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")

    return Handler


class FakeServer:
    def __init__(self, engine, info):
        self.engine, self.samples, self.on_stop = engine, engine.samples, []
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), handler(engine, info))
        self.http.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"
        threading.Thread(target=self.http.serve_forever, kwargs=dict(poll_interval=0.01), daemon=True).start()

    def wait_ready(self, timeout):
        return None

    def stop(self, grace=0):
        self.engine.stopped = True
        self.http.shutdown()
        self.http.server_close()
        for hook in self.on_stop:
            hook()
        return 0


class FakeLauncher:
    """Emulated arms: step time, a running-batch cap or corrupted tokens; receipts like the real workers."""

    def __init__(self, plan_dir, *, step=0.0025, speeds=None, caps=None, corrupt=()):
        self.plan_dir, self.step, self.corrupt = plan_dir, step, corrupt
        self.speeds, self.caps, self.started = speeds or {}, caps or {}, []
        plan = json.loads((plan_dir / "plan.json").read_text())
        self.batches = {point["name"]: point["real_batch"] for point in plan["pressure_points"]}

    def start(self, arm, run_dir, *, config_path=None, wrapper=()):
        receipts = run_dir / "receipts"
        receipts.mkdir(parents=True)
        config = None if arm == runner.NATIVE else sc.read(config_path or self.plan_dir / f"{arm}.json")[0]
        measured = run_dir.parents[3].name == "runs"            # runs/<point>/<pair>/r<k>/<slot>-<arm>
        batch = self.batches[run_dir.parents[2].name] if measured else 1
        engine = Engine(self.speeds.get(arm, self.step), batch=batch, cap=self.caps.get(arm),
                        corrupt=arm in self.corrupt)
        info = dict(max_total_num_tokens=4096, page_size=4, internal_states=[dict(
            memory_usage=dict(kvcache=1.0, token_capacity=4096), effective_max_running_requests_per_dp=64)])
        server = FakeServer(engine, info)
        if config is not None and (config.uses_fixed_layers or config.uses_arena):
            self.receipts(config, receipts)
            if config.diagnostics == "counters":
                server.on_stop.append(lambda: self.counters(receipts))
        self.started.append((arm, tuple(wrapper)))
        return server, ["fake-launch", arm, *wrapper]

    @staticmethod
    def receipts(config, directory):
        offload = list(config.placement.offload_layers) if config.uses_fixed_layers else [0, 1, 2, 3]
        placement = dict(signature=dict(offload_layers=offload, hot_plane_rows=2308,
                                        resident_path="direct" if config.uses_resident_direct else "resolver",
                                        native_swap=None if config.uses_arena else
                                        "split" if config.uses_split_copy else "fused"),
                         bytes=dict(hot_main_kv=1, resident_main_kv=2, host_pinned=3))
        for rank in range(TP):
            if config.uses_arena:
                record = dict(event="hisparse_shared_lru_bound", rank=rank, config_sha256=config.digest,
                              signature=dict(layers=offload, plane_rows=2308))
                if config.uses_fixed_layers:
                    record["placement"] = placement
            else:
                record = dict(placement, event="hisparse_layer_placement_bound", rank=rank, config_sha256=config.digest)
            (directory / f"{record['event']}.rank{rank}.json").write_text(json.dumps(record))

    @staticmethod
    def counters(directory):
        for rank in range(TP):
            layers = {str(layer): dict(zip(runner.arena.reference.COUNTER_FIELDS, range(1, 9))) for layer in (1, 3)}
            (directory / f"hisparse_shared_lru_counters.rank{rank}.json").write_text(json.dumps(dict(
                event="hisparse_shared_lru_counters", rank=rank, status=0, layers=layers)))


def fake_command(argv, env=None, timeout=None):
    """nvidia-smi, nvcc, nsys, interpreter probes and the build/gate entry points of an idle 8-GPU host."""
    if argv[0] == "nvidia-smi" and argv[1].startswith("--query-gpu=index,uuid,name"):
        return 0, "\n".join(f"{i}, GPU-{i}, NVIDIA A100-SXM4-80GB, 81920 MiB, 3 MiB, 0 %, 210 MHz, 1410 MHz, "
                            "580.178.04" for i in range(TP))
    if argv[0] == "nvidia-smi" and argv[1].startswith("--query-gpu=index,uuid,memory"):
        return 0, "\n".join(f"{i}, GPU-{i}, 70000, 81920" for i in range(TP))
    if argv[0] == "nvidia-smi" and argv[1] == "--query-gpu=uuid":
        return 0, "\n".join(f"GPU-{i}" for i in range(TP))
    if argv[:2] == ["nsys", "export"]:                   # rank 1 reads layer 3 slower than the others
        nsys_export.export(Path(argv[argv.index("--output") + 1]), devices=TP)
        return 0, ""
    if argv[0] in ("nvidia-smi", "nvcc", "nsys"):
        return 0, ""
    if argv[1] == "-c":
        return 0, "/pinned/sglang\n" if "sglang" in argv[2] else '["2.15.0", "13.4"]\n'
    Path(argv[argv.index("--output") + 1]).write_text(json.dumps(dict(status="pass", entry=argv[2])))
    return 0, "ok"


def tiny_spec(heldout):
    spec = copy.deepcopy(SMALL)
    spec.update(
        split={"training_sha256": "a" * 64, "heldout_sha256": heldout},
        offload_lists=[{"name": "fixed", "selection": "fixed", "offload_layers": [1, 3]}],
        static_quota_rows={"fixed": {"1": 24, "3": 40}}, resident_direct_lists=["fixed"],
        pressure_points=[{"name": "p1", "input_tokens": [8, 16], "real_batch": 4},
                         {"name": "p2", "input_tokens": [16, 32], "real_batch": 3}],
        measurement=dict(max_new_tokens=64, warmup_requests=1, warmup_steps=4, min_window_steps=16,
                         min_decode_log_samples=3, functional_requests=3, functional_max_new_tokens=6, ttft_slo_s=5,
                         tpot_slo_s=1, server_start_timeout_s=5, request_timeout_s=30, workloads=["fixed", "natural"]),
        acceptance=dict(candidate="fixed_shared_arena", baselines=["stock", "fixed_static_quota"], min_gain=0.05,
                        tpot_p95_regression_max=0.05, ci_level=0.95, min_pressure_points=2, min_pairs=3,
                        quality_mismatch_over_stock_max=0.0))
    return spec


def workspace(tmp, *, placeholder=False):
    """Held-out workloads, their manifest and a plan compiled against the manifest digest."""
    tmp = Path(tmp)
    sizes = dict(functional=(3, 8), p1=(12, 8), p2=(9, 16))
    entries = {}
    for name, (count, low) in sizes.items():
        rows = [dict(request_alias=f"{name}-{i}", workload=name, input_ids=[(i * 13 + j) % 50 + 1
                                                                            for j in range(low + i % 4)],
                     max_new_tokens=4) for i in range(count)]
        path = tmp / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        entries[name] = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = tmp / "manifest.json"
    manifest.write_text(json.dumps(dict(schema=runner.MANIFEST_SCHEMA, split="heldout",
                                        functional=entries.pop("functional"), points=entries)))
    digest = "0" * 64 if placeholder else hashlib.sha256(manifest.read_bytes()).hexdigest()
    (tmp / "spec.json").write_text(json.dumps(tiny_spec(digest)))
    spec, spec_sha256 = planner.read_spec(tmp / "spec.json")
    planner.compile_plan(spec, spec_sha256, tmp / "plan")
    (tmp / "model").mkdir()
    (tmp / "model" / "config.json").write_text("{}")
    return tmp / "plan", manifest, tmp / "model"


def main(tmp, launcher, *extra, command=None):
    plan, manifest, model = Path(tmp) / "plan", Path(tmp) / "manifest.json", Path(tmp) / "model"
    argv = ["--plan-dir", str(plan), "--manifest", str(manifest), "--model-dir", str(model),
            "--devices", ",".join(map(str, range(TP))), "--output-dir", str(Path(tmp) / "out"),
            "--image-id", runner.EXPECTED_IMAGE, "--weights-sha256", "f" * 64, "--allow-gpu", "--max-priority", "1",
            *extra]
    with patch.object(runner.placement_device, "verify_digests", lambda root: {"srt/x.py": "ok"}), \
            patch.object(runner.placement_device, "resolve_symbols", lambda root: [dict(symbol="s", status="ok")]), \
            contextlib.redirect_stdout(io.StringIO()):
        code = runner.main(argv, launcher=launcher, command=command or fake_command)
    out = Path(tmp) / "out"
    return code, json.loads((out / "stages.json").read_text()), out


class EndToEnd(unittest.TestCase):
    def setUp(self):
        # The emulated arms differ by milliseconds per step; with the whole suite loaded a full
        # collection stops every thread for about 30 ms, close to a whole candidate window.
        gc.disable()
        self.addCleanup(gc.enable)

    def test_every_stage_writes_its_evidence_and_a_faster_candidate_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_dir, _, _ = workspace(tmp)
            launcher = FakeLauncher(plan_dir, speeds={"fixed_shared_arena": 0.001})
            code, stages, out = main(tmp, launcher)
            self.assertEqual(code, 0, json.dumps(stages, indent=1)[:3000])
            self.assertEqual({stage: value["status"] for stage, value in stages.items()},
                             dict.fromkeys(runner.STAGES, "pass"))
            identity = json.loads((out / "identity.json").read_text())
            self.assertEqual((len(identity["gpus"]), identity["foreign_processes"]), (TP, []))
            self.assertEqual(sorted(p.name for p in (out / "build").glob("*.json")),
                             ["layer_placement.json", "shared_lru.json", "summary.json"])
            functional = json.loads((out / "functional" / "correctness.json").read_text())
            self.assertEqual([a["arm"] for a in functional["arms"]][:3], [runner.NATIVE, "stock", "stock"])
            self.assertEqual(functional["stock_repeat_mismatch"], dict(sequential=0.0, concurrent=0.0))
            self.assertTrue(all(a["status"] == "pass" for a in functional["arms"]))
            capacity = json.loads((out / "capacity" / "admission.json").read_text())
            self.assertEqual(set(capacity["arms"]), {"stock", "fixed_shared_arena", "fixed_static_quota"})
            self.assertEqual(capacity["infeasible"], [])
            runs = sorted((out / "runs").rglob("metrics.json"))
            self.assertEqual(len(runs), 3 * 2 * 2 * 3 * 2, "three priority-0/1 pairs x two points x two workloads x "
                                                           "three AB/BA rounds")
            metrics = [json.loads(p.read_text()) for p in runs]
            self.assertTrue(all(m["status"] == "valid" for m in metrics), [m["reasons"] for m in metrics])
            first, natural = metrics[0], metrics[2]
            self.assertEqual((first["workload"], natural["workload"]), ("fixed", "natural"))
            self.assertGreaterEqual(first["window"]["steps"], 16)
            self.assertEqual(first["window"]["finish_reasons"], {"length": 4})
            self.assertEqual(natural["window"]["finish_reasons"].get("stop", 0) + natural["window"]["finish_reasons"]
                             .get("length", 0), 4)
            self.assertGreater(natural["window"]["natural_eos_fraction"], 0)
            self.assertEqual(first["real_batch"]["running"], [json.loads((runs[0].parent / "run_config.json")
                                                                         .read_text())["real_batch"]])
            for name in ("run_config.json", "admission.json", "correctness.json"):
                self.assertTrue((runs[0].parent / name).is_file())
            verdict = json.loads(next(out.glob("verdict-*.json")).read_text())
            self.assertEqual({w: a["overall"] for w, a in verdict["acceptance"].items()},
                             dict(fixed="pass", natural="pass"), json.dumps(verdict["acceptance"]))
            results = {(r["point"], r["workload"], r["control"], r["candidate"]): r for r in verdict["results"]}
            noise = results[("p1", "fixed", "stock", "stock")]
            self.assertEqual((noise["blocks"], noise["quality_ok"]), (3, True), "native A/A repeat noise")
            win = results[("p2", "natural", "stock", "fixed_shared_arena")]
            self.assertEqual((win["verdict"], win["blocks"], win["orders"]), ("pass", 3, ["AB", "BA"]))
            gates = sorted(p.name for p in (out / "gate").glob("*.json"))
            self.assertEqual(gates, ["combined-gate.json", "fixed-layer-gate-production.json", "fixed-layer-gate.json"])
            with (out / "diagnostics" / "diagnostic_summary.csv").open() as handle:
                rows = handle.read().splitlines()
            self.assertEqual(len(rows), 1 + 2 * TP * 2 + 1, "header, two arena arms x ranks x layers, stock note")
            self.assertIn(("fixed_shared_arena", ()), launcher.started)

    def test_rejected_rounds_and_negative_results_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_dir, _, _ = workspace(tmp)
            launcher = FakeLauncher(plan_dir, speeds={"fixed_shared_arena": 0.005}, caps={"fixed_static_quota": 3})
            code, stages, out = main(tmp, launcher, "--points", "p1", "--stages",
                                     "identity,build,gate,functional,capacity,performance,verdict")
            self.assertEqual(stages["performance"]["rejected"], 6, "the capped arm never holds B=4 in either workload")
            verdict = json.loads(next(out.glob("verdict-*.json")).read_text())
            results = {(r["point"], r["workload"], r["control"], r["candidate"]): r for r in verdict["results"]}
            for workload in ("fixed", "natural"):
                self.assertEqual(results[("p1", workload, "stock", "fixed_shared_arena")]["verdict"], "fail")
                self.assertNotIn(("p1", workload, "fixed_static_quota", "fixed_shared_arena"), results)
            self.assertEqual(len(verdict["rejected_rounds"]), 6)
            reasons = verdict["rejected_rounds"][0]["runs"]
            capped = [run for run in reasons.values() if run["status"] == "rejected"][0]["reasons"]
            self.assertTrue(any("full-batch" in reason or "running batch" in reason for reason in capped), capped)
            self.assertEqual({w: a["overall"] for w, a in verdict["acceptance"].items()},
                             dict(fixed="fail", natural="fail"))
            self.assertEqual(code, 0)

    def test_wrong_outputs_stop_the_study_before_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_dir, _, _ = workspace(tmp)
            code, stages, out = main(tmp, FakeLauncher(plan_dir, corrupt=("fixed_static_quota",)))
            self.assertEqual(code, 1)
            self.assertEqual(stages["functional"]["status"], "fail")
            self.assertTrue(any("fixed_static_quota" in r and "differ from stock" in r
                                for r in stages["functional"]["reasons"]))
            for stage in ("capacity", "performance", "diagnostics", "verdict"):
                self.assertEqual(stages[stage]["status"], "skipped", stage)
            self.assertFalse((out / "runs").exists())

    def test_guards_deadline_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_dir, _, _ = workspace(tmp)
            launcher = FakeLauncher(plan_dir)
            past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)).isoformat()
            stages_arg = ("--stages", "identity,build,gate,functional,capacity,performance", "--points", "p1",
                          "--max-priority", "0")
            code, stages, out = main(tmp, launcher, *stages_arg, "--deadline", past)
            self.assertEqual((code, stages["performance"]["not_run"], stages["performance"]["status"]), (1, 12, "fail"))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(tmp, launcher, *stages_arg)                       # the output directory is never reused
            code, stages, out = main(tmp, launcher, *stages_arg, "--resume")
            self.assertEqual((code, stages["performance"]["valid"]), (0, 12))
            code, stages, out = main(tmp, launcher, *stages_arg, "--resume")
            self.assertEqual((stages["performance"]["kept"], stages["performance"]["valid"]), (12, 0))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main(["--plan-dir", str(plan_dir), "--manifest", "m", "--model-dir", "d", "--devices", "0",
                             "--output-dir", str(Path(tmp) / "new"), "--image-id", "x"])     # no --allow-gpu

    def test_a_foreign_process_makes_the_runner_yield_the_devices(self):
        def busy(argv, env=None, timeout=None):
            if argv[0] == "nvidia-smi" and argv[1].startswith("--query-compute-apps"):
                return 0, "GPU-3, 4242, 1024 MiB\nGPU-99, 7, 1 MiB"
            return fake_command(argv, env, timeout)

        with tempfile.TemporaryDirectory() as tmp:
            plan_dir, _, _ = workspace(tmp)
            launcher = FakeLauncher(plan_dir)
            main(tmp, launcher, "--stages", "identity,build,gate,functional,capacity", "--points", "p1")
            code, stages, out = main(tmp, launcher, "--stages", "performance", "--points", "p1", "--resume",
                                     command=busy)
            performance = stages["performance"]
            self.assertEqual((code, performance["status"], performance["valid"]), (1, "fail", 0))
            self.assertEqual(performance["not_run"], 3 * 3 * 2 * 2)
            self.assertIn("yielded the devices: 1 foreign compute processes appeared", performance["reasons"])

    def test_placeholder_holdout_digest_blocks_every_workload_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_dir, _, _ = workspace(tmp, placeholder=True)
            code, stages, out = main(tmp, FakeLauncher(plan_dir))
            self.assertEqual((stages["gate"]["status"], stages["functional"]["status"]), ("pass", "fail"))
            self.assertIn("placeholder", stages["functional"]["reasons"][0])

    def test_nsys_waves_give_whole_steps_and_the_bottleneck_rank_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_dir, _, _ = workspace(tmp)
            launcher = FakeLauncher(plan_dir)
            code, stages, out = main(tmp, launcher, "--stages", "identity,build,gate,functional,diagnostics", "--nsys")
            self.assertEqual((code, stages["diagnostics"]["status"]), (0, "pass"), stages["diagnostics"])
            arm = "fixed_shared_arena"
            self.assertIn((arm, ("nsys", "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node",
                                 "--force-overwrite=false", "-o", str(out / "diagnostics" / arm / "nsys" / arm))),
                          launcher.started)
            summary = json.loads((out / "diagnostics" / arm / "nsys" / "timeline.json").read_text())
            self.assertEqual((summary["status"], summary["tp_size"], summary["steps"]), ("pass", TP, 5))
            with (out / "diagnostics" / "diagnostic_summary.csv").open() as handle:
                rows = [row for row in csv.DictReader(handle) if row["arm"] == arm and row["source"] == "nsys"]
            steps = [row for row in rows if row["layer"] == "step"]
            self.assertEqual([row["rank"] for row in steps], [*map(str, range(TP)), "bottleneck", "critical"])
            self.assertEqual([row["bottleneck_steps"] for row in steps], ["0", "5", "0", "0", "5", ""])
            self.assertEqual([row["critical_steps"] for row in steps], ["5", "0", "0", "0", "", "5"],
                             "equal TP waits: the lowest rank")
            self.assertEqual(float(steps[-2]["whole_step_us_p50"]), 346)
            layers = [(row["layer"], float(row["read_path_us"])) for row in rows
                      if row["rank"] == "bottleneck" and row["layer"] != "step"]
            self.assertEqual(layers, [("0", 0), ("1", 20), ("2", 0), ("3", 50)])
            critical = [json.loads(row["critical_ranks"]) for row in rows
                        if row["rank"] == "critical" and row["layer"] != "step"]
            self.assertEqual(critical, [{}, {"0": 5}, {"0": 5}, {"0": 5}])
            identity = json.loads((out / "identity.json").read_text())
            self.assertIn("nsys", identity["toolchain"])


class ProcessPieces(unittest.TestCase):
    def test_launcher_argv_pins_the_plan_config_model_and_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_dir, _, model = workspace(tmp)
            plan = json.loads((plan_dir / "plan.json").read_text())
            launcher = runner.Launcher(plan_dir, plan, devices="0,1,2,3", model_dir=str(model), port=31999)
            argv = launcher.argv("fixed_shared_arena")
            self.assertEqual(argv[:3], [sys.executable, "-m", "runtime.launch"])
            self.assertEqual(argv[argv.index("--hisparse-study-config") + 1], str(plan_dir / "fixed_shared_arena.json"))
            self.assertEqual(argv[argv.index("--model-path") + 1], str(model))
            self.assertEqual(argv[-4:], ["--host", "127.0.0.1", "--port", "31999"])
            native = launcher.argv(runner.NATIVE)
            self.assertNotIn("--hisparse-study-config", native)
            self.assertEqual(native[native.index("--"):], argv[argv.index("--"):])
            env = runner.gpu_env("2,3")
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,3")
            self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(runner.ROOT))

    def test_server_reader_timestamps_decode_lines_and_stop_escalates(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = f"import time; print({LOG.format(n=4, graph=True)!r}, flush=True); time.sleep(60)"
            server = runner.Server([sys.executable, "-c", script], dict(os.environ), Path(tmp), "http://127.0.0.1:9")
            deadline = time.monotonic() + 10
            while not server.samples and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertEqual(server.samples[0][1]["running"], 4)
            self.assertIsNotNone(server.stop(grace=2))
            self.assertIn("Decode batch", (Path(tmp) / "server.log").read_text())
            (Path(tmp) / "dead").mkdir()
            dead = runner.Server([sys.executable, "-c", "raise SystemExit(3)"], dict(os.environ), Path(tmp) / "dead",
                                 "http://127.0.0.1:9")
            self.assertEqual(dead.wait_ready(10), "server exited with code 3 before it was ready")
            self.assertEqual(dead.stop(), 3)


if __name__ == "__main__":
    unittest.main()