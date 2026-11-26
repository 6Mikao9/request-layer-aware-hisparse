"""Execute a compiled combined-study plan on the maintainer's GPU host, one stage at a time.

    python -m runtime.combined_run --plan-dir "$PLAN_DIR" --manifest "$DATA_MANIFEST" --model-dir "$MODEL_DIR" \
        --devices 0,1,2,3,4,5,6,7 --image-id sha256:614b...7985e87 --output-dir "$OUTPUT_DIR/combined" --allow-gpu \
        [--stages identity,build,...] [--max-priority 1] [--points 16-32Ki,32-64Ki] [--deadline ISO8601] [--resume]

The maintainer supplies paths, devices and the authorized window; the runner applies
the plan (arms, pairs, AB/BA order, measurement and acceptance rules) and adds no
choice of its own. Every stage writes its result before the next one starts and no
result file is ever overwritten (``--resume`` continues a directory, skipping finished
runs). ``stages.json`` indexes the stage statuses: pass, fail, skipped (a prerequisite
did not pass) or not_run, each with its reasons.

identity     identity.json: selected GPUs (UUID, clocks, foreign processes), topology, image ID,
             pinned SGLang digests and symbols, weights, host memlock/memory, disk, toolchain
build        build/{shared_lru,layer_placement,split_copy}.json: sm_80 JIT build receipts with artifacts
gate         gate/{combined-gate,fixed-layer-gate,fixed-layer-gate-production}.json: the A100 correctness
             gates of the arms' mechanisms, eager and CUDA Graph
functional   functional/correctness.json: real-model outputs of no config, the all-default config and
             every arm against stock (stock A/A gives the repeat rate), sequential and concurrent;
             per-rank receipts of every arm
capacity     capacity/admission.json: auto-sized pools (/server_info), receipts, GPU memory per arm,
             and which planned batches each arm can hold
performance  runs/<point>/<pair>/r<k>-<workload>/<a|b>-<arm>/{run_config,admission,correctness,metrics}.json:
             fresh server per run, warmup, one full batch; fixed work (ignore_eos) is judged on its steady
             full-batch window, natural load (EOS) on its whole decode phase
diagnostics  diagnostics/diagnostic_summary.csv: counter waves of arena arms; with --nsys a captured run
             per arm whose steady decode graph gives per-rank whole steps and TP waits, each step's
             bottleneck and critical rank and their per-layer split into swap, resolver, attention,
             collective, other and idle (<arm>/nsys/timeline.json keeps every step of every rank)
verdict      verdict-<UTC>.json: paired intervals per point, workload and pair, native A/A noise, and the
             acceptance of each workload side by side

Logs, token-ID ledgers and host details stay in the output directory, the maintainer's
private workspace; only code, synthetic fixtures and anonymous summaries enter the PR.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from benchmarks import sglang_correctness_client as correctness_client
from benchmarks import sglang_stream_client as stream
from . import combined_measure as measure
from . import combined_timeline as timeline
from . import layer_placement_device as placement_device
from . import shared_lru_device as arena
from . import study_config as sc

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("identity", "build", "gate", "functional", "capacity", "performance", "diagnostics", "verdict")
REQUIRES = {"build": ("identity",), "gate": ("build",), "functional": ("gate",), "capacity": ("functional",),
            "performance": ("capacity",), "diagnostics": ("functional",), "verdict": ("performance",)}
EXPECTED_IMAGE = "sha256:614b1509bcb729c653528d448bff9850f8977fcad9d12ba56bd6dbe3d7985e87"
MANIFEST_SCHEMA = "combined-workload-manifest-v1"
NATIVE = "native_no_config"          # runtime.launch without --hisparse-study-config
GPU_FIELDS = ("index", "uuid", "name", "memory_total", "memory_used", "utilization", "clock_sm", "clock_sm_max",
              "driver")
TIMELINE_FIELDS = ("steps", "whole_step_us_p50", "whole_step_us_p95", "bottleneck_steps", "critical_steps",
                   "collective_us_p50", "tp_wait_us_p50", *timeline.LAYER_FIELDS, "critical_ranks")
OPENER = build_opener(ProxyHandler({}))


def sha256(data):
    return hashlib.sha256(data if isinstance(data, bytes) else Path(data).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return value


def http_json(url, timeout=30):
    with OPENER.open(Request(url), timeout=timeout) as response:
        return json.load(response)


def run_command(argv, *, env=None, timeout=None):
    """(exit code, combined output) of one helper command; a missing executable is exit 127."""
    try:
        done = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return 127, f"{argv[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{argv[0]}: timed out after {timeout} s"
    return done.returncode, done.stdout + done.stderr


class Server:
    """One SGLang server in its own process group; a reader thread timestamps every log line."""

    def __init__(self, argv, env, run_dir, url):
        self.url, self.samples = url, []
        self.log = (run_dir / "server.log").open("x", encoding="utf-8")
        self.process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for raw in self.process.stdout:
            now, line = time.perf_counter(), raw.decode("utf-8", "replace")
            self.log.write(line)
            self.log.flush()
            parsed = measure.parse_decode_line(line)
            if parsed is not None:
                self.samples.append((now, parsed))

    def alive(self):
        return self.process.poll() is None

    def wait_ready(self, timeout):
        """None once /health answers, else the reason the server never became ready."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.alive():
                return f"server exited with code {self.process.returncode} before it was ready"
            try:
                with OPENER.open(Request(self.url + "/health"), timeout=5) as response:
                    if response.status == 200:
                        return None
            except (URLError, OSError):
                pass
            time.sleep(2)
        return f"server was not ready after {timeout} s"

    def stop(self, grace=120):
        """SIGINT the whole process group (workers flush receipts at exit), then escalate."""
        for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 30), (signal.SIGKILL, 30)):
            if not self.alive():
                break
            try:
                os.killpg(self.process.pid, sig)
                self.process.wait(wait)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                continue
        self.reader.join(30)
        self.process.stdout.close()
        self.log.close()
        return self.process.returncode


class Launcher:
    """Starts plan arms on the maintainer's devices: a fresh server and receipt directory per run."""

    def __init__(self, plan_dir, plan, *, devices, model_dir, port):
        self.plan_dir, self.plan, self.devices, self.model_dir, self.port = plan_dir, plan, devices, model_dir, port
        self.files = {arm["arm"]: arm["config"] for arm in plan["arms"]}

    def argv(self, arm, config_path=None):
        launch = list(self.plan["launch"]["stock" if arm == NATIVE else arm])
        at = launch.index("--hisparse-study-config")
        if arm == NATIVE:
            del launch[at:at + 2]
        else:
            launch[at + 1] = str(config_path or Path(self.plan_dir) / self.files[arm])
        launch = [sys.executable if token == "python" else self.model_dir if token == "$MODEL_DIR" else token
                  for token in launch]
        return launch + ["--host", "127.0.0.1", "--port", str(self.port)]

    def start(self, arm, run_dir, *, config_path=None, wrapper=()):
        receipts = run_dir / "receipts"
        receipts.mkdir(parents=True)
        env = gpu_env(self.devices)
        env[arena.RECEIPT_DIR_ENV] = str(receipts)
        argv = [*wrapper, *self.argv(arm, config_path)]
        return Server(argv, env, run_dir, f"http://127.0.0.1:{self.port}"), argv


def gpu_env(devices):
    path = os.environ.get("PYTHONPATH")
    return dict(os.environ, CUDA_VISIBLE_DEVICES=devices,
                PYTHONPATH=os.pathsep.join([str(ROOT)] + ([path] if path else [])))


def host_limits(directory):
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    out = dict(memlock_soft=None if soft == resource.RLIM_INFINITY else soft,
               memlock_hard=None if hard == resource.RLIM_INFINITY else hard,
               disk_free_bytes=shutil.disk_usage(directory).free)
    meminfo, cgroup = Path("/proc/meminfo"), Path("/sys/fs/cgroup/memory.max")
    if meminfo.is_file():
        out["mem_available_kib"] = next((int(line.split()[1]) for line in meminfo.read_text().splitlines()
                                         if line.startswith("MemAvailable:")), None)
    if cgroup.is_file():
        out["cgroup_memory_max"] = cgroup.read_text().strip()
    return out


def geometry_args(geo):
    names = ("tp_size", "first_layer", "layers", "hot_rows", "page", "top_k_blocks", "block_size", "head_dim",
             "dtype_bytes", "kv_heads_per_rank", "slots", "plane_rows")
    return [token for name in names for token in (f"--{name.replace('_', '-')}", str(geo[name]))]


def timeline_rows(arm, point, summary):
    """CSV rows of one Nsight timeline: whole step and TP wait per rank and of each step's bottleneck rank (longest
    whole step) and critical rank (least TP wait), then those ranks' own layers (medians over steps)."""
    if summary is None:
        return []
    steps, rows = summary["steps"], []
    picked = dict(bottleneck=dict(summary["bottleneck"], bottleneck_steps=steps))
    if summary["critical"] is not None:
        picked["critical"] = dict(summary["critical"], critical_steps=steps)
    for rank, row in [*((row["rank"], row) for row in summary["ranks"]), *picked.items()]:
        rows.append(dict(arm=arm, point=point, rank=rank, layer="step", source="nsys", steps=steps,
                         whole_step_us_p50=(row["whole_step_us"] or {}).get("p50"),
                         whole_step_us_p95=(row["whole_step_us"] or {}).get("p95"),
                         bottleneck_steps=row.get("bottleneck_steps"), critical_steps=row.get("critical_steps"),
                         collective_us_p50=(row.get("collective_us") or {}).get("p50"),
                         tp_wait_us_p50=(row.get("tp_wait_us") or {}).get("p50")))
    for rank, part in picked.items():
        rows += [dict(arm=arm, point=point, rank=rank, source="nsys", steps=steps,
                      **dict(layer, critical_ranks=json.dumps(layer["critical_ranks"]) if "critical_ranks" in layer
                             else None))
                 for layer in part["layers"] or []]
    return rows


def info_excerpt(info):
    """The /server_info fields a run is judged by: pools, memory, graph and HiSparse settings."""
    if info is None:
        return None
    state = (info.get("internal_states") or [{}])[0]
    keys = ("max_total_num_tokens", "max_req_input_len", "page_size", "hisparse_config", "disable_cuda_graph",
            "cuda_graph_max_bs", "cuda_graph_config", "mem_fraction_static", "max_running_requests", "tp_size",
            "decode_log_interval", "launch_command")
    return dict({key: info.get(key) for key in keys}, memory_usage=state.get("memory_usage"),
                effective_max_running_requests_per_dp=state.get("effective_max_running_requests_per_dp"))


def read_manifest(path, plan):
    """Held-out workloads pinned by the plan's split digest; returns (rows, digest, reasons)."""
    raw = Path(path).read_bytes()
    digest, reasons = sha256(raw), []
    expected = plan["split"]["heldout_sha256"]
    if expected == "0" * 64:
        reasons.append("plan split.heldout_sha256 is a placeholder; put this manifest's SHA-256 in the spec and "
                       "recompile the plan")
    elif digest != expected:
        reasons.append(f"manifest SHA-256 {digest} differs from the plan's held-out digest {expected}")
    data = sc.loads(raw)
    if data.get("schema") != MANIFEST_SCHEMA or data.get("split") != "heldout" or set(data) != {
            "schema", "split", "functional", "points"}:
        raise ValueError(f"Manifest must be {MANIFEST_SCHEMA} with split=heldout, functional and points")
    rows, base = {}, Path(path).parent
    entries = [("functional", data["functional"], None)] + [
        (name, data["points"].get(name), point) for name, point in ((p["name"], p) for p in plan["pressure_points"])]
    for name, entry, point in entries:
        if type(entry) is not dict or set(entry) != {"path", "sha256"}:
            raise ValueError(f"Manifest entry {name} needs path and sha256")
        payload = (base / entry["path"]).read_bytes()
        if sha256(payload) != entry["sha256"]:
            raise ValueError(f"Workload {name} differs from its manifest SHA-256")
        rows[name] = stream.read_workload(payload)
        if point is not None:
            low, high = point["input_tokens"]
            outside = sum(not low <= len(row["input_ids"]) < high for row in rows[name])
            if outside:
                reasons.append(f"{outside} rows of {name} fall outside its input range [{low}, {high})")
            if len(rows[name]) < plan["repeats"] * point["real_batch"]:
                reasons.append(f"{name} needs {plan['repeats']} batches of {point['real_batch']} distinct requests")
    return rows, digest, reasons


def first_divergence(a, b):
    if a is None or b is None:
        return None
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


class Runner:
    """Stage driver over one plan, one held-out manifest and one launcher (fakes in the CPU tests)."""

    def __init__(self, args, plan, manifest, launcher, command=run_command):
        self.args, self.plan, self.launcher, self.command = args, plan, launcher, command
        self.rows, self.manifest_sha256, self.manifest_reasons = manifest
        self.out, self.m, self.acceptance = Path(args.output_dir), plan["measurement"], plan["acceptance"]
        self.points, self.tp = {p["name"]: p for p in plan["pressure_points"]}, plan["geometry"]["tp_size"]
        self.configs = {arm["arm"]: sc.read(Path(args.plan_dir) / arm["config"], arm["file_sha256"])[0]
                        for arm in plan["arms"]}
        points = set(filter(None, args.points.split(",")))
        self.schedule = [entry for entry in plan["schedule"] if entry["priority"] <= args.max_priority
                         and (not points or entry["point"] in points)]

    def run(self, stages):
        index = self.out / "stages.json"
        status = json.loads(index.read_text()) if index.exists() else {}
        for stage in STAGES:
            if stage not in stages or (stage in status and stage not in ("performance", "verdict")):
                continue                    # --resume keeps recorded stages; performance continues, verdict re-reads
            missing = [name for name in REQUIRES.get(stage, ()) if status.get(name, {}).get("status") != "pass"]
            result = (dict(status="skipped", reasons=[f"requires {missing} to pass"]) if missing
                      else getattr(self, f"stage_{stage}")())
            status[stage] = dict(result, finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
            partial = index.with_suffix(".tmp")
            partial.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
            os.replace(partial, index)
        return status

    # -- shared steps --------------------------------------------------------------------------

    def gpu_memory(self):
        code, out = self.command(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
                                  "--format=csv,noheader,nounits", "-i", self.args.devices])
        if code != 0:
            return None
        return [dict(zip(("index", "uuid", "used_mib", "total_mib"), (x.strip() for x in line.split(","))))
                for line in out.splitlines() if line.strip()]

    def foreign_processes(self):
        """Compute processes on the selected GPUs; checked between runs, when none of ours is alive."""
        code, uuids = self.command(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i",
                                    self.args.devices])
        listed, apps = self.command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                     "--format=csv,noheader"])
        if code or listed:
            return ["nvidia-smi query failed"]
        selected = {line.strip() for line in uuids.splitlines() if line.strip()}
        return [line.strip() for line in apps.splitlines() if line.split(",")[0].strip() in selected]

    def check_receipts(self, arm, directory, config=None):
        """Bound receipts must cover every TP rank with this arm's config, S and resident path."""
        config = config or (None if arm == NATIVE else self.configs[arm])
        records = [json.loads(path.read_text()) for path in sorted(Path(directory).glob("*.json"))]
        events = ("hisparse_layer_placement_bound", "hisparse_shared_lru_bound")
        bound = [r for r in records if r.get("event") in events]
        expected = None if config is None or not (config.uses_fixed_layers or config.uses_arena) else (
            events[1] if config.uses_arena else events[0])
        reasons = []
        if expected is None and bound:
            reasons.append("a stock arm printed experimental receipts")
        elif expected is not None:
            ranks = sorted(r.get("rank") for r in bound if r.get("event") == expected)
            if ranks != list(range(self.tp)):
                reasons.append(f"{expected} receipts from ranks {ranks}; TP{self.tp} needs one per rank")
            if any(r.get("config_sha256") != config.digest for r in bound):
                reasons.append("a receipt was bound to another config")
            for r in bound if config.uses_fixed_layers else ():
                signature = ((r.get("placement") if config.uses_arena else r) or {}).get("signature", {})
                if signature.get("offload_layers") != list(config.placement.offload_layers):
                    reasons.append(f"rank {r.get('rank')}: bound offload layers differ from S")
                if signature.get("resident_path") != ("direct" if config.uses_resident_direct else "resolver"):
                    reasons.append(f"rank {r.get('rank')}: bound resident path differs from the config")
                if not config.uses_arena and signature.get("native_swap") != (
                        "split" if config.uses_split_copy else "fused"):
                    reasons.append(f"rank {r.get('rank')}: bound native swap differs from the config")
        counters = [r for r in records if r.get("event") == "hisparse_shared_lru_counters"]
        return dict(reasons=sorted(set(reasons)), bound=bound, counters=counters)

    def hot_plane_rows(self, receipts):
        rows = [(r.get("signature") or {}).get("hot_plane_rows" if r["event"].startswith("hisparse_layer")
                                                else "plane_rows") for r in receipts["bound"]]
        return min(rows) if rows and None not in rows else None

    def fits(self, info, receipts):
        """Per point: can this arm's real pools hold the planned batch? None where a bound is not reported."""
        logical, hot, g = (info or {}).get("max_total_num_tokens"), self.hot_plane_rows(receipts), \
            self.plan["geometry"]
        out = {}
        for name, point in self.points.items():
            need = point["real_batch"] * (point["input_tokens"][1] + self.m["max_new_tokens"])
            logical_ok = None if logical is None else logical >= need
            hot_ok = None if hot is None else (hot - g["page"]) // (g["hot_rows"] + g["page"]) >= point["real_batch"]
            out[name] = dict(logical_tokens=logical, needed_logical_tokens=need, logical_ok=logical_ok,
                             hot_plane_rows=hot, hot_ok=hot_ok, fits=logical_ok is not False and hot_ok is not False)
        return out

    def batch(self, server, rows):
        """Every row at once on one monotonic clock: (stream records, requests for the window, makespan)."""
        starts, timeout = {}, self.m["request_timeout_s"]

        def one(row):
            return stream.generate(server.url + "/generate", row, timeout,
                                   on_start=lambda t, alias=row["request_alias"]: starts.__setitem__(alias, t))

        began = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(rows)) as pool:
            records = list(pool.map(one, rows))
        makespan = time.perf_counter() - began
        requests = [dict(alias=r["request_alias"], start=starts[r["request_alias"]], successful=r["successful"],
                         finish_reason=r["finish_reason"],
                         events=[(starts[r["request_alias"]] + e["arrival_s"], e["new_tokens"])
                                 for e in r["token_events"]])
                    for r in records]
        return records, requests, makespan

    def warmup(self, server):
        rows = [dict(row, request_alias=f"warmup-{i}", max_new_tokens=8)
                for i, row in enumerate(self.rows["functional"][:self.m["warmup_requests"]])]
        return sum(stream.generate(server.url + "/generate", row, self.m["request_timeout_s"])["successful"]
                   for row in rows)

    def batch_rows(self, point, input_batch, workload):
        size = self.points[point]["real_batch"]
        rows = self.rows[point][input_batch * size:(input_batch + 1) * size]
        return [dict(row, max_new_tokens=self.m["max_new_tokens"], ignore_eos=workload == "fixed") for row in rows]

    def measure(self, entry, slot, arm, run_dir, *, config_path=None, wrapper=()):
        """One fresh server, warmup and one full batch; writes the four result files of the run.

        The fixed workload is judged on its steady full-batch window, natural load on its decode phase.
        """
        m, point, batch, workload = self.m, entry["point"], entry["real_batch"], entry["workload"]
        config = sc.read(config_path)[0] if config_path is not None else self.configs[arm]
        rows = self.batch_rows(point, entry["input_batch"], workload)
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        server, argv = self.launcher.start(arm, run_dir, config_path=config_path, wrapper=wrapper)
        records, requests, makespan, info, memory, warm = [], [], None, None, None, None
        try:
            problem = server.wait_ready(m["server_start_timeout_s"])
            if problem is None:
                try:
                    info = http_json(server.url + "/server_info")
                    warm = self.warmup(server)
                    records, requests, makespan = self.batch(server, rows)
                    memory = self.gpu_memory()
                except (URLError, OSError, ValueError) as error:       # a crashed server is evidence, not a stop
                    problem = f"server request failed: {error}"
        finally:
            exit_code = server.stop(grace=1800 if wrapper else 120)     # nsys writes its report after the workers
        receipts = self.check_receipts(arm, run_dir / "receipts", config)
        reasons = ([problem] if problem else []) + receipts["reasons"]
        window, more = (None, [])
        if requests and workload == "fixed":
            window, more = measure.steady_window(requests, batch=batch, warmup_steps=m["warmup_steps"],
                                                 min_steps=m["min_window_steps"])
            if window is not None and set(window["finish_reasons"]) != {"length"}:
                more.append(f"fixed work ended by {window['finish_reasons']}; every request must reach its budget")
        elif requests:
            window, more = measure.natural_window(requests, batch=batch)
        reasons += more
        real = None
        if window is not None:
            origin = min(r["start"] for r in requests)
            real, more = measure.real_batch(server.samples, batch=batch, start=origin + window["start_s"],
                                            end=origin + window["end_s"], min_samples=m["min_decode_log_samples"],
                                            shrinking=workload == "natural")
            reasons += more
        run = dict(run_id=str(run_dir.relative_to(self.out)), point=point, pair=entry.get("pair"),
                   repeat=entry.get("repeat"), workload=workload, slot=slot, arm=arm)
        write_new(run_dir / "run_config.json", dict(
            run, config_sha256=config.digest, resolved_config=config.as_dict(), capabilities=sc.capabilities(config),
            argv=argv, devices=self.args.devices, plan_sha256=sha256(Path(self.args.plan_dir) / "plan.json"),
            spec_sha256=self.plan["spec_sha256"], manifest_sha256=self.manifest_sha256, measurement=m,
            input_batch=entry["input_batch"], real_batch=batch, order=entry.get("order"), started=started,
            input_sha256=sha256(json.dumps([r["input_ids"] for r in rows]).encode()),
            requests=[r["request_alias"] for r in rows]))
        write_new(run_dir / "admission.json", dict(
            run, server_info=info_excerpt(info), gpu_memory_after_batch=memory, receipts=receipts["bound"],
            fits=self.fits(info, receipts), real_batch=real, server_exit_code=exit_code, warmup_successes=warm))
        write_new(run_dir / "correctness.json", dict(run, outputs={r["request_alias"]: dict(
            output_ids_sha256=r["output_ids_sha256"], output_tokens=r["output_tokens"],
            finish_reason=r["finish_reason"], successful=r["successful"], error=r.get("error")) for r in records}))
        summary = (stream.summarize(records, makespan, m["ttft_slo_s"], m["tpot_slo_s"], len(rows))
                   if records else None)
        return write_new(run_dir / "metrics.json", dict(
            run, status="rejected" if reasons or window is None else "valid", reasons=reasons, window=window,
            real_batch=real, batch_summary=summary, counters=receipts["counters"],
            finished=datetime.datetime.now(datetime.timezone.utc).isoformat()))

    # -- stages --------------------------------------------------------------------------------

    def stage_identity(self):
        reasons, devices = [], self.args.devices.split(",")
        query = "index,uuid,name,memory.total,memory.used,utilization.gpu,clocks.sm,clocks.max.sm,driver_version"
        code, out = self.command(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader", "-i",
                                  self.args.devices])
        gpus = [dict(zip(GPU_FIELDS, (x.strip() for x in line.split(",")))) for line in out.splitlines()
                if line.strip()] if code == 0 else []
        if len(gpus) != len(devices) or len(devices) != self.tp:
            reasons.append(f"selected {len(devices)} devices, nvidia-smi listed {len(gpus)}; the plan runs TP{self.tp}")
        if any("A100" not in gpu["name"] for gpu in gpus):
            reasons.append("the plan targets A100 SXM4 80 GB (sm_80) devices")
        foreign = self.foreign_processes()
        if foreign:
            reasons.append(f"{len(foreign)} foreign compute processes hold the selected GPUs; yield or pick others")
        if self.args.image_id != EXPECTED_IMAGE:
            reasons.append(f"image {self.args.image_id} is not the pinned experiment image {EXPECTED_IMAGE}")
        code, found = self.command([sys.executable, "-c", "import os, sglang; print(os.path.dirname(sglang.__file__))"])
        root = Path(found.strip().splitlines()[-1]) if code == 0 and found.strip() else None
        digests = placement_device.verify_digests(root) if root else {}
        symbols = placement_device.resolve_symbols(root) if root else []
        bad = sorted(name for name, state in digests.items() if state != "ok")
        bad += [s["symbol"] for s in symbols if s["status"] != "ok"]
        if root is None or bad:
            reasons.append("pinned SGLang e7399a46 is not importable" if root is None
                           else f"pinned SGLang mismatch: {bad}")
        config_json = Path(self.args.model_dir) / "config.json"
        weights = dict(declared_sha256=self.args.weights_sha256,
                       config_json_sha256=sha256(config_json) if config_json.is_file() else None)
        if weights["config_json_sha256"] is None or not self.args.weights_sha256:
            reasons.append("pass the verified read-only weights digest (--weights-sha256) and a model directory")
        torch = self.command([sys.executable, "-c", "import json, torch; print(json.dumps([torch.__version__, "
                              "torch.version.cuda]))"])[1].strip()
        result = dict(status="fail" if reasons else "pass", reasons=reasons, devices=devices, gpus=gpus,
                      foreign_processes=foreign, topology=self.command(["nvidia-smi", "topo", "-m"])[1],
                      image_id=self.args.image_id, sglang_root=None if root is None else str(root), digests=digests,
                      symbols=symbols, weights=weights, host=host_limits(self.out),
                      toolchain=dict(python=sys.version.split()[0], torch=torch,
                                     nvcc=self.command(["nvcc", "--version"])[1].strip(),
                                     nsys=self.command(["nsys", "--version"])[1].strip()),
                      plan_sha256=sha256(Path(self.args.plan_dir) / "plan.json"), manifest_sha256=self.manifest_sha256)
        write_new(self.out / "identity.json", result)
        return dict(status=result["status"], reasons=reasons)

    def _jobs(self, name, argv, output):
        (self.out / name).mkdir(parents=True, exist_ok=True)
        code, text = self.command([sys.executable, *argv], env=gpu_env(self.args.devices.split(",")[0]), timeout=7200)
        output.with_suffix(".log").write_text(text)
        ok = code == 0 and output.is_file()
        return ok, dict(exit_code=code, output=str(output.relative_to(self.out)) if output.is_file() else None)

    def stage_build(self):
        arms, geo = self.configs.items(), geometry_args(self.plan["geometry"])
        jobs = {"shared_lru": next((a for a, c in arms if c.uses_arena and c.hot_pool.quota == "none"), None),
                "layer_placement": next((a for a, c in arms if c.uses_fixed_layers and not c.uses_resident_direct),
                                        None),
                "split_copy": next((a for a, c in arms if c.uses_split_copy), None)}
        modules = dict(shared_lru="runtime.shared_lru_device", layer_placement="runtime.layer_placement_device",
                       split_copy="runtime.layer_placement_device")
        results, reasons = {}, []
        for name, arm in jobs.items():
            if arm is None:
                results[name] = dict(skipped="the plan has no arm that launches this kernel")
                continue
            output = self.out / "build" / f"{name}.json"
            ok, results[name] = self._jobs("build", ["-m", modules[name], "build-receipt", "--allow-cuda-build",
                                                     "--config", str(Path(self.args.plan_dir) / f"{arm}.json"),
                                                     "--output", str(output), *geo], output)
            reasons += [] if ok else [f"{name} build-receipt failed: exit {results[name]['exit_code']}"]
        write_new(self.out / "build" / "summary.json", dict(status="fail" if reasons else "pass", reasons=reasons,
                                                            jobs=results))
        return dict(status="fail" if reasons else "pass", reasons=reasons)

    def stage_gate(self):
        """A100 gates of the mechanisms the plan launches: the combined gate (arena over S, resident reads) and
        the fixed-layer gate (stock layout, fused and split swaps) at the small and the production geometry."""
        jobs = []
        if any(c.uses_arena for c in self.configs.values()):
            jobs.append(("combined-gate", ["runtime.cuda_combined_correctness"]))
        if any(c.uses_fixed_layers and not c.uses_arena for c in self.configs.values()):
            jobs += [("fixed-layer-gate", ["runtime.cuda_fixed_layer_correctness"]),
                     ("fixed-layer-gate-production", ["runtime.cuda_fixed_layer_correctness", "--production"])]
        reasons = []
        for name, (module, *extra) in jobs:
            output = self.out / "gate" / f"{name}.json"
            ok, job = self._jobs("gate", ["-m", module, "--allow-cuda-test", *extra, "--output", str(output)], output)
            if not (ok and json.loads(output.read_text()).get("status") == "pass"):
                reasons.append(f"{name} failed: {job}")
        return dict(status="fail" if reasons else "pass", reasons=reasons, gates=[name for name, _ in jobs])

    def scheduled_arms(self):
        return sorted({arm for entry in self.schedule for arm in entry["order"]})

    def stage_functional(self):
        """Real-model outputs: no config and all-default against stock, stock A/A, every scheduled arm."""
        if self.manifest_reasons:
            return dict(status="fail", reasons=self.manifest_reasons)
        m = self.m
        rows = [dict(row, max_new_tokens=m["functional_max_new_tokens"])
                for row in self.rows["functional"][:m["functional_requests"]]]
        arms = [NATIVE, "stock", "stock"] + [arm for arm in self.scheduled_arms() if arm != "stock"]
        report = []
        for index, arm in enumerate(arms):
            run_dir = self.out / "functional" / f"{index:02d}-{arm}"
            server, argv = self.launcher.start(arm, run_dir)
            sequential, concurrent, ids = [], [], {}
            try:
                problem = server.wait_ready(m["server_start_timeout_s"])
                if problem is None:
                    with (run_dir / "output_ids.jsonl").open("x") as ledger:      # private: exact output token IDs
                        sequential = [correctness_client.submit(OPENER, server.url + "/generate", row,
                                                                m["request_timeout_s"], ledger) for row in rows]
                    concurrent = self.batch(server, rows)[0]
            except (URLError, OSError, ValueError) as error:
                problem = f"server request failed: {error}"
            finally:
                exit_code = server.stop()
            if (run_dir / "output_ids.jsonl").is_file():
                ids = {line["request_alias"]: line["output_ids"]
                       for line in map(json.loads, (run_dir / "output_ids.jsonl").read_text().splitlines())}
            receipts = self.check_receipts(arm, run_dir / "receipts")
            report.append(dict(arm=arm, label=run_dir.name, ready=problem, exit_code=exit_code, argv=argv,
                               receipt_reasons=receipts["reasons"], receipt_ranks=len(receipts["bound"]),
                               sequential={r["request_alias"]: r.get("output_ids_sha256") for r in sequential},
                               concurrent={r["request_alias"]: r["output_ids_sha256"] for r in concurrent}, ids=ids))
        stock, again = report[1], report[2]
        repeat = dict(sequential=measure.output_mismatch(stock["sequential"], again["sequential"]),
                      concurrent=measure.output_mismatch(stock["concurrent"], again["concurrent"]))
        tolerance, reasons = self.acceptance["quality_mismatch_over_stock_max"], []
        for entry in report:
            entry["against_stock"] = dict(
                sequential=measure.output_mismatch(stock["sequential"], entry["sequential"]),
                concurrent=measure.output_mismatch(stock["concurrent"], entry["concurrent"]),
                first_divergence={alias: first_divergence(stock["ids"].get(alias), entry["ids"].get(alias))
                                  for alias in stock["ids"]})
            limit = 0.0 if entry["arm"] == NATIVE else tolerance
            problems = ([entry["ready"]] if entry["ready"] else []) + entry["receipt_reasons"] + [
                f"{kind} outputs differ from stock on {rate:.0%} of requests (stock A/A {repeat[kind]:.0%})"
                for kind, rate in entry["against_stock"].items() if kind != "first_divergence"
                and rate > repeat[kind] + limit]
            entry["status"], entry["reasons"] = ("fail" if problems else "pass"), problems
            reasons += [f"{entry['label']}: {p}" for p in problems]
        for entry in report:
            del entry["ids"]                                 # token IDs stay in the private ledgers
        write_new(self.out / "functional" / "correctness.json", dict(
            status="fail" if reasons else "pass", reasons=reasons, stock_repeat_mismatch=repeat, arms=report,
            requests=len(rows), note="no config and the all-default config must match stock like stock A/A"))
        return dict(status="fail" if reasons else "pass", reasons=reasons)

    def stage_capacity(self):
        """Auto-sized pools, receipts and GPU memory of every scheduled arm; which batches each can hold."""
        arms, reasons = {}, []
        for arm in self.scheduled_arms():
            run_dir = self.out / "capacity" / arm
            server, argv = self.launcher.start(arm, run_dir)
            info, memory = None, None
            try:
                problem = server.wait_ready(self.m["server_start_timeout_s"])
                if problem is None:
                    info, memory = http_json(server.url + "/server_info"), self.gpu_memory()
            except (URLError, OSError, ValueError) as error:
                problem = f"server request failed: {error}"
            finally:
                exit_code = server.stop()
            receipts = self.check_receipts(arm, run_dir / "receipts")
            problems, fits = receipts["reasons"], self.fits(info, receipts)
            if problem is not None:                          # startup OOM or failure: evidence, every point infeasible
                fits = {name: dict(value, fits=False, failed_to_start=problem) for name, value in fits.items()}
                problems = problems + ([problem] if arm == "stock" else [])
            reasons += [f"{arm}: {p}" for p in problems]
            arms[arm] = dict(ready=problem, exit_code=exit_code, argv=argv, server_info=info_excerpt(info),
                             gpu_memory=memory, receipts=receipts["bound"], fits=fits)
        infeasible = [dict(point=e["point"], workload=e["workload"], pair=e["pair"],
                           arms=[a for a in e["order"] if not arms[a]["fits"][e["point"]]["fits"]])
                      for e in self.schedule if not all(arms[a]["fits"][e["point"]]["fits"] for a in e["order"])]
        write_new(self.out / "capacity" / "admission.json", dict(
            status="fail" if reasons else "pass", reasons=reasons, arms=arms, infeasible=infeasible,
            note="no --max-total-tokens: pools are SGLang's automatic sizing; infeasible pairs are recorded, not run"))
        return dict(status="fail" if reasons else "pass", reasons=reasons)

    def run_dir(self, entry, slot, arm):
        return (self.out / "runs" / entry["point"] / entry["pair"].replace("->", "__")
                / f"r{entry['repeat']}-{entry['workload']}" / f"{slot}-{arm}")

    def stage_performance(self):
        """AB/BA rounds in plan priority order until the deadline; finished runs are kept on --resume."""
        if self.manifest_reasons:
            return dict(status="fail", reasons=self.manifest_reasons)
        capacity = json.loads((self.out / "capacity" / "admission.json").read_text())
        fits = {(arm, point): value["fits"] for arm, data in capacity["arms"].items()
                for point, value in data["fits"].items()}
        counts, reasons = dict(valid=0, rejected=0, infeasible_rounds=0, not_run=0, kept=0), []
        for entry in self.schedule:
            if not all(fits.get((arm, entry["point"]), False) for arm in entry["order"]):
                counts["infeasible_rounds"] += 1
                continue
            for slot, arm in zip("ab", entry["order"]):
                run_dir = self.run_dir(entry, slot, arm)
                if (run_dir / "metrics.json").is_file():
                    counts["kept"] += 1
                    continue
                now = datetime.datetime.now(datetime.timezone.utc)
                if reasons or (self.args.deadline is not None and now >= self.args.deadline):
                    counts["not_run"] += 1
                    continue
                foreign = self.foreign_processes()
                if foreign:                                  # someone else needs the devices: yield them
                    reasons.append(f"yielded the devices: {len(foreign)} foreign compute processes appeared")
                    counts["not_run"] += 1
                    continue
                if run_dir.exists():                         # an interrupted attempt stays as evidence, renamed
                    attempts = len(list(run_dir.parent.iterdir()))
                    run_dir.rename(run_dir.with_name(f"{run_dir.name}.interrupted-{attempts}"))
                metrics = self.measure(entry, slot, arm, run_dir)
                counts["valid" if metrics["status"] == "valid" else "rejected"] += 1
        if not counts["valid"] + counts["kept"]:
            reasons.append("no run produced a valid window")
        return dict(status="pass" if counts["valid"] + counts["kept"] else "fail", reasons=reasons, **counts)

    def stage_diagnostics(self):
        """Separate waves of the acceptance and split-copy arms: arena counters per rank and layer; optional
        Nsight; never used for clean metrics."""
        if self.manifest_reasons:
            return dict(status="fail", reasons=self.manifest_reasons)
        point = self.args.diagnostics_point or self.plan["pressure_points"][0]["name"]
        entry = dict(point=point, real_batch=self.points[point]["real_batch"], input_batch=0, workload="fixed")
        fields = ("arm", "point", "rank", "layer", "source", *arena.reference.COUNTER_FIELDS, *TIMELINE_FIELDS, "note")
        rows, reasons = [], []
        arms = [self.acceptance["candidate"], *self.acceptance["baselines"]]
        for arm in arms + [a for a, c in self.configs.items() if c.uses_split_copy and a not in arms]:
            config, base = self.configs[arm], self.out / "diagnostics" / arm
            if config.uses_arena:
                path = base / "counters_config.json"
                write_new(path, dict(config.as_dict(), diagnostics={"mode": "counters"}))
                metrics = self.measure(entry, "counters", arm, base / "counters", config_path=path)
                snapshots = metrics["counters"]
                if not snapshots:
                    reasons.append(f"{arm}: no counter snapshot (workers must exit through atexit on SIGINT)")
                for snapshot in snapshots:
                    for layer, counts in sorted(snapshot["layers"].items(), key=lambda item: int(item[0])):
                        rows.append(dict(arm=arm, point=point, rank=snapshot["rank"], layer=layer, source="counters",
                                         note=f"status {snapshot['status']}", **counts))
            else:
                rows.append(dict(arm=arm, point=point, note="stock swap kernel has no per-layer counters; "
                                                            "see the Nsight capture"))
            if self.args.nsys:
                report = base / "nsys" / arm
                report.parent.mkdir(parents=True, exist_ok=True)
                wrapper = ["nsys", "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--force-overwrite=false",
                           "-o", str(report)]
                metrics = self.measure(entry, "nsys", arm, base / "nsys-run", wrapper=wrapper)
                summary, problem = self.timeline(report)
                reasons += [f"{arm}: {problem}"] if problem else []
                rows += timeline_rows(arm, point, summary)
                rows.append(dict(arm=arm, point=point, source="nsys", note=f"nsys report {report}.nsys-rep "
                                 f"({metrics['status']}); timeline {problem or 'pass'}"))
        target = self.out / "diagnostics" / "diagnostic_summary.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return dict(status="fail" if reasons else "pass", reasons=reasons)

    def timeline(self, report):
        """Export one capture to SQLite and summarize its steady decode graph into timeline.json."""
        database = report.with_suffix(".sqlite")
        code, text = self.command(["nsys", "export", "--type", "sqlite", "--output", str(database),
                                   f"{report}.nsys-rep"], timeout=3600)
        if code != 0 or not database.is_file():
            return None, f"nsys export failed with exit {code}: {text.strip()[-300:]}"
        g = self.plan["geometry"]
        try:
            summary = timeline.analyze(database, layers=range(g["first_layer"], g["first_layer"] + g["layers"]),
                                       tp_size=self.tp)
        except timeline.TimelineError as error:
            return None, str(error)
        write_new(report.parent / "timeline.json", summary)
        return summary, "; ".join(summary["reasons"]) or None

    def stage_verdict(self):
        """Paired intervals per point, workload and pair, native A/A noise and the preregistered acceptance of
        each workload, reported side by side."""
        acc, by_round = self.acceptance, {}
        for path in sorted((self.out / "runs").rglob("metrics.json")):
            metrics = json.loads(path.read_text())
            outputs = json.loads((path.parent / "correctness.json").read_text())["outputs"]
            by_round.setdefault((metrics["point"], metrics["workload"], metrics["pair"], metrics["repeat"]), {})[
                metrics["slot"]] = (metrics, {alias: value["output_ids_sha256"] for alias, value in outputs.items()})
        entries = {(e["point"], e["workload"], e["pair"], e["repeat"]): e for e in self.plan["schedule"]}
        blocks, rejected = {}, []
        for key, slots in sorted(by_round.items()):
            entry = entries[key]
            runs = [slots.get(slot) for slot in "ab"]
            if any(run is None or run[0]["status"] != "valid" for run in runs):
                rejected.append(dict(point=key[0], workload=key[1], pair=key[2], repeat=key[3], runs={
                    slot: None if run is None else dict(status=run[0]["status"], reasons=run[0]["reasons"])
                    for slot, run in zip("ab", runs)}))
                continue
            control, candidate = runs if entry["order"][0] == entry["control"] else runs[::-1]
            blocks.setdefault((key[0], key[1], entry["control"], entry["candidate"]), []).append(dict(
                round=key[3], order="AB" if entry["order"][0] == entry["control"] else "BA",
                control=control[0]["window"], candidate=candidate[0]["window"],
                mismatch=measure.output_mismatch(control[1], candidate[1])))
        noise = {(point, workload, b["round"]): b["mismatch"] for (point, workload, control, candidate), items
                 in blocks.items() if control == candidate == "stock" for b in items}
        functional = self.out / "functional" / "correctness.json"
        fallback = json.loads(functional.read_text())["stock_repeat_mismatch"]["concurrent"] if functional.is_file() \
            else 0.0
        results, rules = {}, dict(min_gain=acc["min_gain"], tpot_p95_regression_max=acc["tpot_p95_regression_max"],
                                  min_pairs=acc["min_pairs"])
        for key, items in blocks.items():
            for b in items:
                b["quality_ok"] = b["mismatch"] <= noise.get((key[0], key[1], b["round"]), fallback) + \
                    acc["quality_mismatch_over_stock_max"]
            results[key] = measure.paired(items, **rules)
        capacity = self.out / "capacity" / "admission.json"
        for item in json.loads(capacity.read_text())["infeasible"] if capacity.is_file() else []:
            control, candidate = item["pair"].split("->")
            results.setdefault((item["point"], item["workload"], control, candidate),
                               dict(verdict="infeasible", arms=item["arms"]))
        kinds = {(e["control"], e["candidate"]): (e["kind"], e["acceptance"]) for e in self.plan["schedule"]}
        acceptance = {workload: measure.acceptance(
            {(p, c, d): result for (p, w, c, d), result in results.items() if w == workload},
            candidate=acc["candidate"], baselines=acc["baselines"],
            points=[p["name"] for p in self.plan["pressure_points"]], min_points=acc["min_pressure_points"])
            for workload in self.m["workloads"]}
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        write_new(self.out / f"verdict-{stamp}.json", dict(
            schema="combined-verdict-v1", plan_sha256=sha256(Path(self.args.plan_dir) / "plan.json"),
            manifest_sha256=self.manifest_sha256, rules=dict(rules, ci="two-sided 95% t over independent rounds",
                                                             quality="output-hash mismatch <= stock A/A + tolerance",
                                                             workloads=dict(
                                                                 fixed="ignore_eos; steady full-batch window",
                                                                 natural="EOS ends requests; whole decode phase")),
            acceptance=acceptance, rejected_rounds=rejected,
            results=[dict(point=p, workload=w, control=c, candidate=d, kind=kinds.get((c, d), (None, False))[0],
                          acceptance=kinds.get((c, d), (None, False))[1], **result)
                     for (p, w, c, d), result in sorted(results.items())]))
        return dict(status="pass", reasons=[], file=f"verdict-{stamp}.json",
                    acceptance={workload: value["overall"] for workload, value in acceptance.items()})


def parse_deadline(value):
    deadline = datetime.datetime.fromisoformat(value)
    if deadline.tzinfo is None:
        raise argparse.ArgumentTypeError("the deadline needs a UTC offset, e.g. 2026-11-26T09:00:00+08:00")
    return deadline


def main(argv=None, *, launcher=None, command=run_command):
    from .combined_plan import PRIORITY
    parser = argparse.ArgumentParser(description="Run a compiled combined-study plan stage by stage (GPU host).")
    parser.add_argument("--plan-dir", type=Path, required=True, help="output of runtime.combined_plan")
    parser.add_argument("--manifest", type=Path, required=True, help=f"{MANIFEST_SCHEMA} of the held-out workloads")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--devices", required=True, help="comma-separated GPU indices the maintainer assigns")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-id", required=True, help="content ID of the running experiment image")
    parser.add_argument("--weights-sha256", help="digest of the verified read-only weights")
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--max-priority", type=int, default=len(PRIORITY) - 1,
                        help=f"run pairs up to this priority ({', '.join(f'{i}={k}' for i, k in enumerate(PRIORITY))})")
    parser.add_argument("--points", default="", help="comma-separated pressure points (default: all)")
    parser.add_argument("--deadline", type=parse_deadline, help="end of the authorized window; no run starts after")
    parser.add_argument("--diagnostics-point")
    parser.add_argument("--nsys", action="store_true", help="add Nsight Systems waves to the diagnostics stage")
    parser.add_argument("--port", type=int, default=31080)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args(argv)
    stages = args.stages.split(",")
    devices = args.devices.split(",")
    if not args.allow_gpu:
        parser.error("Run only inside an approved, idle GPU window with --allow-gpu")
    if set(stages) - set(STAGES) or not all(d.isdigit() for d in devices) or len(set(devices)) != len(devices):
        parser.error(f"--stages must name {list(STAGES)} and --devices distinct GPU indices")
    if args.output_dir.exists() and not args.resume:
        parser.error(f"Refusing to reuse {args.output_dir}; pass a new directory or --resume")
    for name in ("NO_PROXY", "no_proxy"):                # the clients talk to 127.0.0.1 only
        hosts = [host for host in os.environ.get(name, "").split(",") if host]
        os.environ[name] = ",".join(hosts + [host for host in ("127.0.0.1", "localhost") if host not in hosts])
    try:
        plan = json.loads((args.plan_dir / "plan.json").read_text())
        if plan.get("schema") != "combined-paired-plan-v1":
            raise ValueError("--plan-dir must hold a combined-paired-plan-v1 plan.json")
        args.output_dir.mkdir(parents=True, exist_ok=args.resume)
        manifest = read_manifest(args.manifest, plan)
        launcher = launcher or Launcher(args.plan_dir, plan, devices=args.devices, model_dir=args.model_dir,
                                        port=args.port)
        runner = Runner(args, plan, manifest, launcher, command)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    status = runner.run(stages)
    print(json.dumps({stage: value["status"] for stage, value in status.items()}, sort_keys=True))
    return 0 if all(status.get(stage, {}).get("status") == "pass" for stage in stages) else 1


if __name__ == "__main__":
    sys.exit(main())