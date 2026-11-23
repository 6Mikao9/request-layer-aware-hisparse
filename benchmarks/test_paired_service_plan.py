from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import stat
from types import SimpleNamespace
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from . import paired_service_plan as planner
from runtime.test_layer_service_config import example_config
from runtime.test_layer_service_launch import upstream
from runtime.request_policy import RequestTypePolicy
from runtime.layer_service_config import parse_service_config


class PairedServicePlanTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        original = example_config()
        original["cost_model"]["ranges"].append(dict(deepcopy(original["cost_model"]["ranges"][0]),
                                                     request_type="agent"))
        uniform = deepcopy(original)
        uniform["type_policy"]["tables"]["agent"] = [4096] * 60
        policy = uniform["type_policy"]
        uniform["cost_model"]["policy_sha256"] = RequestTypePolicy(
            policy["hmax"], tuple(sorted((k, tuple(v)) for k, v in policy["tables"].items())),
            policy["default_type"], uniform["expected_tp_size"], 2048).digest
        self.configs = [uniform, original]
        argv = upstream() + ["--random-seed", "7", "--host", "127.0.0.1", "--port", "31080"]
        workload = self.write("work.jsonl", [
            dict(request_alias="private-request", workload="agent", input_ids=[1, 2],
                 max_new_tokens=16380)])
        warmup = self.write("warm.jsonl", [
            dict(request_alias="warm", workload="chat", input_ids=[2], max_new_tokens=16381)])
        self.spec = dict(schema=planner.SCHEMA, stage="closed_loop_smoke", runtime_sha256="a" * 64,
            execution_root="/experiment/input", output_root="/experiment/output", upstream=argv,
            endpoint="http://127.0.0.1:31080/generate", sm80_bf16_projections=True,
            repetitions=2, seed=7, client=dict(concurrency=2, timeout_s=30,
                ttft_slo_s=1, tpot_slo_s=.1), workload=workload, warmup=warmup,
            arrival_schedule=None, arms=[
                dict(name="uniform", kind="uniform_target", service_config=self.write("a.json", uniform)),
                dict(name="typed", kind="type_layer_target", service_config=self.write("b.json", original))])

    def write(self, name, value):
        if isinstance(value, bytes):
            raw = value
        elif name.endswith(".jsonl"):
            raw = ("\n".join(json.dumps(row) for row in value) + "\n").encode()
        else:
            raw = json.dumps(value).encode()
        (self.root / name).write_bytes(raw)
        return dict(path=name, sha256=sha256(raw).hexdigest())

    def manifest(self):
        reference = self.write("plan.json", self.spec)
        return self.root / reference["path"], reference["sha256"]

    def compile(self):
        return planner.compile_plan(*self.manifest())

    def cli(self, *extra):
        path, digest = self.manifest()
        return subprocess.run([sys.executable, "-B", "-c",
            "import sys; from benchmarks.paired_service_plan import main; "
            "main(); assert 'torch' not in sys.modules; "
            "assert not any(n == 'sglang' or n.startswith('sglang.') for n in sys.modules)",
            "--manifest", str(path), "--sha256", digest,
            "--output", str(self.root / "report.json"), *extra],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, encoding="utf-8", timeout=15)

    def test_real_cli_emits_balanced_unique_templates_only(self):
        run = self.cli("--require-command-templates")
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads((self.root / "report.json").read_text())
        self.assertTrue(result["command_templates_ready"])
        jobs = result["jobs"]
        self.assertEqual([job["arm"] for job in jobs], ["uniform", "typed", "typed", "uniform"])
        self.assertEqual(len({job["output_directory"] for job in jobs}), 4)
        self.assertEqual(result["physical_main_bytes_per_rank"], self.configs[0]["main"]["budget_bytes"])
        self.assertEqual(result["host_budget_bytes_per_rank"], self.configs[0]["host_budget_bytes"])
        self.assertEqual(result["logical_token_capacity"], self.configs[0]["logical_token_capacity"])
        self.assertEqual(result["seed"], 7)
        for job in jobs:
            commands = job["commands"]
            self.assertEqual(set(commands), {"server", "warmup", "measurement"})
            self.assertEqual(commands["server"][-len(self.spec["upstream"]):], self.spec["upstream"])
            self.assertIn("--sm80-bf16-projections", commands["server"])
            self.assertIn("/experiment/input/warm.jsonl", commands["warmup"])
            self.assertIn("/experiment/input/work.jsonl", commands["measurement"])
        self.assertEqual({p.name for p in self.root.iterdir()},
                         {"a.json", "b.json", "work.jsonl", "warm.jsonl", "plan.json", "report.json"})

    def test_rejects_undeclared_axis_changes(self):
        mutations = (
            lambda c: c.update(host_budget_bytes=c["host_budget_bytes"] + 1),
            lambda c: c["eviction"].update(cooldown_seconds=2),
            lambda c: c["cost_model"].update(horizon_steps=33),
            lambda c: c["cost_model"]["ranges"][0].update(max_tokens=12000),
        )
        for mutate in mutations:
            config = deepcopy(self.configs[1])
            mutate(config)
            self.spec["arms"][1]["service_config"] = self.write("b.json", config)
            with self.assertRaises(ValueError):
                self.compile()
        self.spec["arms"][1]["service_config"] = self.write("b.json", self.configs[1])
        self.spec["seed"] = 8
        with self.assertRaisesRegex(ValueError, "seed"):
            self.compile()

    def test_open_loop_plan_validates_schedule_and_preserves_common_warmup(self):
        self.spec["stage"] = "open_loop_admission"
        schedule = dict(schema="hisparse-arrival-schedule-v1",
            workload_sha256=self.spec["workload"]["sha256"],
            phases=[dict(name="low", start_s=0, end_s=1), dict(name="high", start_s=1, end_s=2),
                    dict(name="recovery", start_s=2, end_s=3)],
            arrivals=[dict(request_alias="private-request", at_s=1.2, phase="high")])
        self.spec["arrival_schedule"] = self.write("arrivals.json", schedule)
        result = self.compile()
        self.assertTrue(result["command_templates_ready"])
        for job in result["jobs"]:
            commands = job["commands"]
            self.assertIn("benchmarks.sglang_stream_client", commands["warmup"])
            self.assertIn("benchmarks.sglang_arrival_client", commands["measurement"])
            self.assertIn("--max-workers", commands["measurement"])
            self.assertIn("/experiment/input/arrivals.json", commands["measurement"])
            self.assertNotIn("--concurrency", commands["measurement"])
        schedule["arrivals"][0]["request_alias"] = "unbound"
        self.spec["arrival_schedule"] = self.write("arrivals.json", schedule)
        with self.assertRaises(ValueError):
            self.compile()

    def test_executable_multih_must_have_its_own_honest_arm_label(self):
        for c in self.configs:
            for bucket in c["cost_model"]["ranges"]:
                bucket["min_tokens"] = 1
        self.spec["arms"][0]["service_config"] = self.write("a.json", self.configs[0])
        config = deepcopy(self.configs[1])
        small = deepcopy(self.configs[0])
        small["type_policy"]["tables"] = {label: [2048]*60 for label in small["type_policy"]["tables"]}
        p = small["type_policy"]
        small["cost_model"]["policy_sha256"] = RequestTypePolicy(p["hmax"],
            tuple(sorted((k,tuple(v)) for k,v in p["tables"].items())), p["default_type"],
            small["expected_tp_size"], 2048).digest
        config["capacity_profiles"] = [dict(name=name, type_policy=c["type_policy"], cost_model=c["cost_model"])
            for name,c in (("small",small),("large",self.configs[0]))]
        self.spec["arms"][1]["service_config"] = self.write("b.json", config)
        with self.assertRaisesRegex(ValueError, "cannot silently"):
            self.compile()
        self.spec["arms"][1]["kind"] = "multih_time_surrogate"
        result = self.compile()
        self.assertTrue(result["command_templates_ready"])
        arm = result["arms"][1]
        self.assertEqual(arm["selector"], "multih_time_useful_bytes_greedy")
        self.assertEqual([p["name"] for p in arm["capacity_profiles"]], ["small","large"])
        self.spec["arms"][1]["service_config"] = self.write("b.json", self.configs[1])
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            self.compile()

    def test_rejects_enabled_diagnostics_and_ignored_arrivals(self):
        self.spec["arrival_schedule"] = self.spec["workload"]
        with self.assertRaisesRegex(ValueError, "ignored arrival"):
            self.compile()
        self.spec["arrival_schedule"] = None
        config = deepcopy(self.configs[1])
        config["catalog_diagnostics"] = dict(schema="layer-catalog-probe-v1",
            output_directory="/experiment/diagnostics", run_id="diag", workload_id="workload",
            runtime_sha256="a" * 64, first_prepared=0, every_prepared=1, max_captures=1,
            profiles=[dict(name="probe", type_policy=config["type_policy"], cost_model=config["cost_model"])])
        self.spec["arms"][1]["service_config"] = self.write("b.json", config)
        with self.assertRaisesRegex(ValueError, "disable catalog diagnostics"):
            self.compile()

    def test_rejects_workload_budget_type_and_duplicate_keys(self):
        row = dict(request_alias="private-request", workload="agent",
                   input_ids=[1, 2], max_new_tokens=16380)
        for change in (dict(max_new_tokens=16), dict(max_new_tokens=16381), dict(workload="chat")):
            self.spec["workload"] = self.write("work.jsonl", [row | change])
            with self.assertRaises(ValueError):
                self.compile()
        raw = json.dumps(row)[:-1] + ', "max_new_tokens": 16380}'
        self.spec["workload"] = self.write("work.jsonl", raw.encode())
        with self.assertRaises(ValueError):
            self.compile()

    def test_rejects_missing_and_hash_mismatched_pinned_references(self):
        self.spec["workload"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.compile()
        self.spec["workload"]["path"] = "not-found.jsonl"
        with self.assertRaises((ValueError, OSError)):
            self.compile()

    def test_pending_stage_or_selector_cannot_produce_runnable_commands(self):
        self.spec["stage"] = "fixed_batch_intervention"
        result = self.compile()
        self.assertFalse(result["command_templates_ready"])
        self.assertTrue(all(j["commands"] is None for j in result["jobs"]))
        rejected = self.cli("--require-command-templates")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("pending", rejected.stderr)
        self.assertFalse((self.root / "report.json").exists())
        self.spec["stage"] = "closed_loop_smoke"
        self.spec["arms"][1]["kind"] = "whole_plan_time"
        result = self.compile()
        self.assertFalse(result["command_templates_ready"])
        self.assertTrue(all(j["commands"] is None for j in result["jobs"] if j["arm"] == "typed"))

    def test_private_area_is_rejected_without_opening_it(self):
        configured = os.pathsep.join(("local-sensitive", "second-sensitive"))
        with patch.dict(os.environ, {"HISPARSE_PRIVATE_PATH_COMPONENTS": configured}):
            for component in ("private-data", "LOCAL-SENSITIVE", "second-sensitive"):
                fake = self.root / component / "not-created.json"
                with self.subTest(component=component), patch.object(
                        planner, "_read_pin", side_effect=AssertionError("unexpected I/O")):
                    with self.assertRaisesRegex(ValueError, "Private area"):
                        planner.compile_plan(fake, "0" * 64)
                    with self.assertRaisesRegex(ValueError, "canonical absolute container directory"):
                        planner._directory(f"/experiment/{component}/output")

    def test_native_prompt_ceiling_headroom_and_legacy_limit(self):
        config = parse_service_config(self.configs[0])
        with self.assertRaisesRegex(ValueError, "input ceiling"):
            planner.native_generation_budget(config, 16378)
        compact = deepcopy(self.configs[0])
        compact["logical_token_capacity"] = 16384
        config = parse_service_config(compact)
        self.assertEqual(planner.native_generation_budget(config, 1), 16127)
        with self.assertRaisesRegex(ValueError, "no generation capacity"):
            planner.native_generation_budget(config, 16256)
        self.spec["workload"] = self.write("work.jsonl", [
            dict(request_alias="private-request", workload="agent", input_ids=[1, 2],
                 max_new_tokens=16382)])
        with self.assertRaises(ValueError):
            self.compile()

    def test_pressure_workload_requires_agent_cost_coverage(self):
        for index, name in enumerate(("a.json", "b.json")):
            config = deepcopy(self.configs[index])
            config["cost_model"]["ranges"] = [r for r in config["cost_model"]["ranges"]
                                               if r["request_type"] != "agent"]
            self.spec["arms"][index]["service_config"] = self.write(name, config)
        with self.assertRaisesRegex(ValueError, "outside frozen cost coverage"):
            self.compile()

    def test_link_component_rejected_before_pinned_read(self):
        path, digest = self.manifest()
        real_lstat = type(path).lstat
        for mode, attributes in ((stat.S_IFLNK, 0), (stat.S_IFDIR, 0x400)):
            def lstat(part):
                if part == self.root:
                    return SimpleNamespace(st_mode=mode, st_file_attributes=attributes)
                return real_lstat(part)

            with (patch.object(type(path), "lstat", lstat),
                  patch.object(planner, "_read_pin") as reader):
                with self.assertRaisesRegex(ValueError, "Symlink/reparse"):
                    planner.compile_plan(path, digest)
                reader.assert_not_called()
