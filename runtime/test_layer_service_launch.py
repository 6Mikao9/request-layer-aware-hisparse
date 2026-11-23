"""Launch contracts with no CUDA or model loading; stubs are explicit."""
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch

from . import layer_service_launch as launch
from .layer_service_config import parse_service_config
from .test_layer_service_config import example_config


def upstream():
    return ["--model-path", "/readonly/model", "--tp-size", "4", "--context-length", "16384",
        "--max-running-requests", "8", "--max-total-tokens", "65536", "--page-size", "128",
        "--attention-backend", "triton", "--dtype", "bfloat16", "--kv-cache-dtype", "auto",
        "--disable-cuda-graph", "--disable-prefill-cuda-graph", "--disable-overlap-schedule",
        "--disable-radix-cache"]


class ServiceLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.path = Path(self.temp.name)/"config.json"
        self.path.write_text(json.dumps(example_config()), encoding="utf-8")
        self.environment = patch.dict(os.environ)
        self.environment.start()
        for name in (launch.SPAWN_ENV, launch.OLD_SPAWN_ENV, launch.SM80_ENV):
            os.environ.pop(name, None)

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()

    def args(self):
        return ["--service-config", str(self.path), "--"] + upstream()

    def test_valid_aliases_and_mismatched_or_unsafe_cli(self):
        config = parse_service_config(example_config())
        launch.check_upstream(upstream(), config)
        for extra in (["--enable-hisparse"], ["--hisparse-config", "{}"], ["--tensor-parallel-size", "4"],
                      ["--dp-size", "2"], ["--config", "file.json"],
                      ["--tp", "4"], ["--attn-dcp-size", "1"],
                      ["--prefill-attention-backend", "triton"], ["--cuda-graph-bs", "4"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                launch.check_upstream(upstream()+extra, config)
        for flag in ("--disable-overlap-schedule", "--disable-radix-cache"):
            args = upstream()
            args.remove(flag)
            with self.assertRaises(ValueError):
                launch.check_upstream(args, config)

    def test_dry_run_imports_neither_torch_nor_sglang_and_hides_manifest(self):
        command = [sys.executable, "-B", "-c",
            "import sys; from runtime.layer_service_launch import main; "
            "main(sys.argv[1:]); assert 'torch' not in sys.modules; "
            "assert not any(n == 'sglang' or n.startswith('sglang.') for n in sys.modules)",
            "--dry-run"]+self.args()
        result = subprocess.run(command, capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1], timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("private-request", result.stdout)
        self.assertTrue(json.loads(result.stdout)["dry_run"])

    def test_enabled_diagnostics_dry_run_and_spawn_digest(self):
        value = example_config()
        root = Path(self.temp.name).resolve()
        value["catalog_diagnostics"] = dict(schema="layer-catalog-probe-v1",
            output_directory=str(root), run_id="capture-run", workload_id="workload",
            runtime_sha256="a" * 64, first_prepared=0, every_prepared=2, max_captures=1,
            profiles=[dict(name="profile", type_policy=value["type_policy"],
                           cost_model=value["cost_model"])])
        self.path.write_text(json.dumps(value), encoding="utf-8")
        command = [sys.executable, "-B", "-c",
            "import sys; from runtime.layer_service_launch import main; "
            "main(sys.argv[1:]); assert 'torch' not in sys.modules; "
            "assert not any(n == 'sglang' or n.startswith('sglang.') for n in sys.modules)",
            "--dry-run"] + self.args()
        result = subprocess.run(command, capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1], timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertTrue(summary["catalog_diagnostics_enabled"])
        self.assertEqual(summary["measurement_role"], "diagnostic_only")
        self.assertFalse((root / "capture-run").exists())

        plan, _, _, _ = launch.resolve(self.args())
        os.environ[launch.SPAWN_ENV] = json.dumps(asdict(plan))
        value["catalog_diagnostics"]["every_prepared"] = 3
        self.path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            launch.install_spawned_worker()
        self.assertFalse((root / "capture-run").exists())

    def test_changed_config_rejected_before_worker_install(self):
        plan, _, _, _ = launch.resolve(self.args())
        self.path.write_text(self.path.read_text()+" ")
        with self.assertRaisesRegex(ValueError, "changed"):
            launch.install_for_process(plan)

    def test_spawn_env_revalidates_and_rejects_unknown_fields(self):
        plan, _, _, _ = launch.resolve(self.args())
        calls = []
        stub = ModuleType("runtime.layer_worker_bootstrap")
        stub.install = lambda config,digest,enabled: calls.append((config,digest,enabled))
        os.environ[launch.SPAWN_ENV] = json.dumps(asdict(plan))
        with patch.dict(sys.modules, {stub.__name__:stub}):
            launch.install_spawned_worker()
        self.assertEqual(calls[0][1], plan.config_sha256)
        self.assertTrue(calls[0][2])
        os.environ[launch.SPAWN_ENV] = json.dumps(asdict(plan)|{"extra":1})
        with self.assertRaisesRegex(ValueError, "fields"):
            launch.install_spawned_worker()

    def test_parent_launch_injects_entry_and_restores_environment_on_failure(self):
        calls = []
        stub = ModuleType("runtime.layer_worker_bootstrap")
        stub.run_scheduler_process = lambda *args: None
        http = ModuleType("sglang.srt.entrypoints.http_server")
        args_module = ModuleType("sglang.srt.server_args")
        args_module.prepare_server_args = lambda args: args
        def server(args, **kwargs):
            calls.append(kwargs)
            self.assertIn(launch.SPAWN_ENV, os.environ)
            raise RuntimeError("test server failure")
        http.launch_server = server
        with patch.dict(sys.modules,{stub.__name__:stub,http.__name__:http,args_module.__name__:args_module}), \
                patch.object(launch, "install_for_process"), \
                patch.object(launch.multiprocessing, "get_start_method", return_value="spawn"), \
                redirect_stdout(StringIO()), self.assertRaisesRegex(RuntimeError, "test server failure"):
            launch.main(self.args())
        self.assertIs(calls[0]["run_scheduler_process_func"], stub.run_scheduler_process)
        self.assertNotIn(launch.SPAWN_ENV, os.environ)

    def test_real_multiprocessing_spawn_reimports_entry(self):
        # This smoke uses a tiny fake bootstrap, not a fake SGLang service.
        # It proves __mp_main__ executes frozen-plan installation in a child.
        script = Path(self.temp.name)/"spawn_check.py"
        receipt = Path(self.temp.name)/"receipt.txt"
        script.write_text('''import os, sys, types, runpy, multiprocessing
from pathlib import Path
from runtime import layer_service_launch as launch
def child():
    assert os.environ.get("TEST_CHILD_INSTALLED") == "yes"
if __name__ == "__mp_main__":
    stub = types.ModuleType("runtime.layer_worker_bootstrap")
    def install(config, digest, enabled):
        assert enabled and config.expected_tp_size == 4
        os.environ["TEST_CHILD_INSTALLED"] = "yes"
        Path(os.environ["TEST_RECEIPT"]).write_text(digest)
    stub.install = install
    sys.modules[stub.__name__] = stub
    runpy.run_module("runtime.layer_service_launch", run_name="__mp_main__")
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    p=multiprocessing.Process(target=child); p.start(); p.join(20)
    if p.is_alive(): p.terminate(); p.join(); raise RuntimeError("spawn timeout")
    assert p.exitcode == 0
''', encoding="utf-8")
        plan, _, _, _ = launch.resolve(self.args())
        env = dict(os.environ, TEST_RECEIPT=str(receipt),
            PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        env[launch.SPAWN_ENV] = json.dumps(asdict(plan))
        result = subprocess.run([sys.executable,"-B",str(script)],env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(receipt.read_text(), plan.config_sha256)


if __name__ == "__main__":
    unittest.main()
