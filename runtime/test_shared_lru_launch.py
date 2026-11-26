"""CPU tests for --hisparse-study-config in the project launcher (no SGLang import)."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from runtime import launch

ARMS = Path(__file__).parent / "fixtures" / "shared_lru_arms"
BASE = ["--enable-hisparse", "--tp-size", "8", "--hisparse-config", '{"top_k":2048,"device_buffer_size":4096}']
SINGLE_STREAM = ["--disable-overlap-schedule", "--disable-radix-cache"]


def resolve(*argv):
    with contextlib.redirect_stderr(io.StringIO()):
        return launch.resolve(list(argv))


class StudyConfigLauncher(unittest.TestCase):
    def test_sm80_adapter_is_packaged_and_installed(self):
        import os
        from hashlib import sha256
        source = Path(launch.__file__).with_name("compat") / "minimax_sm80.py"
        self.assertEqual(sha256(source.read_bytes()).hexdigest(), launch.SM80_SHA256)
        plan, _, _ = resolve("--hisparse-study-config", str(ARMS / "shared_arena.json"),
                             "--sm80-bf16-projections", "--", *BASE, *SINGLE_STREAM)
        calls = []
        adapter = types.ModuleType("runtime.compat.minimax_sm80")
        adapter.install = lambda: calls.append(("sm80", os.environ.get(launch.SM80_ENV)))
        with patch("runtime.shared_lru_device.install", lambda config, digest: calls.append(("arena", digest))), \
                patch.dict(sys.modules, {"runtime.compat.minimax_sm80": adapter}), patch.dict(os.environ):
            launch.install_for_process(plan)
        self.assertEqual(calls, [("arena", plan.study_config_sha256), ("sm80", "1")])

    def test_arena_arm_resolves_and_prints_receipt(self):
        plan, upstream, dry = resolve("--hisparse-study-config", str(ARMS / "shared_arena.json"), "--dry-run",
                                      "--", *BASE, *SINGLE_STREAM)
        self.assertTrue(dry)
        self.assertEqual(upstream, BASE + SINGLE_STREAM)
        self.assertEqual(Path(plan.study_config_path).name, "shared_arena.json")
        self.assertEqual(len(plan.study_config_sha256), 64)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(launch.main(["--hisparse-study-config", str(ARMS / "shared_arena.json"), "--dry-run",
                                          "--", *BASE, *SINGLE_STREAM]), 0)
        summary = json.loads(out.getvalue())
        self.assertEqual(summary["mode"], "hisparse_study_config")
        self.assertEqual(summary["study_capabilities"]["runtime"], "shared_lru_arena")
        self.assertEqual(summary["study_resolved_config"]["hot_pool"]["mode"], "shared")
        # The spawn payload round-trips through LaunchPlan(**json).
        self.assertEqual(launch.LaunchPlan(**json.loads(json.dumps(launch.asdict(plan)))), plan)

    def test_stock_study_arm_keeps_stock_flags(self):
        plan, _, _ = resolve("--hisparse-study-config", str(ARMS / "stock.json"), "--", *BASE)
        self.assertIsNotNone(plan.study_config_sha256)

    def test_rejections_happen_before_any_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixed = Path(tmp) / "fixed.json"         # route_indirect without eager prefill
            fixed.write_text(json.dumps({"schema": "hisparse-study-config-v1",
                                         "placement": {"mode": "fixed_layers", "offload_layers": [3]},
                                         "features": {"route_indirect": True}}))
            typo = Path(tmp) / "typo.json"
            typo.write_text('{"schema": "hisparse-study-config-v1", "hot_pool": {"mode": "shared"}}')
            shared = str(ARMS / "shared_arena.json")
            cases = [
                ("--hisparse-study-config", shared, "--", *BASE),                                  # needs single stream
                ("--hisparse-study-config", shared, "--", *BASE, "--disable-overlap-schedule"),
                ("--hisparse-study-config", shared, "--", *BASE, *SINGLE_STREAM, "--speculative-algorithm", "EAGLE"),
                ("--hisparse-study-config", shared, "--", *BASE, *SINGLE_STREAM, "--disaggregation-mode", "decode"),
                ("--hisparse-study-config", shared, "--", *BASE, *SINGLE_STREAM, "--enable-two-batch-overlap"),
                ("--hisparse-study-config", shared, "--", *BASE, *SINGLE_STREAM,
                 "--cuda-graph-config", '{"decode": {"backend": "breakable"}}'),
                ("--hisparse-study-config", shared, "--six-layer-timing", "--", *BASE, *SINGLE_STREAM),
                ("--hisparse-study-config", shared, "--policy", shared, "--", *BASE, *SINGLE_STREAM),
                ("--hisparse-study-config", str(fixed), "--", *BASE, *SINGLE_STREAM),
                ("--hisparse-study-config", str(typo), "--", *BASE, *SINGLE_STREAM),
            ]
            for argv in cases:
                with self.subTest(argv=argv[2:]), self.assertRaises(SystemExit):
                    resolve(*argv)
        plan, _, _ = resolve("--hisparse-study-config", shared, "--", *BASE, *SINGLE_STREAM,
                             "--cuda-graph-config", '{"decode": {"backend": "full", "bs": [1, 2, 4]}}')
        self.assertIsNotNone(plan.study_config_path)

    def test_worker_installs_arena_only_for_arena_layouts(self):
        calls = []
        shared_plan, _, _ = resolve("--hisparse-study-config", str(ARMS / "shared_arena.json"), "--",
                                    *BASE, *SINGLE_STREAM)
        stock_plan, _, _ = resolve("--hisparse-study-config", str(ARMS / "stock.json"), "--", *BASE)
        sglang = types.ModuleType("sglang")
        sglang.__file__ = str(Path(tempfile.gettempdir()) / "sglang" / "__init__.py")
        with patch("runtime.shared_lru_device.install", lambda config, digest: calls.append(("arena", digest))), \
                patch("runtime.hook.verify_sources", lambda root: calls.append(("stock", None))), \
                patch.dict(sys.modules, {"sglang": sglang}):
            launch.install_for_process(shared_plan)
            launch.install_for_process(stock_plan)
        self.assertEqual(calls, [("arena", shared_plan.study_config_sha256), ("stock", None)])
        tampered = launch.LaunchPlan(**(launch.asdict(shared_plan) | {"study_config_sha256": "0" * 64}))
        with self.assertRaisesRegex(ValueError, "changed"):
            launch.install_for_process(tampered)


if __name__ == "__main__":
    unittest.main()
