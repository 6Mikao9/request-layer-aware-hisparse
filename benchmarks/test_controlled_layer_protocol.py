import copy
import unittest

from benchmarks.controlled_layer_protocol import SAMPLE, validate_pair


def arms():
    fixed = dict(image_id="sha256:" + "a" * 64, model_config_sha256="a" * 64,
                 runtime_sources_sha256="b" * 64, workload_sha256="c" * 64,
                 warmup_workload_sha256="d" * 64, device_uuids=["GPU-" + str(i) for i in range(8)],
                 tp_size=8, precision="fixed-nvfp4-sm80-bf16", top_k=2048, physical_hmax=8192,
                 pool_capacity=131072, host_ratio=2, context_length=32768, client_concurrency=8,
                 max_running_requests=8, graph_config={"bs": [1, 2, 4, 8]}, overlap_schedule=False,
                 radix_cache=False, generation_settings={"temperature": 0, "ignore_eos": False},
                 cache_seed_sha256="e" * 64, kernel_variant="fused", split_copy_config=None,
                 speculative_decode=False, layer_policy_adapter="static-active-prefix")
    a = dict(fixed=fixed, active_capacity_by_layer=[4096] * 60)
    b = copy.deepcopy(a)
    for layer, capacity in zip(SAMPLE, (3584, 3584, 5120, 5120, 4096, 3072)):
        b["active_capacity_by_layer"][layer] = capacity
    return a, b


class ControlledProtocolTest(unittest.TestCase):
    def test_six_layer_budget_and_same_adapter_are_explicit(self):
        result = validate_pair(*arms(), scope="six_layers")
        self.assertEqual(result["changed_layers"], [5, 15, 25, 35, 55])
        self.assertEqual(result["active_slots_per_request_per_rank"], 245760)
        self.assertFalse(result["timing_measured"])

    def test_rejects_unrelated_optimization_and_runtime_confounders(self):
        for key, value in [("kernel_variant", "split"), ("split_copy_config", {}),
                           ("graph_config", {"bs": [1, 2]}), ("client_concurrency", 4),
                           ("overlap_schedule", True), ("workload_sha256", "f" * 64)]:
            a, b = arms()
            b["fixed"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_pair(a, b, scope="six_layers")

    def test_rejects_budget_drift_and_out_of_scope_changes(self):
        a, b = arms()
        b["active_capacity_by_layer"][0] += 128
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_pair(a, b, scope="six_layers")
        b["active_capacity_by_layer"][1] -= 128
        with self.assertRaisesRegex(ValueError, "prespecified"):
            validate_pair(a, b, scope="six_layers")
        self.assertIn(0, validate_pair(a, b, scope="all_layers")["changed_layers"])

    def test_unaccounted_control_fields_and_duplicate_gpu_rejected(self):
        a, b = arms()
        for arm in (a, b):
            arm["fixed"]["unreviewed_tuning"] = True
        with self.assertRaisesRegex(ValueError, "fields"):
            validate_pair(a, b, scope="six_layers")
        a, b = arms()
        for arm in (a, b):
            arm["fixed"]["device_uuids"][1] = arm["fixed"]["device_uuids"][0]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_pair(a, b, scope="six_layers")


if __name__ == "__main__":
    unittest.main()
