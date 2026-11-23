"""CPU-only validation of budgets, cost coverage and frozen service inputs."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .layer_service_config import parse_service_config, read_service_config
from .request_policy import RequestTypePolicy


def example_config():
    """Synthetic configuration for contracts, not calibrated experiment input."""
    value = dict(schema="layer-hisparse-service-v1", model_identity="synthetic-test@revision",
        expected_tp_size=4, max_context=16384, max_running_requests=8, logical_token_capacity=65536,
        main=dict(layer_count=60, page_tokens=128, kv_bytes_per_token_per_layer=512,
            topk_blocks=16, block_tokens=128, allocatable_pages=[256]*60,
            budget_bytes=257*60*128*512), host_budget_bytes=1<<30, control_timeout_seconds=30,
        mode="pressure", type_policy=dict(hmax=8192, default_type="chat",
            tables=dict(chat=[4096]*60, agent=[8192]*60)),
        request_types={"private-request": "agent"}, unknown_request_action="default",
        cost_model=dict(kind="surrogate", unit="us", horizon_steps=32,
            protocol="synthetic CPU contract test; no measured costs", source_sha256="1"*64,
            ranges=[dict(request_type="chat", min_tokens=4097, max_tokens=16384,
                ranks=[dict(switch_us=[r+1]*60, exposed_us=[2*r+1]*60) for r in range(4)])]),
        eviction=dict(max_migrations=60, cooldown_seconds=.5, min_residency_seconds=1,
            transfer_weight=1, prepared_attempts=2))
    p = value["type_policy"]
    value["cost_model"]["policy_sha256"] = RequestTypePolicy(p["hmax"],
        tuple(sorted((k, tuple(v)) for k,v in p["tables"].items())),
        p["default_type"], value["expected_tp_size"], 2048).digest
    return value


class ServiceConfigTests(unittest.TestCase):
    def test_costs_are_bound_to_exact_frozen_policy(self):
        value = example_config()
        self.assertEqual(parse_service_config(value).cost_model.policy_sha256,
                         parse_service_config(value).type_policy.digest)
        value["type_policy"]["tables"]["chat"][0] = 3072
        with self.assertRaisesRegex(ValueError, "policy_sha256"):
            parse_service_config(value)
        value = example_config()
        del value["cost_model"]["policy_sha256"]
        with self.assertRaises(ValueError):
            parse_service_config(value)

    def test_exact_main_budget_logical_host_separation(self):
        c = parse_service_config(example_config())
        self.assertEqual(c.layout.max_requests, 9)  # row zero is a sink
        self.assertEqual(c.layout.backing_bytes, 257*60*128*512)
        self.assertEqual(c.logical_token_capacity, 65536)
        self.assertEqual(c.host_budget_bytes, 1<<30)
        self.assertEqual(len(c.make_policy()._trigger), 240)
        self.assertEqual(c.make_policy()._cooldown, .5)

    def test_cost_range_and_rank_are_explicit(self):
        c = parse_service_config(example_config())
        self.assertIsNone(c.cost_model.lookup("agent", 0, 8192))
        self.assertIsNone(c.cost_model.lookup("chat", 0, 4096))
        self.assertEqual(c.cost_model.lookup("chat", 3, 4097).exposed_us[55], 7)
        self.assertEqual(c.cost_model.lookup("chat", 0, 16384).switch_us[0], 1)
        self.assertIsNone(c.cost_model.lookup("chat", 0, 16385))
        with self.assertRaises(ValueError):
            c.cost_model.lookup("chat", 4, 5000)

    def test_summary_has_no_manifest_ids(self):
        summary = json.dumps(parse_service_config(example_config()).summary())
        self.assertNotIn("private-request", summary)
        self.assertIn('"request_manifest_count": 1', summary)
        self.assertIn('"cost_kind": "surrogate"', summary)

    def test_reject_malformed_geometry_costs_and_policy(self):
        mutations = [
            lambda v:v["main"].update(budget_bytes=1),
            lambda v:v["main"].update(page_tokens=True),
            lambda v:v.update(logical_token_capacity=16385),
            lambda v:v.update(logical_token_capacity=8192),
            lambda v:v.update(expected_tp_size=2),
            lambda v:v["main"]["allocatable_pages"].__setitem__(2, 10),
            lambda v:v["type_policy"]["tables"]["chat"].__setitem__(2, 2047),
            lambda v:v["cost_model"]["ranges"][0]["ranks"].pop(),
            lambda v:v["cost_model"]["ranges"].append(deepcopy(v["cost_model"]["ranges"][0])),
            lambda v:v["cost_model"]["ranges"][0]["ranks"][0]["switch_us"].__setitem__(0, float("nan")),
            lambda v:v["cost_model"]["ranges"][0]["ranks"][0]["switch_us"].__setitem__(0, True),
            lambda v:v["cost_model"].update(unit="cycles"),
            lambda v:v["cost_model"].update(source_sha256="unknown"),
            lambda v:v["cost_model"].update(ranges=[]),
            lambda v:v.update(host_budget_bytes=0),
            lambda v:v.update(control_timeout_seconds=0),
            lambda v:v.update(unknown_request_action="reject"),
            lambda v:v["request_types"].update(other="unknown-type"),
            lambda v:v["eviction"].update(prepared_attempts=0),
            lambda v:v.update(unrecognized=True),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                value = example_config()
                mutation(value)
                with self.assertRaises((ValueError, TypeError)):
                    parse_service_config(value)

    def test_resident_reference_needs_no_cost_or_host_budget(self):
        value = example_config()
        value["mode"] = "resident_only"
        value["host_budget_bytes"] = 0
        value["cost_model"]["ranges"] = []
        c = parse_service_config(value)
        self.assertEqual(c.mode, "resident_only")

    def test_hash_freeze_duplicates_and_nonfinite_json(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/"config.json"
            raw = json.dumps(example_config()).encode()
            path.write_bytes(raw)
            c, digest = read_service_config(path)
            self.assertEqual(digest, sha256(raw).hexdigest())
            self.assertEqual(read_service_config(path, digest)[0], c)
            path.write_bytes(raw+b" ")
            with self.assertRaisesRegex(ValueError, "changed"):
                read_service_config(path, digest)
            path.write_text('{"schema":"x","schema":"y"}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_service_config(path)
            path.write_text('{"schema":NaN}')
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                read_service_config(path)


if __name__ == "__main__":
    unittest.main()
