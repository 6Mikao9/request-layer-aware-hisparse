"""CPU contracts for immutable split routing; no CUDA or serving side effects."""
from dataclasses import FrozenInstanceError
import json
import unittest
from runtime.policy import StaticPolicy
from runtime.split_copy import SplitCopyConfig, execution_identity


class SplitCopyContracts(unittest.TestCase):
    def test_json_roundtrip_and_frozen_config(self):
        config = SplitCopyConfig((5, 15, 25))
        self.assertEqual(SplitCopyConfig.from_json(json.dumps(config.as_dict())), config)
        self.assertEqual(SplitCopyConfig.from_json(config.as_dict()).digest, config.digest)
        with self.assertRaises(FrozenInstanceError):
            config.copy_grid = 1
        for layers in ((), [5], (5, 5), (15, 5), (-1,), (60,), (True,)):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                SplitCopyConfig(layers)
        for grid, rows in ((1, 2), (64, 3), (True, 2), (64, True)):
            with self.assertRaises(ValueError):
                SplitCopyConfig((5,), grid, rows)

    def test_strict_json_fields_and_duplicates(self):
        config = SplitCopyConfig((5,)).as_dict()
        for data in ({}, dict(config, extra=True), dict(config, layers=(5,)), []):
            with self.assertRaises(ValueError):
                SplitCopyConfig.from_json(data)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            SplitCopyConfig.from_json('{"layers":[5],"copy_grid":64,"plan_rows":2,"plan_rows":2}')

    def test_batch_shape_routing_and_layer_exclusion(self):
        config = SplitCopyConfig((5, 15))
        for batch in (1, 2):
            self.assertTrue(config.uses_split(5, batch))
            self.assertFalse(config.uses_split(6, batch))
        for batch in (0, 3, 8):
            self.assertFalse(config.uses_split(5, batch))

    def test_one_collective_identity_detects_mode_and_layer_mismatch(self):
        policy = StaticPolicy(8192, (4096,) * 60)
        identities = [execution_identity(policy, config) for config in
                      (None, SplitCopyConfig((5,)), SplitCopyConfig((15,)))]
        self.assertEqual(len(set(identities)), 3)
        self.assertNotEqual(identities[1], execution_identity(StaticPolicy(8192, (8192,) * 60), SplitCopyConfig((5,))))


if __name__ == '__main__':
    unittest.main()
