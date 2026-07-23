import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from wan_va.configs.action_schema import (
    ROT6D20_ACTION_DIM,
    ROT6D20_RAW_TO_LINGBOT30_COPY,
    load_rot6d20_quantile_stats,
)
from wan_va.modules.action_io import migrate_action_io_to_native20


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_embedder = nn.Linear(30, 32)
        self.action_proj_out = nn.Linear(32, 30)


class Rot6D20ActionSchemaTest(unittest.TestCase):
    def test_ctrlworld_stat_schema_stays_native20(self):
        payload = {
            "state_01": [0.0] * ROT6D20_ACTION_DIM,
            "state_99": [1.0] * ROT6D20_ACTION_DIM,
            "action_dim": ROT6D20_ACTION_DIM,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stat.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            stats = load_rot6d20_quantile_stats(path)
        self.assertEqual(len(stats["q01"]), ROT6D20_ACTION_DIM)
        self.assertEqual(len(stats["q99"]), ROT6D20_ACTION_DIM)

    def test_migration_replaces_only_action_boundary_shapes(self):
        torch.manual_seed(7)
        model = DummyModel()
        old_embedder = model.action_embedder.weight.detach().clone()
        old_projector = model.action_proj_out.weight.detach().clone()
        result = migrate_action_io_to_native20(model)

        self.assertTrue(result["changed"])
        self.assertEqual(model.action_embedder.in_features, 20)
        self.assertEqual(model.action_proj_out.out_features, 20)
        self.assertEqual(
            result["reinitialized_channels"],
            sorted(set(range(20)) - set(ROT6D20_RAW_TO_LINGBOT30_COPY)),
        )
        for raw_channel, old_channel in ROT6D20_RAW_TO_LINGBOT30_COPY.items():
            torch.testing.assert_close(
                model.action_embedder.weight[:, raw_channel],
                old_embedder[:, old_channel],
            )
            torch.testing.assert_close(
                model.action_proj_out.weight[raw_channel],
                old_projector[old_channel],
            )


if __name__ == "__main__":
    unittest.main()
