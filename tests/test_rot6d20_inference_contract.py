import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file


SCRIPT = Path(__file__).parents[1] / "script" / "prepare_rot6d20_inference_bundle.py"
SPEC = importlib.util.spec_from_file_location("prepare_rot6d20_inference_bundle", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Rot6D20InferenceContractTest(unittest.TestCase):
    def test_prepare_bundle_repairs_config_without_modifying_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            checkpoint = tmp_path / "checkpoint_step_50000"
            transformer = checkpoint / "transformer"
            transformer.mkdir(parents=True)
            source_config = transformer / "config.json"
            source_config.write_text(json.dumps({"action_dim": 30, "attn_mode": "flex"}), encoding="utf-8")
            save_file(
                {
                    "action_embedder.weight": torch.zeros(8, 20),
                    "action_proj_out.weight": torch.zeros(20, 8),
                },
                transformer / "diffusion_pytorch_model.safetensors",
            )
            base = tmp_path / "base"
            for name in ("vae", "tokenizer", "text_encoder"):
                (base / name).mkdir(parents=True)
            stat = tmp_path / "stat.json"
            stat.write_text(
                json.dumps({"action_dim": 20, "q01": [0] * 20, "q99": [1] * 20}), encoding="utf-8"
            )

            output = tmp_path / "bundle"
            audit = MODULE.prepare_bundle(checkpoint, base, output, stat)
            repaired = json.loads((output / "transformer" / "config.json").read_text())
            original = json.loads(source_config.read_text())

            self.assertEqual(original, {"action_dim": 30, "attn_mode": "flex"})
            self.assertEqual(repaired["action_dim"], 20)
            self.assertEqual(repaired["attn_mode"], "torch")
            self.assertEqual(audit["tensor_shapes"]["action_embedder.weight"], [8, 20])
            self.assertTrue((output / "vae").is_symlink())
            self.assertEqual((output / "vae").resolve(), (base / "vae").resolve())

    def test_config_registry_contains_rot6d20_i2va(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stat = Path(directory) / "stat.json"
            stat.write_text(
                json.dumps({"action_dim": 20, "q01": [0] * 20, "q99": [1] * 20}), encoding="utf-8"
            )
            old_stat = os.environ.get("LINGBOT_ROT6D20_STAT_PATH")
            os.environ["LINGBOT_ROT6D20_STAT_PATH"] = str(stat)
            try:
                from wan_va.configs import VA_CONFIGS

                config = VA_CONFIGS["robotwin_rot6d20_i2va"]
                self.assertEqual(config.action_dim, 20)
                self.assertEqual(config.action_schema, "canonical_rot6d20")
                self.assertEqual(config.infer_mode, "i2va")
                self.assertEqual(config.future_rgb_frames, 32)
                self.assertEqual(config.vae_temporal_stride, 4)
            finally:
                if old_stat is None:
                    os.environ.pop("LINGBOT_ROT6D20_STAT_PATH", None)
                else:
                    os.environ["LINGBOT_ROT6D20_STAT_PATH"] = old_stat


if __name__ == "__main__":
    unittest.main()
