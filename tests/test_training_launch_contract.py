import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


class TrainingLaunchContractTest(unittest.TestCase):
    def test_fixed_aihc_bundle_freezes_native20_40k_contract(self):
        script = (
            REPO
            / "docs/actionfollowing/aihc/run_lingbot_full50_40k_native20_fixed.sh"
        ).read_text(encoding="utf-8")
        required = (
            "export LINGBOT_ENV_FILE=/dev/null",
            "clean) precompute_samples=472622",
            "mix4) precompute_samples=600000",
            "export LINGBOT_NUM_STEPS=40000",
            "export LINGBOT_SAVE_INTERVAL=5000",
            "action_shape=[32,20]",
            "action_dim=20",
            "effective_batch=8",
            "checkpoint_step_40000",
        )
        for token in required:
            self.assertIn(token, script)

    def test_legacy_preflight_no_longer_expands_to_30d(self):
        script = (REPO / "script/preflight_rot6d20_dataset.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("LINGBOT30", script)
        self.assertNotIn("model_action_dim=30", script)
        self.assertIn("model_action_dim=20", script)


if __name__ == "__main__":
    unittest.main()
