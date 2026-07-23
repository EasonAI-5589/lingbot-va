import unittest
from pathlib import Path

from script.precompute_partition import partition_sample_indices


REPO = Path(__file__).resolve().parents[1]


class TrainingLaunchContractTest(unittest.TestCase):
    def test_fixed_aihc_bundle_freezes_native20_40k_contract(self):
        script = (
            REPO
            / "docs/actionfollowing/aihc/run_lingbot_full50_40k_native20_fixed.sh"
        ).read_text(encoding="utf-8")
        required = (
            "export LINGBOT_ENV_FILE=/dev/null",
            "export LINGBOT_PRECOMPUTE_PYTHON=/mnt/gyc/cosmos-framework/.venv/bin/python",
            "[PRECOMPUTE_ENV]",
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

    def test_precompute_can_resume_across_a_new_rank_topology(self):
        precompute = (
            REPO / "script/precompute_actionfollowing_native20.py"
        ).read_text(encoding="utf-8")
        launcher = (
            REPO / "script/run_rot6d20_native20_train_aihc.sh"
        ).read_text(encoding="utf-8")
        bootstrap = (
            REPO
            / "docs/actionfollowing/aihc/run_lingbot_full50_40k_native20_fixed.sh"
        ).read_text(encoding="utf-8")

        for token in (
            '"--resume"',
            "collect_resume_records",
            "partition_sample_indices",
            "resume_manifest.jsonl",
            "publish_sample",
            "PRECOMPUTE_RESUME_DISCOVERY",
            "PRECOMPUTE_RANK_RESULT",
            "timeout=timedelta(hours=6)",
        ):
            self.assertIn(token, precompute)
        for token in (
            '--nnodes="${NODE_WORLD_SIZE}"',
            '--node_rank="${NODE_RANK}"',
            "PRECOMPUTE_WORKER_DONE",
            "LINGBOT_PRECOMPUTE_RESUME",
        ):
            self.assertIn(token, launcher)
        for token in (
            "LINGBOT_PRECOMPUTE_REUSE_ROOT",
            "PRECOMPUTE_CONTRACT",
            "bootstrap.worker",
            "PRECOMPUTE_WORKER_EXIT",
        ):
            self.assertIn(token, bootstrap)

    def test_resume_partition_balances_missing_work_across_ranks(self):
        num_samples = 600
        world_size = 16
        reusable = set(range(503))
        partitions = [
            partition_sample_indices(num_samples, reusable, rank, world_size)
            for rank in range(world_size)
        ]

        flattened = [index for partition in partitions for index in partition]
        self.assertEqual(sorted(flattened), list(range(num_samples)))
        self.assertEqual(len(flattened), len(set(flattened)))

        computed_counts = [
            sum(index not in reusable for index in partition)
            for partition in partitions
        ]
        reused_counts = [
            sum(index in reusable for index in partition)
            for partition in partitions
        ]
        self.assertLessEqual(max(computed_counts) - min(computed_counts), 1)
        self.assertLessEqual(max(reused_counts) - min(reused_counts), 1)


if __name__ == "__main__":
    unittest.main()
