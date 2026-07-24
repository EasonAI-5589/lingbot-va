import unittest

import numpy as np

from wan_va.dataset.rot6d20_precomputed_dataset import (
    Rot6D20PrecomputedDataset,
)
from wan_va.temporal_contract import (
    build_current1_future_video_contract,
    split_condition_and_future_frames,
)


class Rot6D20TemporalParityTest(unittest.TestCase):
    def setUp(self):
        self.contract = build_current1_future_video_contract()

    def test_video_only_inference_matches_training_latent_timeline(self):
        self.assertEqual(self.contract.condition_rgb_frames, 1)
        self.assertEqual(self.contract.future_rgb_frames, 32)
        self.assertEqual(self.contract.full_chunk_latent_frames, 8)
        self.assertEqual(self.contract.tail_video_latent_frames, 1)
        self.assertEqual(self.contract.total_video_latent_frames, 9)
        self.assertEqual(self.contract.decoded_rgb_frames_with_condition, 33)
        self.assertEqual(self.contract.internal_aux_action_steps, 32)

    def test_training_dataloader_masks_only_ninth_action_group(self):
        dataset = Rot6D20PrecomputedDataset.__new__(Rot6D20PrecomputedDataset)
        dataset.q01 = np.zeros((1, 20), dtype=np.float32)
        dataset.q99 = np.ones((1, 20), dtype=np.float32)
        dataset.action_per_frame = self.contract.action_steps_per_latent
        actions = np.full((32, 20), 0.5, dtype=np.float32)

        action_tensor, action_mask = dataset._prepare_actions(
            actions,
            latent_frames=self.contract.total_video_latent_frames,
        )

        self.assertEqual(tuple(action_tensor.shape), (20, 9, 4, 1))
        self.assertTrue(action_mask[:, :8].all())
        self.assertFalse(action_mask[:, 8:].any())
        self.assertTrue((action_tensor[:, 8:] == 0).all())

    def test_decoder_exports_exactly_32_future_frames(self):
        decoded = np.arange(33)
        future = split_condition_and_future_frames(
            decoded,
            future_rgb_frames=self.contract.future_rgb_frames,
        )
        np.testing.assert_array_equal(future, np.arange(1, 33))

    def test_legacy_29_frame_output_fails_parity(self):
        with self.assertRaisesRegex(
            ValueError,
            "actual=29, expected=33",
        ):
            split_condition_and_future_frames(
                np.arange(29),
                future_rgb_frames=self.contract.future_rgb_frames,
            )


if __name__ == "__main__":
    unittest.main()
