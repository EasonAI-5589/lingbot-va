import unittest

import numpy as np

from wan_va.dataset.rot6d20_latent_manifest_dataset import (
    Rot6D20LatentManifestDataset,
)


class Rot6D20LatentManifestDatasetTest(unittest.TestCase):
    def test_action_shape_matches_lingbot_temporal_layout(self):
        dataset = Rot6D20LatentManifestDataset.__new__(
            Rot6D20LatentManifestDataset
        )
        dataset.q01 = np.zeros((1, 20), dtype=np.float32)
        dataset.q99 = np.ones((1, 20), dtype=np.float32)
        actions = np.full((240, 20), 0.5, dtype=np.float32)
        frame_ids = list(range(0, 241, 4))

        action_tensor, action_mask = dataset._prepare_actions(
            actions, frame_ids, latent_frames=16
        )
        self.assertEqual(tuple(action_tensor.shape), (20, 16, 16, 1))
        self.assertEqual(tuple(action_mask.shape), (20, 16, 16, 1))
        self.assertTrue(action_mask.all())


if __name__ == "__main__":
    unittest.main()
