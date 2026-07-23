import unittest

import numpy as np

from wan_va.dataset.rot6d20_precomputed_dataset import (
    Rot6D20PrecomputedDataset,
)


class Rot6D20PrecomputedDatasetTest(unittest.TestCase):
    def test_native_actions_group_four_per_wan_latent(self):
        dataset = Rot6D20PrecomputedDataset.__new__(Rot6D20PrecomputedDataset)
        dataset.q01 = np.zeros((1, 20), dtype=np.float32)
        dataset.q99 = np.ones((1, 20), dtype=np.float32)
        dataset.action_per_frame = 4
        actions = np.full((32, 20), 0.5, dtype=np.float32)
        action_tensor, mask = dataset._prepare_actions(actions, latent_frames=8)
        self.assertEqual(tuple(action_tensor.shape), (20, 8, 4, 1))
        self.assertEqual(tuple(mask.shape), (20, 8, 4, 1))
        self.assertTrue(mask.all())

    def test_tail_padding_is_masked(self):
        dataset = Rot6D20PrecomputedDataset.__new__(Rot6D20PrecomputedDataset)
        dataset.q01 = np.zeros((1, 20), dtype=np.float32)
        dataset.q99 = np.ones((1, 20), dtype=np.float32)
        dataset.action_per_frame = 4
        actions = np.full((32, 20), 0.5, dtype=np.float32)
        action_tensor, mask = dataset._prepare_actions(actions, latent_frames=9)
        self.assertEqual(tuple(action_tensor.shape), (20, 9, 4, 1))
        self.assertFalse(mask[:, -1].any())
        self.assertTrue((action_tensor[:, -1] == 0).all())


if __name__ == "__main__":
    unittest.main()
