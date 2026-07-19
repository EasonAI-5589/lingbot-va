# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Datasets with lazy legacy LeRobot imports."""

from .rot6d20_latent_manifest_dataset import Rot6D20LatentManifestDataset
from .rot6d20_precomputed_dataset import Rot6D20PrecomputedDataset

__all__ = [
    "MultiLatentLeRobotDataset",
    "Rot6D20LatentManifestDataset",
    "Rot6D20PrecomputedDataset",
]


def __getattr__(name):
    if name == "MultiLatentLeRobotDataset":
        from .lerobot_latent_dataset import MultiLatentLeRobotDataset

        return MultiLatentLeRobotDataset
    raise AttributeError(name)
