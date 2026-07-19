#!/usr/bin/env python3
"""Load real paired latents/actions before allocating a training process group."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wan_va.configs import VA_CONFIGS
from wan_va.dataset import Rot6D20PrecomputedDataset


def main() -> None:
    config = VA_CONFIGS["robotwin_rot6d20_train"]
    if config.dataset_backend != "rot6d20_precomputed":
        raise ValueError(
            f"Expected rot6d20_precomputed backend, got {config.dataset_backend}"
        )
    dataset = Rot6D20PrecomputedDataset(config)
    checked = []
    for index in sorted(set((0, len(dataset) - 1))):
        sample = dataset[index]
        shapes = {
            key: list(value.shape)
            for key, value in sample.items()
            if hasattr(value, "shape")
        }
        if shapes["latents"][0] != 48:
            raise ValueError(f"Expected 48 latent channels, got {shapes['latents']}")
        if shapes["actions"][0] != 20:
            raise ValueError(f"Expected native 20D actions, got {shapes['actions']}")
        if shapes["latents"][1] != shapes["actions"][1]:
            raise ValueError(f"Latent/action temporal mismatch: {shapes}")
        checked.append({"index": index, "shapes": shapes})

    summary = {
        "status": "PASS",
        "dataset_backend": config.dataset_backend,
        "records": len(dataset),
        "action_dim": config.action_dim,
        "manifest": config.precomputed_manifest,
        "checked": checked,
        "job_name": os.getenv("AIHC_JOB_NAME", "local-preflight"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
