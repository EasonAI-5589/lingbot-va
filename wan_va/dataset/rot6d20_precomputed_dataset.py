# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Dataset for native rot6d20 samples precomputed from canonical AFD video."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from einops import rearrange


class Rot6D20PrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        if getattr(config, "norm_stat_is_placeholder", False):
            raise ValueError(
                "rot6d20 training requires LINGBOT_ROT6D20_STAT_PATH"
            )
        self.q01 = np.asarray(config.norm_stat["q01"], dtype=np.float32)[None]
        self.q99 = np.asarray(config.norm_stat["q99"], dtype=np.float32)[None]
        self.action_per_frame = int(config.action_per_frame)
        manifest = Path(config.precomputed_manifest)
        if not manifest.is_file():
            raise FileNotFoundError(f"Precomputed manifest missing: {manifest}")
        self.records = [json.loads(line) for line in manifest.open() if line.strip()]
        if config.max_samples > 0:
            self.records = self.records[: config.max_samples]
        if not self.records:
            raise ValueError(f"No samples in {manifest}")
        self.root = manifest.parent
        for record in self.records:
            path = self.root / record["file"]
            if not path.is_file():
                raise FileNotFoundError(f"Precomputed sample missing: {path}")

    def _prepare_actions(
        self, actions: np.ndarray, latent_frames: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 20:
            raise ValueError(f"Expected [T,20] rot6d20 actions, got {actions.shape}")
        required = latent_frames * self.action_per_frame
        mask = np.ones_like(actions, dtype=bool)
        if actions.shape[0] < required:
            padding = required - actions.shape[0]
            actions = np.pad(actions, ((0, padding), (0, 0)))
            mask = np.pad(mask, ((0, padding), (0, 0)), constant_values=False)
        actions = actions[:required]
        mask = mask[:required]
        actions = (actions - self.q01) / (self.q99 - self.q01 + 1e-6) * 2.0 - 1.0
        actions = np.clip(actions, -1.5, 1.5)
        actions = rearrange(
            actions, "(f n) c -> c f n 1", f=latent_frames
        )
        mask = rearrange(mask, "(f n) c -> c f n 1", f=latent_frames)
        actions *= mask
        return torch.from_numpy(actions).float(), torch.from_numpy(mask)

    def __getitem__(self, index: int) -> dict:
        path = self.root / self.records[index]["file"]
        sample = torch.load(path, map_location="cpu", weights_only=False)
        latents = sample["latents"]
        text_emb = sample["text_emb"]
        actions, action_mask = self._prepare_actions(
            sample["actions"], int(latents.shape[1])
        )
        if latents.ndim != 4 or latents.shape[0] != 48:
            raise ValueError(f"Expected [48,F,H,W] Wan latent in {path}")
        return {
            "latents": latents,
            "text_emb": text_emb,
            "actions": actions,
            "actions_mask": action_mask,
        }

    def __len__(self) -> int:
        return len(self.records)
