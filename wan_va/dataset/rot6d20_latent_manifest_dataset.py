# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Pair canonical rot6d20 actions with existing LingBot Wan latents."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from einops import rearrange


class Rot6D20LatentManifestDataset(torch.utils.data.Dataset):
    """Native-20D training data without a LeRobot-version conversion.

    The Action-Following manifest supplies the canonical frame-wise DeltaEE
    rot6d20 actions. The official LingBot RoboTwin dataset supplies Wan2.2
    48-channel latents for the exact same clean task/episode trajectories.
    """

    def __init__(self, config):
        if getattr(config, "norm_stat_is_placeholder", False):
            raise ValueError(
                "rot6d20 training requires LINGBOT_ROT6D20_STAT_PATH"
            )
        if config.dataset_family != "clean":
            raise ValueError(
                "The existing LingBot latent bridge is verified only for the "
                f"clean family, got {config.dataset_family!r}"
            )

        self.config = config
        self.cfg_prob = float(config.cfg_prob)
        self.used_video_keys = tuple(config.obs_cam_keys)
        self.q01 = np.asarray(config.norm_stat["q01"], dtype=np.float32)[None]
        self.q99 = np.asarray(config.norm_stat["q99"], dtype=np.float32)[None]
        self.empty_emb = torch.load(
            config.empty_emb_path, map_location="cpu", weights_only=False
        )

        self.manifest_path = Path(config.dataset_manifest)
        self.action_root = Path(config.action_data_root)
        self.latent_root = (
            Path(config.lingbot_latent_root)
            / "lerobot_robotwin_eef_clean_50"
        )
        self.records = self._build_records()

    def _task_roots(self) -> dict[str, Path]:
        if not self.latent_root.is_dir():
            raise FileNotFoundError(f"LingBot latent root missing: {self.latent_root}")
        roots = {}
        for path in self.latent_root.iterdir():
            if not path.is_dir() or "-" not in path.name:
                continue
            task = path.name.split("-", 1)[0]
            if task in roots:
                raise ValueError(f"Duplicate LingBot clean latent root for {task}")
            roots[task] = path
        return roots

    def _build_records(self) -> list[dict]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"rot6d20 manifest missing: {self.manifest_path}")
        task_roots = self._task_roots()
        selected_tasks = set(self.config.dataset_tasks)
        records = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                if record.get("family") != self.config.dataset_family:
                    continue
                task = record["task"]
                if selected_tasks and task not in selected_tasks:
                    continue
                if task not in task_roots:
                    raise FileNotFoundError(
                        f"No LingBot clean latent root for manifest task {task} "
                        f"at line {line_number}"
                    )

                episode_index = int(record["episode_index"])
                latent_length = int(record["latent_length"])
                episode_chunk = episode_index // 1000
                latent_paths = []
                for camera in self.used_video_keys:
                    latent_path = (
                        task_roots[task]
                        / "latents"
                        / f"chunk-{episode_chunk:03d}"
                        / camera
                        / (
                            f"episode_{episode_index:06d}_0_"
                            f"{latent_length}.pth"
                        )
                    )
                    if not latent_path.is_file():
                        raise FileNotFoundError(
                            f"Missing LingBot latent paired with {task}/"
                            f"episode_{episode_index:04d}: {latent_path}"
                        )
                    latent_paths.append(latent_path)

                action_path = self.action_root / record["action_file"]
                if not action_path.is_file():
                    raise FileNotFoundError(f"Missing rot6d20 action: {action_path}")
                action = np.load(action_path, mmap_mode="r")
                expected_shape = (int(record["action_length"]), 20)
                if action.shape != expected_shape:
                    raise ValueError(
                        f"{action_path}: expected {expected_shape}, got {action.shape}"
                    )

                records.append(
                    {
                        "task": task,
                        "episode_index": episode_index,
                        "latent_length": latent_length,
                        "latent_paths": tuple(latent_paths),
                        "action_path": action_path,
                    }
                )
                if self.config.max_samples > 0 and len(records) >= self.config.max_samples:
                    break
        if not records:
            raise ValueError(
                f"No {self.config.dataset_family} records selected from {self.manifest_path}"
            )
        return records

    @staticmethod
    def _unflatten_latent(data: dict) -> torch.Tensor:
        return rearrange(
            data["latent"],
            "(f h w) c -> f h w c",
            f=int(data["latent_num_frames"]),
            h=int(data["latent_height"]),
            w=int(data["latent_width"]),
        )

    def _load_latents(self, record: dict) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        camera_data = [
            torch.load(path, map_location="cpu", weights_only=False)
            for path in record["latent_paths"]
        ]
        frame_ids = list(camera_data[0]["frame_ids"])
        for data, path in zip(camera_data, record["latent_paths"]):
            if list(data["frame_ids"]) != frame_ids:
                raise ValueError(f"Camera frame_ids disagree for {path}")
            if data["latent"].shape[-1] != 48:
                raise ValueError(f"Expected 48-channel Wan latent in {path}")

        latents = [self._unflatten_latent(data) for data in camera_data]
        wrist_latent = torch.cat(latents[1:], dim=2)
        combined = torch.cat([wrist_latent, latents[0]], dim=1)
        text_emb = camera_data[0]["text_emb"]
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb
        return combined.permute(3, 0, 1, 2), text_emb, frame_ids

    def _prepare_actions(
        self, actions: np.ndarray, frame_ids: list[int], latent_frames: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(frame_ids) < 2:
            raise ValueError("At least two sampled frame_ids are required")
        frame_stride = int(frame_ids[1] - frame_ids[0])
        if frame_stride <= 0 or any(
            int(right - left) != frame_stride
            for left, right in zip(frame_ids[:-1], frame_ids[1:])
        ):
            raise ValueError(f"Non-uniform latent frame_ids: {frame_ids[:12]}")

        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 20:
            raise ValueError(f"Expected [T,20] rot6d20 actions, got {actions.shape}")
        actions = actions[int(frame_ids[0]) :]
        prefix_actions = frame_stride * 4
        actions = np.pad(actions, ((prefix_actions, 0), (0, 0)))
        required_actions = latent_frames * frame_stride * 4
        if actions.shape[0] < required_actions:
            raise ValueError(
                f"Only {actions.shape[0]} actions after history padding, "
                f"need {required_actions}"
            )
        actions = actions[:required_actions]

        action_mask = np.ones_like(actions, dtype=bool)
        actions = (actions - self.q01) / (self.q99 - self.q01 + 1e-6) * 2.0 - 1.0
        actions = np.clip(actions, -1.5, 1.5)
        actions = rearrange(
            actions, "(f n) c -> c f n 1", f=latent_frames
        )
        action_mask = rearrange(
            action_mask, "(f n) c -> c f n 1", f=latent_frames
        )
        return torch.from_numpy(actions).float(), torch.from_numpy(action_mask)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        latents, text_emb, frame_ids = self._load_latents(record)
        actions = np.load(record["action_path"])
        action_tensor, action_mask = self._prepare_actions(
            actions, frame_ids, latents.shape[1]
        )
        return {
            "latents": latents,
            "text_emb": text_emb,
            "actions": action_tensor,
            "actions_mask": action_mask,
        }

    def __len__(self) -> int:
        return len(self.records)
