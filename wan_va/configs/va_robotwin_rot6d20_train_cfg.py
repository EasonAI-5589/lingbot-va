# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_robotwin_rot6d20_cfg import va_robotwin_rot6d20_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg


va_robotwin_rot6d20_train_cfg = EasyDict(
    __name__="Config: VA robotwin native rot6d20 train"
)
va_robotwin_rot6d20_train_cfg.update(va_robotwin_train_cfg)
va_robotwin_rot6d20_train_cfg.update(va_robotwin_rot6d20_cfg)

va_robotwin_rot6d20_train_cfg.dataset_backend = os.getenv(
    "LINGBOT_DATASET_BACKEND", "rot6d20_precomputed"
)
va_robotwin_rot6d20_train_cfg.precomputed_manifest = os.getenv(
    "LINGBOT_PRECOMPUTED_MANIFEST", "/path/to/precomputed/manifest.jsonl"
)
va_robotwin_rot6d20_train_cfg.dataset_manifest = os.getenv(
    "LINGBOT_ROT6D20_MANIFEST_PATH", "/path/to/rot6d20/manifests/train.jsonl"
)
va_robotwin_rot6d20_train_cfg.action_data_root = os.getenv(
    "LINGBOT_ROT6D20_ACTION_ROOT", "/path/to/rot6d20"
)
va_robotwin_rot6d20_train_cfg.lingbot_latent_root = os.getenv(
    "LINGBOT_ROT6D20_LATENT_ROOT", "/path/to/robotwin-clean-and-aug-lerobot"
)
va_robotwin_rot6d20_train_cfg.empty_emb_path = os.path.join(
    va_robotwin_rot6d20_train_cfg.lingbot_latent_root, "empty_emb.pt"
)
va_robotwin_rot6d20_train_cfg.dataset_family = os.getenv(
    "LINGBOT_ROT6D20_FAMILY", "clean"
)
task_filter = os.getenv("LINGBOT_ROT6D20_TASKS", "")
va_robotwin_rot6d20_train_cfg.dataset_tasks = [
    task.strip() for task in task_filter.split(",") if task.strip()
]
va_robotwin_rot6d20_train_cfg.max_samples = int(
    os.getenv("LINGBOT_ROT6D20_MAX_SAMPLES", "0")
)
va_robotwin_rot6d20_train_cfg.require_action_norm_stat = True
va_robotwin_rot6d20_train_cfg.enable_wandb = os.getenv(
    "LINGBOT_ENABLE_WANDB", "0"
) == "1"
va_robotwin_rot6d20_train_cfg.load_worker = int(
    os.getenv("LINGBOT_LOAD_WORKERS", "8")
)
va_robotwin_rot6d20_train_cfg.num_steps = int(
    os.getenv("LINGBOT_NUM_STEPS", "50000")
)
va_robotwin_rot6d20_train_cfg.save_interval = int(
    os.getenv("LINGBOT_SAVE_INTERVAL", "1000")
)
