# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .action_schema import (
    ROT6D20_ACTION_DIM,
    load_rot6d20_quantile_stats,
    unresolved_rot6d20_quantile_stats,
)
from .va_robotwin_cfg import va_robotwin_cfg


va_robotwin_rot6d20_cfg = EasyDict(__name__="Config: VA robotwin rot6d20")
va_robotwin_rot6d20_cfg.update(va_robotwin_cfg)

# Cosmos and Ctrl-World both consume canonical rot6d20 natively. LingBot uses
# the same representation; only its two 30D checkpoint boundary layers are
# migrated after checkpoint loading.
va_robotwin_rot6d20_cfg.action_schema = "canonical_deltaee_rot6d20"
va_robotwin_rot6d20_cfg.action_dim = ROT6D20_ACTION_DIM
va_robotwin_rot6d20_cfg.raw_action_dim = ROT6D20_ACTION_DIM
va_robotwin_rot6d20_cfg.action_per_frame = int(
    os.getenv("LINGBOT_ACTION_PER_FRAME", "4")
)
va_robotwin_rot6d20_cfg.used_action_channel_ids = list(
    range(ROT6D20_ACTION_DIM)
)
va_robotwin_rot6d20_cfg.inverse_used_action_channel_ids = list(
    range(ROT6D20_ACTION_DIM)
)

model_path = os.getenv("LINGBOT_WAN22_PATH")
if model_path:
    va_robotwin_rot6d20_cfg.wan22_pretrained_model_name_or_path = model_path

stat_path = os.getenv("LINGBOT_ROT6D20_STAT_PATH")
va_robotwin_rot6d20_cfg.norm_stat_path = stat_path
va_robotwin_rot6d20_cfg.norm_stat_is_placeholder = not bool(stat_path)
va_robotwin_rot6d20_cfg.norm_stat = (
    load_rot6d20_quantile_stats(stat_path)
    if stat_path
    else unresolved_rot6d20_quantile_stats()
)
