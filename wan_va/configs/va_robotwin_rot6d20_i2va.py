# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_robotwin_rot6d20_cfg import va_robotwin_rot6d20_cfg


va_robotwin_rot6d20_i2va_cfg = EasyDict(
    __name__="Config: VA robotwin native rot6d20 minimal i2va"
)
va_robotwin_rot6d20_i2va_cfg.update(va_robotwin_rot6d20_cfg)
va_robotwin_rot6d20_i2va_cfg.input_img_path = os.getenv(
    "LINGBOT_INPUT_IMAGE_DIR", "/path/to/actionfollowing/input"
)
va_robotwin_rot6d20_i2va_cfg.num_chunks_to_infer = int(
    os.getenv("LINGBOT_NUM_CHUNKS_TO_INFER", "4")
)
va_robotwin_rot6d20_i2va_cfg.prompt = os.getenv(
    "LINGBOT_PROMPT", "Place the burger beside the fries."
)
va_robotwin_rot6d20_i2va_cfg.save_root = os.getenv(
    "LINGBOT_INFERENCE_OUTPUT", "visualization/rot6d20_minimal"
)
va_robotwin_rot6d20_i2va_cfg.infer_mode = "i2va"
