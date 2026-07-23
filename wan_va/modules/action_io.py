# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Checkpoint migration for a native rot6d20 LingBot action boundary."""

from __future__ import annotations

import torch
from torch import nn

from wan_va.configs.action_schema import ROT6D20_RAW_TO_LINGBOT30_COPY


def migrate_action_io_to_native20(model, target_action_dim: int = 20) -> dict:
    """Replace only LingBot's 30D action input/output layers with native 20D.

    Position and gripper parameters are copied because their physical meaning
    is unchanged. Rotation parameters are intentionally initialized by
    ``nn.Linear`` because Rot6D and quaternion coordinates have no fixed linear
    channel correspondence.
    """

    old_embedder = model.action_embedder
    old_projector = model.action_proj_out
    source_action_dim = old_embedder.in_features
    if old_projector.out_features != source_action_dim:
        raise ValueError(
            "LingBot action input/output dimensions disagree: "
            f"{source_action_dim} vs {old_projector.out_features}"
        )
    if source_action_dim == target_action_dim:
        return {
            "source_action_dim": source_action_dim,
            "target_action_dim": target_action_dim,
            "copied_channels": list(range(target_action_dim)),
            "reinitialized_channels": [],
            "changed": False,
        }
    if source_action_dim != 30 or target_action_dim != 20:
        raise ValueError(
            f"Unsupported action migration {source_action_dim}D -> {target_action_dim}D"
        )

    new_embedder = nn.Linear(
        target_action_dim,
        old_embedder.out_features,
        bias=old_embedder.bias is not None,
        device=old_embedder.weight.device,
        dtype=old_embedder.weight.dtype,
    )
    new_projector = nn.Linear(
        old_projector.in_features,
        target_action_dim,
        bias=old_projector.bias is not None,
        device=old_projector.weight.device,
        dtype=old_projector.weight.dtype,
    )

    copied_channels = sorted(ROT6D20_RAW_TO_LINGBOT30_COPY)
    with torch.no_grad():
        for raw_channel, old_channel in ROT6D20_RAW_TO_LINGBOT30_COPY.items():
            new_embedder.weight[:, raw_channel].copy_(
                old_embedder.weight[:, old_channel]
            )
            new_projector.weight[raw_channel].copy_(
                old_projector.weight[old_channel]
            )
            if new_projector.bias is not None:
                new_projector.bias[raw_channel].copy_(
                    old_projector.bias[old_channel]
                )
        if new_embedder.bias is not None:
            new_embedder.bias.copy_(old_embedder.bias)

    model.action_embedder = new_embedder
    model.action_proj_out = new_projector
    if hasattr(model, "config"):
        model.config.action_dim = target_action_dim

    return {
        "source_action_dim": source_action_dim,
        "target_action_dim": target_action_dim,
        "copied_channels": copied_channels,
        "reinitialized_channels": sorted(
            set(range(target_action_dim)) - set(copied_channels)
        ),
        "changed": True,
    }
