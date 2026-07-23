# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Canonical Action-Following rot6d20 schema and stats helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence


ROT6D20_ACTION_DIM = 20

ROT6D20_CHANNEL_NAMES = (
    "left.pos.x",
    "left.pos.y",
    "left.pos.z",
    "left.rot6.r00",
    "left.rot6.r10",
    "left.rot6.r20",
    "left.rot6.r01",
    "left.rot6.r11",
    "left.rot6.r21",
    "left.gripper",
    "right.pos.x",
    "right.pos.y",
    "right.pos.z",
    "right.rot6.r00",
    "right.rot6.r10",
    "right.rot6.r20",
    "right.rot6.r01",
    "right.rot6.r11",
    "right.rot6.r21",
    "right.gripper",
)

# Only channels with the same physical meaning are transplanted from the
# pretrained LingBot 30D boundary layers. Rot6D has no linear one-to-one map
# to the old quaternion channels, so all twelve rotation channels are freshly
# initialized.
ROT6D20_RAW_TO_LINGBOT30_COPY = {
    0: 0,
    1: 1,
    2: 2,
    9: 28,
    10: 7,
    11: 8,
    12: 9,
    19: 29,
}


def _extract_quantiles(raw: Mapping) -> tuple[Sequence[float], Sequence[float]]:
    block = raw.get("global", raw)
    if "q01" in block and "q99" in block:
        return block["q01"], block["q99"]
    if "state_01" in block and "state_99" in block:
        return block["state_01"], block["state_99"]
    raise KeyError(
        "Stats JSON must contain q01/q99, global.q01/global.q99, or "
        "state_01/state_99"
    )


def validate_quantile_stats(
    q01: Sequence[float], q99: Sequence[float]
) -> dict[str, list[float]]:
    q01 = [float(value) for value in q01]
    q99 = [float(value) for value in q99]
    if len(q01) != ROT6D20_ACTION_DIM or len(q99) != ROT6D20_ACTION_DIM:
        raise ValueError(
            f"Expected {ROT6D20_ACTION_DIM}D quantiles, "
            f"got q01={len(q01)} and q99={len(q99)}"
        )
    for index, (lo, hi) in enumerate(zip(q01, q99)):
        if hi < lo:
            raise ValueError(f"q99[{index}]={hi} is below q01[{index}]={lo}")
    return {"q01": q01, "q99": q99}


def load_rot6d20_quantile_stats(path: str | Path) -> dict[str, list[float]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"rot6d20 stats file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("action_dim", ROT6D20_ACTION_DIM) != ROT6D20_ACTION_DIM:
        raise ValueError(
            f"Expected action_dim={ROT6D20_ACTION_DIM} in {path}, "
            f"got {raw.get('action_dim')}"
        )
    return validate_quantile_stats(*_extract_quantiles(raw))


def unresolved_rot6d20_quantile_stats() -> dict[str, list[float]]:
    """Import-safe placeholder; runtime entry points reject this value."""

    return validate_quantile_stats(
        [0.0] * ROT6D20_ACTION_DIM,
        [1.0] * ROT6D20_ACTION_DIM,
    )
