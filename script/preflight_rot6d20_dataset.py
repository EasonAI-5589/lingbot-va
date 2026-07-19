#!/usr/bin/env python3
"""CPU-only preflight for canonical rot6d20 and LingBot-prepared roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wan_va.configs.action_schema import (
    ROT6D20_ACTION_DIM,
    ROT6D20_INVERSE_CHANNEL_IDS,
    ROT6D20_TO_LINGBOT30,
    load_rot6d20_quantile_stats,
)


CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def feature_width(info: dict, key: str) -> int | None:
    shape = info.get("features", {}).get(key, {}).get("shape", [])
    return int(shape[-1]) if shape else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--stat", type=Path, required=True)
    parser.add_argument(
        "--require-lingbot-latents",
        action="store_true",
        help="Also require v2.1 episodes.jsonl and at least one latent per root",
    )
    args = parser.parse_args()

    stats = load_rot6d20_quantile_stats(args.stat)
    if len(stats["q01"]) != 30 or len(stats["q99"]) != 30:
        raise ValueError("Expanded LingBot stats must be 30D")

    info_paths = sorted(args.dataset_root.rglob("meta/info.json"))
    if not info_paths:
        raise FileNotFoundError(f"No meta/info.json below {args.dataset_root}")

    raw = list(range(ROT6D20_ACTION_DIM)) + [0]
    expanded = [raw[index] for index in ROT6D20_INVERSE_CHANNEL_IDS]
    recovered = [expanded[index] for index in ROT6D20_TO_LINGBOT30]
    if recovered != list(range(ROT6D20_ACTION_DIM)):
        raise AssertionError("rot6d20 -> LingBot30 -> rot6d20 round-trip failed")

    failures = []
    latent_files = 0
    for info_path in info_paths:
        root = info_path.parent.parent
        with info_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
        if feature_width(info, "action") != ROT6D20_ACTION_DIM:
            failures.append(f"{root}: action width is not 20")
        missing = [key for key in CAMERAS if key not in info.get("features", {})]
        if missing:
            failures.append(f"{root}: missing cameras {missing}")
        if args.require_lingbot_latents:
            if not (root / "meta" / "episodes.jsonl").is_file():
                failures.append(f"{root}: missing meta/episodes.jsonl")
            current_latents = sum(1 for _ in (root / "latents").rglob("*.pth"))
            latent_files += current_latents
            if current_latents == 0:
                failures.append(f"{root}: no LingBot Wan latent files")

    if failures:
        preview = "\n".join(f"- {item}" for item in failures[:20])
        raise SystemExit(f"Preflight failed ({len(failures)} errors):\n{preview}")

    mode = "LingBot-prepared" if args.require_lingbot_latents else "canonical source"
    print(
        f"PASS mode={mode} datasets={len(info_paths)} action_dim=20 "
        f"model_action_dim=30 latent_files={latent_files}"
    )


if __name__ == "__main__":
    main()
