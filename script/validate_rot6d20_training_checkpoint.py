#!/usr/bin/env python3
"""Validate that a saved LingBot checkpoint is self-consistent native 20D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def find_key(keys: list[str], suffix: str) -> str:
    matches = [key for key in keys if key.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one *{suffix} tensor, got {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    transformer = args.checkpoint / "transformer"
    config_path = transformer / "config.json"
    weights_path = transformer / "diffusion_pytorch_model.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("action_dim") != 20:
        raise ValueError(f"Expected config.action_dim=20, got {config.get('action_dim')}")

    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        embed_key = find_key(keys, "action_embedder.weight")
        proj_key = find_key(keys, "action_proj_out.weight")
        embed_shape = list(handle.get_slice(embed_key).get_shape())
        proj_shape = list(handle.get_slice(proj_key).get_shape())
    if embed_shape[-1] != 20:
        raise ValueError(f"Action embedder input must be 20, got {embed_shape}")
    if proj_shape[0] != 20:
        raise ValueError(f"Action projector output must be 20, got {proj_shape}")

    audit = {
        "status": "PASS",
        "checkpoint": str(args.checkpoint),
        "config_action_dim": config["action_dim"],
        "action_embedder_key": embed_key,
        "action_embedder_shape": embed_shape,
        "action_proj_out_key": proj_key,
        "action_proj_out_shape": proj_shape,
    }
    rendered = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
