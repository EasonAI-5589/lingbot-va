#!/usr/bin/env python3
"""Build a reload-clean LingBot native-20D inference bundle.

Training checkpoints contain the fine-tuned transformer only.  This command
creates a lightweight model root that links the immutable base tokenizer,
text encoder, and VAE, links the checkpoint weights, and writes a corrected
inference-only transformer config.  The source checkpoint is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from safetensors import safe_open


ACTION_INPUT_KEY = "action_embedder.weight"
ACTION_OUTPUT_KEY = "action_proj_out.weight"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = source.resolve(strict=True)
    destination_parent = destination.parent.resolve(strict=True)
    if destination.is_symlink() or destination.exists():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"Refusing to replace existing bundle path: {destination}")
    # AIHC exposes the same PFS through stable aliases such as /mnt/public_ckp
    # and /mnt/gyc_ckp.  Computing relpath from those aliases creates a link
    # that becomes invalid once the destination is followed through its mount
    # alias.  Resolve both endpoints first so the link is valid from the real
    # mounted directories as well as through the aliases.
    destination.symlink_to(
        os.path.relpath(source, destination_parent),
        target_is_directory=source.is_dir(),
    )


def inspect_action_shapes(weights_path: Path) -> dict[str, list[int]]:
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        missing = sorted({ACTION_INPUT_KEY, ACTION_OUTPUT_KEY} - keys)
        if missing:
            raise KeyError(f"Missing action boundary tensors: {missing}")
        shapes = {
            ACTION_INPUT_KEY: list(handle.get_slice(ACTION_INPUT_KEY).get_shape()),
            ACTION_OUTPUT_KEY: list(handle.get_slice(ACTION_OUTPUT_KEY).get_shape()),
        }
    if shapes[ACTION_INPUT_KEY][1] != 20 or shapes[ACTION_OUTPUT_KEY][0] != 20:
        raise ValueError(f"Checkpoint is not native Rot6D20: {shapes}")
    return shapes


def prepare_bundle(checkpoint: Path, base_model: Path, output: Path, stat_path: Path) -> dict:
    transformer = checkpoint / "transformer"
    source_config = transformer / "config.json"
    weights = transformer / "diffusion_pytorch_model.safetensors"
    for required in (source_config, weights, stat_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    for required_dir in (base_model / "vae", base_model / "tokenizer", base_model / "text_encoder"):
        if not required_dir.is_dir():
            raise FileNotFoundError(required_dir)

    shapes = inspect_action_shapes(weights)
    output.mkdir(parents=True, exist_ok=True)
    _relative_symlink(base_model / "vae", output / "vae")
    _relative_symlink(base_model / "tokenizer", output / "tokenizer")
    _relative_symlink(base_model / "text_encoder", output / "text_encoder")
    _relative_symlink(weights, output / "transformer" / weights.name)

    config = json.loads(source_config.read_text(encoding="utf-8"))
    source_action_dim = config.get("action_dim")
    source_attn_mode = config.get("attn_mode")
    config["action_dim"] = 20
    config["attn_mode"] = "torch"
    config.pop("_name_or_path", None)
    output_config = output / "transformer" / "config.json"
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(json.dumps(config, indent=2), encoding="utf-8")

    audit = {
        "status": "prepared",
        "source_checkpoint": str(checkpoint.resolve()),
        "base_model": str(base_model.resolve()),
        "output_bundle": str(output.resolve()),
        "normalization_stat": str(stat_path.resolve()),
        "source_config_action_dim": source_action_dim,
        "source_config_attn_mode": source_attn_mode,
        "inference_config_action_dim": 20,
        "inference_config_attn_mode": "torch",
        "tensor_shapes": shapes,
        "source_checkpoint_modified": False,
        "sha256": {
            "source_config": _sha256(source_config),
            "inference_config": _sha256(output_config),
            "normalization_stat": _sha256(stat_path),
        },
    }
    (output / "INFERENCE_BUNDLE_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stat-path", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(prepare_bundle(args.checkpoint, args.base_model, args.output, args.stat_path), indent=2))
