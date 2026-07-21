#!/usr/bin/env python3
"""Precompute LingBot Wan latents from canonical Action-Following rot6d20."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

TEXT_EMB_DIRNAME = "text_emb"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--afd-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--protocol", choices=("clean", "mix4"), default="clean")
    return parser.parse_args()


def text_emb_key(prompt: str) -> str:
    """Content-addressed key for a prompt's T5 embedding.

    Keying on the prompt itself lets every distributed rank derive the same
    filename without coordinating, so identical prompts collapse to one file.
    """
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]


def store_text_emb(text_emb_root: Path, key: str, prompt: str, emb) -> None:
    """Persist one shared T5 embedding, atomically and only once.

    AFD gives every task a single canonical instruction, so a full-scale run has
    on the order of 50 unique prompts against millions of samples. Writing the
    4 MB embedding into each sample would multiply the dataset by ~12x and put a
    50-task clean run in the petabyte range; storing it once keeps it in the
    hundreds of gigabytes.

    Ranks race on the same key, so write to a rank-private temp file and rename.
    Rename is atomic within a directory, so a reader never observes a partial
    file, and the loser of the race simply overwrites with identical bytes.
    """
    final_path = text_emb_root / f"{key}.pt"
    if final_path.exists():
        return
    tmp_path = text_emb_root / f".{key}.{os.getpid()}.tmp"
    torch.save({"prompt": prompt, "text_emb": emb}, tmp_path)
    os.replace(tmp_path, final_path)


def encode_text(tokenizer, text_encoder, prompt, device, dtype):
    inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    mask = inputs.attention_mask
    seq_len = int(mask.sum())
    embeds = text_encoder(
        inputs.input_ids.to(device), mask.to(device)
    ).last_hidden_state.to(dtype=dtype)
    output = embeds.new_zeros((512, embeds.shape[-1]))
    output[:seq_len] = embeds[0, :seq_len]
    return output.cpu()


def encode_camera(vae, video, size, dtype):
    video = F.interpolate(
        video.float(), size=size, mode="bilinear", align_corners=False
    )
    video = video.permute(1, 0, 2, 3).unsqueeze(0)
    video = video.to(next(vae.parameters()).device, dtype=dtype) * 2.0 - 1.0
    # AutoencoderKLWan's native offline encoder applies the exact causal
    # schedule required by Wan: frame 0 first, then groups of four while
    # carrying feature caches. Passing all 32 frames to the streaming wrapper
    # in one call breaks temporal residual shapes (32 vs 16).
    encoded = vae._encode(video)
    mu, _ = torch.chunk(encoded, 2, dim=1)
    mean = torch.tensor(vae.config.latents_mean, device=mu.device).view(
        1, -1, 1, 1, 1
    )
    std = torch.tensor(vae.config.latents_std, device=mu.device).view(
        1, -1, 1, 1, 1
    )
    return ((mu.float() - mean) / std).to(dtype=dtype)


def sample_prompt(sample, task):
    value = sample.get("task")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
        return value[0].strip()
    return task.replace("_", " ")


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16

    from cosmos_framework.data.generator.action.datasets.actionfollowing_lerobot_dataset import (
        ActionFollowingLeRobotDataset,
        CAMERA_FEATURES,
    )
    from wan_va.modules.utils import (
        load_text_encoder,
        load_tokenizer,
        load_vae,
    )

    if not args.afd_root.is_dir():
        raise FileNotFoundError(f"AFD root missing: {args.afd_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    sample_root = args.output_root / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    text_emb_root = args.output_root / TEXT_EMB_DIRNAME
    text_emb_root.mkdir(parents=True, exist_ok=True)

    dataset = ActionFollowingLeRobotDataset(
        root=str(args.afd_root),
        protocol=args.protocol,
        fps=30.0,
        chunk_length=32,
        audit_num_samples=1000,
        audit_max_abs_error=0.05,
        max_loaded_datasets=8,
    )
    raw_virtual_length = len(dataset)
    if args.num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {args.num_samples}")
    if args.protocol == "clean" and args.num_samples > raw_virtual_length:
        raise ValueError(
            "Clean precompute cannot exceed the genuine current1+future32 "
            f"window count: requested={args.num_samples}, available={raw_virtual_length}"
        )
    # For mix4 the Cosmos loader's length is the sum of all raw family windows,
    # while LingBot intentionally materializes a smaller weighted training set.
    # Make that materialized length the deterministic virtual sampling length so
    # audit_sampling(num_samples=N) describes the exact indices [0, N) that are
    # written below, rather than a prefix of a larger permutation.
    if args.protocol == "mix4":
        dataset._num_valid_indices = args.num_samples
    source_audit = dataset.audit_sampling(num_samples=args.num_samples)
    if rank == 0:
        print(
            "[DATA_AUDIT] "
            + json.dumps(
                {
                    "protocol": args.protocol,
                    "raw_virtual_length": raw_virtual_length,
                    "materialized_samples": args.num_samples,
                    "effective_counts": dataset.family_effective_counts,
                    **source_audit,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if source_audit["max_abs_error"] > 1e-4:
        raise ValueError(
            "Protocol sampling audit exceeded 1e-4: "
            f"{source_audit}"
        )
    vae = load_vae(
        str(args.model_path / "vae"), torch_dtype=dtype, torch_device=device
    ).eval()
    tokenizer = load_tokenizer(str(args.model_path / "tokenizer"))
    text_encoder = load_text_encoder(
        str(args.model_path / "text_encoder"),
        torch_dtype=dtype,
        torch_device=device,
    ).eval()
    text_cache = {}

    rank_manifest = args.output_root / f"manifest.rank{rank:02d}.jsonl"
    with rank_manifest.open("w", encoding="utf-8") as manifest_handle:
        for sample_index in range(rank, args.num_samples, world_size):
            output_path = sample_root / f"sample_{sample_index:06d}.pt"
            mode, dataset_index, _, sample = dataset._fetch_sample(sample_index)
            task = dataset._task_by_source[dataset_index]
            family = dataset._family_by_source[dataset_index]
            actions = sample["action"].float()
            if tuple(actions.shape) != (32, 20):
                raise ValueError(
                    f"Expected canonical action [32,20], got {tuple(actions.shape)}"
                )

            with torch.inference_mode():
                high = encode_camera(
                    vae, sample[CAMERA_FEATURES[0]], (256, 320), dtype
                )
                left = encode_camera(
                    vae, sample[CAMERA_FEATURES[1]], (128, 160), dtype
                )
                right = encode_camera(
                    vae, sample[CAMERA_FEATURES[2]], (128, 160), dtype
                )
                if high.shape[2] != left.shape[2] or high.shape[2] != right.shape[2]:
                    raise ValueError("Camera VAE temporal shapes disagree")
                wrists = torch.cat([left, right], dim=-1)
                latents = torch.cat([wrists, high], dim=-2)[0].cpu()

                prompt = sample_prompt(sample, task)
                emb_key = text_emb_key(prompt)
                if prompt not in text_cache:
                    text_cache[prompt] = encode_text(
                        tokenizer, text_encoder, prompt, device, dtype
                    )
                    store_text_emb(
                        text_emb_root, emb_key, prompt, text_cache[prompt]
                    )

            # text_emb lives once in text_emb/<key>.pt; the sample only refers
            # to it. The loader resolves and caches it.
            payload = {
                "latents": latents,
                "text_emb_key": emb_key,
                "actions": actions.cpu(),
                "task": task,
                "family": family,
                "source_index": sample_index,
                "mode": mode,
                "prompt": prompt,
            }
            torch.save(payload, output_path)
            record = {
                "index": sample_index,
                "file": str(output_path.relative_to(args.output_root)),
                "task": task,
                "family": family,
                "text_emb_key": emb_key,
                "latent_shape": list(latents.shape),
                "action_shape": list(actions.shape),
            }
            manifest_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest_handle.flush()
            print(json.dumps({"rank": rank, **record}), flush=True)

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        records = []
        for worker_rank in range(world_size):
            path = args.output_root / f"manifest.rank{worker_rank:02d}.jsonl"
            records.extend(json.loads(line) for line in path.open() if line.strip())
        records.sort(key=lambda record: record["index"])
        if len(records) != args.num_samples:
            raise ValueError(
                f"Expected {args.num_samples} precomputed samples, got {len(records)}"
            )
        manifest = args.output_root / "manifest.jsonl"
        with manifest.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        family_counts = Counter(record["family"] for record in records)
        task_count = len({record["task"] for record in records})
        observed = {
            family: family_counts[family] / float(len(records))
            for family in source_audit["target_family_probabilities"]
        }
        max_abs_error = max(
            abs(observed[family] - target)
            for family, target in source_audit["target_family_probabilities"].items()
        )
        if task_count != 50:
            raise ValueError(f"Expected 50 tasks, got {task_count}")
        if max_abs_error > 1e-4:
            raise ValueError(
                "Materialized manifest sampling audit exceeded 1e-4: "
                f"observed={observed}"
            )
        manifest_audit = {
            "status": "PASS",
            "protocol": args.protocol,
            "samples": len(records),
            "tasks": task_count,
            "action_shape": [32, 20],
            "family_counts": dict(sorted(family_counts.items())),
            "observed_family_probabilities": observed,
            "target_family_probabilities": source_audit[
                "target_family_probabilities"
            ],
            "max_abs_error": max_abs_error,
        }
        (args.output_root / "DATA_AUDIT.json").write_text(
            json.dumps(manifest_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("[MANIFEST_AUDIT] " + json.dumps(manifest_audit, sort_keys=True), flush=True)
        print(
            json.dumps(
                {
                    "status": "PRECOMPUTE_COMPLETE",
                    "samples": len(records),
                    "manifest": str(manifest),
                }
            ),
            flush=True,
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
