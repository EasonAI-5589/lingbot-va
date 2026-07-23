#!/usr/bin/env python3
"""Precompute LingBot Wan latents from canonical Action-Following rot6d20."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from precompute_partition import partition_sample_indices

TEXT_EMB_DIRNAME = "text_emb"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--afd-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--protocol", choices=("clean", "mix4"), default="clean")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse complete samples recorded by earlier rank manifests in the "
            "same output root, then rebuild manifests for the current world size."
        ),
    )
    return parser.parse_args()


def _record_file(output_root: Path, record: dict) -> Path:
    relative = Path(record["file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe resume record path: {relative}")
    return output_root / relative


def collect_resume_records(output_root: Path, num_samples: int) -> dict[int, dict]:
    """Collect complete legacy/current rank records before manifests are rebuilt.

    A record is only reusable after the corresponding sample and content-addressed
    text embedding have both been atomically published.  Rank manifests are
    written after ``torch.save`` completes, so a killed writer cannot make a
    partially-written sample look reusable merely because the filename exists.
    """
    records: dict[int, dict] = {}
    sample_root = output_root / "samples"
    text_emb_root = output_root / TEXT_EMB_DIRNAME
    valid_sample_names = {
        entry.name
        for entry in os.scandir(sample_root)
        if entry.is_file() and entry.stat().st_size > 0
    }
    valid_emb_names = {
        entry.name
        for entry in os.scandir(text_emb_root)
        if entry.is_file() and entry.stat().st_size > 0
    }
    for manifest_path in sorted(output_root.glob("manifest.rank*.jsonl")):
        with manifest_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    index = int(record["index"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Invalid resume record {manifest_path}:{line_number}"
                    ) from exc
                if not 0 <= index < num_samples:
                    raise ValueError(f"Resume index outside [0,{num_samples}): {index}")
                if record.get("action_shape") != [32, 20]:
                    raise ValueError(
                        f"Resume action contract mismatch at index {index}: "
                        f"{record.get('action_shape')}"
                    )
                if record.get("latent_shape") != [48, 9, 24, 20]:
                    raise ValueError(
                        f"Resume latent contract mismatch at index {index}: "
                        f"{record.get('latent_shape')}"
                    )
                sample_path = _record_file(output_root, record)
                if sample_path.parent != sample_root:
                    raise ValueError(
                        f"Resume sample is outside samples/: {sample_path}"
                    )
                if sample_path.name not in valid_sample_names:
                    continue
                if f"{record['text_emb_key']}.pt" not in valid_emb_names:
                    continue
                previous = records.setdefault(index, record)
                if previous != record:
                    raise ValueError(f"Conflicting resume records for index {index}")
    return records


def publish_json(path: Path, payload: dict) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp_path, path)


def publish_sample(path: Path, payload: dict, rank: int) -> None:
    tmp_path = path.with_name(f".{path.name}.rank{rank:02d}.{os.getpid()}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


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
    mean = torch.tensor(vae.config.latents_mean, device=mu.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=mu.device).view(1, -1, 1, 1, 1)
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
        dist.init_process_group("nccl", timeout=timedelta(hours=6))
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

    contract_path = args.output_root / "PRECOMPUTE_CONTRACT.json"
    contract = {
        "protocol": args.protocol,
        "num_samples": args.num_samples,
        "action_shape": [32, 20],
        "latent_shape": [48, 9, 24, 20],
        "afd_root": str(args.afd_root),
        "model_path": str(args.model_path),
    }
    if rank == 0:
        if contract_path.exists():
            existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
            if existing_contract != contract:
                raise ValueError(
                    "Precompute resume contract mismatch: "
                    f"existing={existing_contract} requested={contract}"
                )
        else:
            publish_json(contract_path, contract)
    if world_size > 1:
        dist.barrier()

    resume_snapshot = args.output_root / "resume_manifest.jsonl"
    if args.resume:
        if rank == 0:
            resume_records = collect_resume_records(args.output_root, args.num_samples)
            tmp_snapshot = resume_snapshot.with_name(
                f".{resume_snapshot.name}.{os.getpid()}.tmp"
            )
            with tmp_snapshot.open("w", encoding="utf-8") as handle:
                for index in sorted(resume_records):
                    handle.write(
                        json.dumps(resume_records[index], ensure_ascii=False) + "\n"
                    )
            os.replace(tmp_snapshot, resume_snapshot)
            print(
                "[PRECOMPUTE_RESUME_DISCOVERY] "
                + json.dumps(
                    {
                        "reusable_samples": len(resume_records),
                        "requested_samples": args.num_samples,
                        "remaining_samples": args.num_samples - len(resume_records),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if world_size > 1:
            dist.barrier()
        resume_records = {
            int(record["index"]): record
            for record in (
                json.loads(line)
                for line in resume_snapshot.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
    else:
        stale_manifests = list(args.output_root.glob("manifest.rank*.jsonl"))
        stale_samples = next(sample_root.glob("sample_*.pt"), None)
        if stale_manifests or stale_samples is not None:
            raise ValueError(
                f"Output root is not empty; pass --resume to reuse it: {args.output_root}"
            )
        resume_records = {}
    # Every process must finish reading the snapshot before ranks 0..7 truncate
    # legacy 8-rank manifests and rebuild them for the current (possibly 16-rank)
    # topology.
    if world_size > 1:
        dist.barrier()

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
        raise ValueError(f"Protocol sampling audit exceeded 1e-4: {source_audit}")
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
    reused_count = 0
    computed_count = 0
    rank_sample_indices = partition_sample_indices(
        args.num_samples,
        resume_records,
        rank,
        world_size,
    )
    with rank_manifest.open("w", encoding="utf-8") as manifest_handle:
        for sample_index in rank_sample_indices:
            output_path = sample_root / f"sample_{sample_index:06d}.pt"
            if sample_index in resume_records:
                record = resume_records[sample_index]
                manifest_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest_handle.flush()
                reused_count += 1
                continue
            mode, dataset_index, _, sample = dataset._fetch_sample(sample_index)
            task = dataset._task_by_source[dataset_index]
            family = dataset._family_by_source[dataset_index]
            actions = sample["action"].float()
            if tuple(actions.shape) != (32, 20):
                raise ValueError(
                    f"Expected canonical action [32,20], got {tuple(actions.shape)}"
                )

            with torch.inference_mode():
                high = encode_camera(vae, sample[CAMERA_FEATURES[0]], (256, 320), dtype)
                left = encode_camera(vae, sample[CAMERA_FEATURES[1]], (128, 160), dtype)
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
                    store_text_emb(text_emb_root, emb_key, prompt, text_cache[prompt])

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
            publish_sample(output_path, payload, rank)
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
            computed_count += 1
            print(json.dumps({"rank": rank, **record}), flush=True)

    print(
        "[PRECOMPUTE_RANK_RESULT] "
        + json.dumps(
            {
                "rank": rank,
                "world_size": world_size,
                "reused": reused_count,
                "computed": computed_count,
            },
            sort_keys=True,
        ),
        flush=True,
    )

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
            "target_family_probabilities": source_audit["target_family_probabilities"],
            "max_abs_error": max_abs_error,
        }
        (args.output_root / "DATA_AUDIT.json").write_text(
            json.dumps(manifest_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "[MANIFEST_AUDIT] " + json.dumps(manifest_audit, sort_keys=True), flush=True
        )
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
