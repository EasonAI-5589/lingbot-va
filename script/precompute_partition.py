"""Deterministic work partitioning for resumable offline precompute."""

from __future__ import annotations


def partition_sample_indices(
    num_samples: int,
    reusable_indices,
    rank: int,
    world_size: int,
) -> list[int]:
    """Balance expensive missing samples independently from cheap reused ones.

    A static ``range(rank, num_samples, world_size)`` partition balances the
    total number of records, but not the amount of VAE work after an interrupted
    run. Earlier ranks can have hundreds more missing samples than later ranks,
    leaving the fast ranks waiting in the final collective long enough to hit
    the default NCCL timeout. Sharding the reusable and missing sets separately
    keeps both counts within one item across ranks.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} outside [0,{world_size})")

    reusable = sorted({int(index) for index in reusable_indices})
    if reusable and (reusable[0] < 0 or reusable[-1] >= num_samples):
        raise ValueError("reusable index outside requested sample range")
    reusable_set = set(reusable)
    missing = [index for index in range(num_samples) if index not in reusable_set]
    return reusable[rank::world_size] + missing[rank::world_size]
