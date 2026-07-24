"""Temporal contracts shared by LingBotVA training and minimal inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Current1FutureVideoContract:
    """Wan temporal layout for one condition frame plus future RGB frames."""

    full_chunks: int
    latent_frames_per_full_chunk: int
    action_steps_per_latent: int
    vae_temporal_stride: int
    condition_rgb_frames: int
    future_rgb_frames: int
    full_chunk_latent_frames: int
    tail_video_latent_frames: int
    total_video_latent_frames: int
    decoded_rgb_frames_with_condition: int
    internal_aux_action_steps: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def build_current1_future_video_contract(
    *,
    full_chunks: int = 4,
    latent_frames_per_full_chunk: int = 2,
    action_steps_per_latent: int = 4,
    vae_temporal_stride: int = 4,
    future_rgb_frames: int = 32,
) -> Current1FutureVideoContract:
    """Build and validate the native LingBot current1+future32 timeline.

    Wan's causal VAE represents ``1 + stride * (F_latent - 1)`` RGB frames.
    Four regular two-latent chunks therefore contain the condition latent plus
    only seven future latents (29 RGB frames).  One video-only tail latent is
    required to recover the ninth latent used by the training dataloader.
    """
    values = {
        "full_chunks": full_chunks,
        "latent_frames_per_full_chunk": latent_frames_per_full_chunk,
        "action_steps_per_latent": action_steps_per_latent,
        "vae_temporal_stride": vae_temporal_stride,
        "future_rgb_frames": future_rgb_frames,
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"Temporal contract values must be positive: {values}")
    if future_rgb_frames % vae_temporal_stride:
        raise ValueError(
            "future_rgb_frames must be divisible by vae_temporal_stride: "
            f"{future_rgb_frames} % {vae_temporal_stride}"
        )

    full_chunk_latent_frames = full_chunks * latent_frames_per_full_chunk
    total_video_latent_frames = future_rgb_frames // vae_temporal_stride + 1
    tail_video_latent_frames = total_video_latent_frames - full_chunk_latent_frames
    if tail_video_latent_frames != 1:
        raise ValueError(
            "Expected exactly one video-only tail latent after the regular "
            f"chunks, got {tail_video_latent_frames}"
        )

    decoded_rgb_frames_with_condition = (
        1 + vae_temporal_stride * (total_video_latent_frames - 1)
    )
    return Current1FutureVideoContract(
        full_chunks=full_chunks,
        latent_frames_per_full_chunk=latent_frames_per_full_chunk,
        action_steps_per_latent=action_steps_per_latent,
        vae_temporal_stride=vae_temporal_stride,
        condition_rgb_frames=1,
        future_rgb_frames=future_rgb_frames,
        full_chunk_latent_frames=full_chunk_latent_frames,
        tail_video_latent_frames=tail_video_latent_frames,
        total_video_latent_frames=total_video_latent_frames,
        decoded_rgb_frames_with_condition=decoded_rgb_frames_with_condition,
        internal_aux_action_steps=(
            full_chunk_latent_frames * action_steps_per_latent
        ),
    )


def split_condition_and_future_frames(decoded_video, *, future_rgb_frames: int):
    """Validate a decoded current+future video and return future frames only."""
    expected_total = future_rgb_frames + 1
    actual_total = len(decoded_video)
    if actual_total != expected_total:
        raise ValueError(
            "Decoded video violates temporal parity: "
            f"actual={actual_total}, expected={expected_total} "
            "(one condition frame plus future frames)"
        )
    future_video = decoded_video[1:]
    if len(future_video) != future_rgb_frames:
        raise AssertionError(
            f"Future video has {len(future_video)} frames, expected {future_rgb_frames}"
        )
    return future_video
