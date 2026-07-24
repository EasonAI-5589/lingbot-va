#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${LINGBOT_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
CHECKPOINT="${LINGBOT_CHECKPOINT:?Set LINGBOT_CHECKPOINT to checkpoint_step_*}"
BASE_MODEL="${LINGBOT_BASE_MODEL:-/mnt/dataset/public_data/lingbot-va-base}"
STAT_PATH="${LINGBOT_ROT6D20_STAT_PATH:?Set LINGBOT_ROT6D20_STAT_PATH to the exact training q01/q99 JSON}"
INPUT_DIR="${LINGBOT_INPUT_IMAGE_DIR:?Set LINGBOT_INPUT_IMAGE_DIR to three current camera PNGs}"
PROMPT_FILE="${LINGBOT_PROMPT_FILE:?Set LINGBOT_PROMPT_FILE to the RoboTwin full_description text file}"
HANDOFF_ROOT="${LINGBOT_HANDOFF_ROOT:-${CHECKPOINT}/handoff_inference/place_burger_fries_clean0_video32_steps25}"
BUNDLE="${LINGBOT_INFERENCE_BUNDLE:-${HANDOFF_ROOT}/model_bundle}"
OUTPUT="${LINGBOT_INFERENCE_OUTPUT:-${HANDOFF_ROOT}/output}"
LOG_PATH="${LINGBOT_INFERENCE_LOG:-${HANDOFF_ROOT}/minimal_inference.log}"

test -f "${PROMPT_FILE}"
for camera in \
  observation.images.cam_high \
  observation.images.cam_left_wrist \
  observation.images.cam_right_wrist; do
  test -f "${INPUT_DIR}/${camera}.png"
done

mkdir -p "${HANDOFF_ROOT}" "${OUTPUT}"
"${PYTHON}" "${REPO_ROOT}/script/prepare_rot6d20_inference_bundle.py" \
  --checkpoint "${CHECKPOINT}" \
  --base-model "${BASE_MODEL}" \
  --output "${BUNDLE}" \
  --stat-path "${STAT_PATH}"

export PYTHONUNBUFFERED=1
export LINGBOT_WAN22_PATH="${BUNDLE}"
export LINGBOT_ROT6D20_STAT_PATH="${STAT_PATH}"
export LINGBOT_INPUT_IMAGE_DIR="${INPUT_DIR}"
export LINGBOT_PROMPT="$(<"${PROMPT_FILE}")"
export LINGBOT_NUM_CHUNKS_TO_INFER="${LINGBOT_NUM_CHUNKS_TO_INFER:-4}"
export LINGBOT_VIDEO_ONLY=1
export LINGBOT_FUTURE_RGB_FRAMES="${LINGBOT_FUTURE_RGB_FRAMES:-32}"
export LINGBOT_VAE_TEMPORAL_STRIDE="${LINGBOT_VAE_TEMPORAL_STRIDE:-4}"
export LINGBOT_INFERENCE_OUTPUT="${OUTPUT}"
export LINGBOT_ACTION_PER_FRAME="${LINGBOT_ACTION_PER_FRAME:-4}"
export NGPU=1

cd "${REPO_ROOT}"
"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port "${MASTER_PORT:-29621}" \
  -m wan_va.wan_va_server \
  --config-name robotwin_rot6d20_i2va \
  --save_root "${OUTPUT}" \
  2>&1 | tee "${LOG_PATH}"

test -s "${OUTPUT}/demo.mp4"
test -s "${OUTPUT}/demo_with_condition.mp4"
test -s "${OUTPUT}/inference_metadata.json"
test ! -e "${OUTPUT}/pred_actions_physical_rot6d20.npy"

FUTURE_FRAMES="$(ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_read_frames -of csv=p=0 "${OUTPUT}/demo.mp4")"
WITH_CONDITION_FRAMES="$(ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_read_frames -of csv=p=0 "${OUTPUT}/demo_with_condition.mp4")"
test "${FUTURE_FRAMES}" = "32"
test "${WITH_CONDITION_FRAMES}" = "33"

"${PYTHON}" - "${OUTPUT}/inference_metadata.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    metadata = json.load(handle)
assert metadata["mode"] == "video_only_current1_future32", metadata
assert metadata["action_output_persisted"] is False, metadata
assert metadata["temporal_contract"]["total_video_latent_frames"] == 9, metadata
assert metadata["outputs"]["demo.mp4"] == 32, metadata
assert metadata["outputs"]["demo_with_condition.mp4"] == 33, metadata
PY

cat > "${HANDOFF_ROOT}/INFERENCE_RESULT.txt" <<EOF
status=passed
model=lingbot-va
checkpoint=${CHECKPOINT}
bundle=${BUNDLE}
normalization_stat=${STAT_PATH}
input=${INPUT_DIR}
output=${OUTPUT}
mode=video_only_current1_future32
future_rgb_frames=32
rgb_frames_with_condition=33
video_latent_frames=9
action_output_persisted=false
camera_layout=left_wrist_top_left_right_wrist_top_right_head_bottom
EOF

echo "[PASS] LingBot video-only current1+future32 inference: ${OUTPUT}"
