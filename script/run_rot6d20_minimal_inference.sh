#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/gyc/LingbotVA2.0/lingbot-va}"
PYTHON="${LINGBOT_PYTHON:-/mnt/gyc/miniconda3/envs/lingbot-va/bin/python}"
CHECKPOINT="${LINGBOT_CHECKPOINT:?Set LINGBOT_CHECKPOINT to checkpoint_step_*}"
BASE_MODEL="${LINGBOT_BASE_MODEL:-/mnt/public_ckp/lingbot-va-base}"
STAT_PATH="${LINGBOT_ROT6D20_STAT_PATH:?Set LINGBOT_ROT6D20_STAT_PATH to the exact training q01/q99 JSON}"
INPUT_DIR="${LINGBOT_INPUT_IMAGE_DIR:?Set LINGBOT_INPUT_IMAGE_DIR to three current camera PNGs}"
PROMPT_FILE="${LINGBOT_PROMPT_FILE:?Set LINGBOT_PROMPT_FILE to the RoboTwin full_description text file}"
HANDOFF_ROOT="${LINGBOT_HANDOFF_ROOT:-${CHECKPOINT}/handoff_inference/place_burger_fries_clean0_steps25_50}"
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
test -s "${OUTPUT}/pred_actions_physical_rot6d20.npy"
test -s "${OUTPUT}/inference_metadata.json"

"${PYTHON}" - "${OUTPUT}/pred_actions_physical_rot6d20.npy" <<'PY'
import sys
import numpy as np

actions = np.load(sys.argv[1])
assert actions.shape == (32, 20), actions.shape
assert np.isfinite(actions).all()
PY

cat > "${HANDOFF_ROOT}/INFERENCE_RESULT.txt" <<EOF
status=passed
model=lingbot-va
checkpoint=${CHECKPOINT}
bundle=${BUNDLE}
normalization_stat=${STAT_PATH}
input=${INPUT_DIR}
output=${OUTPUT}
action_space=physical_robot_base_frame_deltaee_rot6d20
camera_layout=left_wrist_top_left_right_wrist_top_right_head_bottom
EOF

echo "[PASS] LingBot Rot6D20 minimal inference: ${OUTPUT}"
