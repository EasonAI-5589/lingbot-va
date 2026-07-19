#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LingBot-VA 2.0 post-training on Action-Following rot6d20 (20D delta_ee).
#
# All site-specific paths come from a local, gitignored config file. See
# script/lingbotva_env.example.sh for the template and the meaning of each var.
#
#     cp script/lingbotva_env.example.sh script/lingbotva_env.local.sh
#     $EDITOR script/lingbotva_env.local.sh
#     bash script/run_rot6d20_native20_train_aihc.sh
#
# Any variable can also be overridden inline:
#     LINGBOT_PROTOCOL=mix4 LINGBOT_PRECOMPUTE_SAMPLES=100000 bash script/...
# ---------------------------------------------------------------------------
set -euo pipefail

umask 007

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Load site configuration ----------------------------------------------
ENV_FILE="${LINGBOT_ENV_FILE:-${SCRIPT_DIR}/lingbotva_env.local.sh}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
else
  echo "WARNING: no site config at ${ENV_FILE}" >&2
  echo "         cp ${SCRIPT_DIR}/lingbotva_env.example.sh ${ENV_FILE} and edit it," >&2
  echo "         or export the variables yourself before running." >&2
fi

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: required variable ${name} is not set." >&2
    echo "       Define it in ${ENV_FILE} (template: script/lingbotva_env.example.sh)." >&2
    exit 1
  fi
}

require_path() {
  local name="$1"
  require_var "${name}"
  if [[ ! -e "${!name}" ]]; then
    echo "ERROR: ${name}=${!name} does not exist." >&2
    exit 1
  fi
}

require_path LINGBOT_PYTHON
require_path LINGBOT_PRECOMPUTE_PYTHON
require_path LINGBOT_WAN22_PATH
require_path AFD_ROOT
require_path LINGBOT_ROT6D20_LATENT_ROOT
require_path LINGBOT_ROT6D20_ACTION_ROOT
require_path LINGBOT_ROT6D20_STAT_PATH
require_var  LINGBOT_PRECOMPUTE_ROOT
require_var  LINGBOT_SAVE_ROOT

# --- Optional container mount bootstrap ------------------------------------
# Only symlinks when both the source mount and the link target are configured
# and the link does not already exist. Disabled when the vars are empty.
maybe_link() {
  local src="${1:-}" dst="${2:-}"
  if [[ -n "${src}" && -n "${dst}" && -e "${src}" && ! -e "${dst}" ]]; then
    ln -s "${src}" "${dst}"
    echo "[mount] linked ${dst} -> ${src}"
  fi
}
maybe_link "${LINGBOT_WORKSPACE_MOUNT:-}" "${LINGBOT_WORKSPACE_LINK:-}"
maybe_link "${LINGBOT_CKP_MOUNT:-}"       "${LINGBOT_CKP_LINK:-}"

# --- Derived defaults ------------------------------------------------------
PYTHON="${LINGBOT_PYTHON}"
PRECOMPUTE_PYTHON="${LINGBOT_PRECOMPUTE_PYTHON}"
PRECOMPUTE_ROOT="${LINGBOT_PRECOMPUTE_ROOT}"
SAVE_ROOT="${LINGBOT_SAVE_ROOT}"
PROTOCOL="${LINGBOT_PROTOCOL:-clean}"
PRECOMPUTE_SAMPLES="${LINGBOT_PRECOMPUTE_SAMPLES:-128}"

export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export AFD_ROOT
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export LINGBOT_WAN22_PATH
export LINGBOT_ROT6D20_MANIFEST_PATH="${LINGBOT_ROT6D20_MANIFEST_PATH:-${LINGBOT_ROT6D20_ACTION_ROOT}/manifests/train.jsonl}"
export LINGBOT_ROT6D20_ACTION_ROOT
export LINGBOT_ROT6D20_LATENT_ROOT
export LINGBOT_ROT6D20_STAT_PATH
export LINGBOT_ROT6D20_FAMILY="${LINGBOT_ROT6D20_FAMILY:-clean}"
export LINGBOT_DATASET_BACKEND=rot6d20_precomputed
export LINGBOT_PRECOMPUTED_MANIFEST="${PRECOMPUTE_ROOT}/manifest.jsonl"
export LINGBOT_ACTION_PER_FRAME="${LINGBOT_ACTION_PER_FRAME:-4}"
export LINGBOT_NUM_STEPS="${LINGBOT_NUM_STEPS:-50000}"
export LINGBOT_SAVE_INTERVAL="${LINGBOT_SAVE_INTERVAL:-1000}"
export LINGBOT_LOAD_WORKERS="${LINGBOT_LOAD_WORKERS:-8}"
export LINGBOT_ENABLE_WANDB="${LINGBOT_ENABLE_WANDB:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${SAVE_ROOT}"

echo "=== LingBot-VA rot6d20 run ==="
echo "  protocol          : ${PROTOCOL}"
echo "  family            : ${LINGBOT_ROT6D20_FAMILY}"
echo "  precompute samples: ${PRECOMPUTE_SAMPLES}"
echo "  train steps       : ${LINGBOT_NUM_STEPS}"
echo "  precompute root   : ${PRECOMPUTE_ROOT}"
echo "  save root         : ${SAVE_ROOT}"
echo "=============================="

cd "${REPO}"

# --- 1. Precompute Wan latents --------------------------------------------
PRECOMPUTE_PYTHONPATH="${REPO}"
if [[ -n "${LINGBOT_PRECOMPUTE_PYTHONPATH:-}" ]]; then
  PRECOMPUTE_PYTHONPATH="${PRECOMPUTE_PYTHONPATH}:${LINGBOT_PRECOMPUTE_PYTHONPATH}"
fi

PYTHONPATH="${PRECOMPUTE_PYTHONPATH}:${PYTHONPATH:-}" \
"${PRECOMPUTE_PYTHON}" -m torch.distributed.run \
  --nproc_per_node="${NGPU:-8}" \
  --master_port="${PRECOMPUTE_MASTER_PORT:-29617}" \
  script/precompute_actionfollowing_native20.py \
  --afd-root "${AFD_ROOT}" \
  --model-path "${LINGBOT_WAN22_PATH}" \
  --output-root "${PRECOMPUTE_ROOT}" \
  --num-samples "${PRECOMPUTE_SAMPLES}" \
  --protocol "${PROTOCOL}"

# --- 2. Preflight ----------------------------------------------------------
"${PYTHON}" script/preflight_native20_training_data.py

# --- 3. Train --------------------------------------------------------------
NGPU="${NGPU:-8}" \
CONFIG_NAME=robotwin_rot6d20_train \
MASTER_PORT="${MASTER_PORT:-29618}" \
bash script/run_va_posttrain.sh --save-root "${SAVE_ROOT}"
