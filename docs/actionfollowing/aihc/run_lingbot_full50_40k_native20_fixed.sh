#!/usr/bin/env bash
# LingBot-VA ActionFollowing full-50 40K bootstrap.
#
# The AIHC job supplies LINGBOT_PROTOCOL=clean or mix4 and an exact git commit.
# All other data/model/training parameters are frozen here so a gitignored local
# environment file cannot silently turn a 40K job into clean128/50K again.
set -Eeuo pipefail

die() { echo "[FATAL] $*" >&2; exit 1; }
ensure_mount_alias() {
  local alias_path="$1" target_path="$2"
  [[ -d "$target_path" ]] || die "missing AIHC mount: $target_path"
  if [[ ! -e "$alias_path" ]]; then ln -s "$target_path" "$alias_path"; fi
  [[ "$(readlink -f "$alias_path")" == "$(readlink -f "$target_path")" ]] || \
    die "$alias_path does not resolve to $target_path"
}

ensure_mount_alias /mnt/gyc        /mnt/dataset/csx_workspace
ensure_mount_alias /mnt/gyc_ckp    /mnt/dataset/csx_ckp
ensure_mount_alias /mnt/public_ckp /mnt/dataset/public_data

REPO=/mnt/gyc/LingbotVA2.0/lingbot-va-actionfollowing-fixed
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "fixed worktree missing: $REPO"
require_commit="${LINGBOT_EXPECTED_COMMIT:?LINGBOT_EXPECTED_COMMIT is required}"
actual_commit="$(git -C "$REPO" rev-parse HEAD)"
[[ "$actual_commit" == "$require_commit" ]] || \
  die "repo commit mismatch: expected=$require_commit actual=$actual_commit"

protocol="${LINGBOT_PROTOCOL:?LINGBOT_PROTOCOL=clean or mix4 is required}"
case "$protocol" in
  clean) precompute_samples=472622 ;;
  mix4) precompute_samples=600000 ;;
  *) die "unsupported protocol: $protocol" ;;
esac

run_tag="${AIHC_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
node_rank="${RANK:-0}"
node_world_size="${WORLD_SIZE:-1}"
out_base="/mnt/gyc_ckp/Action-Following/outputs/lingbot/full50/${protocol}/train_40000_20260721_native20_fixed_${run_tag}"
bootstrap_marker="$out_base/.bootstrap_job_id"
if (( node_rank == 0 )); then
  [[ ! -e "$out_base" ]] || die "refusing to reuse output root: $out_base"
  mkdir -p "$out_base"
  printf '%s\n' "$run_tag" > "$bootstrap_marker"
else
  for _ in $(seq 1 120); do
    [[ -f "$bootstrap_marker" ]] && break
    sleep 1
  done
  [[ -f "$bootstrap_marker" ]] || die "master did not initialize output: $out_base"
  [[ "$(cat "$bootstrap_marker")" == "$run_tag" ]] || die "output marker mismatch"
fi
if (( node_rank == 0 )); then
  bootstrap_log="$out_base/bootstrap.log"
else
  bootstrap_log="$out_base/bootstrap.worker${node_rank}.log"
fi
exec > >(tee -a "$bootstrap_log") 2>&1
trap 'rc=$?; printf "[BOOTSTRAP_ERROR] rc=%s line=%s command=%q\n" "$rc" "${BASH_LINENO[0]}" "$BASH_COMMAND"; exit "$rc"' ERR

export LINGBOT_ENV_FILE=/dev/null
export LINGBOT_PYTHON=/mnt/gyc/miniconda3/envs/lingbot-va/bin/python
export LINGBOT_PRECOMPUTE_PYTHON=/mnt/gyc/cosmos-framework/.venv/bin/python
export LINGBOT_PRECOMPUTE_PYTHONPATH=/mnt/gyc/cosmos-framework
export LINGBOT_WAN22_PATH=/mnt/public_ckp/lingbot-va-base
export AFD_ROOT=/mnt/public_ckp/cscsx_projects/data/ActionFollowingData_LeRobot_Rot6D_nosymlink/train
export LINGBOT_ROT6D20_LATENT_ROOT=/mnt/public_ckp/robotwin-clean-and-aug-lerobot
export LINGBOT_ROT6D20_ACTION_ROOT=/mnt/public_ckp/cscsx_projects/ctrl_world_train/latents/action_following_current1_future32_clean_enhanced_explore_test4v2_fulldesc_rot6d20_frameidxfix_20260711
export LINGBOT_ROT6D20_STAT_PATH=/mnt/public_ckp/cscsx_projects/ctrl_world_train/dataset_meta_info/action_following_current1_future32_clean_enhanced_explore_test4v2_fulldesc_rot6d20_frameidxfix_20260711/stat.json
if [[ -n "${LINGBOT_PRECOMPUTE_REUSE_ROOT:-}" ]]; then
  [[ "${LINGBOT_PRECOMPUTE_RESUME:-0}" == "1" ]] || \
    die "LINGBOT_PRECOMPUTE_REUSE_ROOT requires LINGBOT_PRECOMPUTE_RESUME=1"
  [[ -d "$LINGBOT_PRECOMPUTE_REUSE_ROOT/samples" ]] || \
    die "resume sample root missing: $LINGBOT_PRECOMPUTE_REUSE_ROOT/samples"
  export LINGBOT_PRECOMPUTE_ROOT="$LINGBOT_PRECOMPUTE_REUSE_ROOT"
else
  export LINGBOT_PRECOMPUTE_ROOT="$out_base/precompute"
fi
export LINGBOT_SAVE_ROOT="$out_base/train"
export LINGBOT_PROTOCOL="$protocol"
export LINGBOT_ROT6D20_FAMILY=clean
export LINGBOT_PRECOMPUTE_SAMPLES="$precompute_samples"
export LINGBOT_NUM_STEPS=40000
export LINGBOT_SAVE_INTERVAL=5000
export LINGBOT_ACTION_PER_FRAME=4
export LINGBOT_LOAD_WORKERS=8
export LINGBOT_ENABLE_WANDB=0
export NGPU=8
export LINGBOT_TRAIN_NGPU=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1

[[ -x "$LINGBOT_PYTHON" ]] || die "python missing: $LINGBOT_PYTHON"
[[ -x "$LINGBOT_PRECOMPUTE_PYTHON" ]] || die "precompute python missing: $LINGBOT_PRECOMPUTE_PYTHON"
[[ -d "$LINGBOT_WAN22_PATH/vae" ]] || die "Wan VAE missing"
[[ -f "$AFD_ROOT/demo_clean_zed2i_visible/turn_switch/meta/info.json" ]] || die "clean data missing"
[[ -f "$LINGBOT_ROT6D20_STAT_PATH" ]] || die "normalization stats missing"
[[ -f "$LINGBOT_ROT6D20_LATENT_ROOT/empty_emb.pt" ]] || die "empty_emb missing"

echo "[TRAIN_CONTRACT] protocol=$protocol samples=$precompute_samples tasks=50 action_shape=[32,20] action_dim=20 steps=40000 per_rank_batch=1 train_world_size=8 effective_batch=8 checkpoint_interval=5000"
echo "[PRECOMPUTE_CONTRACT] nodes=$node_world_size gpus_per_node=${NPROC_PER_NODE:-8} global_ranks=$(( node_world_size * ${NPROC_PER_NODE:-8} )) resume=${LINGBOT_PRECOMPUTE_RESUME:-0} root=$LINGBOT_PRECOMPUTE_ROOT"
echo "[PROVENANCE] commit=$actual_commit repo=$REPO output=$out_base node_rank=$node_rank"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
PYTHONPATH="$REPO:$LINGBOT_PRECOMPUTE_PYTHONPATH" \
  "$LINGBOT_PRECOMPUTE_PYTHON" - <<'PY'
import diffusers
import pandas
import torch
from cosmos_framework.data.generator.action.datasets.actionfollowing_lerobot_dataset import (
    ActionFollowingLeRobotDataset,
)
from wan_va.modules.utils import load_text_encoder, load_tokenizer, load_vae
print(
    "[PRECOMPUTE_ENV] "
    f"torch={torch.__version__} pandas={pandas.__version__} "
    f"diffusers={diffusers.__version__} dataset={ActionFollowingLeRobotDataset.__name__}"
)
PY

cd "$REPO"
bash script/run_rot6d20_native20_train_aihc.sh

if (( node_rank != 0 )); then
  echo "[PRECOMPUTE_WORKER_EXIT] node_rank=$node_rank"
  exit 0
fi

checkpoint="$LINGBOT_SAVE_ROOT/checkpoints/checkpoint_step_40000"
[[ -d "$checkpoint" ]] || die "final checkpoint missing: $checkpoint"
"$LINGBOT_PYTHON" script/validate_rot6d20_training_checkpoint.py \
  "$checkpoint" --output "$out_base/CHECKPOINT_AUDIT.json"

"$LINGBOT_PYTHON" - "$out_base" "$protocol" "$LINGBOT_PRECOMPUTE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
protocol = sys.argv[2]
precompute_root = Path(sys.argv[3])
data_audit = json.loads((precompute_root / "DATA_AUDIT.json").read_text())
checkpoint_audit = json.loads((root / "CHECKPOINT_AUDIT.json").read_text())
audit = {
    "status": "PASS",
    "model": "LingBot-VA-2.0",
    "protocol": protocol,
    "optimizer_steps": 40000,
    "effective_batch": 8,
    "checkpoint_interval": 5000,
    "precompute_root": str(precompute_root),
    "data": data_audit,
    "checkpoint": checkpoint_audit,
}
(root / "TRAIN_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
(root / "TRAIN_RESULT.txt").write_text(
    "status=passed\n"
    f"protocol={protocol}\n"
    "action_shape=[32,20]\n"
    "action_dim=20\n"
    "optimizer_steps=40000\n"
    "effective_batch=8\n"
    "checkpoint=checkpoint_step_40000\n"
)
print("[TRAIN_AUDIT] " + json.dumps(audit, sort_keys=True))
PY

echo "[DONE] output=$out_base"
