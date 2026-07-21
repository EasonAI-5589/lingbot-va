#!/usr/bin/env bash
# LingBot-VA checkpoint_step_50000 native-20D minimal-inference AIHC bootstrap.
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
expected_commit="${LINGBOT_EXPECTED_COMMIT:?LINGBOT_EXPECTED_COMMIT is required}"
actual_commit="$(git -C "$REPO" rev-parse HEAD)"
[[ "$actual_commit" == "$expected_commit" ]] || \
  die "repo commit mismatch: expected=$expected_commit actual=$actual_commit"

export REPO_ROOT="$REPO"
export LINGBOT_CHECKPOINT=/mnt/public_ckp/cscsx_projects/lingbotva_train/native20_rot6d20_clean4_20260718_v1/checkpoints/checkpoint_step_50000
export LINGBOT_BASE_MODEL=/mnt/public_ckp/lingbot-va-base
export LINGBOT_ROT6D20_STAT_PATH=/mnt/public_ckp/cscsx_projects/ctrl_world_train/dataset_meta_info/action_following_current1_future32_clean_enhanced_explore_test4v2_fulldesc_rot6d20_frameidxfix_20260711/stat.json
export LINGBOT_INPUT_IMAGE_DIR=/mnt/gyc_ckp/Action-Following/outputs/handoff_inputs/place_burger_fries_clean0_20260721
export LINGBOT_PROMPT_FILE="${LINGBOT_INPUT_IMAGE_DIR}/full_description.txt"
export LINGBOT_HANDOFF_ROOT=/mnt/gyc_ckp/Action-Following/outputs/lingbot/minimal_inference/checkpoint_step_50000_place_burger_fries_clean0_aihc_train_20260721

[[ ! -e "$LINGBOT_HANDOFF_ROOT" ]] || die "refusing to reuse handoff root: $LINGBOT_HANDOFF_ROOT"
mkdir -p "$LINGBOT_HANDOFF_ROOT"
exec > >(tee -a "$LINGBOT_HANDOFF_ROOT/bootstrap.log") 2>&1
trap 'rc=$?; printf "[BOOTSTRAP_ERROR] rc=%s line=%s command=%q\n" "$rc" "${BASH_LINENO[0]}" "$BASH_COMMAND"; exit "$rc"' ERR

echo "[INFERENCE_CONTRACT] model=lingbot checkpoint=checkpoint_step_50000 action_shape=[32,20] action_dim=20 cameras=3"
echo "[PROVENANCE] commit=$actual_commit repo=$REPO output=$LINGBOT_HANDOFF_ROOT"
[[ -f "$LINGBOT_CHECKPOINT/transformer/config.json" ]] || die "checkpoint config missing"
[[ -f "$LINGBOT_ROT6D20_STAT_PATH" ]] || die "normalization stats missing"
[[ -f "$LINGBOT_PROMPT_FILE" ]] || die "prompt missing"
for camera in observation.images.cam_high observation.images.cam_left_wrist observation.images.cam_right_wrist; do
  [[ -s "$LINGBOT_INPUT_IMAGE_DIR/$camera.png" ]] || die "camera missing: $camera"
done

bash "$REPO/script/run_rot6d20_minimal_inference.sh"
