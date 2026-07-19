#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LingBot-VA 2.0 / Action-Following rot6d20 — site configuration TEMPLATE
#
# Every path below is site-specific. Do NOT commit real values.
#
# Setup:
#     cp script/lingbotva_env.example.sh script/lingbotva_env.local.sh
#     $EDITOR script/lingbotva_env.local.sh
#
# The launcher sources script/lingbotva_env.local.sh automatically.
# Keep it somewhere else? Point LINGBOT_ENV_FILE at it instead.
# lingbotva_env.local.sh is gitignored.
# ---------------------------------------------------------------------------

# --- Interpreters ----------------------------------------------------------
# Training / preflight environment (needs torch + diffusers).
export LINGBOT_PYTHON=/path/to/conda/envs/lingbot-va/bin/python

# Latent precompute environment. May differ from LINGBOT_PYTHON, but BEWARE:
# a diffusers/torch version skew between the two changes VAE numerics.
# Verify they match before a full-scale precompute.
export LINGBOT_PRECOMPUTE_PYTHON=/path/to/precompute-venv/bin/python
# Extra PYTHONPATH entry the precompute environment needs (leave empty if none).
export LINGBOT_PRECOMPUTE_PYTHONPATH=/path/to/precompute-framework

# --- Model weights ---------------------------------------------------------
# lingbot-va-base checkpoint (~23G). Contains transformer/ vae/ text_encoder/.
# NOTE: transformer/config.json's attn_mode must be "flex" for training and
# "torch"/"flashattn" for inference. If this checkpoint is on shared storage,
# treat it as read-only — flipping attn_mode breaks other people's runs.
export LINGBOT_WAN22_PATH=/path/to/lingbot-va-base

# --- Source data -----------------------------------------------------------
# Action-Following LeRobot Rot6D root (the 'train' subdir; 350 datasets,
# action/state = 20D, LeRobot codebase_version v3.0).
export AFD_ROOT=/path/to/ActionFollowingData_LeRobot_Rot6D/train

# Dataset supplying empty_emb.pt for classifier-free guidance.
export LINGBOT_ROT6D20_LATENT_ROOT=/path/to/robotwin-clean-and-aug-lerobot

# Canonical rot6d20 action root + its train manifest.
export LINGBOT_ROT6D20_ACTION_ROOT=/path/to/rot6d20/action_root
export LINGBOT_ROT6D20_MANIFEST_PATH="${LINGBOT_ROT6D20_ACTION_ROOT}/manifests/train.jsonl"

# Action normalization quantiles (q01/q99), 20 channels.
# WARNING: these must be computed over the SAME task set and the SAME action
# convention (delta_ee rot6d, not state) as the run. Reusing stats from a
# different task subset silently skews normalization — it does not error.
export LINGBOT_ROT6D20_STAT_PATH=/path/to/dataset_meta_info/<stat-dir>/stat.json

# --- Outputs ---------------------------------------------------------------
export LINGBOT_PRECOMPUTE_ROOT=/path/to/outputs/precomputed_<tag>
export LINGBOT_SAVE_ROOT=/path/to/outputs/train_<tag>

# --- Optional: container mount bootstrap -----------------------------------
# Only used when the launcher runs somewhere the expected mounts are missing
# (e.g. an AIHC container) and needs to symlink them into place.
# Leave both empty to disable the symlink bootstrap entirely.
export LINGBOT_WORKSPACE_MOUNT=
export LINGBOT_WORKSPACE_LINK=
export LINGBOT_CKP_MOUNT=
export LINGBOT_CKP_LINK=

# --- Run parameters --------------------------------------------------------
export LINGBOT_PROTOCOL=clean          # clean | mix4
export LINGBOT_ROT6D20_FAMILY=clean    # clean | perturbed | random_feasible | counterfactual_replay | exploration
export LINGBOT_PRECOMPUTE_SAMPLES=128  # raise for a real run; 128 is smoke-scale
export LINGBOT_NUM_STEPS=50000
export LINGBOT_SAVE_INTERVAL=1000
export LINGBOT_ACTION_PER_FRAME=4      # 8 latent frames x 4 = 32 action steps
export LINGBOT_LOAD_WORKERS=8
export LINGBOT_ENABLE_WANDB=0          # set 1 only after WANDB_* are configured
export NGPU=8
