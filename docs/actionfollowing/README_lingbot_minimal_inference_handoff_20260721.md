# LingBot-VA ActionFollowing 最小推理交接（2026-07-21）

本文提供 LingBot-VA native Rot6D20 checkpoint 的最小 reload 和 image-to-video-action 推理路径。结构参考 [Ctrl-World 单任务模型交接](https://github.com/Ricardo520nono/ctrl-world-train-wjx/blob/dev-csx-codex/code/scripts_daily/20260719/README_ctrlworld_single_task_handoff_20260719.md)，明确 checkpoint、训练配置、归一化、action space、相机排布、输入输出和当前证据边界。

## 当前结论

- 可用 checkpoint 来自 `clean128` bring-up，而不是完整 clean/mix4 baseline：它只预计算 128 个 clean 样本、训练 50,000 steps。
- 权重的 action boundary 已经是 native 20D，但旧保存逻辑生成的 `transformer/config.json` 仍写 `action_dim=30`；这会导致 `from_pretrained` 按 30D 建层后加载 20D tensor 失败。
- 训练 config 的 `attn_mode=flex` 也不能直接用于 inference；推理必须为 `torch` 或 `flashattn`。
- 修复策略是不修改原 checkpoint：创建派生 inference bundle，链接原权重和 base tokenizer/text encoder/VAE，只写一份 `action_dim=20, attn_mode=torch` 的推理 config，并记录 audit。
- 最小推理 retry job `job-51cqdsi6mjj4` 已 `Succeeded` 并完成独立验收：源 checkpoint 未修改，派生 bundle 为 `action_dim=20, attn_mode=torch`，三路原始相机、视频、physical Rot6D20 `[32,20]` action、metadata 和 `INFERENCE_RESULT.txt` 均已通过。
- 2026-07-21 启动的两条名义 full50 clean/mix4 job 不只是从 40K 漂移到 50K：真实持久化 bootstrap 日志证明两条都被 `lingbotva_env.local.sh` 覆盖成 `clean + 128 samples + 50000 steps`，并写入同一个 `save root`。它们不能作为 clean/mix4 baseline 结果，且存在并发覆盖 checkpoint 的风险。
- launcher 已改为“调用者/AIHC 显式环境变量优先于本地 env 默认值”；修复后的 full50 任务还使用独立 worktree、固定源码 commit、独立输出目录和训练前 contract gate。
- 修复后的 full50 clean job `job-ih9xybn3brte` 被外部人工终止在 precompute 阶段，没有进入 optimizer training，也没有 checkpoint；不得把它写成代码失败或已完成 baseline。
- 修复后的 full50 mix4 retry `job-uoma5dw5mcfi` 当前在 `train` 队列运行 precompute，已证明 600,000 样本、50 tasks、physical Rot6D20 `[32,20]`、runtime action_dim 20、40,000 steps、effective batch 8、每 5,000 steps checkpoint 的合同；尚未产生训练 checkpoint，不能标记为训练通过。

## 模型与 checkpoint

```text
public repo:
https://github.com/EasonAI-5589/lingbot-va

branch:
codex/rot6d20-action-adapter-20260718

PR:
https://github.com/EasonAI-5589/lingbot-va/pull/1

checkpoint:
/mnt/public_ckp/cscsx_projects/lingbotva_train/native20_rot6d20_clean4_20260718_v1/checkpoints/checkpoint_step_50000

base model assets:
/mnt/public_ckp/lingbot-va-base
```

训练 job `job-9xnzba90ngou` 已 `Succeeded`，但实验名为 `ACWM_lingbotva_native20_rot6d20_clean128_8gpu_50k_retry7_20260718`。它只能证明 20D adapter 可以训练，不能证明 full50 clean/mix4 baseline 已复现。

### 修复后 full50 40K 的当前状态（2026-07-22）

| protocol | AIHC job | 当前状态 | 已证明 / 未证明 |
| --- | --- | --- | --- |
| clean | `job-ih9xybn3brte` | `ManualTermination` | 已启动真实 clean precompute，但在完成 472,622 个 genuine current1+future32 window 前被人工终止；无 optimizer step、无 checkpoint、无最终 audit。若要补齐 clean baseline，需要明确授权重新提交。 |
| mix4 | `job-uoma5dw5mcfi` | `Running`（precompute） | `[TRAIN_CONTRACT]` 已证明 600,000 样本、50 tasks、`[32,20]`、action_dim 20、40K、batch 8、ckpt/5K；`[PROVENANCE]` 为 `667bb9eebc742de2439b9c2b2d6a34902c9dd015`。尚未进入训练，不能推断 loss 或 checkpoint。 |

修复后输出根互相独立：

```text
clean:
/mnt/gyc_ckp/Action-Following/outputs/lingbot/full50/clean/train_40000_20260721_native20_fixed_job-ih9xybn3brte

mix4:
/mnt/gyc_ckp/Action-Following/outputs/lingbot/full50/mix4/train_40000_20260721_native20_fixed_job-uoma5dw5mcfi
```

mix4 当前使用固定源码 `667bb9eebc742de2439b9c2b2d6a34902c9dd015`。只有出现 finite optimizer loss、`checkpoint_step_5000` 的 20D tensor/config audit，并最终生成 `checkpoint_step_40000`、`CHECKPOINT_AUDIT.json`、`TRAIN_AUDIT.json` 和 `TRAIN_RESULT.txt`，才能验收为合格 baseline。

### 当前 full50 训练参数覆盖告警（高严重度）

```text
clean job: job-5j57hh9vd8op
mix4 job:  job-c0dngmocgbcd

两条任务的真实持久化 bootstrap 日志均为：
protocol           : clean
family             : clean
precompute samples : 128
train steps        : 50000
precompute root    : /mnt/public_ckp/cscsx_projects/lingbotva_train/precomputed_native20_clean128_20260718_v1
save root          : /mnt/public_ckp/cscsx_projects/lingbotva_train/native20_rot6d20_clean4_20260718_v1
```

预期与实际的差异：

| 项目 | clean job 预期 | mix4 job 预期 | 两条任务实际值 |
| --- | --- | --- | --- |
| protocol | full50 clean | full50 mix4 | clean |
| data volume | 正式 clean 资产 | 正式 mix4 资产与约定采样比例 | 128 个 clean 样本 |
| optimizer steps | 40,000 | 40,000 | 50,000 |
| output root | clean 独立目录 | mix4 独立目录 | 同一个旧 clean bring-up 目录 |

影响不是“实验名写错”这么简单：

1. 没有产生可比较的 full50 clean/mix4 两组结果；
2. 训练步数不满足约定的 40K；
3. 两个独立训练进程可能同时创建或覆盖相同的 `checkpoint_step_*` 目录；
4. 该共享目录中的新旧 checkpoint 不能仅凭目录名确定 provenance，必须结合 job 持久化日志、文件 mtime 和 tensor/config audit；
5. 这些 job 即使调度状态变成 `Succeeded`，也不能通过 ActionFollowing baseline 验收。

根因是旧 launcher 在 AIHC bootstrap 已导出 `LINGBOT_PROTOCOL`、`LINGBOT_PRECOMPUTE_SAMPLES`、`LINGBOT_NUM_STEPS`、`LINGBOT_PRECOMPUTE_ROOT` 和 `LINGBOT_SAVE_ROOT` 后，又 source 了 gitignored 的 `script/lingbotva_env.local.sh`，本地旧默认值反向覆盖了调用者参数。当前 `script/run_rot6d20_native20_train_aihc.sh` 会先保存调用者环境，source 本地默认值后再恢复调用者显式值，因此未来提交不会再发生同类覆盖。

这两条任务没有被停止。停止 Running job 属于单独授权动作；在获得授权前只能记录风险，不能自动终止。后续严格复现应使用修复后的 launcher，分别提交独立 clean/mix4 任务，并在训练开始前从 bootstrap 日志硬验收 protocol、sample count、steps、precompute root 与 save root。

### 保存时 config bug 与 30D 结论

线上对 `checkpoint_step_30000` 和 `checkpoint_step_50000` 的只读检查得到完全一致的结果：

```text
checkpoint_step_30000:
  transformer/config.json action_dim = 30
  transformer/config.json attn_mode  = flex
  action_embedder.weight             = [3072,20]
  action_proj_out.weight             = [20,3072]

checkpoint_step_50000:
  transformer/config.json action_dim = 30
  transformer/config.json attn_mode  = flex
  action_embedder.weight             = [3072,20]
  action_proj_out.weight             = [20,3072]
```

因此 30D 的准确结论是：

- 上游 LingBot 的默认 30D action layout 对其原生任务不一定错误；
- 对本 ActionFollowing checkpoint，训练运行时和真实权重边界都是 native Rot6D20 20D；
- 错的是保存产物中的 stale `action_dim=30`，不是把训练数据证明成了 30D；
- 直接 `from_pretrained()` 会先按 config 创建 30D 层，再加载 20D tensor，触发 size mismatch；
- 不能把物理 20D action 补零到 30D 来绕过检查，因为多出的通道没有本任务定义，会改变 action space 语义并掩盖 checkpoint provenance 问题。

代码现已在 `wan_va/train.py` 的保存路径显式写入 runtime `config.action_dim`，防止以后生成同类坏 checkpoint。已有 checkpoint 通过 `script/prepare_rot6d20_inference_bundle.py` 派生，不原地修改：派生 bundle 链接原 weights/base assets，只生成一份 inference-only `action_dim=20, attn_mode=torch` config，并在 `INFERENCE_BUNDLE_AUDIT.json` 中记录源/派生 config SHA256、stats SHA256、tensor shape 和 `source_checkpoint_modified=false`。

## Action space 与归一化

输入/输出 physical action 都是 robot base frame 的 delta-EE Rot6D20：

```text
[left_dx, left_dy, left_dz,
 left_r00, left_r10, left_r20, left_r01, left_r11, left_r21,
 left_gripper,
 right_dx, right_dy, right_dz,
 right_r00, right_r10, right_r20, right_r01, right_r11, right_r21,
 right_gripper]
```

```text
rot6d = concat(R[:,0], R[:,1])
```

坐标轴为 `+X robot forward / +Y robot left / +Z robot up`。20D action 不是官方 LingBot 旧 30D layout，也不是 14D Euler、16D quaternion-relative 或 joint qpos。

LingBot 和 Cosmos 最大的 action boundary 差异是归一化：

```text
LingBot model input = (physical - q01) / (q99 - q01 + 1e-6) * 2 - 1
LingBot public output = (model_output + 1) / 2 * (q99 - q01 + 1e-6) + q01
```

本 checkpoint 必须复用其训练时的 exact stats：

```text
/mnt/public_ckp/cscsx_projects/ctrl_world_train/dataset_meta_info/action_following_current1_future32_clean_enhanced_explore_test4v2_fulldesc_rot6d20_frameidxfix_20260711/stat.json
```

这个 stats 来源是当时 bring-up 使用的 test4v2 clean+enhanced 资产，不等于 full50 clean stats，也不等于 full50 mix4 stats。换成正式 full50 clean/mix4 checkpoint 后，必须分别使用各自训练时冻结并保存的 20D q01/q99；不得跨协议复用。也不能把 policy normalized action、Ctrl-World normalized action或 Cosmos physical action在未经过该 stats bridge 时直接当作 LingBot internal action。

## 相机排布

原始输入文件读取顺序：

```text
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
```

LingBot `robotwin_tshape` 在 latent 中的 native 排布是：

```text
+-------------+-------------+
| left_wrist  | right_wrist |
+-------------+-------------+
|          cam_high         |
+---------------------------+
```

这与 Cosmos3 的 `head top / wrists bottom` 相反。输入目录仍提供三个独立 RGB PNG，由 LingBot `_encode_obs` 按上述 native 布局编码；不要先用 Cosmos 的 `current_tshape.png` 替代三个原始相机文件。

## 最小推理输入输出

为了与 Cosmos 对照，固定复用同一个 ActionFollowing clean `place_burger_fries` 当前 observation 和 RoboTwin full_description：

```text
input directory:
observation.images.cam_high.png
observation.images.cam_left_wrist.png
observation.images.cam_right_wrist.png
full_description.txt

output:
demo.mp4
pred_actions_physical_rot6d20.npy   # [32,20], finite, 已反归一化
inference_metadata.json
```

LingBot 不是 Cosmos 的纯 forward-dynamics API：该 standalone i2va 入口由当前三视角和 prompt 自回归联合生成 video 与 action。它生成的 `[32,20]` 是 policy/action branch 预测，不是输入给 Cosmos 的 expert action。两者的最小推理任务应分别报告，不能用同一个 “action-conditioned video” 标签混写。

## 运行命令

```bash
cd /mnt/gyc/LingbotVA2.0/lingbot-va

export LINGBOT_CHECKPOINT=/mnt/public_ckp/cscsx_projects/lingbotva_train/native20_rot6d20_clean4_20260718_v1/checkpoints/checkpoint_step_50000
export LINGBOT_BASE_MODEL=/mnt/public_ckp/lingbot-va-base
export LINGBOT_ROT6D20_STAT_PATH=/mnt/public_ckp/cscsx_projects/ctrl_world_train/dataset_meta_info/action_following_current1_future32_clean_enhanced_explore_test4v2_fulldesc_rot6d20_frameidxfix_20260711/stat.json
export LINGBOT_INPUT_IMAGE_DIR=/mnt/gyc_ckp/Action-Following/outputs/handoff_inputs/place_burger_fries_clean0_20260721
export LINGBOT_PROMPT_FILE=${LINGBOT_INPUT_IMAGE_DIR}/full_description.txt
export LINGBOT_HANDOFF_ROOT=/mnt/gyc_ckp/Action-Following/outputs/lingbot/minimal_inference/checkpoint_step_50000_place_burger_fries_clean0_20260721

bash script/run_rot6d20_minimal_inference.sh
```

脚本会：

1. 检查 checkpoint tensor 确实为 `[3072,20]`/`[20,3072]`；
2. 创建 reload-clean inference bundle，不改源 checkpoint；
3. 固定加载 20D q01/q99；
4. 使用 1 GPU、4 chunks、每 chunk `2 latent frames x 4 action/frame`，生成正好 32 个 action；
5. 核验 action shape `[32,20]` 和 finite；
6. 生成 `INFERENCE_RESULT.txt`。

## 输出目录和验收

```text
<LINGBOT_HANDOFF_ROOT>/
  model_bundle/
    transformer/config.json
    transformer/diffusion_pytorch_model.safetensors -> source checkpoint
    vae/ -> base model
    tokenizer/ -> base model
    text_encoder/ -> base model
    INFERENCE_BUNDLE_AUDIT.json
  output/
    demo.mp4
    pred_actions_physical_rot6d20.npy
    inference_metadata.json
  minimal_inference.log
  INFERENCE_RESULT.txt
```

本次 AIHC job 的固定输出根为：

```text
/mnt/gyc_ckp/Action-Following/outputs/lingbot/minimal_inference/checkpoint_step_50000_place_burger_fries_clean0_aihc_train_20260721_job-51cqdsi6mjj4
```

`job-51cqdsi6mjj4` 的实际验收结果：

- source checkpoint 的 SHA/config 未改变，`source_checkpoint_modified=false`；
- bundle config 为 `action_dim=20`、`attn_mode=torch`，tensor boundary 为 native 20D；
- exact q01/q99 stats SHA256 以 `2ccaf0` 开头，并记录在 audit/metadata；
- 输入为三路原始相机，LingBot native 排布为 wrists-top、head-bottom；
- `demo.mp4` 为 H.264、320×384、29 帧，可解码且非空；
- `pred_actions_physical_rot6d20.npy` 恰为 `[32,20]`、全部 finite，metadata 明确为 physical Rot6D20；
- `INFERENCE_RESULT.txt` 为 passed；
- 该结果只证明 clean128 checkpoint 的正确 20D reload/inference，不证明 full50 clean/mix4 baseline 已完成。

## 代码位置

```text
wan_va/train.py
wan_va/configs/va_robotwin_rot6d20_cfg.py
wan_va/configs/va_robotwin_rot6d20_i2va.py
wan_va/wan_va_server.py
script/prepare_rot6d20_inference_bundle.py
script/run_rot6d20_minimal_inference.sh
tests/test_rot6d20_inference_contract.py
docs/actionfollowing/README_lingbot_minimal_inference_handoff_20260721.md
```
