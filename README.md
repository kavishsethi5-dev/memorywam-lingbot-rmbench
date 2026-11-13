# LingBot-VA on RMBench

Setup for training LingBot-VA on RMBench memory tasks on the cluster.

Status (13 Nov): latents ready for press_button, rearrange_blocks and observe_and_pickup, two setups each. All pass the trainer's dataset loader. On a Colab L4, the eval server ran open-loop with the posttrain-robotwin weights and the training forward pass ran on our latents. Backward pass and simulator eval are untested.

|            | EEF (main)                    | Joint (backup)        |
|------------|-------------------------------|-----------------------|
| Start from | lingbot-va-posttrain-robotwin | lingbot-va-base       |
| Actions    | 16 (hand pose + gripper)      | 14 (joints + gripper) |
| Config     | rmbench_eef_train             | robotwin30_train      |
| Eval       | patched model.py (ee mode)    | stock eval.sh         |

The released RoboTwin weights are in EEF space, but XPolicyLab's adapter only does joints, hence two setups.

Note: observe_and_pickup on joint clips 5.2% of steps with the robotwin30 norm stats. Use EEF for it, or give it its own norm stats.

## Precautions

- Use this repo for the code. The Drive folder is only for the latent zips. If you find an older bundle zip anywhere, ignore it.
- Download the zips one at a time, or with `gdown --folder <link>`. Downloading the whole Drive folder from the browser makes Google re-zip it and sometimes split it into parts.
- Unzip the zips directly. Don't copy the extracted folders around, since the trainer checks that every episode's mp4 exists.
- After `apply_patches.sh`, every step should print "patched" or "already patched".

## Install

```bash
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RMBench.git
cd RMBench
bash scripts/_install.sh
bash scripts/_download_assets.sh
```

LingBot env: `XPolicyLab/policy/LingBot_VA/install.sh` (tested on XPolicyLab fa431ec). Keep `av<16`, lerobot 0.3.3 breaks with newer av.

```bash
bash apply_patches.sh /path/to/RMBench
```

Adds the EEF config and eval mode, the memory window env var, an optional flash-attn import, and a fix for "File name too long" in the eval server. Only touches `XPolicyLab/policy/LingBot_VA/`.

## Weights and data

```bash
huggingface-cli download robbyant/lingbot-va-base --local-dir /path/to/lingbot-va-base
huggingface-cli download robbyant/lingbot-va-posttrain-robotwin --local-dir /path/to/lingbot-va-posttrain-robotwin
```

Latents are on Drive: `<task>_eef_latents_fps4.zip` and `<task>_latents_joint_fps5.zip` (press_button joint is `press_button_latents_fps5.zip`). See Precautions above.

```bash
unzip press_button_eef_latents_fps4.zip -d /path/to/latents   # -> RMBench-press_button-aloha_agilex-ee
```

CPU checks in `checks/` if needed.

Running on a shared cluster (disk, no resume, `.merged_ckpt`, multi-task layout, SLURM example): see [CLUSTER.md](CLUSTER.md).

## Train

From `XPolicyLab/policy/LingBot_VA/`. Re-link `.merged_ckpt` before every run (eval overwrites it).

```bash
# EEF
ln -sfn /path/to/lingbot-va-posttrain-robotwin .merged_ckpt
LINGBOT_VA_CONFIG_NAME=rmbench_eef_train \
LINGBOT_VA_DATASET_PATH=/path/to/latents/RMBench-press_button-aloha_agilex-ee \
bash train.sh RMBench press_button aloha_agilex ee 0 0,1,2,3,4,5,6,7

# Joint
ln -sfn /path/to/lingbot-va-base .merged_ckpt
LINGBOT_VA_DATASET_PATH=/path/to/latents/RMBench-press_button-aloha_agilex-joint \
bash train.sh RMBench press_button aloha_agilex joint 0 0,1,2,3,4,5,6,7
```

For other tasks, change the task name in the path and the `train.sh` argument.

- Joint config has wandb on with placeholder keys and will crash. Set `enable_wandb = False` in `configs/va_robotwin30_train_cfg.py` or add keys in `run_va_posttrain.sh`. EEF has it off.
- Full fine-tune, FSDP, roughly 80 GB total. Won't fit on one GPU.
- lr 1e-5, batch 1, grad accum 8, 5000 steps, checkpoint every 200. Training randomises the window (4 to 64) and chunk size (1 to 4).

## Eval

Needs SAPIEN (Vulkan), about 24 GB VRAM.

```bash
# EEF
LINGBOT_VA_CONFIG_NAME=rmbench_eef LINGBOT_VA_BASE_MODEL_PATH=/path/to/lingbot-va-base \
bash eval.sh RMBench press_button /path/to/ckpt aloha_agilex ee 0 0 0 <policy_env> <eval_env>

# Joint
LINGBOT_VA_BASE_MODEL_PATH=/path/to/lingbot-va-base \
bash eval.sh RMBench press_button RMBench-press_button-aloha_agilex-joint-0 aloha_agilex joint 0 0 0 <policy_env> <eval_env>
```

`EVAL_ENV_TYPE=debug` tests wiring without the simulator. EEF eval hasn't run in the sim yet. If it breaks, fall back to LingBot's own client (`lingbot_va/evaluation/robotwin/eval_polict_client_openpi.py`).

Zero-shot, the released weights don't solve press_button, so fine-tune first.

## Memory knobs

- `LINGBOT_VA_ATTN_WINDOW=<even n>` keeps n/2 chunks (default 72). A chunk is 32 steps for EEF, 24 for joint.
- Longest press_button demo is 629 steps, so the default window never forgets. observe_and_pickup is ~160 steps, so only small windows matter there.
- No-memory baseline: `rollout_mode: receding_horizon` in `deploy.yml`.
- Imagined video: `save_imagined_video: true` in `deploy.yml`.
- Hidden-variable logging (press counts etc.) not added yet.

## Other tasks

1. `bash scripts/_download_data.sh <task>`
2. `ln -s demo_clean data/RMBench`, then `python XPolicyLab/scripts/transform_lerobot_v21_format.py "RMBench.<task>.aloha_agilex" --repo_id rmbench_<task>`
3. EEF: `python make_eef_lerobot.py --src <joint dataset> --hdf5-dir data/demo_clean/<task>/aloha_agilex/data --dst <eef dataset>`
4. Latents:
   - joint: `LINGBOT_VA_TARGET_FPS=5 bash process_data.sh RMBench <task> aloha_agilex joint`
   - EEF: `python process_data_eef.py --source-dataset <eef dataset> --output-dataset <out> --base-model <base> --target-fps 4`

The default fps (10) gives the wrong stride on 15 Hz data. Check the log says stride 3 (joint) or 4 (EEF), and check norm clipping before training.
