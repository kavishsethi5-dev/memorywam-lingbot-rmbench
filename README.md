# LingBot-VA on RMBench (memory tasks): setup for cluster training

Status 12 Nov 2026: data for RMBench `press_button` is ready for two setups. Both pass the trainer's own dataset loader. Nothing has been trained or evaluated yet.

| | EEF (suggested main) | Joint (backup) |
|---|---|---|
| Start from | `lingbot-va-posttrain-robotwin` | `lingbot-va-base` |
| Actions | 16-dim hand pose + gripper | 14-dim joints + gripper |
| Config | `rmbench_eef_train` | `robotwin30_train` |
| Latents | `press_button_eef_latents_fps4.zip` | `press_button_latents_fps5.zip` |
| Eval | patched `model.py` (`ee` mode), untested | stock XPolicyLab `eval.sh` |

Why two: the released RoboTwin weights are in EEF space, and XPolicyLab's LingBot adapter only does joint space. Joint from base has to learn the robot from 50 demos. EEF starts from weights that already know this robot.

## 1. Install

RMBench with the pinned XPolicyLab submodule (tested on XPolicyLab `fa431ec`), on a GPU machine:

```bash
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RMBench.git
cd RMBench
bash scripts/_install.sh            # conda env, CuRobo, XPolicyLab
bash scripts/_download_assets.sh
```

LingBot env: follow `XPolicyLab/policy/LingBot_VA/install.sh` (python 3.10, torch 2.9 cu126, diffusers 0.36.0, transformers 4.55.2, `lerobot==0.3.3`). Keep `av<16`, lerobot 0.3.3 breaks with newer av.

Then apply our patches:

```bash
bash apply_patches.sh /path/to/RMBench
```

This adds the EEF config, the EEF mode in `model.py`, an env var for the memory window, and makes the flash-attn import optional. It only touches files inside `XPolicyLab/policy/LingBot_VA/`.

## 2. Weights and data

```bash
huggingface-cli download robbyant/lingbot-va-base --local-dir /path/to/lingbot-va-base
huggingface-cli download robbyant/lingbot-va-posttrain-robotwin --local-dir /path/to/lingbot-va-posttrain-robotwin
```

Latents: unzip one of the zips (shared Drive folder). Use the zip as is, do not copy the folders: the trainer checks that every episode's mp4 exists, and the zip has real files.

```bash
unzip press_button_eef_latents_fps4.zip -d /path/to/latents    # -> RMBench-press_button-aloha_agilex-ee
unzip press_button_latents_fps5.zip -d /path/to/latents        # -> RMBench-press_button-aloha_agilex-joint
```

Optional checks on CPU: `checks/eef_loader_test.py`, `checks/joint_loader_test.py` (shapes), and `checks/verify_eef_eval.py` (feeds the training targets through the patched EEF eval code as if the server had predicted them, and checks the absolute poses against the HDF5).

## 3. Train

Run from `XPolicyLab/policy/LingBot_VA/`. The trainer loads `.merged_ckpt/transformer`, so point `.merged_ckpt` at the starting weights first. Eval rewrites this link, so re-link before every training run.

**EEF:**
```bash
ln -sfn /path/to/lingbot-va-posttrain-robotwin .merged_ckpt
LINGBOT_VA_CONFIG_NAME=rmbench_eef_train \
LINGBOT_VA_DATASET_PATH=/path/to/latents/RMBench-press_button-aloha_agilex-ee \
bash train.sh RMBench press_button aloha_agilex ee 0 0,1,2,3,4,5,6,7
```

**Joint:**
```bash
ln -sfn /path/to/lingbot-va-base .merged_ckpt
LINGBOT_VA_DATASET_PATH=/path/to/latents/RMBench-press_button-aloha_agilex-joint \
bash train.sh RMBench press_button aloha_agilex joint 0 0,1,2,3,4,5,6,7
```

Things that will break otherwise:
- **wandb:** the joint config has `enable_wandb = True` and `lingbot_va/script/run_va_posttrain.sh` has placeholder keys, so `wandb.login` crashes. Either fill in `WANDB_API_KEY` etc. in that script, or set `va_robotwin30_train_cfg.enable_wandb = False` in `configs/va_robotwin30_train_cfg.py`. The EEF config already has it off.
- Memory: full fine-tune in fp32 with FSDP, about 80 GB in total before activations (our estimate). A single GPU does not work.
- `attn_mode`: no manual edit needed. `train.py` forces `flex` and `wan_va_server.py` forces `torch`.

Settings for both: lr 1e-5, batch 1, grad accum 8, 5000 steps, checkpoint every 200 steps to `checkpoints/RMBench-press_button-aloha_agilex-<ee|joint>-0/checkpoints/checkpoint_step_<N>/transformer/`. Training randomises the attention window (4 to 64) and chunk size (1 to 4) every step (`train.py::_prepare_input_dict`).

## 4. Evaluate

Needs the simulator (SAPIEN, Vulkan) and about 24 GB VRAM for the model.

```bash
# EEF (zero-shot released weights: pass the posttrain folder as the checkpoint)
LINGBOT_VA_CONFIG_NAME=rmbench_eef LINGBOT_VA_BASE_MODEL_PATH=/path/to/lingbot-va-base \
bash eval.sh RMBench press_button /path/to/ckpt_or_run_dir aloha_agilex ee 0 0 0 <policy_env> <eval_env>

# Joint
LINGBOT_VA_BASE_MODEL_PATH=/path/to/lingbot-va-base \
bash eval.sh RMBench press_button RMBench-press_button-aloha_agilex-joint-0 aloha_agilex joint 0 0 0 <policy_env> <eval_env>
```

`EVAL_ENV_TYPE=debug` checks the wiring without the simulator.

**EEF eval is untested.** The model predicts poses relative to the pose at the last server reset. The patched `model.py` adds that pose back (same maths as LingBot's own RoboTwin client; round trip checked numerically) and sends absolute poses to `take_action(..., action_type='ee')`. If something fails, the fallback is LingBot's own client in `lingbot_va/evaluation/robotwin/eval_polict_client_openpi.py` (set `robowin_root`, symlink `task_config -> env_cfg/task_config`).

## 5. Memory experiments

- **Window (memory dial):** `LINGBOT_VA_ATTN_WINDOW=<even n>` before `eval.sh` keeps n/2 chunks of history (default 72 = 36 chunks). A chunk is 32 env steps for EEF (stride 4) and 24 for joint (stride 3). The longest press_button demo is 629 steps, so at the default window nothing is ever forgotten. Shrinking the window creates forgetting.
- **No-memory baseline:** `rollout_mode: receding_horizon` in `deploy.yml` resets the cache every chunk.
- **Imagined video:** `save_imagined_video: true` in `deploy.yml`.
- **Saved by default:** the server writes predicted latents and actions per chunk to `policy/LingBot_VA/visualization/real/`.
- Press Button's hidden variables (`card_id_1/2`, `press_cnt_1/2`, check button) are attributes of the task env (`envs/press_button.py`). Logging them per step is not added yet.

## 6. How the data was made (for other tasks)

1. `bash scripts/_download_data.sh <task>` (HDF5, 15 Hz).
2. `ln -s demo_clean data/RMBench`, then `python XPolicyLab/scripts/transform_lerobot_v21_format.py "RMBench.<task>.aloha_agilex" --repo_id rmbench_<task>` (LeRobot v2.1, joint).
3. EEF only: `python make_eef_lerobot.py --src <joint dataset> --hdf5-dir data/demo_clean/<task>/aloha_agilex/data --dst <eef dataset>`. Episodes are matched to HDF5 files by exact joint-action match.
4. Latents (1 GPU, needs `lingbot-va-base` vae + text encoder):
   - joint: `LINGBOT_VA_TARGET_FPS=5 bash process_data.sh RMBench <task> aloha_agilex joint` (stride 3 at 15 Hz = 12 actions per latent frame, matches the config).
   - EEF: `python process_data_eef.py --source-dataset <eef dataset> --output-dataset <out> --base-model <base> --target-fps 4` (stride 4 = 16 actions per latent frame).
   - The default target fps (10) gives stride 2 on 15 Hz data, which silently mismatches the configs. Check the stride in the log for any new task.
5. `checks/eef_norm_check.py` checks the RoboTwin normalisation fits (press_button: 0% clipped).
