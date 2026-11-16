# LingBot-VA on RMBench

Setup for training LingBot-VA on RMBench memory tasks. Based on XPolicyLab `fa431ec` (pinned by RMBench `7d1cdfc`).

## Status (16 Nov)

- EEF latents: all 10 RMBench tasks, on Drive. Each passes the trainer's loader check on every episode.
- Joint latents: press_button, rearrange_blocks, observe_and_pickup only.
- Multi-task loader check (all 10 EEF tasks under one folder, trainer's `MultiLatentLeRobotDataset`): pass.
- Tested on a Colab L4: training forward pass on our latents; eval server open-loop with posttrain-robotwin.
- Training launch on 2x H200: crashed on `--seed` (upstream bug), fixed by `apply_patches.sh` step 7.
- Multi-task dataset start-up: upstream Pool hung once with 10 tasks; `apply_patches.sh` step 8 loads tasks one by one.
- Second H200 launch: crashed at the first compile with `[Errno 13] Permission denied: 'nvcc'` (torch 2.9 probes `nvcc --version` on every compile; the cluster's PATH had a broken nvcc). `apply_patches.sh` step 9 guards the probe.
- Not tested: backward pass / full FSDP run, simulator eval.

## Setups

|            | EEF (main)                    | Joint (backup)        |
|------------|-------------------------------|-----------------------|
| Start from | lingbot-va-posttrain-robotwin | lingbot-va-base       |
| Actions    | 16 (hand pose + gripper)      | 14 (joints + gripper) |
| Config     | rmbench_eef_train             | robotwin30_train      |
| Eval       | patched model.py (ee mode)    | stock eval.sh         |

- Released RoboTwin weights are in EEF space; XPolicyLab's adapter only does joints. Hence two setups.
- Joint on observe_and_pickup clips 5.2% of steps with the robotwin30 norm stats. Use EEF for it.

## Precautions

- Code from this repo only. Drive holds only the latent zips. Ignore any older bundle zip.
- Download zips one at a time or with `gdown`. Browser download of the whole folder re-zips and splits it.
- Unzip in place. Do not move extracted folders (the trainer checks every episode's mp4 exists).
- `apply_patches.sh`: every step must print "patched" or "already patched".

## Install

```bash
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RMBench.git
cd RMBench
bash scripts/_install.sh
bash scripts/_download_assets.sh
```

- LingBot env: `XPolicyLab/policy/LingBot_VA/install.sh` (tested on XPolicyLab fa431ec).
- Keep `av<16` (lerobot 0.3.3 breaks with newer av).

```bash
bash apply_patches.sh /path/to/RMBench
```

Patches (only `XPolicyLab/policy/LingBot_VA/`):
1. EEF config, registered as `rmbench_eef` / `rmbench_eef_train`
2. `LINGBOT_VA_ATTN_WINDOW` env override (eval)
3. EEF control mode in `model.py`
4. optional flash-attn import
5. EEF data scripts
6. short eval output folder names ("File name too long" fix)
7. drop `--seed` from `train.sh` (`train.py` rejects it)
8. build per-task datasets without a multiprocessing Pool (`LINGBOT_VA_INIT_POOL=1` restores it)
9. guard torch's `nvcc --version` probe (a broken `nvcc` on PATH crashes the first compile)
10. wan_va's folder first on `sys.path` in `train.py` and `wan_va_server.py` (Dask's `distributed`, preinstalled on Colab, hid `wan_va/distributed`)

## Weights and data

```bash
huggingface-cli download robbyant/lingbot-va-base --local-dir /path/to/lingbot-va-base
huggingface-cli download robbyant/lingbot-va-posttrain-robotwin --local-dir /path/to/lingbot-va-posttrain-robotwin
```

- EEF zips: `<task>_eef_latents_fps4.zip`
- Joint zips: `<task>_latents_joint_fps5.zip` (press_button: `press_button_latents_fps5.zip`)
- Other zips in the Drive folder (raw data, logs) are not needed.

```bash
unzip press_button_eef_latents_fps4.zip -d /path/to/latents   # -> RMBench-press_button-aloha_agilex-ee
```

- CPU checks: `checks/`.

### Multi-task layout

The trainer loads every folder under `LINGBOT_VA_DATASET_PATH` that has `meta/info.json`.

```
/path/to/latents_ee/
  empty_emb.pt                          <- copy from any task folder (same in all)
  RMBench-press_button-aloha_agilex-ee/
  RMBench-rearrange_blocks-aloha_agilex-ee/
  ...
```

- Config reads `empty_emb.pt` from the parent itself.
- Do not mix `-ee` and `-joint` folders under one parent.
- Real folders only, no symlinks (discovery uses `os.walk`).
- Checked: all 10 EEF tasks under one parent load through `MultiLatentLeRobotDataset`.

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

- Other tasks: change the task in the path and the `train.sh` argument.
- All tasks: `LINGBOT_VA_DATASET_PATH=/path/to/latents_ee`, e.g. `bash train.sh RMBench all_tasks aloha_agilex ee 0 0,1,2,3,4,5,6,7`.
- Config: `train.sh` defaults to `robotwin30_train` (joint). EEF needs `LINGBOT_VA_CONFIG_NAME=rmbench_eef_train`.
- Joint config: wandb on with placeholder keys, crashes. Set `enable_wandb = False` in `configs/va_robotwin30_train_cfg.py`. EEF: off.
- Full fine-tune, FSDP, ~80 GB total (weights + grads + AdamW). Does not fit on one GPU.
- lr 1e-5, batch 1, grad accum 8, 5000 steps, checkpoint every 200.
- Short test run (EEF config only): `LINGBOT_VA_NUM_STEPS=10 LINGBOT_VA_SAVE_INTERVAL=5` (also `LINGBOT_VA_GRAD_ACCUM`, `LINGBOT_VA_LOAD_WORKER`). Unset = defaults.
- Training randomises window (4 to 64) and chunk size (1 to 4).

### Cluster notes

- Single node only (`run_va_posttrain.sh` uses `torch.distributed.run --nproc_per_node`).
- Last `train.sh` arg = GPU list -> `CUDA_VISIBLE_DEVICES` and GPU count. Under SLURM pass the IDs the job sees (`"$CUDA_VISIBLE_DEVICES"`).
- `.merged_ckpt` (training reads `.merged_ckpt/transformer`): `eval.sh` rebuilds it. Re-link before every run; one checkout per concurrently running job.
- No resume: only transformer weights are saved (optimizer state not). A killed job restarts at step 0. To continue from weights (fresh optimizer), point `.merged_ckpt` at `checkpoint_step_<N>`.
- Disk: every checkpoint kept, ~10.6 GB each (bf16), save every 200 -> ~265 GB per 5000-step run. Symlink `checkpoints/` to scratch or raise `save_interval` in `va_rmbench_eef_cfg.py`.
- Checkpoint path: `checkpoints/<bench>-<ckpt>-<env>-<action>-<seed>/checkpoints/checkpoint_step_<N>/transformer/`.
- Each rank loads the fp32 transformer on CPU first: ask for plenty of RAM.
- First step compiles flex attention (`torch.compile`): needs a C compiler and `Python.h` on the node, and an Ampere or newer GPU (bf16 + flex attention; T4/V100 don't work).
- torch 2.9 runs `nvcc --version` on every compile. Step 9 makes a broken `nvcc` on PATH harmless; no nvcc at all is fine too.
- flash-attn not needed (patch 4 makes its import optional).
- No internet on compute nodes: download weights and zips first, set `HF_HUB_OFFLINE=1`.

## Eval

Needs SAPIEN (Vulkan), ~24 GB VRAM.

```bash
# EEF
LINGBOT_VA_CONFIG_NAME=rmbench_eef LINGBOT_VA_BASE_MODEL_PATH=/path/to/lingbot-va-base \
bash eval.sh RMBench press_button /path/to/ckpt aloha_agilex ee 0 0 0 <policy_env> <eval_env>

# Joint
LINGBOT_VA_BASE_MODEL_PATH=/path/to/lingbot-va-base \
bash eval.sh RMBench press_button RMBench-press_button-aloha_agilex-joint-0 aloha_agilex joint 0 0 0 <policy_env> <eval_env>
```

- `EVAL_ENV_TYPE=debug`: wiring test without the simulator.
- EEF eval not run in the sim yet. Fallback: LingBot's client `lingbot_va/evaluation/robotwin/eval_polict_client_openpi.py`.
- Released weights do not solve press_button zero-shot. Fine-tune first.

## Memory

- `LINGBOT_VA_ATTN_WINDOW=<even n>`: keeps n/2 chunks (default 72). Chunk = 32 steps (EEF), 24 (joint).
- press_button: longest demo 629 steps, default window never forgets.
- observe_and_pickup: ~160 steps, only small windows matter.
- No-memory baseline: `rollout_mode: receding_horizon` in `deploy.yml`.
- Imagined video: `save_imagined_video: true` in `deploy.yml`.
- Hidden-variable logging (press counts etc.): not added.

## Other tasks (making latents)

1. `bash scripts/_download_data.sh <task>`
2. `ln -s demo_clean data/RMBench`, then `python XPolicyLab/scripts/transform_lerobot_v21_format.py "RMBench.<task>.aloha_agilex" --repo_id rmbench_<task>`
3. EEF: `python make_eef_lerobot.py --src <joint dataset> --hdf5-dir data/demo_clean/<task>/aloha_agilex/data --dst <eef dataset>`
4. Latents:
   - joint: `LINGBOT_VA_TARGET_FPS=5 bash process_data.sh RMBench <task> aloha_agilex joint`
   - EEF: `python process_data_eef.py --source-dataset <eef dataset> --output-dataset <out> --base-model <base> --target-fps 4`

- Default fps (10) gives the wrong stride on 15 Hz data. Log must say stride 3 (joint) or 4 (EEF).
- Check norm clipping before training.
