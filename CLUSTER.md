# Running on a cluster

What you might need to change to run this on a shared cluster. Everything here is from the code at
XPolicyLab `fa431ec` (the commit RMBench `7d1cdfc` pins). Paths are relative to
`XPolicyLab/policy/LingBot_VA/` unless stated.

## What runs where

| Step | Needs | Status |
|---|---|---|
| Latents | 1 GPU (~24 GB) | Done, zips on Drive |
| Training | 1 node, 8 x 80 GB | Not run yet |
| Eval | 1 GPU (~24 GB), Vulkan, CuRobo | Not run in the sim yet |

- Training is single-node only: `lingbot_va/script/run_va_posttrain.sh` uses
  `torch.distributed.run --nproc_per_node`, no multi-node arguments.
- Training uses flex attention and the eval server uses torch attention (both set in code), so
  flash-attn is not needed. `apply_patches.sh` makes its import optional, so you can skip the
  flash-attn build in `install.sh` if it fails.

## Environment

- `install.sh` creates a conda env (Python 3.10, torch 2.9 cu126), installs lerobot 0.3.3
  with `--no-deps`, and `pip install -e` both `lingbot_va` and XPolicyLab. Keep `av<16`.
- Then `bash apply_patches.sh /path/to/RMBench` from this repo. Every step should print
  "patched" or "already patched".
- **No internet on compute nodes:** download both weights (`lingbot-va-base`,
  `lingbot-va-posttrain-robotwin`) and the latent zips on a login node first. Setting
  `HF_HUB_OFFLINE=1` in jobs avoids surprise download attempts.

## Data layout (one or many tasks)

The trainer (`MultiLatentLeRobotDataset`) loads every folder under `LINGBOT_VA_DATASET_PATH`
that has a `meta/info.json`, so multi-task training needs no code change:

```
/scratch/latents_ee/
  empty_emb.pt                          <- copy one here (from any task)
  RMBench-press_button-aloha_agilex-ee/
  RMBench-rearrange_blocks-aloha_agilex-ee/
  ...
```

- The config reads `empty_emb.pt` from the parent folder itself (`dataset_path/empty_emb.pt`).
  It is the empty-prompt text embedding, the same for every task.
- **Do not mix EEF (`-ee`) and joint (`-joint`) datasets under one parent.** They have
  different action layouts and configs.
- For one task, point `LINGBOT_VA_DATASET_PATH` at that task's folder directly.
- Unzip the zips in place; don't copy the extracted folders around (the loader checks every
  episode's mp4 exists).

## Checkpoints and disk

- Written inside the checkout:
  `checkpoints/<bench>-<ckpt>-<env>-<action>-<seed>/checkpoints/checkpoint_step_<N>/transformer/`.
- Every `save_interval` = 200 steps, and all are kept. About 10.6 GB each (bf16 transformer),
  so about **265 GB per 5000-step run**.
- Either symlink `checkpoints/` to scratch before the run, or raise `save_interval` in the
  train config (`va_rmbench_eef_cfg.py` for EEF, `va_robotwin30_train_cfg.py` for joint).

## No resume

- `train.py::save_checkpoint` saves only the transformer weights; the optimizer state line is
  commented out. A killed job starts again from step 0.
- To continue from a checkpoint's weights (fresh optimizer), point `.merged_ckpt` at that
  `checkpoint_step_<N>` folder (training reads `.merged_ckpt/transformer`).
- Set the SLURM time limit with this in mind.

## `.merged_ckpt` (one per checkout)

- Training loads the starting weights from `.merged_ckpt/transformer`.
  - EEF: `ln -sfn /path/to/lingbot-va-posttrain-robotwin .merged_ckpt`
  - Joint: `ln -sfn /path/to/lingbot-va-base .merged_ckpt`
- `eval.sh` rebuilds `.merged_ckpt` (`prepare_merged_ckpt.py`). So **re-link before every
  training run**, and use **one checkout per job that runs at the same time**.

## Things to set per job

- **GPUs:** the last `train.sh` argument is the GPU list; it becomes `CUDA_VISIBLE_DEVICES` and
  the GPU count. Under SLURM, pass the IDs the job sees (usually `0,1,...,7`).
- **Port:** `MASTER_PORT` defaults to 29501. Set a different one per job if jobs can share a node.
- **CPUs:** the dataset start-up uses a pool of 128 processes, plus 16 dataloader workers.
  Ask for plenty of CPUs (32 or more), or startup will be slow.
- **wandb:** EEF config has it off. The joint config (`robotwin30_train`) has it on, and
  `run_va_posttrain.sh` has placeholder keys, so it crashes. Set `enable_wandb = False` in
  `configs/va_robotwin30_train_cfg.py`, or put real keys in `run_va_posttrain.sh`.
- **Config:** `train.sh` defaults to `robotwin30_train` (joint). For EEF set
  `LINGBOT_VA_CONFIG_NAME=rmbench_eef_train`.

## SLURM example (EEF, all tasks)

```bash
#!/bin/bash
#SBATCH --job-name=lingbot-ee
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lingbot_va
export HF_HUB_OFFLINE=1
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

cd /path/to/RMBench/XPolicyLab/policy/LingBot_VA
ln -sfn /path/to/lingbot-va-posttrain-robotwin .merged_ckpt
LINGBOT_VA_CONFIG_NAME=rmbench_eef_train \
LINGBOT_VA_DATASET_PATH=/scratch/latents_ee \
bash train.sh RMBench all_tasks aloha_agilex ee 0 0,1,2,3,4,5,6,7
```

Partition names, memory and time limits depend on your cluster.

## Checklist per run

1. `apply_patches.sh` printed "patched" / "already patched" for every step.
2. `.merged_ckpt` points at the right starting weights.
3. `LINGBOT_VA_DATASET_PATH` parent has `empty_emb.pt` and only one path type.
4. `checkpoints/` is on scratch (or `save_interval` raised).
5. Config name matches the path (`rmbench_eef_train` for `-ee`).
6. Unique `MASTER_PORT` if nodes are shared.

## Not tested yet

- The backward pass and a full FSDP training run (only the forward pass ran, on a Colab L4).
- EEF eval in the simulator (the patched `model.py` ee mode ran open-loop only).
- A multi-task training run (the multi-task loader was checked on the data, not in training).
- Joint path on observe_and_pickup clips 5.2% of steps with the robotwin30 norm stats; use EEF
  for it or give it its own norm stats.
