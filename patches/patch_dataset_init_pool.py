#!/usr/bin/env python3
"""Build the per-task datasets in the training process instead of a multiprocessing Pool.

construct_lerobot_multi_processor builds one LatentLeRobotDataset per task folder with
    with Pool(num_init_worker) as pool: pool.map(...)
(num_init_worker=128 from MultiLatentLeRobotDataset). train.py calls it after the model,
FSDP and CUDA are set up, i.e. it forks a large, multi-threaded process.

Seen on Sharanga (15 Nov, all 10 RMBench EEF tasks, 4 workers): the pool hung for good
after every task's parquet was read; one worker had been replaced (died), the rest idle.
The same load in one process (no pool) passed every check. A pool started from a fresh
process also passed, so the hang depends on the state of the process that forks.

After patching: tasks are built one after another in the calling process (default).
LINGBOT_VA_INIT_POOL=1 restores the original Pool.

Usage (from the RMBench root):
  python patch_dataset_init_pool.py XPolicyLab/policy/LingBot_VA/lingbot_va/wan_va/dataset/lerobot_latent_dataset.py
Safe to re-run.
"""
import sys

p = sys.argv[1]
s = open(p).read()
MARK = "# memorywam: no Pool by default"
if MARK in s:
    print("already patched")
    sys.exit(0)

old = (
    "    with Pool(num_init_worker) as pool:\n"
    "        datasets_out_lst = pool.map(construct_func, repo_list)\n"
)
if s.count(old) != 1:
    sys.exit(f"ERROR: Pool lines not found exactly once in {p} (XPolicyLab changed?)")
new = (
    f"    {MARK} (forking the training process can hang); LINGBOT_VA_INIT_POOL=1 restores it\n"
    "    if os.environ.get(\"LINGBOT_VA_INIT_POOL\", \"0\") == \"1\" and num_init_worker > 1:\n"
    "        with Pool(num_init_worker) as pool:\n"
    "            datasets_out_lst = pool.map(construct_func, repo_list)\n"
    "    else:\n"
    "        datasets_out_lst = [construct_func(r) for r in repo_list]\n"
)
open(p, "w").write(s.replace(old, new))
print("patched", p)
