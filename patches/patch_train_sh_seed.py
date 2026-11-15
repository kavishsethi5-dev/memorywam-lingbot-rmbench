#!/usr/bin/env python3
"""Remove --seed from the train.sh -> run_va_posttrain.sh call.

XPolicyLab fa431ec train.sh passes `--seed "${seed}"`, but wan_va/train.py only accepts
--config-name and --save-root, so every rank exits at once:
    train.py: error: unrecognized arguments: --seed 0
train.py fixes its own seed (42). The seed argument of train.sh still sets PYTHONHASHSEED
and the checkpoint folder name.

Usage (from the RMBench root):
  python patch_train_sh_seed.py XPolicyLab/policy/LingBot_VA/train.sh
Safe to re-run.
"""
import sys

p = sys.argv[1]
s = open(p).read()
MARK = "# memorywam: no --seed"
if MARK in s or '--seed "${seed}"' not in s:
    print("already patched")
    sys.exit(0)

old = '  --save-root "${ckpt_dir}" \\\n  --seed "${seed}"\n'
if s.count(old) != 1:
    sys.exit(f"ERROR: --save-root/--seed lines not found exactly once in {p} (XPolicyLab changed?)")
new = f'  --save-root "${{ckpt_dir}}"  {MARK} (wan_va/train.py rejects it)\n'
open(p, "w").write(s.replace(old, new))
print("patched", p)
