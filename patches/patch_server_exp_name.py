#!/usr/bin/env python3
"""Shorten the eval server's per-episode output folder name.

wan_va_server._reset names a folder after the full task instruction:
    self.exp_name = f"{prompt}_{timestamp}"
RMBench instructions can be ~300 characters (press_button), which is over the
255-character file name limit, so the eval server crashes with
"OSError: File name too long" on the first episode. Instructions can also
contain "/" or other characters that are not safe in a folder name.

After patching: first 60 characters of the instruction, anything other than
letters, digits, "-" and "_" replaced by "_", plus the timestamp.

Usage (from the RMBench root):
  python patch_server_exp_name.py XPolicyLab/policy/LingBot_VA/lingbot_va/wan_va/wan_va_server.py
Safe to re-run.
"""
import sys

p = sys.argv[1]
s = open(p).read()
MARK = "# memorywam: short, safe exp_name"
if MARK in s:
    print("already patched")
    sys.exit(0)

old = "        self.exp_name = f\"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}\" if prompt else \"default\"\n"
if s.count(old) != 1:
    sys.exit(f"ERROR: exp_name line not found exactly once in {p} (XPolicyLab changed?)")
new = (
    f"        {MARK} (the full instruction can exceed the 255-char file name limit)\n"
    "        _safe = ''.join(c if (c.isalnum() or c in '-_') else '_' for c in (prompt or ''))[:60]\n"
    "        self.exp_name = f\"{_safe}_{time.strftime('%Y%m%d_%H%M%S')}\" if prompt else \"default\"\n"
)
open(p, "w").write(s.replace(old, new))
print("patched", p)
