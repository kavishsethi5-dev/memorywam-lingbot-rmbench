#!/usr/bin/env python3
"""Put wan_va's own folder FIRST on sys.path in train.py and wan_va_server.py.

Both files do `sys.path.append(<wan_va dir>)` and then `from distributed.fsdp import ...`,
`from configs import ...`, `from modules... / utils...`. append puts the folder LAST, so any
installed package with the same name wins. Colab ships Dask's `distributed`, which gives:
    ModuleNotFoundError: No module named 'distributed.fsdp'
insert(0, ...) makes wan_va's packages win, which is what the code means.

Usage: python patch_sys_path_first.py <wan_va dir>   (patches train.py and wan_va_server.py)
Safe to re-run.
"""
import os
import sys

d = sys.argv[1]
OLD = "sys.path.append(os.path.dirname(os.path.abspath(__file__)))\n"
NEW = "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # memorywam: wan_va first\n"
for name in ("train.py", "wan_va_server.py"):
    p = os.path.join(d, name)
    s = open(p).read()
    if NEW in s:
        print(f"{name}: already patched")
        continue
    if s.count(OLD) != 1:
        sys.exit(f"ERROR: sys.path.append line not found exactly once in {p} (XPolicyLab changed?)")
    open(p, "w").write(s.replace(OLD, NEW))
    print(f"{name}: patched")
