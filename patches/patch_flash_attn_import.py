#!/usr/bin/env python3
"""Make the flash-attn import in LingBot's model.py optional.

Without flash-attn installed, `import modules` crashes (model.py imports
flash_attn_func at the top with no fallback), even for code that never uses it
(e.g. process_data.py). Only needed on machines without flash-attn.
Usage (from the RMBench root):
  python patch_flash_attn_import.py XPolicyLab/policy/LingBot_VA/lingbot_va/wan_va/modules/model.py
"""
import re, sys
p = sys.argv[1]
s = open(p).read()
if "    try:\n        from flash_attn import flash_attn_func" in s:
    print("already patched"); sys.exit(0)
m = re.search(r"^( *)from flash_attn import flash_attn_func *$", s, re.M)
ind = m.group(1)
new = (f"{ind}try:\n{ind}    from flash_attn import flash_attn_func\n"
       f"{ind}except Exception:\n{ind}    flash_attn_func = None")
open(p, "w").write(s[:m.start()] + new + s[m.end():])
print("patched", p)
