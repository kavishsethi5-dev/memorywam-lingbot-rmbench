#!/usr/bin/env python3
"""Make the KV-cache window (the memory dial) settable with an env var.

attn_window is only used by the eval server (wan_va_server -> create_empty_cache:
attn_window // 2 latent chunks + attn_window // 2 action chunks of history).
Training ignores it (train.py randomises its own window). So this only changes eval.

After patching:
  LINGBOT_VA_ATTN_WINDOW=8 bash eval.sh ...    # 4 chunks of history
  (unset = config default, 72 = 36 chunks)
Chunk length in env steps: joint path 24 (stride 3), EEF path 32 (stride 4).

Patches both configs (joint robotwin30 and EEF rmbench_eef). Safe to re-run.
Usage (from ~/RMBench):
  python patch_attn_window.py XPolicyLab/policy/LingBot_VA/lingbot_va/wan_va/configs
"""
import sys
from pathlib import Path

d = Path(sys.argv[1])
SNIPPET = '''
_attn_override = os.environ.get("LINGBOT_VA_ATTN_WINDOW")
if _attn_override:
    _w = int(_attn_override)
    assert _w >= 2 and _w % 2 == 0, f"LINGBOT_VA_ATTN_WINDOW must be even and >= 2, got {_w}"
    {cfg}.attn_window = _w
    print(f"[config] attn_window overridden to {_w} ({_w // 2} chunks of history)", flush=True)
'''

targets = [
    ("va_robotwin30_train_cfg.py", "va_robotwin30_cfg.attn_window = 72\n", "va_robotwin30_cfg"),
    ("va_rmbench_eef_cfg.py", "va_rmbench_eef_cfg.enable_offload = True\n", "va_rmbench_eef_cfg"),
]
for name, anchor, cfg in targets:
    p = d / name
    s = p.read_text()
    if "LINGBOT_VA_ATTN_WINDOW" in s:
        print(f"{name}: already patched")
        continue
    assert s.count(anchor) == 1, f"{name}: anchor not found once"
    assert "import os" in s, f"{name}: needs 'import os'"
    s = s.replace(anchor, anchor + SNIPPET.replace("{cfg}", cfg))
    p.write_text(s)
    print(f"{name}: patched")
