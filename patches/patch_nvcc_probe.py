#!/usr/bin/env python3
"""Stop a broken `nvcc` on PATH from crashing training (and eval) at the first torch.compile.

torch 2.9 runs `nvcc --version` on every inductor compile, only to write a comment into a
debug repro string (torch/_dynamo/debug_utils.py::_cuda_system_info_comment, called from
torch/_inductor/compile_fx.py via torch/_dynamo/repro/after_aot.py). It catches
FileNotFoundError and CalledProcessError only. If an `nvcc` on PATH can't be executed,
Python raises PermissionError and the first training step dies:
    [Errno 13] Permission denied: 'nvcc'
(seen on BITS Sharanga, inside FlexAttnFunc.init_mask). nvcc is not needed for training.

This patch adds a guard to wan_va/modules/model.py (imported by both train.py and the eval
server) that turns any OSError from that probe into "# nvcc not usable". Nothing else changes.

Usage (from the RMBench root):
  python patch_nvcc_probe.py XPolicyLab/policy/LingBot_VA/lingbot_va/wan_va/modules/model.py
Safe to re-run.
"""
import sys

p = sys.argv[1]
s = open(p).read()
MARK = "# memorywam: safe nvcc probe"
if MARK in s:
    print("already patched")
    sys.exit(0)

anchor = "__all__ = ['WanTransformer3DModel']\n"
if s.count(anchor) != 1:
    sys.exit(f"ERROR: {anchor.strip()!r} not found exactly once in {p} (XPolicyLab changed?)")

guard = f'''
{MARK}: torch 2.9 runs `nvcc --version` on every compile (debug text only) and crashes on
# PermissionError if a broken nvcc is on PATH. See memorywam-lingbot-rmbench patches/patch_nvcc_probe.py.
def _memorywam_safe_nvcc_probe():
    try:
        import torch._dynamo.debug_utils as _du
        import torch._dynamo.repro.after_aot as _aa
    except Exception:
        return
    _orig = getattr(_du, "_cuda_system_info_comment", None)
    if _orig is None or getattr(_orig, "_memorywam_safe", False):
        return

    def _safe():
        try:
            return _orig()
        except OSError:
            return "# nvcc not usable (memorywam guard)\\n"

    _safe._memorywam_safe = True
    _du._cuda_system_info_comment = _safe
    if hasattr(_aa, "_cuda_system_info_comment"):
        _aa._cuda_system_info_comment = _safe


_memorywam_safe_nvcc_probe()
'''
open(p, "w").write(s.replace(anchor, anchor + guard, 1))
print("patched", p)
