#!/usr/bin/env bash
# Apply the memorywam patches to an RMBench checkout (XPolicyLab submodule at fa431ec).
# Usage: bash apply_patches.sh /path/to/RMBench
# Safe to re-run: every step checks whether it is already done.
set -euo pipefail
RMB="$(cd "${1:?usage: bash apply_patches.sh /path/to/RMBench}" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LVA="${RMB}/XPolicyLab/policy/LingBot_VA"
CFG="${LVA}/lingbot_va/wan_va/configs"

echo "XPolicyLab commit: $(git -C "${RMB}/XPolicyLab" rev-parse --short HEAD) (tested on fa431ec)"

# 1. EEF config + registration
cp "${HERE}/patches/va_rmbench_eef_cfg.py" "${CFG}/"
python3 - "${CFG}/__init__.py" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
if "rmbench_eef" in s:
    print("configs/__init__.py: already registered")
else:
    s = s.replace("VA_CONFIGS = {",
        "from .va_rmbench_eef_cfg import va_rmbench_eef_cfg, va_rmbench_eef_train_cfg\n\nVA_CONFIGS = {\n"
        "    'rmbench_eef': va_rmbench_eef_cfg,\n"
        "    'rmbench_eef_train': va_rmbench_eef_train_cfg,", 1)
    open(p, "w").write(s); print("configs/__init__.py: registered rmbench_eef, rmbench_eef_train")
EOF

# 2. memory dial: LINGBOT_VA_ATTN_WINDOW env override (eval only)
python3 "${HERE}/patches/patch_attn_window.py" "${CFG}"

# 3. EEF control mode in the XPolicyLab adapter (action_type ee)
python3 "${HERE}/patches/patch_model_ee.py" "${LVA}/model.py"

# 4. optional flash-attn import (harmless if flash-attn is installed)
python3 "${HERE}/patches/patch_flash_attn_import.py" "${LVA}/lingbot_va/wan_va/modules/model.py"

# 5. EEF data scripts next to the originals
cp "${HERE}/patches/process_data_eef.py" "${LVA}/"
cp "${HERE}/patches/make_eef_lerobot.py" "${RMB}/"

# 6. short eval output folder names (long instructions -> "File name too long" in the eval server)
python3 "${HERE}/patches/patch_server_exp_name.py" "${LVA}/lingbot_va/wan_va/wan_va_server.py"

echo "done. git -C ${RMB}/XPolicyLab status --short:"
git -C "${RMB}/XPolicyLab" status --short
