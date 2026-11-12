#!/usr/bin/env python3
"""Offline check of the EEF eval path (no GPU, no simulator, no server).

What it does: takes episode 0, builds exactly what wan_va_server would send back
if the model predicted the training targets perfectly (dataset actions ->
server postprocess_action: denormalise + select used channels), runs that through
the PATCHED model.py code (_format_action_chunk, _convert_to_ee_control_chunk,
encode_obs), and compares with the absolute EEF actions stored in the HDF5.
Also checks the memory-window env var.

If everything matches, the remaining unknowns are only the live parts (server,
websocket, simulator), not the maths or the channel order.

Run on the laptop from ~/RMBench (rmb venv), after apply_patches.sh:
  python verify_eef_eval.py
"""
import os, sys, types, importlib, subprocess
from pathlib import Path
import numpy as np

RMB = Path(os.environ.get("RMB", "~/RMBench")).expanduser()
LAT = Path(os.environ.get("EEF_LATENTS", "~/RMBench/latents/RMBench-press_button-aloha_agilex-ee")).expanduser()
H5 = RMB / "data/demo_clean/press_button/aloha_agilex/data/episode_0000000.hdf5"
WAN_VA = RMB / "XPolicyLab/policy/LingBot_VA/lingbot_va/wan_va"
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


# ---- 0. patches present ---------------------------------------------------------
mp = (RMB / "XPolicyLab/policy/LingBot_VA/model.py").read_text()
check("model.py has EEF mode", "EE_CONTROL_INDICES" in mp and "(T, 16) or (T, 30)" in mp,
      "(re-run apply_patches.sh from the latest bundle if this fails)")
cfgs = (WAN_VA / "configs/__init__.py").read_text()
check("rmbench_eef registered", "rmbench_eef_train" in cfgs)

# ---- 1. memory window env var -----------------------------------------------------
code = "import sys; sys.path.insert(0, %r); from configs import VA_CONFIGS as V; print(V['rmbench_eef'].attn_window, V['robotwin30_train'].attn_window)" % str(WAN_VA)
out_default = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.split()[-2:]
out_8 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, "LINGBOT_VA_ATTN_WINDOW": "8"}).stdout.split()[-2:]
check("attn_window default", out_default == ["72", "72"], f"(got {out_default})")
check("LINGBOT_VA_ATTN_WINDOW=8", out_8 == ["8", "8"], f"(got {out_8})")

# ---- 2. import the patched adapter (websocket client stubbed: no server needed) ---
# wan_va/__init__ imports the whole model stack (diffusers, flash-attn, ...), which is not
# needed here. Pre-register light stand-ins so only configs + model.py are imported.
sys.path.insert(0, str(RMB))
sys.path.insert(0, str(WAN_VA))
import configs as _configs                      # wan_va/configs, imported directly
from configs import VA_CONFIGS
P = "XPolicyLab.policy.LingBot_VA.lingbot_va"
for name, path in [(P, WAN_VA.parent), (P + ".wan_va", WAN_VA),
                   (P + ".evaluation", WAN_VA.parent / "evaluation"),
                   (P + ".evaluation.robotwin", WAN_VA.parent / "evaluation/robotwin")]:
    mod = types.ModuleType(name); mod.__path__ = [str(path)]; sys.modules[name] = mod
sys.modules[P + ".wan_va.configs"] = _configs
ws = types.ModuleType(P + ".evaluation.robotwin.websocket_client_policy")
ws.WebsocketClientPolicy = object
sys.modules[ws.__name__] = ws
M = importlib.import_module("XPolicyLab.policy.LingBot_VA.model")
cfg = VA_CONFIGS["rmbench_eef"]

m = object.__new__(M.Model)          # skip __init__ (it connects to the server)
m.action_type, m.ee_mode, m.action_dim, m._ee_init_pose = "ee", True, int(cfg.action_dim), None

# ---- 3. what the server would send for the training targets of episode 0 --------
os.environ["LINGBOT_VA_DATASET_PATH"] = str(LAT)
from dataset.lerobot_latent_dataset import LatentLeRobotDataset
ds = LatentLeRobotDataset(repo_id=str(LAT), config=VA_CONFIGS["rmbench_eef_train"])
norm = ds[0]["actions"][..., 0].numpy().astype(np.float64)        # (30, F, 16), normalised
q01 = np.array(cfg.norm_stat["q01"])[:, None, None]
q99 = np.array(cfg.norm_stat["q99"])[:, None, None]
server_out = ((norm + 1) / 2 * (q99 - q01 + 1e-6) + q01)[cfg.used_action_channel_ids]   # postprocess_action
check("server output shape", server_out.shape[0] == 16, f"{server_out.shape} (16 channels, F, 16 per frame)")

chunk = m._format_action_chunk(server_out)                         # (F*16, 16)
check("_format_action_chunk", chunk.shape[1] == 16, f"{chunk.shape}")

# ---- 4. reference pose + conversion, against the HDF5 ----------------------------
import h5py
with h5py.File(H5) as h:
    def g(grp):
        return np.concatenate([h[f"{grp}/left_ee_poses"][()], h[f"{grp}/left_ee_joint_states"][()].reshape(-1, 1),
                               h[f"{grp}/right_ee_poses"][()], h[f"{grp}/right_ee_joint_states"][()].reshape(-1, 1)], 1).astype(np.float64)
    act, st = g("action"), g("state")

# encode_obs in ee mode: state vector must be [L7, Lg, R7, Rg] from obs['state']
img = np.zeros((240, 320, 3), np.uint8)
obs = {"vision": {"cam_high": img, "cam_left_wrist": img, "cam_right_wrist": img},
       "state": {"left_ee_pose": st[0, :7], "left_ee_joint_state": st[0, 7:8],
                 "right_ee_pose": st[0, 8:15], "right_ee_joint_state": st[0, 15:16]},
       "instruction": "x"}
enc = M.encode_obs(obs, "ee", None, "x")
check("encode_obs ee state", np.allclose(enc["observation.state"], st[0], atol=1e-6), f"{enc['observation.state'].shape}")

# training made poses relative to act[0]; give the model the same reference
m._ee_init_pose = act[0]
absolute = m._convert_to_ee_control_chunk(chunk)
skip = int(cfg.action_per_frame)          # first latent frame = zero padding; model skips it (initial_action_skip)
pred = absolute[skip:]
true = act[:len(pred)]
pos_err = np.abs(pred[:, [0, 1, 2, 8, 9, 10]] - true[:, [0, 1, 2, 8, 9, 10]]).max()
def qerr(a, b):
    return np.minimum(np.abs(a - b).max(1), np.abs(a + b).max(1)).max()
q_err = max(qerr(pred[:, 3:7], true[:, 3:7]), qerr(pred[:, 11:15], true[:, 11:15]))
g_err = np.abs(pred[:, [7, 15]] - true[:, [7, 15]]).max()
check("absolute positions match HDF5", pos_err < 1e-4, f"max err {pos_err:.2e} m over {len(pred)} steps")
check("quaternions match HDF5 (up to sign)", q_err < 1e-4, f"max err {q_err:.2e}")
check("grippers match HDF5", g_err < 1e-4, f"max err {g_err:.2e}")
check("unit quaternions", np.allclose(np.linalg.norm(pred[:, 3:7], axis=1), 1, atol=1e-5))

# same thing if a 30-channel chunk arrives
full = (norm + 1) / 2 * (q99 - q01 + 1e-6) + q01
abs30 = m._convert_to_ee_control_chunk(np.transpose(full, (1, 2, 0)).reshape(-1, 30))
check("30-channel input gives same result", np.allclose(abs30, absolute, atol=1e-6))

# the eval loop's state at step 0 vs the training reference (one step apart)
print(f"[info] state[0] vs action[0] position gap: {np.abs(st[0, :3] - act[0, :3]).max():.2e} m "
      "(eval uses the pose at reset, training used the next step; expected tiny)")
print("\nALL PASS" if ok_all else "\nSOME CHECKS FAILED")
