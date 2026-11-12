# Loader smoke test for the EEF latents (laptop, CPU), same idea as the joint one.
# Needs: config va_rmbench_eef_cfg.py registered in configs/__init__.py,
# latents zip unzipped to ~/RMBench/latents/RMBench-press_button-aloha_agilex-ee
import os, sys
import numpy as np

D = os.path.expanduser("~/RMBench/latents/RMBench-press_button-aloha_agilex-ee")
os.environ["LINGBOT_VA_DATASET_PATH"] = D
sys.path.insert(0, os.path.expanduser(
    "~/RMBench/XPolicyLab/policy/LingBot_VA/lingbot_va/wan_va"))
from configs import VA_CONFIGS
from dataset.lerobot_latent_dataset import LatentLeRobotDataset

cfg = VA_CONFIGS["rmbench_eef_train"]
print("env_type", cfg.env_type, "action_per_frame", cfg.action_per_frame)
ds = LatentLeRobotDataset(repo_id=D, config=cfg)
print("segments", len(ds), "(expect 50)")
x = ds[0]
lat, act, msk = x["latents"], x["actions"], x["actions_mask"]
print("latents", tuple(lat.shape), "(expect 48, 36, 24, 20 for ep 0: wrists 8x20 stacked on head 16x20)")
print("actions", tuple(act.shape), "(expect 30, 36, 16, 1)")
print("actions per latent frame", act.shape[2], "config", cfg.action_per_frame,
      "MATCH" if act.shape[2] == cfg.action_per_frame else "MISMATCH")
a = act[..., 0].reshape(30, -1).numpy()
m = msk[..., 0].reshape(30, -1).numpy()
used = list(range(0, 14)) + [28, 29]
names = ["Lx", "Ly", "Lz", "Lq0", "Lq1", "Lq2", "Lq3",
         "Rx", "Ry", "Rz", "Rq0", "Rq1", "Rq2", "Rq3", "Lg", "Rg"]
for c, s in zip(used, names):
    at_clip = 100 * np.mean(np.abs(a[c]) >= 1.4999)
    print(f"ch{c:2d} {s:3s} norm {a[c].min():6.2f}..{a[c].max():6.2f}  at clip {at_clip:4.1f}%")
print("finite", bool(np.isfinite(a).all()),
      "| mask on used channels all", bool(m[used].all()),
      "| mask on unused channels none", bool(~m[[c for c in range(30) if c not in used]].any()))
