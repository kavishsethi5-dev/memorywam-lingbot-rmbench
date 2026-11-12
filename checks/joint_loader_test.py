# Loader smoke test for the joint latents (CPU). Set D to the unzipped latents folder.
import os, sys, numpy as np
D = os.path.expanduser(os.environ.get("D", "~/RMBench/latents/RMBench-press_button-aloha_agilex-joint"))
os.environ["LINGBOT_VA_DATASET_PATH"] = D
sys.path.insert(0, os.path.expanduser(os.environ.get(
    "WAN_VA", "~/RMBench/XPolicyLab/policy/LingBot_VA/lingbot_va/wan_va")))
from configs import VA_CONFIGS
from dataset.lerobot_latent_dataset import LatentLeRobotDataset
cfg = VA_CONFIGS["robotwin30_train"]
ds = LatentLeRobotDataset(repo_id=D, config=cfg)
x = ds[0]
print("segments", len(ds), "| latents", tuple(x["latents"].shape), "| actions", tuple(x["actions"].shape),
      "| per frame", x["actions"].shape[2], "config", cfg.action_per_frame)
