#!/usr/bin/env python3
"""Build an EEF copy of the joint-space RMBench LeRobot v2.1 dataset.

The XPolicyLab converter writes 14-dim joint action/state. LingBot-VA's
RoboTwin weights (posttrain-robotwin, config 'robotwin', env_type
'robotwin_tshape') expect 16-dim absolute EEF instead:
    [left xyz + quat (7), left gripper, right xyz + quat (7), right gripper]
The trainer itself turns these into poses relative to the first frame.

This script copies the joint dataset (videos unchanged) and rewrites the
parquet action / observation.state columns from the HDF5 ee_poses. Each
episode is matched to its HDF5 file by checking that the parquet joint
action equals the HDF5 joint action, so a wrong file order cannot slip
through silently.

Quaternions are copied as stored (w, x, y, z from RoboTwin). Do not reorder:
LingBot reads them the same way in training and eval, and so did the
released weights.

Usage (laptop, rmb venv):
  python make_eef_lerobot.py \
    --src ~/RMBench/lerobot_data/rmbench_press_button \
    --hdf5-dir ~/RMBench/data/demo_clean/press_button/aloha_agilex/data \
    --dst ~/RMBench/lerobot_data/rmbench_press_button_eef
"""
import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

JOINT_KEYS = ["left_arm_joint_states", "left_ee_joint_states",
              "right_arm_joint_states", "right_ee_joint_states"]
EEF_KEYS = ["left_ee_poses", "left_ee_joint_states",
            "right_ee_poses", "right_ee_joint_states"]
EEF_NAMES = ["left_x", "left_y", "left_z", "left_q0", "left_q1", "left_q2", "left_q3",
             "left_gripper",
             "right_x", "right_y", "right_z", "right_q0", "right_q1", "right_q2", "right_q3",
             "right_gripper"]


def read_group(h, group, keys):
    parts = []
    for k in keys:
        a = h[f"{group}/{k}"][()].astype(np.float32)
        parts.append(a.reshape(a.shape[0], -1))
    return np.concatenate(parts, axis=1)


def stats(arr):
    return {"min": arr.min(0).tolist(), "max": arr.max(0).tolist(),
            "mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(),
            "count": [int(arr.shape[0])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--hdf5-dir", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    src, dst = Path(args.src).expanduser(), Path(args.dst).expanduser()
    h5_files = sorted(Path(args.hdf5_dir).expanduser().rglob("*.hdf5"))  # same order as converter
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=False)

    info = json.loads((dst / "meta/info.json").read_text())
    chunks_size = int(info.get("chunks_size", 1000))
    episodes = [json.loads(l) for l in (dst / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    ep_stats = {json.loads(l)["episode_index"]: json.loads(l)
                for l in (dst / "meta/episodes_stats.jsonl").read_text().splitlines() if l.strip()}
    print(f"episodes {len(episodes)}, hdf5 files {len(h5_files)}")

    # cache joint actions for fallback matching
    h5_joint = {}
    def joint_of(f):
        if f not in h5_joint:
            with h5py.File(f) as h:
                h5_joint[f] = read_group(h, "action", JOINT_KEYS)
        return h5_joint[f]

    used, mapping = set(), {}
    for ep in episodes:
        i = ep["episode_index"]
        pq = dst / "data" / f"chunk-{i // chunks_size:03d}" / f"episode_{i:06d}.parquet"
        df = pd.read_parquet(pq)
        pq_joint = np.stack(df["action"].to_numpy()).astype(np.float32)

        cands = ([h5_files[i]] if i < len(h5_files) else []) + h5_files
        match = None
        for f in cands:
            if f in used:
                continue
            j = joint_of(f)
            if j.shape == pq_joint.shape and np.allclose(j, pq_joint, atol=1e-5):
                match = f
                break
        if match is None:
            raise RuntimeError(f"episode {i}: no HDF5 file matches its joint actions")
        if match != (h5_files[i] if i < len(h5_files) else None):
            print(f"  note: episode {i} matched {match.name}, not sorted position {i}")
        used.add(match)
        mapping[i] = match.name

        with h5py.File(match) as h:
            act = read_group(h, "action", EEF_KEYS)
            st = read_group(h, "state", EEF_KEYS)
        assert act.shape == (len(df), 16) and st.shape == (len(df), 16), (act.shape, st.shape, len(df))
        df["action"] = list(act)
        df["observation.state"] = list(st)
        df.to_parquet(pq, index=False)

        s = ep_stats.get(i, {"episode_index": i, "stats": {}})
        s["stats"]["action"] = stats(act)
        s["stats"]["observation.state"] = stats(st)
        ep_stats[i] = s

    for key in ("action", "observation.state"):
        info["features"][key]["shape"] = [16]
        info["features"][key]["names"] = [list(EEF_NAMES)]
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))
    (dst / "meta/episodes_stats.jsonl").write_text(
        "\n".join(json.dumps(ep_stats[k]) for k in sorted(ep_stats)) + "\n")
    (dst / "eef_episode_to_hdf5.json").write_text(json.dumps(mapping, indent=1))

    print(f"done: {len(mapping)} episodes -> {dst}")
    print("episode -> hdf5 (first 5):", list(mapping.items())[:5])
    a = act
    print("last episode action[0]:", np.round(a[0], 3).tolist())


if __name__ == "__main__":
    main()
