# EEF norm check for LingBot-VA RoboTwin config on RMBench HDF5.
# Run on laptop: cd ~/RMBench && source ~/rmb/bin/activate && python eef_norm_check.py
import glob, h5py, numpy as np
from scipy.spatial.transform import Rotation as R

def rel(p):  # same as lingbot get_relative_pose
    rot = R.from_quat(p[:, 3:7]); r0 = R.from_quat(np.tile(p[:1, 3:7], (len(p), 1)))
    return np.concatenate([p[:, :3] - p[:1, :3], (r0.inv() * rot).as_quat()], 1)

q01 = np.array([-0.0617, -3.7e-05, -0.0878, -1, -1, -1, -1,
                -0.3547, -1.3e-06, -0.1198, -1, -1, -1, -1])
q99 = np.array([0.3463, 0.3997, 0.1475, 1, 1, 1, 1,
                0.0342, 0.3914, 0.1792, 1, 1, 1, 1])
fs = sorted(glob.glob("data/demo_clean/press_button/aloha_agilex/data/*.hdf5"))
rows = []
for f in fs:
    with h5py.File(f) as h:
        L = h["action/left_ee_poses"][()].astype(float)
        Rr = h["action/right_ee_poses"][()].astype(float)
    rows.append(np.concatenate([rel(L), rel(Rr)], 1))
A = np.concatenate(rows)
n = (A - q01) / (q99 - q01 + 1e-6) * 2 - 1
names = [f"L{c}" for c in "xyz"] + ["Lq0", "Lq1", "Lq2", "Lq3"] + \
        [f"R{c}" for c in "xyz"] + ["Rq0", "Rq1", "Rq2", "Rq3"]
print("episodes", len(fs), "steps", len(A))
for i, s in enumerate(names):
    print(f"{s:4s} data {A[:,i].min():7.3f}..{A[:,i].max():7.3f}"
          f"  norm {n[:,i].min():6.2f}..{n[:,i].max():6.2f}"
          f"  clipped {100*np.mean(np.abs(n[:,i]) > 1.5):5.1f}%")
