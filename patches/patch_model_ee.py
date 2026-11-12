#!/usr/bin/env python3
"""Add an EEF mode (action_type: ee) to XPolicyLab's LingBot_VA model.py.

Why: XPolicyLab's adapter only does joint control. The RoboTwin weights
(config rmbench_eef) output 16 EEF channels [0-6, 28, 7-13, 29] as poses
RELATIVE to the episode's first pose. This patch makes the adapter:
  1. remember the absolute EEF pose [L7, Lg, R7, Rg] whenever the server is reset
     (episode start; or every chunk in receding_horizon mode),
  2. turn the predicted relative poses back into absolute ones exactly like the
     upstream RoboTwin client (add_eef_pose: trans + init, R(init) * R(rel), same
     scipy quaternion reading as training), normalise the quats,
  3. return a flat (T, 16) chunk, which RMBench's eval passes straight to
     take_action(..., action_type='ee').
Joint mode is unchanged.

Usage (laptop, from ~/RMBench):
  python patch_model_ee.py XPolicyLab/policy/LingBot_VA/model.py
Re-running is safe (it checks first).
"""
import sys

p = sys.argv[1]
s = open(p).read()
if "EE_CONTROL_INDICES" in s:
    if "(T, 16) or (T, 30)" in s:
        print("already patched")
        sys.exit(0)
    # upgrade the first version (expected 30 channels; the server sends 16)
    old = '''        action_chunk = np.asarray(action_chunk, dtype=np.float64)
        if action_chunk.ndim != 2 or action_chunk.shape[1] != 30:
            raise ValueError(f"EEF conversion expects (T, 30), got {action_chunk.shape}")
        if self._ee_init_pose is None:
            raise RuntimeError("EEF mode: no reference pose stored (server reset not seen)")
        rel = action_chunk[:, EE_CONTROL_INDICES]'''
    new = '''        # wan_va_server.postprocess_action already returns only used_action_channel_ids,
        # so with the robotwin config the server sends 16 channels [L7, Lg, R7, Rg].
        # Accept 30 too (full layout) and select.
        action_chunk = np.asarray(action_chunk, dtype=np.float64)
        if action_chunk.ndim != 2 or action_chunk.shape[1] not in (16, 30):
            raise ValueError(f"EEF conversion expects (T, 16) or (T, 30), got {action_chunk.shape}")
        if self._ee_init_pose is None:
            raise RuntimeError("EEF mode: no reference pose stored (server reset not seen)")
        rel = action_chunk if action_chunk.shape[1] == 16 else action_chunk[:, EE_CONTROL_INDICES]'''
    assert s.count(old) == 1, "unexpected model.py state; restore it with git checkout and re-run"
    open(p, "w").write(s.replace(old, new))
    print("upgraded", p, "(now accepts the 16-channel server output)")
    sys.exit(0)


def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, f"expected 1 match, found {n}: {old[:70]!r}"
    s = s.replace(old, new)


# 1. imports + channel map
rep('''import numpy as np
''', '''import numpy as np
from scipy.spatial.transform import Rotation as _R
''')
rep('''def extract_image(observation, candidate_names):''',
'''# 30-dim LingBot layout -> 16-dim EEF [L xyz+quat, L grip, R xyz+quat, R grip]
# (va_robotwin_cfg.used_action_channel_ids). Poses are relative to the first frame.
EE_CONTROL_INDICES = np.array(list(range(0, 7)) + [28] + list(range(7, 14)) + [29], dtype=np.int64)
_EE_STATE_KEYS = ("left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state")


def _ee_state_from_obs(observation):
    st = observation.get("state", {})
    missing = [k for k in _EE_STATE_KEYS if k not in st]
    if missing:
        raise KeyError(f"EEF mode needs obs['state'] keys {missing} (RMBench convert_state sets them from endpose)")
    return np.concatenate([np.asarray(st[k], dtype=np.float64).reshape(-1) for k in _EE_STATE_KEYS])


def _add_eef_pose(new_pose, init_pose):
    # Same as upstream lingbot_va/evaluation/robotwin/eval_polict_client_openpi.py::add_eef_pose
    out_rot = (_R.from_quat(init_pose[3:7][None]) * _R.from_quat(new_pose[3:7][None])).as_quat().reshape(-1)
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]])


def extract_image(observation, candidate_names):''')

# 2. encode_obs: in ee mode take the state directly (pack_robot_state uses arm_dim 6 and would fail)
rep('''    if robot_action_dim_info is None:
        state = np.zeros((1,), dtype=np.float32)''',
'''    if action_type == "ee":
        state = _ee_state_from_obs(observation).astype(np.float32)
    elif robot_action_dim_info is None:
        state = np.zeros((1,), dtype=np.float32)''')

# 3. state for the reference pose
rep('''        self._first_observation: dict[str, Any] | None = None
        self._latest_raw_action_chunk: np.ndarray | None = None
        self._exec_obs_buffer: list[dict[str, Any]] = []
        self._last_exec_step_count = 0''',
'''        self._first_observation: dict[str, Any] | None = None
        self._latest_raw_action_chunk: np.ndarray | None = None
        self._exec_obs_buffer: list[dict[str, Any]] = []
        self._last_exec_step_count = 0
        self.ee_mode = self.action_type == "ee"
        self._ee_init_pose: np.ndarray | None = None  # absolute [L7, Lg, R7, Rg] at last server reset
        if self.ee_mode:
            print("[LingBot_VA] EEF mode: channels [0-6,28,7-13,29], relative to pose at reset", flush=True)''')

# 4. receding horizon: reference = current pose (server forgets everything at reset)
rep('''    def _soft_reset_inference_state(self):
        """Clear server KV/VAE state (receding-horizon replan)."""
        self._ws.infer({"reset": True, "prompt": self.default_prompt})''',
'''    def _soft_reset_inference_state(self):
        """Clear server KV/VAE state (receding-horizon replan)."""
        self._ws.infer({"reset": True, "prompt": self.default_prompt})
        if self.ee_mode and self.observation_window is not None:
            self._ee_init_pose = np.asarray(self.observation_window[0]["observation.state"], dtype=np.float64)''')

# 5. conversion
rep('''    def _resolve_keyframe_stride(self) -> int:''',
'''    def _convert_to_ee_control_chunk(self, action_chunk):
        # wan_va_server.postprocess_action already returns only used_action_channel_ids,
        # so with the robotwin config the server sends 16 channels [L7, Lg, R7, Rg].
        # Accept 30 too (full layout) and select.
        action_chunk = np.asarray(action_chunk, dtype=np.float64)
        if action_chunk.ndim != 2 or action_chunk.shape[1] not in (16, 30):
            raise ValueError(f"EEF conversion expects (T, 16) or (T, 30), got {action_chunk.shape}")
        if self._ee_init_pose is None:
            raise RuntimeError("EEF mode: no reference pose stored (server reset not seen)")
        rel = action_chunk if action_chunk.shape[1] == 16 else action_chunk[:, EE_CONTROL_INDICES]
        init = self._ee_init_pose
        out = []
        for row in rel:
            left = _add_eef_pose(row[:8], init[:8])
            right = _add_eef_pose(row[8:], init[8:])
            left[3:7] /= np.linalg.norm(left[3:7])
            right[3:7] /= np.linalg.norm(right[3:7])
            out.append(np.concatenate([left, right]))
        return np.asarray(out, dtype=np.float32)

    def _resolve_keyframe_stride(self) -> int:''')

rep('''        action = self._format_action_chunk(result["action"])
        action = self._convert_to_joint_control_chunk(action)''',
'''        action = self._format_action_chunk(result["action"])
        if self.ee_mode:
            action = self._convert_to_ee_control_chunk(action)
        else:
            action = self._convert_to_joint_control_chunk(action)''')

# 6. closed loop: reference = pose at the episode's first observation
rep('''        print(f"[LingBot_VA] reset server with instruction: {prompt!r}", flush=True)
        self._ws.infer({"reset": True, "prompt": prompt})''',
'''        print(f"[LingBot_VA] reset server with instruction: {prompt!r}", flush=True)
        self._ws.infer({"reset": True, "prompt": prompt})
        if self.ee_mode:
            self._ee_init_pose = np.asarray(observation["observation.state"], dtype=np.float64)''')

# 7. return flat (T, 16) in ee mode (unpack_robot_state would use arm_dim 6)
rep('''        if self.robot_action_dim_info is None:
            return action_chunk

        return unpack_robot_state(''',
'''        if self.ee_mode or self.robot_action_dim_info is None:
            return action_chunk

        return unpack_robot_state(''')

# 8. full reset clears the reference
rep('''        self._ws.infer({"reset": True, "prompt": self.default_prompt})
        self.observation_window = None''',
'''        self._ws.infer({"reset": True, "prompt": self.default_prompt})
        self._ee_init_pose = None
        self.observation_window = None''')

open(p, "w").write(s)
print("patched", p)
