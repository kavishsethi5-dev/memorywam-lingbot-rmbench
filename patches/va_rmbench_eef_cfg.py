# EEF config for LingBot-VA on RMBench, starting from lingbot-va-posttrain-robotwin.
# Put in: XPolicyLab/policy/LingBot_VA/lingbot_va/wan_va/configs/
# Register in configs/__init__.py (see bottom of this file).
#
# Model/inference settings = upstream va_robotwin_cfg (what posttrain-robotwin was
# trained with: robotwin_tshape, 256x320 head + 128x160 wrists, 16 EEF channels,
# action_per_frame 16, its own norm_stat for relative EEF poses).
# Training settings = XPolicyLab's robotwin30 fine-tune recipe (5000 steps, accum 8).
import os

from easydict import EasyDict

from .va_robotwin_cfg import va_robotwin_cfg

_POLICY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

va_rmbench_eef_cfg = EasyDict(__name__="Config: VA RMBench EEF (robotwin_tshape)")
va_rmbench_eef_cfg.update(va_robotwin_cfg)
va_rmbench_eef_cfg.infer_mode = "server"
# Training reads <this>/transformer. Link .merged_ckpt -> lingbot-va-posttrain-robotwin before training.
va_rmbench_eef_cfg.wan22_pretrained_model_name_or_path = os.path.join(_POLICY_ROOT, ".merged_ckpt")
va_rmbench_eef_cfg.enable_offload = True

va_rmbench_eef_train_cfg = EasyDict(__name__="Config: VA RMBench EEF train")
va_rmbench_eef_train_cfg.update(va_rmbench_eef_cfg)
va_rmbench_eef_train_cfg.dataset_path = os.environ.get("LINGBOT_VA_DATASET_PATH", "/path/to/latents")
va_rmbench_eef_train_cfg.empty_emb_path = os.path.join(va_rmbench_eef_train_cfg.dataset_path, "empty_emb.pt")
va_rmbench_eef_train_cfg.enable_wandb = False  # set True + fill run_va_posttrain.sh wandb vars if wanted
va_rmbench_eef_train_cfg.load_worker = 16
va_rmbench_eef_train_cfg.save_interval = 200
va_rmbench_eef_train_cfg.gc_interval = 50
va_rmbench_eef_train_cfg.cfg_prob = 0.1
va_rmbench_eef_train_cfg.learning_rate = 1e-5
va_rmbench_eef_train_cfg.beta1 = 0.9
va_rmbench_eef_train_cfg.beta2 = 0.95
va_rmbench_eef_train_cfg.weight_decay = 0.1
va_rmbench_eef_train_cfg.warmup_steps = 10
va_rmbench_eef_train_cfg.batch_size = 1
va_rmbench_eef_train_cfg.gradient_accumulation_steps = 8
va_rmbench_eef_train_cfg.num_steps = 5000

# --- add to configs/__init__.py -------------------------------------------
# from .va_rmbench_eef_cfg import va_rmbench_eef_cfg, va_rmbench_eef_train_cfg
# and inside VA_CONFIGS = {...}:
#     'rmbench_eef': va_rmbench_eef_cfg,
#     'rmbench_eef_train': va_rmbench_eef_train_cfg,
