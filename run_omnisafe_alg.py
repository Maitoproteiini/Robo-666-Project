"""
Script for training a chosen algorithm in a chosen environment using Omnisafe library 
and uploading progress to Wandb.ai. You can alter the hyperparameters like shown using 
custom_cfgs, otherwise the run uses the default values in Omnisafe.

You need to input your Wandb.ai details, environment, hyperparameters and algorithm.
Original script is from Naeim Ebrahimi Toulkani.

Run this code on path with the following command:

python3 run_omnisafe_alg.py

"""

import os
import torch
import omnisafe

# --- W&B allow config value changes  ---
import wandb
_orig_update = wandb.sdk.wandb_config.Config.update
def _patched_update(self, d=None, allow_val_change=False):
    return _orig_update(self, d, allow_val_change=True)
wandb.sdk.wandb_config.Config.update = _patched_update

# ------------------------------------------------------------------------------

os.environ.setdefault("WANDB_ENTITY", " ") # Add your Wandb.ai entity
os.environ.setdefault("WANDB_PROJECT", " ") # Add your Wandb.ai project

print("PyTorch CUDA available:", torch.cuda.is_available())

env_id = "SafetyCarGoal1-v0" # Change to the environment you want to run
device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

# Here you can customize the run's hyperparameters, check the config block below for 
# hyperparameters' names and default values
custom_cfgs = {
    "seed": 0,
    "train_cfgs": {
        "device": device_str,
        "torch_threads": 16,
        "vector_env_nums": 1,
        "parallel": 1,
        "total_steps": 2_000_000,
    },
    
}

# Commented-out config block 
"""
"algo_cfgs": {
    "steps_per_epoch": 20000,
    "update_iters": 10,
    "batch_size": 64,
    "target_kl": 0.015,
    "entropy_coef": 0.0,
    "reward_normalize": False,
    "cost_normalize": False,
    "obs_normalize": True,
    "kl_early_stop": True,
    "use_max_grad_norm": True,
    "max_grad_norm": 40.0,
    "use_critic_norm": True,
    "critic_norm_coef": 0.001,
    "gamma": 0.99,
    "cost_gamma": 0.99,
    "lam": 0.95,
    "lam_c": 0.95,
    "clip": 0.2,
    "adv_estimation_method": "gae",
    "standardized_rew_adv": True,
    "standardized_cost_adv": True,
    "penalty_coef": 0.0,
    "use_cost": True,
},
"logger_cfgs": {
    "use_wandb": True,
    "wandb_project": os.environ.get("WANDB_PROJECT", "omnisafe-ppolag"),
    "use_tensorboard": True,
    "save_model_freq": 100,
    "log_dir": "./runs",
    "window_lens": 100,
},
"model_cfgs": {
    "weight_initialization_mode": "kaiming_uniform",
    "actor_type": "gaussian_learning",
    "linear_lr_decay": True,
    "exploration_noise_anneal": False,
    "std_range": [0.5, 0.1],
    "actor": {"hidden_sizes": [64, 64], "activation": "tanh", "lr": 3e-4},
    "critic": {"hidden_sizes": [64, 64], "activation": "tanh", "lr": 3e-4},
},
"lagrange_cfgs": {
    "cost_limit": 25.0,
    "lagrangian_multiplier_init": 0.001,
    "lambda_lr": 0.035,
    "lambda_optimizer": "Adam",
},
"env_cfgs": {},
"""

print("W&B target → entity:", os.environ.get("WANDB_ENTITY"),
      "project:", os.environ.get("WANDB_PROJECT"))

agent = omnisafe.Agent("PPOLag", env_id, custom_cfgs=custom_cfgs) # Change to your algorithm
agent.learn()
