# run_ppolag_car.py
"""
Run this code on bath with the following command:

python3 run_ppolag_car.py

"""

import os
import torch
import omnisafe

import logging, sys
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger('PPO-Lag Car Goal 1')

CONFIGURATION_FILE = 'SACLag.yaml'
CONFIGURATION_DIR = os.path.join(os.path.dirname(__file__), 'configurations')
PATH_TO_CONFIG = os.path.join(CONFIGURATION_DIR, CONFIGURATION_FILE)

def main():
    env_id = "SafetyCarGoal1-v0"
    logger.info(f"Starting training with SACLag on {env_id} environment")
    if torch.xpu.is_available():
        device_str = "xpu:0"
    elif torch.cuda.is_available():
        device_str = "cuda"
    device_str = "cpu"

    logger.info(f"Using device: {device_str}")

    with open(PATH_TO_CONFIG, "r") as f:
        custom_cfg = yaml.safe_load(f)["defaults"]

    custom_cfg['train_cfgs']['device'] = device_str
    logger.info(f"Configuration loaded from {PATH_TO_CONFIG}:")
    logger.info(custom_cfg)

    my_cfg = {    "train_cfgs": {
        "device": device_str,
        "torch_threads": 16,
        "vector_env_nums": 1,
        "parallel": 1,
        "total_steps": 2_000_000,
    },}
    

    agent = omnisafe.Agent(algo="SACLag", env_id=env_id, custom_cfgs=custom_cfg)
    agent.learn()

if __name__ == "__main__":
    main()