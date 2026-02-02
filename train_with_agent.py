#!/usr/bin/env python3
"""
This module trains a given safety-gymnasium agent and an environment using omnisafe.
The module can be run with command line arguments.
By default it trains a PPOLag agent on the SafetyCarGoal1-v0 environment with default configurations.
The trained model will be saved under runs folder.
The training process can be tracked using Weights & Biases if configured.

Flags:
--config: Path to the configuration YAML/YML file. Default is 'configurations/default_configurations.yaml'.
--algorithm: The algorithm to be used for training. Default is 'PPOLag'.
--env_id: The environment ID to train the agent on. Default is 'SafetyCarGoal1-v0'.
--total_steps: Total number of training steps. Default is 2,000,000.
--wandb-project: Weights & Biases project name. Default is an empty string. If not set, wandb tracking is disabled.

Example usage:
python3 -m train_with_agent --config configurations/custom_config.yaml --algorithm PPOLag --env_id SafetyCarGoal1-v0 --total_steps 3000000 --use_wandb True
"""

import os
import torch
import omnisafe
import argparse

import logging, sys
import yaml
import wandb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger('train_with_agent')

CONFIGURATION_FILE = 'default_configurations.yaml'
CONFIGURATION_DIR = os.path.join(os.path.dirname(__file__), 'configurations')
PATH_TO_CONFIG = os.path.join(CONFIGURATION_DIR, CONFIGURATION_FILE)

DEFAULT_ENV_ID = "SafetyCarGoal1-v0"
DEFAULT_TOTAL_STEPS = 2_000_000
DEFAULT_ALGORITHM = "PPOLag"

def train_with_agent(env_id=DEFAULT_ENV_ID, algorithm=DEFAULT_ALGORITHM, config_path=PATH_TO_CONFIG, wandb_project="") -> None:
    """
    This function initializes and trains an agent bas
    
    :param env_id: The evnvironment ID to train the agent on.
    :param algorithm: The algorithm to be used for training.
    :param config_path: The path to the configuration file.
    """
    logger.info(f"Starting training with {algorithm} on {env_id} environment.")
    
    device_str = return_device()
    logger.info(f"Using device: {device_str}")

    configuration = load_configuration(config_path)

    if len(wandb_project) > 0:
        configuration = set_wandb_config(configuration, wandb_project)
        # wandb.init(project=wandb_project, config=configuration, name=f"{algorithm}_{env_id}")

    agent = omnisafe.Agent(algo=algorithm, env_id=env_id, custom_cfgs=configuration)
    agent.learn()

def set_wandb_config(config, wandb_project) -> dict:
    """
    This function sets up Weights & Biases (wandb) configuration
    based on the provided configuration dictionary.

    :param config: A dictionary containing configuration parameters.
    :param wandb_project: The wandb project name.

    :return: Updated configuration dictionary with wandb settings.
    """
    
    config['logger_cfgs']['use_wandb'] = True
    config['logger_cfgs']['wandb_project'] = wandb_project

    _orig_update = wandb.sdk.wandb_config.Config.update
    def _patched_update(self, d=None, allow_val_change=False):
        return _orig_update(self, d, allow_val_change=True)
    wandb.sdk.wandb_config.Config.update = _patched_update

    return config

def return_device() -> str:
    """
    This function checks for the availability of XPU, CUDA, and CPU devices
    in that order and returns the appropriate device string.

    :return: A string representing the available device ("xpu:0", "cuda:0", or "cpu").
    """
    try:
        import torch_xpu
        if torch.xpu.is_available():
            return "xpu:0"
    except ImportError:
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

def load_configuration(config_path) -> dict:
    """
    This function loads configuration from a YAML file
    and returns it as a dictionary.
    
    :param config_path: The path to the configuration file. Must be .yaml or .yml
    :return: A dictionary containing the configuration parameters.
    """

    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as file:
            configuration = yaml.safe_load(file)
    else:
        raise ValueError("Configuration file must be a .yaml, or .yml file")
    return configuration

def parse_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train agent on SafetyCarGoal1-v0")
    parser.add_argument(
        "--config",
        type=str,
        default=PATH_TO_CONFIG,
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=DEFAULT_ALGORITHM,
        help="The algorithm to be used for training."
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default=DEFAULT_ENV_ID,
        help="The environment ID to train the agent on."
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=DEFAULT_TOTAL_STEPS,
        help="Total number of training steps."
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="",
        help="Weights & Biases project name."
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Get arguments from command line
    args = parse_args()
    train_with_agent(env_id=args.env_id, algorithm=args.algorithm, config_path=args.config, wandb_project=args.wandb_project)