from typing import Optional
import torch, sys, shutil, logging, time
from pathlib import Path
from tqdm import tqdm
from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train
import multiprocessing
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('PPO-Lag Minimal')

ENVIRONMENT = ['SafetyCarGoal1-v0']
PPO_LAG = ['PPOLag']


def get_unique_exp_dir(base: Path, name: str) -> Path:
    """Return a unique experiment directory path that doesn't yet exist."""
    exp_dir = base / name
    i = 1
    while exp_dir.exists():
        exp_dir = base / f"{name}_{i}"
        i += 1
    return exp_dir


def return_gpu():
    available_gpus = list(range(torch.cuda.device_count()))
    gpu_id = [0, 1, 2, 3]
    if gpu_id and not set(gpu_id).issubset(available_gpus):
        logger.info("GPU not available. Using CPU instead.")
        gpu_id = None
    return gpu_id


def train_with_progress(eg: ExperimentGrid, num_pool: int, gpu_id, total_steps: int, steps_per_epoch: int):
    total_epochs = (total_steps + steps_per_epoch - 1) // steps_per_epoch
    start_time = time.time()
    logger.info(f"Starting training for ~{total_epochs} epochs...")
    eg.run(train, num_pool=num_pool, gpu_id=gpu_id)
    elapsed = time.time() - start_time
    logger.info(f"✅ Training finished in {elapsed/60:.1f} minutes.")


def train_ppo(force_clean: bool = False):
    experiment_name = 'PPO-Lag_ObstacleAvoid'
    base_directory = Path("exp-x")
    base_directory.mkdir(parents=True, exist_ok=True)

    gpu_id = return_gpu()
    log_weights_and_biases = False
    number_of_envs_to_run_in_parallel = 4
    pytorch_thread_count = 1
    number_of_steps_per_epoch = 32000
    maximum_number_of_steps = 3200000
    seeds = [0]
    cost_limit = 25
    number_of_experiments_run_at_the_same_time = 1

    # Choose experiment directory
    exp_dir = base_directory / experiment_name
    if exp_dir.exists() and not force_clean:
        logger.info(f"Existing experiment '{exp_dir}' found — creating a new unique run folder.")
        exp_dir = get_unique_exp_dir(base_directory, experiment_name)
    elif exp_dir.exists() and force_clean:
        shutil.rmtree(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
    else:
        exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using experiment directory: {exp_dir}")

    # Set up ExperimentGrid
    eg = ExperimentGrid(exp_name=exp_dir.name)
    eg.add('env_id', ENVIRONMENT)
    eg.add('algo', PPO_LAG)
    eg.add('logger_cfgs:use_wandb', [log_weights_and_biases])
    eg.add('train_cfgs:vector_env_nums', [number_of_envs_to_run_in_parallel])
    eg.add('train_cfgs:torch_threads', [pytorch_thread_count])
    eg.add('algo_cfgs:steps_per_epoch', [number_of_steps_per_epoch])
    eg.add('train_cfgs:total_steps', [maximum_number_of_steps])
    eg.add('seed', seeds)
    eg.add('lagrange_cfgs:cost_limit', [cost_limit])

    eg.add('algo_cfgs:entropy_coef', [0.01])      # encourages exploration
   
    
    # Train
    train_with_progress(
        eg,
        num_pool=number_of_experiments_run_at_the_same_time,
        gpu_id=gpu_id,
        total_steps=maximum_number_of_steps,
        steps_per_epoch=number_of_steps_per_epoch
    )

    # Post-train
    eg.analyze(parameter='env_id', values=None, compare_num=1)
    eg.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    eg.evaluate(num_episodes=1)
    logger.info("✅ Training Complete!")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-clean', action='store_true', help='Forcefully clean directory before running.')
    args = parser.parse_args()
    train_ppo(force_clean=args.force_clean)

