

import torch

import sys, shutil, logging
from pathlib import Path

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)


logger = logging.getLogger('PPO-Lag-With-Obstacles')

ENVIRONMENT = ['SafetyRacecarGoal1-v0']
PPO_LAG = ['PPOLag']


def ensure_experiment_dir_is_clean(exp_base: str, experiment_name: str, *, ask: bool = True, force: bool = False) -> Path:
    base = Path(exp_base).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    exp_dir = base / experiment_name

    if exp_dir.exists():
        if force:
            logging.info(f"Removing existing: {exp_dir}")
            shutil.rmtree(exp_dir)
        elif ask and sys.stdin.isatty():
            reply = input(f"The directory '{exp_dir}' already exists. Clean it? [y/N]: ").strip().lower()
            if reply in {"y", "yes"}:
                logging.info(f"Removing existing: {exp_dir}")
                shutil.rmtree(exp_dir)
            else:
                logging.info(f"Keeping existing directory: {exp_dir}")
        else:
            raise FileExistsError(f"Directory exists: {exp_dir}. Run with force=True to clean.")
    if exp_dir.exists():
        raise FileExistsError(f"Directory still exists: {exp_dir}. Failed to clean it. Omnisafe needs the directory to not exists.")
    return exp_dir

def return_gpu():
    avaliable_gpus = list(range(torch.cuda.device_count()))
    gpu_id = [0, 1, 2, 3]

    if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
        logger.info('Gpu not available. Using CPU instead.')
        gpu_id = None
    return gpu_id


def train_ppo():

    # Set variables
    experiment_name = 'PPO-Lag-With-Obstacles'
    gpu_id = return_gpu()
    log_weights_and_biases = False
    number_of_envs_to_run_in_parallel = 8
    pytorch_thread_count = 1
    number_of_steps_per_epoch = 4000
    maximum_number_of_steps = 4000000 # maximum_number_of_steps / number_of_steps_per_epoch = 10 epochs
    seeds = [1,2,3,4]
    cost_limit = 25
    entropy_coefficient = 0.005
    kl_divergence = 0.02
    lambda_init = 0.001
    lambda_learn_rate = 0.0
    # The num_pool is different from number_of_envs_to_run_in_parallel
    # One experiment can have multiple envs run.
    # number_of_experiments_run_at_the_same_time must be divisible by number_of_envs_to_run_in_parallel 
    number_of_experiments_run_at_the_same_time = 1

    logger.info(f"Experiment name            : {experiment_name}")
    logger.info(f"Log to Weights & Biases    : {log_weights_and_biases}")
    logger.info(f"Vector envs in parallel    : {number_of_envs_to_run_in_parallel}")
    logger.info(f"PyTorch thread count       : {pytorch_thread_count}")
    logger.info(f"Steps per epoch            : {number_of_steps_per_epoch}")

    epochs = (maximum_number_of_steps + number_of_steps_per_epoch - 1) // number_of_steps_per_epoch
    steps_per_env_per_epoch = number_of_steps_per_epoch // max(1, number_of_envs_to_run_in_parallel)

    logger.info(f"Max total steps            : {maximum_number_of_steps} (~{epochs} epochs)")
    logger.info(f"~Steps/env/epoch (approx)  : {steps_per_env_per_epoch}")
    logger.info(f"Seeds                      : {seeds}")

    base_directory = "exp-x"
    ensure_experiment_dir_is_clean(base_directory, experiment_name)

    eg = ExperimentGrid(exp_name=experiment_name)

    # Set environment.
    eg.add('env_id', ENVIRONMENT)

    # Set PPO-Lag as the algortihm to use
    eg.add('algo', PPO_LAG )
    
    # Don't log weights and biases. Set True to log.
    eg.add('logger_cfgs:use_wandb', [log_weights_and_biases])

    # Log to tensorboard
    eg.add('logger_cfgs:use_tensorboard', [True])
    
    # Set the number of envs to run in parallel 
    eg.add('train_cfgs:vector_env_nums', [number_of_envs_to_run_in_parallel])

    # Set thread count
    eg.add('train_cfgs:torch_threads', [pytorch_thread_count])

    # Set number of steps
    eg.add('algo_cfgs:steps_per_epoch', [number_of_steps_per_epoch])

    # Set entropy bonus, the higher the value the more exploration is encouraged
    #eg.add('algo_cfgs:entropy_coef', [entropy_coefficient])
    
    # Set target_kl, the higher the value the more stable the training
    #eg.add('algo_cfgs:target_kl', [kl_divergence])
    
    # Set the number of epochs
    eg.add('train_cfgs:total_steps', [maximum_number_of_steps])
    
    # Set no seeds
    eg.add('seed', seeds)

    # Set cost limit, the higher the limit the more the agent can afford speed-costs
    # i.e. it’ll drive quicker
    eg.add('lagrange_cfgs:cost_limit', [cost_limit])

    # Set starting weight on safety cost, if too big the robot might not move
    eg.add('lagrange_cfgs:lagrangian_multiplier_init', lambda_init)

    # Set learning rate for updating lambda, updates using the dual gradient step
    # eg.add('lagrange_cfgs:lambda_lr', [lambda_learn_rate])

    # Train
    eg.run(train, num_pool=number_of_experiments_run_at_the_same_time, gpu_id=gpu_id)

    # analyze learning speed vs stability
    # Note that the parameter has to be larger that compare_num
    eg.analyze(parameter='env_id', values=None, compare_num=1)

    # render videos
    eg.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    eg.evaluate(num_episodes=1)
    logger.info("Done!")


if __name__ == '__main__':
    train_ppo()