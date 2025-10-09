#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import json
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from omnisafe.common.statistics_tools import StatisticsTools
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger('Plot Statistics')

DEFAULT_PATH = './exp-x/PPO-Lag Minimal'
DEFAULT_GRID_CONFIG_PATH = f"{DEFAULT_PATH}/grid_config.json"
DEFAULT_SAVED_PLOTS_DIRECTORY = './published/PPO-Lag-Minimal/plots'
PLOT_ALL_PARAMETERS = False # Don't switch, no point unless multiple
PARAMETERS_TO_IGNORE = ["algo"]

# draw_graph parameters and definitions according to documentation: https://omnisafe.readthedocs.io/en/latest/common/stastics_tool.html
# parameter (str) – The parameter to compare.
PARAMETER = "env_id"

# values (list[Any] or None, optional) – The values of the parameter to compare. Defaults to None.
# Note: values and compare_num cannot be set at the same time.
VALUES = ["SafetyRacecarGoal0-v0"]

# compare_num (int or None, optional) – The number of values to compare. Defaults to None. 
# Note: values and compare_num cannot be set at the same time.
COMPARE_NUM = None

# cost_limit (float or None, optional) – The cost limit of the experiment. Defaults to None.
COST_LIMIT = None

# smooth (int, optional) – The smooth window size. Defaults to 1.
SMOOTH = 1

# show_image (bool) – Whether to show graph image in GUI windows.
SHOW_IMAGE=True

def list_directory_contents(path: str):
    logger.info(f"Contents of directory: {os.path.abspath(path)}\n")
    try:
        entries = os.listdir(path)
    except PermissionError:
        logger.info("Error: Permission denied.")
        sys.exit(1)

    for item in entries:
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            logger.info(f"[DIR]  {item}")
        else:
            logger.info(f"       {item}")

def return_paths():
    parser = argparse.ArgumentParser(description="List files and subdirectories in a directory.")
    parser.add_argument("path", nargs="?", help="Path to the directory (optional). If omitted, uses DEFAULT_PATH.")
    parser.add_argument("saved_plots_path", nargs="?", help="Path to the directory where to save plots (optional). If omitted, uses DEFAULT_SAVED_PLOTS_DIRECTORY.")
    args = parser.parse_args()

    path = args.path if args.path is not None else DEFAULT_PATH
    saved_plots_path = args.saved_plots_path if args.saved_plots_path is not None else DEFAULT_SAVED_PLOTS_DIRECTORY

    if args.path is None:
        logger.info(f"No path provided. Using default: {path}")
    if args.saved_plots_path is None:
        logger.info(f"No saved plots path provided. Using default: {saved_plots_path}")
    return path, saved_plots_path

def verify_argument_path_exists(path: str, saved_plots_path: str):
    if not os.path.exists(path):
        logger.info(f"Error: The path '{path}' does not exist.")
        sys.exit(1)

    if not os.path.isdir(path):
        logger.info(f"Error: The path '{path}' is not a directory.")
        sys.exit(1)

    if not os.path.exists(saved_plots_path):
        logger.info(f"Error: The path to save plots '{saved_plots_path}' does not exist.")
        sys.exit(1)
    
    grid_config_path = Path(path) / "grid_config.json"
    if not os.path.exists(grid_config_path) and PLOT_ALL_PARAMETERS:
        logger.info(f"Warning: The file grid_config.json does not exist in '{grid_config_path}'")
        logger.info(f"Can not plot all parameters. Will only plot the given parameter: {PARAMETER}")

def return_all_parameters_and_values(path):
    exp_dir = Path(path)
    grid_config_path = exp_dir / "grid_config.json"

    with open(grid_config_path, "r") as f:
        grid_config = json.load(f)
    logger.info(grid_config)
    return grid_config

def plot_and_save_one_single_graph(statistics_tool: StatisticsTools, saved_plots_path: str, parameter:str = PARAMETER, value: list[Any] = VALUES, headless:bool = True):
    outfile = f"{saved_plots_path}/{parameter}_plot.png"
    if not headless:
        statistics_tool.draw_graph(parameter=parameter, values=value, compare_num=COMPARE_NUM, cost_limit=COST_LIMIT, show_image=SHOW_IMAGE)
    logger.info(f"Saving plot to {outfile}")
    matplotlib.use("Agg")
    statistics_tool.draw_graph(parameter=parameter, values=value, compare_num=COMPARE_NUM, cost_limit=COST_LIMIT, show_image=SHOW_IMAGE)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close("all")
    print(f"Saved plot to: {outfile}")

def plot_statistics(path: str, saved_plots_path: str):
    st = StatisticsTools()
    st.load_source(path)
    if not PLOT_ALL_PARAMETERS:
        plot_and_save_one_single_graph(st, saved_plots_path, headless=False)
    else:
        all_parameters_and_values = return_all_parameters_and_values(path)
        for parameter, values in all_parameters_and_values.items():
            if len(values) <= 1:
                continue
            plot_and_save_one_single_graph(st, saved_plots_path, parameter=parameter, value=values, headless=True)


if __name__ == "__main__":
    path, saved_plots_path = return_paths()
    verify_argument_path_exists(path, saved_plots_path)
    list_directory_contents(path)
    plot_statistics(path, saved_plots_path)