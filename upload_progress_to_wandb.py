"""
Script for retro-importing an existing progress.csv into Wandb.ai.
You can import old runs or runs from other sources to your own Wandb.ai.

Fill in "EDIT THESE". Make sure you are logged in Wandb.ai.

Origin: Made for the Safe RL course project (ROBO.666 Project Work).

Run this code on path with the following command:

python3 upload_progress_to_wandb.py

"""
import os
import sys 
import pandas as pd
import wandb
from pathlib import Path

# ---- EDIT THESE ----
RUN_DIR = Path(" ")     # path to your directory that contains the progress.csv
PROJECT  = " "          # your W&B project
ENTITY   = " "          # your W&B entity/org
RUN_NAME = RUN_DIR.name 
CSV_PATH = RUN_DIR / "progress.csv"
# --------------------

if not CSV_PATH.exists():
    print(f"progress.csv not found at: {CSV_PATH}")
    sys.exit(1)

# Load the logged metrics into DataFrame
df = pd.read_csv(CSV_PATH)

# Pick a sensible x-axis/step. Try common OmniSafe columns; else use the row index.
CANDIDATE_STEPS = [
    "total_steps","TotalEnvSteps","Steps","global_step","epoch","Epoch","iteration","Itr"
]

# Find the first column name from CANDIDATE_STEPS that is present in df.columns
step_col = next((c for c in CANDIDATE_STEPS if c in df.columns), None)

# Fall back to using row index as the step, if step_col found None
if step_col is None:
    df["_step"] = range(len(df))
    step_col = "_step"

# Optional: cast non-numerics to strings for cleaner plotting
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            df[col] = df[col].astype(str)

# Start the Wandb.ai run
wandb.login()  # Assumes you have logged in in the shell

run = wandb.init(
    project=PROJECT,
    entity=ENTITY,
    name=RUN_NAME,
    config={
        "env_id": "SafetyCarGoal1-v0",  # Change to your run's environment for metadata (no affect to logging)
        "algo": "PPOLag",   # Change to your run's algorithm for metadata (no affect to logging)
        "source": "retro-import", 
    },
    settings=wandb.Settings(start_method="thread"),
)

# Log each row as a step
for _, row in df.iterrows():
    step = int(row[step_col]) if pd.api.types.is_integer_dtype(type(row[step_col])) or str(row[step_col]).isdigit() else None
    # Convert row to dict
    log_dict = row.to_dict()
    # Optional: replace slashes in keys (W&B can handle '/', but cleaner graphs without)
    # log_dict = {k.replace("/", "_"): v for k, v in log_dict.items()}
    wandb.log(log_dict, step=step)

# Finish the run to ensure everything is in Wandb.ai
run.finish()
print("Done. Check your W&B project dashboard.")
