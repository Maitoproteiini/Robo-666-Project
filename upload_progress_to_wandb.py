# Exporting old runs or just data from somewhere else using "progress.csv" to your Wandb.ai
# Fill in "EDIT THESE" and the run's details in rows 45 and 46. Make sure you are logged in Wandb.ai.

import os, sys, pandas as pd, wandb
from pathlib import Path

# ---- EDIT THESE ----
RUN_DIR = Path(" ")     # path to your directory
PROJECT  = " "           # your W&B project
ENTITY   = " "  # your W&B entity/org
RUN_NAME = RUN_DIR.name             # name on W&B
CSV_PATH = RUN_DIR / "progress.csv"
# --------------------

if not CSV_PATH.exists():
    print(f"progress.csv not found at: {CSV_PATH}")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)

# Pick a sensible x-axis/step. Try common OmniSafe columns; else use the row index.
CANDIDATE_STEPS = [
    "total_steps","TotalEnvSteps","Steps","global_step","epoch","Epoch","iteration","Itr"
]
step_col = next((c for c in CANDIDATE_STEPS if c in df.columns), None)
if step_col is None:
    df["_step"] = range(len(df))
    step_col = "_step"

# Optional: cast non-numerics to strings (W&B accepts mixed, but cleaner).
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            df[col] = df[col].astype(str)

# Start the run
wandb.login()  # make sure you're logged in
run = wandb.init(
    project=PROJECT,
    entity=ENTITY,
    name=RUN_NAME,
    config={
        "env_id": "SafetyCarGoal1-v0",  # change to your run's environment
        "algo": "PPOLag",   # change to your run's algorithm
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

run.finish()
print("Done. Check your W&B project dashboard.")
