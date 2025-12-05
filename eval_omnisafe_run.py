"""
Script for evaluating a specific run using Omnisafe. This script generates two 
(unless specified otherwise) video simulations of the run.

Original script: Naeim Ebrahimi Toulkani
Current version: Documented for Safe RL course project (ROBO.666 Project Work)

In the run command you specify the directory and how many videos to generate.
Run this code on path with the following command:

python3 eval_omnisafe_run.py \
  --run-dir ./runs/PPOLag-{SafetyCarGoal1-v0} \
  --episodes 2

"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List

# File name patterns for model checkpoints
CHECKPOINT_PATTERNS = [
    "*.pt", "*.pth", "model.pt",
    "epoch-*.pt", "checkpoint-*.pt",
]

def find_checkpoint(run_dir: Path) -> Optional[Path]:
    """
    Find the most recent-looking checkpoint file under a run dir.
    
    Parameters:
    run_dir : Path to the Omnisafe run directory.

    Returns:
    candidates (Optional[Path]) : The best candidate checkpoint file, or 
    None if no candidates are found.
    
    """
    if not run_dir.exists():
        print(f"[!] Run dir does not exist: {run_dir}", file=sys.stderr)
        return None
    
    candidates: List[Path] = []

    for pat in CHECKPOINT_PATTERNS:
        candidates.extend(run_dir.rglob(pat))

    # Prefer files under a 'models' or 'checkpoints' subdir if present
    def score(p: Path) -> tuple:
        pref = 0
        if "models" in p.parts: pref -= 2
        if "checkpoints" in p.parts: pref -= 1
        # newer mtime wins
        return (pref, -p.stat().st_mtime)
    
    candidates = [c for c in candidates if c.is_file()]

    if not candidates:
        return None
    
    candidates.sort(key=score)
    return candidates[0]

def main():
    """
    Parse command-line arguments, locate a checkpoint and call Omnisafe's eval.
    
    """
    ap = argparse.ArgumentParser(description="Render a trained PPO-Lag policy in Safety-Gymnasium.")
    ap.add_argument("--run-dir", type=str, required=True,
                    help="Path to the training run directory that contains config.json and checkpoints (e.g., ./runs/PPOLag-{SafetyCarGoal1-v0}/seed-000-YYYY.../)")
    ap.add_argument("--env-id", type=str, default="SafetyCarGoal1-v0",
                    help="Environment ID (default: SafetyCarGoal1-v0)")
    ap.add_argument("--episodes", type=int, default=1,
                    help="How many episodes to render")
    ap.add_argument("--device", type=str, default=None,
                    help="Device string for eval (e.g., cpu, cuda:0). If omitted, OmniSafe default is used.")
    ap.add_argument("--model-path", type=str, default=None,
                    help="Optional explicit path to a checkpoint .pt/.pth. If omitted, auto-detect from run-dir.")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="Anything after --extra is forwarded to `omnisafe eval` (advanced).")
    args = ap.parse_args()

    # Locate omnisafe executable in the current path
    omnisafe_exe = shutil.which("omnisafe")
    if omnisafe_exe is None:
        print("[!] Could not find `omnisafe` CLI in PATH. Activate the env where OmniSafe is installed.", file=sys.stderr)
        sys.exit(1)

    # Normalise and resolve paths
    run_dir = Path(args.run_dir).resolve()
    model_path: Optional[Path] = Path(args.model_path).resolve() if args.model_path else None

    # Before calling omnisafe, if there is not a model path provided, check for a checkpoint
    if model_path is None:
        model_path = find_checkpoint(run_dir)
        if model_path is None:
            print(f"[!] No checkpoint found under {run_dir}\n"
                  f"    Looked for: {', '.join(CHECKPOINT_PATTERNS)}", file=sys.stderr)
            sys.exit(2)

    # Guard against the case where the user passed a non-existing model path
    if not model_path.exists():
        print(f"[!] Model path does not exist: {model_path}", file=sys.stderr)
        sys.exit(3)

    print(f"[i] Using model checkpoint: {model_path}")
    print(f"[i] Evaluating on: {args.env_id}  |  episodes: {args.episodes}")

    # Base command for Omnisafe evaluation
    cmd = [
    omnisafe_exe, "eval",
    "--render",
    "--num-episode", str(args.episodes),  # Omnisafe uses the singular flag name
    str(run_dir)
    ]

    # Optionally propagate device choice
    if args.device:
        cmd += ["--device", args.device]

    # Forward any extra arguments to 'omnisafe eval'
    if args.extra:
        cmd += args.extra

    print(f"[i] Command: {' '.join(cmd)}")

    # Run the command us subprocess, inherit current env (so WANDB_* etc still apply if present)
    try:
        proc = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] `omnisafe eval` failed with return code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
