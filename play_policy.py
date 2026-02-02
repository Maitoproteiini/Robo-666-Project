#!/usr/bin/env python3
"""
This module exports videos of trained OmniSafe policies.
The module can be run with command line arguments.
This module takes at least one argument, the path to the trained model's run directory.
Optionally, the number of evaluation episodes can be specified (default is 1).
Flags:
--run_path: Path to the trained model's run directory.
--episodes: Number of evaluation episodes. Default is 1.

Example usage:
python3 -m play_policy runs/PPOLag-SafetyRacecarGoal0-v0
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Run a trained OmniSafe policy.")
    p.add_argument("run_path", type=Path, help="Path to run dir (e.g., .../PPOLag-{SafetyRacecarGoal0-v0})")
    p.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes")
    args = p.parse_args()

    if not args.run_path.exists():
        print(f"ERROR: {args.run_path} not found", file=sys.stderr)
        sys.exit(1)

    omnisafe_bin = shutil.which("omnisafe")
    if omnisafe_bin:
        cmd = [omnisafe_bin, "eval", str(args.run_path), "--num-episode", str(args.episodes)]
    else:
        cmd = [sys.executable, "-m", "omnisafe", "eval", str(args.run_path), "--num-episode", str(args.episodes)]

    print(">> Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Eval failed (exit {e.returncode}). Is omnisafe installed? `pip install omnisafe`", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
