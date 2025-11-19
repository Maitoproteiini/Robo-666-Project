#!/usr/bin/env python3
import argparse
import sys
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def sanitize_filename(name: str) -> str:
    bad = "/\\<>:*?|\"'"
    out = []
    for ch in name:
        if ch in bad or ch.isspace():
            out.append("_")
        else:
            out.append(ch.replace("/", "_"))
    return "".join(out)

def verify_arguments_and_load_data(csv_path: str, args):
    if not csv_path.exists():
        raise SystemExit(f"[error] csv not found: {csv_path}")

    try:
        data_file = pd.read_csv(csv_path)
    except Exception as e:
        raise SystemExit(f"[error] Failed to read csv: {e}")
    
    if args.metric == "all":
        all_metrics = list(data_file.columns)
        args.metric = all_metrics
        return data_file
    
    args.metric = [args.metric]

    if args.metric[0] not in data_file.columns:
        cols = list(data_file.columns)
        suggestions = [c for c in cols if args.metric in c or c in args.metric]
        msg = f"[error] Metric '{args.metric}' not found.\nAvailable columns include:\n  - " + "\n  - ".join(cols[:30])
        if suggestions:
            msg += "\n\nDid you mean:\n  - " + "\n  - ".join(suggestions)
        raise SystemExit(msg)
    return data_file

def plot_metric(data_file: pd.DataFrame, metric: str, out_path: Path, show: bool = True):
    # Assume row index corresponds to epoch number starting from the second row.
    x = (data_file.index + 1).values
    y = data_file[metric].values

    # Plot
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Epoch")
    plt.grid(True)

    file_name = sanitize_filename(metric) + "_vs_epoch.png"

    if out_path:
        out_path = f"{out_path}/{file_name}"
    else:
        out_path = f"{file_name}" 

    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    # Also show for interactive use
    try:
        if show:
            plt.show()
    except Exception:
        pass
    plt.close()
    logger.info(f"[ok] Saved plot to: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Plot a metric vs epochs from an progress csv file.")
    p.add_argument("--csv", required=True, help="Path to progress csv file.")
    p.add_argument("--metric", required=True, default="Metrics/EpRet", help="Column name of the metric to plot. Defaults to 'Metrics/EpRet'. Use 'all' to plot all columns.")
    p.add_argument("--out", help="Optional output directory path. Defaults to current directory.")
    args = p.parse_args()

    csv_path = Path(args.csv)
    data_file = verify_arguments_and_load_data(csv_path, args)

    show = True

    if len(args.metric) > 1:
        logger.info(f"[info] Plotting multiple metrics: {args.metric}")
        show = False

    for metric in args.metric:
        plot_metric(data_file, metric, args.out, show)


if __name__ == "__main__":
    main()
