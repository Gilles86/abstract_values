#!/usr/bin/env python3
"""Compare grand-average gaze trajectories between two groups of trials —
side-by-side grand averages plus their per-orientation and
across-orientation difference.

Two modes:
  1. Two files (default): different trial epochs, e.g. gabor presentation
     vs. response_bar/value-estimation.
  2. One file, split by a column (--split-col): e.g. `mapping` (cdf vs
     inverse_cdf) within a single epoch. Values sorted alphabetically;
     override the printed default labels with --label-a/--label-b if the
     sort order doesn't read naturally.

Same QC/recentring/two-stage-averaging pipeline as plot_gaze_trajectories.py
in both modes, reused via that module's aggregate_df()/load_and_aggregate()
rather than duplicated.

Caveat for mode 1 (state this whenever quoting the difference panel):
sample_idx is normalized time (0..1 fraction of each trial's OWN epoch
window), not wall-clock time — comparing epochs of different duration
compares trajectory *shape* at matched fractional progress, not literal
simultaneous timepoints. Not a concern for mode 2 (same epoch, same
window definition, just a trial subset).

Usage (local, no cluster needed):
    # mode 1: gabor vs response_bar epoch
    python -m abstract_values.visualize.compare_gaze_epochs \\
        --tsv-a notes/data/gaze_trajectories_gabor.tsv --label-a "Gabor presentation" \\
        --tsv-b notes/data/gaze_trajectories_all.tsv   --label-b "Response bar" \\
        --out notes/figures/gaze_trajectories_epoch_comparison.pdf

    # mode 2: cdf vs inverse_cdf mapping, within the gabor epoch
    python -m abstract_values.visualize.compare_gaze_epochs \\
        --tsv-a notes/data/gaze_trajectories_gabor.tsv --split-col mapping \\
        --out notes/figures/gaze_trajectories_mapping_comparison_gabor.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abstract_values.visualize.plot_gaze_trajectories import (
    aggregate_df,
    draw_trajectories,
    load_and_aggregate,
    smooth_path,
    symmetric_limits,
)


def compute_difference(grand_a: pd.DataFrame, grand_b: pd.DataFrame) -> pd.DataFrame:
    """Per (orientation, sample_idx): grand_a - grand_b."""
    m = grand_a.merge(grand_b, on=["orientation", "sample_idx"], suffixes=("_a", "_b"))
    return pd.DataFrame({
        "orientation": m["orientation"], "sample_idx": m["sample_idx"],
        "x_deg": m["x_deg_a"] - m["x_deg_b"], "y_deg": m["y_deg_a"] - m["y_deg_b"],
    })


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tsv-a", default="notes/data/gaze_trajectories_gabor.tsv")
    p.add_argument("--label-a", default=None)
    p.add_argument("--tsv-b", default="notes/data/gaze_trajectories_all.tsv")
    p.add_argument("--label-b", default=None)
    p.add_argument("--split-col", default=None,
                   help="Instead of comparing --tsv-a vs --tsv-b, split --tsv-a's trials "
                        "by this column (must have exactly 2 unique values), e.g. 'mapping'.")
    p.add_argument("--comparison-title", default=None,
                   help="Human-readable name for the compared factor, e.g. 'trial epoch' or "
                        "'value mapping'. Defaults to --split-col or 'trial epoch'.")
    p.add_argument("--out", default="notes/figures/gaze_trajectories_epoch_comparison.pdf")
    args = p.parse_args()

    if args.split_col:
        df = pd.read_csv(args.tsv_a, sep="\t", dtype={"subject": str})
        vals = sorted(df[args.split_col].unique())
        if len(vals) != 2:
            p.error(f"--split-col {args.split_col!r} must have exactly 2 unique values, got {vals}")
        label_a = args.label_a or str(vals[0])
        label_b = args.label_b or str(vals[1])
        agg_a = aggregate_df(df[df[args.split_col] == vals[0]], qc_label=f"{args.tsv_a} [{label_a}]")
        agg_b = aggregate_df(df[df[args.split_col] == vals[1]], qc_label=f"{args.tsv_a} [{label_b}]")
        comparison_title = args.comparison_title or args.split_col
    else:
        agg_a, agg_b = load_and_aggregate(args.tsv_a), load_and_aggregate(args.tsv_b)
        label_a = args.label_a or "Group A"
        label_b = args.label_b or "Group B"
        comparison_title = args.comparison_title or "trial epoch"

    grand_a, grand_b = agg_a["grand"], agg_b["grand"]
    n_subj = len(set(agg_a["subjects"]) & set(agg_b["subjects"]))

    diff = compute_difference(grand_a, grand_b)
    diff_grand = diff.groupby("sample_idx")[["x_deg", "y_deg"]].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)

    shared_lim = symmetric_limits(
        np.concatenate([grand_a["x_deg"], grand_b["x_deg"]]),
        np.concatenate([grand_a["y_deg"], grand_b["y_deg"]]))
    for ax, grand, label in ((axes[0], grand_a, label_a), (axes[1], grand_b, label_b)):
        draw_trajectories(ax, grand, lw=1.6, dot_ms=4.5)
        ax.set_xlim(shared_lim)
        ax.set_ylim(shared_lim)
        ax.set_xlabel("Gaze x (deg)")
        ax.set_title(f"Grand average — {label}", fontsize=9)
        sns.despine(ax=ax, offset=3, trim=True)
    axes[0].set_ylabel("Gaze y (deg)")

    ax = axes[2]
    draw_trajectories(ax, diff, lw=0.9, dot_ms=2.5)
    x_g, y_g = smooth_path(diff_grand["x_deg"], diff_grand["y_deg"])
    ax.plot(x_g, y_g, color="0.15", lw=2.2, zorder=5, solid_capstyle="round")
    ax.plot(x_g[-1], y_g[-1], "o", color="0.15", ms=5, zorder=6)
    diff_lim = symmetric_limits(diff["x_deg"].to_numpy(), diff["y_deg"].to_numpy())
    ax.set_xlim(diff_lim)
    ax.set_ylim(diff_lim)
    ax.set_xlabel("Δ Gaze x (deg)")
    ax.set_ylabel("Δ Gaze y (deg)")
    ax.set_title(f"Difference ({label_a} − {label_b})\n"
                 "thin = per orientation, bold = averaged across orientation", fontsize=8)
    sns.despine(ax=ax, offset=3, trim=True)

    suffix = ("; matched on normalized within-epoch time, not wall-clock time — "
              "epochs have different durations)" if not args.split_col else ")")
    fig.suptitle(f"Gaze trajectory shape by {comparison_title} (N={n_subj} subjects in both"
                 f"{suffix}", fontsize=8.5)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
