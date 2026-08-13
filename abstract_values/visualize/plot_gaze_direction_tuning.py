#!/usr/bin/env python3
"""Gaze-direction tuning curves: endpoint direction angle and displacement
magnitude as a function of grating orientation, for one or two groups of
trials (default: gabor presentation vs. response_bar epoch).

Distils the "fan" pattern visible in plot_gaze_trajectories.py's grand-
average panel into an explicit function of orientation — easier to read
the *shape* of the orientation -> gaze-direction relationship off a
tuning curve than off overlapping 2D paths.

Endpoint = mean of the trajectory's last END_SAMPLES resampled points
(robust tail estimate, same idea as ORIGIN_SAMPLES in
plot_gaze_trajectories.recenter_trials). angle_deg = atan2(y, x) of that
endpoint relative to the recentred trial-onset origin (0 = rightward,
90 = up, ±180 = left, -90 = down). magnitude_deg = its length.

Two modes, same as compare_gaze_epochs.py:
  1. Two files (default): different trial epochs.
  2. One file, split by a column (--split-col), e.g. `mapping`.

Circular-data caveat: connecting per-orientation points into a line uses
np.unwrap on the angle sequence (sorted by orientation) so a genuine
smooth drift doesn't get chopped into a fake +/-360 deg jump — but this
can misrepresent a *real* large jump between adjacent orientations as
smooth. Per-subject points are always shown unconnected (scatter only)
to avoid compounding that assumption across noisier individual data.

Usage (local, no cluster needed):
    python -m abstract_values.visualize.plot_gaze_direction_tuning \\
        --tsv-a notes/data/gaze_trajectories_gabor.tsv --label-a "Gabor presentation" \\
        --tsv-b notes/data/gaze_trajectories_all.tsv   --label-b "Response bar" \\
        --out notes/figures/gaze_direction_tuning.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abstract_values.visualize.plot_gaze_trajectories import aggregate_df, load_and_aggregate

END_SAMPLES = 3
PALETTE = ["#3B5BA5", "#C44E52"]  # blue, red — hue is free here (x-axis is orientation)


def endpoint_stats(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """One row per (group_cols..., orientation): endpoint angle (deg,
    atan2 convention) and magnitude (deg), from the mean of each trial's
    (or each pre-aggregated group's) last END_SAMPLES resampled points."""
    n_max = df["sample_idx"].max()
    end = df[df["sample_idx"] > n_max - END_SAMPLES]
    end = end.groupby(group_cols + ["orientation"])[["x_deg", "y_deg"]].mean().reset_index()
    end["angle_deg"] = np.degrees(np.arctan2(end["y_deg"], end["x_deg"]))
    end["magnitude_deg"] = np.hypot(end["x_deg"], end["y_deg"])
    return end


def unwrap_by_orientation(sub: pd.DataFrame) -> np.ndarray:
    sub = sub.sort_values("orientation")
    return np.degrees(np.unwrap(np.radians(sub["angle_deg"].to_numpy())))


def plot_group(ax_angle, ax_mag, agg: dict, label: str, color: str):
    grand_end = endpoint_stats(agg["grand"], group_cols=[]).sort_values("orientation")
    subj_end = endpoint_stats(agg["per_subj"], group_cols=["subject"])

    ax_angle.scatter(subj_end["orientation"], subj_end["angle_deg"], s=6, color=color,
                      alpha=0.25, linewidth=0, zorder=2)
    angle_y = unwrap_by_orientation(grand_end)
    ax_angle.plot(grand_end["orientation"], angle_y, color=color, lw=2, marker="o", ms=4, zorder=3)
    ax_angle.annotate(label, xy=(grand_end["orientation"].iloc[-1], angle_y[-1]),
                      xytext=(6, 0), textcoords="offset points", color=color,
                      fontsize=7.5, ha="left", va="center", fontweight="bold")

    ax_mag.scatter(subj_end["orientation"], subj_end["magnitude_deg"], s=6, color=color,
                   alpha=0.25, linewidth=0, zorder=2)
    ax_mag.plot(grand_end["orientation"], grand_end["magnitude_deg"],
               color=color, lw=2, marker="o", ms=4, zorder=3)
    ax_mag.annotate(label, xy=(grand_end["orientation"].iloc[-1], grand_end["magnitude_deg"].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", color=color,
                    fontsize=7.5, ha="left", va="center", fontweight="bold")


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
    p.add_argument("--title", default=None)
    p.add_argument("--out", default="notes/figures/gaze_direction_tuning.pdf")
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
        title = args.title or f"Gaze-direction tuning by {args.split_col}"
    else:
        agg_a, agg_b = load_and_aggregate(args.tsv_a), load_and_aggregate(args.tsv_b)
        label_a, label_b = args.label_a or "Group A", args.label_b or "Group B"
        title = args.title or "Gaze-direction tuning by trial epoch"

    n_subj = len(set(agg_a["subjects"]) & set(agg_b["subjects"]))

    fig, (ax_angle, ax_mag) = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)

    plot_group(ax_angle, ax_mag, agg_a, label_a, PALETTE[0])
    plot_group(ax_angle, ax_mag, agg_b, label_b, PALETTE[1])

    ori_ticks = [0, 45, 90, 135, 180]
    ax_angle.set_xticks(ori_ticks)
    ax_angle.set_xlabel("Grating orientation (deg)")
    ax_angle.set_ylabel("Gaze direction (deg)")
    ax_angle.set_yticks([-180, -90, 0, 90, 180])
    for y, ref in ((0, "Right"), (90, "Up"), (-90, "Down"), (180, "Left")):
        ax_angle.axhline(y, color="0.75", lw=0.6, ls="--", zorder=0)
        ax_angle.text(182, y, ref, fontsize=6.5, color="0.4", va="center", ha="left")
    ax_angle.set_xlim(0, 205)
    ax_angle.set_ylim(-200, 200)
    ax_angle.set_title("Endpoint direction", fontsize=9)
    sns.despine(ax=ax_angle, offset=3, trim=True)

    ax_mag.set_xticks(ori_ticks)
    ax_mag.set_xlim(0, 205)
    ax_mag.set_xlabel("Grating orientation (deg)")
    ax_mag.set_ylabel("Endpoint displacement (deg)")
    ax_mag.set_title("Endpoint magnitude", fontsize=9)
    sns.despine(ax=ax_mag, offset=3, trim=True)

    fig.suptitle(f"{title} (N={n_subj} subjects in both; dots = individual subjects, "
                 "line = grand average)", fontsize=8.5)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
