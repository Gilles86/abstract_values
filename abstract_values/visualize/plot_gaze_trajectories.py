#!/usr/bin/env python3
"""Per-trial gaze trajectories during the response_bar (estimation) phase,
averaged per orientation, one panel per subject.

Input: notes/data/gaze_trajectories_all.tsv, produced by
abstract_values.eyetracking.extract_gaze_trajectories (run on
sciencecluster — needs the Linux edf2asc binary, not available locally)
and rsynced back. Columns: subject, session, mapping, run, trial_nr,
orientation, sample_idx (0..N_RESAMPLE-1, normalized time within the
response_bar window), x_deg, y_deg (visual angle, screen-centre-relative,
already converted from EyeLink pixels using each run's own
expsettings.yml geometry — see that script's docstring).

For each (subject, orientation), trials are averaged sample_idx-wise (no
per-trial recentring — x_deg/y_deg are screen-centre-relative already, so
the shared "+" at (0, 0) marks the fixation/screen-centre reference, not
a per-trial computed origin). A filled dot marks the trajectory's end
(mean gaze position at feedback onset).

Orientation is cyclic (0 deg == 180 deg, gratings are axis-symmetric) so
colour uses the 'hsv' colormap wrapped at 180 deg, matching the
convention in visualize_mean_orientation_fsaverage.py (ORI_CMAP = 'hsv').

xlim/ylim are shared across every subject panel (computed from the full
across-subject data extent, deliberately not robustified — an outlier
subject with a large offset is exactly the kind of thing this plot is
meant to surface, e.g. a mis-set eye tracker or a real perceptual/motor
bias).

Usage (local, plain matplotlib/seaborn — no cluster needed):
    python -m abstract_values.visualize.plot_gaze_trajectories \\
        --tsv notes/data/gaze_trajectories_all.tsv \\
        --out notes/figures/gaze_trajectories.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ORI_CMAP = "hsv"       # cyclic — orientation wraps at 180 deg
ORI_PERIOD = 180.0
MIN_TRIALS = 3          # minimum trials per (subject, orientation) to plot

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 150,
    "savefig.dpi": 300,
})
sns.set_context("paper")


def orientation_color(orientation: float) -> tuple:
    cmap = plt.get_cmap(ORI_CMAP)
    return cmap((orientation % ORI_PERIOD) / ORI_PERIOD)


def subject_sort_key(s: str):
    return (0, int(s)) if s.isdigit() else (1, s)


def plot_orientation_wheel(ax):
    """Compact polar swatch: colour = orientation (0-180, mirrored at 360)."""
    n = 360
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    orientation = np.degrees(theta) % ORI_PERIOD
    colors = plt.get_cmap(ORI_CMAP)(orientation / ORI_PERIOD)
    ax.axis("off")
    ax_polar = ax.figure.add_axes(ax.get_position(), projection="polar")
    ax_polar.bar(theta, np.ones(n), width=2 * np.pi / n, bottom=0.6,
                 color=colors, linewidth=0)
    ax_polar.set_ylim(0, 1.6)
    ax_polar.set_xticks([])
    ax_polar.set_yticks([])
    ax_polar.spines["polar"].set_visible(False)
    ax_polar.set_title("Orientation", fontsize=8, pad=2)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tsv", default="notes/data/gaze_trajectories_all.tsv")
    p.add_argument("--out", default="notes/figures/gaze_trajectories.pdf")
    p.add_argument("--ncols", type=int, default=6)
    args = p.parse_args()

    df = pd.read_csv(args.tsv, sep="\t", dtype={"subject": str})

    n_trials = (df.drop_duplicates(["subject", "session", "mapping", "run", "trial_nr", "orientation"])
                  .groupby(["subject", "orientation"]).size())
    agg = (df.groupby(["subject", "orientation", "sample_idx"])[["x_deg", "y_deg"]]
             .mean().reset_index())
    agg = agg.set_index(["subject", "orientation"])
    keep = n_trials[n_trials >= MIN_TRIALS].index
    agg = agg.loc[agg.index.isin(keep)].reset_index()

    pad_frac = 0.08
    xr = agg["x_deg"].max() - agg["x_deg"].min()
    yr = agg["y_deg"].max() - agg["y_deg"].min()
    xlim = (agg["x_deg"].min() - pad_frac * xr, agg["x_deg"].max() + pad_frac * xr)
    ylim = (agg["y_deg"].min() - pad_frac * yr, agg["y_deg"].max() + pad_frac * yr)

    subjects = sorted(agg["subject"].unique(), key=subject_sort_key)
    ncols = args.ncols
    nrows = int(np.ceil((len(subjects) + 1) / ncols))  # +1 slot for the orientation legend

    panel_size = 1.7
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * panel_size, nrows * panel_size),
                              constrained_layout=True)
    axes = np.atleast_2d(axes)

    col_bottom_row = {}
    for i in range(len(subjects)):
        r, c = divmod(i, ncols)
        col_bottom_row[c] = max(col_bottom_row.get(c, -1), r)

    data_axes = []
    for i, subject in enumerate(subjects):
        ax = axes.flat[i]
        data_axes.append(ax)
        row, col = divmod(i, ncols)
        d = agg[agg["subject"] == subject]
        for orientation, g in d.groupby("orientation"):
            g = g.sort_values("sample_idx")
            color = orientation_color(orientation)
            ax.plot(g["x_deg"], g["y_deg"], color=color, lw=0.9, alpha=0.85, zorder=2)
            ax.plot(g["x_deg"].iloc[-1], g["y_deg"].iloc[-1], "o", color=color,
                     ms=2.5, zorder=3, markeredgewidth=0)
        ax.plot(0, 0, "+", color="0.15", ms=5, mew=1, zorder=4)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        label = f"sub-{subject}" if subject.isdigit() else subject
        ax.set_title(label, fontsize=8)
        if col == 0:
            ax.set_ylabel("Gaze y (deg)")
        else:
            ax.set_yticklabels([])
        if row == col_bottom_row[col]:
            ax.set_xlabel("Gaze x (deg)")
        else:
            ax.set_xticklabels([])

    plot_orientation_wheel(axes.flat[len(subjects)])
    for j in range(len(subjects) + 1, len(axes.flat)):
        axes.flat[j].axis("off")

    for ax in data_axes:
        sns.despine(ax=ax, offset=3, trim=True)
    fig.suptitle("Gaze trajectories during value estimation, by orientation "
                 f"(N={len(subjects)} subjects, mean of ≥{MIN_TRIALS} trials/orientation)",
                 fontsize=9, y=1.01)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
