#!/usr/bin/env python3
"""Per-trial gaze trajectories during the response_bar (estimation) phase,
averaged per orientation: one large grand-average panel plus one small
panel per subject.

Input: notes/data/gaze_trajectories_all.tsv, produced by
abstract_values.eyetracking.extract_gaze_trajectories (run on
sciencecluster — needs the Linux edf2asc binary, not available locally)
and rsynced back. Columns: subject, session, mapping, run, trial_nr,
orientation, sample_idx (0..N_RESAMPLE-1, normalized time within the
response_bar window), x_deg, y_deg (visual angle, screen-centre-relative,
converted from EyeLink pixels using each run's own expsettings.yml
geometry — see that script's docstring).

Per-trial recentring
---------------------
x_deg/y_deg as extracted are relative to *screen centre*, which is a
geometric assumption, not where gaze actually was. EyeLink calibration
drifts across runs/sessions, and gaze at response_bar onset needn't sit
exactly on the pixel screen centre anyway — so plotting raw
screen-centre-relative trajectories conflates real within-trial gaze
movement with between-run calibration offset (this is why the first cut
of this figure looked bad: panels didn't line up at a common start, and
one subject's calibration offset blew out the shared axis for everyone).

Fix: recentre every trial on its OWN first resampled sample (sample_idx
0, i.e. gaze position right as the response_bar phase begins — the
participant was on/near the preceding fixation cross a moment before).
After recentring, "(0, 0)" means "gaze position when the response bar
appeared" for every trial, subject, and session alike, and the plotted
paths show only genuine within-trial drift.

Axes are NOT shared across subject panels — deliberately reverted from
an earlier version. A shared axis let one high-amplitude subject flatten
every other panel to a sliver; per-panel axes (still symmetric around
zero and equal-aspect) show each subject's own trajectory shape clearly.
Cross-subject amplitude comparison, if needed, is what the grand-average
panel plus notes/data/gaze_trajectories_all.tsv are for.

Orientation is cyclic (0 deg == 180 deg, gratings are axis-symmetric) so
colour uses the 'hsv' colormap wrapped at 180 deg, matching the
convention in visualize_mean_orientation_fsaverage.py (ORI_CMAP = 'hsv').

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

ORI_CMAP = "hsv"        # cyclic — orientation wraps at 180 deg
ORI_PERIOD = 180.0
MIN_TRIALS = 3           # minimum trials per (subject, orientation) to plot
TRIAL_KEYS = ["subject", "session", "mapping", "run", "trial_nr"]

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


def recenter_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract each trial's own sample_idx==0 position from the whole trial."""
    origin = (df.loc[df["sample_idx"] == 0, TRIAL_KEYS + ["x_deg", "y_deg"]]
                .rename(columns={"x_deg": "x0", "y_deg": "y0"}))
    df = df.merge(origin, on=TRIAL_KEYS, how="left")
    df["x_deg"] = df["x_deg"] - df["x0"]
    df["y_deg"] = df["y_deg"] - df["y0"]
    return df.drop(columns=["x0", "y0"])


def symmetric_limits(x: np.ndarray, y: np.ndarray, pad_frac: float = 0.15) -> tuple:
    m = max(np.abs(x).max(), np.abs(y).max()) * (1 + pad_frac)
    m = max(m, 0.1)  # guard against degenerate all-zero panels
    return (-m, m)


def smooth_path(x: pd.Series, y: pd.Series, window: int = 3) -> tuple:
    """Light centred rolling mean, display-only — the underlying per-sample
    mean is noisy (SEM comparable to point-to-point steps even pooling
    ~400 trials), so unsmoothed paths read as jagged zig-zags that
    overstate how much the true average trajectory actually wiggles."""
    x_s = x.rolling(window, center=True, min_periods=1).mean().to_numpy()
    y_s = y.rolling(window, center=True, min_periods=1).mean().to_numpy()
    return x_s, y_s


def draw_trajectories(ax, d: pd.DataFrame, lw: float, dot_ms: float):
    for orientation, g in d.groupby("orientation"):
        g = g.sort_values("sample_idx")
        x, y = smooth_path(g["x_deg"], g["y_deg"])
        color = orientation_color(orientation)
        ax.plot(x, y, color=color, lw=lw, alpha=0.85, zorder=2)
        ax.plot(x[-1], y[-1], "o", color=color, ms=dot_ms, zorder=3, markeredgewidth=0)
    ax.plot(0, 0, "+", color="0.15", ms=6, mew=1.2, zorder=4)
    ax.set_aspect("equal", adjustable="box")


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
    ax_polar.set_title("Orientation", fontsize=9, pad=2)


def annotate_onset_offset(ax, offset_xy: tuple):
    ax.annotate(
        "Response-bar onset\n(gaze recentred here)",
        xy=(0, 0), xytext=(0.35, 0.92), textcoords="axes fraction",
        fontsize=8, ha="left", va="top",
        arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=70",
                         color="0.2", lw=1.1, mutation_scale=11, shrinkA=3, shrinkB=8,
                         relpos=(0.0, 0.0)),
    )
    ax.annotate(
        "Feedback onset\n(trial end)",
        xy=offset_xy, xytext=(0.05, 0.08), textcoords="axes fraction",
        fontsize=8, ha="left", va="bottom",
        arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=-70",
                         color="0.2", lw=1.1, mutation_scale=11, shrinkA=3, shrinkB=8,
                         relpos=(0.0, 1.0)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tsv", default="notes/data/gaze_trajectories_all.tsv")
    p.add_argument("--out", default="notes/figures/gaze_trajectories.pdf")
    p.add_argument("--ncols", type=int, default=6)
    args = p.parse_args()

    df = pd.read_csv(args.tsv, sep="\t", dtype={"subject": str})
    df = recenter_trials(df)

    n_trials = (df.drop_duplicates(TRIAL_KEYS + ["orientation"])
                  .groupby(["subject", "orientation"]).size())
    keep = n_trials[n_trials >= MIN_TRIALS].index

    per_subj = (df.groupby(["subject", "orientation", "sample_idx"])[["x_deg", "y_deg"]]
                  .mean().reset_index().set_index(["subject", "orientation"]))
    per_subj = per_subj.loc[per_subj.index.isin(keep)].reset_index()

    grand = (df.groupby(["orientation", "sample_idx"])[["x_deg", "y_deg"]]
               .mean().reset_index())

    subjects = sorted(per_subj["subject"].unique(), key=subject_sort_key)
    ncols = args.ncols
    nrows_subj = int(np.ceil(len(subjects) / ncols))
    top_rows = 3  # grand-average panel + wheel occupy this many grid rows

    panel_size = 1.7
    fig_w = ncols * panel_size
    fig_h = (top_rows + nrows_subj) * panel_size
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(top_rows + nrows_subj, ncols)

    grand_cols = max(3, ncols // 2)
    ax_grand = fig.add_subplot(gs[0:top_rows, 0:grand_cols])
    draw_trajectories(ax_grand, grand, lw=1.8, dot_ms=5)
    lim = symmetric_limits(grand["x_deg"].to_numpy(), grand["y_deg"].to_numpy())
    ax_grand.set_xlim(lim)
    ax_grand.set_ylim(lim)
    ax_grand.set_xlabel("Gaze x (deg)")
    ax_grand.set_ylabel("Gaze y (deg)")
    ax_grand.set_title(f"Grand average (N={len(subjects)} subjects, "
                        f"mean of {n_trials.loc[n_trials.index.isin(keep)].sum()} trials)",
                        fontsize=9)
    end_xy = (grand.sort_values("sample_idx").groupby("orientation").tail(1)[["x_deg", "y_deg"]].mean())
    annotate_onset_offset(ax_grand, (end_xy["x_deg"], end_xy["y_deg"]))
    sns.despine(ax=ax_grand, offset=3, trim=True)

    ax_wheel = fig.add_subplot(gs[0:top_rows, grand_cols:grand_cols + 2])
    plot_orientation_wheel(ax_wheel)

    col_bottom_row = {}
    for i in range(len(subjects)):
        r, c = divmod(i, ncols)
        col_bottom_row[c] = max(col_bottom_row.get(c, -1), r)

    for i, subject in enumerate(subjects):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[top_rows + row, col])
        d = per_subj[per_subj["subject"] == subject]
        draw_trajectories(ax, d, lw=0.9, dot_ms=2.5)
        lim = symmetric_limits(d["x_deg"].to_numpy(), d["y_deg"].to_numpy())
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        label = f"sub-{subject}" if subject.isdigit() else subject
        ax.set_title(label, fontsize=8)
        if col == 0:
            ax.set_ylabel("Gaze y (deg)")
        if row == col_bottom_row[col]:
            ax.set_xlabel("Gaze x (deg)")
        n_ticks = 3
        ax.set_xticks(np.linspace(*lim, n_ticks).round(1))
        ax.set_yticks(np.linspace(*lim, n_ticks).round(1))
        sns.despine(ax=ax, offset=2, trim=True)

    fig.suptitle("Gaze trajectories during value estimation, by orientation "
                 "(per-trial recentred on response-bar onset; "
                 f"per-panel axes, mean of ≥{MIN_TRIALS} trials/orientation)",
                 fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
