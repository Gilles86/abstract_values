#!/usr/bin/env python3
"""Microsaccade direction histogram during the pre-stimulus fixation
window (trial onset -> gabor onset), compared between two conditions
(default: cdf vs inverse_cdf mapping) — this is the window BEFORE the
grating orientation is shown, so there is nothing stimulus-driven to
condition on; the only thing that can differ between trials here is
which mapping block the subject is currently in.

Input: notes/data/microsaccades_pregabor.tsv from
abstract_values.eyetracking.extract_microsaccades (run on sciencecluster
— raw full-sampling-rate detection, not available locally). Filters to
amplitude < --max-amplitude (default 1 deg, the standard microsaccade/
saccade cutoff in the literature) before any comparison — larger events
in that TSV are regular saccades the detector also picked up, not
microsaccades.

Statistics: same "difference of the mean directions, not the mean of the
differences" approach as plot_gaze_direction_tuning.py (imported, not
reimplemented) — each subject's own circular-mean microsaccade direction
per condition is a well-defined number; the paired per-subject angular
difference is what gets averaged (circular mean) and bootstrap-CI'd,
rather than pooling all microsaccades into one big vector per condition
and risking the same near-zero-vector pathology that motivated that
rewrite.

Usage (local, no cluster needed):
    python -m abstract_values.visualize.plot_microsaccade_directions \\
        --tsv notes/data/microsaccades_pregabor.tsv \\
        --out notes/figures/microsaccade_directions_mapping.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abstract_values.visualize.plot_gaze_direction_tuning import circular_diff_stats, linear_diff_stats

MAX_AMPLITUDE_DEG = 1.0   # standard micro- vs regular-saccade cutoff
MIN_EVENTS_PER_CELL = 5   # minimum microsaccades for a subject x condition mean direction to count
N_BINS = 24               # 15 deg polar-histogram bins
PALETTE = ["#3B5BA5", "#C44E52"]


def circular_mean_deg(directions_deg: np.ndarray) -> float:
    rad = np.radians(directions_deg)
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())))


def per_subject_condition_stats(df: pd.DataFrame, cond_col: str) -> pd.DataFrame:
    rows = []
    for (subj, cond), g in df.groupby(["subject", cond_col]):
        if len(g) < MIN_EVENTS_PER_CELL:
            continue
        rows.append(dict(subject=subj, condition=cond, n=len(g),
                          mean_dir_deg=circular_mean_deg(g["direction_deg"].to_numpy()),
                          mean_amp_deg=g["amplitude_deg"].mean()))
    return pd.DataFrame(rows)


def polar_histogram(ax, directions_deg: np.ndarray, color: str, label: str, n_bins: int = N_BINS):
    edges = np.linspace(-180, 180, n_bins + 1)
    counts, _ = np.histogram(directions_deg, bins=edges)
    density = counts / counts.sum()  # proportion, comparable across conditions with different N
    theta = np.radians((edges[:-1] + edges[1:]) / 2)
    width = np.radians(edges[1] - edges[0]) * 0.85
    ax.bar(theta, density, width=width, color=color, alpha=0.55, edgecolor="none",
          label=f"{label} (n={len(directions_deg)})", zorder=2)
    mean_dir = np.radians(circular_mean_deg(directions_deg))
    ax.plot([mean_dir, mean_dir], [0, density.max() * 1.15], color=color, lw=2.2, zorder=3)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tsv", default="notes/data/microsaccades_pregabor.tsv")
    p.add_argument("--cond-col", default="mapping")
    p.add_argument("--label-a", default="cdf")
    p.add_argument("--label-b", default="inverse_cdf")
    p.add_argument("--max-amplitude", type=float, default=MAX_AMPLITUDE_DEG)
    p.add_argument("--out", default="notes/figures/microsaccade_directions_mapping.pdf")
    args = p.parse_args()

    df = pd.read_csv(args.tsv, sep="\t", dtype={"subject": str})
    n_before = len(df)
    df = df[df["amplitude_deg"] < args.max_amplitude]
    print(f"Kept {len(df)}/{n_before} events as microsaccades (amplitude < {args.max_amplitude} deg); "
          f"{n_before - len(df)} larger events excluded (regular saccades)")

    per_subj = per_subject_condition_stats(df, args.cond_col)
    wide_dir = per_subj.pivot(index="subject", columns="condition", values="mean_dir_deg").dropna()
    wide_amp = per_subj.pivot(index="subject", columns="condition", values="mean_amp_deg").dropna()
    n_subj = len(wide_dir)

    delta_angle = (((wide_dir[args.label_a] - wide_dir[args.label_b]) + 180) % 360 - 180).to_numpy()
    delta_amp = (wide_amp[args.label_a] - wide_amp[args.label_b]).to_numpy()
    mean_delta_deg, delta_lo, delta_hi, R, p_dir = circular_diff_stats(delta_angle)
    mean_delta_amp, amp_lo, amp_hi, p_amp = linear_diff_stats(delta_amp)

    print(f"Direction: N={n_subj} subjects, mean difference ({args.label_a} - {args.label_b}) = "
          f"{mean_delta_deg:.1f} deg [{delta_lo:.1f}, {delta_hi:.1f}], R={R:.2f}, p={p_dir:.3f}")
    print(f"Amplitude: mean difference = {mean_delta_amp:+.3f} deg "
          f"[{amp_lo:+.3f}, {amp_hi:+.3f}], p={p_amp:.3f}")

    fig = plt.figure(figsize=(9, 4.2), constrained_layout=True)
    ax_hist = fig.add_subplot(1, 2, 1, projection="polar")
    ax_hist.set_theta_zero_location("E")
    ax_hist.set_theta_direction(1)
    dir_a = df.loc[df[args.cond_col] == args.label_a, "direction_deg"].to_numpy()
    dir_b = df.loc[df[args.cond_col] == args.label_b, "direction_deg"].to_numpy()
    polar_histogram(ax_hist, dir_a, PALETTE[0], args.label_a)
    polar_histogram(ax_hist, dir_b, PALETTE[1], args.label_b)
    ax_hist.set_xticks(np.radians([0, 90, 180, 270]))
    ax_hist.set_xticklabels(["Right", "Up", "Left", "Down"], fontsize=8)
    ax_hist.set_yticklabels([])
    ax_hist.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7.5, frameon=False)
    ax_hist.set_title("Microsaccade directions\n(pooled trials/orientations; radial line = group mean)",
                      fontsize=9)

    ax_scatter = fig.add_subplot(1, 2, 2)
    for i, (subj, row) in enumerate(wide_dir.iterrows()):
        ax_scatter.plot([0, 1], [row[args.label_a], row[args.label_b]], color="0.75", lw=0.6, zorder=1)
    ax_scatter.scatter(np.zeros(n_subj), wide_dir[args.label_a], color=PALETTE[0], s=18, zorder=2)
    ax_scatter.scatter(np.ones(n_subj), wide_dir[args.label_b], color=PALETTE[1], s=18, zorder=2)
    ax_scatter.set_xlim(-0.4, 1.4)
    ax_scatter.set_xticks([0, 1])
    ax_scatter.set_xticklabels([args.label_a, args.label_b])
    ax_scatter.set_ylim(-200, 200)
    ax_scatter.set_yticks([-180, -90, 0, 90, 180])
    ax_scatter.set_ylabel("Subject's own mean microsaccade direction (deg)")
    sig_txt = "significant" if p_dir < 0.05 else "not significant"
    ax_scatter.set_title(f"Per-subject mean direction (N={n_subj} paired)\n"
                         f"Difference: {mean_delta_deg:+.1f} deg [{delta_lo:+.1f}, {delta_hi:+.1f}], "
                         f"p={p_dir:.3f} ({sig_txt})", fontsize=8.5)
    sns.despine(ax=ax_scatter, offset=3, trim=True)

    fig.suptitle(f"Pre-stimulus microsaccade direction by {args.cond_col} "
                 f"(amplitude < {args.max_amplitude} deg, >= {MIN_EVENTS_PER_CELL} events/subject/condition)",
                 fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
