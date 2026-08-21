"""Does head motion explain a subject's decoding quality?

One panel per (decoded quantity × ROI): each point is a subject, x = mean
framewise displacement, y = decoding correlation. A highlighted subject is
called out so a single restless participant can be read against the cohort
trend rather than against an absolute threshold.

Inputs are the two summary TSVs, both small enough to work with locally:

    notes/data/motion_summary.tsv                  <- check_motion.py --summary-tsv
    notes/data/decoding_quality_<...>.tsv          <- decoding_quality_scatter.py

    python -m abstract_values.visualize.motion_vs_decoding \
        --motion-tsv notes/data/motion_summary.tsv \
        --decoding-tsv notes/data/decoding_quality_spherical_nv250.tsv \
        --highlight 28 --out notes/figures/motion_vs_decoding.pdf

The reported r is Spearman across subjects — with ~25 points and one subject
far out in x, a Pearson coefficient would largely be reporting that subject's
leverage. The highlighted subject is included in the fit (excluding the point
you are asking about would beg the question); the annotation says so.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})

HIGHLIGHT_COLOUR = "#E76F51"
COHORT_COLOUR = "#3B5BA5"

PANELS = [
    ("gabor", "BensonV1", "Orientation · V1"),
    ("gabor", "NPCr", "Orientation · NPCr"),
    ("value", "BensonV1", "Value · V1"),
    ("value", "NPCr", "Value · NPCr"),
]


def run(motion_tsv, decoding_tsv, highlight, out, study_only=True):
    motion = pd.read_csv(motion_tsv, sep="\t", dtype={"subject": str})
    dec = pd.read_csv(decoding_tsv, sep="\t", dtype={"subject": str})

    fd = motion.groupby("subject")["mean_fd"].mean().rename("mean_fd")
    df = dec.merge(fd, left_on="subject", right_index=True, how="inner")
    if study_only:
        df = df[[s.isdigit() for s in df["subject"]]]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
    for ax, (quantity, roi, title) in zip(axes, PANELS):
        sub = df[(df["quantity"] == quantity) & (df["roi"] == roi)].dropna(
            subset=["mean_fd", "r"])
        if sub.empty:
            ax.set_axis_off()
            continue
        is_hl = (sub["subject"] == str(highlight)).to_numpy()
        ax.scatter(sub.loc[~is_hl, "mean_fd"], sub.loc[~is_hl, "r"],
                   s=26, color=COHORT_COLOUR, alpha=0.75, linewidths=0)
        if is_hl.any():
            ax.scatter(sub.loc[is_hl, "mean_fd"], sub.loc[is_hl, "r"],
                       s=52, color=HIGHLIGHT_COLOUR, linewidths=0, zorder=4)
            for _, row in sub[is_hl].iterrows():
                ax.annotate(f"sub-{row['subject']}",
                            (row["mean_fd"], row["r"]),
                            textcoords="offset points", xytext=(-6, 6),
                            ha="right", fontsize=8, color=HIGHLIGHT_COLOUR)

        rho, p = spearmanr(sub["mean_fd"], sub["r"])
        ax.axhline(0, color="0.75", lw=0.8, ls="--", zorder=0)
        ax.set_title(title, fontsize=9, color="0.2")
        ax.set_xlabel("Mean FD (mm)")
        ax.set_ylabel("Decoding correlation")
        ax.text(0.95, 0.95, rf"Spearman $\rho$ = {rho:.2f}" + "\n" + f"p = {p:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.5))

    fig.suptitle("Head motion vs decoding quality (one point per subject; "
                 "the highlighted subject is included in the fit)",
                 fontsize=10, y=1.08)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion-tsv", default="notes/data/motion_summary.tsv")
    p.add_argument("--decoding-tsv", required=True)
    p.add_argument("--highlight", default=None)
    p.add_argument("--out", default="notes/figures/motion_vs_decoding.pdf")
    args = p.parse_args()
    run(args.motion_tsv, args.decoding_tsv, args.highlight, args.out)


if __name__ == "__main__":
    main()
