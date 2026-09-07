"""Cross-condition decoding, split by which mapping was decoded with which.

The pooled version averages the two transfers, which hides whether they behave
alike. They are not symmetric situations: the CDF and inverse-CDF mappings put
the same orientations at opposite ends of the value range, so if the shift
between conditions were a translation of the value code, one direction would
push decoded values up and the other down. Splitting them is the test.

Left two panels: decoded against true for the held-out session, with its own
tuning (matched) and with the other condition's (cross). Right panel: the
difference, one line per direction.

    python -m abstract_values.visualize.plot_cross_condition_by_direction
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
import seaborn as sns

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "lines.linewidth": 1.2, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})

MATCHED_C, CROSS_C = "#2A9D8F", "#C4442B"
DIR_C = {"cdf": "#2A6F97", "inverse_cdf": "#C4442B"}
NICE = {"cdf": "CDF", "inverse_cdf": "Inverse CDF"}
BINS = [0, 8, 14, 20, 26, 32, 38, 45]


def profile(d, col):
    d = d.copy()
    d["bin"] = pd.cut(d["true_value"], BINS).apply(lambda i: i.mid).astype(float)
    return (d.groupby(["bin", "subject"], observed=True)[col].mean()
              .groupby("bin", observed=True).agg(["mean", "sem"]).reset_index())


def main(tsv, out):
    df = pd.read_csv(tsv, sep="\t")
    df["d_dec"] = df.cross_mean - df.matched_mean

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), constrained_layout=True)

    for ax, test in zip(axes[:2], ["cdf", "inverse_cdf"]):
        d = df[df.test_condition == test]
        other = NICE["inverse_cdf" if test == "cdf" else "cdf"]
        for col, c, lab in [("matched_mean", MATCHED_C, "Own tuning"),
                            ("cross_mean", CROSS_C, f"{other} tuning")]:
            per = profile(d, col)
            ax.plot(per["bin"], per["mean"], color=c, marker="o", ms=3)
            ax.fill_between(per["bin"], per["mean"] - per["sem"],
                            per["mean"] + per["sem"], color=c, alpha=.22, lw=0)
            dy = 7 if col == "cross_mean" else -9
            ax.annotate(lab, (per["bin"].iloc[1], per["mean"].iloc[1]),
                        xytext=(2, dy), textcoords="offset points", color=c,
                        fontsize=6.5, va="center", ha="left")
        ax.plot([0, 45], [0, 45], color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=0)
        ax.set_title(f"{NICE[test]} session", fontsize=8, color="0.2")
        ax.set_xlabel("True value (CHF)")
        ax.set_ylabel("Decoded value (CHF)")
        ax.set_xlim(0, 45); ax.set_ylim(0, 45)
        ax.set_xticks([0, 15, 30, 45]); ax.set_yticks([0, 15, 30, 45])

    ax = axes[2]
    for train, d in df.groupby("train_condition"):
        per = profile(d, "d_dec")
        c = DIR_C[train]
        ax.plot(per["bin"], per["mean"], color=c, marker="o", ms=3.5)
        ax.fill_between(per["bin"], per["mean"] - per["sem"],
                        per["mean"] + per["sem"], color=c, alpha=.22, lw=0)
        lab = f"{NICE[train]} tuning\non {NICE['inverse_cdf' if train == 'cdf' else 'cdf']} data"
        ax.annotate(lab, (per["bin"].iloc[-1], per["mean"].iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", color=c,
                    fontsize=6.5, va="center", ha="left")
    ax.axhline(0, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.annotate("No cost", (2, 0), xytext=(0, 3), textcoords="offset points",
                fontsize=6.5, color="0.45", ha="left")
    ax.set_xlabel("True value (CHF)")
    ax.set_ylabel("Cross − matched decoded (CHF)")
    ax.set_xlim(0, 62)
    ax.set_xticks([0, 15, 30, 45])

    sns.despine(fig=fig, offset=4, trim=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsv", default="notes/data/cross_condition_decoding.tsv")
    p.add_argument("--out",
                   default="notes/figures/cross_condition_by_direction.pdf")
    a = p.parse_args()
    main(Path(a.tsv), Path(a.out))
