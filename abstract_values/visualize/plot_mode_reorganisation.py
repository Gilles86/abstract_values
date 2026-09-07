"""Does the value code reorganise between the two mappings, or just move?

Three views of the same question, from encoding parameters and from decoding:

  a  preferred value in one condition against the other, per voxel
  b  the shift, against where the voxel sat in the CDF condition
  c  cross minus matched decoded value, against the true value

A translation would show as a constant offset, an inversion as a negative
slope, and neither is what happens: preferred values stay positively related
across conditions but contract sharply toward the middle of the range.

IMPORTANT -- contraction is also exactly what measurement noise produces. If
mode_cdf and mode_inv are noisy readings of one stable preferred value, the
regression slope is attenuated to the reliability, and here slope (0.34) and
correlation (0.32) are close enough to be that. Calling this reorganisation
needs a within-condition reliability benchmark (split-half inside a session)
to compare the across-condition correlation against; that fit does not exist
yet. The panels show the effect, not its interpretation.

    python -m abstract_values.visualize.plot_mode_reorganisation
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

C_MODE, C_DEC = "#6A4C93", "#C4442B"


def main(modes_tsv, cross_tsv, out):
    m = pd.read_csv(modes_tsv, sep="\t")
    x = pd.read_csv(cross_tsv, sep="\t")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)

    # a — preferred value, condition against condition
    ax = axes[0]
    ax.scatter(m.mode_cdf, m.mode_inv, s=4, alpha=.25, color=C_MODE,
               linewidths=0, rasterized=True)
    lim = (0, 45)
    ax.plot(lim, lim, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=1)
    sl = m.groupby("subject").apply(lambda d: pd.Series(
        dict(zip(("slope", "inter"), np.polyfit(d.mode_cdf, d.mode_inv, 1)))),
        include_groups=False)
    xs = np.linspace(*lim, 50)
    ax.plot(xs, sl["inter"].mean() + sl["slope"].mean() * xs, color=C_MODE, lw=1.6)
    ax.annotate(f"Slope {sl['slope'].mean():.2f}", (34, sl['inter'].mean() + sl['slope'].mean() * 34),
                xytext=(0, -12), textcoords="offset points", fontsize=6.5,
                color=C_MODE, ha="center")
    ax.annotate("Identity", (40, 40), xytext=(-2, 2), textcoords="offset points",
                fontsize=6.5, color="0.45", ha="right")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xticks([0, 15, 30, 45]); ax.set_yticks([0, 15, 30, 45])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Preferred value, CDF (CHF)")
    ax.set_ylabel("Preferred value, inv. CDF (CHF)")

    # b — the shift, against where the voxel started
    ax = axes[1]
    m["bin"] = pd.cut(m.mode_cdf, [0, 10, 18, 26, 34, 45]).apply(lambda i: i.mid).astype(float)
    per = (m.groupby(["bin", "subject"], observed=True)["dshift"].mean()
             .groupby("bin", observed=True).agg(["mean", "sem"]).reset_index())
    ax.plot(per["bin"], per["mean"], color=C_MODE, marker="o", ms=3.5)
    ax.fill_between(per["bin"], per["mean"] - per["sem"], per["mean"] + per["sem"],
                    color=C_MODE, alpha=.22, lw=0)
    ax.axhline(0, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.annotate("No shift", (44, 0), xytext=(0, 2), textcoords="offset points",
                fontsize=6.5, color="0.45", ha="right")
    ax.set_xlabel("Preferred value, CDF (CHF)")
    ax.set_ylabel("Shift to inv. CDF (CHF)")
    ax.set_xticks([0, 15, 30, 45])

    # c — the decoder's view of the same swap
    ax = axes[2]
    x["d_dec"] = x.cross_mean - x.matched_mean
    x["bin"] = pd.cut(x.true_value, [0, 8, 14, 20, 26, 32, 38, 45]).apply(lambda i: i.mid).astype(float)
    per = (x.groupby(["bin", "subject"], observed=True)["d_dec"].mean()
             .groupby("bin", observed=True).agg(["mean", "sem"]).reset_index())
    ax.plot(per["bin"], per["mean"], color=C_DEC, marker="o", ms=3.5)
    ax.fill_between(per["bin"], per["mean"] - per["sem"], per["mean"] + per["sem"],
                    color=C_DEC, alpha=.22, lw=0)
    ax.axhline(0, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.annotate("No cost", (44, 0), xytext=(0, 2), textcoords="offset points",
                fontsize=6.5, color="0.45", ha="right")
    ax.set_xlabel("True value (CHF)")
    ax.set_ylabel("Cross − matched decoded (CHF)")
    ax.set_xticks([0, 15, 30, 45])

    sns.despine(fig=fig, offset=4, trim=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modes", default="notes/data/session_shift_modes_npcr.tsv")
    p.add_argument("--cross", default="notes/data/cross_condition_decoding.tsv")
    p.add_argument("--out", default="notes/figures/mode_reorganisation.pdf")
    a = p.parse_args()
    main(Path(a.modes), Path(a.cross), Path(a.out))
