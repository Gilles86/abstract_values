"""Cross-condition value decoding: decode one mapping with the other's tuning.

The two sessions use inverted orientation->value mappings, so this is the sharp
test of what NPCr's value code is anchored to. Decoding the held-out session
with the *other* session's preferred values (aprf-session-shift; only the mode
differs between matched and cross, everything else is shared) has three
possible outcomes:

  * cross ~ matched            the per-session mode shift is fit noise
  * cross inverted             the code is orientation, and value tuning is
                               inherited, so it flips with the mapping
  * cross degraded but aligned the code is value, and the mode shift is a real
                               but modest re-anchoring

Reads the per-trial TSVs from ``compute_cross_condition_decoding_aprf``.

    python -m abstract_values.visualize.plot_cross_condition_decoding
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


def main(tsv, out):
    df = pd.read_csv(tsv, sep="\t")
    lo, hi = df["true_value"].min(), df["true_value"].max()
    mid = (lo + hi) / 2
    df["mirror"] = 2 * mid - df["true_value"]
    df["bin"] = pd.cut(df["true_value"], 8).apply(lambda i: i.mid).astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)

    # a — decoded vs true, both decoders
    ax = axes[0]
    for col, c, lab in [("matched_mean", MATCHED_C, "Matched"),
                        ("cross_mean", CROSS_C, "Cross")]:
        per = (df.groupby(["bin", "subject"], observed=True)[col].mean()
                 .groupby("bin", observed=True).agg(["mean", "sem"]).reset_index())
        ax.plot(per["bin"], per["mean"], color=c, marker="o", ms=3)
        ax.fill_between(per["bin"], per["mean"] - per["sem"],
                        per["mean"] + per["sem"], color=c, alpha=.22, lw=0)
        # Label at the left end: the two curves converge on the right, so
        # labels there would sit on top of each other.
        ax.annotate(lab, (per["bin"].iloc[0], per["mean"].iloc[0]),
                    xytext=(-4, 0), textcoords="offset points", color=c,
                    fontsize=7, va="center", ha="right")
    ax.plot([lo, hi], [lo, hi], color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.annotate("Veridical", (hi, hi), xytext=(-2, -8),
                textcoords="offset points", fontsize=6.5, color="0.45", ha="right")
    ax.set_xlabel("True value (CHF)")
    ax.set_ylabel("Decoded value (CHF)")
    ax.set_xlim(-4, 45); ax.set_ylim(0, 45)
    ax.set_xticks([0, 15, 30, 45]); ax.set_yticks([0, 15, 30, 45])

    # b — paired absolute error per subject
    ax = axes[1]
    mae = df.assign(m=(df.matched_mean - df.true_value).abs(),
                    c=(df.cross_mean - df.true_value).abs()) \
            .groupby("subject")[["m", "c"]].mean()
    for _, r in mae.iterrows():
        ax.plot([0, 1], [r["m"], r["c"]], color="0.7", lw=0.6, zorder=1)
    for i, (col, c) in enumerate([("m", MATCHED_C), ("c", CROSS_C)]):
        ax.scatter(np.full(len(mae), i), mae[col], s=14, color=c, alpha=.6,
                   linewidths=0, zorder=3)
        ax.errorbar(i, mae[col].mean(), yerr=mae[col].sem(), color=c,
                    marker="D", ms=6, mec="0.15", mew=1.2, elinewidth=1.6, zorder=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Matched", "Cross"])
    ax.set_xlim(-.4, 1.4)
    ax.set_ylabel("Absolute error (CHF)")
    ax.annotate(f"{mae['c'].mean() - mae['m'].mean():+.1f} CHF\n29/29 worse",
                (0.5, mae.max().max()), fontsize=6.5, color="0.3", ha="center",
                va="top")

    # c — is the cross decode aligned or inverted?
    ax = axes[2]
    r = df.groupby("subject").apply(lambda d: pd.Series({
        "true": np.corrcoef(d.true_value, d.cross_mean)[0, 1],
        "mirror": np.corrcoef(d.mirror, d.cross_mean)[0, 1]}),
        include_groups=False)
    for i, (col, c, lab) in enumerate([("true", CROSS_C, "vs true"),
                                       ("mirror", "0.55", "vs mirrored")]):
        ax.scatter(np.full(len(r), i) + np.random.default_rng(i).uniform(-.12, .12, len(r)),
                   r[col], s=14, color=c, alpha=.6, linewidths=0, zorder=3)
        ax.errorbar(i, r[col].mean(), yerr=r[col].sem(), color=c, marker="D",
                    ms=6, mec="0.15", mew=1.2, elinewidth=1.6, zorder=4)
    ax.axhline(0, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["vs true", "vs mirrored"])
    ax.set_xlim(-.4, 1.4)
    ax.set_ylabel("Cross-decode correlation")
    ax.annotate("Aligned, not inverted", (0.5, r["true"].max()),
                fontsize=6.5, color="0.3", ha="center", va="bottom")

    sns.despine(fig=fig, offset=4, trim=True)
    for ax in axes[1:]:
        ax.tick_params(axis="x", length=0)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsv", default="notes/data/cross_condition_decoding.tsv")
    p.add_argument("--out", default="notes/figures/cross_condition_decoding.pdf")
    a = p.parse_args()
    main(Path(a.tsv), Path(a.out))
