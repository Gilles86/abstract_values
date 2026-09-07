"""Cross-condition decoding, split by which mapping was decoded with which.

The pooled version averages the two transfers, which hides whether they behave
alike. The two mappings are NOT inversions of one another -- both are monotone in
orientation with Spearman rho = 1, differing only in where they bunch values
(cdf at the high end, inverse_cdf at the low end; mean gap 4.1 CHF). So the
transfer is between two warps of the same axis, and a translation of the value
code would still show as opposite-signed shifts in the two directions.
Splitting them is that test.

Left two panels: the decoded quantity against true value for the held-out
session, with its own tuning (matched) and with the other condition's (cross).
Right panel: the difference, one line per direction.

``--x orientation`` puts the gabor orientation on the x-axis instead of the
value, which is how the behavioural analyses are read. The two axes are the
same trials seen through the two mappings: value is a monotone function of
orientation within a session and the two sessions invert it, so an effect
fixed in value space becomes mirrored in orientation space and vice versa.
The veridical value for that condition is drawn as the reference.

``--metric sd`` draws the posterior SD instead of the posterior mean -- how
uncertain the decoder is rather than where it lands. The two answer different
questions: a mismatched tuning set could in principle leave the estimate alone
and only widen it, or shift it while staying just as confident.

    python -m abstract_values.visualize.plot_cross_condition_by_direction
    python -m abstract_values.visualize.plot_cross_condition_by_direction \
        --metric sd --out notes/figures/cross_condition_sd_by_direction.pdf
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
ORI_BINS = [0, 30, 60, 90, 120, 150, 180]


def add_orientation(df, lut_tsv):
    """Attach the gabor orientation via the condition's value<->orientation map.

    Exact, not approximate: within a mapping each presented CHF value comes
    from exactly one of the 23 orientations, so the lookup is a bijection.
    """
    lut = pd.read_csv(lut_tsv, sep="\t")
    return df.merge(lut.rename(columns={"mapping": "test_condition",
                                        "value": "true_value"}),
                    on=["test_condition", "true_value"], how="left")


def profile(d, col, x="value"):
    d = d.copy()
    if x == "value":
        d["bin"] = pd.cut(d["true_value"], BINS).apply(lambda i: i.mid).astype(float)
    else:
        d["bin"] = pd.cut(d["orientation"], ORI_BINS).apply(lambda i: i.mid).astype(float)
    return (d.groupby(["bin", "subject"], observed=True)[col].mean()
              .groupby("bin", observed=True).agg(["mean", "sem"]).reset_index())


def main(tsv, out, metric="mean", x="value", lut=None):
    df = pd.read_csv(tsv, sep="\t")
    if x == "orientation":
        df = add_orientation(df, lut)
        if df["orientation"].isna().any():
            raise SystemExit("Some trials have no orientation in the lookup.")
    xlab = "True value (CHF)" if x == "value" else "Orientation (deg)"
    xlim = (0, 45) if x == "value" else (0, 180)
    xticks = [0, 15, 30, 45] if x == "value" else [0, 45, 90, 135, 180]
    m_col, c_col = f"matched_{metric}", f"cross_{metric}"
    df["d_dec"] = df[c_col] - df[m_col]
    ylab = ("Decoded value (CHF)" if metric == "mean"
            else "Posterior SD (CHF)")
    dlab = (f"Cross − matched decoded (CHF)" if metric == "mean"
            else "Cross − matched SD (CHF)")

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), constrained_layout=True)

    for ax, test in zip(axes[:2], ["cdf", "inverse_cdf"]):
        d = df[df.test_condition == test]
        other = NICE["inverse_cdf" if test == "cdf" else "cdf"]
        for col, c, lab in [(m_col, MATCHED_C, "Own tuning"),
                            (c_col, CROSS_C, f"{other} tuning")]:
            per = profile(d, col, x)
            ax.plot(per["bin"], per["mean"], color=c, marker="o", ms=3)
            ax.fill_between(per["bin"], per["mean"] - per["sem"],
                            per["mean"] + per["sem"], color=c, alpha=.22, lw=0)
            dy = 7 if col == c_col else -9
            ax.annotate(lab, (per["bin"].iloc[1], per["mean"].iloc[1]),
                        xytext=(2, dy), textcoords="offset points", color=c,
                        fontsize=6.5, va="center", ha="left")
        if metric == "mean":
            # Reference: what a perfect decode would give. On the value axis
            # that is the identity; on the orientation axis it is this
            # condition's own value-orientation mapping, which is exactly what
            # the two sessions invert.
            if x == "value":
                ax.plot([0, 45], [0, 45], color="0.55", lw=0.8,
                        ls=(0, (4, 3)), zorder=0)
            else:
                ref = profile(d, "true_value", x)
                ax.plot(ref["bin"], ref["mean"], color="0.55", lw=0.9,
                        ls=(0, (4, 3)), zorder=0)
        ax.set_title(f"{NICE[test]} session", fontsize=8, color="0.2")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        if metric == "mean":
            ax.set_ylim(0, 45); ax.set_yticks([0, 15, 30, 45])
        else:
            ax.set_ylim(0, 10)

    ax = axes[2]
    for train, d in df.groupby("train_condition"):
        per = profile(d, "d_dec", x)
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
    ax.set_xlabel(xlab)
    ax.set_ylabel(dlab)
    ax.set_xlim(xlim[0], xlim[1] * 1.38)
    ax.set_xticks(xticks)

    sns.despine(fig=fig, offset=4, trim=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsv", default="notes/data/cross_condition_decoding.tsv")
    p.add_argument("--x", default="value", choices=["value", "orientation"],
                   help="Stimulus axis: CHF value, or the gabor orientation "
                        "the behavioural analyses use.")
    p.add_argument("--lut", default="notes/data/value_orientation_lut.tsv")
    p.add_argument("--metric", default="mean", choices=["mean", "sd"],
                   help="Posterior mean (where the decoder lands) or posterior "
                        "SD (how sure it is).")
    p.add_argument("--out",
                   default="notes/figures/cross_condition_by_direction.pdf")
    a = p.parse_args()
    main(Path(a.tsv), Path(a.out), metric=a.metric, x=a.x, lut=Path(a.lut))
