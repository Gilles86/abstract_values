"""What the encoding models say about precision across stimulus space.

Expected decoded uncertainty is simulated from each fitted encoding model: for
every stimulus, many noisy response vectors are decoded and the spread of the
decoded estimates recorded. It is the model's own precision profile, free of
the trial-to-trial BOLD noise that dominates single-trial decoded SD.

    NPCr / aPRF       uncertainty about VALUE,       in CHF
    V1  / von Mises   uncertainty about ORIENTATION, in degrees

Both are per subject and per session; sessions are averaged here.

Writes notes/figures/expected_uncertainty.pdf.
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
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})

VAL_C, ORI_C = "#2A9D8F", "#3B5BA5"
COND_C = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}
COND_L = {"cdf": "CDF", "inverse_cdf": "Inverse CDF"}


def profile_panel(ax, df, xcol, color, xlabel, ylabel, title, xticks=None):
    prof = df.groupby(["subject", xcol]).sd_E.mean().reset_index()
    for s, g in prof.groupby("subject"):
        g = g.sort_values(xcol)
        ax.plot(g[xcol], g.sd_E, color=color, lw=0.6, alpha=0.30)
    m = prof.groupby(xcol).sd_E.agg(["mean", "sem"])
    ax.fill_between(m.index, m["mean"] - m["sem"], m["mean"] + m["sem"],
                    color=color, alpha=0.30, lw=0)
    ax.plot(m.index, m["mean"], color=color, lw=2.0)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(xticks)
    return prof, m


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--value-tsv", default="notes/data/expected_uncertainty_value_npcr.tsv")
    p.add_argument("--ori-tsv", default="notes/data/expected_uncertainty_orientation_v1.tsv")
    p.add_argument("--paradigm-tsv", default="notes/data/efficient_coding_paradigm.tsv")
    p.add_argument("--out", default="notes/figures/expected_uncertainty.pdf")
    a = p.parse_args()

    val = pd.read_csv(a.value_tsv, sep="\t")
    ori = pd.read_csv(a.ori_tsv, sep="\t")
    key = pd.read_csv("notes/data/session_mapping_key.tsv", sep="\t")
    val = val.merge(key, on=["subject", "session"], how="left").dropna(subset=["mapping"])
    ori = ori.merge(key, on=["subject", "session"], how="left").dropna(subset=["mapping"])

    par = pd.read_csv(a.paradigm_tsv, sep="\t")
    from bauer.efficient_coding import MAPPING_ORIENTATIONS_DEG as O, MAPPING_VALUES as G
    par["value"] = [np.interp(o, O, G[m]) for o, m in zip(par.orientation, par.mapping)]

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.2), constrained_layout=True)

    # --- a: NPCr value uncertainty, per condition -----------------------
    ax = axes[0, 0]
    for cond in ("cdf", "inverse_cdf"):
        g = val[val.mapping == cond]
        prof = g.groupby(["subject", "value"]).sd_E.mean().reset_index()
        m = prof.groupby("value").sd_E.agg(["mean", "sem"])
        ax.fill_between(m.index, m["mean"]-m["sem"], m["mean"]+m["sem"],
                        color=COND_C[cond], alpha=0.28, lw=0)
        ax.plot(m.index, m["mean"], color=COND_C[cond], lw=1.8)
        ax.text(0.03, 0.97 if cond == "cdf" else 0.89, COND_L[cond], color=COND_C[cond],
                fontsize=7, transform=ax.transAxes, va="top")
    ax.set_xlabel("True value (CHF)"); ax.set_ylabel("Expected decoded SD (CHF)")
    ax.set_title(f"NPCr value uncertainty, by mapping (n = {val.subject.nunique()})")

    # --- b: the value density each mapping imposes ----------------------
    ax = axes[0, 1]
    # Density over the CHF axis, not per distinct value: each mapping shows its
    # own 23 values about equally often, so counting per value is flat by
    # construction and says nothing. What differs is WHERE on the scale those
    # values fall.
    bins = np.arange(2, 43, 2.5)
    for cond in ("cdf", "inverse_cdf"):
        v = par[par.mapping == cond].value.values
        h, edges = np.histogram(v, bins=bins, density=True)
        ctr = (edges[:-1] + edges[1:]) / 2
        ax.plot(ctr, h, color=COND_C[cond], lw=1.6, marker="o", ms=3.2,
                mec="white", mew=0.4)
    ax.set_xlabel("True value (CHF)"); ax.set_ylabel("Density of presented values")
    ax.set_title("Where each mapping puts its values")
    ax.text(0.03, 0.97, "Efficient coding: precision should be best\nwhere the values pile up",
            fontsize=6.3, color="0.4", transform=ax.transAxes, va="top")

    # --- c: V1 orientation uncertainty, per condition (the control) -----
    ax = axes[1, 0]
    # Subjects differ 3x in overall V1 uncertainty (7-21 deg), so an unscaled
    # group mean is dominated by level rather than shape. z-score each subject
    # within condition, which is what the condition comparison is about anyway.
    prof = ori.groupby(["subject", "mapping", "orientation"]).sd_E.mean().reset_index()
    prof["z"] = prof.groupby(["subject", "mapping"]).sd_E.transform(
        lambda x: (x - x.mean()) / x.std())
    for cond in ("cdf", "inverse_cdf"):
        m = prof[prof.mapping == cond].groupby("orientation").z.agg(["mean", "sem"])
        ax.fill_between(m.index, m["mean"]-m["sem"], m["mean"]+m["sem"],
                        color=COND_C[cond], alpha=0.28, lw=0)
        ax.plot(m.index, m["mean"], color=COND_C[cond], lw=1.6)
    ax.axhline(0, color="0.6", lw=0.8, ls="--", zorder=0)
    for c in (0, 90, 180):
        ax.axvline(c, color="0.92", lw=0.7, ls=":", zorder=0)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("Orientation θ (deg)")
    ax.set_ylabel("Expected decoded SD (z, within subject)")
    ax.set_title("V1 orientation uncertainty — should not move")

    # --- d: within-subject reallocation ---------------------------------
    ax = axes[1, 1]
    w = (val.groupby(["subject", "mapping", "value"]).sd_E.mean().reset_index()
            .pivot_table(index=["subject", "value"], columns="mapping", values="sd_E").dropna())
    w["diff"] = w["cdf"] - w["inverse_cdf"]
    m = w.groupby("value")["diff"].agg(["mean", "sem"])
    ax.axhline(0, color="0.5", lw=0.9, ls="--")
    ax.fill_between(m.index, m["mean"]-m["sem"], m["mean"]+m["sem"], color="0.35", alpha=0.25, lw=0)
    ax.plot(m.index, m["mean"], color="0.2", lw=1.6)
    ax.set_xlabel("True value (CHF)")
    ax.set_ylabel("CDF − inverse CDF  (CHF)")
    ax.set_title("Within-subject difference in uncertainty")
    n_pairs = w.index.get_level_values("subject").nunique()
    ax.text(0.03, 0.97, f"same {n_pairs} subjects in both conditions",
            fontsize=6.3, color="0.4", transform=ax.transAxes, va="top")

    for cond in ("cdf", "inverse_cdf"):
        g = val[val.mapping == cond].groupby("value").sd_E.mean()
        print(f"  NPCr {COND_L[cond]:12s}: mean {g.mean():.2f} CHF, "
              f"min at {g.idxmin():.1f} CHF ({g.min():.2f}), max at {g.idxmax():.1f} ({g.max():.2f})")
    sns.despine(fig=fig, offset=4)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
