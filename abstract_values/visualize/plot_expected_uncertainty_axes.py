"""Expected uncertainty on both stimulus axes, for both ROIs.

Four views of the same two profiles: each ROI's expected SD against the axis
its model lives on, and against the other one.

  a  V1 uncertainty vs orientation
  b  V1 uncertainty vs value
  c  NPCr uncertainty vs value
  d  NPCr uncertainty vs orientation

It is tempting to read the condition split as a test of which space the
precision is anchored in -- one curve on the right axis, two on the wrong one.
It is NOT that test here. Both models are fitted with weights shared across
sessions and only the noise model re-fitted per session, so a single curve in
the model's own space is largely built in, and the split on the other axis
follows mechanically from the mapping difference. What the panels do show is
the SHAPE of each profile, which is not built in.

The anchoring test needs per-session tuning fits, so that the two conditions
could in principle disagree on either axis.

Expected uncertainty is simulated from each fitted encoding model -- the noise
is drawn afresh rather than measured -- so this is the model's own precision
profile, free of the trial-to-trial BOLD noise that dominates decoded SD.

    python -m abstract_values.visualize.plot_expected_uncertainty_axes
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

COND_C = {"cdf": "#2A6F97", "inverse_cdf": "#C4442B"}
NICE = {"cdf": "CDF", "inverse_cdf": "Inverse CDF"}


def load(eu_tsv, key_tsv, lut_tsv):
    d = pd.read_csv(eu_tsv, sep="\t")
    key = pd.read_csv(key_tsv, sep="\t")
    lut = pd.read_csv(lut_tsv, sep="\t")

    d["sn"] = d.subject.astype(str).str.replace("pil", "", regex=False).astype(int)
    d = d.merge(key.rename(columns={"subject": "sn", "mapping": "cond"}),
                on=["sn", "session"], how="left")
    d["condition"] = d["condition"].fillna(d["cond"])

    ori = d[d.space == "orientation"].copy()
    ori["sd"] = np.rad2deg(ori["sd_E"])            # period-pi radians -> degrees
    # the simulation grid carries float error (7.500017 deg); the lookup keys
    # are exact, so round before joining or every row misses.
    ori["orientation"] = ori["value_deg"].round(1)
    # the orientation file's own "value" column is the stimulus in radians;
    # drop it so the lookup can supply the CHF value for this condition.
    ori = ori.drop(columns=["value"]).merge(
        lut.rename(columns={"mapping": "condition"}),
        on=["condition", "orientation"], how="left")

    val = d[d.space == "value"].copy()
    val["sd"] = val["sd_E"]
    val["value"] = val["value"].round(1)
    lut = lut.assign(value=lut["value"].round(1))
    val = val.merge(lut.rename(columns={"mapping": "condition"}),
                    on=["condition", "value"], how="left")
    return ori, val


def curve(ax, d, xcol, ycol, bins):
    for cond, g in d.groupby("condition"):
        g = g.copy()
        g["bin"] = pd.cut(g[xcol], bins).apply(lambda i: i.mid).astype(float)
        per = (g.groupby(["bin", "subject"], observed=True)[ycol].mean()
                 .groupby("bin", observed=True).agg(["mean", "sem"]).reset_index())
        c = COND_C[cond]
        ax.plot(per["bin"], per["mean"], color=c, marker="o", ms=3)
        ax.fill_between(per["bin"], per["mean"] - per["sem"],
                        per["mean"] + per["sem"], color=c, alpha=.22, lw=0)
        ax.annotate(NICE[cond], (per["bin"].iloc[-1], per["mean"].iloc[-1]),
                    xytext=(3, 0), textcoords="offset points", color=c,
                    fontsize=6.5, va="center")


def main(eu, key, lut, out):
    ori, val = load(eu, key, lut)
    ob = [0, 30, 60, 90, 120, 150, 180]
    vb = [0, 8, 14, 20, 26, 32, 38, 45]

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0), constrained_layout=True)

    curve(axes[0, 0], ori, "orientation", "sd", ob)
    axes[0, 0].set_title("V1 · orientation uncertainty", fontsize=8, color="0.2")
    axes[0, 0].set_xlabel("Orientation (deg)"); axes[0, 0].set_xticks([0, 90, 180])

    curve(axes[0, 1], ori, "value", "sd", vb)
    axes[0, 1].set_xlabel("Value (CHF)"); axes[0, 1].set_xticks([0, 15, 30, 45])

    curve(axes[1, 0], val, "value", "sd", vb)
    axes[1, 0].set_title("NPCr · value uncertainty", fontsize=8, color="0.2")
    axes[1, 0].set_xlabel("Value (CHF)"); axes[1, 0].set_xticks([0, 15, 30, 45])

    curve(axes[1, 1], val, "orientation", "sd", ob)
    axes[1, 1].set_xlabel("Orientation (deg)"); axes[1, 1].set_xticks([0, 90, 180])

    for ax in axes[0]:
        ax.set_ylabel("Expected SD (deg)")
        ax.set_ylim(0, None)
    for ax in axes[1]:
        ax.set_ylabel("Expected SD (CHF)")
        ax.set_ylim(0, None)
    for ax in axes.ravel():
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 1.22)

    sns.despine(fig=fig, offset=4, trim=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")
    for name, d, x in [("V1 vs orientation", ori, "orientation"),
                       ("V1 vs value", ori, "value"),
                       ("NPCr vs value", val, "value"),
                       ("NPCr vs orientation", val, "orientation")]:
        s = d.groupby(["condition", "subject"])["sd"].mean().groupby("condition").mean()
        print(f"  {name:22s} cdf {s.get('cdf', np.nan):.2f}  "
              f"inv {s.get('inverse_cdf', np.nan):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eu", default="notes/data/expected_uncertainty_alpha10.tsv")
    p.add_argument("--key", default="notes/data/session_mapping_key.tsv")
    p.add_argument("--lut", default="notes/data/value_orientation_lut.tsv")
    p.add_argument("--out", default="notes/figures/expected_uncertainty_axes.pdf")
    a = p.parse_args()
    main(Path(a.eu), Path(a.key), Path(a.lut), Path(a.out))
