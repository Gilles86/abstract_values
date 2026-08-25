"""Prior attraction or likelihood repulsion?  Bias as a function of value.

A scalar bias per subject cannot tell the two apart -- both make people wrong.
The signature is in the SHAPE of bias(value), and this design separates them
cleanly because the two mappings have complementary stimulus densities:

  cdf          density peaks at ~12 and ~32 CHF, TROUGH at 22
  inverse_cdf  density peaks at ~2.5, 22 and ~41.5, troughs at ~12 and ~32

Both conditions have the same mean (22.0 CHF), so:

  Prior attraction (bias toward the distribution mean) predicts a NEGATIVE
  bias slope at 22 in BOTH conditions.

  Likelihood repulsion (bias down the density gradient, away from peaks)
  predicts a zero crossing with POSITIVE slope at every density peak and
  NEGATIVE slope at every trough -- so negative at 22 under cdf, POSITIVE at
  22 under inverse_cdf.  Opposite signs.

The 16-28 CHF window around 22 is therefore the decisive test, and it is well
clear of the response bar's 0-42 limits, so bid truncation cannot drive it.

Usage:
    python -m abstract_values.visualize.bias_attraction_vs_repulsion
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
from scipy import stats

from abstract_values.behavior.data import get_all_behavioral_data

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelpad": 4, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2, "legend.frameon": False, "legend.handlelength": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})
sns.set_context("paper")

COND_C = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}
COND_L = {"cdf": "CDF", "inverse_cdf": "Inverse CDF"}
WIN = (16.0, 28.0)   # decisive window around the shared mean of 22 CHF


def load():
    b = get_all_behavioral_data()
    b = b[b.event_type == "feedback"].copy()
    b["response"] = pd.to_numeric(b["response"], errors="coerce")
    b = b.reset_index()
    b = b[b.subject.between(3, 28)].dropna(subset=["response"])
    b["bias"] = b["response"] - b["value"]
    return b


def predictors(values):
    dens = 1.0 / np.gradient(values)
    A = -(values - values.mean())                 # toward the distribution mean
    R = -np.gradient(np.log(dens), values)        # down the density gradient
    z = lambda x: (x - x.mean()) / x.std()
    return pd.DataFrame({"value": values, "A": z(A), "R": z(R), "dens": dens})


def panel_bias_curve(ax, bb, pred, cond):
    d = bb[bb.mapping == cond]
    g = d.groupby("value")["bias"].agg(["mean", "sem"]).reset_index()
    P = pred[cond]

    axd = ax.twinx()
    axd.fill_between(P["value"], 0, P["dens"], color="0.85", zorder=0, lw=0)
    axd.set_ylim(0, P["dens"].max() * 3.2)
    axd.set_yticks([])
    axd.spines["right"].set_visible(False)

    ax.axhline(0, color="0.4", lw=0.8, ls="--", zorder=1)
    ax.axvspan(*WIN, color=COND_C[cond], alpha=0.09, zorder=0, lw=0)
    ax.errorbar(g["value"], g["mean"], yerr=g["sem"], color=COND_C[cond],
                marker="o", ms=3.5, lw=1.4, capsize=0, zorder=3)
    for v in P.loc[np.r_[True, (P["dens"].values[1:-1] >
                               np.maximum(P["dens"].values[:-2],
                                          P["dens"].values[2:])), True], "value"]:
        pass
    ax.set_zorder(axd.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_xlabel("True value (CHF)")
    ax.set_ylabel("Bias, bid − true value (CHF)")
    ax.set_title(f"{COND_L[cond]}   (grey = stimulus density)",
                 fontsize=9, color="0.2")
    return ax


def slopes_in_window(bb, cond, lo=WIN[0], hi=WIN[1]):
    out = {}
    for s, d in bb[bb.mapping == cond].groupby("subject"):
        w = d[(d.value >= lo) & (d.value <= hi)]
        if len(w) >= 4:
            out[s] = stats.linregress(w["value"], w["bias"]).slope
    return pd.Series(out)


def panel_slopes(ax, bb):
    sl = {c: slopes_in_window(bb, c) for c in COND_C}
    common = sl["cdf"].index.intersection(sl["inverse_cdf"].index)
    for va, vb in zip(sl["cdf"][common], sl["inverse_cdf"][common]):
        ax.plot([0, 1], [va, vb], color="0.75", lw=0.7, zorder=1)
    for x, c in [(0, "cdf"), (1, "inverse_cdf")]:
        v = sl[c][common]
        ax.scatter(np.full(len(v), x), v, s=24, color=COND_C[c], zorder=3,
                   edgecolor="white", linewidth=0.5)
        ax.plot([x - 0.17, x + 0.17], [v.mean()] * 2, color=COND_C[c], lw=2.4, zorder=4)
        t, p = stats.ttest_1samp(v, 0)
        ax.annotate(f"{v.mean():+.3f}\np = {p:.0e}", xy=(x, ax.get_ylim()[1]),
                    ha="center", va="top", fontsize=7.5, color=COND_C[c])
    ax.axhline(0, color="0.4", lw=0.9, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["CDF\n(22 = trough)", "Inv CDF\n(22 = peak)"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel(f"Bias slope, {WIN[0]:.0f}–{WIN[1]:.0f} CHF")
    ax.set_title("Decisive test: slope at 22 CHF\nAttraction predicts negative in both",
                 fontsize=8.5, color="0.2")


def panel_betas(ax, bb, pred):
    rows = []
    for cond in COND_C:
        P = pred[cond]
        for s, d in bb[bb.mapping == cond].groupby("subject"):
            d = d.merge(P, on="value")
            X = np.c_[np.ones(len(d)), d["A"], d["R"]]
            beta = np.linalg.lstsq(X, d["bias"], rcond=None)[0]
            rows.append(dict(subject=s, cond=cond, attraction=beta[1], repulsion=beta[2]))
    B = pd.DataFrame(rows)
    xs = []
    for i, cond in enumerate(COND_C):
        for j, term in enumerate(["attraction", "repulsion"]):
            v = B[B.cond == cond][term]
            x = i * 2.4 + j
            xs.append((x, f"{COND_L[cond]}\n{term}"))
            colour = COND_C[cond] if term == "repulsion" else "0.6"
            ax.scatter(np.full(len(v), x) + np.random.default_rng(2).uniform(-0.11, 0.11, len(v)),
                       v, s=18, color=colour, alpha=0.75, edgecolor="white", linewidth=0.4)
            ax.plot([x - 0.22, x + 0.22], [v.mean()] * 2, color=colour, lw=2.4, zorder=4)
            t, p = stats.ttest_1samp(v, 0)
            ax.annotate(f"p = {p:.0e}" if p < 0.01 else f"p = {p:.2f}",
                        xy=(x, v.max()), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7, color="0.3")
    ax.axhline(0, color="0.4", lw=0.9, ls="--")
    ax.set_xticks([x for x, _ in xs]); ax.set_xticklabels([l for _, l in xs], fontsize=7)
    ax.set_ylabel("Regression weight on bias(value)")
    ax.set_title("Both predictors in one model\n(orthogonal under CDF, r = −0.10)",
                 fontsize=8.5, color="0.2")
    return B


def run(out):
    b = load()
    bb = b.groupby(["subject", "mapping", "value"])["bias"].mean().reset_index()
    pred = {m: predictors(np.sort(d["value"].unique())) for m, d in b.groupby("mapping")}

    fig = plt.figure(figsize=(12.0, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    panel_bias_curve(fig.add_subplot(gs[0, 0]), bb, pred, "cdf")
    panel_bias_curve(fig.add_subplot(gs[0, 1]), bb, pred, "inverse_cdf")
    panel_slopes(fig.add_subplot(gs[1, 0]), bb)
    panel_betas(fig.add_subplot(gs[1, 1]), bb, pred)

    fig.suptitle("Bias vs value: likelihood repulsion, not prior attraction  ·  n = 26",
                 fontsize=10, y=1.03, color="0.15")
    sns.despine(fig=fig, offset=4, trim=False)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="notes/figures/bias_attraction_vs_repulsion.pdf")
    run(p.parse_args().out)


if __name__ == "__main__":
    main()
