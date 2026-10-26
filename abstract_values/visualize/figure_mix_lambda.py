"""Where does tuning sit, and how wide is it? Position/width mixing model, fitted to the betas.

a: what the four corner models predict for one voxel in the two sessions
   (MixPositionWidthGaussianPRF: lam_mu = position follows the quantile,
   lam_sigma = width follows the quantile).
b: observed, leave-one-run-out cvR^2 of each corner relative to the value code
   (0, 0), per ROI; mean +- s.e.m. across participants, filled = p < .05.
c: the same contrasts, IPS (mean of IPS0/1-2/3-5) and V1, observed vs betas
   simulated with a known code (simulate_lambda_pipeline; its lambda = 1 moves
   only the preferred value, i.e. corner (1, 0)), at noise x1-3.
d: which corner wins per participant and ROI.

Input: notes/data/mix_lambda_width.tsv (fit_mix_lambda.py --model width, collected).

    python -m abstract_values.visualize.figure_mix_lambda
    -> notes/figures/fig_mix_lambda.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from abstract_values.encoding_models import hierarchical_lambda as hl
from abstract_values.encoding_models.models import MixPositionWidthGaussianPRF
from abstract_values.visualize import qc_value_shift as q   # noqa: F401 house rcParams
from abstract_values.visualize.figure_tuning_connectivity import ORI_COL, VAL_COL

REPO = Path(__file__).resolve().parents[2]
CORNERS = {"0_0": ("Value code", VAL_COL), "1_0": ("Position follows quantile", "#C9A0DC"),
           "0_1": ("Width follows quantile", "0.6"), "1_1": ("Quantile code", ORI_COL)}
ROIS = ["V1", "hV4", "V3ab", "IPS0", "IPS12", "IPS345"]
NAME = {"V1": "V1", "hV4": "hV4", "V3ab": "V3a/b", "IPS0": "IPS0", "IPS12": "IPS1–2",
        "IPS345": "IPS3–5", "IPS": "IPS"}
MAP_COL = {0: "#3B5BA5", 1: "#C44E52"}


def schematic(axes):
    g = np.linspace(2, 42, 400)
    v = np.r_[g, g]
    ses = np.r_[np.zeros(400), np.ones(400)]
    F = lambda m: 2 + 40 * np.interp(g, np.r_[2, hl.VALUES[m], 42],          # noqa: E731
                                     np.r_[0, (np.arange(23) + .5) / 23, 1])
    par = pd.DataFrame({"x": v, "q": np.r_[F("cdf"), F("inv")], "m": ses}).astype(np.float32)
    q16 = 2 + 40 * np.interp(16, np.r_[2, hl.VALUES["cdf"], 42],
                             np.r_[0, (np.arange(23) + .5) / 23, 1])
    m = MixPositionWidthGaussianPRF()
    for ax, (k, (lab, col)) in zip(axes, CORNERS.items()):
        lm, ls = map(float, k.split("_"))
        p = pd.DataFrame({"mode": [(1 - lm) * 16 + lm * q16], "fwhm": [7.], "amplitude": [1.],
                          "baseline": [0.], "lam_mu": [lm], "lam_sigma": [ls]}).astype(np.float32)
        y = m.predict(parameters=p, paradigm=par).values[:, 0]
        for s in (0, 1):
            ax.plot(v[ses == s], y[ses == s], color=MAP_COL[s], lw=1.1)
        ax.set_title(lab, fontsize=6.5, color=col if k != "0_1" else "0.35", loc="left")
        ax.set_yticks([])
        ax.set_xticks([2, 22, 42])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(2, 42)
    axes[0].text(.98, .95, "Misaligned", color=MAP_COL[0], fontsize=5.8, ha="right", va="top",
                 transform=axes[0].transAxes)
    axes[0].text(.98, .8, "Aligned", color=MAP_COL[1], fontsize=5.8, ha="right", va="top",
                 transform=axes[0].transAxes)
    for ax in axes:
        ax.set_xlabel("Value (CHF)", fontsize=6.5)


def contrasts(d):
    for k in ("1_0", "0_1", "1_1"):
        d[f"d{k}"] = 1e3 * (d[f"cvr2_fixed{k}"] - d["cvr2_fixed0_0"])
    return d


def main(out=REPO / "notes/figures/fig_mix_lambda.pdf"):
    d = contrasts(pd.read_csv(REPO / "notes/data/mix_lambda_width.tsv", sep="\t",
                              dtype={"subject": str}))
    fig = plt.figure(figsize=(7.25, 5.3))
    top = fig.add_gridspec(1, 4, left=.05, right=.99, top=.9, bottom=.74, wspace=.15)
    schematic([fig.add_subplot(top[i]) for i in range(4)])
    fig.text(.01, .965, "a  Four ways a voxel's tuning can respond to the change in mapping "
             "(preferred value 16 CHF in the Misaligned session)", fontweight="bold", fontsize=8)

    bot = fig.add_gridspec(1, 3, left=.08, right=.99, top=.58, bottom=.09, wspace=.38,
                           width_ratios=[1.25, 1.15, .8])
    ax = fig.add_subplot(bot[0])
    o = d[d.cond == "observed"]
    for j, k in enumerate(("1_0", "0_1", "1_1")):
        lab, col = CORNERS[k]
        for i, r in enumerate(ROIS):
            x = o[o.roi == r][f"d{k}"]
            m, se = x.mean(), x.std() / np.sqrt(len(x))
            p = stats.ttest_1samp(x, 0).pvalue
            y = i + (j - 1) * .22
            ax.plot([m - se, m + se], [y, y], color=col, lw=1.2)
            ax.plot(m, y, "o", color=col, ms=3.6, mew=.8, mfc=col if p < .05 else "white", mec=col)
    ax.axvline(0, color=VAL_COL, lw=.7)
    ax.set_yticks(range(len(ROIS)), [NAME[r] for r in ROIS])
    ax.set_ylim(len(ROIS) - .4, -1.6)
    for j, k in enumerate(("1_1", "1_0", "0_1")):
        lab, col = CORNERS[k]
        ax.text(.02, .97 - .055 * j, lab, color=col if k != "0_1" else "0.45", fontsize=6,
                transform=ax.transAxes, va="top")
    ax.set_xlabel("cvR² relative to the value code (×10⁻³)")
    ax.set_title("b  Observed: which code predicts held-out runs", loc="left", fontweight="bold")

    ax = fig.add_subplot(bot[1])
    ips = d[d.roi.isin(["IPS0", "IPS12", "IPS345"])].groupby(["cond", "subject"])[
        ["d1_0", "d1_1"]].mean().reset_index()
    conds = [("observed", "Observed"), ("sim-lam0-noise1", "λ = 0, ×1"),
             ("sim-lam0-noise2", "λ = 0, ×2"), ("sim-lam0-noise3", "λ = 0, ×3"),
             ("sim-lam1-noise1", "Position, ×1"), ("sim-lam1-noise2", "Position, ×2"),
             ("sim-lam1-noise3", "Position, ×3")]
    for i, (c, lab) in enumerate(conds):
        for j, k in enumerate(("1_0", "1_1")):
            x = ips[ips.cond == c][f"d{k}"]
            if len(x) < 3:
                continue
            m, se = x.mean(), x.std() / np.sqrt(len(x))
            y = i + (j - .5) * .3
            col = CORNERS[k][1]
            ax.plot([m - se, m + se], [y, y], color=col, lw=1.2)
            ax.plot(m, y, "o", color=col, ms=3.4, mew=0)
            if len(x) < 30:
                ax.text(max(m + se, 0) + .3, y, f"n = {len(x)}", fontsize=5.3, va="center",
                        color="0.5")
    ax.axvline(0, color=VAL_COL, lw=.7)
    ax.set_yticks(range(len(conds)), [c[1] for c in conds])
    ax.set_ylim(len(conds) - .5, -.6)
    ax.set_xlim(-6, 4)
    ax.set_xlabel("cvR² relative to the value code (×10⁻³)")
    ax.set_title("c  IPS: observed vs simulated", loc="left", fontweight="bold")
    ax.text(.98, .03, "Simulated: betas from each voxel's\nreal fit with a known code",
            transform=ax.transAxes, fontsize=5.6, ha="right", color="0.4")

    ax = fig.add_subplot(bot[2])
    cor = [f"cvr2_fixed{k}" for k in CORNERS]
    d["win"] = d[cor].idxmax(axis=1).str.replace("cvr2_fixed", "")
    w = d.groupby("cond").win.value_counts(normalize=True).unstack().reindex(
        columns=list(CORNERS)).fillna(0)
    for i, (c, lab) in enumerate(conds):
        if c not in w.index or (d.cond == c).sum() < 30 * 6:
            continue
        left = 0
        for k, (_, col) in CORNERS.items():
            ax.barh(i, w.loc[c, k], left=left, color=col, height=.7, lw=0)
            left += w.loc[c, k]
    ax.set_yticks(range(len(conds)), [c[1] for c in conds])
    ax.set_ylim(len(conds) - .5, -.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of participants × ROIs")
    ax.set_title("d  Winning code", loc="left", fontweight="bold")
    for a in fig.axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    out = Path(out)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=250)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
