"""Does the shape of each ROI's code predict how noisy a participant's bids are?

Per participant and ROI: wrap index = cvR²(wrapped quantile) − cvR²(unwrapped
quantile) over all ROI voxels (positive: a circle fits better, orientation-like;
negative: an ordered line fits better). Correlated (Spearman, bootstrap 95% CI
over participants) with the total noise of the categorical cognitive model
(free prior, n = 35). Total rather than the perceptual / value split, which is
weakly identified per participant.

  a  ρ(wrap index, total noise) along the hierarchy, unsmoothed and smoothed
  b, c  the two ends: V3 and IPS0 (smoothed), one dot per participant

Usage:
    python -m abstract_values.visualize.figure_wrap_behavior
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from abstract_values.visualize import qc_value_shift as q   # noqa: F401 house rcParams

REPO = Path(__file__).resolve().parents[2]
ROIS = [("BensonV1", "V1"), ("BensonV2", "V2"), ("BensonV3", "V3"),
        ("hV4", "hV4"), ("V3ab", "V3a/b"), ("IPS0", "IPS0"),
        ("IPS12", "IPS1–2"), ("IPS345", "IPS3–5")]
COL = "#8172B2"


def sid(s):
    s = str(s)
    return int(s.replace("pil", "")) if "pil" in s else int(s)


def boot_rho(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    r = [stats.spearmanr(x[i], y[i])[0] for i in idx]
    return np.percentile(r, [2.5, 97.5])


def load(qtsv, params):
    cm = pd.read_csv(params, sep="\t")
    cm["sid"] = cm.subject.astype(int)
    d = pd.read_csv(qtsv, sep="\t", dtype={"subject": str})
    d = d[d.selection == "All voxels"].copy()
    d["wrap"] = d["quantile_wrapped"] - d["quantile"]
    d["sid"] = d.subject.map(sid)
    return d.merge(cm[["sid", "total_noise"]], on="sid")


def main(qtsv, params, out):
    d = load(qtsv, params)
    fig = plt.figure(figsize=(7.25, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 1, 1], wspace=.45, left=.08,
                          right=.98, top=.83, bottom=.24)
    ax = [fig.add_subplot(gs[0, i]) for i in range(3)]
    rows = []
    for sm, dx, mfc in ((False, .12, "white"), (True, -.12, COL)):
        for i, (roi, lab) in enumerate(ROIS):
            x = d[(d.smoothed == sm) & (d.roi == roi)]
            r, p = stats.spearmanr(x.wrap, x.total_noise)
            lo, hi = boot_rho(x.wrap.to_numpy(), x.total_noise.to_numpy())
            ax[0].plot([i + dx] * 2, [lo, hi], color=COL, lw=1)
            ax[0].plot(i + dx, r, "o", ms=3.5, color=COL, mfc=mfc, mew=.8)
            if p < .05:
                ax[0].text(i + dx, hi + .04, "*", ha="center", fontsize=7)
            rows.append(dict(smoothed=sm, roi=roi, rho=r, p=p, lo=lo, hi=hi, n=len(x)))
    ax[0].axhline(0, color="0.5", lw=.6, ls=":")
    ax[0].axvspan(4.5, 7.6, color="0.94", lw=0, zorder=0)
    ax[0].set_xticks(range(len(ROIS)))
    ax[0].set_xticklabels([l for _, l in ROIS], rotation=45, ha="right")
    ax[0].set_xlim(-.5, len(ROIS) - .4)
    ax[0].set_ylim(-.9, .9)
    ax[0].set_ylabel("ρ(wrap index, bid noise)")
    ax[0].set_title("Code shape predicts bid noise, with opposite signs",
                    loc="left", fontweight="bold")
    ax[0].text(.02, .95, "Circle fits better: noisier bids", transform=ax[0].transAxes,
               fontsize=6, color="0.4", va="top")
    ax[0].text(.02, .05, "Circle fits better: more precise bids", transform=ax[0].transAxes,
               fontsize=6, color="0.4")
    for i, (lab, mfc) in enumerate((("Smoothed", COL), ("Unsmoothed", "white"))):
        y = .2 - i * .08
        ax[0].plot(.78, y, "o", ms=3.5, mfc=mfc, mew=.8, color=COL,
                   transform=ax[0].transAxes)
        ax[0].text(.81, y, lab, transform=ax[0].transAxes, fontsize=6, va="center")

    for a, (roi, lab) in zip(ax[1:], (("BensonV3", "V3"), ("IPS0", "IPS0"))):
        x = d[d.smoothed & (d.roi == roi)]
        r, p = stats.spearmanr(x.wrap, x.total_noise)
        a.plot(x.wrap, x.total_noise, "o", ms=3, color=COL, alpha=.7, mew=0)
        b = np.polyfit(x.wrap, x.total_noise, 1)
        xx = np.linspace(x.wrap.min(), x.wrap.max(), 10)
        a.plot(xx, np.polyval(b, xx), color="0.3", lw=1)
        a.axvline(0, color="0.8", lw=.8, zorder=0)
        a.set_title(lab, loc="left", fontweight="bold")
        a.text(.97, .95, f"ρ = {r:.2f}, p = {p:.2g}", transform=a.transAxes,
               ha="right", va="top", fontsize=6)
        a.set_xlabel("Wrap index (pp cvR²)")
        a.text(.5, -.36, "Line fits better  |  circle fits better", transform=a.transAxes,
               fontsize=5.5, color="0.4", ha="center")
    ax[1].set_ylabel("Bid noise (model total)")
    for a, l in zip(ax, "abc"):
        a.text(-.15 if a is ax[0] else -.3, 1.1, l, transform=a.transAxes,
               fontsize=9, fontweight="bold")
    pd.DataFrame(rows).to_csv(REPO / "notes/data/wrap_behavior_correlations.tsv",
                              sep="\t", index=False)
    out = Path(out)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=250)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quantile-tsv", default=str(REPO / "notes/data/quantile_models_per_subject.tsv"))
    ap.add_argument("--params", default=str(REPO / "notes/data/categorical_freeprior_n35_subject_params.tsv"))
    ap.add_argument("--out", default=str(REPO / "notes/figures/fig_wrap_behavior.pdf"))
    a = ap.parse_args()
    main(a.quantile_tsv, a.params, a.out)
