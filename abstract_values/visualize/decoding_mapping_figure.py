"""Decoding under the two orientation-to-value mappings, in one figure.

The panels are ordered as the argument runs:

  a  what the two mappings actually present — same orientations, same mean
     value, different value SPREAD (10.3 vs 12.3 CHF). Everything else on the
     page follows from this.
  b  raw mean absolute error per ROI. inverse_cdf looks worse in every ROI,
     which is the trap: wider stimuli make larger absolute errors at equal
     correlation.
  c  the same error divided by the stimulus SD. The ordering reverses — cdf is
     the relatively harder condition.
  d  posterior width, normalised the same way: the decoder is reliably less
     sure under cdf, the same direction as c.
  e  decoded orientation is biased in opposite directions under the two
     mappings in V1, although the orientation stimulus set is identical.
  f  how well posterior width predicts trial-wise error. Positive but small:
     the posteriors carry some calibration, not much.

Reads ``notes/figures/decoding_by_roi_condition.tsv`` (written by
``decoding_by_roi_condition.py``) plus the decoded posteriors themselves for
panel a.

Usage
-----
    python -m abstract_values.visualize.decoding_mapping_figure
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from abstract_values.utils.data import BIDS_FOLDER
from abstract_values.visualize.decoding_by_roi_condition import mapping_for

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300,
})

COND = {"cdf": "#3B5BA5", "inverse_cdf": "#C44E52"}
ORDER = ["BensonV1", "NPCr", "vmPFCOFC"]


def mapping_curves(bids_folder):
    """value as a function of orientation, per mapping, from the events."""
    out = {}
    for fn in sorted(glob.glob(f"{bids_folder}/sourcedata/behavior/sub-*/"
                               f"ses-*/*events.tsv")):
        cond = "cdf" if ".cdf_" in fn else "inverse_cdf"
        if cond in out:
            continue
        d = pd.read_csv(fn, sep="\t")
        if not {"orientation", "value"} <= set(d.columns):
            continue          # practice / non-estimate runs log fewer columns
        d = d.dropna(subset=["orientation", "value"])
        d = d[d.value > 0].groupby("orientation").value.first().sort_index()
        out[cond] = d
        if len(out) == 2:
            break
    return out


def true_values_by_condition(deriv, mask="NPCr"):
    out = {}
    pat = (f"{deriv}/decoding/value/sub-*/func/"
           f"sub-*_mask-{mask}_nvoxels-100_noise-spherical_lambda-0.1_pars.tsv")
    for fn in sorted(glob.glob(pat)):
        s = re.search(r"sub-([^_]+)_", Path(fn).name).group(1)
        d = pd.read_csv(fn, sep="\t", usecols=["session", "true_value_chf"])
        d["condition"] = [mapping_for(s, x) for x in d.session]
        for cond, g in d.groupby("condition"):
            out.setdefault(cond, []).append(g.true_value_chf.to_numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def paired(df, column, roi, space="value"):
    d = df[(df.roi == roi) & (df.space == space)]
    piv = d.pivot_table(index="subject", columns="condition", values=column)
    piv = piv.dropna()
    return piv


def dots(ax, df, column, rois, space, ylabel, ylim=None):
    """Mean +/- SEM per ROI, one offset cloud per mapping."""
    present = [r for r in rois if r in set(df[df.space == space].roi)]
    x = np.arange(len(present))
    for cond, dx in (("cdf", -0.15), ("inverse_cdf", 0.15)):
        m, e = [], []
        for roi in present:
            piv = paired(df, column, roi, space)
            m.append(piv[cond].mean() if cond in piv else np.nan)
            e.append(piv[cond].std(ddof=1) / np.sqrt(len(piv))
                     if cond in piv else np.nan)
        ax.errorbar(x + dx, m, yerr=e, fmt="o", ms=3.6, lw=0, elinewidth=1.0,
                    color=COND[cond], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("Benson", "") for r in present],
                       rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.55, len(present) - 0.45)
    if ylim:
        ax.set_ylim(*ylim)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--tsv", default="notes/figures/decoding_by_roi_condition.tsv")
    p.add_argument("--out", default="notes/figures/decoding_mapping.pdf")
    args = p.parse_args()

    df = pd.read_csv(args.tsv, sep="\t", dtype={"subject": str})
    deriv = Path(args.bids_folder) / "derivatives"
    n = df[df.space == "value"].subject.nunique()

    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.6))
    a, b, c, d, e, f = axes.ravel()

    # (a) the design fact: same gabors, same mean value, different spread
    tv = true_values_by_condition(deriv)
    curves = mapping_curves(args.bids_folder)
    for i, cond in enumerate(("cdf", "inverse_cdf")):
        cur = curves[cond]
        a.plot(cur.index, cur.values, "o-", ms=2.4, color=COND[cond], lw=1.2)
        a.text(0.04, 0.96 - 0.11 * i, f"{cond} · SD {tv[cond].std():.1f} CHF",
               transform=a.transAxes, color=COND[cond], fontsize=6.5,
               fontweight="bold", va="top")
    a.set_xlabel("Gabor orientation (deg)")
    a.set_ylabel("Value (CHF)")
    a.set_xticks([0, 45, 90, 135, 180])
    a.set_yticks([0, 10, 20, 30, 40])

    # (b,c,d) error and uncertainty, raw then normalised
    dots(b, df, "abs_error", ORDER, "value", "|Error| (CHF)")
    dots(c, df, "norm_abs_error", ORDER, "value", "|Error| / SD of true value")
    dots(d, df, "norm_uncertainty", ORDER, "value", "Posterior SD / SD of true value")
    for ax, ann in ((b, "Wider stimuli,\nlarger errors"),
                    (c, "Reverses once\nnormalised")):
        ax.annotate(ann, (0.97, 0.06), xycoords="axes fraction", ha="right",
                    va="bottom", fontsize=6.5, color="0.25")

    # (e) orientation bias in V1, per subject
    piv = paired(df, "bias", "BensonV1", "gabor")
    for i, (cond, dx) in enumerate((("cdf", 0), ("inverse_cdf", 1))):
        jitter = np.random.default_rng(0).uniform(-0.09, 0.09, len(piv))
        e.scatter(dx + jitter, piv[cond], s=9, alpha=.45, linewidths=0,
                  color=COND[cond], zorder=3)
        e.hlines(piv[cond].mean(), dx - 0.22, dx + 0.22, color=COND[cond],
                 lw=2.2, zorder=4)
    for y0, y1 in zip(piv["cdf"], piv["inverse_cdf"]):
        e.plot([0.1, 0.9], [y0, y1], color="0.75", lw=0.4, zorder=1)
    t = stats.ttest_rel(piv["cdf"], piv["inverse_cdf"])
    e.axhline(0, color="0.7", lw=0.6, ls=(0, (4, 3)), zorder=0)
    e.set_xticks([0, 1])
    e.set_xticklabels(["cdf", "inverse_cdf"], rotation=20, ha="right")
    e.set_ylabel("Decoded orientation bias (deg)")
    e.set_title(f"V1 · p = {t.pvalue:.3f}", fontsize=7)
    e.set_xlim(-0.45, 1.45)

    # (f) calibration
    dots(f, df, "unc_err_r", ORDER, "value", "Posterior SD ~ |error| (r)")
    f.axhline(0, color="0.7", lw=0.6, ls=(0, (4, 3)), zorder=0)

    for ax, letter in zip(axes.ravel(), "abcdef"):
        ax.text(-0.22, 1.06, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="left")
        for side in ("left", "bottom"):
            ax.spines[side].set_position(("outward", 4))
    fig.suptitle(f"Value decoding under the two mappings — n={n}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=130)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
