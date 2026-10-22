"""lambda-hat per participant (fine mix-lambda profile, run-bootstrap 95% CI).

Rows: unsmoothed / smoothed; columns: ROIs. Each participant is a row of the
forest, sorted by lambda-hat; the band on top is the group-mean cvR^2 profile
(relative to lambda = 0) with its argmax. Input: per-participant TSVs from
``encoding_models/fit_lambda_profile.py`` (rsynced into --in-dir).

Usage:
    python -m abstract_values.visualize.figure_lambda_distribution \
        --in-dir notes/data/lambda_profile
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from abstract_values.visualize import qc_value_shift as q   # noqa: F401 house rcParams
from abstract_values.visualize.figure_tuning_connectivity import ORI_COL, ROI_COL, ROI_NAME, VAL_COL

REPO = Path(__file__).resolve().parents[2]
ROIS = ["IPS0", "IPS12", "IPS345", "V3ab", "hV4"]


def read(in_dir, kind):
    fs = sorted(Path(in_dir).rglob(f"*_{kind}.tsv"))
    d = pd.concat([pd.read_csv(f, sep="\t", dtype={"subject": str}) for f in fs])
    d["smoothed"] = d["smoothed"].astype(str) == "True"
    return d


def main(in_dir, out):
    hat, prof = read(in_dir, "lamhat"), read(in_dir, "profile")
    fig, ax = plt.subplots(4, len(ROIS), figsize=(7.25, 6.4), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 2.6, 1, 2.6], hspace=.12,
                                            wspace=.25, left=.08, right=.98, top=.95,
                                            bottom=.07))
    summary = []
    for r, sm in enumerate((False, True)):
        for c, roi in enumerate(ROIS):
            col = ROI_COL[roi]
            # group profile
            pa = ax[2 * r, c]
            p = prof[(prof.smoothed == sm) & (prof.roi == roi)].pivot(
                index="subject", columns="lam", values="cvr2")
            p = (p.sub(p[0.0], axis=0)) * 1e3
            m, s = p.mean(), p.sem()
            L = p.columns.to_numpy(float)
            pa.fill_between(L, m - s, m + s, color=col, alpha=.25, lw=0)
            pa.plot(L, m, color=col, lw=1.2)
            pa.axhline(0, color="0.7", lw=.5)
            pa.plot(L[np.argmax(m)], m.max(), "v", ms=3, color=col, clip_on=False)
            pa.set_yticks([])
            pa.spines["left"].set_visible(False)
            pa.spines["bottom"].set_visible(False)
            pa.tick_params(axis="x", length=0)
            if r == 0:
                pa.set_title(ROI_NAME[roi], fontweight="bold")
            # forest
            fa = ax[2 * r + 1, c]
            h = hat[(hat.smoothed == sm) & (hat.roi == roi)].sort_values("lam_hat")
            y = np.arange(len(h))
            for l, lc in ((0, VAL_COL), (1, ORI_COL)):
                fa.axvline(l, color=lc, lw=.8, alpha=.6, zorder=0)
            fa.hlines(y, h.ci_lo, h.ci_hi, color=col, lw=.8, alpha=.6)
            fa.plot(h.lam_hat, y, "o", ms=2.6, color=col, mew=0)
            med = h.lam_hat.median()
            fa.axvline(med, color="k", lw=.8, ls=(0, (2, 1.5)))
            fa.set_ylim(-1, len(h))
            fa.set_yticks([])
            fa.spines["left"].set_visible(False)
            ex0 = int((h.ci_lo > 0).sum())
            ex1 = int(((h.ci_hi < 1) | (h.ci_lo > 1)).sum())
            fa.text(.02, .99, f"Median {med:.2f}\nCI > 0: {ex0}/{len(h)}\nCI ∌ 1: {ex1}/{len(h)}",
                    transform=fa.transAxes, fontsize=5.5, va="top", color="0.3")
            summary.append(dict(smoothed=sm, roi=roi, median=med, n=len(h),
                                ci_excl0_pos=ex0, ci_excl1=ex1,
                                group_argmax=L[np.argmax(m)],
                                median_ci_width=(h.ci_hi - h.ci_lo).median()))
        ax[2 * r + 1, 0].set_ylabel(("Smoothed" if sm else "Unsmoothed")
                                    + ": participants")
        ax[2 * r, 0].set_ylabel("Group\nΔ cvR²", fontsize=6)
    for a in ax[-1]:
        a.set_xlim(-1.05, 2.05)
        a.set_xticks([-1, 0, 1, 2])
        a.set_xlabel("λ̂")
    ax[1, 0].text(0, -.02, "Fixed value", color=VAL_COL, fontsize=5.5, ha="center",
                  va="top", transform=ax[1, 0].get_xaxis_transform())
    ax[1, 0].text(1, -.02, "Quantile", color=ORI_COL, fontsize=5.5, ha="center",
                  va="top", transform=ax[1, 0].get_xaxis_transform())
    out = Path(out)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=250)
    print(f"Wrote {out}")
    print(pd.DataFrame(summary).round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=str(REPO / "notes/data/lambda_profile"))
    ap.add_argument("--out", default=str(REPO / "notes/figures/fig_lambda_distribution.pdf"))
    a = ap.parse_args()
    main(a.in_dir, a.out)
