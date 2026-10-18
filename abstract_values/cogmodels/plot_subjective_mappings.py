"""The objective orientation -> value mappings against participants' subjective ones.

The subjective (internal) mapping of each participant and session type is the
one an efficient-coding fit with mapping shrinkage and/or offsets implies,
G_int = (1 - lambda) G + lambda L + B . offset (see plot_internal_mappings).

  a  per mapping: the objective mapping (black), every participant's subjective
     mapping (thin) and their mean (thick), in CHF
  b  every participant, both mappings: subjective mapping with its 95% CrI
     over the objective one, and the observed mean bid per orientation

    python -m abstract_values.cogmodels.plot_subjective_mappings --trace <..._trace.nc>
    -> notes/figures/fig_subjective_mappings_<tag>.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abstract_values.cogmodels.plot_internal_mappings import (MAP_COL, REPO,
                                                              internal_deviation)
from abstract_values.visualize.plot_vmpfc_dropout import style

MAP_LAB = {"cdf": "CDF mapping", "inverse_cdf": "Inverse-CDF mapping"}
ORDER = ("cdf", "inverse_cdf")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True)
    p.add_argument("--paradigm-tsv", default=str(REPO / "notes/data/efficient_coding_paradigm_n35.tsv"))
    a = p.parse_args()
    style()
    dev, theta, maps, subs, G = internal_deviation(a.trace, a.paradigm_tsv)
    Gs = {mp: G[maps.index(mp)] for mp in ORDER}
    subj = {mp: G[maps.index(mp)][None, None] + dev[:, :, maps.index(mp)] for mp in ORDER}
    d = pd.read_csv(a.paradigm_tsv, sep="\t")
    obs = d.groupby(["subject", "mapping", "orientation"]).response.mean()

    ncol = 7
    nrow = int(np.ceil(len(subs) / ncol))
    fig = plt.figure(figsize=(7.25, 3.0 + 1.12 * nrow))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.12 * nrow], hspace=0.17,
                          left=0.08, right=0.985, top=0.965, bottom=0.04)

    ga = gs[0].subgridspec(1, 2, wspace=0.14)
    for j, mp in enumerate(ORDER):
        ax = fig.add_subplot(ga[j])
        mean = subj[mp].mean(0)                                         # (S, T)
        for s in range(len(subs)):
            ax.plot(theta, mean[s], color=MAP_COL[mp], lw=0.5, alpha=0.35)
        ax.plot(theta, mean.mean(0), color=MAP_COL[mp], lw=2.0)
        ax.plot(theta, Gs[mp], color="k", lw=1.1, ls=(0, (4, 2)))
        ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_ylim(0, 44); ax.set_yticks([2, 12, 22, 32, 42])
        ax.set_xlabel("Orientation (°)")
        if j == 0:
            ax.set_ylabel("Value (CHF)")
            ax.text(-0.15, 1.04, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.set_title(MAP_LAB[mp], loc="left", color=MAP_COL[mp], fontweight="bold",
                     fontsize=7.5)
        tf = ax.transAxes
        for i, (lab, col, lw, ls) in enumerate((("Objective", "k", 1.1, (0, (4, 2))),
                                                 ("Subjective, mean", MAP_COL[mp], 2.0, "-"),
                                                 ("Subjective, participants", MAP_COL[mp],
                                                  0.6, "-"))):
            yy = 0.95 - i * 0.075
            ax.plot([0.04, 0.12], [yy, yy], transform=tf, color=col, lw=lw, ls=ls,
                    alpha=0.6 if i == 2 else 1)
            ax.text(0.14, yy, lab, transform=tf, fontsize=6.3, va="center", color="0.25")
        sns.despine(ax=ax, offset=3)

    rms = np.sqrt((dev.mean(0) ** 2).mean((1, 2)))
    order = np.argsort(rms)
    gb = gs[1].subgridspec(nrow, ncol, hspace=0.42, wspace=0.12)
    for k, si in enumerate(order):
        ax = fig.add_subplot(gb[k // ncol, k % ncol])
        s = subs[si]
        for mp in ORDER:
            lo, hi = np.percentile(subj[mp][:, si], [2.5, 97.5], axis=0)
            ax.plot(theta, Gs[mp], color=MAP_COL[mp], lw=0.6, ls=(0, (2, 1.5)), alpha=0.9)
            ax.fill_between(theta, lo, hi, color=MAP_COL[mp], alpha=0.3, lw=0)
            ax.plot(theta, subj[mp][:, si].mean(0), color=MAP_COL[mp], lw=0.8)
            if (s, mp) in obs.index.droplevel(2):
                o = obs.loc[(s, mp)]
                ax.plot(o.index, o.values, "o", ms=1.4, color=MAP_COL[mp], mew=0)
        ax.set_xlim(0, 180); ax.set_ylim(0, 44)
        ax.set_xticks([0, 90, 180]); ax.set_yticks([2, 22, 42])
        if k % ncol:
            ax.set_yticklabels([])
        if k < len(subs) - ncol:
            ax.set_xticklabels([])
        ax.tick_params(labelsize=5.5, length=2)
        ax.text(0.03, 0.97, f"sub-{s:02d}", transform=ax.transAxes, fontsize=5.6,
                va="top", color="0.3")
        sns.despine(ax=ax, offset=2)
        if k == 0:
            ax.text(-0.55, 1.18, "b", transform=ax.transAxes, fontsize=9, fontweight="bold")
            ax.text(0.0, 1.18, "Participants, from closest to furthest from the objective "
                    "mapping (dashed: objective, solid with band: subjective, dots: mean bid)",
                    transform=ax.transAxes, fontsize=6.3, color="0.3")
    tag = Path(a.trace).name.replace("efficient_coding_", "").replace("_trace.nc", "")
    out = REPO / f"notes/figures/fig_subjective_mappings_{tag}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
