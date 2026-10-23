"""Hierarchical lambda vs voxel selection.

Columns: ROI groups. Rows: group mean mu (95% CrI); between-participant SD tau;
Spearman rho of the per-participant posterior means with the cognitive
model's value noise (sigma_rep, shrink + motor-tail fit); participants kept.
x: the cvR^2-over-null threshold. Colours: the gate (which aPRF's cvR^2 is
compared with the null). Filled markers: p < .05.

Input: ``encoding_models/hierlambda_selection_sweep.py`` summary.

    python -m abstract_values.visualize.figure_hierlambda_selection
    -> notes/figures/fig_hierlambda_selection.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from abstract_values.visualize import qc_value_shift as q   # noqa: F401 house rcParams
from abstract_values.visualize.figure_tuning_connectivity import ORI_COL, VAL_COL

REPO = Path(__file__).resolve().parents[2]
GATE = {"std": ("Value aPRF", VAL_COL), "neutral": ("Best of value / quantile", "0.35"),
        "shift": ("Session-shift aPRF", "#2E5E73")}
ROIS = {"IPS": "IPS (pooled)", "IPS0": "IPS0", "IPS12": "IPS1–2", "IPS345": "IPS3–5"}
DX = {"std": -.0004, "neutral": 0., "shift": .0004}


def main(out=REPO / "notes/figures/fig_hierlambda_selection.pdf"):
    d = pd.read_csv(REPO / "notes/data/hierlambda_selection/summary.tsv", sep="\t")
    fig, axes = plt.subplots(4, 4, figsize=(7.25, 6.6), sharex=True,
                             gridspec_kw=dict(hspace=.35, wspace=.35, left=.09, right=.98,
                                              top=.93, bottom=.08,
                                              height_ratios=[1.4, 1, 1, .8]))
    for c, (roi, name) in enumerate(ROIS.items()):
        for sel, (lab, col) in GATE.items():
            x = d[(d.roi == roi) & (d.select == sel)].sort_values("min_gain")
            g = x.min_gain + DX[sel]
            ax = axes[0, c]
            ax.vlines(g, x.mu_lo, x.mu_hi, color=col, lw=.8)
            ax.plot(g, x.mu, "o-", color=col, ms=3, lw=1)
            axes[1, c].plot(g, x.tau, "o-", color=col, ms=3, lw=1)
            ax = axes[2, c]
            ax.plot(g, x.rho_sigma_rep, "-", color=col, lw=1)
            sig = x.p_sigma_rep < .05
            ax.plot(g[sig], x.rho_sigma_rep[sig], "o", color=col, ms=3.4)
            ax.plot(g[~sig], x.rho_sigma_rep[~sig], "o", mfc="white", mec=col, ms=3.4,
                    mew=.8)
            axes[3, c].plot(g, x.n_subjects, "o-", color=col, ms=3, lw=1)
        axes[0, c].axhline(0, color=VAL_COL, lw=.6, alpha=.6, zorder=0)
        axes[0, c].axhline(1, color=ORI_COL, lw=.6, alpha=.6, zorder=0)
        axes[0, c].set_ylim(-1.05, 2.05)
        axes[2, c].axhline(0, color="0.8", lw=.6, zorder=0)
        axes[2, c].set_ylim(-.7, .4)
        axes[3, c].set_ylim(15, 37)
        axes[0, c].set_title(name, loc="left", fontweight="bold")
        axes[-1, c].set_xlabel("cvR² over null >")
        axes[-1, c].set_xticks([0, .01, .02])
    axes[0, 0].set_ylabel("Group mean μ")
    axes[0, 0].text(.021, 1.05, "Quantile", color=ORI_COL, fontsize=6, ha="right", va="bottom")
    axes[0, 0].text(.021, .05, "Value", color=VAL_COL, fontsize=6, ha="right", va="bottom")
    axes[1, 0].set_ylabel("Spread τ")
    axes[2, 0].set_ylabel("ρ with value noise\n(cognitive model)")
    axes[3, 0].set_ylabel("Participants")
    for k, (lab, col) in enumerate(GATE.values()):
        fig.text(.09 + k * .25, .975, "— " + lab, color=col, fontsize=7, va="center")
    fig.text(.86, .975, "Filled: p < .05", fontsize=6.5, color="0.4", va="center")
    out = Path(out)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=250)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
