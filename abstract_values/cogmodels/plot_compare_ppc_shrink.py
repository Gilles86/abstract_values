"""Mapping shrinkage: group PPCs, residuals, LOO and parameter recovery (categorical, n = 35).

Reads compare_ppc.py output for four fits (fixed / free prior steepness, each
with and without mapping shrinkage) from notes/data/compare_ppc_shrink/, and the
recovery fit (data simulated with known lambda) from the local cogmodels_n35
copy.

  a  bid error and bid SD per orientation, per mapping: observed points over
     the 95% predictive band of the free-prior fit without (grey) and with
     (orange) shrinkage
  b  residuals (observed - predicted mean) for all four fits
  c  ELPD relative to the fixed prior
  d  recovered vs true lambda
  e  fitted lambda per participant

    python -m abstract_values.cogmodels.plot_compare_ppc_shrink    # abstract_values env
    -> notes/figures/fig_ppc_mapping_shrinkage.pdf
"""
from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abstract_values.visualize.plot_vmpfc_dropout import style

REPO = Path(__file__).resolve().parents[2]
DATA, FIG = REPO / "notes/data", REPO / "notes/figures"
IN = DATA / "compare_ppc_shrink"
FITS = Path("/data/ds-abstractvalue/derivatives/cogmodels_n35")
TAG = "efficient_coding_categorical_freeprior_grid161_trunc_hn_total-share_erf_shrink"
MAP_COL = {"cdf": "#5B4A9E", "inverse_cdf": "#B23A6E"}
MAP_LAB = {"cdf": "CDF", "inverse_cdf": "Inverse CDF"}
FIX, FREE = "Fixed prior", "Free prior steepness"
SHR, BOTH = "Mapping shrinkage", "Free prior + shrinkage"
FIT_COL = {FIX: "0.7", FREE: "0.35", SHR: "#E8A867", BOTH: "#C2560A"}
FIT_LS = {FIX: (0, (3, 2)), FREE: "-", SHR: (0, (3, 2)), BOTH: "-"}


def stats_by(grp, fit, m, stat):
    s = grp[(grp.fit == fit) & (grp.mapping == m)]
    sim = s[s.draw >= 0].groupby("orientation")[stat]
    q = sim.quantile([0.025, 0.975]).unstack()
    ob = s[s.draw < 0].set_index("orientation")[stat]
    return q, sim.mean(), ob


def main():
    style()
    grp = pd.read_csv(IN / "ppc_compare_group.tsv", sep="\t")
    loo = pd.read_csv(IN / "ppc_compare_loo.tsv", sep="\t").set_index("fit")

    fig = plt.figure(figsize=(7.25, 9.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.6, 1.25], hspace=0.42,
                          left=0.11, right=0.98, top=0.96, bottom=0.055)

    # a: bands, free prior with and without shrinkage
    ga = gs[0].subgridspec(2, 2, hspace=0.15, wspace=0.14)
    for i, (stat, yl) in enumerate((("bias", "Bid − value (CHF)"), ("sd", "Bid SD (CHF)"))):
        for j, m in enumerate(("cdf", "inverse_cdf")):
            ax = fig.add_subplot(ga[i, j])
            cov = {}
            for fit, alpha in ((FREE, 0.3), (BOTH, 0.35)):
                q, _, ob = stats_by(grp, fit, m, stat)
                ax.fill_between(q.index, q[0.025], q[0.975], color=FIT_COL[fit], alpha=alpha, lw=0)
                cov[fit] = ((ob >= q[0.025]) & (ob <= q[0.975])).mean()
            ax.plot(ob.index, ob.values, "o", ms=3, color=MAP_COL[m], mec="white", mew=0.4,
                    zorder=3)
            if stat == "bias":
                ax.axhline(0, color="0.7", lw=0.6, ls=(0, (3, 2)))
                ax.set_title(MAP_LAB[m] + " sessions", loc="left", color=MAP_COL[m],
                             fontweight="bold", fontsize=7.5)
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Orientation (°)")
            ax.set_xticks([0, 45, 90, 135, 180]); ax.set_xlim(0, 180)
            if j == 0:
                ax.set_ylabel(yl)
            ax.text(0.99, 0.03, f"Inside 95% band: {100 * cov[FREE]:.0f}% without, "
                    f"{100 * cov[BOTH]:.0f}% with shrinkage", transform=ax.transAxes, ha="right",
                    fontsize=5.8, color="0.35")
            sns.despine(ax=ax, offset=3)
            if i == 0 and j == 0:
                ax.text(-0.14, 1.08, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")
                for k, fit in enumerate((FREE, BOTH)):
                    ax.add_patch(plt.Rectangle((0.03, 0.9 - k * 0.1), 0.06, 0.06,
                                               transform=ax.transAxes, color=FIT_COL[fit],
                                               alpha=0.45, lw=0))
                    ax.text(0.11, 0.93 - k * 0.1, fit, transform=ax.transAxes, fontsize=6.3,
                            va="center", color="0.25")

    # b: residuals, all four fits
    gb = gs[1].subgridspec(2, 2, hspace=0.15, wspace=0.14)
    for i, (stat, yl) in enumerate((("bias", "Residual error (CHF)"),
                                    ("sd", "Residual SD (CHF)"))):
        for j, m in enumerate(("cdf", "inverse_cdf")):
            ax = fig.add_subplot(gb[i, j])
            for fit in (FIX, FREE, SHR, BOTH):
                _, mu, ob = stats_by(grp, fit, m, stat)
                r = ob - mu
                ax.plot(r.index, r.values, color=FIT_COL[fit], ls=FIT_LS[fit], lw=1.1)
                if i == 0 and j == 1:
                    pass
            ax.axhline(0, color="0.6", lw=0.6)
            ax.set_xticks([0, 45, 90, 135, 180]); ax.set_xlim(0, 180)
            if i == 0:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Orientation (°)")
            if j == 0:
                ax.set_ylabel(yl)
            sns.despine(ax=ax, offset=3)
            if i == 0 and j == 0:
                ax.text(-0.14, 1.08, "b", transform=ax.transAxes, fontsize=9, fontweight="bold")
    for k, fit in enumerate((FIX, FREE, SHR, BOTH)):
        x0 = 0.3 + k * 0.17
        fig.add_artist(plt.Line2D([x0, x0 + 0.03], [0.588, 0.588], color=FIT_COL[fit],
                                  ls=FIT_LS[fit], lw=1.1, transform=fig.transFigure))
        fig.text(x0 + 0.035, 0.588, fit, fontsize=6, va="center", color="0.25")
    fig.text(0.11, 0.588, "Observed − predicted mean:", fontsize=6, color="0.4", va="center")

    # c: ELPD ; d: recovery ; e: lambda
    gc = gs[2].subgridspec(1, 3, wspace=0.5, width_ratios=[1.0, 1, 1])
    ax = fig.add_subplot(gc[0])
    order = [FIX, FREE, SHR, BOTH]
    base = loo.loc[FIX, "elpd_loo"]
    gain = [loo.loc[f, "elpd_loo"] - base for f in order]
    se = [0 if f == FIX else np.nan for f in order]
    ax.barh(range(4), gain, color=[FIT_COL[f] for f in order], height=0.6)
    best = loo.index[loo["rank"] == 0][0]
    for k, f in enumerate(order):
        txt = f"+{gain[k]:.0f}" if gain[k] > 0 else "0"
        ax.text(gain[k] + 8, k, txt, va="center", fontsize=6.2, color="0.25")
    ax.set_yticks(range(4))
    ax.set_yticklabels(["Fixed prior", "Free w", "Shrinkage", "Free w +\nshrinkage"],
                       fontsize=6.3)
    ax.invert_yaxis()
    ax.set_xlabel("ELPD vs fixed prior")
    ax.set_xlim(0, max(gain) * 1.25)
    d = loo.loc[FREE]
    ax.set_title(f"Free w + shrinkage beats free w\nby {d.elpd_diff - loo.loc[best, 'elpd_diff']:.0f}"
                 f" ± {d.dse:.0f} (PSIS-LOO)", loc="left", fontsize=6.3, color="0.3")
    ax.text(-0.6, 1.12, "c", transform=ax.transAxes, fontsize=9, fontweight="bold")
    sns.despine(ax=ax, offset=3)
    del se

    truth = pd.read_csv(DATA / "efficient_coding_paradigm_n35_sim-shrink_truth.tsv", sep="\t"
                        ).set_index("subject").mapping_shrinkage
    rec = az.from_netcdf(FITS / "recovery_shrink" / f"{TAG}_trace.nc").posterior.mapping_shrinkage
    rq = rec.quantile([0.025, 0.5, 0.975], dim=("chain", "draw")).to_pandas().T
    rq.index = rq.index.astype(int)
    rq = rq.loc[truth.index]
    ax = fig.add_subplot(gc[1])
    ax.vlines(truth, rq[0.025], rq[0.975], color="#C2560A", lw=0.6, alpha=0.6)
    ax.plot(truth, rq[0.5], "o", ms=3, color="#C2560A", mec="white", mew=0.4)
    ax.plot([0, 0.6], [0, 0.6], color="0.7", lw=0.6, ls=(0, (3, 2)))
    ax.set_xlabel("True λ (simulated)")
    ax.set_ylabel("Recovered λ")
    ax.set_title(f"Parameter recovery: r = {np.corrcoef(truth, rq[0.5])[0, 1]:.2f}",
                 loc="left", fontsize=6.3, color="0.3")
    ax.text(-0.38, 1.12, "d", transform=ax.transAxes, fontsize=9, fontweight="bold")
    sns.despine(ax=ax, offset=3)

    lam = az.from_netcdf(FITS / f"{TAG}_trace.nc").posterior.mapping_shrinkage
    lq = lam.quantile([0.025, 0.5, 0.975], dim=("chain", "draw")).to_pandas().T.sort_values(0.5)
    ax = fig.add_subplot(gc[2])
    y = np.arange(len(lq))
    ax.hlines(y, lq[0.025], lq[0.975], color="#C2560A", lw=0.7, alpha=0.6)
    ax.plot(lq[0.5], y, "o", ms=2.6, color="#C2560A", mec="white", mew=0.3)
    ax.set_yticks([])
    ax.set_ylabel("Participants (sorted)")
    ax.set_xlabel("λ (shrinkage toward the line)")
    ax.set_title("Fitted λ, free prior + shrinkage", loc="left", fontsize=6.3, color="0.3")
    ax.text(-0.2, 1.12, "e", transform=ax.transAxes, fontsize=9, fontweight="bold")
    sns.despine(ax=ax, offset=3, left=True)

    fig.savefig(FIG / "fig_ppc_mapping_shrinkage.pdf")
    fig.savefig(FIG / "fig_ppc_mapping_shrinkage.png", dpi=200)
    print(loo[["elpd_loo", "elpd_diff", "dse", "p_loo"]].round(1).to_string())
    print(f"Wrote {FIG / 'fig_ppc_mapping_shrinkage.pdf'}")


if __name__ == "__main__":
    main()
