"""How do per-participant parameters move when a model component is added?

Scatter of posterior means (95% CrI whiskers both ways) of the shared
per-participant parameters of two efficient-coding fits of the same data, one
panel per parameter, with the identity line, Spearman rho and the median
change.

    python -m abstract_values.cogmodels.plot_param_shift \\
        --a "No offsets=<..._shrink_freelapse_trace.nc>" \\
        --b "Spline offsets=<..._shrink_splineoffsets_freelapse_trace.nc>"
    -> notes/figures/fig_param_shift_<b-tag>.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from abstract_values.visualize.plot_vmpfc_dropout import style

REPO = Path(__file__).resolve().parents[2]
PARAMS = [("kappa_r", "Perceptual precision κ_r", True),
          ("sigma_rep", "Value noise σ_rep", True),
          ("perceptual_share", "Perceptual share of noise", False),
          ("prior_weight", "Prior steepness w", False),
          ("mapping_shrinkage", "Mapping shrinkage λ", False),
          ("lapse", "Lapse rate", True)]


def quantiles(trace, name):
    post = az.from_netcdf(trace).posterior
    if name not in post:
        return None
    q = post[name].quantile([0.025, 0.5, 0.975], dim=("chain", "draw")).to_pandas().T
    m = post[name].mean(("chain", "draw")).to_pandas()
    q["mean"] = m
    q.index = q.index.astype(int)
    return q


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="label=trace (reference fit)")
    p.add_argument("--b", required=True, help="label=trace (extended fit)")
    a = p.parse_args()
    (la, ta), (lb, tb) = (s.split("=", 1) for s in (a.a, a.b))
    style()
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.9), constrained_layout=True)
    for ax, (name, lab, log) in zip(axes.ravel(), PARAMS):
        qa, qb = quantiles(ta, name), quantiles(tb, name)
        if qa is None or qb is None:
            ax.set_axis_off()
            continue
        s = qa.index.intersection(qb.index)
        qa, qb = qa.loc[s], qb.loc[s]
        ax.errorbar(qa["mean"], qb["mean"],
                    xerr=[qa["mean"] - qa[0.025], qa[0.975] - qa["mean"]],
                    yerr=[qb["mean"] - qb[0.025], qb[0.975] - qb["mean"]],
                    fmt="o", ms=3, color="#C2560A", ecolor="#C2560A", elinewidth=0.4,
                    alpha=0.75, mec="white", mew=0.3)
        lo = min(qa[0.025].min(), qb[0.025].min())
        hi = max(qa[0.975].max(), qb[0.975].max())
        if log:
            ax.set_xscale("log"); ax.set_yscale("log")
            lo = max(lo, 1e-4)
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.6, ls=(0, (3, 2)), zorder=0)
        rho = stats.spearmanr(qa["mean"], qb["mean"]).statistic
        chg = (np.median(qb["mean"] / qa["mean"]) - 1) * 100
        ax.set_title(f"{lab}\nρ = {rho:.2f}, median change {chg:+.0f}%", loc="left",
                     fontsize=7)
        ax.set_xlabel(la)
        ax.set_ylabel(lb)
        sns.despine(ax=ax, offset=3)
    tag = Path(tb).name.replace("efficient_coding_", "").replace("_trace.nc", "")
    out = REPO / f"notes/figures/fig_param_shift_{tag}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
