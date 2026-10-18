"""What do participants' internal ('wrong') orientation -> value mappings look like?

From an efficient-coding fit with mapping shrinkage and/or mapping offsets,
reconstructs each participant's internal mapping per session,

    G_int(theta) = (1 - lambda) G(theta) + lambda L(theta) + B(theta) . offset

(L the straight line 2 -> 42 CHF, B the offset basis of the fit: per stimulus or
cubic B-spline) for every posterior draw, and plots it as the deviation from the
true mapping, G_int - G:

  a  every participant (thin) and the group mean (thick), per mapping
  b  six example participants (largest deviations), 95% CrI band, with the
     observed mean bid error per orientation for reference (which also carries
     the perceptual-stage bias, so it need not match the band)

    python -m abstract_values.cogmodels.plot_internal_mappings --trace <..._trace.nc>
    -> notes/figures/fig_internal_mappings_<tag>.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abstract_values.cogmodels.fit_efficient_coding import make_model
from abstract_values.cogmodels.ppc_efficient_coding import load_paradigm
from abstract_values.visualize.plot_vmpfc_dropout import style

REPO = Path(__file__).resolve().parents[2]
MAP_COL = {"cdf": "#5B4A9E", "inverse_cdf": "#B23A6E"}
MAP_LAB = {"cdf": "CDF sessions", "inverse_cdf": "Inverse-CDF sessions"}


def internal_deviation(trace, paradigm_tsv, n_draws=400):
    """(draw, subject, mapping, theta) deviation G_int - G on the 161-point grid."""
    from bauer.efficient_coding import linear_value_np
    idata = az.from_netcdf(trace)
    post = idata.posterior
    attrs = post.attrs
    basis = attrs.get("fit_mapping_offsets", "") or ""
    par = load_paradigm(paradigm_tsv)
    subs = [int(s) for s in post.subject.values]
    par = par[par.index.get_level_values("subject").isin(subs)]
    m = make_model(par, "categorical", int(attrs.get("grid_resolution", 161)),
                   cardinal_truncation=True, fit_mapping_offsets=basis or False)
    m._setup_grids(par)
    theta = np.append(m.ori_grid / (2 * np.pi) * 180, 180.0)
    maps = list(m.value_on_ori_grid)
    G = np.stack([m.value_on_ori_grid_ext[k] for k in maps])            # (C, O+1)
    L = linear_value_np(theta)
    flat = lambda v: v.values.reshape(-1, *v.shape[2:])
    rng = np.random.default_rng(0)
    nd = post.sizes["chain"] * post.sizes["draw"]
    pick = rng.choice(nd, min(n_draws, nd), replace=False)
    dev = np.zeros((len(pick), len(subs), len(maps), len(theta)))
    if "mapping_shrinkage" in post:
        lam = flat(post.mapping_shrinkage)[pick]                       # (D, S)
        dev += lam[:, :, None, None] * (L[None, None, None] - G[None, None])
    if "mapping_offset" in post:
        off = post.mapping_offset
        order = [list(map(str, off.offset_mapping.values)).index(k) for k in maps]
        o = flat(off)[pick][:, :, order]                               # (D, S, C, J)
        dev += o @ m.offset_basis.T
    return dev, theta, maps, subs, G


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True)
    p.add_argument("--paradigm-tsv", default=str(REPO / "notes/data/efficient_coding_paradigm_n35.tsv"))
    a = p.parse_args()
    style()
    dev, theta, maps, subs, G = internal_deviation(a.trace, a.paradigm_tsv)
    mean = dev.mean(0)                                                 # (S, C, T)
    lo, hi = np.percentile(dev, [2.5, 97.5], axis=0)

    d = pd.read_csv(a.paradigm_tsv, sep="\t")
    obs = (d.assign(err=d.response - d.value)
             .groupby(["subject", "mapping", "orientation"]).err.mean())

    tag = Path(a.trace).name.replace("efficient_coding_", "").replace("_trace.nc", "")
    fig = plt.figure(figsize=(7.25, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 2.0], hspace=0.35,
                          left=0.09, right=0.98, top=0.93, bottom=0.07)
    ga = gs[0].subgridspec(1, 2, wspace=0.18)
    for j, mp in enumerate(maps):
        ax = fig.add_subplot(ga[j])
        for s in range(len(subs)):
            ax.plot(theta, mean[s, j], color=MAP_COL[mp], lw=0.5, alpha=0.3)
        ax.plot(theta, mean[:, j].mean(0), color=MAP_COL[mp], lw=2.0)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_title(MAP_LAB[mp], loc="left", color=MAP_COL[mp], fontweight="bold",
                     fontsize=7.5)
        ax.set_xlabel("Orientation (°)")
        if j == 0:
            ax.set_ylabel("Internal − true\nmapping (CHF)")
            ax.text(-0.17, 1.1, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")
            ax.text(0.02, 0.04, "Thin: participants; thick: mean", transform=ax.transAxes,
                    fontsize=6, color="0.4")
        sns.despine(ax=ax, offset=3)

    size = np.sqrt((mean ** 2).mean((1, 2)))
    ex = np.argsort(size)[::-1][:6]
    gb = gs[1].subgridspec(2, 3, hspace=0.45, wspace=0.25)
    for k, s in enumerate(ex):
        ax = fig.add_subplot(gb[k // 3, k % 3])
        for j, mp in enumerate(maps):
            ax.fill_between(theta, lo[s, j], hi[s, j], color=MAP_COL[mp], alpha=0.2, lw=0)
            ax.plot(theta, mean[s, j], color=MAP_COL[mp], lw=1.2)
            if (subs[s], mp) in obs.index.droplevel(2).unique():
                o = obs.loc[(subs[s], mp)]
                ax.plot(o.index, o.values, "o", ms=2.2, color=MAP_COL[mp], mec="white",
                        mew=0.3, alpha=0.8)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xlim(0, 180); ax.set_xticks([0, 90, 180])
        ax.set_title(f"sub-{subs[s]:02d}", loc="left", fontsize=7)
        if k % 3 == 0:
            ax.set_ylabel("Internal − true (CHF)")
        if k >= 3:
            ax.set_xlabel("Orientation (°)")
        if k == 0:
            ax.text(-0.3, 1.12, "b", transform=ax.transAxes, fontsize=9, fontweight="bold")
            ax.text(0.02, 0.04, "Line ± band: internal mapping; dots: mean bid error",
                    transform=ax.transAxes, fontsize=5.6, color="0.4")
        sns.despine(ax=ax, offset=3)
    fig.suptitle("Participants' internal orientation-to-value mappings, relative to the true one",
                 x=0.09, ha="left", fontsize=8, fontweight="bold")
    out = REPO / f"notes/figures/fig_internal_mappings_{tag}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"RMS internal-true deviation per participant: median {np.median(size):.2f} CHF, "
          f"range {size.min():.2f}-{size.max():.2f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
