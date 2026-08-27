"""Per-subject parameter estimates from an efficient-coding fit.

Forest plots of the posterior (median + 95% HDI) per subject, plus the
comparison that says whether the two-stage model is doing anything the
one-stage models were not.

Writes notes/figures/params_<tag>.pdf.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})

K_COL, S_COL = "#3B5BA5", "#C44E52"


def forest(ax, samples, subjects, color, xlabel, log=False, ref=None, ref_label=None):
    """Median + 95% HDI per subject, sorted by median."""
    med = np.median(samples, axis=0)
    hdi = np.stack([az.hdi(samples[:, i], hdi_prob=0.95) for i in range(samples.shape[1])])
    order = np.argsort(med)
    y = np.arange(len(order))
    ax.hlines(y, hdi[order, 0], hdi[order, 1], color=color, lw=1.1, alpha=0.75)
    ax.plot(med[order], y, "o", ms=3.2, color=color, mec="white", mew=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{subjects[i]:02d}" for i in order], fontsize=5.5)
    ax.set_ylim(-1, len(y))
    if log:
        ax.set_xscale("log")
    if ref is not None:
        ax.axvline(ref, color="0.6", lw=0.7, ls="--", zorder=0)
        ax.text(ref, len(y) - 0.5, "  " + ref_label, color="0.45", fontsize=6.5,
                va="top", ha="left")
    ax.set_xlabel(xlabel)
    return med


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True)
    p.add_argument("--compare-kappa", default=None,
                   help="Trace of the perception-only fit, for the reallocation panel.")
    p.add_argument("--compare-sigma", default=None,
                   help="Trace of the valuation-only fit.")
    p.add_argument("--grid-resolution", type=int, default=101)
    p.add_argument("--out", default="notes/figures/params_sequential.pdf")
    a = p.parse_args()

    d = az.from_netcdf(a.trace)
    subs = [int(s) for s in d.posterior.kappa_r.coords["subject"].values]
    k = d.posterior.kappa_r.values.reshape(-1, len(subs))
    s = d.posterior.sigma_rep.values.reshape(-1, len(subs))
    ceiling = (a.grid_resolution / (2 * np.pi)) ** 2

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.6), constrained_layout=True)
    kmed = forest(axes[0], k, subs, K_COL, "κ_r (perceptual precision)", log=True,
                  ref=ceiling, ref_label=f"grid limit\n({ceiling:.0f})")
    smed = forest(axes[1], s, subs, S_COL, "σ_rep (value noise, CHF)")
    axes[0].set_ylabel("Subject")
    axes[0].set_title("Perceptual stage")
    axes[1].set_title("Valuation stage")

    # Does the value stage switch off for some subjects?
    ax = axes[2]
    ax.plot(kmed, smed, "o", ms=4, color="0.25", mec="white", mew=0.5)
    for x, y, sub in zip(kmed, smed, subs):
        if y < 0.5 or y > 2.6:
            ax.annotate(f"{sub:02d}", (x, y), fontsize=5.5, color="0.45",
                        xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("κ_r"); ax.set_ylabel("σ_rep (CHF)")
    ax.set_title("Where the noise is assigned")
    lo = (smed < 0.5).sum()
    ax.axhspan(-0.05, 0.5, color=S_COL, alpha=0.08, lw=0)
    ax.text(kmed.min(), 0.42, f" {lo}/{len(subs)} subjects: value stage\n effectively noiseless",
            fontsize=6.5, color=S_COL, va="top")

    print(f"kappa_r  median {np.median(kmed):.0f}  range {kmed.min():.0f}-{kmed.max():.0f} "
          f"| grid ceiling {ceiling:.0f} | above it: {(kmed > ceiling).sum()}/{len(subs)}")
    print(f"sigma_rep median {np.median(smed):.2f} range {smed.min():.2f}-{smed.max():.2f} "
          f"| below 0.5: {lo}/{len(subs)}")
    for name, tr, med in (("kappa_r vs perception-only", a.compare_kappa, kmed),
                          ("sigma_rep vs valuation-only", a.compare_sigma, smed)):
        if tr:
            o = az.from_netcdf(tr)
            v = "kappa_r" if "kappa" in name else "sigma_rep"
            other = np.median(o.posterior[v].values.reshape(-1, len(subs)), axis=0)
            print(f"  corr({name}) = {np.corrcoef(med, other)[0,1]:+.3f}")

    sns.despine(fig=fig, offset=4, trim=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
