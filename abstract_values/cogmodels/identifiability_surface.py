"""Can the data separate perceptual from valuation noise?  A likelihood surface.

The fastest honest answer to an identifiability question is not a fit -- it is
the shape of the likelihood.  This simulates one subject's worth of trials from
known (kappa_r, sigma_rep) under a given architecture, then evaluates the
log-likelihood of that same architecture on a 2-D grid of the two parameters.

    a well-defined peak  ->  the design separates the two stages
    a ridge              ->  they trade off; a fit will wander along it
    flat in kappa_r      ->  the perceptual stage is unidentified, and no
                             amount of sampling effort will change that

Run for the uniform and long-term priors to see the thing that makes the
difference: under a uniform orientation prior the perceptual stage adds
variance but no bias, so it has almost nothing to be estimated from.

Writes notes/figures/identifiability_surface.pdf.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
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

KAPPA_TRUE, SIGMA_TRUE = 30.0, 1.5   # overridden by --kappa-true / --sigma-true


def one_subject_paradigm(tsv, subject=3):
    d = pd.read_csv(tsv, sep="\t")
    d = d[d.subject == subject].copy()
    p = d[["orientation", "response", "mapping"]].copy()
    p = p.set_index([pd.Index([subject] * len(p), name="subject"),
                     pd.RangeIndex(len(p), name="trial")])
    return p


def trial_dist_fn(model, paradigm):
    """Compile trial_dist as a function of (kappa_r, sigma_rep).

    Compiled once and re-evaluated by setting the pm.Data values -- rebuilding
    the graph per grid point is what makes the naive version of this unusable.
    """
    model._setup_grids(paradigm)
    with pm.Model() as m:
        paradigm_ = model._get_paradigm(paradigm=paradigm)
        model.set_paradigm(paradigm_)
        pm.Data("kappa_r", np.array([KAPPA_TRUE]))
        pm.Data("sigma_rep", np.array([SIGMA_TRUE]))
        params = model.get_parameter_values()
        td = model._compute_trial_distributions(model.get_model_inputs(params))
    f = pytensor.function([], td, on_unused_input="ignore")
    return m, f


def loglik(dist, responses, grid, bin_width=0.5, lapse=0.01):
    """Binned log-likelihood, matching bauer's bin_probability in numpy."""
    dv = grid[1] - grid[0]
    cdf = np.concatenate([np.zeros((dist.shape[0], 1)),
                          np.cumsum((dist[:, :-1] + dist[:, 1:]) / 2 * dv, axis=1)], axis=1)
    lo = np.clip(responses - bin_width / 2, grid[0], grid[-1])
    hi = np.clip(responses + bin_width / 2, grid[0], grid[-1])
    def interp(frac):
        i = np.clip(np.floor(frac).astype(int), 0, len(grid) - 2)
        w = frac - i
        return cdf[np.arange(len(frac)), i] * (1 - w) + cdf[np.arange(len(frac)), i + 1] * w
    p = interp((hi - grid[0]) / dv) - interp((lo - grid[0]) / dv)
    p = (1 - lapse) * p + lapse * bin_width / (grid[-1] - grid[0])
    return np.log(np.clip(p, 1e-12, None)).sum()


def surface(model, paradigm, kappas, sigmas, seed=1):
    m, f = trial_dist_fn(model, paradigm)
    grid = model._get_response_grid()

    # simulate at the truth, through the same graph
    with m:
        pm.set_data({"kappa_r": np.array([KAPPA_TRUE]), "sigma_rep": np.array([SIGMA_TRUE])})
    d0 = f()
    rng = np.random.default_rng(seed)
    dv = grid[1] - grid[0]
    pdf = d0 / (d0.sum(axis=1, keepdims=True) * dv)
    cdf = np.cumsum(pdf * dv, axis=1)
    u = rng.uniform(size=len(pdf))
    y = grid[np.argmax(cdf >= u[:, None], axis=1)]
    y = np.round(y / 0.5) * 0.5                     # the real response lattice

    ll = np.full((len(sigmas), len(kappas)), np.nan)
    for i, s in enumerate(sigmas):
        for j, k in enumerate(kappas):
            with m:
                pm.set_data({"kappa_r": np.array([k]), "sigma_rep": np.array([s])})
            ll[i, j] = loglik(f(), y, grid)
    return ll


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paradigm-tsv", default="notes/data/efficient_coding_paradigm.tsv")
    ap.add_argument("--grid-resolution", type=int, default=51)
    ap.add_argument("--n-kappa", type=int, default=24)
    ap.add_argument("--n-sigma", type=int, default=18)
    ap.add_argument("--kappa-true", type=float, default=30.0)
    ap.add_argument("--sigma-true", type=float, default=1.5)
    ap.add_argument("--out", default="notes/figures/identifiability_surface.pdf")
    a = ap.parse_args()

    global KAPPA_TRUE, SIGMA_TRUE
    KAPPA_TRUE, SIGMA_TRUE = a.kappa_true, a.sigma_true
    from abstract_values.cogmodels.fit_efficient_coding import make_model
    paradigm = one_subject_paradigm(a.paradigm_tsv)
    print(f"{len(paradigm)} trials, one subject")

    kappas = np.exp(np.linspace(np.log(3), np.log(2000), a.n_kappa))
    sigmas = np.linspace(0.3, 4.0, a.n_sigma)

    configs = [("Sequential, uniform prior", "sequential", dict(perceptual_prior="uniform")),
               ("Sequential, long-term prior", "sequential", dict(perceptual_prior="long_term")),
               ("Categorical + no seam, long-term", "categorical",
                dict(perceptual_prior="long_term", no_seam_crossing=True))]

    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.5), constrained_layout=True)
    profiles = []
    for ax, (title, name, kw) in zip(axes, configs):
        model = make_model(paradigm, name, a.grid_resolution, lapse_rate=0.01, **kw)
        ll = surface(model, paradigm, kappas, sigmas)
        rel = ll - np.nanmax(ll)
        im = ax.contourf(kappas, sigmas, rel, levels=np.linspace(-30, 0, 16),
                         cmap="mako", extend="min")
        ax.contour(kappas, sigmas, rel, levels=[-6.0], colors="w", linewidths=0.9)
        ax.plot(KAPPA_TRUE, SIGMA_TRUE, "o", ms=5, mfc="none", mec="#E76F51", mew=1.4)
        ax.set_xscale("log")
        ax.set_xlabel("κ_r (perceptual precision)")
        ax.set_title(title)
        j, i = np.unravel_index(np.nanargmax(ll), ll.shape)[::-1]
        prof = np.nanmax(ll, axis=0)            # profile over sigma_rep
        prof = prof - prof.max()
        profiles.append((title, prof))
        drop = prof[-1]
        print(f"{title:36s} peak kappa={kappas[i]:7.1f} sigma={sigmas[j]:.2f} | "
              f"log-lik drop from peak to kappa={kappas[-1]:.0f}: {drop:6.1f}")
    axes[0].set_ylabel("σ_rep (value noise)")
    fig.colorbar(im, ax=axes[2], label="Δ log-likelihood")

    # Profile over sigma_rep: the flat-direction test, made quantitative.
    ax = axes[3]
    cols = ["#E76F51", "#3B5BA5", "#2A9D8F"]
    for (title, prof), c in zip(profiles, cols):
        ax.plot(kappas, prof, color=c)
        ax.text(kappas[-1], prof[-1], "  " + title.split(",")[0], color=c,
                fontsize=6.5, va="center", ha="left")
    ax.axhline(-3.0, color="0.7", lw=0.7, ls="--", zorder=0)
    ax.text(4, -2.6, "2 log-units", color="0.5", fontsize=6.5, va="bottom")
    ax.axvline(KAPPA_TRUE, color="#E76F51", lw=0.7, ls=":", zorder=0)
    ax.set_xscale("log"); ax.set_ylim(-25, 2); ax.set_xlim(3, 6000)
    ax.set_xlabel("κ_r (perceptual precision)")
    ax.set_ylabel("Profile Δ log-likelihood")
    ax.set_title("Flat direction?")
    fig.suptitle("Can the design separate the two noise sources?  "
                 "White contour = 6 log-units below the peak; circle = truth",
                 fontsize=8, y=1.04, color="0.15")
    sns.despine(fig=fig, offset=3)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
