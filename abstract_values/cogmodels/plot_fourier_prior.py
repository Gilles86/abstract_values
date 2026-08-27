"""What a circular Fourier orientation prior can look like, and what it predicts.

The efficient-coding models need a prior over orientation: which orientations
the observer expects.  It matters because efficient coding spends
representational resolution in proportion to the prior, so the prior IS the
prediction about where perception is precise and where it is noisy.

Everything here is written in orientation theta (0-180 deg).  bauer works
internally in the doubled angle phi = 2*theta, which is why the paper's prior
is written 2 - |sin phi|; in theta that is a shape with peaks at the cardinals
(0 and 90 deg) and troughs at the obliques (45 and 135 deg).

Current options in `bauer`:
    fixed       p ~ 2 - |sin 2theta|      the paper's long-term prior
    uniform     p ~ const
    1-parameter p ~ 1 - w|sin 2theta|     (--fit-prior-weight)

Proposed instead, a two-term circular Fourier prior:

    p(theta) ~ exp(a1 cos 2theta + a2 cos 4theta)

positive and exactly periodic by construction (no knot placement, no positivity
constraint to enforce, unlike a spline), nesting uniform at a = 0, and splitting
the shape into the two things that actually differ:

    a2   cardinals vs obliques -- the classic cardinal prior, and the term that
         reproduces the paper's 2 - |sin| shape at a2 ~ 0.31
    a1   horizontal vs vertical -- an asymmetry the fixed prior cannot express
         at all, because 2 - |sin| is symmetric about 45 deg and so forces the
         0 deg and 90 deg cardinals to be equally expected

Writes notes/figures/fourier_prior_family.pdf.
"""
from __future__ import annotations

import argparse
from pathlib import Path

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
    "axes.labelpad": 4, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})

THETA = np.linspace(0, 180, 721)          # orientation, degrees
PHI = np.deg2rad(THETA) * 2               # doubled angle, radians
TICKS = [0, 45, 90, 135, 180]


def fourier_prior(a1, a2, phi=PHI):
    p = np.exp(a1 * np.cos(phi) + a2 * np.cos(2 * phi))
    return p / np.trapezoid(p, phi)


def long_term_prior(phi=PHI):
    p = 2.0 - np.abs(np.sin(phi))
    return p / np.trapezoid(p, phi)


def weight_prior(w, phi=PHI):
    p = np.maximum(1.0 - w * np.abs(np.sin(phi)), 1e-6)
    return p / np.trapezoid(p, phi)


def encoding_cdf(p, phi=PHI):
    """F(phi): the efficient-coding transform the prior induces."""
    c = np.concatenate([[0.0], np.cumsum((p[:-1] + p[1:]) / 2 * np.diff(phi))])
    return c / c[-1] * 180.0                       # in degrees of coding space


def fit_to_long_term():
    """Least-squares (a1, a2) matching the paper's 2 - |sin phi| prior."""
    y = np.log(long_term_prior())
    X = np.column_stack([np.cos(PHI), np.cos(2 * PHI), np.ones_like(PHI)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef[0], coef[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="notes/figures/fourier_prior_family.pdf")
    a = ap.parse_args()

    a1_lt, a2_lt = fit_to_long_term()
    print(f"Least-squares match to 2 - |sin phi|:  a1 = {a1_lt:.3f}, a2 = {a2_lt:.3f}")
    approx = fourier_prior(a1_lt, a2_lt)
    print(f"  max abs deviation from it: {np.abs(approx - long_term_prior()).max():.2e} "
          f"(density scale {long_term_prior().mean():.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), constrained_layout=True)

    # Three named shapes, not a parameter sweep: the panel has to be readable
    # before the reader has decoded a1/a2.
    shapes = [
        ("Uniform",             fourier_prior(0.0, 0.0),   "0.55",
         "a₁ = 0, a₂ = 0"),
        ("Cardinals favoured",  fourier_prior(0.0, a2_lt), "#3B5BA5",
         f"a₁ = 0, a₂ = {a2_lt:.2f}"),
        ("Horizontal favoured", fourier_prior(0.5, a2_lt), "#C44E52",
         f"a₁ = 0.5, a₂ = {a2_lt:.2f}"),
    ]

    # --- a: the prior itself -------------------------------------------
    ax = axes[0]
    for name, prior, c, sub in shapes:
        rel = prior / prior.mean()
        ax.plot(THETA, rel, color=c)
        ax.text(184, rel[-1], f"{name}\n{sub}", color=c, fontsize=6.5,
                va="center", ha="left")
    lt = long_term_prior() / long_term_prior().mean()
    ax.plot(THETA, lt, color="#3B5BA5", ls=(0, (1.5, 1.6)), lw=1.6)
    ax.set_ylim(0.3, 2.45)
    ax.set_ylabel("Prior density (uniform = 1)")
    ax.set_title("Which orientations are expected")
    ax.text(45, 0.36, "Obliques", color="0.45", fontsize=6.5, ha="center")
    ax.text(135, 0.36, "Obliques", color="0.45", fontsize=6.5, ha="center")
    ax.text(90, 2.33, "Cardinal", color="0.45", fontsize=6.5, ha="center")
    ax.annotate("Dashed: the paper's 2 − |sin|\nprior, on top of a₂ = 0.31",
                xy=(112, lt[THETA.searchsorted(112)]),
                xytext=(20, 1.95), fontsize=6.5, color="#3B5BA5", ha="left",
                arrowprops=dict(arrowstyle="-|>", color="#3B5BA5", lw=0.9,
                                mutation_scale=7, shrinkB=4,
                                connectionstyle="angle3,angleA=0,angleB=65",
                                relpos=(1.0, 0.5)))

    # --- b: the behavioural consequence --------------------------------
    # Efficient coding spends coding space in proportion to the prior, so the
    # perceptual SD at an orientation goes as 1 / p(theta).
    ax = axes[1]
    for name, prior, c, sub in shapes:
        ax.plot(THETA, prior.mean() / prior, color=c)
    ax.set_ylim(0.3, 2.45)
    ax.set_ylabel("Perceptual noise (uniform = 1)")
    ax.set_title("What that predicts about precision")
    ax.text(90, 2.33, "Same three priors", color="0.45", fontsize=6.5, ha="center")
    ax.annotate("Precision is spent where\nthe prior is high",
                xy=(90, (fourier_prior(0.0, a2_lt).mean()
                         / fourier_prior(0.0, a2_lt))[THETA.searchsorted(90)]),
                xytext=(104, 1.55), fontsize=6.5, color="0.35", ha="left",
                arrowprops=dict(arrowstyle="-|>", color="0.45", lw=0.9,
                                mutation_scale=7, shrinkB=5,
                                connectionstyle="angle3,angleA=0,angleB=70",
                                relpos=(0.0, 0.5)))

    for ax in axes:
        ax.set_xlabel("Orientation θ (deg)")
        ax.set_xticks(TICKS)
        for cardinal in (0, 90, 180):
            ax.axvline(cardinal, color="0.9", lw=0.6, ls=":", zorder=0)
    axes[0].set_xlim(0, 290)
    axes[1].set_xlim(0, 180)
    sns.despine(fig=fig, offset=4, trim=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
