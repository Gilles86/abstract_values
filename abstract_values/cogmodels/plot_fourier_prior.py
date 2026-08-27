"""What a circular Fourier orientation prior can look like.

The efficient-coding models need a prior over orientation, p(phi), and it
enters only through its CDF F(phi) -- the encoding transform that decides how
much representational space each orientation gets.  So the question is what
family of shapes to allow.

Current options in `bauer`:
    fixed       p(phi) ~ 2 - |sin phi|          (the paper's long-term prior)
    uniform     p(phi) ~ const
    1-parameter p(phi) ~ 1 - w|sin phi|         (--fit-prior-weight)

Proposed: a two-term circular Fourier prior in the DOUBLED angle phi = 2*theta,

    p(phi) ~ exp(a1 cos(phi) + a2 cos(2 phi))

which is positive and exactly periodic by construction (no knot placement, no
positivity constraint to enforce, unlike a spline), nests uniform at a = 0, and
separates the two things a spline with knots at 0/45/90/135/180 would be
buying:

    a2  (= cos 4 theta)   cardinal-vs-oblique -- the classic cardinal prior,
                          and the term that reproduces 2 - |sin phi|
    a1  (= cos 2 theta)   horizontal-vs-vertical asymmetry, which the fixed
                          prior cannot express at all: it is symmetric about
                          45 deg, so it forces the 0 deg and 90 deg cardinals
                          to be equally likely.

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

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.4), constrained_layout=True)
    cmap = sns.color_palette("mako", as_cmap=True)
    XMAX = 232          # room to direct-label at the right end; spine trims to 180

    def label_at(ax, x, y, text, color, va="center"):
        ax.text(x, y, text, color=color, fontsize=6.5, va=va, ha="left")

    # --- a: cardinal-vs-oblique strength -------------------------------
    ax = axes[0]
    a2s = [0.0, 0.15, 0.35, 0.7]
    for i, a2 in enumerate(a2s):
        c = cmap(np.linspace(0.72, 0.15, len(a2s))[i])
        y = fourier_prior(0.0, a2)
        ax.plot(THETA, y, color=c)
        # a2 = 0.35 is deliberately next to the least-squares match, so one
        # label carries the whole point of the panel.
        txt = f"a₂ = {a2:g}" + ("\n≈ 2 − |sin φ|" if a2 == 0.35 else "")
        label_at(ax, 184, y[-1], txt, c)
    y_lt = long_term_prior()
    ax.plot(THETA, y_lt, color="#C44E52", ls="--", lw=1.0)
    ax.set_title("Cardinal vs oblique (a₂)")
    ax.set_ylabel("Prior density")

    # --- b: horizontal-vs-vertical asymmetry ---------------------------
    ax = axes[1]
    a1s = [0.0, 0.15, 0.3, 0.5]
    for i, a1 in enumerate(a1s):
        c = cmap(np.linspace(0.72, 0.15, len(a1s))[i])
        y = fourier_prior(a1, a2_lt)
        ax.plot(THETA, y, color=c)
        label_at(ax, 184, y[-1], f"a₁ = {a1:g}", c)
    ax.set_title("Horizontal vs vertical (a₁)")
    ax.annotate("Only the Fourier prior\ncan break this symmetry",
                xy=(90, fourier_prior(0.5, a2_lt)[THETA.searchsorted(90)]),
                xytext=(16, 0.30), fontsize=6.5, color="0.35", ha="left",
                arrowprops=dict(arrowstyle="-|>", color="0.45", lw=0.9,
                                mutation_scale=7, shrinkB=5,
                                connectionstyle="angle3,angleA=0,angleB=70",
                                relpos=(1.0, 0.5)))

    # --- c: what the model actually uses -------------------------------
    ax = axes[2]
    shapes = [("1 − w|sin φ|,\nw = 0.9", weight_prior(0.9), "#E76F51"),
              ("a₂ = 0.35", fourier_prior(0.0, 0.35), cmap(0.45)),
              ("a₁ = 0.5,\na₂ = 0.35", fourier_prior(0.5, 0.35), cmap(0.15))]
    for label, prior, c in shapes:
        dev = encoding_cdf(prior) - THETA
        ax.plot(THETA, dev, color=c)
        label_at(ax, 184, dev[THETA.searchsorted(155)], label, c)
    ax.axhline(0, color="0.7", lw=0.7, ls="--", zorder=0)
    ax.text(6, 0.6, "Uniform", color="0.45", fontsize=6.5, va="bottom")
    ax.set_title("Coding space taken / given")
    ax.set_ylabel("F(θ) − θ (deg)")

    for i, ax in enumerate(axes):
        ax.set_xlabel("Orientation θ (deg)")
        ax.set_xticks(TICKS)
        ax.set_xlim(0, XMAX)
        for cardinal in (90,):
            ax.axvline(cardinal, color="0.88", lw=0.6, ls=":", zorder=0)
    # Same quantity in a and b, so one scale (skill: align y across panels
    # showing the same thing), sized to whichever panel needs more room.
    lo = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    hi = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(lo, hi); axes[1].set_ylim(lo, hi)
    axes[1].set_yticklabels([])
    sns.despine(fig=fig, offset=4, trim=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
