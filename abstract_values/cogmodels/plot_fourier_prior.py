"""What the Fourier-prior fit actually says about the orientation prior.

The model fits p(phi) ~ exp(sum_k a_k cos(k phi) + b_k sin(k phi)) with phi the
DOUBLED angle, so each harmonic has a specific, interpretable meaning:

    a1  cos(2 theta)   horizontal vs vertical      (0 deg vs 90 deg)
    b1  sin(2 theta)   oblique vs oblique          (45 deg vs 135 deg)
    a2  cos(4 theta)   cardinal vs oblique         (the paper's term, a2 ~ 0.31)
    b2  sin(4 theta)   within-quadrant skew        (22.5 deg vs 67.5 deg)

Getting these labels right is the whole point of the figure: b1 and b2 are both
"asymmetries", but they are asymmetries of different things, and only b1 answers
"are the two obliques different?".

Usage:
    python -m abstract_values.cogmodels.plot_fourier_prior \
        --trace derivatives/cogmodels/efficient_coding_categorical_fourier2_noseam_trace.nc \
        --out notes/figures/fourier_prior_k2.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import seaborn as sns

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "mathtext.fontset": "stixsans",
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelpad": 4, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2, "lines.markersize": 4,
    "legend.frameon": False, "legend.handlelength": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 150, "savefig.dpi": 300,
})

THETA = np.linspace(0, 180, 721)
PHI = np.deg2rad(THETA) * 2

MEANING = {
    "a1": "Horizontal vs vertical",
    "b1": "Oblique vs oblique",
    "a2": "Cardinal vs oblique",
    "b2": "Within-quadrant skew",
}
C_FIT = "#3B5BA5"
C_REF = "#C44E52"


def harmonics(post):
    """Harmonic names present in this trace, in reading order."""
    ks = sorted({int(v[-1]) for v in post.data_vars
                 if len(v) == 8 and v.startswith("prior_") and v[6] in "ab"})
    return [f"{c}{k}" for k in ks for c in "ab"]


def prior_curves(coef_draws):
    """(n_draw, n_theta) prior densities, each normalised to mean 1."""
    logp = np.zeros((len(next(iter(coef_draws.values()))), PHI.size))
    for name, vals in coef_draws.items():
        k = int(name[-1])
        f = np.cos if name[0] == "a" else np.sin
        logp += vals[:, None] * f(k * PHI)[None, :]
    p = np.exp(logp - logp.max(axis=1, keepdims=True))
    return p / p.mean(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", default="notes/figures/fourier_prior.pdf")
    ap.add_argument("--rhat-flag", type=float, default=1.05,
                    help="Subjects whose coefficients exceed this r_hat are drawn "
                         "in grey: their per-subject prior is not estimated well "
                         "enough to read, and hiding that would be dishonest.")
    a = ap.parse_args()

    idata = az.from_netcdf(a.trace)
    post = idata.posterior
    names = harmonics(post)
    subs = [int(x) for x in post[f"prior_{names[0]}"].coords["subject"].values]

    # --- group level -------------------------------------------------------
    grp = {n: post[f"prior_{n}_mu"].values.ravel() for n in names}
    gc = prior_curves(grp)
    g_med = np.median(gc, axis=0)
    g_lo, g_hi = np.percentile(gc, [2.5, 97.5], axis=0)

    # --- per subject -------------------------------------------------------
    summ = az.summary(idata, var_names=[f"prior_{n}" for n in names])
    bad = set()
    for lab, r in summ["r_hat"].items():
        if r > a.rhat_flag:
            bad.add(int(lab.split("[")[1].rstrip("]")))
    sub_curves = {}
    for i, s in enumerate(subs):
        d = {n: post[f"prior_{n}"].values[:, :, i].ravel() for n in names}
        sub_curves[s] = np.median(prior_curves(d), axis=0)

    # The paper's fixed prior, p ~ 2 - |sin phi|, i.e. free-w at w = 0.5.
    ref = 2 - np.abs(np.sin(PHI))
    ref = ref / ref.mean()

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.35), constrained_layout=True)

    # (a) group prior
    ax = axes[0]
    ax.axhline(1.0, color="0.7", lw=0.6, ls=":", zorder=0)
    ax.plot(THETA, ref, color=C_REF, lw=1.1, ls=(0, (2.5, 1.8)), zorder=2)
    ax.fill_between(THETA, g_lo, g_hi, color=C_FIT, alpha=0.25, lw=0, zorder=1)
    ax.plot(THETA, g_med, color=C_FIT, lw=1.5, zorder=3)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlim(0, 180)
    ax.set_xlabel("Orientation (deg)")
    ax.set_ylabel("Prior density (uniform = 1)")
    ax.text(0.03, 0.97, "Fitted", transform=ax.transAxes, color=C_FIT,
            fontsize=7.5, va="top", ha="left")
    ax.text(0.03, 0.86, "Paper (a₂ = 0.31)", transform=ax.transAxes, color=C_REF,
            fontsize=7.5, va="top", ha="left")

    # (b) per-subject priors
    ax = axes[1]
    ax.axhline(1.0, color="0.7", lw=0.6, ls=":", zorder=0)
    for s, c in sub_curves.items():
        if s in bad:
            ax.plot(THETA, c, color="0.82", lw=0.7, zorder=1)
        else:
            ax.plot(THETA, c, color=C_FIT, lw=0.7, alpha=0.5, zorder=2)
    ax.plot(THETA, g_med, color=C_FIT, lw=1.8, zorder=4)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlim(0, 180)
    ax.set_xlabel("Orientation (deg)")
    ax.set_ylabel("Prior density (uniform = 1)")
    if bad:
        ax.text(0.03, 0.97, f"Grey: r̂ > {a.rhat_flag:g} (n = {len(bad)})",
                transform=ax.transAxes, color="0.45", fontsize=7, va="top")

    # (c) coefficient forest
    ax = axes[2]
    ax.axvline(0, color="0.7", lw=0.6, ls=":", zorder=0)
    ys = np.arange(len(names))[::-1]
    for y, n in zip(ys, names):
        v = grp[n]
        lo, hi = np.percentile(v, [2.5, 97.5])
        signif = lo > 0 or hi < 0
        col = C_FIT if signif else "0.55"
        ax.plot([lo, hi], [y, y], color=col, lw=1.4, solid_capstyle="butt", zorder=2)
        ax.plot([np.median(v)], [y], "o", color=col, ms=5, mec="white", mew=0.8,
                zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{n[0]}{n[1]}  {MEANING[n]}" for n in names])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("Coefficient (group mean, 95% HDI)")
    ax.text(0.98, 0.04, "Blue: HDI excludes 0", transform=ax.transAxes,
            color=C_FIT, fontsize=7, va="bottom", ha="right")

    for ax, letter in zip(axes, "abc"):
        ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="right")
    sns.despine(fig=fig, offset=4, trim=False)
    # Panel c's y-axis is a list of labels, not a scale; its spine is noise.
    axes[2].spines["left"].set_visible(False)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {out}")

    print("\nGroup-level harmonics (mean [95% HDI]):")
    for n in names:
        v = grp[n]
        lo, hi = np.percentile(v, [2.5, 97.5])
        mark = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {mark} {n}  {v.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]   {MEANING[n]}")
    pk = THETA[np.argmax(g_med)]
    tr = THETA[np.argmin(g_med)]
    print(f"\n  Prior peak at {pk:.1f} deg, trough at {tr:.1f} deg, "
          f"peak/trough ratio {g_med.max()/g_med.min():.2f}")
    print(f"  Paper's prior for comparison: ratio {ref.max()/ref.min():.2f}")
    if bad:
        print(f"\n  Subjects with r_hat > {a.rhat_flag:g} on a coefficient: "
              f"{sorted(bad)}")


if __name__ == "__main__":
    main()
