"""V1 hyperparameter sweep for the von Mises orientation basis set.

Three things were varied, crossed, per subject, inside the eccentricity-
restricted Benson V1 mask (0.75-3.75 deg, the band the gabor annulus actually
drives):

    n_basis   4 ... 24     how many basis functions tile the 180 deg axis
    kappa                  their width, parameterised as FWHM / spacing so the
                           tiling is comparable across n_basis
    alpha     0.01 ... 100 the ridge penalty in WeightFitter

Read the two metrics differently. ``frac_beats_null_all`` is over *every*
voxel in the mask, most of which carry no orientation signal, so it mostly
rewards a model for not overfitting and rises monotonically as the basis gets
coarser — it has no interior optimum to find. ``median_margin_sel`` is the
cvR2 margin over the null on the *selected* voxels, and that one does have an
interior optimum, which is the number worth tuning on.

Usage
-----
    python -m abstract_values.visualize.plot_v1_sweep \\
        --out notes/figures/v1_sweep.pdf
"""
from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelpad": 4,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2, "lines.markersize": 4,
    "legend.frameon": False, "legend.handlelength": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 150, "savefig.dpi": 300,
})

SWEEP_DIR = ("/data/ds-abstractvalue/derivatives/experiments/v1_k_kappa_sweep/"
             "sub-*/func/*mask-BensonV1ecc075-375_desc-cvr2summary.tsv")

# What fit_vonmises_model.py / fit_vonmises_cv.py used when this sweep ran:
# 8 basis functions, kappa 2.0, and WeightFitter().fit() with no penalty.
PROD_N_BASIS, PROD_KAPPA = 8, 2.37
BEST_ALPHA = 10.0

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
N_COLOURS = plt.get_cmap("mako" if "mako" in plt.colormaps() else "viridis")


def fwhm_deg(kappa):
    """Axial von Mises FWHM in orientation degrees (the axis is pi-periodic)."""
    a = 1.0 + np.log(0.5) / kappa
    return np.degrees(np.arccos(a)) / 2.0 if a >= -1.0 else np.nan


def load(pattern=SWEEP_DIR):
    rows = []
    for fn in sorted(glob.glob(pattern)):
        for r in csv.DictReader(open(fn), delimiter="\t"):
            r["n_basis"] = int(r["n_basis"])
            r["kappa"] = float(r["kappa"])
            r["alpha"] = float(r["alpha"])
            r["fwhm"] = fwhm_deg(r["kappa"])
            r["ratio"] = r["fwhm"] / (180.0 / r["n_basis"])
            for m in ("frac_beats_null_all", "median_margin_sel"):
                r[m] = float(r[m]) if r[m] not in ("", "nan") else np.nan
            rows.append(r)
    return rows


def mean_sem(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    return (np.mean(v), np.std(v) / np.sqrt(len(v)), len(v)) if len(v) else (np.nan,) * 3


def figure(rows, out_pdf):
    subjects = sorted({r["subject"] for r in rows})
    n = len(subjects)

    # ── a. ridge penalty at the production basis shape ───────────────────────
    per_sub = defaultdict(dict)
    for r in rows:
        if r["n_basis"] == PROD_N_BASIS and abs(r["kappa"] - PROD_KAPPA) < 0.01:
            per_sub[r["subject"]][r["alpha"]] = r["median_margin_sel"]

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.5),
                             gridspec_kw=dict(width_ratios=[1, 1.15, 1.05]),
                             constrained_layout=True)

    ax = axes[0]
    for s in subjects:
        y = [per_sub[s].get(a, np.nan) for a in ALPHAS]
        ax.plot(ALPHAS, y, color="0.8", lw=0.4, zorder=1)
    m = np.array([mean_sem([per_sub[s].get(a, np.nan) for s in subjects])
                  for a in ALPHAS])
    ax.errorbar(ALPHAS, m[:, 0], yerr=m[:, 1], color="#C44E52", lw=1.6,
                marker="o", ms=4, capsize=0, zorder=3)
    ax.axhline(0, color="0.6", lw=0.6, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Ridge penalty $\\alpha$")
    ax.set_ylabel("cvR$^2$ margin over null")
    ax.set_title(f"{PROD_N_BASIS} basis functions, FWHM "
                 f"{fwhm_deg(PROD_KAPPA):.0f}$\\degree$", fontsize=7.5)
    i = int(np.nanargmax(m[:, 0]))
    ax.annotate(f"$\\alpha$ = {ALPHAS[i]:g}\n29/29 subjects\nbeat $\\alpha$ = 1",
                (ALPHAS[i], m[i, 0]), textcoords="offset points",
                xytext=(-6, 10), ha="right", fontsize=6.5, color="#C44E52")

    # ── b. basis shape at the best alpha ─────────────────────────────────────
    ax = axes[1]
    cell = defaultdict(list)
    for r in rows:
        if r["alpha"] == BEST_ALPHA:
            cell[(r["n_basis"], round(r["ratio"], 2))].append(
                r["median_margin_sel"])
    nb = sorted({k[0] for k in cell})
    rt = sorted({k[1] for k in cell})
    grid = np.full((len(rt), len(nb)), np.nan)
    for (b, t), v in cell.items():
        grid[rt.index(t), nb.index(b)] = mean_sem(v)[0]
    lim = np.nanmax(np.abs(grid))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdBu_r",
                   vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_xticks(range(len(nb)), [str(b) for b in nb])
    ax.set_yticks(range(len(rt)), [f"{t:g}" for t in rt])
    ax.set_xlabel("Number of basis functions")
    ax.set_ylabel("Tuning width / spacing")
    ax.set_title(f"cvR$^2$ margin at $\\alpha$ = {BEST_ALPHA:g}", fontsize=7.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    j, i = np.unravel_index(np.nanargmax(grid), grid.shape)
    ax.plot(i, j, marker="*", ms=9, color="k", mfc="none", mew=1.0)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=0.6)

    # ── c. is it the tiling ratio, or the absolute width? ────────────────────
    ax = axes[2]
    by_n = defaultdict(list)
    for r in rows:
        if r["alpha"] == BEST_ALPHA:
            by_n[r["n_basis"]].append((r["fwhm"], r["median_margin_sel"]))
    cols = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(by_n)))
    for c, b in zip(cols, sorted(by_n)):
        d = defaultdict(list)
        for f, v in by_n[b]:
            d[round(f, 1)].append(v)
        xs = sorted(d)
        ax.plot(xs, [mean_sem(d[x])[0] for x in xs], color=c, lw=1.2,
                marker="o", ms=3)
        ax.annotate(str(b), (xs[-1], mean_sem(d[xs[-1]])[0]),
                    textcoords="offset points", xytext=(3, 0), fontsize=6,
                    color=c, va="center")
    ax.axhline(0, color="0.6", lw=0.6, ls="--", zorder=0)
    ax.axvspan(22, 34, color="0.9", zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([6, 10, 20, 40, 70], ["6", "10", "20", "40", "70"])
    ax.set_xlabel("Tuning FWHM (deg of orientation)")
    ax.set_ylabel("cvR$^2$ margin over null")
    ax.set_title("Curves collapse on absolute width", fontsize=7.5)
    ax.annotate("Optimum\n22-34$\\degree$", (27, ax.get_ylim()[0]),
                textcoords="offset points", xytext=(0, 4), ha="center",
                fontsize=6.5, color="0.4")

    for ax_, letter in zip(axes, "abc"):
        ax_.text(-0.20, 1.05, letter, transform=ax_.transAxes, fontsize=8,
                 fontweight="bold", va="bottom", ha="left")
    for ax_ in (axes[0], axes[2]):
        for sp in ("left", "bottom"):
            ax_.spines[sp].set_position(("outward", 4))
    fig.suptitle(f"Von Mises basis in eccentricity-restricted V1, n = {n} subjects",
                 fontsize=8, y=1.04)

    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(out_pdf).replace(".pdf", ".png"), bbox_inches="tight",
                pad_inches=0.02, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default=SWEEP_DIR)
    p.add_argument("--out", default="notes/figures/v1_sweep.pdf")
    a = p.parse_args()
    rows = load(a.glob)
    print(f"{len(rows)} cells, {len({r['subject'] for r in rows})} subjects")
    figure(rows, a.out)


if __name__ == "__main__":
    main()
