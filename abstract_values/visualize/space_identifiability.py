"""Can a value model be told apart from an orientation model at all?

Value is a deterministic, invertible function of gabor orientation within a
session, so a basis set in value space and a basis set in orientation space
can express nearly the same functions of the stimulus. This figure measures
that overlap on the experiment's own stimulus sequence, rather than arguing
about it:

  a  the two bases drawn against the same axis — 8 axial von Mises in
     orientation (the ``vonmises`` model) and 8 log-Gaussians in value (the
     ``aprf-weighted`` model), the latter plotted through the mapping. They
     are the same family of bumps under a warp of the x axis.
  b  canonical correlations between the two design matrices. Within one
     mapping they are ~1 for the first few components: the two models are
     interchangeable. Pooling both mappings pulls them apart — that flip is
     the only thing in the design that separates the two spaces.
  c  and a value basis refit per mapping (what the session-shift models do)
     spends exactly that: the orientation basis is back inside its span.

So a session-shifted value model that beats an orientation model is not
evidence about representational space, which is why ``model_winner_maps``
keeps the shift variants out of its default comparison.

Usage
-----
    python -m abstract_values.visualize.space_identifiability
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import svd

from abstract_values.utils.data import BIDS_FOLDER

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300,
})

ORI_COLOUR, VAL_COLOUR = "#25457F", "#B33A22"
REGIME_COLOUR = {"Within one mapping": "#3B5BA5",
                 "Both mappings pooled": "#5D8C3F",
                 "Value basis refit per mapping": "#C44E52"}


def ori_basis(theta_deg, n=8, kappa=2.0):
    """fit_vonmises_model: mu = linspace(0, pi, n, endpoint=False), axial."""
    th = np.deg2rad(np.asarray(theta_deg, float))
    mus = np.linspace(0, np.pi, n, endpoint=False)
    X = np.exp(kappa * np.cos(2 * (th[:, None] - mus[None, :])))
    return X / X.max(axis=0, keepdims=True)


def val_basis(v, vmin, vmax, n=8):
    """fit_aprf_weighted: modes linspace(vmin, vmax, n), fwhm = 2 x spacing."""
    v = np.asarray(v, float)
    modes = np.linspace(vmin, vmax, n)
    fwhm = 2.0 * (modes[1] - modes[0])
    lm = np.log(np.clip(modes, 1e-3, None))
    sd = np.log1p(fwhm / np.clip(modes, 1e-3, None)) / (2 * np.sqrt(2 * np.log(2)))
    X = np.exp(-0.5 * ((np.log(np.clip(v, 1e-3, None))[:, None] - lm[None, :])
                       / sd[None, :]) ** 2)
    return X / X.max(axis=0, keepdims=True)


def canon_corr(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    Qa = np.linalg.qr(A)[0]
    Qb = np.linalg.qr(B)[0]
    return np.clip(svd(Qa.T @ Qb, compute_uv=False), 0, 1)


def span_r2(target, X):
    X = np.column_stack([np.ones(len(X)), X - X.mean(0)])
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    resid = target - X @ beta
    ss = ((target - target.mean(0)) ** 2).sum(0)
    return 1 - (resid ** 2).sum(0) / np.maximum(ss, 1e-12)


def load_stimuli(bids_folder, subject="03"):
    files = sorted(glob.glob(f"{bids_folder}/sourcedata/behavior/sub-{subject}/"
                             f"ses-*/*events.tsv"))
    frames = []
    for fn in files:
        d = pd.read_csv(fn, sep="\t")
        if not {"orientation", "value"} <= set(d.columns):
            continue
        d = d.dropna(subset=["orientation", "value"])
        frames.append(d[d.value > 0].assign(
            mapping="cdf" if ".cdf_" in fn else "inverse_cdf"))
    return pd.concat(frames)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--subject", default="03")
    p.add_argument("--out", default="notes/figures/space_identifiability.pdf")
    args = p.parse_args()

    df = load_stimuli(args.bids_folder, args.subject)
    vmin, vmax = df.value.min(), df.value.max()
    one = df[df.mapping == "cdf"]

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.25, 2.35))

    # (a) both bases against orientation, under one mapping
    curve = one.groupby("orientation").value.first().sort_index()
    th = np.linspace(curve.index.min(), curve.index.max(), 400)
    v_of_th = np.interp(th, curve.index.to_numpy(), curve.to_numpy())
    for col in ori_basis(th).T:
        a.plot(th, col, color=ORI_COLOUR, lw=0.9, alpha=.85)
    for col in val_basis(v_of_th, vmin, vmax).T:
        a.plot(th, col, color=VAL_COLOUR, lw=0.9, alpha=.85, ls=(0, (3, 1.6)))
    a.text(0.03, 0.97, "Orientation basis", transform=a.transAxes, fontsize=6.5,
           color=ORI_COLOUR, fontweight="bold", va="top")
    a.text(0.03, 0.86, "Value basis, drawn\nthrough the mapping",
           transform=a.transAxes, fontsize=6.5, color=VAL_COLOUR,
           fontweight="bold", va="top")
    a.set_xlabel("Gabor orientation (deg)")
    a.set_ylabel("Basis response")
    a.set_xticks([0, 45, 90, 135, 180])
    a.set_ylim(0, 1.45)

    # (b) canonical correlations per regime
    Xo_all = ori_basis(df.orientation)
    Xv_all = val_basis(df.value, vmin, vmax)
    sess = (df.mapping == "cdf").to_numpy(float)[:, None]
    regimes = {}
    for cond in ("cdf", "inverse_cdf"):
        d = df[df.mapping == cond]
        regimes.setdefault("Within one mapping", []).append(
            canon_corr(ori_basis(d.orientation), val_basis(d.value, vmin, vmax)))
    regimes["Within one mapping"] = np.mean(regimes["Within one mapping"], axis=0)
    regimes["Both mappings pooled"] = canon_corr(Xo_all, Xv_all)
    Xv_flex = np.hstack([Xv_all * sess, Xv_all * (1 - sess)])
    regimes["Value basis refit per mapping"] = canon_corr(Xo_all, Xv_flex)[:8]

    for i, (name, cc) in enumerate(regimes.items()):
        b.plot(np.arange(1, len(cc) + 1), cc, "o-", ms=3, color=REGIME_COLOUR[name])
        b.text(0.96, 0.97 - 0.1 * i, name, transform=b.transAxes, ha="right",
               va="top", fontsize=6.2, color=REGIME_COLOUR[name], fontweight="bold")
    b.set_xlabel("Canonical component")
    b.set_ylabel("Canonical correlation")
    b.set_ylim(0, 1.05)
    b.set_xticks([1, 2, 4, 6, 8])

    # (c) how much of one model's span the other covers
    bars = {
        "Within one mapping": np.median(span_r2(
            val_basis(one.value, vmin, vmax) - val_basis(one.value, vmin, vmax).mean(0),
            ori_basis(one.orientation))),
        "Both mappings pooled": np.median(span_r2(Xv_all - Xv_all.mean(0), Xo_all)),
        "Value basis refit per mapping": np.median(span_r2(
            Xo_all - Xo_all.mean(0), Xv_flex)),
    }
    for i, (name, val) in enumerate(bars.items()):
        c.bar(i, val, width=.62, color=REGIME_COLOUR[name], lw=0)
        c.text(i, val + 0.02, f"{val:.2f}", ha="center", fontsize=6.5,
               color=REGIME_COLOUR[name], fontweight="bold")
    c.set_xticks(range(len(bars)))
    c.set_xticklabels(["Within\none mapping", "Both\npooled", "Refit per\nmapping"],
                      fontsize=6.5)
    c.set_ylabel("R² of one basis in the other's span")
    c.set_ylim(0, 1.1)
    c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    for ax, letter in zip((a, b, c), "abc"):
        ax.text(-0.2, 1.06, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="left")
        for side in ("left", "bottom"):
            ax.spines[side].set_position(("outward", 4))
    fig.suptitle("Orientation and value bases are nearly the same model",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=130)
    print(f"Wrote {out}")
    for name, cc in regimes.items():
        print(f"  {name:32s} canonical r: " + " ".join(f"{x:.3f}" for x in cc))


if __name__ == "__main__":
    main()
