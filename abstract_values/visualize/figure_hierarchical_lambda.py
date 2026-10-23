"""Hierarchical lambda from per-session preferred values: group, participants, validity.

a: posterior of the population mean mu per ROI group (0 = tuned to fixed
value, 1 = tuned to the value's quantile); b: posterior of the population SD
tau; c: per-participant posteriors in pooled IPS (partially pooled, 95% CrI),
with the no-pooling estimate; d-f: the per-participant IPS lambda against the
same model fitted to each hemisphere separately, against the cvR^2-profile
lambda-hat (``fit_lambda_profile``, voxel-weighted mean over IPS0/1-2/3-5) and
against behavioural lambda (the mix axis that best predicts the bids).
Correlations are Spearman rho integrated over the hierarchical posterior
(one draw of every participant's lambda per sample; 95% interval of rho over
draws). Unsmoothed, log-Gaussian session-shift fits.

Input: ``encoding_models/hierarchical_lambda.py`` outputs in notes/data.

    python -m abstract_values.visualize.figure_hierarchical_lambda
    -> notes/figures/fig_hierarchical_lambda.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from abstract_values.visualize import qc_value_shift as q   # noqa: F401 house rcParams
from abstract_values.visualize.figure_tuning_connectivity import ORI_COL, ROI_COL, ROI_NAME, VAL_COL

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "notes/data"
GROUPS = ["IPS", "IPS0", "IPS12", "IPS345", "V3ab"]
COL = {**ROI_COL, "IPS": "k"}
NAME = {**ROI_NAME, "IPS": "IPS (pooled)"}
IPS_C = "#2E5E73"
N_DRAWS = 4000


def sid(s):
    return int(str(s).replace("pil", ""))


def refs(ax, labels=False):
    for l, c, t in ((0, VAL_COL, "Fixed value"), (1, ORI_COL, "Quantile")):
        ax.axvline(l, color=c, lw=.8, alpha=.7, zorder=0)
        if labels:
            ax.text(l, 1.0, t, color=c, fontsize=6, ha="center", va="bottom",
                    transform=ax.get_xaxis_transform())


def draw_rho(post, lam, x, rng):
    """Spearman rho between lambda draws (rows of post over lam) and x."""
    cdf = np.cumsum(post, 1)
    cdf /= cdf[:, -1:]
    u = rng.random((N_DRAWS, len(post), 1))
    draws = lam[(u > cdf[None]).sum(-1).clip(max=len(lam) - 1)]
    r = np.array([stats.spearmanr(d, x)[0] for d in draws])
    return r.mean(), *np.percentile(r, [2.5, 97.5])


def fmt_rho(m, lo, hi):
    return f"ρ = {m:.2f} [{lo:.2f}, {hi:.2f}]".replace("-", "−")


def scatter(ax, x, h, xlab, rho, col=IPS_C):
    ax.vlines(x, h.lo, h.hi, color=col, lw=.5, alpha=.35)
    ax.plot(x, h["mean"], "o", ms=3, color=col, mew=0)
    ax.set_xlabel(xlab)
    m, lo, hi = rho
    ax.text(.03, .97, fmt_rho(m, lo, hi), transform=ax.transAxes,
            fontsize=6.5, va="top", color="0.2")


def main(out=REPO / "notes/figures/fig_hierarchical_lambda.pdf"):
    rng = np.random.default_rng(0)
    grp = pd.read_csv(DATA / "hierlambda_lognormal_group.tsv", sep="\t").set_index("roi")
    sub = pd.read_csv(DATA / "hierlambda_lognormal_subjects.tsv", sep="\t", dtype={"subject": str})
    pop = np.load(DATA / "hierlambda_lognormal_population.npz")
    MU, TAU, LAM = pop["MU"], pop["TAU"], pop["LAM"]

    fig = plt.figure(figsize=(7.25, 5.0))
    gs = fig.add_gridspec(2, 3, left=.07, right=.98, top=.93, bottom=.09, hspace=.55,
                          wspace=.38)
    # a: mu
    ax = fig.add_subplot(gs[0, 0])
    refs(ax, labels=True)
    for g in GROUPS:
        p = pop[f"{g}_mu"] / np.diff(MU)[0]
        ax.plot(MU, p, color=COL[g], lw=1.6 if g == "IPS" else 1.0)
    for k, g in enumerate(GROUPS):                                # key, top left
        yy = .93 - .085 * k
        ax.plot([.03, .11], [yy, yy], color=COL[g], lw=1.6 if g == "IPS" else 1.0,
                transform=ax.transAxes)
        ax.text(.14, yy, NAME[g], color=COL[g], fontsize=6, va="center",
                transform=ax.transAxes)
    ax.set_xlim(-1, 2)
    ax.set_ylim(0, None)
    ax.set_xlabel("Population mean μ")
    ax.set_ylabel("Posterior density")
    ax.set_title("a  Group λ", loc="left", fontweight="bold")
    # b: tau
    ax = fig.add_subplot(gs[0, 1])
    for g in GROUPS:
        ax.plot(TAU, pop[f"{g}_tau"] / np.diff(TAU)[0], color=COL[g],
                lw=1.6 if g == "IPS" else 1.0)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, None)
    ax.set_xlabel("Between-participant SD τ")
    ax.set_ylabel("Posterior density")
    ax.set_title("b  Spread across participants", loc="left", fontweight="bold")
    # c: participants, pooled IPS
    ax = fig.add_subplot(gs[0, 2])
    h = sub[sub.roi == "IPS"].sort_values("mean").reset_index(drop=True)
    y = np.arange(len(h))
    refs(ax)
    g = grp.loc["IPS"]
    ax.fill_betweenx([-1, len(h) - .5], g.mu_lo, g.mu_hi, color="0.9", lw=0, zorder=-1)
    ax.hlines(y, h.lo, h.hi, color=IPS_C, lw=.7, alpha=.6)
    ax.plot(h["mean"], y, "o", ms=2.8, color=IPS_C, mew=0, label="Partially pooled")
    ax.plot(h.nopool_mean, y, "o", ms=2.8, mfc="none", mec="0.45", mew=.5,
            label="No pooling")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-1.05, 2.05)
    ax.set_ylim(-1, len(h) + 6)
    ax.set_xlabel("λ (IPS)")
    ax.set_ylabel("Participants")
    ax.set_title("c  Per participant (IPS)", loc="left", fontweight="bold")
    for k, (lab, kw) in enumerate((("Partially pooled", dict(color=IPS_C, mew=0)),
                                   ("No pooling", dict(mfc="none", mec="0.45", mew=.5, color="0.45")))):
        yy = .99 - .058 * k
        ax.plot(.04, yy, "o", ms=2.8, transform=ax.transAxes, **kw)
        ax.text(.07, yy, lab, fontsize=6, va="center", color=kw["color"], transform=ax.transAxes)
    ax.add_patch(plt.Rectangle((.015, .99 - .058 * 2 - .02), .05, .04, color="0.85", lw=0,
                               transform=ax.transAxes))
    ax.text(.07, .99 - .058 * 2, "Group μ, 95% CrI", fontsize=6, va="center", color="0.45",
            transform=ax.transAxes)

    # d-f: validity
    h = sub[sub.roi == "IPS"].copy()
    h["sid"] = h.subject.map(sid)
    post = np.stack(h.posterior.str.split().map(lambda v: np.array(v, float)).to_list())
    post = post / post.sum(1, keepdims=True)
    # d: hemispheres (normal approximation to each hemisphere's posterior)
    ax = fig.add_subplot(gs[1, 0])
    dl = rng.normal(h.L_mean, h.L_sd, (N_DRAWS, len(h)))
    dr = rng.normal(h.R_mean, h.R_sd, (N_DRAWS, len(h)))
    r = np.array([stats.spearmanr(a, b)[0] for a, b in zip(dl, dr)])
    ax.plot([-1, 2], [-1, 2], color="0.8", lw=.6, zorder=0)
    ax.errorbar(h.L_mean, h.R_mean, xerr=h.L_sd, yerr=h.R_sd, fmt="o", ms=3, color=IPS_C,
                mew=0, elinewidth=.5, alpha=.8, ecolor=(*matplotlib.colors.to_rgb(IPS_C), .35))
    ax.text(.03, .97, fmt_rho(r.mean(), *np.percentile(r, [2.5, 97.5])),
            transform=ax.transAxes, fontsize=6.5, va="top", color="0.2")
    ax.set_xlim(-1.1, 2.1)
    ax.set_ylim(-1.1, 2.6)
    ax.set_yticks([-1, 0, 1, 2])
    ax.set_xlabel("λ, left IPS")
    ax.set_ylabel("λ, right IPS")
    ax.set_title("d  Hemispheres (same sessions)", loc="left", fontweight="bold")
    # e: cvR2-profile lambda-hat
    lh = pd.read_csv(DATA / "lambda_hat.tsv", sep="\t", dtype={"subject": str})
    lh = lh[(lh.smoothed.astype(str) == "False") & lh.roi.isin(["IPS0", "IPS12", "IPS345"])]
    lh = (lh.assign(w=lh.lam_hat * lh.n_voxels).groupby("subject")[["w", "n_voxels"]].sum()
          .pipe(lambda d: d.w / d.n_voxels))
    lh.index = lh.index.map(sid)
    ax = fig.add_subplot(gs[1, 1], sharey=None)
    keep = h.sid.isin(lh.index).to_numpy()
    xe = lh.loc[h.sid[keep]].to_numpy()
    scatter(ax, xe, h[keep], "cvR²-profile λ̂ (IPS)", draw_rho(post[keep], LAM, xe, rng))
    ax.set_ylabel("Hierarchical λ (IPS)")
    ax.set_title("e  vs cvR² profile", loc="left", fontweight="bold")
    # f: behaviour
    lb = pd.read_csv(DATA / "lambda_behaviour.tsv", sep="\t", dtype={"subject": str})
    lb = lb.set_index(lb.subject.map(sid)).lam_b
    ax = fig.add_subplot(gs[1, 2])
    keep = h.sid.isin(lb.index).to_numpy()
    xb = lb.loc[h.sid[keep]].to_numpy()
    xj = xb + rng.uniform(-.012, .012, len(xb))                 # lam_b is on a 0.05 grid
    scatter(ax, xj, h[keep], "Behavioural λ (bids)", draw_rho(post[keep], LAM, xb, rng))
    ax.set_ylabel("Hierarchical λ (IPS)")
    ax.set_title("f  vs behaviour", loc="left", fontweight="bold")
    for a in fig.axes[-2:]:
        a.axhline(0, color=VAL_COL, lw=.6, alpha=.5, zorder=0)
        a.axhline(1, color=ORI_COL, lw=.6, alpha=.5, zorder=0)
        a.set_ylim(-1.1, 2.6)
        a.set_yticks([-1, 0, 1, 2])
    for a in fig.axes:
        a.set_title(a.get_title("left"), loc="left", fontweight="bold", pad=11)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    out = Path(out)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=250)
    print(f"Wrote {out}; n = {len(h)} (e: {len(xe)}, f: {len(xb)})")


if __name__ == "__main__":
    main()
