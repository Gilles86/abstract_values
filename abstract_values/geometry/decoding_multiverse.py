"""Plain decoding as the neural measure: does the gradient survive decoder choices?

Trial-wise leave-one-run-out value decoding (decode_value.py; log-Gaussian
pRF, top-100 voxels, lambda 0.1) stores the LIKELIHOOD over a 50-point CHF
grid, so the prior can be applied afterwards. Specifications:

  noise   spherical | full covariance (Student-t residual model)
  prior   flat (likelihood only) | presented (density of the values actually
          shown in that session, Gaussian kernel 1.5 CHF -- the objective prior)
  measure accuracy  r(posterior mean, true value)
          precision -log posterior SD, averaged over trials

Each specification gives one subject x ROI table; the gradient analysis of
figure_gradient_versions (joint regression on perceptual and value
precision + motion, interaction parietal - early) is run on every one.

    python -m abstract_values.geometry.decoding_multiverse
    -> notes/data/decoding_multiverse_{measures,gradients}.tsv,
       notes/figures/fig_decoding_multiverse.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abstract_values.visualize import figure_gradient_versions as fv

REPO = Path(__file__).resolve().parents[2]
DEC = Path("/data/ds-abstractvalue/derivatives/decoding/value")
ROI = {"BensonV1": "V1", "BensonV2": "V2", "BensonV3": "V3", "V3ab": "V3a/b",
       "IPS0": "IPS0", "IPS12": "IPS1–2", "IPS345": "IPS3–5"}
PIL = {"pil01": "01", "pil02": "02"}
KERNEL = 1.5


def summarise(f):
    d = pd.read_csv(f, sep="\t")
    grid = np.array([float(c) for c in d.columns[4:]])
    L = d.iloc[:, 4:].to_numpy(float)
    rows = []
    for prior in ("flat", "presented"):
        P = L.copy()
        if prior == "presented":
            for ses, idx in d.groupby("session").groups.items():
                shown = d.loc[idx, "true_value_chf"].to_numpy()
                dens = np.exp(-0.5 * ((grid[:, None] - shown[None]) / KERNEL) ** 2).sum(1)
                P[np.asarray(idx)] *= dens / dens.sum()
        P = P / P.sum(1, keepdims=True)
        mu = P @ grid
        sd = np.sqrt(np.maximum(P @ grid ** 2 - mu ** 2, 1e-12))
        ok = np.isfinite(mu)
        rows.append(dict(prior=prior,
                         accuracy=np.corrcoef(mu[ok], d.true_value_chf[ok])[0, 1],
                         precision=np.mean(-np.log(sd[ok]))))
    return rows


def measures():
    out = []
    for mask, roi in ROI.items():
        for noise in ("spherical", "full"):
            for f in DEC.glob(f"sub-*/func/*_mask-{mask}_nvoxels-100_noise-{noise}"
                              "_lambda-0.1_pars.tsv"):
                s = f.name.split("_")[0].replace("sub-", "")
                for r in summarise(f):
                    out.append(dict(subject=PIL.get(s, s), roi=roi, noise=noise, **r))
    return pd.DataFrame(out)


def main():
    m = measures()
    m.to_csv(REPO / "notes/data/decoding_multiverse_measures.tsv", sep="\t", index=False)
    tab = fv.subject_table(fv.PARAMS)
    rows, profiles = [], {}
    for (noise, prior), g in m.groupby(["noise", "prior"]):
        for meas in ("accuracy", "precision"):
            wide = g.pivot(index="subject", columns="roi", values=meas)
            if wide.shape[1] < len(fv.DORSAL) or wide.notna().sum().min() < 20:
                print(f"skip {noise}/{prior}/{meas}: ROIs {list(wide.columns)}")
                continue
            regs = ["perc_prec", "val_prec"]
            subs = [s for s in tab.index if s in wide.index]
            est = fv.profile(wide, tab, regs)
            B = fv.boot(lambda i: fv.profile(wide, tab, regs, i), len(subs))
            gr = fv.gradient_stat(est)[0]
            Bg = np.array([fv.gradient_stat(b)[0] for b in B])
            rows.append(dict(noise=noise, prior=prior, measure=meas, n=len(subs),
                             gradient=gr, se=Bg.std(), p=np.mean(Bg <= 0),
                             group_mean_parietal=wide[fv.LATE].mean().mean(),
                             group_mean_early=wide[fv.EARLY].mean().mean()))
            profiles[(noise, prior, meas)] = est
    res = pd.DataFrame(rows)
    res.to_csv(REPO / "notes/data/decoding_multiverse_gradients.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.4), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.6, 1]))
    x = np.arange(len(fv.DORSAL))
    ax = axes[0]
    styles = {("spherical", "flat"): "-", ("spherical", "presented"): (0, (4, 2)),
              ("full", "flat"): (0, (1, 1.5)), ("full", "presented"): (0, (5, 1, 1, 1))}
    for (noise, prior, meas), est in profiles.items():
        if meas != "precision":
            continue
        for k, col in enumerate((fv.PER_C, fv.REP_C)):
            ax.plot(x, [est[r][k] for r in fv.DORSAL], color=col, lw=1.3,
                    ls=styles[(noise, prior)], alpha=0.9)
    ax.axhline(0, color="0.7", lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(x)
    ax.set_xticklabels(fv.DORSAL)
    ax.set_ylabel("Weight on decoding\nprecision (std. β)")
    ax.text(0.02, 0.97, "Value precision", color=fv.REP_C, transform=ax.transAxes,
            fontsize=7, fontweight="bold", va="top")
    ax.text(0.02, 0.86, "Perceptual precision", color=fv.PER_C, transform=ax.transAxes,
            fontsize=7, fontweight="bold", va="top")
    for i, ((noise, prior), ls) in enumerate(styles.items()):
        ax.plot([0.62, 0.70], [0.97 - 0.1 * i] * 2, color="0.3", ls=ls, lw=1.2,
                transform=ax.transAxes)
        ax.text(0.72, 0.97 - 0.1 * i, f"{noise.capitalize()} noise, {prior} prior",
                transform=ax.transAxes, fontsize=6, va="center", color="0.3")
    sns.despine(ax=ax, offset=4, trim=True)
    ax = axes[1]
    res = res.reset_index(drop=True)
    yy = np.arange(len(res))[::-1]
    ax.errorbar(res.gradient, yy, xerr=res.se, fmt="o", color="0.2", ms=4,
                elinewidth=1, capsize=0)
    ax.axvline(0, color="0.7", lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{r.noise}, {r.prior}, {r.measure}  (P = {r.p:.2f})"
                        for r in res.itertuples()], fontsize=6)
    ax.set_xlabel("Interaction (std. β), ±1 SE")
    sns.despine(ax=ax, offset=4, trim=True)
    out = REPO / "notes/figures/fig_decoding_multiverse.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
