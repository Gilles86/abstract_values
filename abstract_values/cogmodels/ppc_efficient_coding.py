"""Posterior predictive checks for the efficient-coding fits.

The thing these models have to reproduce is not the average bid -- almost any
model gets that -- but the SHAPE of bias as a function of value, and how that
shape flips between the two mappings.  So the PPC is built around bias(value):

  page 1  group level, one panel per mapping: observed mean bias per stimulus
          value with the posterior predictive ribbon over it, stimulus density
          shaded behind.  Plus observed vs predicted response SD, since a model
          can match the bias curve and still get the noise badly wrong.
  page 2+ one panel per subject, both mappings overlaid, same quantity.

The predictive band is the honest one: each posterior draw SIMULATES a full
synthetic dataset with the same trial structure, and the summary is recomputed
on it, so the band carries parameter uncertainty PLUS the trial-level sampling
noise the model predicts.  Ribboning the predicted MEAN instead (the obvious
thing) gives a band that is invisible at this N.

The observed points get NO error band.  The predictive band already contains
the measurement noise -- it is the distribution of datasets like this one --
so the observed summary is one realisation to be checked against it.  Adding
SEM or bootstrap ink to the points double-counts that noise and lets a real
miss read as overlapping uncertainty.

Draws are subsampled (--n-draws) because the predictive is simulated per trial
per draw and the full posterior is far more than the plot can use.

Usage:
    python -m abstract_values.cogmodels.ppc_efficient_coding \
        --trace derivatives/cogmodels/efficient_coding_sequential_trace.nc \
        --paradigm-tsv notes/data/efficient_coding_paradigm.tsv \
        --model sequential --out notes/figures/ppc_sequential.pdf
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
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

mpl.rcParams.update({
    # scientific-figures house style. NB: do NOT add sns.set_context("paper")
    # after this -- it does not scale the block, it replaces 13 of 15 keys.
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelpad": 4, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2, "lines.markersize": 4,
    "legend.frameon": False, "legend.handlelength": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 150, "savefig.dpi": 300,
})

COND_C = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}
COND_L = {"cdf": "CDF", "inverse_cdf": "Inverse CDF"}
PAR_C = "#3B5BA5"


def load_paradigm(tsv):
    df = pd.read_csv(tsv, sep="\t")
    p = df.set_index(["subject", df.groupby("subject").cumcount()])
    p.index.names = ["subject", "trial"]
    return p


def posterior_subject_draws(idata, subjects, params, n_draws, seed=1):
    """Yield per-subject parameter frames for a random subset of posterior draws.

    The subject axis of each posterior variable is aligned to `subjects` via the
    InferenceData coordinate when one is present, and only falls back to
    positional order when it is not -- getting this wrong would silently swap
    subjects' parameters.
    """
    post = idata.posterior
    rng = np.random.default_rng(seed)
    n_chain, n_draw = post.sizes["chain"], post.sizes["draw"]
    picks = rng.choice(n_chain * n_draw, size=min(n_draws, n_chain * n_draw),
                       replace=False)

    aligned = {}
    for par in params:
        if par not in post:
            raise KeyError(f"{par} not in posterior (have: {list(post.data_vars)})")
        v = post[par]
        subj_dim = [d for d in v.dims if d not in ("chain", "draw")]
        if not subj_dim:
            # Non-hierarchical fit: one scalar per parameter, shared by every
            # row of the paradigm. Broadcast it to the subject axis.
            aligned[par] = np.repeat(v.values[:, :, None], len(subjects), axis=2)
            continue
        dim = subj_dim[0]
        coord = list(v.coords[dim].values) if dim in v.coords else None
        if coord is not None and set(map(str, coord)) == set(map(str, subjects)):
            order = [list(map(str, coord)).index(str(sub)) for sub in subjects]
            aligned[par] = v.values[:, :, order]
        else:
            if coord is not None:
                print(f"  ! {par}: coord {coord[:4]}… does not match subjects "
                      f"{list(subjects)[:4]}…, falling back to positional order")
            if v.sizes[dim] != len(subjects):
                raise ValueError(f"{par} has {v.sizes[dim]} entries for "
                                 f"{len(subjects)} subjects")
            aligned[par] = v.values
    for flat in picks:
        c, d = divmod(int(flat), n_draw)
        yield pd.DataFrame({par: aligned[par][c, d, :] for par in params},
                           index=pd.Index(subjects, name="subject"))


def run_ppc(model, paradigm, idata, params, n_draws, seed=1):
    """One simulated dataset per posterior draw, same trial structure as the data."""
    subjects = list(paradigm.index.get_level_values("subject").unique())
    # bauer.simulate draws through the global numpy RNG.
    np.random.seed(seed)
    draws = []
    for i, pars in enumerate(posterior_subject_draws(idata, subjects, params,
                                                     n_draws, seed=seed)):
        sim = model.simulate(paradigm, pars, n_samples=1).reset_index()
        # simulate() joins the paradigm back on, so the OBSERVED response rides
        # along; drop it before the simulated one takes the name, or every
        # downstream d["response"] silently becomes a two-column frame.
        sim = (sim.drop(columns=["response"])
                  .rename(columns={"simulated_response": "response"}))
        draws.append(sim)
        if (i + 1) % 10 == 0:
            print(f"  draw {i+1}/{n_draws}")
    return draws


def coverage(obs_curve, lo, hi):
    """Fraction of observed points inside the band.  For a nominal 95% band
    this should sit near 0.95: far below means the band is too narrow (the
    classic PPC bug), far above means the check has been smoothed until it
    can no longer fail.  The scientific-figures skill asks for this number
    every time, so it is printed and put on the panel."""
    o = obs_curve.reindex(lo.index)
    ok = ((o >= lo) & (o <= hi))
    return float(ok.mean())


def _bias_by_value(df, resp_col):
    """Mean (resp - value) per (mapping, value)."""
    d = df.copy()
    d["bias"] = d[resp_col] - d["value"]
    return d.groupby(["mapping", "value"])["bias"].mean()


def page_group(obs, pred_draws, out_pdf):
    """Observed bias(value) per condition with the posterior predictive ribbon."""
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.6), constrained_layout=True)
    bias_pads, sd_pads = [], []      # shared y-limits: same quantity, same scale
    for j, cond in enumerate(["cdf", "inverse_cdf"]):
        o = obs[obs.mapping == cond]
        og = o.groupby("value").agg(bias=("bias", "mean"), sd=("response", "std"))

        # posterior predictive ribbon over draws
        sim = [d[d.mapping == cond] for d in pred_draws]
        pb = [d.groupby("value")["bias"].mean() for d in sim]
        pb = pd.concat(pb, axis=1)
        lo, mid, hi = pb.quantile(.025, axis=1), pb.median(axis=1), pb.quantile(.975, axis=1)

        ax = axes[0, j]
        vals = np.sort(o["value"].unique())
        dens = 1.0 / np.gradient(vals)
        axd = ax.twinx(); axd.fill_between(vals, 0, dens, color="0.88", lw=0, zorder=0)
        axd.set_ylim(0, dens.max() * 3.2); axd.set_yticks([])
        axd.spines["right"].set_visible(False)
        ax.set_zorder(axd.get_zorder() + 1); ax.patch.set_visible(False)

        ax.axhline(0, color="0.4", lw=0.8, ls="--")
        ax.fill_between(mid.index, lo, hi, color=COND_C[cond], alpha=0.28, lw=0,
                        label="Predicted (95% PPC, simulated data)")
        ax.plot(og.index, og["bias"], "-o", color=COND_C[cond], ms=5.5, lw=1.2,
                mec="white", mew=0.6, zorder=3)
        cov_bias = coverage(og["bias"], lo, hi)
        print(f"  {COND_L[cond]:<12} band coverage: bias {cov_bias:.2f}", end="")
        # The predictive blows up in the outermost value bins: probability mass
        # falls off the end of the response grid there, dragging the predicted
        # mean down and inflating its SD. Clip to the observed range so the real
        # structure stays visible, and mark the affected bins.
        bias_pads.append(1.6 * max(og["bias"].abs().max(), 0.5))
        for edge in (vals[0], vals[-1]):
            ax.axvline(edge, color="0.6", lw=0.7, ls=":", zorder=0)
        off = ((mid < -bias_pads[-1]) | (mid > bias_pads[-1]))
        if off.any():
            ax.annotate("edge bins: predictive off-scale\n(grid truncation)",
                        xy=(0.5, 0.03), xycoords="axes fraction", ha="center",
                        va="bottom", fontsize=6.5, color="0.45")
        ax.set_xlabel("True value (CHF)"); ax.set_ylabel("Bias (CHF)")
        ax.set_title(f"{COND_L[cond]} — bias vs value", color="0.2")
        # Direct labels rather than a legend (house style): the reader should
        # not have to look away from the data to decode it.
        ax.text(0.02, 0.97, "Dots: observed", transform=ax.transAxes,
                color=COND_C[cond], fontsize=7, va="top", ha="left")
        ax.text(0.02, 0.88, "Band: predicted 95%", transform=ax.transAxes,
                color=COND_C[cond], fontsize=7, va="top", ha="left")
        ax.text(0.98, 0.03, f"95% band covers {cov_bias:.0%} of points",
                transform=ax.transAxes, color="0.45", fontsize=6.5,
                va="bottom", ha="right")

        # response SD: does the model get the noise level right?  Computed the
        # same way on both sides -- SD of the responses in each value bin --
        # so the two are directly comparable.
        ps = [d.groupby("value")["response"].std() for d in sim]
        ax = axes[1, j]
        ax.plot(og.index, og["sd"], "-o", color=COND_C[cond], ms=5.5, lw=1.2,
                mec="white", mew=0.6, zorder=3)
        if ps:
            ps = pd.concat(ps, axis=1)
            s_lo, s_hi = ps.quantile(.025, axis=1), ps.quantile(.975, axis=1)
            ax.fill_between(ps.index, s_lo, s_hi, color=COND_C[cond], alpha=0.28, lw=0)
            cov_sd = coverage(og["sd"], s_lo, s_hi)
            print(f" | SD {cov_sd:.2f}")
            ax.text(0.98, 0.03, f"95% band covers {cov_sd:.0%} of points",
                    transform=ax.transAxes, color="0.45", fontsize=6.5,
                    va="bottom", ha="right")
        sd_pads.append(1.8 * max(og["sd"].max(), 0.5))
        ax.set_xlabel("True value (CHF)"); ax.set_ylabel("Response SD (CHF)")
        ax.set_title(f"{COND_L[cond]} — spread", color="0.2")
    # One scale per row: the two mappings show the same quantity, so panel-
    # specific autoscaling would make a small bias look like a large one.
    for j in range(2):
        axes[0, j].set_ylim(-max(bias_pads), max(bias_pads))
        axes[1, j].set_ylim(0, max(sd_pads))
    fig.suptitle("Group posterior predictive check  ·  band = 95% of simulated "
                 "datasets (parameter + trial-level noise)",
                 fontsize=8, y=1.03, color="0.15")
    sns.despine(fig=fig, offset=5, trim=False)
    out_pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


THETA = np.linspace(0, 180, 361)
PHI = np.deg2rad(THETA) * 2


def implied_prior(post, i, n_sub):
    """That subject's orientation prior, from whichever parameterisation was fitted."""
    if "prior_weight" in post.data_vars:
        w = float(np.median(post.prior_weight.values.reshape(-1, n_sub)[:, i]))
        pr = np.maximum(1.0 - w * np.abs(np.sin(PHI)), 1e-6)
        return pr / np.trapezoid(pr, PHI), w
    coefs = sorted(v for v in post.data_vars
                   if len(v) == 8 and v.startswith("prior_") and v[6] in "ab")
    if not coefs:
        return None, None
    logp = np.zeros_like(PHI)
    for v in coefs:
        k = int(v[-1]); c = np.cos if v[6] == "a" else np.sin
        logp += float(np.median(post[v].values.reshape(-1, n_sub)[:, i])) * c(k * PHI)
    pr = np.exp(logp - logp.max())
    return pr / np.trapezoid(pr, PHI), None


def page_implied_priors(idata, out_pdf):
    """The orientation prior each subject's fitted parameters imply."""
    post = idata.posterior
    pars = _par_names(post)
    subs = [int(x) for x in post[pars[0]].coords["subject"].values]
    curves = [implied_prior(post, i, len(subs)) for i in range(len(subs))]
    if curves[0][0] is None:
        return
    fig, ax = plt.subplots(figsize=(4.4, 3.0), constrained_layout=True)
    ws = [w for _, w in curves if w is not None]
    cmap = sns.color_palette("mako", as_cmap=True)
    lo, hi = (min(ws), max(ws)) if ws else (0, 1)
    for (pr, w), sub in zip(curves, subs):
        c = cmap(0.15 + 0.6 * (1 - (w - lo) / (hi - lo + 1e-12))) if w is not None else "0.5"
        ax.plot(THETA, pr / pr.mean(), color=c, lw=0.9, alpha=0.85)
    ref = np.maximum(1.0 - 0.5 * np.abs(np.sin(PHI)), 1e-6)
    ax.plot(THETA, ref / ref.mean(), color="#C44E52", lw=1.6, ls=(0, (2, 1.6)))
    ax.axhline(1.0, color="0.75", lw=0.7, ls=":")
    ax.set_xticks([0, 45, 90, 135, 180]); ax.set_xlim(0, 180)
    ax.set_xlabel("Orientation θ (deg)")
    ax.set_ylabel("Prior density (uniform = 1)")
    ax.set_title("Implied orientation prior, one line per subject", color="0.2")
    ax.text(90, ax.get_ylim()[1] * 0.97, "Dashed red: the paper's fixed prior (w = 0.5)",
            color="#C44E52", fontsize=6.5, ha="center", va="top")
    if ws:
        ax.text(2, ax.get_ylim()[1] * 0.97, f"w = {lo:.2f} – {hi:.2f}\ndarker = more peaked",
                color="0.35", fontsize=6.5, ha="left", va="top")
    sns.despine(fig=fig, offset=4)
    out_pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _par_names(post):
    """Free parameters present in this trace, in a sensible reading order."""
    base = [v for v in ("kappa_r", "sigma_rep", "sigma_motor", "prior_weight")
            if v in post.data_vars]
    fourier = sorted(v for v in post.data_vars
                     if len(v) == 8 and v.startswith("prior_") and v[6] in "ab")
    return base + fourier


def page_group_parameters(idata, out_pdf):
    """Group-level posteriors: the population mean and spread of each parameter."""
    post = idata.posterior
    pars = _par_names(post)
    pars = [p for p in pars if f"{p}_mu" in post.data_vars]
    if not pars:
        return
    fig, axes = plt.subplots(2, len(pars), figsize=(2.0 * len(pars) + 0.8, 4.0),
                             constrained_layout=True, squeeze=False)
    for j, par in enumerate(pars):
        for i, (suffix, label) in enumerate((("_mu", "Group mean"),
                                             ("_sd", "Between-subject SD"))):
            ax = axes[i, j]
            v = post.get(par + suffix)
            if v is None:
                ax.set_visible(False)
                continue
            x = v.values.ravel()
            sns.kdeplot(x=x, ax=ax, color=PAR_C, fill=True, alpha=0.25, lw=1.2)
            lo, hi = az.hdi(x, hdi_prob=0.95)
            ax.axvspan(lo, hi, color=PAR_C, alpha=0.12, lw=0)
            ax.axvline(np.median(x), color=PAR_C, lw=1.2)
            ax.set_ylabel(label if j == 0 else "")
            ax.set_yticks([])
            ax.set_xlabel(par.replace("_", " "))
            ax.set_title(f"{np.median(x):.2f}  [{lo:.2f}, {hi:.2f}]",
                         fontsize=7, color="0.35")
    fig.suptitle("Group-level parameters  ·  median and 95% HDI",
                 fontsize=8, y=1.04, color="0.15")
    sns.despine(fig=fig, offset=4)
    out_pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_subject_parameters(idata, out_pdf):
    """Per-subject posteriors: median and 95% HDI, sorted within each parameter."""
    post = idata.posterior
    pars = _par_names(post)
    subs = [int(x) for x in post[pars[0]].coords["subject"].values]
    fig, axes = plt.subplots(1, len(pars), figsize=(1.9 * len(pars) + 0.9, 3.8),
                             constrained_layout=True, squeeze=False)
    for j, par in enumerate(pars):
        ax = axes[0, j]
        x = post[par].values.reshape(-1, len(subs))
        med = np.median(x, axis=0)
        hdi = np.stack([az.hdi(x[:, i], hdi_prob=0.95) for i in range(len(subs))])
        order = np.argsort(med)
        y = np.arange(len(subs))
        ax.hlines(y, hdi[order, 0], hdi[order, 1], color=PAR_C, lw=1.0, alpha=0.7)
        ax.plot(med[order], y, "o", ms=3.4, color=PAR_C, mec="white", mew=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{subs[i]:02d}" for i in order], fontsize=5.5)
        ax.set_ylim(-1, len(subs))
        ax.set_xlabel(par.replace("_", " "))
        if par == "kappa_r":
            ax.set_xscale("log")
        if j == 0:
            ax.set_ylabel("Subject")
    fig.suptitle("Per-subject parameters  ·  median and 95% HDI, sorted",
                 fontsize=8, y=1.03, color="0.15")
    sns.despine(fig=fig, offset=4)
    out_pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_parameter_correlations(idata, out_pdf):
    """Across-subject correlations between the per-subject posterior medians."""
    post = idata.posterior
    pars = _par_names(post)
    if len(pars) < 2:
        return
    subs = [int(x) for x in post[pars[0]].coords["subject"].values]
    med, hdi = {}, {}
    for par in pars:
        x = post[par].values.reshape(-1, len(subs))
        med[par] = np.median(x, axis=0)
        hdi[par] = np.stack([az.hdi(x[:, i], hdi_prob=0.95) for i in range(len(subs))])
    pairs = [(a, b) for i, a in enumerate(pars) for b in pars[i + 1:]]
    ncol = min(4, len(pairs))
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.3 * nrow),
                             constrained_layout=True, squeeze=False)
    for ax, (a, b) in zip(axes.flat, pairs):
        x, y = med[a], med[b]
        # Credible intervals in both directions: a subject whose kappa is only
        # known to within a factor of three should not read as a point.
        xerr = np.abs(np.stack([x - hdi[a][:, 0], hdi[a][:, 1] - x]))
        yerr = np.abs(np.stack([y - hdi[b][:, 0], hdi[b][:, 1] - y]))
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none",
                    ecolor=PAR_C, elinewidth=0.7, alpha=0.35, zorder=1)
        ax.plot(x, y, "o", ms=5, color=PAR_C, mec="white", mew=0.6, zorder=2)
        r = np.corrcoef(np.log(x) if a == "kappa_r" else x,
                        np.log(y) if b == "kappa_r" else y)[0, 1]
        ax.set_xlabel(a.replace("_", " ")); ax.set_ylabel(b.replace("_", " "))
        if a == "kappa_r":
            ax.set_xscale("log")
        if b == "kappa_r":
            ax.set_yscale("log")
        ax.set_title(f"r = {r:+.2f}", fontsize=7.5, color="0.35")
    for ax in axes.flat[len(pairs):]:
        ax.set_visible(False)
    fig.suptitle("Across-subject parameter correlations  ·  posterior median "
                 "and 95% HDI per subject  (κ_r on a log axis)",
                 fontsize=8, y=1.03, color="0.15")
    sns.despine(fig=fig, offset=4)
    out_pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def pages_subjects(obs, pred_draws, out_pdf, per_page=14, par_med=None,
                   priors=None):
    subs = sorted(obs["subject"].unique())
    # Balance the pages: 26 subjects at 12 per page leaves a page with two
    # panels on it. Round the page count first, then split evenly.
    n_pages = max(1, int(np.ceil(len(subs) / per_page)))
    per_page = int(np.ceil(len(subs) / n_pages))
    for start in range(0, len(subs), per_page):
        chunk = subs[start:start + per_page]
        ncol = 4; nrow = int(np.ceil(len(chunk) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.5 * nrow),
                                 constrained_layout=True, squeeze=False)
        for ax, s in zip(axes.flat, chunk):
            for cond in ["cdf", "inverse_cdf"]:
                o = obs[(obs.subject == s) & (obs.mapping == cond)]
                if o.empty:
                    continue
                og = o.groupby("value")["bias"].mean()
                pb = pd.concat([d[(d.subject == s) & (d.mapping == cond)]
                                .groupby("value")["bias"].mean() for d in pred_draws], axis=1)
                ax.fill_between(pb.index, pb.quantile(.025, axis=1), pb.quantile(.975, axis=1),
                                color=COND_C[cond], alpha=0.25, lw=0)
                ax.plot(og.index, og, color=COND_C[cond], marker="o", ms=2.5, lw=1.1)
            if priors is not None and s in priors and priors[s] is not None:
                # The subject's own orientation prior, in the corner: the shape
                # that produced this panel's predictions.
                ins = ax.inset_axes([0.72, 0.70, 0.27, 0.28])
                pr = priors[s]
                ins.plot(THETA, pr / pr.mean(), color="#3B5BA5", lw=0.8)
                ins.axhline(1.0, color="0.8", lw=0.5, ls=":")
                ins.set_xticks([]); ins.set_yticks([])
                ins.set_xlim(0, 180)
                # Fixed across subjects, or every inset autoscales to its own
                # peak and a flat prior looks identical to a sharp one.
                ins.set_ylim(0, 2.8)
                for sp in ins.spines.values():
                    sp.set_linewidth(0.5); sp.set_color("0.7")
                ins.patch.set_alpha(0.6)
            ax.axhline(0, color="0.5", lw=0.7, ls="--")
            obs_s = obs[obs.subject == s]
            pad = 1.6 * max((obs_s["response"] - obs_s["value"]).abs()
                            .groupby([obs_s["mapping"], obs_s["value"]]).mean().abs().max(), 0.5)
            ax.set_ylim(-pad, pad)
            title = f"sub-{int(s):02d}"
            if par_med is not None and s in par_med:
                # The parameters that produced this panel's prediction, on the
                # panel: a misfit is only interpretable next to the values that
                # caused it.
                fmt = {"κ": "{:.0f}", "σ": "{:.2f}", "σm": "{:.2f}", "w": "{:.2f}"}
                title += "   " + "  ".join(
                    f"{k}={fmt.get(k, '{:.3g}').format(v)}" for k, v in par_med[s].items())
            ax.set_title(title, fontsize=6.5, color="0.2")
            ax.tick_params(labelsize=7)
        for ax in axes.flat[len(chunk):]:
            ax.set_visible(False)
        for ax in axes[-1, :]:
            ax.set_xlabel("Value (CHF)", fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel("Bias (CHF)", fontsize=8)
        fig.suptitle("Per-subject posterior predictive check  ·  line = observed, "
                     "band = 95% of simulated datasets", fontsize=8, y=1.02,
                     color="0.15")
        sns.despine(fig=fig, offset=3, trim=False)
        out_pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def run(trace, paradigm_tsv, model_name, out, n_draws, grid_resolution,
        perceptual_prior="long_term", lapse_rate=0.01,
        fit_prior_weight=False, no_seam_crossing=False,
        cardinal_truncation=False):
    import arviz as az
    from abstract_values.cogmodels.fit_efficient_coding import make_model

    paradigm = load_paradigm(paradigm_tsv)
    idata = az.from_netcdf(trace)
    # The prior has to match the one the trace was fitted under, or the PPC
    # scores the posterior against a different generative model than the one
    # that produced it.
    model = make_model(paradigm, model_name, grid_resolution,
                       lapse_rate=lapse_rate, perceptual_prior=perceptual_prior,
                       fit_prior_weight=fit_prior_weight,
                       no_seam_crossing=no_seam_crossing,
                       cardinal_truncation=cardinal_truncation)

    params = {"perception": ["kappa_r"], "valuation": ["sigma_rep"],
              "sequential": ["kappa_r", "sigma_rep"],
              "categorical": ["kappa_r", "sigma_rep"]}[model_name]
    if fit_prior_weight:
        params = params + ["prior_weight"]
    print(f"Running PPC ({n_draws} draws, params {params})…")
    pred_draws = run_ppc(model, paradigm, idata, params, n_draws)
    for d in pred_draws:
        d["bias"] = d["response"] - d["value"]
    print(f"  {len(pred_draws)} predictive draws")

    obs = paradigm.reset_index()
    obs["bias"] = obs["response"] - obs["value"]

    # per-subject posterior medians, for the panel annotations
    post = idata.posterior
    par_list = _par_names(post)
    par_subs = [int(x) for x in post[par_list[0]].coords["subject"].values]
    short = {"kappa_r": "κ", "sigma_rep": "σ", "sigma_motor": "σm",
             "prior_weight": "w"}
    par_med = {sub: {short.get(par, par.replace("prior_", "")):
                     float(np.median(post[par].values.reshape(-1, len(par_subs))[:, i]))
                     for par in par_list}
               for i, sub in enumerate(par_subs)}
    priors = {sub: implied_prior(post, i, len(par_subs))[0]
              for i, sub in enumerate(par_subs)}

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        page_group(obs, pred_draws, pdf)
        page_group_parameters(idata, pdf)
        page_subject_parameters(idata, pdf)
        page_parameter_correlations(idata, pdf)
        page_implied_priors(idata, pdf)
        pages_subjects(obs, pred_draws, pdf, par_med=par_med, priors=priors)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True)
    p.add_argument("--paradigm-tsv", default="notes/data/efficient_coding_paradigm.tsv")
    p.add_argument("--model", default="sequential",
                   choices=["perception", "valuation", "sequential", "categorical"])
    p.add_argument("--grid-resolution", type=int, default=101)
    p.add_argument("--n-draws", type=int, default=200)
    p.add_argument("--perceptual-prior", default="long_term",
                   choices=["long_term", "uniform"],
                   help="Must match the prior the trace was fitted under.")
    p.add_argument("--lapse-rate", type=float, default=0.01)
    p.add_argument("--fit-prior-weight", action="store_true",
                   help="Match a trace fitted with a free prior peakedness.")
    p.add_argument("--no-seam-crossing", action="store_true",
                   help="Match a trace fitted with the 0/180 deg seam closed.")
    p.add_argument("--cardinal-truncation", action="store_true",
                   help="Match a trace fitted with perception truncated at 0/90/180.")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = a.out or f"notes/figures/ppc_{a.model}.pdf"
    run(a.trace, a.paradigm_tsv, a.model, out, a.n_draws, a.grid_resolution,
        perceptual_prior=a.perceptual_prior, lapse_rate=a.lapse_rate,
        fit_prior_weight=a.fit_prior_weight, no_seam_crossing=a.no_seam_crossing,
        cardinal_truncation=a.cardinal_truncation)


if __name__ == "__main__":
    main()
