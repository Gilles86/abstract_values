"""Posterior predictive checks for the efficient-coding fits.

The thing these models have to reproduce is not the average bid -- almost any
model gets that -- but the SHAPE of bias as a function of value, and how that
shape flips between the two mappings.  So the PPC is built around bias(value):

  page 1  group level, one panel per mapping: observed mean bias per stimulus
          value with the posterior predictive ribbon over it, stimulus density
          shaded behind.  Plus observed vs predicted response SD, since a model
          can match the bias curve and still get the noise badly wrong.
  page 2+ one panel per subject, both mappings overlaid, same quantity.

Draws are subsampled (--n-draws) because the predictive is computed per trial
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
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelpad": 4, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2, "legend.frameon": False, "legend.handlelength": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})
sns.set_context("paper")

COND_C = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}
COND_L = {"cdf": "CDF", "inverse_cdf": "Inverse CDF"}


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
            raise ValueError(f"{par} has no subject dimension; is this a hierarchical fit?")
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
    """Predicted response distribution per trial, for a subset of posterior draws."""
    subjects = list(paradigm.index.get_level_values("subject").unique())
    draws = []
    for i, pars in enumerate(posterior_subject_draws(idata, subjects, params,
                                                     n_draws, seed=seed)):
        pred = model.predict(paradigm, pars)
        draws.append(pred.reset_index())
        if (i + 1) % 10 == 0:
            print(f"  draw {i+1}/{n_draws}")
    return draws


def _bias_by_value(df, resp_col):
    """Mean (resp - value) per (mapping, value)."""
    d = df.copy()
    d["bias"] = d[resp_col] - d["value"]
    return d.groupby(["mapping", "value"])["bias"].mean()


def page_group(obs, pred_draws, out_pdf):
    """Observed bias(value) per condition with the posterior predictive ribbon."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.4), constrained_layout=True)
    for j, cond in enumerate(["cdf", "inverse_cdf"]):
        o = obs[obs.mapping == cond]
        og = o.groupby("value").agg(bias=("bias", "mean"), sd=("response", "std"))

        # posterior predictive ribbon over draws
        pb = [d[d.mapping == cond].groupby("value")["bias"].mean()
              for d in pred_draws]
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
                        label="Predicted (95% PPC)")
        ax.plot(mid.index, mid, color=COND_C[cond], lw=1.2, ls="--")
        ax.plot(og.index, og["bias"], color="0.15", marker="o", ms=3.5, lw=1.3,
                label="Observed")
        # The predictive blows up in the outermost value bins: probability mass
        # falls off the end of the response grid there, dragging the predicted
        # mean down and inflating its SD. Clip to the observed range so the real
        # structure stays visible, and mark the affected bins.
        pad = 1.6 * max(og["bias"].abs().max(), 0.5)
        ax.set_ylim(-pad, pad)
        for edge in (vals[0], vals[-1]):
            ax.axvline(edge, color="0.6", lw=0.7, ls=":", zorder=0)
        off = ((mid < -pad) | (mid > pad))
        if off.any():
            ax.annotate("edge bins: predictive off-scale\n(grid truncation)",
                        xy=(0.5, 0.03), xycoords="axes fraction", ha="center",
                        va="bottom", fontsize=6.5, color="0.45")
        ax.set_xlabel("True value (CHF)"); ax.set_ylabel("Bias (CHF)")
        ax.set_title(f"{COND_L[cond]} — bias vs value", fontsize=9, color="0.2")
        ax.legend(loc="upper left", fontsize=7)

        # response SD: does the model get the noise level right?
        ps = [d[d.mapping == cond].groupby("value")["predicted_sd"].mean()
              for d in pred_draws if "predicted_sd" in d]
        ax = axes[1, j]
        ax.plot(og.index, og["sd"], color="0.15", marker="o", ms=3.5, lw=1.3,
                label="Observed SD")
        if ps:
            ps = pd.concat(ps, axis=1)
            ax.fill_between(ps.index, ps.quantile(.025, axis=1), ps.quantile(.975, axis=1),
                            color=COND_C[cond], alpha=0.28, lw=0)
            ax.plot(ps.index, ps.median(axis=1), color=COND_C[cond], lw=1.2, ls="--",
                    label="Predicted SD")
        ax.set_ylim(0, 1.8 * max(og["sd"].max(), 0.5))
        ax.set_xlabel("True value (CHF)"); ax.set_ylabel("Response SD (CHF)")
        ax.set_title(f"{COND_L[cond]} — spread", fontsize=9, color="0.2")
        ax.legend(loc="best", fontsize=7)
    fig.suptitle("Group posterior predictive check", fontsize=10, y=1.03, color="0.15")
    sns.despine(fig=fig, offset=4, trim=False)
    out_pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def pages_subjects(obs, pred_draws, out_pdf, per_page=12):
    subs = sorted(obs["subject"].unique())
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
            ax.axhline(0, color="0.5", lw=0.7, ls="--")
            obs_s = obs[obs.subject == s]
            pad = 1.6 * max((obs_s["response"] - obs_s["value"]).abs()
                            .groupby([obs_s["mapping"], obs_s["value"]]).mean().abs().max(), 0.5)
            ax.set_ylim(-pad, pad)
            ax.set_title(f"sub-{int(s):02d}", fontsize=8, color="0.2")
            ax.tick_params(labelsize=7)
        for ax in axes.flat[len(chunk):]:
            ax.set_visible(False)
        for ax in axes[-1, :]:
            ax.set_xlabel("Value (CHF)", fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel("Bias (CHF)", fontsize=8)
        fig.suptitle("Per-subject posterior predictive check  ·  "
                     "line = observed, band = 95% PPC", fontsize=10, y=1.02, color="0.15")
        sns.despine(fig=fig, offset=3, trim=False)
        out_pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def run(trace, paradigm_tsv, model_name, out, n_draws, grid_resolution):
    import arviz as az
    from abstract_values.cogmodels.fit_efficient_coding import make_model

    paradigm = load_paradigm(paradigm_tsv)
    idata = az.from_netcdf(trace)
    model = make_model(paradigm, model_name, grid_resolution)

    params = {"perception": ["kappa_r"], "valuation": ["sigma_rep"],
              "sequential": ["kappa_r", "sigma_rep"]}[model_name]
    print(f"Running PPC ({n_draws} draws, params {params})…")
    pred_draws = run_ppc(model, paradigm, idata, params, n_draws)
    for d in pred_draws:
        d["bias"] = d["predicted_mean"] - d["value"]
    print(f"  {len(pred_draws)} predictive draws")

    obs = paradigm.reset_index()
    obs["bias"] = obs["response"] - obs["value"]

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        page_group(obs, pred_draws, pdf)
        pages_subjects(obs, pred_draws, pdf)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True)
    p.add_argument("--paradigm-tsv", default="notes/data/efficient_coding_paradigm.tsv")
    p.add_argument("--model", default="sequential",
                   choices=["perception", "valuation", "sequential"])
    p.add_argument("--grid-resolution", type=int, default=31)
    p.add_argument("--n-draws", type=int, default=100)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = a.out or f"notes/figures/ppc_{a.model}.pdf"
    run(a.trace, a.paradigm_tsv, a.model, out, a.n_draws, a.grid_resolution)


if __name__ == "__main__":
    main()
