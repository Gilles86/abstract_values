"""Do the cognitive-model noise parameters track neural acuity, stage by stage?

The two-stage model says perception is noisy in ORIENTATION space and valuation
in VALUE space.  If those stages are real, the per-subject parameters should
line up with the acuity of the corresponding neural representation, and only
that one:

    kappa_r    (perceptual precision)  <->  V1 orientation decoding
    sigma_rep  (value noise)           <->  NPC (IPS) value decoding

and the crossed pairs -- kappa_r against NPC, sigma_rep against V1 -- should
not.  That double dissociation is the test; a single correlation is not, since
a subject who is simply a better participant is better at everything.

Correlations are computed PER POSTERIOR DRAW, so the interval on r carries the
uncertainty in the parameter itself.  With 26 subjects and kappa_r known only
to within a factor of ~3 for many of them, a point estimate of r would overstate
what we know.

Writes notes/figures/brain_behavior_<tag>.pdf and a TSV of the correlations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

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

MATCH, CROSS = "#3B5BA5", "#9C9C9C"


def load_parameters(trace):
    d = az.from_netcdf(trace)
    post = d.posterior
    subs = [int(x) for x in post.kappa_r.coords["subject"].values]
    out = {}
    for par in ("kappa_r", "sigma_rep", "sigma_motor", "prior_weight"):
        if par in post.data_vars:
            out[par] = post[par].values.reshape(-1, len(subs))
    return subs, out


def draws_correlation(x_draws, y, n_draws=600, seed=0):
    """Spearman rho carrying BOTH sources of uncertainty.

    Per iteration: take one posterior draw of the parameter AND bootstrap the
    subjects. Posterior spread alone gives an interval an order of magnitude too
    narrow -- with n = 26 the sampling error on rho is about +-0.4, and that
    dominates. Quoting the posterior-only interval would turn a null result into
    a confident one.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    rs = []
    for _ in range(n_draws):
        i = rng.integers(x_draws.shape[0])
        b = rng.integers(0, n, size=n)                    # cluster bootstrap
        if len(np.unique(y[b])) < 3:
            continue
        rs.append(stats.spearmanr(x_draws[i][b], y[b]).statistic)
    rs = np.array(rs)
    return np.median(rs), np.percentile(rs, 2.5), np.percentile(rs, 97.5)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True)
    p.add_argument("--brain-tsv", default="notes/data/brain_behavior_subject_summary.tsv")
    p.add_argument("--tag", default="sequential")
    p.add_argument("--eu-tsv",
                   default="notes/data/brain_expected_uncertainty_spherical_dense.tsv",
                   help="Per-subject expected decoded SD (NPCr value, V1 orientation). "
                        "Default: the spherical-noise / dense-grid run; the older "
                        "brain_expected_uncertainty.tsv used the full empirical noise "
                        "covariance and the presented-value grid.")
    p.add_argument("--acuity", default="expected",
                   choices=["expected", "extent", "r", "precision"],
                   help="'extent' (default) is the amount of cortex with a "
                        "significant encoding model -- log tuned vertices, "
                        "vonmises for orientation and aprf for value. That is "
                        "the measure that has carried the signal in this "
                        "dataset before; decoding accuracy ('r') and decoded "
                        "precision are the alternatives.")
    a = p.parse_args()

    subs, pars = load_parameters(a.trace)
    brain = pd.read_csv(a.brain_tsv, sep="\t").set_index("subject")
    brain = brain.reindex(subs)
    keep = brain.notna().any(axis=1).values
    if not keep.all():
        print(f"  ! {(~keep).sum()} subjects missing from the brain table, dropped")

    if a.acuity == "expected":
        # Model-derived precision: the encoding model's own expected decoded
        # uncertainty, not trial-wise decoded SD, so it does not inherit BOLD
        # noise. Available for all 28 subjects with MRI, pilots included.
        eu = pd.read_csv(a.eu_tsv, sep="\t").set_index("subject")
        eu = eu.reindex(subs)
        neural = {"NPC value precision": 1.0 / eu.npcr_expected_sd_chf.values,
                  "V1 orientation precision": 1.0 / eu.v1_expected_sd_deg.values}
        neural["NPC value precision (cross)"] = neural["NPC value precision"]
        neural["V1 orientation precision (cross)"] = neural["V1 orientation precision"]
        nlab = "1 / expected SD"
    elif a.acuity == "extent":
        neural = {"Orientation encoding": brain["log_tuned_vonmises"].values,
                  "Value encoding": brain["log_tuned_aprf"].values,
                  "Value encoding (cross)": brain["log_tuned_aprf"].values,
                  "Orientation encoding (cross)": brain["log_tuned_vonmises"].values}
        nlab = "log tuned vertices"
    elif a.acuity == "r":
        neural = {"V1 orientation": brain["v1_ori_r"].values,
                  "NPC value": brain["npcr_val_r"].values,
                  "V1 value": brain["v1_val_r"].values,
                  "NPC orientation": brain["npcr_ori_r"].values}
        nlab = "decoding r"
    else:
        neural = {"V1 orientation": 1.0 / brain["v1_sd_mean"].values,
                  "NPC value": 1.0 / brain["npcr_sd_mean"].values,
                  "V1 value": np.full(len(brain), np.nan),
                  "NPC orientation": np.full(len(brain), np.nan)}
        nlab = "1 / decoded SD"

    # kappa_r is precision, sigma_rep is noise: flip sigma so "more" means
    # "better" on both axes, otherwise the predicted signs disagree for no
    # reason and the double dissociation is harder to read.
    behav = {"κ_r (perceptual precision)": (pars["kappa_r"], +1),
             "1/σ_rep (value precision)": (1.0 / np.maximum(pars["sigma_rep"], 1e-6), +1)}
    if a.acuity == "expected":
        pairs = [("κ_r (perceptual precision)", "V1 orientation precision", True),
                 ("1/σ_rep (value precision)", "NPC value precision", True),
                 ("κ_r (perceptual precision)", "NPC value precision (cross)", False),
                 ("1/σ_rep (value precision)", "V1 orientation precision (cross)", False)]
    elif a.acuity == "extent":
        pairs = [("κ_r (perceptual precision)", "Orientation encoding", True),
                 ("1/σ_rep (value precision)", "Value encoding", True),
                 ("κ_r (perceptual precision)", "Value encoding (cross)", False),
                 ("1/σ_rep (value precision)", "Orientation encoding (cross)", False)]
    else:
        pairs = [("κ_r (perceptual precision)", "V1 orientation", True),
                 ("1/σ_rep (value precision)", "NPC value", True),
                 ("κ_r (perceptual precision)", "NPC orientation", False),
                 ("1/σ_rep (value precision)", "V1 value", False)]

    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.2), constrained_layout=True)
    for ax, (bname, nname, matched) in zip(axes.flat, pairs):
        xd = behav[bname][0][:, keep]
        y = np.asarray(neural[nname], dtype=float)[keep]
        ok = np.isfinite(y)
        if ok.sum() < 5:
            ax.set_visible(False)
            continue
        med, lo, hi = draws_correlation(xd[:, ok], y[ok])
        rows.append(dict(behaviour=bname, neural=nname, matched=matched,
                         rho=med, lo=lo, hi=hi, n=int(ok.sum())))
        col = MATCH if matched else CROSS
        x = np.median(xd, axis=0)[ok]
        hdi = np.stack([az.hdi(xd[:, ok][:, i], hdi_prob=0.95) for i in range(ok.sum())])
        ax.errorbar(x, y[ok], xerr=np.abs(np.stack([x - hdi[:, 0], hdi[:, 1] - x])),
                    fmt="none", ecolor=col, elinewidth=0.7, alpha=0.35)
        ax.plot(x, y[ok], "o", ms=5, color=col, mec="white", mew=0.6)
        ax.set_xscale("log")
        ax.set_xlabel(bname); ax.set_ylabel(f"{nname}  ({nlab})")
        ax.set_title(f"ρ = {med:+.2f}  [{lo:+.2f}, {hi:+.2f}]"
                     + ("" if matched else "   (cross)"),
                     color="0.2" if matched else "0.5")
    fig.suptitle("Model parameters vs neural acuity  ·  matched pairs in blue, "
                 "crossed in grey\nSpearman ρ: posterior draws × subject bootstrap, "
                 "median and 95% interval",
                 fontsize=8, y=1.06, color="0.15")
    sns.despine(fig=fig, offset=4)
    out = Path(f"notes/figures/brain_behavior_{a.tag}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")

    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_csv(f"notes/data/brain_behavior_correlations_{a.tag}.tsv", sep="\t", index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
