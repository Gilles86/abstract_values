"""Bayesian model comparison for the hierarchical lambda model: value code (lambda = 0)
vs quantile code (lambda = 1) vs free lambda.

Uses the log marginal likelihoods written by ``hierarchical_lambda.py``.
Per participant: log Bayes factors (quantile vs value, free vs each).
Across participants: random-effects (over all three models, and over value vs
quantile alone: the *2 columns) Bayesian model selection (Stephan et al.,
2009, variational) with protected exceedance probabilities (Rigoux et al.,
2014). Group (fixed effects): everyone at 0, everyone at 1, one shared lambda,
and the hierarchical lambda_s ~ N(mu, tau).

    python -m abstract_values.encoding_models.lambda_model_comparison
    -> notes/data/hierlambda_lognormal_bms.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats

REPO = Path(__file__).resolve().parents[2]
MODELS = {"logev_lam0": "Value (λ = 0)", "logev_lam1": "Quantile (λ = 1)",
          "logev_free": "Free λ"}


def rfx_bms(L, alpha0=1., tol=1e-6, n_samples=200_000, seed=0):
    """Variational random-effects BMS. L: subjects x models log evidence.
    Returns Dirichlet alpha, expected frequencies, exceedance probabilities,
    protected exceedance probabilities and the Bayesian omnibus risk."""
    n, k = L.shape
    a0 = np.full(k, alpha0)
    a = a0.copy()
    for _ in range(10_000):
        lu = L + special.digamma(a) - special.digamma(a.sum())
        g = np.exp(lu - special.logsumexp(lu, 1, keepdims=True))
        a_new = a0 + g.sum(0)
        if np.abs(a_new - a).max() < tol:
            a = a_new
            break
        a = a_new
    r = np.random.default_rng(seed).dirichlet(a, n_samples)
    xp = np.bincount(r.argmax(1), minlength=k) / n_samples
    # Bayesian omnibus risk: evidence of the null (all frequencies equal) vs the RFX model
    lu = L + special.digamma(a) - special.digamma(a.sum())
    g = np.exp(lu - special.logsumexp(lu, 1, keepdims=True))
    elbo = ((g * (L + special.digamma(a) - special.digamma(a.sum()))).sum()
            - (g * np.log(g + 1e-300)).sum()
            + special.gammaln(a0.sum()) - special.gammaln(a0).sum()
            - special.gammaln(a.sum()) + special.gammaln(a).sum()
            + ((a0 - a) * (special.digamma(a) - special.digamma(a.sum()))).sum())
    f0 = special.logsumexp(L, 1).sum() - n * np.log(k)
    bor = 1 / (1 + np.exp(elbo - f0))
    pxp = xp * (1 - bor) + bor / k
    return a, a / a.sum(), xp, pxp, bor


def main(tag, out):
    sub = pd.read_csv(REPO / f"notes/data/hierlambda_{tag}_subjects.tsv", sep="\t",
                      dtype={"subject": str})
    grp = pd.read_csv(REPO / f"notes/data/hierlambda_{tag}_group.tsv", sep="\t").set_index("roi")
    rows = []
    for roi, d in sub.groupby("roi", sort=False):
        L = d[list(MODELS)].to_numpy()
        bf10 = d.logev_lam1 - d.logev_lam0
        a, ef, xp, pxp, bor = rfx_bms(L)
        _, ef2, xp2, pxp2, bor2 = rfx_bms(L[:, :2])                  # value vs quantile only
        g = grp.loc[roi]
        best_fixed = max(g.logev_all0, g.logev_all1, g.logev_shared)
        row = dict(roi=roi, n=len(d),
                   n_quantile_bf3=int((bf10 > np.log(3)).sum()),   # BF > 3 for quantile
                   n_value_bf3=int((bf10 < -np.log(3)).sum()),
                   n_quantile_bf20=int((bf10 > np.log(20)).sum()),
                   n_value_bf20=int((bf10 < -np.log(20)).sum()),
                   median_logbf_q_vs_v=bf10.median(),
                   sum_logbf_q_vs_v=bf10.sum(),
                   wilcoxon_p=stats.wilcoxon(bf10).pvalue,
                   bor=bor,
                   group_logbf_all1_vs_all0=g.logev_all1 - g.logev_all0,
                   group_logbf_shared_vs_all0=g.logev_shared - g.logev_all0,
                   group_logbf_hier_vs_bestfixed=g.logev_hier - best_fixed)
        for m, e, x, p in zip(MODELS, ef, xp, pxp):
            row.update({f"freq_{m[6:]}": e, f"xp_{m[6:]}": x, f"pxp_{m[6:]}": p})
        row.update(freq2_quantile=ef2[1], xp2_quantile=xp2[1], pxp2_quantile=pxp2[1],
                   bor2=bor2)
        rows.append(row)
    r = pd.DataFrame(rows)
    r.to_csv(out, sep="\t", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    print(r.round(3).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="lognormal")
    ap.add_argument("--out", default=str(REPO / "notes/data/hierlambda_lognormal_bms.tsv"))
    a = ap.parse_args()
    main(a.tag, a.out)
