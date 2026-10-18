"""Posterior predictive checks + PSIS-LOO for a set of efficient-coding fits, side by side.

For each trace (``--fit label=path:flags``) simulates ``--n-draws`` full
datasets from the posterior (same trials as the data), and evaluates the
pointwise log-likelihood on ``--n-loo`` posterior draws. Writes, into --out-dir:

  ppc_compare_group.tsv     per fit x draw x mapping x orientation: mean bid
                            error and the within-subject bid SD (averaged over
                            subjects), plus the observed rows (draw = -1)
  ppc_compare_subjects.tsv  per fit x subject x mapping x orientation: the
                            observed bid error / SD and quantiles of the
                            simulated ones
  ppc_compare_loo.tsv       az.compare table
  ppc_compare_loo_subj.tsv  ELPD per subject per fit (sum of pointwise elpd_loo)

Flags after the colon: ``prior_weight`` (fit had --fit-prior-weight), ``trunc``
(--cardinal-truncation), ``noseam``, ``fourierK``, ``shrink``
(--fit-mapping-shrinkage).

    python -m abstract_values.cogmodels.compare_ppc \\
        --paradigm-tsv notes/data/efficient_coding_paradigm_n35.tsv \\
        --fit "Fixed prior=<dir>/efficient_coding_categorical_..._trace.nc:trunc" \\
        --fit "Free prior=<dir>/efficient_coding_categorical_freeprior_..._trace.nc:trunc,prior_weight"
Run in the ``bauer`` env.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from abstract_values.cogmodels.fit_efficient_coding import make_model
from abstract_values.cogmodels.loo_efficient_coding import pointwise_loglik
from abstract_values.cogmodels.ppc_efficient_coding import (load_paradigm, run_ppc,
                                                           subset_paradigm_to_trace)

GRID = 161
LAPSE = 0.01
QS = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]


def parse_fit(s):
    label, rest = s.split("=", 1)
    path, _, flags = rest.partition(":")
    flags = set(f for f in flags.split(",") if f)
    K = next((int(f[7:]) for f in flags if f.startswith("fourier")), 0)
    kw = dict(model_name="categorical" if "categorical" in Path(path).name else "sequential",
              cardinal_truncation="trunc" in flags, fit_prior_weight="prior_weight" in flags,
              no_seam_crossing="noseam" in flags, prior_fourier_order=K, value_kernel="cell",
              fit_mapping_shrinkage="shrink" in flags)
    params = (["kappa_r", "sigma_rep"] + (["prior_weight"] if "prior_weight" in flags else [])
              + (["mapping_shrinkage"] if "shrink" in flags else [])
              + [f"prior_{c}{k}" for k in range(1, K + 1) for c in "ab"])
    return label, path, kw, params


def per_subject(df):
    d = df.assign(err=df.response - df.value)
    g = d.groupby(["subject", "mapping", "orientation"])
    return pd.DataFrame({"bias": g.err.mean(), "sd": g.response.std()})


def group(sub):
    return sub.groupby(["mapping", "orientation"]).mean()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paradigm-tsv", required=True)
    p.add_argument("--fit", action="append", required=True)
    p.add_argument("--n-draws", type=int, default=200)
    p.add_argument("--n-loo", type=int, default=400)
    p.add_argument("--out-dir", default="notes/data")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    grp_rows, sub_rows, loos, ll_subj = [], [], {}, []
    for spec in a.fit:
        label, path, kw, params = parse_fit(spec)
        print(f"== {label}: {path}  params={params}", flush=True)
        idata = az.from_netcdf(path)
        par = subset_paradigm_to_trace(load_paradigm(a.paradigm_tsv), idata)
        kw = dict(kw)
        model = make_model(par, kw.pop("model_name"), GRID, lapse_rate=LAPSE,
                           perceptual_prior="long_term", **kw)
        obs = per_subject(par.reset_index())
        g = group(obs).reset_index().assign(fit=label, draw=-1)
        grp_rows.append(g)
        sims = []
        for i, sim in enumerate(run_ppc(model, par, idata, params, a.n_draws)):
            s = per_subject(sim)
            sims.append(s.assign(draw=i))
            grp_rows.append(group(s).reset_index().assign(fit=label, draw=i))
        sims = pd.concat(sims)
        q = sims.groupby(["subject", "mapping", "orientation"])[["bias", "sd"]].quantile(QS)
        q = q.unstack(-1)
        q.columns = [f"{c}_{qq}" for c, qq in q.columns]
        sub_rows.append(q.join(obs).reset_index().assign(fit=label))

        ll = pointwise_loglik(model, par, idata, params, n_draws=a.n_loo)
        da = xr.DataArray(ll[None], dims=("chain", "draw", "trial"))
        idl = az.InferenceData(
            posterior=xr.Dataset({"x": xr.DataArray(np.zeros((1, ll.shape[0])),
                                                    dims=("chain", "draw"))}),
            log_likelihood=xr.Dataset({"ll": da}))
        loo = az.loo(idl, pointwise=True)
        loos[label] = loo
        subj = par.index.get_level_values("subject").to_numpy()
        ll_subj.append(pd.DataFrame({"subject": subj, "elpd": loo.loo_i.values,
                                     "pareto_k": loo.pareto_k.values})
                       .groupby("subject").agg(elpd=("elpd", "sum"),
                                               max_k=("pareto_k", "max"))
                       .reset_index().assign(fit=label))
        print(f"   elpd_loo {float(loo.elpd_loo):.1f} ± {float(loo.se):.1f}, "
              f"p_loo {float(loo.p_loo):.1f}, k>0.7 {(loo.pareto_k > 0.7).mean().item():.3f}",
              flush=True)

    pd.concat(grp_rows).to_csv(out / "ppc_compare_group.tsv", sep="\t", index=False)
    pd.concat(sub_rows).to_csv(out / "ppc_compare_subjects.tsv", sep="\t", index=False)
    pd.concat(ll_subj).to_csv(out / "ppc_compare_loo_subj.tsv", sep="\t", index=False)
    cmp = az.compare(loos, ic="loo").reset_index().rename(columns={"index": "fit"})
    cmp.to_csv(out / "ppc_compare_loo.tsv", sep="\t", index=False)
    print(cmp.to_string())


if __name__ == "__main__":
    main()
