"""Pointwise log-likelihood and LOO for the efficient-coding fits.

bauer attaches its likelihood with ``pm.Potential``, so ``model.observed_RVs``
is empty and neither pymc's samplers nor arviz can produce a log-likelihood
group on their own -- asking the numpyro sampler for one crashes it outright,
after sampling. The per-trial log-probability is available though: it is exactly
what ``build_likelihood`` sums into the Potential. This rebuilds that vector
outside the estimation model, evaluates it at each posterior draw, and hands
arviz a proper ``log_likelihood`` group so ``az.loo`` / ``az.compare`` work.

Usage:
    python -m abstract_values.cogmodels.loo_efficient_coding \
        --trace-dir <dir> --paradigm-tsv notes/data/..._paradigm.tsv \
        --out notes/data/efficient_coding_loo.tsv
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

PARAMS = {"perception": ["kappa_r"], "valuation": ["sigma_rep"],
          "sequential": ["kappa_r", "sigma_rep"]}


def pointwise_loglik(model, paradigm, idata, params, n_draws=400, seed=1):
    """(n_draws, n_trials) log-likelihood, evaluated on posterior draws."""
    from abstract_values.cogmodels.ppc_efficient_coding import posterior_subject_draws

    subjects = list(paradigm.index.get_level_values("subject").unique())
    model._setup_grids(paradigm)
    with pm.Model() as g:
        par_ = model._get_paradigm(paradigm=paradigm)
        model.set_paradigm(par_)
        holders = {p: pm.Data(p, np.zeros(len(subjects))) for p in params}
        mi = model.get_model_inputs(holders)
        ll = model._get_response_log_likelihood(mi)
        pm.Deterministic("pointwise_ll", ll)

    rows = []
    for pars in posterior_subject_draws(idata, subjects, params, n_draws, seed=seed):
        for p in params:
            pm.set_data({p: pars[p].values}, model=g)
        rows.append(np.asarray(g["pointwise_ll"].eval()))
    return np.stack(rows)


def attach_loo(idata, ll):
    """Give arviz a log_likelihood group shaped (chain, draw, trial)."""
    import xarray as xr
    n = ll.shape[0]
    da = xr.DataArray(ll[None, :, :], dims=("chain", "draw", "trial"),
                      coords={"chain": [0], "draw": np.arange(n),
                              "trial": np.arange(ll.shape[1])})
    idata.add_groups({"log_likelihood": xr.Dataset({"ll": da})})
    return idata


def main():
    import arviz as az
    from abstract_values.cogmodels.fit_efficient_coding import make_model, get_paradigm

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--paradigm-tsv", required=True)
    p.add_argument("--grid-resolution", type=int, default=101)
    p.add_argument("--lapse-rate", type=float, default=0.025)
    p.add_argument("--n-draws", type=int, default=300)
    p.add_argument("--out", default="notes/data/efficient_coding_loo.tsv")
    a = p.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(a.trace_dir, "efficient_coding_*_trace.nc"))):
        base = os.path.basename(f).replace("efficient_coding_", "").replace("_trace.nc", "")
        prior = "uniform" if "prior-uniform" in base else "long_term"
        model_name = base.replace("_prior-uniform", "")
        if model_name not in PARAMS:
            continue
        par = get_paradigm(paradigm_tsv=a.paradigm_tsv)
        model = make_model(par, model_name, a.grid_resolution,
                           lapse_rate=a.lapse_rate, perceptual_prior=prior)
        idata = az.from_netcdf(f)
        ll = pointwise_loglik(model, par, idata, PARAMS[model_name], a.n_draws)
        loo = az.loo(attach_loo(idata, ll), pointwise=True)
        rows.append(dict(model=model_name, prior=prior,
                         elpd_loo=float(loo.elpd_loo), se=float(loo.se),
                         p_loo=float(loo.p_loo), n_trials=ll.shape[1]))
        print(f"  {model_name:11s} {prior:10s} elpd_loo={loo.elpd_loo:10.2f} "
              f"+-{loo.se:5.2f}  p_loo={loo.p_loo:5.2f}")
    out = pd.DataFrame(rows).sort_values("elpd_loo", ascending=False)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, sep="\t", index=False)
    print(f"\nWrote {a.out}\n{out.to_string(index=False)}")


if __name__ == "__main__":
    main()
