#!/usr/bin/env python3
"""A6 step 1: per-cell predictive bid distributions from the motor-noise cogmodel.

The motor stage in bauer.efficient_coding is a convolution of the response
distribution with a kernel of SD ``sigma_motor`` (Student-t-like when
``motor_tail`` > 0), applied to the whole (subject, mapping, orientation, value)
table: it is the SAME for every trial of a subject, whatever the bid, so the
model has no distance-dependent (Fitts-like) motor noise. That makes it the
right null for the A6 distance-variance test.

For every subject x mapping x orientation cell we store, averaged over
``--n-draws`` posterior draws (the predictive distribution is the mixture over
draws, so mean/variance are computed from the mixture):

    pred_mean_full, pred_sd_full, pred_madsd_full   motor noise as fitted
    pred_mean_nomotor, pred_sd_nomotor, ...          sigma_motor -> 1e-3 (off)
    pred_absdev_*                                    E|bid - pred_mean|

The model is rebuilt from the flags stamped in the trace attrs, and the
grand-mean predicted bid is checked against the observed grand mean.

    ~/mambaforge/envs/bauer/bin/python model_cell_predictions.py --n-draws 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from abstract_values.cogmodels.fit_efficient_coding import make_model
from abstract_values.cogmodels.ppc_efficient_coding import (load_paradigm,
                                                           posterior_subject_draws,
                                                           subset_paradigm_to_trace)

REPO = Path(__file__).resolve().parents[3]
TRACE = Path("/data/ds-abstractvalue/derivatives/cogmodels_n35/"
             "efficient_coding_categorical_freeprior_grid161_trunc_hn_motor_lapse0_"
             "total-share_erf_shrink_mtail_trace.nc")
PARADIGM = REPO / "notes/data/efficient_coding_paradigm_n35.tsv"
OUT = Path("/data/ds-abstractvalue/derivatives/followup_2026-10/a6_m1_motor_noise")
PARAMS = ["kappa_r", "sigma_rep", "prior_weight", "mapping_shrinkage", "sigma_motor",
          "motor_tail"]


def trial_dist(model, paradigm, pars):
    """(n_trials, R) predictive density on the response grid, as in Estimation.simulate."""
    import pymc as pm
    if isinstance(pars, pd.DataFrame):
        pars = pars.to_dict(orient="list")
    model._setup_grids(paradigm)
    with pm.Model() as m:
        model.set_paradigm(model._get_paradigm(paradigm=paradigm))
        for k, v in pars.items():
            pm.Data(k, v)
        mi = model.get_model_inputs(model.get_parameter_values())
        pm.Deterministic("trial_dist", model._compute_trial_distributions(mi))
    d = np.asarray(m["trial_dist"].eval())
    return d / d.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-draws", type=int, default=40)
    a = ap.parse_args()
    idata = az.from_netcdf(TRACE)
    at = idata.posterior.attrs
    assert at["fit_motor_noise"] == 1 and at["motor_tail"] == 1 and at["fit_lapse"] == 0
    par = subset_paradigm_to_trace(load_paradigm(PARADIGM), idata)
    model = make_model(par, at["model"], int(at["grid_resolution"]),
                       lapse_rate=float(at["lapse_rate"]),
                       perceptual_prior=at["perceptual_prior"],
                       fit_prior_weight=bool(at["fit_prior_weight"]),
                       cardinal_truncation=bool(at["cardinal_truncation"]),
                       fit_motor_noise=True, param=at["param"],
                       fit_mapping_shrinkage=bool(at["fit_mapping_shrinkage"]),
                       fit_mapping_offsets=False, fit_lapse=False,
                       perceptual_tail=bool(at["perceptual_tail"]), motor_tail=True)
    # one representative trial per cell is enough: every trial of a cell shares its
    # predictive distribution
    flat = par.reset_index()
    cells = flat.drop_duplicates(["subject", "mapping", "orientation"]).copy()
    cpar = cells.set_index(["subject", cells.groupby("subject").cumcount()])
    cpar.index.names = ["subject", "trial"]
    subjects = list(par.index.get_level_values("subject").unique())
    acc = {k: [] for k in ("full", "nomotor")}
    for i, p in enumerate(posterior_subject_draws(idata, subjects, PARAMS, a.n_draws)):
        acc["full"].append(trial_dist(model, cpar, p))
        q = p.copy()
        q["sigma_motor"] = 1e-3
        acc["nomotor"].append(trial_dist(model, cpar, q))
        print(f"draw {i + 1}/{a.n_draws}", flush=True)

    grid = np.asarray(model._get_response_grid())
    assert grid.size == acc["full"][0].shape[1]
    out = cells[["subject", "mapping", "orientation", "value"]].reset_index(drop=True)
    for k, ds in acc.items():
        mix = np.mean(ds, 0)                                    # mixture over draws
        mu = mix @ grid
        var = mix @ grid ** 2 - mu ** 2
        cdf = np.cumsum(mix, 1)
        med = grid[np.argmax(cdf >= .5, 1)]
        # robust SD of the mixture: 1.4826 x median |x - median|
        mad = []
        for j in range(len(mix)):
            dev = np.abs(grid - med[j])
            o = np.argsort(dev)
            c = np.cumsum(mix[j, o])
            mad.append(dev[o][np.argmax(c >= .5)])
        out[f"pred_mean_{k}"] = mu
        out[f"pred_sd_{k}"] = np.sqrt(var)
        out[f"pred_madsd_{k}"] = 1.4826 * np.asarray(mad)
        out[f"pred_absdev_{k}"] = (mix * np.abs(grid[None] - mu[:, None])).sum(1)
        out[f"pred_median_{k}"] = med
    # grand-mean gate
    obs = flat.groupby(["subject", "mapping", "orientation"]).response.mean()
    m = out.set_index(["subject", "mapping", "orientation"]).join(obs)
    print("observed vs predicted cell mean: r = %.3f, mean diff %.3f CHF"
          % (np.corrcoef(m.response, m.pred_mean_full)[0, 1],
             (m.response - m.pred_mean_full).mean()))
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "model_cell_predictions.tsv", sep="\t", index=False, float_format="%.5g")


if __name__ == "__main__":
    main()
