"""Fit the Bedi et al. (2026) efficient-coding models to the abstract-values study.

The three architectures from the paper, all implemented in ``bauer``:

    perception  EfficientPerceptionModel        efficient coding + Bayesian
                                                decoding in ORIENTATION space
                                                only; valuation is the veridical
                                                v = G(theta_hat).  1 parameter:
                                                kappa_r.
    valuation   EfficientValuationModel         perception veridical; efficient
                                                coding + decoding in VALUE space.
                                                1 parameter: sigma_rep.
    sequential  SequentialEfficientCodingModel  both stages, perceptual
                                                uncertainty marginalised into
                                                the value stage.  2 parameters:
                                                kappa_r, sigma_rep.

bauer's mapping tables are the same 25-point orientation->CHF lookups this study
presented (our 23 trained orientations are that table minus the 0 deg and 180 deg
endpoints), so the paradigm needs no rescaling -- just orientation, response,
mapping per trial.

Fits hierarchically across subjects so the per-subject parameters are shrunk
sensibly, then writes the per-subject posterior means to a TSV for correlating
against the neural measures.

Usage (cluster):
    python -m abstract_values.cogmodels.fit_efficient_coding --model sequential \
        --subjects 03 ... 28 --out-dir derivatives/cogmodels
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ("perception", "valuation", "sequential", "categorical")


def write_paradigm_tsv(path, subjects=None):
    """Dump the trial table the fit needs.  Run this in an env that has the
    abstract_values stack; the fit itself then only needs bauer + pymc."""
    from abstract_values.behavior.data import get_all_behavioral_data

    df = get_all_behavioral_data()
    df = df[df["event_type"] == "feedback"].copy()
    df["response"] = pd.to_numeric(df["response"], errors="coerce")
    df = df.reset_index()
    if subjects is not None:
        df = df[df["subject"].isin({int(s) for s in subjects})]
    df = df.dropna(subset=["response", "orientation"])
    cols = ["subject", "orientation", "response", "mapping", "value"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(path, sep="\t", index=False)
    print(f"Wrote {path}  ({df['subject'].nunique()} subjects, {len(df)} trials)")
    return df[cols]


def get_paradigm(subjects=None, paradigm_tsv=None, condition=None):
    """Trial-level paradigm: orientation (deg), response (CHF), mapping."""
    if paradigm_tsv is not None:
        df = pd.read_csv(paradigm_tsv, sep="\t")
    else:
        from abstract_values.behavior.data import get_all_behavioral_data
        df = get_all_behavioral_data()
        df = df[df["event_type"] == "feedback"].copy()
        df["response"] = pd.to_numeric(df["response"], errors="coerce")
        df = df.reset_index()
    if subjects is not None:
        df = df[df["subject"].isin({int(s) for s in subjects})]
    if condition is not None:
        df = df[df["mapping"] == condition]
        if df.empty:
            raise SystemExit(f"No trials for mapping={condition!r}")
    df = df.dropna(subset=["response", "orientation"])

    p = df[["subject", "orientation", "response", "mapping"]].copy()
    p = p.set_index(["subject", p.groupby("subject").cumcount()])
    p.index.names = ["subject", "trial"]

    print(f"{p.index.get_level_values('subject').nunique()} subjects, {len(p)} trials")
    for m, n in p["mapping"].value_counts().items():
        print(f"  {m}: {n} trials")
    return p


def make_model(paradigm, model_name, grid_resolution, lapse_rate=0.01,
               perceptual_prior="long_term", fit_prior_weight=False,
               no_seam_crossing=False, prior_fourier_order=0):
    if model_name == "perception":
        from bauer.efficient_coding import EfficientPerceptionModel
        return EfficientPerceptionModel(paradigm, grid_resolution=grid_resolution,
                                        perceptual_prior=perceptual_prior,
                                        lapse_rate=lapse_rate,
                                        fit_prior_weight=fit_prior_weight,
                                        prior_fourier_order=prior_fourier_order,
                                        no_seam_crossing=no_seam_crossing)
    if model_name == "valuation":
        from bauer.efficient_coding import EfficientValuationModel
        return EfficientValuationModel(paradigm, grid_resolution=grid_resolution,
                                       lapse_rate=lapse_rate)
    if model_name in ("sequential", "categorical"):
        if model_name == "sequential":
            from bauer.efficient_coding import SequentialEfficientCodingModel as cls
        else:
            # Paper Fig. 6: hard three-category gate around the 90 deg cardinal.
            from bauer.efficient_coding import CategoricalSequentialModel as cls
        return cls(paradigm, grid_resolution=grid_resolution,
                   perceptual_prior=perceptual_prior,
                   lapse_rate=lapse_rate,
                   fit_prior_weight=fit_prior_weight,
                   prior_fourier_order=prior_fourier_order,
                   no_seam_crossing=no_seam_crossing)
    raise ValueError(f"Unknown model: {model_name}")


def subject_parameters(idata, paradigm, model_name):
    """Per-subject posterior means of the model's free parameters."""
    import arviz as az

    subs = list(paradigm.index.get_level_values("subject").unique())
    post = idata.posterior
    rows = {}
    fourier = [f"prior_{c}{k}" for k in range(1, 9) for c in "ab"]
    for par in ("kappa_r", "sigma_rep", "prior_weight", *fourier):
        cands = [v for v in post.data_vars if v == par or v.startswith(f"{par}_subject")]
        if not cands:
            continue
        v = post[cands[0]]
        arr = np.atleast_1d(v.mean(dim=("chain", "draw")).values).ravel()
        if arr.size == 1 and len(subs) == 1:
            rows[par] = pd.Series(arr, index=subs)
        elif arr.size == len(subs):
            rows[par] = pd.Series(arr, index=subs)
        else:
            print(f"  ! {cands[0]} has {arr.size} values for {len(subs)} subjects, skipping")
    out = pd.DataFrame(rows)
    out.index.name = "subject"
    out["model"] = model_name
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="sequential", choices=MODELS,
                   help="'categorical' is 'sequential' plus the paper's "
                        "cardinal category gate (no extra free parameters).")
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--condition", default=None, choices=["cdf", "inverse_cdf"],
                   help="Fit one mapping only. Bedi et al. is between-subject, "
                        "but this study is within-subject and their Table 4 shows "
                        "kappa_r/sigma_rep differ several-fold by condition, so a "
                        "single pooled value per subject describes neither session.")
    p.add_argument("--grid-resolution", type=int, default=101,
                   help="Paper uses 101 points over [2, 42] CHF. Below ~101 the\n                        likelihood is flat in kappa_r above ~20 and sigma_rep is\n                        biased low by 10-35%.")
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--target-accept", type=float, default=0.9)
    p.add_argument("--nuts-sampler", default="numpyro",
                   choices=["pymc", "numpyro", "nutpie"])
    p.add_argument("--chain-method", default="sequential",
                   choices=["sequential", "parallel", "vectorized"],
                   help="numpyro only. 'parallel'/'vectorized' hold every chain's "
                        "intermediates at once, multiplying peak memory by "
                        "--chains; the sequential model needs more than 192 GB "
                        "that way at grid 101.")
    p.add_argument("--paradigm-tsv", default=None,
                   help="Read the trial table from here instead of importing "
                        "the abstract_values stack (see --write-paradigm).")
    p.add_argument("--write-paradigm", default=None,
                   help="Only dump the trial table to this path and exit.")
    p.add_argument("--perceptual-prior", default="long_term",
                   choices=["long_term", "uniform"],
                   help="Orientation prior used for encoding/decoding. "
                        "'long_term' is the fixed 2-|sin| cardinal prior; "
                        "'uniform' is the short-term prior this experiment "
                        "actually imposed (orientations were sampled "
                        "uniformly). The paper fits both and finds uniform "
                        "better for the perception-only architecture.")
    p.add_argument("--fit-prior-weight", action="store_true",
                   help="Fit the peakedness of the orientation prior, "
                        "p(phi) ~ 1 - w|sin phi|. Nests uniform (w=0) and the "
                        "paper's long-term prior (w=0.5).")
    p.add_argument("--prior-fourier-order", type=int, default=0,
                   help="Fit the orientation prior as a circular Fourier "
                        "series, p(phi) ~ exp(sum_k a_k cos k phi + b_k sin k phi), "
                        "with K harmonics. k=1 is the horizontal-vs-vertical "
                        "asymmetry, k=2 the cardinal-vs-oblique term that "
                        "reproduces the paper's prior at a_2 ~ 0.31, k>=3 "
                        "refines further. Coefficients carry a 0.6/k shrinkage "
                        "prior, so higher harmonics must earn their amplitude. "
                        "Mutually exclusive with --fit-prior-weight.")
    p.add_argument("--no-seam-crossing", action="store_true",
                   help="Forbid perceptual confusion across the 0/180 deg seam, "
                        "where G jumps from 42 CHF back to 2 CHF.")
    p.add_argument("--lapse-rate", type=float, default=0.01,
                   help="Probability of a uniformly random bid. Without it a "
                        "single far-off response can dominate the likelihood. "
                        "The paper fixes 0.01 for its primary comparisons.")
    p.add_argument("--no-hierarchical", action="store_true",
                   help="Fit subjects independently. Required for a single "
                        "subject: with one subject the group SD is unidentified "
                        "and its HalfCauchy tail lets kappa_r run away.")
    p.add_argument("--out-dir", default="derivatives/cogmodels")
    a = p.parse_args()

    if a.write_paradigm:
        write_paradigm_tsv(a.write_paradigm, a.subjects)
        return

    # NOTE: do NOT ask the sampler for log_likelihood. bauer attaches the
    # likelihood with pm.Potential, so model.observed_RVs is empty and pymc's
    # JAX path dies in _get_log_likelihood with "'NoneType' object is not
    # iterable" -- after sampling, so the whole fit is lost. Pointwise
    # log-likelihood for LOO is computed separately (see loo_efficient_coding).
    import arviz as az

    paradigm = get_paradigm(a.subjects, paradigm_tsv=a.paradigm_tsv,
                            condition=a.condition)
    model = make_model(paradigm, a.model, a.grid_resolution,
                       lapse_rate=a.lapse_rate,
                       perceptual_prior=a.perceptual_prior,
                       fit_prior_weight=a.fit_prior_weight,
                       prior_fourier_order=a.prior_fourier_order,
                       no_seam_crossing=a.no_seam_crossing)

    print(f"\nBuilding {a.model} model (grid={a.grid_resolution}, "
          f"hierarchical={not a.no_hierarchical}, lapse={a.lapse_rate})…")
    model.build_estimation_model(hierarchical=not a.no_hierarchical)

    print(f"Sampling: {a.chains} chains x {a.draws} draws (tune {a.tune}), "
          f"sampler={a.nuts_sampler}")
    sampler_kwargs = ({"chain_method": a.chain_method}
                      if a.nuts_sampler == "numpyro" else {})
    idata = model.sample(draws=a.draws, tune=a.tune, chains=a.chains,
                         target_accept=a.target_accept,
                         nuts_sampler=a.nuts_sampler,
                         nuts_sampler_kwargs=sampler_kwargs)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model}" + (f"_{a.condition}" if a.condition else "")
    if a.perceptual_prior != "long_term":
        tag += f"_prior-{a.perceptual_prior}"
    if a.fit_prior_weight:
        tag += "_freeprior"
    if a.prior_fourier_order:
        tag += f"_fourier{a.prior_fourier_order}"
    # The grid is part of the model, not just its accuracy (it sets the kappa_r
    # ceiling), so a grid sweep must not overwrite the reference fit.
    if a.grid_resolution != 101:
        tag += f"_grid{a.grid_resolution}"
    if a.no_seam_crossing:
        tag += "_noseam"
    nc = out / f"efficient_coding_{tag}_trace.nc"
    idata.to_netcdf(str(nc))
    print(f"\nWrote {nc}")

    summ = az.summary(idata, var_names=["~_log", "~p_"], filter_vars="regex")
    print(summ.head(30).to_string())
    bad = summ[summ["r_hat"] > 1.01]
    print(f"\n{len(bad)} parameters with r_hat > 1.01"
          + (f":\n{bad.head(10).to_string()}" if len(bad) else ""))

    pars = subject_parameters(idata, paradigm, a.model)
    pars["condition"] = a.condition or "both"
    pars["perceptual_prior"] = a.perceptual_prior
    pars["lapse_rate"] = a.lapse_rate
    tsv = out / f"efficient_coding_{tag}_subject_params.tsv"
    pars.to_csv(tsv, sep="\t")
    print(f"Wrote {tsv}\n{pars.to_string()}")


if __name__ == "__main__":
    main()
