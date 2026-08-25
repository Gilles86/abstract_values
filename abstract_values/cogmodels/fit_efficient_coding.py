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

MODELS = ("perception", "valuation", "sequential")


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


def get_paradigm(subjects=None, paradigm_tsv=None):
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
    df = df.dropna(subset=["response", "orientation"])

    p = df[["subject", "orientation", "response", "mapping"]].copy()
    p = p.set_index(["subject", p.groupby("subject").cumcount()])
    p.index.names = ["subject", "trial"]

    print(f"{p.index.get_level_values('subject').nunique()} subjects, {len(p)} trials")
    for m, n in p["mapping"].value_counts().items():
        print(f"  {m}: {n} trials")
    return p


def make_model(paradigm, model_name, grid_resolution):
    if model_name == "perception":
        from bauer.efficient_coding import EfficientPerceptionModel
        return EfficientPerceptionModel(paradigm, grid_resolution=grid_resolution,
                                        perceptual_prior="long_term")
    if model_name == "valuation":
        from bauer.efficient_coding import EfficientValuationModel
        return EfficientValuationModel(paradigm, grid_resolution=grid_resolution)
    if model_name == "sequential":
        from bauer.efficient_coding import SequentialEfficientCodingModel
        return SequentialEfficientCodingModel(paradigm, grid_resolution=grid_resolution,
                                              perceptual_prior="long_term")
    raise ValueError(f"Unknown model: {model_name}")


def subject_parameters(idata, paradigm, model_name):
    """Per-subject posterior means of the model's free parameters."""
    import arviz as az

    subs = list(paradigm.index.get_level_values("subject").unique())
    post = idata.posterior
    rows = {}
    for par in ("kappa_r", "sigma_rep"):
        cands = [v for v in post.data_vars if v == par or v.startswith(f"{par}_subject")]
        if not cands:
            continue
        v = post[cands[0]]
        arr = v.mean(dim=("chain", "draw")).values.ravel()
        if arr.size == len(subs):
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
    p.add_argument("--model", default="sequential", choices=MODELS)
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--grid-resolution", type=int, default=31)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--target-accept", type=float, default=0.9)
    p.add_argument("--nuts-sampler", default="numpyro",
                   choices=["pymc", "numpyro", "nutpie"])
    p.add_argument("--paradigm-tsv", default=None,
                   help="Read the trial table from here instead of importing "
                        "the abstract_values stack (see --write-paradigm).")
    p.add_argument("--write-paradigm", default=None,
                   help="Only dump the trial table to this path and exit.")
    p.add_argument("--out-dir", default="derivatives/cogmodels")
    a = p.parse_args()

    if a.write_paradigm:
        write_paradigm_tsv(a.write_paradigm, a.subjects)
        return

    import arviz as az

    paradigm = get_paradigm(a.subjects, paradigm_tsv=a.paradigm_tsv)
    model = make_model(paradigm, a.model, a.grid_resolution)

    print(f"\nBuilding {a.model} model (grid={a.grid_resolution})…")
    model.build_estimation_model(hierarchical=True)

    print(f"Sampling: {a.chains} chains x {a.draws} draws (tune {a.tune}), "
          f"sampler={a.nuts_sampler}")
    idata = model.sample(draws=a.draws, tune=a.tune, chains=a.chains,
                         target_accept=a.target_accept,
                         nuts_sampler=a.nuts_sampler)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    nc = out / f"efficient_coding_{a.model}_trace.nc"
    idata.to_netcdf(str(nc))
    print(f"\nWrote {nc}")

    summ = az.summary(idata, var_names=["~_log", "~p_"], filter_vars="regex")
    print(summ.head(30).to_string())
    bad = summ[summ["r_hat"] > 1.01]
    print(f"\n{len(bad)} parameters with r_hat > 1.01"
          + (f":\n{bad.head(10).to_string()}" if len(bad) else ""))

    pars = subject_parameters(idata, paradigm, a.model)
    tsv = out / f"efficient_coding_{a.model}_subject_params.tsv"
    pars.to_csv(tsv, sep="\t")
    print(f"Wrote {tsv}\n{pars.to_string()}")


if __name__ == "__main__":
    main()
