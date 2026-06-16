#!/usr/bin/env python3
"""
Out-of-sample VALUE decoding tiebreaker: single pRF vs weighted basis,
joint vs separate condition handling, on the model-neutral union voxel set.

For each (model family x condition), leave-one-run-out:
  1. fit the encoding model on the training runs (single: grid+Adam
     log-Gaussian; weighted: OLS basis weights),
  2. fit a spherical Student-t noise model (ResidualFitter),
  3. P(data | value) over a value grid (get_stimulus_pdf) -> posterior
     mean = decoded value.
Reports mean abs error (CHF) and Pearson r (true vs decoded).

Voxels = the UNION FDR set (passes aprf OR aprf-weighted whole-brain FDR),
read from the `select_value_voxels_fdr` output, so the SAME model-neutral
voxels are used for every model -- no selection bias toward either family.
'shift' is omitted here: the encoding sweep showed joint ~ shift ~ separate,
and the session-conditioned likelihood complicates decoding; joint vs
separate is the decisive 2x2.

Output
------
  derivatives/experiments/npc_value_sweep/sub-<S>/func/
    sub-<S>_..._desc-decodesummary{_smoothed}.tsv   (one row per model x cond)
    sub-<S>_..._desc-decodetrials{_smoothed}.tsv     (per-trial true/decoded)

Usage
-----
  python -m abstract_values.encoding_models.decode_value_modelcomp 03
  python -m abstract_values.encoding_models.decode_value_modelcomp 03 \
      --n-basis 8 --fwhm 6 --noise-iter 1000
"""
from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import LogGaussianPRF
from braincoder.optimize import WeightFitter, ResidualFitter

from abstract_values.utils.data import Subject, BIDS_FOLDER
from abstract_values.encoding_models.model_specs import get_spec
from abstract_values.encoding_models.fit_pipeline import fit_one_model
from abstract_values.encoding_models.sweep_npc_value import get_value_paradigm

MODELS = ("single", "weighted")
CONDS = ("joint", "separate")
N_GRID = 60                       # value posterior grid points


def _union_voxels(bids_folder, subject, smooth_label):
    p = (bids_folder / "derivatives" / "experiments" / "npc_value_sweep"
         / f"sub-{subject}" / "func"
         / f"sub-{subject}_task-abstractvalue_mask-NPCr"
           f"_desc-voxelselect{smooth_label}.tsv")
    if not p.exists():
        raise SystemExit(f"Missing voxel-selection TSV {p.name} — run "
                         "select_value_voxels_fdr first.")
    df = pd.read_csv(p, sep="\t")
    return df.loc[df["in_union"] == 1, "voxel"].to_numpy()


def _decode(data, paradigm, model_kind, cond, value_min, value_max,
            n_iter, n_basis, fwhm, spherical=True, noise_iter=1000):
    """Leave-one-run-out value decoding; returns (true, decoded) arrays."""
    grid = np.linspace(value_min, value_max, N_GRID).astype(np.float32)
    grid_df = pd.DataFrame({"x": grid})
    s = paradigm.index.get_level_values("session")
    r = paradigm.index.get_level_values("run")
    folds = sorted(set(zip(s, r)))

    if model_kind == "weighted":
        modes = np.linspace(value_min, value_max, n_basis).astype(np.float32)
        basis_pars = pd.DataFrame({
            "mode": modes, "fwhm": np.full(n_basis, fwhm, np.float32),
            "amplitude": np.ones(n_basis, np.float32),
            "baseline": np.zeros(n_basis, np.float32)})

    trues, decodeds = [], []
    for ts, tr in folds:
        test_mask = (s == ts) & (r == tr)
        train_mask = ((~test_mask) & (s == ts)) if cond == "separate" \
            else ~test_mask
        train_data = data.loc[train_mask].reset_index(drop=True)
        train_par = paradigm.loc[train_mask, ["x"]].reset_index(drop=True)
        test_data = data.loc[test_mask].reset_index(drop=True)
        test_x = paradigm.loc[test_mask, "x"].to_numpy()

        if model_kind == "single":
            spec = get_spec("standard")
            pars, _ = fit_one_model(spec, train_data, train_par,
                                    value_min=value_min, value_max=value_max,
                                    n_iterations=n_iter, log_prefix="        ")
            model = spec.cls(**spec.cls_kwargs)
            resid = ResidualFitter(model, train_data, train_par, parameters=pars)
            omega, dof = resid.fit(init_sigma2=0.1, init_dof=10.0, method="t",
                                   learning_rate=0.05, spherical=spherical,
                                   max_n_iterations=noise_iter)
            pdf = model.get_stimulus_pdf(test_data, grid, parameters=pars,
                                         omega=omega, dof=dof, normalize=True)
        else:
            model = LogGaussianPRF(parameterisation="mode_fwhm_natural")
            w = WeightFitter(model, basis_pars, train_data, train_par).fit()
            resid = ResidualFitter(model, train_data, train_par,
                                   parameters=basis_pars, weights=w)
            omega, dof = resid.fit(init_sigma2=0.1, init_dof=10.0, method="t",
                                   learning_rate=0.05, spherical=spherical,
                                   max_n_iterations=noise_iter)
            pdf = model.get_stimulus_pdf(test_data, grid, parameters=basis_pars,
                                         weights=w, omega=omega, dof=dof,
                                         normalize=True)
        p = np.asarray(pdf.values, dtype=np.float64)
        p = p / p.sum(axis=1, keepdims=True)
        decodeds.append(p @ grid)               # posterior mean = E[value]
        trues.append(test_x)
    return np.concatenate(trues), np.concatenate(decodeds)


def run_one(subject, n_basis=8, fwhm=6.0, n_iter=500, noise_iter=1000,
            spherical=True, smoothed=False, bids_folder=BIDS_FOLDER):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    smooth_label = "_smoothed" if smoothed else ""

    mask_img = sub.get_roi_mask("NPCr", hemi=None)
    betas_img = sub.get_single_trial_estimates(sessions, desc="gabor",
                                               smoothed=smoothed)
    masker = NiftiMasker(mask_img=mask_img, target_affine=betas_img.affine,
                         target_shape=betas_img.shape[:3]).fit()
    paradigm = get_value_paradigm(sub, sessions)
    data = pd.DataFrame(masker.transform(betas_img).astype(np.float32),
                        index=paradigm.index)
    value_min, value_max = float(paradigm["x"].min()), float(paradigm["x"].max())

    union = _union_voxels(bids_folder, subject, smooth_label)
    data = data.iloc[:, union]
    print(f"sub-{subject}  {len(union)} union voxels · weighted k={n_basis} "
          f"fwhm={fwhm} · spherical={spherical}")

    out_dir = (bids_folder / "derivatives" / "experiments"
               / "npc_value_sweep" / f"sub-{subject}" / "func")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"sub-{subject}_task-abstractvalue_mask-NPCr"
    p_sum = out_dir / f"{base}_desc-decodesummary{smooth_label}.tsv"
    p_trial = out_dir / f"{base}_desc-decodetrials{smooth_label}.tsv"

    with ExitStack() as stack:
        f_sum = stack.enter_context(open(p_sum, "w", newline=""))
        f_tr = stack.enter_context(open(p_trial, "w", newline=""))
        w_sum = csv.writer(f_sum, delimiter="\t")
        w_tr = csv.writer(f_tr, delimiter="\t")
        w_sum.writerow(["subject", "model", "cond", "n_union",
                        "mae_chf", "pearson_r"])
        w_tr.writerow(["subject", "model", "cond", "true_chf", "decoded_chf"])
        f_sum.flush(); f_tr.flush()

        for model_kind in MODELS:
            for cond in CONDS:
                true, dec = _decode(data, paradigm, model_kind, cond,
                                    value_min, value_max, n_iter, n_basis, fwhm,
                                    spherical=spherical, noise_iter=noise_iter)
                mae = float(np.mean(np.abs(dec - true)))
                r = float(np.corrcoef(true, dec)[0, 1])
                w_sum.writerow([subject, model_kind, cond, len(union),
                                f"{mae:.4f}", f"{r:.4f}"])
                for t, d in zip(true, dec):
                    w_tr.writerow([subject, model_kind, cond,
                                   f"{t:.2f}", f"{d:.2f}"])
                f_sum.flush(); f_tr.flush()
                print(f"  {model_kind:<8} {cond:<8} "
                      f"MAE={mae:.2f} CHF  r={r:+.3f}", flush=True)
    print(f"  wrote {p_sum.name}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subject")
    p.add_argument("--n-basis", type=int, default=8)
    p.add_argument("--fwhm", type=float, default=6.0)
    p.add_argument("--n-iter", type=int, default=500)
    p.add_argument("--noise-iter", type=int, default=1000)
    p.add_argument("--full-noise", action="store_true")
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    args = p.parse_args()
    run_one(args.subject, n_basis=args.n_basis, fwhm=args.fwhm,
            n_iter=args.n_iter, noise_iter=args.noise_iter,
            spherical=not args.full_noise, smoothed=args.smoothed,
            bids_folder=args.bids_folder)


if __name__ == "__main__":
    main()
