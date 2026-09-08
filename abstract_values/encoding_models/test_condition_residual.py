#!/usr/bin/env python3
"""Generalise a value tuning curve ACROSS conditions, and let orientation compete.

The design's real leverage. Both mappings rise monotonically with orientation
(Spearman rho = 1, Pearson 0.93), so a transfer test passes under either
hypothesis. What separates them is that at a given orientation theta the two
conditions assign different CHF values -- 4.1 CHF apart on average.

So fit each voxel's value tuning f on ONE condition's runs, and predict the
OTHER condition's response to each orientation two ways:

    value hypothesis        f(v_other(theta))   the voxel tracks what it is worth
    orientation hypothesis  f(v_train(theta))   the voxel tracks theta, so it
                                                responds as it did in training

Both predictions come from the same curve fitted on the same data; they differ
only in where that curve is read out, so nothing about fit quality or flexibility
separates them. Whichever correlates better with the held-out condition's
responses wins that voxel. Averaged over both directions of generalisation.

This replaces an earlier within-condition version that fitted f on BOTH
conditions pooled and correlated the session difference with the difference the
curve predicted. That was circular: a pooled value fit has to reconcile two
different value assignments for the same orientation, so its per-session
residuals are structured by the mapping difference whatever the voxel codes.
It duly produced the same positive effect in V1 as in NPCr, which is the
signature of the artefact rather than of value coding in V1.

Voxel selection uses the training condition's own R2, so it never sees the
tested data.

Output: one TSV per subject with a row per selected voxel and direction.

    python -m abstract_values.encoding_models.test_condition_residual 03 \
        --roi NPCr --hemi None --out residual_sub-03.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import LogGaussianPRF
from braincoder.optimize import WeightFitter
from braincoder.utils import get_rsq

from abstract_values.encoding_models.decode_value import make_value_basis_parameters
from abstract_values.encoding_models.ridge_alpha import (
    DEFAULT_RIDGE_ALPHA, enforce_default_alpha)
from abstract_values.utils.data import Subject, BIDS_FOLDER


def get_paradigm(sub, sessions):
    """One row per gabor trial: value (CHF), orientation (deg), session, run."""
    rows = []
    for session in sessions:
        cond = sub.get_mapping(session)
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            ev = events.loc[run].reset_index().sort_values("onset")
            for _, r in ev[ev["event_type"] == "gabor"].iterrows():
                rows.append({"session": session, "run": run,
                             "condition": cond,
                             "x": np.float32(r["value"]),
                             "orientation": np.float32(r["orientation"])})
    return pd.DataFrame(rows)


def mapping_table(paradigm):
    """orientation -> value in each condition, from the trials themselves."""
    t = (paradigm.groupby(["condition", "orientation"])["x"].first()
                 .unstack("condition"))
    if t.isna().any().any():
        raise SystemExit("An orientation is missing from one condition.")
    return t


def generalise(data, paradigm, train_cond, basis_pars, model, alpha, n_voxels):
    """Fit on ``train_cond``, predict the other condition's orientation profile."""
    test_cond = "inverse_cdf" if train_cond == "cdf" else "cdf"
    tr = (paradigm["condition"] == train_cond).to_numpy()
    train_d, train_p = data[tr], paradigm[tr].reset_index(drop=True)

    weights = WeightFitter(model, basis_pars, train_d,
                           train_p[["x"]]).fit(alpha=alpha)
    pred = pd.DataFrame(model.basis_predictions(train_p[["x"]], basis_pars)
                        @ weights.values,
                        index=train_d.index, columns=train_d.columns)
    r2 = get_rsq(train_d, pred)
    sel = r2.sort_values(ascending=False).index[:n_voxels]

    m = mapping_table(paradigm)                       # orientation x condition
    grid = pd.DataFrame({"x": np.concatenate(
        [m[test_cond].values, m[train_cond].values]).astype(np.float32)})
    resp = (model.basis_predictions(grid, basis_pars) @ weights[sel].values)
    n_ori = len(m)
    pred_value = resp[:n_ori]        # read the curve at the NEW values
    pred_orientation = resp[n_ori:]  # read it where training put this orientation

    te = ~tr
    test_d, test_p = data.loc[te, sel], paradigm[te]
    observed = (test_d.groupby(test_p["orientation"].values).mean()
                      .reindex(m.index).to_numpy())
    return sel, r2.loc[sel].to_numpy(), observed, pred_value, pred_orientation


def _corr_slope(observed, predicted):
    """Column-wise Pearson r and OLS slope of observed on predicted.

    Both are centred first, so the comparison is about the shape of the
    orientation profile, not about session-level offsets.
    """
    o = observed - np.nanmean(observed, axis=0)
    p = predicted - np.nanmean(predicted, axis=0)
    cov = np.nansum(o * p, axis=0)
    var_p = np.nansum(p ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(np.nansum(o ** 2, axis=0) * var_p)
        slope = cov / var_p
    return r, slope


def main(subject, roi="NPCr", hemi=None, n_voxels=100, n_basis=8,
         alpha=DEFAULT_RIDGE_ALPHA, smoothed=False, bids_folder=BIDS_FOLDER,
         fmriprep_deriv="fmriprep"):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)
    sessions = sorted(sub.get_sessions())
    if len(sessions) < 2:
        raise SystemExit(f"sub-{subject}: needs both sessions.")

    paradigm = get_paradigm(sub, sessions)
    betas = sub.get_single_trial_estimates(sessions, desc="gabor",
                                           smoothed=smoothed)
    masker = NiftiMasker(mask_img=sub.get_roi_mask(roi=roi, hemi=hemi),
                         target_affine=betas.affine,
                         target_shape=betas.shape[:3]).fit()
    data = pd.DataFrame(masker.transform(betas).astype(np.float32))
    assert len(data) == len(paradigm), "beta/paradigm mismatch"
    print(f"sub-{subject}  {data.shape[1]} voxels in {roi}  "
          f"{len(paradigm)} trials  alpha={alpha}")

    vmin, vmax = float(paradigm["x"].min()), float(paradigm["x"].max())
    basis_pars = make_value_basis_parameters(n_basis, vmin, vmax)
    model = LogGaussianPRF(parameterisation="mode_fwhm_natural")

    rows = []
    for train_cond in ("cdf", "inverse_cdf"):
        sel, r2, observed, p_val, p_ori = generalise(
            data, paradigm, train_cond, basis_pars, model, alpha, n_voxels)
        r_val, _ = _corr_slope(observed, p_val)
        r_ori, _ = _corr_slope(observed, p_ori)
        rows.append(pd.DataFrame({
            "subject": subject, "roi": roi, "train_condition": train_cond,
            "voxel": sel, "train_r2": r2,
            "r_value": r_val, "r_orientation": r_ori,
            "r_advantage": r_val - r_ori,
            "predicted_gap": np.abs(p_val - p_ori).mean(axis=0)}))
        print(f"  train on {train_cond:12s}: {len(sel)} voxels  "
              f"r(value)={np.nanmean(r_val):+.4f}  "
              f"r(orientation)={np.nanmean(r_ori):+.4f}  "
              f"advantage={np.nanmean(r_val - r_ori):+.4f}")
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subject")
    p.add_argument("--roi", default="NPCr")
    p.add_argument("--hemi", default="None")
    p.add_argument("--n-voxels", type=int, default=100)
    p.add_argument("--n-basis", type=int, default=8)
    p.add_argument("--weight-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    p.add_argument("--allow-nondefault-alpha", action="store_true")
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--fmriprep-deriv", default="fmriprep")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    enforce_default_alpha(a.weight_alpha, a.allow_nondefault_alpha,
                          "condition-residual tuning")
    df = main(a.subject, roi=a.roi, hemi=None if a.hemi == "None" else a.hemi,
              n_voxels=a.n_voxels, n_basis=a.n_basis, alpha=a.weight_alpha,
              smoothed=a.smoothed, bids_folder=a.bids_folder,
              fmriprep_deriv=a.fmriprep_deriv)
    df.to_csv(a.out, sep="\t", index=False)
    print(f"wrote {a.out}")
