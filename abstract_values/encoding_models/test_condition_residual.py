#!/usr/bin/env python3
"""Does the response to the SAME orientation change the way value predicts?

The design's real leverage. The two mappings both rise monotonically with
orientation (Spearman rho = 1, Pearson 0.93), so a transfer test passes under
either hypothesis. What separates value from orientation is the 13% where they
disagree: at a given orientation theta the two conditions assign different CHF
values, v_cdf(theta) and v_inv(theta), differing by 4.1 CHF on average.

So ask each voxel a question only a value code can answer:

    predicted  d(theta) = f(v_inv(theta)) - f(v_cdf(theta))
    observed   d(theta) = mean beta at theta in the inv session
                          - mean beta at theta in the cdf session

where f is the voxel's own value tuning curve. A voxel tuned to orientation
responds to theta the same way whatever it is worth, so its observed d is
unrelated to the predicted one. A voxel tuned to value follows f.

Two things keep this honest:

* Circularity. f is fitted on one half of the runs (odd/even within each
  session) and the observed difference is computed on the other half, then the
  halves are swapped and the two folds averaged. Voxel selection uses the
  training half's R2 only.
* A control the counterbalancing gives for free -- but only at the group
  level. The same contrast computed as (session 2 - session 1) rather than
  (inverse_cdf - cdf) carries every session-level nuisance (drift, attention,
  adaptation) and none of the condition effect. Within one subject it is just
  the condition contrast with a known sign, so its per-subject r is trivially
  +/- the condition r; what matters is that session order is counterbalanced
  across subjects, so the group mean of the control is ~0 while the group mean
  of the condition statistic is not. Read the control across subjects, never
  within one.

Per-voxel statistics are the correlation and the regression slope of observed
on predicted across the 23 presented orientations.

Output: one TSV per subject with a row per selected voxel.

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


def fold_statistics(data, paradigm, train_mask, basis_pars, model, alpha,
                    n_voxels):
    """Fit tuning on the training runs, test the difference on the rest."""
    train_d, train_p = data[train_mask], paradigm[train_mask]
    weights = WeightFitter(model, basis_pars, train_d,
                           train_p[["x"]].reset_index(drop=True)).fit(alpha=alpha)
    pred = pd.DataFrame(
        model.basis_predictions(train_p[["x"]].reset_index(drop=True), basis_pars)
        @ weights.values, index=train_d.index, columns=train_d.columns)
    r2 = get_rsq(train_d, pred)
    sel = r2.sort_values(ascending=False).index[:n_voxels]

    # predicted difference: the voxel's own tuning read at the two values the
    # conditions assign to each orientation.
    m = mapping_table(paradigm)
    grid = pd.DataFrame({"x": np.concatenate([m["cdf"].values,
                                              m["inverse_cdf"].values])
                         .astype(np.float32)})
    resp = pd.DataFrame(model.basis_predictions(grid, basis_pars)
                        @ weights[sel].values, columns=sel)
    n_ori = len(m)
    predicted = resp.iloc[n_ori:].to_numpy() - resp.iloc[:n_ori].to_numpy()

    # observed difference on the held-out runs, each session centred per voxel
    test_d, test_p = data.loc[~train_mask, sel], paradigm[~train_mask]
    centred = test_d - test_d.groupby(test_p["session"].values).transform("mean")

    def _diff(labels, hi, lo):
        g = centred.groupby([labels.values, test_p["orientation"].values]).mean()
        return (g.loc[hi].reindex(m.index).to_numpy()
                - g.loc[lo].reindex(m.index).to_numpy())

    observed = _diff(test_p["condition"], "inverse_cdf", "cdf")
    ses = sorted(test_p["session"].unique())
    control = _diff(test_p["session"], ses[-1], ses[0])
    return sel, r2.loc[sel].to_numpy(), predicted, observed, control


def _corr_slope(observed, predicted):
    """Column-wise Pearson r and OLS slope of observed on predicted."""
    o = observed - np.nanmean(observed, axis=0)
    p = predicted - np.nanmean(predicted, axis=0)
    denom_p = np.sqrt(np.nansum(p ** 2, axis=0))
    denom_o = np.sqrt(np.nansum(o ** 2, axis=0))
    cov = np.nansum(o * p, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / (denom_o * denom_p)
        slope = cov / (denom_p ** 2)
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

    # Two folds: odd runs train / even test, then swapped. Splitting by run
    # inside each session keeps both conditions in both halves.
    odd = (paradigm["run"] % 2 == 1).to_numpy()
    rows = []
    for fold, train_mask in enumerate((odd, ~odd), start=1):
        sel, r2, predicted, observed, control = fold_statistics(
            data, paradigm, train_mask, basis_pars, model, alpha, n_voxels)
        r, slope = _corr_slope(observed, predicted)
        r_ctrl, slope_ctrl = _corr_slope(control, predicted)
        rows.append(pd.DataFrame({
            "subject": subject, "roi": roi, "fold": fold, "voxel": sel,
            "train_r2": r2, "r": r, "slope": slope,
            "r_session_control": r_ctrl, "slope_session_control": slope_ctrl,
            "predicted_range": predicted.max(axis=0) - predicted.min(axis=0)}))
        print(f"  fold {fold}: {len(sel)} voxels, mean r = {np.nanmean(r):+.4f} "
              f"(session-aligned {np.nanmean(r_ctrl):+.4f}; sign-flipped copy "
              f"of the same contrast -- only its GROUP mean is a control)")
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
