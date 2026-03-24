#!/usr/bin/env python3
"""
Leave-one-run-out cross-validated fitting of a *weighted* abstract pRF.

Instead of fitting a single log-Gaussian per voxel (mode, fwhm, amplitude,
baseline), a fixed population of N log-Gaussian pRFs is defined spanning the
CHF value range.  A linear combination of these basis pRFs is then fitted per
voxel using least squares (WeightFitter) — analogous to the Von Mises
orientation model, but for the abstract value dimension.

Basis parameters
----------------
  mode   : N equally-spaced values from value_min to value_max
  fwhm   : fixed (default 2× inter-basis spacing)
  amplitude / baseline : 1 / 0  (fixed; absorbed into fitted weights)

Output
------
  derivatives/encoding_models/aprf-weighted.cv/sub-<subject>/[ses_label/]func/
    sub-<subject>[_ses]_task-abstractvalue_space-T1w_run-<r>_desc-cvr2_pe.nii.gz
    sub-<subject>[_ses]_task-abstractvalue_space-T1w_desc-cvr2_pe.nii.gz  (mean)

Usage
-----
  python fit_aprf_weighted_cv.py pil01 --sessions 1 2
  python fit_aprf_weighted_cv.py pil01 --sessions 1 2 --n-basis 12
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import LogGaussianPRF
from braincoder.optimize import WeightFitter
from braincoder.utils import get_rsq

from abstract_values.utils.data import Subject, BIDS_FOLDER


def get_value_paradigm_with_runs(sub, sessions):
    """Return paradigm DataFrame indexed by (session, run, trial_idx).

    Column 'x' = objective CHF value (float32).
    """
    rows = []
    for session in sessions:
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            run_ev = events.loc[run].reset_index().sort_values('onset')
            for _, row in run_ev[run_ev['event_type'] == 'gabor'].iterrows():
                rows.append({'session': session, 'run': run,
                             'x': float(row['value'])})
    df = pd.DataFrame(rows).astype({'x': np.float32})
    df.index = pd.MultiIndex.from_frame(
        df[['session', 'run']].assign(
            trial=df.groupby(['session', 'run']).cumcount()),
        names=['session', 'run', 'trial'])
    return df[['x']]


def make_basis_parameters(n_basis, value_min, value_max, fwhm=None):
    """Return a DataFrame of fixed log-Gaussian basis pRF parameters.

    Modes are equally spaced in CHF space from value_min to value_max.
    fwhm defaults to 2× the inter-basis spacing.
    """
    modes = np.linspace(value_min, value_max, n_basis).astype(np.float32)
    spacing = modes[1] - modes[0] if n_basis > 1 else (value_max - value_min)
    if fwhm is None:
        fwhm = float(2.0 * spacing)
    return pd.DataFrame({
        'mode':      modes,
        'fwhm':      np.full(n_basis, fwhm, dtype=np.float32),
        'amplitude': np.ones(n_basis,  dtype=np.float32),
        'baseline':  np.zeros(n_basis, dtype=np.float32),
    })


def main(subject, sessions=None, n_basis=8, fwhm=None, mask=None,
         bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep',
         smoothed=False):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder,
                  fmriprep_deriv=fmriprep_deriv)

    if sessions is None:
        sessions = sub.get_sessions()

    ses_dir    = f'ses-{sessions[0]}' if len(sessions) == 1 else ''
    ses_entity = f'_ses-{sessions[0]}' if len(sessions) == 1 else ''
    print(f'sub-{subject}  {ses_dir or "all-sessions"}  '
          f'n_basis={n_basis}  [weighted aPRF CV]')

    # ── paradigm ──────────────────────────────────────────────────────────────
    paradigm  = get_value_paradigm_with_runs(sub, sessions)
    value_min = float(paradigm['x'].min())
    value_max = float(paradigm['x'].max())
    print(f'  {len(paradigm)} trials  value range: {value_min:.1f}–{value_max:.1f} CHF')

    # ── betas ─────────────────────────────────────────────────────────────────
    betas_img = sub.get_single_trial_estimates(sessions, desc='gabor',
                                               smoothed=smoothed)
    assert betas_img.shape[3] == len(paradigm), (
        f'Beta count mismatch: {betas_img.shape[3]} vs {len(paradigm)}')

    # ── masker ────────────────────────────────────────────────────────────────
    if mask is None:
        mask = sub.get_brain_mask(sessions[0])
    masker = NiftiMasker(mask_img=mask,
                         target_affine=betas_img.affine,
                         target_shape=betas_img.shape[:3]).fit()
    data = pd.DataFrame(masker.transform(betas_img).astype(np.float32),
                        index=paradigm.index)
    print(f'  {data.shape[1]} voxels in mask')

    # ── fixed basis parameters (identical across folds) ───────────────────────
    basis_pars = make_basis_parameters(n_basis, value_min, value_max, fwhm)
    eff_fwhm   = float(basis_pars['fwhm'].iloc[0])
    print(f'  basis fwhm = {eff_fwhm:.2f} CHF')
    model = LogGaussianPRF(parameterisation='mode_fwhm_natural')

    # ── output directory ──────────────────────────────────────────────────────
    smooth_label = '_smoothed' if smoothed else ''
    out_dir = (bids_folder / 'derivatives' / 'encoding_models'
               / 'aprf-weighted.cv' / f'sub-{subject}')
    if ses_dir:
        out_dir = out_dir / ses_dir
    out_dir = out_dir / 'func'
    out_dir.mkdir(parents=True, exist_ok=True)

    fn_run  = (f'sub-{subject}_ses-{{ses}}_task-abstractvalue'
               f'_space-T1w_run-{{run}}_desc-cvr2{smooth_label}_pe.nii.gz')
    fn_mean = (f'sub-{subject}{ses_entity}_task-abstractvalue'
               f'_space-T1w_desc-cvr2{smooth_label}_pe.nii.gz')

    # ── leave-one-run-out CV ──────────────────────────────────────────────────
    folds = sorted(set(zip(
        paradigm.index.get_level_values('session'),
        paradigm.index.get_level_values('run'),
    )))
    all_cvr2 = []

    for test_session, test_run in folds:
        print(f'  fold ses-{test_session} run-{test_run}:')

        test_mask  = (paradigm.index.get_level_values('session') == test_session) & \
                     (paradigm.index.get_level_values('run')     == test_run)
        train_mask = ~test_mask

        train_paradigm = paradigm.loc[train_mask].reset_index(drop=True)[['x']]
        train_data     = data.loc[train_mask].reset_index(drop=True)
        test_paradigm  = paradigm.loc[test_mask].reset_index(drop=True)[['x']]
        test_data      = data.loc[test_mask].reset_index(drop=True)

        # Fit weights (closed-form least squares)
        weights = WeightFitter(model, basis_pars,
                               train_data, train_paradigm).fit()

        # Predict on held-out run
        basis_pred = model.basis_predictions(test_paradigm, basis_pars)
        test_pred  = pd.DataFrame(basis_pred @ weights.values,
                                  index=test_data.index,
                                  columns=test_data.columns)
        cv_r2 = get_rsq(test_data, test_pred)
        print(f'    mean CV R² = {float(cv_r2.mean()):.4f}')

        masker.inverse_transform(cv_r2).to_filename(
            str(out_dir / fn_run.format(ses=test_session, run=test_run)))
        all_cvr2.append(cv_r2)

    # ── mean CV R² ────────────────────────────────────────────────────────────
    mean_cvr2 = pd.concat(all_cvr2, axis=1).mean(axis=1)
    print(f'  mean CV R² (all folds) = {float(mean_cvr2.mean()):.4f}')
    masker.inverse_transform(mean_cvr2).to_filename(str(out_dir / fn_mean))
    print(f'  saved to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-'")
    parser.add_argument('--sessions', type=int, nargs='+', default=None)
    parser.add_argument('--n-basis', type=int, default=8,
                        help='Number of log-Gaussian basis pRFs (default: 8)')
    parser.add_argument('--fwhm', type=float, default=None,
                        help='FWHM in CHF of each basis pRF '
                             '(default: 2× inter-basis spacing)')
    parser.add_argument('--mask', default=None)
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep',
                        choices=['fmriprep', 'fmriprep-t2w'])
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()

    main(args.subject, sessions=args.sessions, n_basis=args.n_basis,
         fwhm=args.fwhm, mask=args.mask, bids_folder=args.bids_folder,
         fmriprep_deriv=args.fmriprep_deriv, smoothed=args.smoothed)
