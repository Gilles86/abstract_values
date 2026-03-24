#!/usr/bin/env python3
"""
Fit a *weighted* abstract pRF to all data (no cross-validation).

A fixed population of N log-Gaussian pRFs is defined spanning the CHF value
range.  A linear combination of these basis pRFs is fitted per voxel using
WeightFitter (closed-form least squares).  Analogous to the Von Mises
orientation model, but for the abstract value dimension.

Basis parameters
----------------
  mode   : N equally-spaced values from value_min to value_max
  fwhm   : fixed (default 2× inter-basis spacing)
  amplitude / baseline : 1 / 0  (fixed; absorbed into fitted weights)

Output
------
  derivatives/encoding_models/aprf-weighted/sub-<subject>/[ses_label/]func/
    sub-<subject>[_ses]_task-abstractvalue_space-T1w_desc-weight_<k>_pe.nii.gz
    sub-<subject>[_ses]_task-abstractvalue_space-T1w_desc-r2_pe.nii.gz

Usage
-----
  python fit_aprf_weighted.py pil01 --sessions 1 2
  python fit_aprf_weighted.py pil01 --sessions 1 2 --n-basis 12
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


def get_value_paradigm(sub, sessions):
    """Return paradigm DataFrame with column 'x' = objective CHF value (float32)."""
    rows = []
    for session in sessions:
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            run_ev = events.loc[run].reset_index().sort_values('onset')
            for _, row in run_ev[run_ev['event_type'] == 'gabor'].iterrows():
                rows.append(float(row['value']))
    return pd.DataFrame({'x': np.array(rows, dtype=np.float32)})


def make_basis_parameters(n_basis, value_min, value_max, fwhm=None):
    """Return a DataFrame of fixed log-Gaussian basis pRF parameters."""
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
    sessions = sorted(sessions)

    ses_dir    = f'ses-{sessions[0]}' if len(sessions) == 1 else ''
    ses_entity = f'_ses-{sessions[0]}' if len(sessions) == 1 else ''
    print(f'sub-{subject}  {ses_dir or "all-sessions"}  '
          f'n_basis={n_basis}  [weighted aPRF]')

    # ── paradigm ──────────────────────────────────────────────────────────────
    paradigm  = get_value_paradigm(sub, sessions)
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
    data = pd.DataFrame(masker.transform(betas_img).astype(np.float32))
    print(f'  {data.shape[1]} voxels in mask')

    # ── fixed basis parameters ────────────────────────────────────────────────
    basis_pars = make_basis_parameters(n_basis, value_min, value_max, fwhm)
    eff_fwhm   = float(basis_pars['fwhm'].iloc[0])
    print(f'  basis fwhm = {eff_fwhm:.2f} CHF')
    model = LogGaussianPRF(parameterisation='mode_fwhm_natural')

    # ── fit weights (closed-form least squares) ───────────────────────────────
    print('  fitting weights...')
    weights = WeightFitter(model, basis_pars, data, paradigm).fit()

    # ── R² on training data ───────────────────────────────────────────────────
    basis_pred = model.basis_predictions(paradigm, basis_pars)
    pred = pd.DataFrame(basis_pred @ weights.values,
                        index=data.index, columns=data.columns)
    r2 = get_rsq(data, pred)
    print(f'  mean R² = {float(r2.mean()):.4f}')

    # ── output directory ──────────────────────────────────────────────────────
    smooth_label = '_smoothed' if smoothed else ''
    out_dir = (bids_folder / 'derivatives' / 'encoding_models'
               / 'aprf-weighted' / f'sub-{subject}')
    if ses_dir:
        out_dir = out_dir / ses_dir
    out_dir = out_dir / 'func'
    out_dir.mkdir(parents=True, exist_ok=True)

    fn = (f'sub-{subject}{ses_entity}_task-abstractvalue'
          f'_space-T1w_desc-{{desc}}{smooth_label}_pe.nii.gz')

    for k in range(n_basis):
        masker.inverse_transform(weights.iloc[k]).to_filename(
            str(out_dir / fn.format(desc=f'weight_{k}')))
    masker.inverse_transform(r2).to_filename(str(out_dir / fn.format(desc='r2')))
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
