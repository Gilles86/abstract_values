#!/usr/bin/env python3
"""
Leave-one-run-out cross-validated fitting of the abstract pRF encoding model.

Supports four model variants (same as fit_aprf.py):

  standard            : LogGaussianPRF (mode, fwhm, amplitude, baseline)
  session-shift       : SessionShiftedLogGaussianPRF (mode_1, mode_2, fwhm, amp, baseline)
  gaussian            : GaussianValuePRF — symmetric Gaussian, no rightward skew
  gauss-session-shift : SessionShiftedGaussianValuePRF — symmetric Gaussian with shift

Each fold fits on N-1 runs and evaluates on the held-out run; per-fold CV R²
NIfTIs and a mean image are written under
``derivatives/encoding_models/<out>/sub-<subject>[/<ses>]/func/``.

Output subdir per model
-----------------------
  standard             → aprf.cv
  session-shift        → aprf-shift.cv
  gaussian             → aprf-gauss.cv
  gauss-session-shift  → aprf-gauss-shift.cv

Usage
-----
  python fit_aprf_cv.py pil01 --sessions 1
  python fit_aprf_cv.py pil01 --sessions 1 2 --model session-shift
  python fit_aprf_cv.py pil01 --sessions 1 --model gaussian
  python fit_aprf_cv.py pil01 --sessions 1 2 --model gauss-session-shift --debug
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import LogGaussianPRF
from braincoder.optimize import ParameterFitter
from braincoder.utils import get_rsq

from abstract_values.encoding_models.models import (
    GaussianValuePRF,
    SessionShiftedGaussianValuePRF,
    SessionShiftedLogGaussianPRF,
)
from abstract_values.utils.data import Subject, BIDS_FOLDER


# ── model registry ──────────────────────────────────────────────────────────

_SHIFT_MODELS = {'session-shift', 'gauss-session-shift'}


def _build_model(model_type):
    if model_type == 'standard':
        return LogGaussianPRF(allow_neg_amplitudes=False,
                              parameterisation='mode_fwhm_natural')
    if model_type == 'session-shift':
        return SessionShiftedLogGaussianPRF(allow_neg_amplitudes=False)
    if model_type == 'gaussian':
        return GaussianValuePRF(allow_neg_amplitudes=False)
    if model_type == 'gauss-session-shift':
        return SessionShiftedGaussianValuePRF(allow_neg_amplitudes=False)
    raise ValueError(f'Unknown model_type: {model_type!r}')


def _out_subdir(model_type):
    return {
        'standard':            'aprf.cv',
        'session-shift':       'aprf-shift.cv',
        'gaussian':            'aprf-gauss.cv',
        'gauss-session-shift': 'aprf-gauss-shift.cv',
    }[model_type]


# ── paradigm ────────────────────────────────────────────────────────────────

def _get_paradigm(sub, sessions, needs_session):
    """Return paradigm DataFrame indexed by (session, run, trial).

    Columns:
      'x'                    — objective CHF value (float32)
      'session' (if shift)   — 0-based session index (float32), which is what
                               the SessionShifted* models consume.

    The MultiIndex 'session' level always holds the actual session number
    (e.g. 1, 2) so leave-one-run-out masking can be done by session+run.
    """
    rows = []
    for session_idx, session in enumerate(sorted(sessions)):
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            run_ev = events.loc[run].reset_index().sort_values('onset')
            for _, row in run_ev[run_ev['event_type'] == 'gabor'].iterrows():
                rows.append({
                    'session':     session,
                    'run':         run,
                    'x':           float(row['value']),
                    'session_idx': float(session_idx),
                })
    df = pd.DataFrame(rows).astype({'x': np.float32, 'session_idx': np.float32})
    df.index = pd.MultiIndex.from_frame(
        df[['session', 'run']].assign(trial=df.groupby(['session', 'run']).cumcount()),
        names=['session', 'run', 'trial'],
    )
    if needs_session:
        return df[['x', 'session_idx']].rename(columns={'session_idx': 'session'})
    return df[['x']]


# ── main ────────────────────────────────────────────────────────────────────

def main(subject, sessions=None, n_iterations=1000, mask=None,
         bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep',
         smoothed=False, debug=False, model_type='standard'):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)

    if sessions is None:
        sessions = sub.get_sessions()
    sessions = sorted(sessions)

    needs_session = model_type in _SHIFT_MODELS
    if needs_session and len(sessions) < 2:
        raise ValueError(f'--model {model_type} requires at least 2 sessions.')

    ses_dir    = f'ses-{sessions[0]}' if len(sessions) == 1 else ''
    ses_entity = f'_ses-{sessions[0]}' if len(sessions) == 1 else ''
    print(f'sub-{subject}  {ses_dir or "all-sessions"}  '
          f'[abstract pRF CV  model={model_type}]')

    if debug:
        n_iterations = 50

    # ── paradigm ─────────────────────────────────────────────────────────────
    paradigm = _get_paradigm(sub, sessions, needs_session)
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

    # ── grid (reused across folds; value range from full dataset) ────────────
    if needs_session:
        n_mode = 8 if debug else 12
        n_fwhm = 5 if debug else 8
    else:
        n_mode = 12 if debug else 20
        n_fwhm = 8  if debug else 15
    modes      = np.linspace(value_min, value_max, n_mode).astype(np.float32)
    fwhms      = np.linspace(1.0, value_max - value_min, n_fwhm).astype(np.float32)
    amplitudes = np.array([1.0], dtype=np.float32)
    baselines  = np.array([0.0], dtype=np.float32)
    if needs_session:
        print(f'  grid: {n_mode}×{n_mode}×{n_fwhm} '
              f'= {n_mode*n_mode*n_fwhm} points per fold')
    else:
        print(f'  grid: {n_mode}×{n_fwhm} = {n_mode*n_fwhm} points per fold')

    # ── output directory ──────────────────────────────────────────────────────
    smooth_label = '_smoothed' if smoothed else ''
    out_dir = bids_folder / 'derivatives' / 'encoding_models' / _out_subdir(model_type) / f'sub-{subject}'
    if ses_dir:
        out_dir = out_dir / ses_dir
    out_dir = out_dir / 'func'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-fold filenames always carry ses+run to avoid cross-session collisions
    fn_run  = (f'sub-{subject}_ses-{{ses}}_task-abstractvalue'
               f'_space-T1w_run-{{run}}_desc-cvr2{smooth_label}_pe.nii.gz')
    fn_mean = (f'sub-{subject}{ses_entity}_task-abstractvalue'
               f'_space-T1w_desc-cvr2{smooth_label}_pe.nii.gz')

    paradigm_cols = ['x', 'session'] if needs_session else ['x']

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

        train_paradigm = paradigm.loc[train_mask].reset_index(drop=True)[paradigm_cols]
        train_data     = data.loc[train_mask].reset_index(drop=True)
        test_paradigm  = paradigm.loc[test_mask].reset_index(drop=True)[paradigm_cols]
        test_data      = data.loc[test_mask].reset_index(drop=True)

        model  = _build_model(model_type)
        fitter = ParameterFitter(model, train_data, train_paradigm)

        print('    grid search...')
        if needs_session:
            grid_pars = fitter.fit_grid(modes, modes, fwhms, amplitudes, baselines,
                                        use_correlation_cost=True)
        else:
            grid_pars = fitter.fit_grid(modes, fwhms, amplitudes, baselines,
                                        use_correlation_cost=True)
        grid_pars = fitter.refine_baseline_and_amplitude(grid_pars)

        print(f'    gradient descent ({n_iterations} iters)...')
        pars = fitter.fit(max_n_iterations=n_iterations, init_pars=grid_pars)

        test_pred = model.predict(parameters=pars, paradigm=test_paradigm)
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
    parser.add_argument('--model', default='standard',
                        choices=['standard', 'session-shift',
                                 'gaussian', 'gauss-session-shift'],
                        help='Model type (default: standard)')
    parser.add_argument('--n-iterations', type=int, default=1000)
    parser.add_argument('--mask', default=None)
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep',
                        choices=['fmriprep', 'fmriprep-t2w'])
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help='Only 50 GD iterations per fold, small grid (fast test)')
    args = parser.parse_args()

    main(args.subject, sessions=args.sessions, n_iterations=args.n_iterations,
         mask=args.mask, bids_folder=args.bids_folder,
         fmriprep_deriv=args.fmriprep_deriv, smoothed=args.smoothed,
         debug=args.debug, model_type=args.model)
