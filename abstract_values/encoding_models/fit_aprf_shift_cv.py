#!/usr/bin/env python3
"""
Leave-one-run-out cross-validated fitting of the session-shifted abstract pRF.

SessionShiftedLogGaussianPRF has a mode that shifts freely between sessions
(mode_1, mode_2) while fwhm/amplitude/baseline are shared across sessions.

For each fold (session, run) the model is fitted on the remaining runs via
grid search + gradient descent, then evaluated on the held-out run.  The
per-fold CV R² images are averaged and saved alongside the per-fold files.

Requires at least 2 sessions.

Output
------
  derivatives/encoding_models/aprf-shift.cv/sub-<subject>/func/
    sub-<subject>_task-abstractvalue_space-T1w_ses-<s>_run-<r>_desc-cvr2_pe.nii.gz
    sub-<subject>_task-abstractvalue_space-T1w_desc-cvr2_pe.nii.gz  (mean)

Usage
-----
  python fit_aprf_shift_cv.py pil01 --sessions 1 2
  python fit_aprf_shift_cv.py pil01 --sessions 1 2 --debug
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.optimize import ParameterFitter
from braincoder.utils import get_rsq

from abstract_values.encoding_models.models import SessionShiftedLogGaussianPRF
from abstract_values.utils.data import Subject, BIDS_FOLDER



def get_value_paradigm_with_runs(sub, sessions):
    """Return paradigm DataFrame indexed by (session, run, trial_idx).

    Columns:
      'x'       = objective CHF value (float32)
      'session' = 0-based session index (float32) for SessionShiftedLogGaussianPRF
                  (0 = first session in sorted order, 1 = second, …)

    The MultiIndex 'session' level holds the actual session number (e.g. 1, 2).
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
        names=['session', 'run', 'trial']
    )
    # Return 'x' and 0-based session indicator (renamed from session_idx → session)
    return df[['x', 'session_idx']].rename(columns={'session_idx': 'session'})


def main(subject, sessions=None, n_iterations=1000, mask=None,
         bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep',
         smoothed=False, debug=False):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)

    if sessions is None:
        sessions = sub.get_sessions()
    sessions = sorted(sessions)

    if len(sessions) < 2:
        raise ValueError('Session-shift model requires at least 2 sessions.')

    print(f'sub-{subject}  all-sessions  [abstract pRF session-shift CV]')

    if debug:
        n_iterations = 50

    # ── paradigm with run + session index ─────────────────────────────────────
    paradigm = get_value_paradigm_with_runs(sub, sessions)
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

    # ── grid (reused across folds; value range from full dataset) ─────────────
    n_mode = 8 if debug else 12
    n_fwhm = 5 if debug else 8
    modes      = np.linspace(value_min, value_max, n_mode).astype(np.float32)
    fwhms      = np.linspace(1.0, value_max - value_min, n_fwhm).astype(np.float32)
    amplitudes = np.array([1.0], dtype=np.float32)
    baselines  = np.array([0.0], dtype=np.float32)
    print(f'  grid: {n_mode}×{n_mode}×{n_fwhm} = {n_mode*n_mode*n_fwhm} points per fold')

    # ── output directory ──────────────────────────────────────────────────────
    smooth_label = '_smoothed' if smoothed else ''
    out_dir = (bids_folder / 'derivatives' / 'encoding_models'
               / 'aprf-shift.cv' / f'sub-{subject}' / 'func')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-fold: include ses+run to avoid collisions across sessions
    fn_run  = (f'sub-{subject}_task-abstractvalue'
               f'_space-T1w_ses-{{ses}}_run-{{run}}_desc-cvr2{smooth_label}_pe.nii.gz')
    fn_mean = (f'sub-{subject}_task-abstractvalue'
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

        train_paradigm = paradigm.loc[train_mask].reset_index(drop=True)[['x', 'session']]
        train_data     = data.loc[train_mask].reset_index(drop=True)
        test_paradigm  = paradigm.loc[test_mask].reset_index(drop=True)[['x', 'session']]
        test_data      = data.loc[test_mask].reset_index(drop=True)

        model  = SessionShiftedLogGaussianPRF(allow_neg_amplitudes=False)
        fitter = ParameterFitter(model, train_data, train_paradigm)

        print('    grid search...')
        grid_pars = fitter.fit_grid(modes, modes, fwhms, amplitudes, baselines,
                                    use_correlation_cost=True)
        grid_pars = fitter.refine_baseline_and_amplitude(grid_pars)

        print(f'    gradient descent ({n_iterations} iters)...')
        pars = fitter.fit(max_n_iterations=n_iterations, init_pars=grid_pars)

        # CV R² on held-out run
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
         debug=args.debug)
