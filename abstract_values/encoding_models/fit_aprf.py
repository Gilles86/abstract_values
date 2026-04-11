#!/usr/bin/env python3
"""
Fit an abstract pRF (Gaussian) encoding model to single-trial GLMsingle betas
using the OBJECTIVE VALUE of each trial's gabor stimulus as the paradigm.

Model
-----
  standard (default):
    LogGaussianPRF with parameters [mu, sd, amplitude, baseline].
    Fits per-session or across sessions treating value as the only stimulus dim.

  session-shift:
    SessionShiftedLogGaussianPRF — mu shifts freely between sessions while
    sd, amplitude, baseline are shared.  Requires at least 2 sessions.
    Parameters: [mu_1, mu_2, sd, amplitude, baseline].

  gaussian:
    GaussianValuePRF — symmetric Gaussian (no rightward skew).
    Parameters: [mode, fwhm, amplitude, baseline].

  gauss-session-shift:
    SessionShiftedGaussianValuePRF — symmetric Gaussian with mode shifting
    between sessions.  Requires at least 2 sessions.
    Parameters: [mode_1, mode_2, fwhm, amplitude, baseline].

Fitting is non-linear: grid search (correlation cost) followed by
gradient descent (Adam) via braincoder.optimize.ParameterFitter.

Output
------
  standard:
    derivatives/encoding_models/aprf/sub-<subject>/<ses_label>/func/
      params: mu, sd, amplitude, baseline, r2, fwhm

  session-shift:
    derivatives/encoding_models/aprf-session-shift/sub-<subject>/<ses_label>/func/
      params: mode_1, mode_2, fwhm, amplitude, baseline, r2

  gaussian:
    derivatives/encoding_models/aprf-gauss/sub-<subject>/<ses_label>/func/
      params: mode, fwhm, amplitude, baseline, r2

  gauss-session-shift:
    derivatives/encoding_models/aprf-gauss-session-shift/sub-<subject>/<ses_label>/func/
      params: mode_1, mode_2, fwhm, amplitude, baseline, r2

Usage
-----
  python fit_aprf.py pil01 --sessions 1
  python fit_aprf.py pil01 --sessions 1 --mask /path/to/mask.nii.gz
  python fit_aprf.py pil01 --sessions 1 2 --model session-shift
  python fit_aprf.py pil01 --sessions 1 --model gaussian
  python fit_aprf.py pil01 --sessions 1 2 --model gauss-session-shift --debug
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import LogGaussianPRF
from braincoder.optimize import ParameterFitter
from braincoder.utils import get_rsq

from abstract_values.utils.data import Subject, BIDS_FOLDER


def save_f32(img, path):
    """Save a NIfTI image as float32, regardless of mask dtype."""
    nib.Nifti1Image(img.get_fdata().astype(np.float32),
                    img.affine).to_filename(str(path))


def get_value_paradigm(sub, sessions):
    """Return DataFrame with column 'x' = objective CHF value (float32).

    Row order matches the gabor betas from fit_glmsingle:
    session → run (sorted) → events sorted by onset, gabor only.
    """
    rows = []
    for session in sessions:
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            run_ev = events.loc[run].reset_index().sort_values('onset')
            for _, row in run_ev[run_ev['event_type'] == 'gabor'].iterrows():
                rows.append(float(row['value']))
    return pd.DataFrame({'x': np.array(rows, dtype=np.float32)})


def get_value_session_paradigm(sub, sessions):
    """Return DataFrame with columns 'x' (CHF value) and 'session' (0-based index).

    Session 0 = first session in sorted order, session 1 = second, etc.
    """
    xs, session_ids = [], []
    for session_idx, session in enumerate(sorted(sessions)):
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            run_ev = events.loc[run].reset_index().sort_values('onset')
            for _, row in run_ev[run_ev['event_type'] == 'gabor'].iterrows():
                xs.append(float(row['value']))
                session_ids.append(float(session_idx))
    session_arr = np.array(session_ids, dtype=np.float32)
    unique_sessions = set(session_arr)
    if not unique_sessions <= {0.0, 1.0}:
        raise ValueError(
            f'Session column must contain only 0 and 1, got {unique_sessions}. '
            f'SessionShiftedLogGaussianPRF uses session=0 for the first session '
            f'and session=1 for the second.')
    return pd.DataFrame({
        'x':       np.array(xs, dtype=np.float32),
        'session': session_arr,
    })


def main(subject, sessions=None, mask=None, n_iterations=1000, model_type='standard',
         bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep',
         smoothed=False, debug=False):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)

    if sessions is None:
        sessions = sub.get_sessions()
    sessions = sorted(sessions)

    if model_type in ('session-shift', 'gauss-session-shift') and len(sessions) < 2:
        raise ValueError(f'--model {model_type} requires at least 2 sessions')

    ses_dir    = f'ses-{sessions[0]}' if len(sessions) == 1 else ''
    ses_entity = f'_ses-{sessions[0]}' if len(sessions) == 1 else ''
    print(f'sub-{subject}  {ses_dir or "all-sessions"}  '
          f'[abstract pRF on objective value  model={model_type}]')

    if debug:
        n_iterations = 50

    # ── paradigm ─────────────────────────────────────────────────────────────
    if model_type in ('session-shift', 'gauss-session-shift'):
        paradigm = get_value_session_paradigm(sub, sessions)
    else:
        paradigm = get_value_paradigm(sub, sessions)
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
    masker = NiftiMasker(mask_img=mask).fit()
    data = pd.DataFrame(masker.transform(betas_img).astype(np.float32))
    print(f'  {data.shape[1]} voxels in mask')

    _skip_save = False
    if debug and data.shape[1] > 1000:
        rng = np.random.default_rng(0)
        debug_cols = rng.choice(data.shape[1], 1000, replace=False)
        data = data.iloc[:, debug_cols]
        _skip_save = True
        print(f'  [debug] subsampled to {data.shape[1]} voxels (saving skipped)')

    # ── model + grid search + gradient descent ────────────────────────────────
    smooth_label = '_smoothed' if smoothed else ''

    if model_type == 'session-shift':
        from abstract_values.encoding_models.models import SessionShiftedLogGaussianPRF

        model  = SessionShiftedLogGaussianPRF(allow_neg_amplitudes=False)
        fitter = ParameterFitter(model, data, paradigm)

        n_mode = 8 if debug else 15
        n_fwhm = 6 if debug else 10
        modes      = np.linspace(value_min, value_max, n_mode).astype(np.float32)
        fwhms      = np.linspace(1.0, value_max - value_min, n_fwhm).astype(np.float32)
        amplitudes = np.array([1.0], dtype=np.float32)
        baselines  = np.array([0.0], dtype=np.float32)

        print(f'  grid search ({n_mode}×{n_mode}×{n_fwhm} = {n_mode*n_mode*n_fwhm} points)...')
        grid_pars = fitter.fit_grid(modes, modes, fwhms, amplitudes, baselines,
                                    use_correlation_cost=True)
        grid_pars = fitter.refine_baseline_and_amplitude(grid_pars)

        print(f'  gradient descent ({n_iterations} iterations)...')
        pars = fitter.fit(max_n_iterations=n_iterations, init_pars=grid_pars)

        pred = model.predict(parameters=pars, paradigm=paradigm)
        r2   = get_rsq(data, pred)
        print(f'  mean R²={float(r2.mean()):.4f}')

        if not _skip_save:
            out_dir = (bids_folder / 'derivatives' / 'encoding_models'
                       / 'aprf-session-shift' / f'sub-{subject}')
            if ses_dir:
                out_dir = out_dir / ses_dir
            out_dir = out_dir / 'func'
            out_dir.mkdir(parents=True, exist_ok=True)

            fn = (f'sub-{subject}{ses_entity}_task-abstractvalue'
                  f'_space-T1w_desc-{{desc}}{smooth_label}_pe.nii.gz')

            for param in ['mode_1', 'mode_2', 'fwhm', 'amplitude', 'baseline']:
                save_f32(masker.inverse_transform(pars[param]),
                         out_dir / fn.format(desc=param))
            save_f32(masker.inverse_transform(r2), out_dir / fn.format(desc='r2'))

            print(f'  saved to {out_dir}')

    elif model_type == 'gaussian':
        from abstract_values.encoding_models.models import GaussianValuePRF

        model  = GaussianValuePRF(allow_neg_amplitudes=False)
        fitter = ParameterFitter(model, data, paradigm)

        n_mode = 12 if debug else 20
        n_fwhm = 8  if debug else 15
        modes      = np.linspace(value_min, value_max, n_mode).astype(np.float32)
        fwhms      = np.linspace(1.0, value_max - value_min, n_fwhm).astype(np.float32)
        amplitudes = np.array([1.0], dtype=np.float32)
        baselines  = np.array([0.0], dtype=np.float32)

        print(f'  grid search ({n_mode}×{n_fwhm} = {n_mode*n_fwhm} points)...')
        grid_pars = fitter.fit_grid(modes, fwhms, amplitudes, baselines,
                                    use_correlation_cost=True)
        grid_pars = fitter.refine_baseline_and_amplitude(grid_pars)

        print(f'  gradient descent ({n_iterations} iterations)...')
        pars = fitter.fit(max_n_iterations=n_iterations, init_pars=grid_pars)

        pred = model.predict(parameters=pars, paradigm=paradigm)
        r2   = get_rsq(data, pred)
        print(f'  mean R²={float(r2.mean()):.4f}')

        if not _skip_save:
            out_dir = (bids_folder / 'derivatives' / 'encoding_models'
                       / 'aprf-gauss' / f'sub-{subject}')
            if ses_dir:
                out_dir = out_dir / ses_dir
            out_dir = out_dir / 'func'
            out_dir.mkdir(parents=True, exist_ok=True)

            fn = (f'sub-{subject}{ses_entity}_task-abstractvalue'
                  f'_space-T1w_desc-{{desc}}{smooth_label}_pe.nii.gz')

            for param in ['mode', 'fwhm', 'amplitude', 'baseline']:
                save_f32(masker.inverse_transform(pars[param]),
                         out_dir / fn.format(desc=param))
            save_f32(masker.inverse_transform(r2), out_dir / fn.format(desc='r2'))

            print(f'  saved to {out_dir}')

    elif model_type == 'gauss-session-shift':
        from abstract_values.encoding_models.models import SessionShiftedGaussianValuePRF

        model  = SessionShiftedGaussianValuePRF(allow_neg_amplitudes=False)
        fitter = ParameterFitter(model, data, paradigm)

        n_mode = 8 if debug else 15
        n_fwhm = 6 if debug else 10
        modes      = np.linspace(value_min, value_max, n_mode).astype(np.float32)
        fwhms      = np.linspace(1.0, value_max - value_min, n_fwhm).astype(np.float32)
        amplitudes = np.array([1.0], dtype=np.float32)
        baselines  = np.array([0.0], dtype=np.float32)

        print(f'  grid search ({n_mode}×{n_mode}×{n_fwhm} = {n_mode*n_mode*n_fwhm} points)...')
        grid_pars = fitter.fit_grid(modes, modes, fwhms, amplitudes, baselines,
                                    use_correlation_cost=True)
        grid_pars = fitter.refine_baseline_and_amplitude(grid_pars)

        print(f'  gradient descent ({n_iterations} iterations)...')
        pars = fitter.fit(max_n_iterations=n_iterations, init_pars=grid_pars)

        pred = model.predict(parameters=pars, paradigm=paradigm)
        r2   = get_rsq(data, pred)
        print(f'  mean R²={float(r2.mean()):.4f}')

        if not _skip_save:
            out_dir = (bids_folder / 'derivatives' / 'encoding_models'
                       / 'aprf-gauss-session-shift' / f'sub-{subject}')
            if ses_dir:
                out_dir = out_dir / ses_dir
            out_dir = out_dir / 'func'
            out_dir.mkdir(parents=True, exist_ok=True)

            fn = (f'sub-{subject}{ses_entity}_task-abstractvalue'
                  f'_space-T1w_desc-{{desc}}{smooth_label}_pe.nii.gz')

            for param in ['mode_1', 'mode_2', 'fwhm', 'amplitude', 'baseline']:
                save_f32(masker.inverse_transform(pars[param]),
                         out_dir / fn.format(desc=param))
            save_f32(masker.inverse_transform(r2), out_dir / fn.format(desc='r2'))

            print(f'  saved to {out_dir}')

    else:  # standard LogGaussianPRF
        model  = LogGaussianPRF(allow_neg_amplitudes=False, parameterisation='mode_fwhm_natural')
        fitter = ParameterFitter(model, data, paradigm)

        n_mode = 12 if debug else 20
        n_fwhm = 8  if debug else 15
        modes      = np.linspace(value_min, value_max, n_mode).astype(np.float32)
        fwhms      = np.linspace(1.0, value_max - value_min, n_fwhm).astype(np.float32)
        amplitudes = np.array([1.0], dtype=np.float32)
        baselines  = np.array([0.0], dtype=np.float32)

        print(f'  grid search ({n_mode}×{n_fwhm} = {n_mode*n_fwhm} points)...')
        grid_pars = fitter.fit_grid(modes, fwhms, amplitudes, baselines,
                                    use_correlation_cost=True)
        grid_pars = fitter.refine_baseline_and_amplitude(grid_pars)

        print(f'  gradient descent ({n_iterations} iterations)...')
        pars = fitter.fit(max_n_iterations=n_iterations, init_pars=grid_pars)

        pred = model.predict(parameters=pars, paradigm=paradigm)
        r2   = get_rsq(data, pred)
        print(f'  mean R²={float(r2.mean()):.4f}')

        if not _skip_save:
            out_dir = (bids_folder / 'derivatives' / 'encoding_models'
                       / 'aprf' / f'sub-{subject}')
            if ses_dir:
                out_dir = out_dir / ses_dir
            out_dir = out_dir / 'func'
            out_dir.mkdir(parents=True, exist_ok=True)

            fn = (f'sub-{subject}{ses_entity}_task-abstractvalue'
                  f'_space-T1w_desc-{{desc}}{smooth_label}_pe.nii.gz')

            for param in ['mode', 'fwhm', 'amplitude', 'baseline']:
                save_f32(masker.inverse_transform(pars[param]),
                         out_dir / fn.format(desc=param))
            save_f32(masker.inverse_transform(r2), out_dir / fn.format(desc='r2'))

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
    parser.add_argument('--mask', default=None,
                        help='Brain mask NIfTI (default: fmriprep brain mask)')
    parser.add_argument('--n-iterations', type=int, default=1000,
                        help='Max gradient descent iterations (default: 1000)')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep',
                        choices=['fmriprep', 'fmriprep-t2w'])
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help='Fast local test: 50 iterations, 500 voxels, small grid')
    args = parser.parse_args()

    main(args.subject, sessions=args.sessions, mask=args.mask,
         n_iterations=args.n_iterations, model_type=args.model,
         bids_folder=args.bids_folder, fmriprep_deriv=args.fmriprep_deriv,
         smoothed=args.smoothed, debug=args.debug)
