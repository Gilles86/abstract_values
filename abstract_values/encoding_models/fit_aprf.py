#!/usr/bin/env python3
"""
Fit an abstract pRF (Gaussian) encoding model to single-trial GLMsingle betas
using the OBJECTIVE VALUE of each trial's gabor stimulus as the paradigm.

Model
-----
  GaussianPRF (braincoder) with parameters [mu, sd, amplitude, baseline].
  Fitting is non-linear: grid search (correlation cost) followed by
  gradient descent (Adam) via braincoder.optimize.ParameterFitter.

Paradigm
--------
  The 'value' column from gabor events — the objective CHF value assigned to
  the orientation shown on each trial. Same trial ordering as the gabor betas
  written by fit_glmsingle (session → run → event sorted by onset).

Output
------
  derivatives/encoding_models/aprf/sub-<subject>/<ses_label>/func/
    sub-<subject>_<ses_label>_task-abstractvalue_space-T1w_desc-mu_pe.nii.gz
    sub-<subject>_<ses_label>_task-abstractvalue_space-T1w_desc-sd_pe.nii.gz
    sub-<subject>_<ses_label>_task-abstractvalue_space-T1w_desc-amplitude_pe.nii.gz
    sub-<subject>_<ses_label>_task-abstractvalue_space-T1w_desc-baseline_pe.nii.gz
    sub-<subject>_<ses_label>_task-abstractvalue_space-T1w_desc-r2_pe.nii.gz

Usage
-----
  python fit_aprf.py pil01 --sessions 1
  python fit_aprf.py pil01 --sessions 1 --mask /path/to/mask.nii.gz
  python fit_aprf.py pil01 --sessions 1 --smoothed --debug
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import LogGaussianPRF
from braincoder.optimize import ParameterFitter
from braincoder.utils import get_rsq

from abstract_values.utils.data import Subject, BIDS_FOLDER


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


def main(subject, sessions=None, mask=None, n_iterations=1000,
         bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep-flair',
         smoothed=False, debug=False):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)

    if sessions is None:
        sessions = sub.get_sessions()

    ses_label = f'ses-{sessions[0]}' if len(sessions) == 1 else 'ses-all'
    print(f'sub-{subject}  {ses_label}  [abstract pRF on objective value]')

    if debug:
        n_iterations = 100

    # ── paradigm ─────────────────────────────────────────────────────────────
    paradigm = get_value_paradigm(sub, sessions)
    value_min, value_max = float(paradigm['x'].min()), float(paradigm['x'].max())
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

    # ── model ─────────────────────────────────────────────────────────────────
    model = LogGaussianPRF(allow_neg_amplitudes=True, parameterisation='mu_sd_natural')
    fitter = ParameterFitter(model, data, paradigm)

    # Grid search over mu and sd; amplitude and baseline refined afterwards
    mus        = np.linspace(value_min, value_max, 20).astype(np.float32)
    sds        = np.linspace(1.0, (value_max - value_min) / 2, 15).astype(np.float32)
    amplitudes = np.array([1.0], dtype=np.float32)
    baselines  = np.array([0.0], dtype=np.float32)

    print('  grid search...')
    grid_pars = fitter.fit_grid(mus, sds, amplitudes, baselines,
                                use_correlation_cost=True)
    grid_pars = fitter.refine_baseline_and_amplitude(grid_pars)

    # Gradient descent
    print(f'  gradient descent ({n_iterations} iterations)...')
    pars = fitter.fit(max_n_iterations=n_iterations, init_pars=grid_pars)

    # ── R² ────────────────────────────────────────────────────────────────────
    pred = model.predict(parameters=pars, paradigm=paradigm)
    r2 = get_rsq(data, pred)
    print(f'  mean R²={float(r2.mean()):.4f}')

    # ── save ──────────────────────────────────────────────────────────────────
    smooth_label = '_smoothed' if smoothed else ''
    out_dir = (bids_folder / 'derivatives' / 'encoding_models' / 'aprf'
               / f'sub-{subject}' / ses_label / 'func')
    out_dir.mkdir(parents=True, exist_ok=True)

    fn = (f'sub-{subject}_{ses_label}_task-abstractvalue'
          f'_space-T1w_desc-{{desc}}{smooth_label}_pe.nii.gz')

    for param in ['mu', 'sd', 'amplitude', 'baseline']:
        masker.inverse_transform(pars[param]).to_filename(
            str(out_dir / fn.format(desc=param)))

    # ── FWHM in natural (CHF) space ───────────────────────────────────────────
    # For LogGaussianPRF(mu_sd_natural): σ_log = sqrt(log(1 + (sd/mu)²))
    # Half-max points: exp(μ_log ± σ_log·sqrt(2·log2)), μ_log = log(mu) - σ_log²/2
    sigma_log = np.sqrt(np.log(1 + (pars['sd'] / pars['mu'].clip(1e-6)) ** 2))
    mu_log    = np.log(pars['mu'].clip(1e-6)) - 0.5 * sigma_log ** 2
    k         = sigma_log * np.sqrt(2 * np.log(2))
    fwhm      = np.exp(mu_log + k) - np.exp(mu_log - k)
    masker.inverse_transform(fwhm).to_filename(str(out_dir / fn.format(desc='fwhm')))

    masker.inverse_transform(r2).to_filename(str(out_dir / fn.format(desc='r2')))

    print(f'  saved to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-'")
    parser.add_argument('--sessions', type=int, nargs='+', default=None)
    parser.add_argument('--mask', default=None,
                        help='Brain mask NIfTI (default: fmriprep brain mask)')
    parser.add_argument('--n-iterations', type=int, default=1000,
                        help='Max gradient descent iterations (default: 1000)')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep-flair',
                        choices=['fmriprep', 'fmriprep-flair', 'fmriprep-noflair', 'fmriprep-t2w'])
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help='Run only 100 gradient descent iterations (fast test)')
    args = parser.parse_args()

    main(args.subject, sessions=args.sessions, mask=args.mask,
         n_iterations=args.n_iterations, bids_folder=args.bids_folder,
         fmriprep_deriv=args.fmriprep_deriv, smoothed=args.smoothed,
         debug=args.debug)
