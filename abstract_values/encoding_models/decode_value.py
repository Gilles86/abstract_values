#!/usr/bin/env python3
"""
Bayesian decoding of abstract value (CHF) from single-trial fMRI amplitudes.

Overview
--------
Leave-one-run-out cross-validation.  In each fold:

  1. Fit a Gaussian nPRF encoding model on training runs (grid search +
     gradient descent via ParameterFitter; same approach as fit_aprf.py).
  2. Select voxels: top n_voxels by training R², or (when n_voxels=0) all
     voxels with nested cross-validated R² > 0 (inner leave-one-run-out CV
     within the training set — no circularity).
  3. Fit a multivariate Student-t residual noise model (ResidualFitter).
  4. Evaluate P(data | value) over the stimulus grid for each test trial.
     This unnormalised likelihood serves as the posterior PDF under a flat
     prior.

Mask suggestions
----------------
  brain (default): fmriprep whole-brain mask
  V1/V2/V3:        derivatives/masks/sub-<s>/ses-<n>/anat/
                   sub-<s>_ses-<n>_space-T1w_hemi-LR_desc-BensonV1_mask.nii.gz
  IPS:             create with a script analogous to create_roi_masks.py
                   (project IPS surface labels to T1w volume)

Output
------
  derivatives/decoding/value/sub-<subject>/<ses_label>/func/
    sub-<subject>_<ses_label>_mask-<mask_desc>_nvoxels-<n>
    _noise-<spherical|full>[_smoothed]_pars.tsv

  One row per test trial, columns = value grid (CHF).
  Row index: (session, run, trial_nr, true_value_chf).

Usage
-----
  python decode_value.py pil01 --sessions 1
  python decode_value.py pil01 --sessions 1 --n-voxels 200
  python decode_value.py pil01 --sessions 1 \\
      --mask /data/ds-abstractvalue/derivatives/masks/sub-pil01/ses-1/anat/\\
             sub-pil01_ses-1_space-T1w_hemi-LR_desc-BensonV1_mask.nii.gz \\
      --mask-desc BensonV1
  python decode_value.py pil01 --sessions 1 --spherical-noise --smoothed
  python decode_value.py pil01 --sessions 1 --debug
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import LogGaussianPRF
from braincoder.optimize import ParameterFitter, ResidualFitter
from braincoder.utils import get_rsq

from abstract_values.utils.data import Subject, BIDS_FOLDER


def get_value_paradigm(sub, sessions):
    """Return DataFrame indexed by (session, run, trial_nr) with column 'x'.

    x = objective CHF value, in the same order as the gabor betas written by
    fit_glmsingle (session → run → events sorted by onset, gabor only).
    """
    rows = []
    for session in sessions:
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            run_ev = events.loc[run].reset_index().sort_values('onset')
            for _, row in run_ev[run_ev['event_type'] == 'gabor'].iterrows():
                rows.append({
                    'session':  session,
                    'run':      run,
                    'trial_nr': int(row['trial_nr']),
                    'x':        np.float32(float(row['value'])),
                })
    df = pd.DataFrame(rows).set_index(['session', 'run', 'trial_nr'])
    return df[['x']]


def main(subject, sessions=None, n_voxels=100, n_iterations=1000,
         n_grid_mus=20, n_grid_sds=15, n_stimulus_grid=50,
         lambd=0.0, mask=None, mask_desc=None, spherical_noise=False,
         bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep',
         smoothed=False, debug=False):

    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)

    if sessions is None:
        sessions = sub.get_sessions()

    if debug:
        n_iterations = 100

    ses_dir    = f'ses-{sessions[0]}' if len(sessions) == 1 else ''
    ses_entity = f'_ses-{sessions[0]}' if len(sessions) == 1 else ''
    print(f'sub-{subject}  {ses_dir or "all-sessions"}  [abstract value decoding]')

    # ── paradigm + data ───────────────────────────────────────────────────────
    paradigm = get_value_paradigm(sub, sessions)
    value_min = float(paradigm['x'].min())
    value_max = float(paradigm['x'].max())
    print(f'  {len(paradigm)} trials  value range: {value_min:.1f}–{value_max:.1f} CHF')

    betas_img = sub.get_single_trial_estimates(sessions, desc='gabor',
                                               smoothed=smoothed)
    assert betas_img.shape[3] == len(paradigm)

    if mask is None:
        raise ValueError('Please provide --mask and --mask-desc (whole-brain decoding is not supported)')
    masker = NiftiMasker(mask_img=mask,
                        target_affine=betas_img.affine,
                        target_shape=betas_img.shape[:3]).fit()

    data = pd.DataFrame(
        masker.transform(betas_img).astype(np.float32),
        index=paradigm.index)
    print(f'  {data.shape[1]} voxels in mask ({mask_desc})')

    # ── stimulus grid ─────────────────────────────────────────────────────────
    stimulus_range = np.linspace(value_min, value_max, n_stimulus_grid,
                                 dtype=np.float32)
    print(f'  stimulus grid: {n_stimulus_grid} values '
          f'({value_min:.1f}–{value_max:.1f} CHF)')

    # ── output ────────────────────────────────────────────────────────────────
    out_dir = bids_folder / 'derivatives' / 'decoding' / 'value' / f'sub-{subject}'
    if ses_dir:
        out_dir = out_dir / ses_dir
    out_dir = out_dir / 'func'
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_label  = 'spherical' if spherical_noise else 'full'
    smooth_label = '_smoothed' if smoothed else ''
    lambd_label  = f'_lambda-{lambd}' if lambd != 0.0 else ''
    out_fn = (out_dir /
              f'sub-{subject}{ses_entity}_mask-{mask_desc}'
              f'_nvoxels-{n_voxels}_noise-{noise_label}{smooth_label}{lambd_label}_pars.tsv')

    # ── leave-one-run-out cross-validation ────────────────────────────────────
    all_pdfs = []
    all_runs = [(s, r) for s in sessions for r in sub.get_runs(s)]

    for test_session, test_run in all_runs:
        print(f'\n  [fold] hold-out ses-{test_session} run-{test_run}')

        test_idx  = (paradigm.index.get_level_values('session') == test_session) & \
                    (paradigm.index.get_level_values('run') == test_run)
        train_idx = ~test_idx

        train_paradigm = paradigm.loc[train_idx]
        test_paradigm  = paradigm.loc[test_idx]
        train_data     = data.loc[train_idx]
        test_data      = data.loc[test_idx]

        # ── fit Log-Gaussian nPRF (grid search → gradient descent) ───────────
        model = LogGaussianPRF(allow_neg_amplitudes=True, parameterisation='mu_sd_natural')
        fitter = ParameterFitter(model, train_data, train_paradigm)

        mus        = np.linspace(value_min, value_max, n_grid_mus,
                                 dtype=np.float32)
        sds        = np.linspace(1.0, (value_max - value_min) / 2, n_grid_sds,
                                 dtype=np.float32)
        amplitudes = np.array([1.0], dtype=np.float32)
        baselines  = np.array([0.0], dtype=np.float32)

        grid_pars = fitter.fit_grid(mus, sds, amplitudes, baselines,
                                    use_correlation_cost=True)
        grid_pars = fitter.refine_baseline_and_amplitude(grid_pars)
        pars = fitter.fit(max_n_iterations=n_iterations, init_pars=grid_pars)

        # ── voxel selection ────────────────────────────────────────────────────
        if n_voxels == 0:
            # Nested CV: leave-one-run-out within training set to get unbiased R²
            inner_runs = list(zip(
                train_paradigm.index.get_level_values('session'),
                train_paradigm.index.get_level_values('run')))
            inner_runs = sorted(set(inner_runs))

            inner_r2s = []
            for inner_ses, inner_run in inner_runs:
                print(f'      [inner CV] hold-out ses-{inner_ses} run-{inner_run}')
                inner_test_idx = (
                    (train_paradigm.index.get_level_values('session') == inner_ses) &
                    (train_paradigm.index.get_level_values('run') == inner_run))
                inner_train_paradigm = train_paradigm.loc[~inner_test_idx]
                inner_test_paradigm  = train_paradigm.loc[inner_test_idx]
                inner_train_data     = train_data.loc[~inner_test_idx]
                inner_test_data      = train_data.loc[inner_test_idx]

                inner_model = LogGaussianPRF(allow_neg_amplitudes=True,
                                             parameterisation='mu_sd_natural')
                inner_fitter = ParameterFitter(inner_model, inner_train_data,
                                               inner_train_paradigm)
                inner_grid = inner_fitter.fit_grid(
                    mus, sds, amplitudes, baselines, use_correlation_cost=True)
                inner_grid = inner_fitter.refine_baseline_and_amplitude(inner_grid)
                inner_pars = inner_fitter.fit(max_n_iterations=n_iterations,
                                              init_pars=inner_grid)

                inner_pred = inner_model.predict(parameters=inner_pars,
                                                  paradigm=inner_test_paradigm)
                inner_r2s.append(get_rsq(inner_test_data, inner_pred))

            cv_r2 = pd.concat(inner_r2s, axis=1).mean(axis=1)
            sel = cv_r2[cv_r2 > 0.0].index
            print(f'    {len(sel)} voxels selected  '
                  f'(nested CV R² > 0, mean={float(cv_r2.loc[sel].mean()):.3f})')
        else:
            pred_train = model.predict(parameters=pars, paradigm=train_paradigm)
            r2_train   = get_rsq(train_data, pred_train)
            sel = r2_train.sort_values(ascending=False).index[:n_voxels]
            print(f'    {len(sel)} voxels selected  '
                  f'(train R² ≥ {float(r2_train.loc[sel].min()):.3f})')

        pars_sel       = pars.loc[sel]
        train_data_sel = train_data[sel]
        test_data_sel  = test_data[sel]

        # ── fit noise model ───────────────────────────────────────────────────
        # Re-create model so state is clean for the selected voxels.
        # LogGaussianPRF uses a pseudo-WWT, so init_pseudoWWT must be called first.
        model_sel = LogGaussianPRF(allow_neg_amplitudes=True, parameterisation='mu_sd_natural')
        model_sel.init_pseudoWWT(stimulus_range, pars_sel)
        n_iter_noise = 100 if debug else 5000
        residfit = ResidualFitter(model_sel, train_data_sel, train_paradigm,
                                  parameters=pars_sel, lambd=lambd)
        omega, dof = residfit.fit(
            init_sigma2=0.1, init_dof=10.0, method='t',
            learning_rate=0.05, spherical=spherical_noise,
            max_n_iterations=n_iter_noise)
        print(f'    noise model: dof={float(dof):.1f}')

        # ── decode ────────────────────────────────────────────────────────────
        pdf = model_sel.get_stimulus_pdf(test_data_sel, stimulus_range,
                                         parameters=pars_sel,
                                         omega=omega, dof=dof,
                                         normalize=False)
        pdf.columns = stimulus_range
        pdf.index = pd.MultiIndex.from_arrays([
            test_paradigm.index.get_level_values('session'),
            test_paradigm.index.get_level_values('run'),
            test_paradigm.index.get_level_values('trial_nr'),
            test_paradigm['x'].values,
        ], names=['session', 'run', 'trial_nr', 'true_value_chf'])

        all_pdfs.append(pdf)

    # ── save ──────────────────────────────────────────────────────────────────
    pdfs = pd.concat(all_pdfs).sort_index()
    pdfs.to_csv(out_fn, sep='\t')
    print(f'\n  saved to {out_fn}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-'")
    parser.add_argument('--sessions', type=int, nargs='+', default=None)
    parser.add_argument('--n-voxels', type=int, default=100,
                        help='Top-N voxels by training R² (0 = nested CV R²>0)')
    parser.add_argument('--n-iterations', type=int, default=1000,
                        help='Max gradient descent iterations (default: 1000)')
    parser.add_argument('--n-stimulus-grid', type=int, default=50,
                        help='Number of value bins in stimulus grid (default: 50)')
    parser.add_argument('--lambd', type=float, default=0.0,
                        help='Lambda regularization for noise model (default: 0)')
    parser.add_argument('--mask', default=None,
                        help='Brain mask NIfTI (default: fmriprep brain mask)')
    parser.add_argument('--mask-desc', default=None,
                        help='Short label for mask used in output filename')
    parser.add_argument('--spherical-noise', action='store_true',
                        help='Fit isotropic noise model instead of full covariance')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep',
                        choices=['fmriprep', 'fmriprep-t2w'])
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help='100 iterations each (fast test)')
    args = parser.parse_args()

    main(args.subject, sessions=args.sessions, n_voxels=args.n_voxels,
         n_iterations=args.n_iterations, n_stimulus_grid=args.n_stimulus_grid,
         lambd=args.lambd, mask=args.mask, mask_desc=args.mask_desc,
         spherical_noise=args.spherical_noise,
         bids_folder=args.bids_folder, fmriprep_deriv=args.fmriprep_deriv,
         smoothed=args.smoothed, debug=args.debug)
