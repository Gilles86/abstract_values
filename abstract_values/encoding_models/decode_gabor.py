#!/usr/bin/env python3
"""
Bayesian decoding of gabor orientation from single-trial fMRI amplitudes.

Overview
--------
Leave-one-run-out cross-validation.  In each fold:

  1. Fit VonMisesPRF encoding model on the training runs (grid search over
     mu/kappa, then Adam gradient descent; amplitude + baseline refined after
     the grid).
  2. Select the top n_voxels by training-set R² (or all voxels if n_voxels=0).
  3. Fit a multivariate Student-t residual noise model (ResidualFitter) on the
     training set to get a noise covariance omega and degrees of freedom dof.
  4. Evaluate P(data | orientation) over all 23 presented orientations for each
     test trial via model.get_stimulus_pdf().  This unnormalised likelihood
     serves as the posterior PDF under a flat prior.

Output
------
  derivatives/decoding/gabor/sub-<subject>/<ses_label>/func/
    sub-<subject>_<ses_label>_mask-<mask_desc>_nvoxels-<n>_pars.tsv

  One row per test trial, columns = decoded orientation grid (radians).
  Row index: (session, run, trial_nr, true_orientation_rad).

Usage
-----
  python decode_gabor.py pil01 --sessions 1
  python decode_gabor.py pil01 --sessions 1 --n-voxels 200
  python decode_gabor.py pil01 --sessions 1 --n-voxels 0   # all R²>0 voxels
  python decode_gabor.py pil01 --sessions 1 \\
      --mask /data/ds-abstractvalue/derivatives/masks/sub-pil01/ses-1/anat/\\
             sub-pil01_ses-1_space-T1w_hemi-LR_desc-BensonV1_mask.nii.gz \\
      --mask-desc BensonV1
  python decode_gabor.py pil01 --sessions 1 --debug
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import VonMisesPRF
from braincoder.optimize import ParameterFitter, ResidualFitter
from braincoder.utils import get_rsq

from abstract_values.utils.data import Subject, BIDS_FOLDER


def get_gabor_paradigm(sub, sessions):
    """Return DataFrame indexed by (session, run, trial_nr) with column 'x'.

    x = gabor orientation in radians, in the same order as the gabor betas
    written by fit_glmsingle (session → run → events sorted by onset).
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
                    'x':        np.float32(np.deg2rad(float(row['orientation']))),
                })
    df = pd.DataFrame(rows).set_index(['session', 'run', 'trial_nr'])
    return df[['x']]


def fit_encoding_model(fitter, n_iterations):
    """Grid search + Adam for VonMisesPRF.  Returns per-voxel parameters."""
    mus        = np.linspace(0, np.pi, 20).astype(np.float32)
    kappas     = np.linspace(0.5, 8.0,  15).astype(np.float32)
    amplitudes = np.array([1.0], dtype=np.float32)
    baselines  = np.array([0.0], dtype=np.float32)

    grid_pars = fitter.fit_grid(mus, kappas, amplitudes, baselines,
                                use_correlation_cost=True)
    grid_pars = fitter.refine_baseline_and_amplitude(grid_pars)
    return fitter.fit(max_n_iterations=n_iterations, init_pars=grid_pars)



def main(subject, sessions=None, n_voxels=100, mask=None, mask_desc='brain',
         n_iterations=1000, spherical_noise=False,
         bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep-flair',
         smoothed=False, debug=False):

    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)

    if sessions is None:
        sessions = sub.get_sessions()

    ses_label = f'ses-{sessions[0]}' if len(sessions) == 1 else 'ses-all'
    print(f'sub-{subject}  {ses_label}  [gabor orientation decoding]')

    if debug:
        n_iterations = 100

    # ── paradigm + data ───────────────────────────────────────────────────────
    paradigm = get_gabor_paradigm(sub, sessions)
    print(f'  {len(paradigm)} trials')

    betas_img = sub.get_single_trial_estimates(sessions, desc='gabor',
                                               smoothed=smoothed)
    assert betas_img.shape[3] == len(paradigm)

    if mask is None:
        mask = sub.get_brain_mask(sessions[0])
    masker = NiftiMasker(mask_img=mask).fit()

    data = pd.DataFrame(
        masker.transform(betas_img).astype(np.float32),
        index=paradigm.index)
    print(f'  {data.shape[1]} voxels in mask ({mask_desc})')

    # ── stimulus range (unique orientations, used for decoding grid) ──────────
    stimulus_range = np.sort(paradigm['x'].unique()).astype(np.float32)
    print(f'  stimulus grid: {len(stimulus_range)} orientations '
          f'({np.rad2deg(stimulus_range[[0,-1]]).round(1)} deg)')

    # ── output dir ───────────────────────────────────────────────────────────
    out_dir = (bids_folder / 'derivatives' / 'decoding' / 'gabor'
               / f'sub-{subject}' / ses_label / 'func')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fn = (out_dir /
              f'sub-{subject}_{ses_label}_mask-{mask_desc}'
              f'_nvoxels-{n_voxels}_pars.tsv')

    # ── leave-one-run-out cross-validation ────────────────────────────────────
    all_pdfs = []

    all_runs = [(s, r)
                for s in sessions
                for r in sub.get_runs(s)]

    for test_session, test_run in all_runs:
        print(f'\n  [fold] hold-out ses-{test_session} run-{test_run}')

        test_idx   = paradigm.index.get_level_values('session') == test_session
        test_idx  &= paradigm.index.get_level_values('run') == test_run
        train_idx  = ~test_idx

        train_paradigm = paradigm.loc[train_idx]
        test_paradigm  = paradigm.loc[test_idx]
        train_data     = data.loc[train_idx]
        test_data      = data.loc[test_idx]

        # ── fit encoding model ────────────────────────────────────────────────
        model = VonMisesPRF(allow_neg_amplitudes=True)
        fitter = ParameterFitter(model, train_data, train_paradigm)
        pars = fit_encoding_model(fitter, n_iterations)

        # ── voxel selection by training R² ────────────────────────────────────
        pred_train = model.predict(parameters=pars, paradigm=train_paradigm)
        r2_train = get_rsq(train_data, pred_train)

        if n_voxels == 0:
            sel = r2_train[r2_train > 0.0].index
        else:
            sel = r2_train.sort_values(ascending=False).index[:n_voxels]

        print(f'    {len(sel)} voxels selected  '
              f'(train R² ≥ {float(r2_train.loc[sel].min()):.3f})')

        pars_sel       = pars.loc[sel]
        train_data_sel = train_data[sel]
        test_data_sel  = test_data[sel]
        model.apply_mask(sel)

        # ── fit noise model ───────────────────────────────────────────────────
        model.init_pseudoWWT(stimulus_range, pars_sel)
        residfit = ResidualFitter(model, train_data_sel, train_paradigm,
                                  parameters=pars_sel)
        omega, dof = residfit.fit(
            init_sigma2=0.1, init_dof=10.0, method='t',
            learning_rate=0.05, spherical=spherical_noise,
            max_n_iterations=5000 if not debug else 100)
        print(f'    noise model: dof={float(dof):.1f}')

        # ── decode ────────────────────────────────────────────────────────────
        pdf = model.get_stimulus_pdf(test_data_sel, stimulus_range,
                                     pars_sel, omega=omega, dof=dof,
                                     normalize=False)
        # pdf: DataFrame (n_test_trials, n_orientations)
        pdf.columns = stimulus_range
        pdf.index = pd.MultiIndex.from_arrays([
            test_paradigm.index.get_level_values('session'),
            test_paradigm.index.get_level_values('run'),
            test_paradigm.index.get_level_values('trial_nr'),
            test_paradigm['x'].values,
        ], names=['session', 'run', 'trial_nr', 'true_orientation_rad'])

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
                        help='Top-N voxels by training R² (0 = all R²>0)')
    parser.add_argument('--mask', default=None,
                        help='Brain mask NIfTI (default: fmriprep brain mask)')
    parser.add_argument('--mask-desc', default='brain',
                        help='Short label for mask used in output filename')
    parser.add_argument('--n-iterations', type=int, default=1000)
    parser.add_argument('--spherical-noise', action='store_true',
                        help='Fit isotropic noise model instead of full covariance')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep-flair',
                        choices=['fmriprep', 'fmriprep-flair', 'fmriprep-noflair'])
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help='100 encoding + 100 noise iterations (fast test)')
    args = parser.parse_args()

    main(args.subject, sessions=args.sessions, n_voxels=args.n_voxels,
         mask=args.mask, mask_desc=args.mask_desc,
         n_iterations=args.n_iterations, spherical_noise=args.spherical_noise,
         bids_folder=args.bids_folder, fmriprep_deriv=args.fmriprep_deriv,
         smoothed=args.smoothed, debug=args.debug)
