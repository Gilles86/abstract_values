#!/usr/bin/env python3
"""
Fit a Von Mises basis set encoding model to single-trial GLMsingle betas.

Models orientation tuning with N linearly spaced Von Mises basis functions.
The basis function parameters (mu, kappa) are fixed; per-voxel weights are
solved in closed form using braincoder.optimize.WeightFitter (lstsq).

Basis functions
---------------
  N Von Mises RFs with mus at np.linspace(0, π, N, endpoint=False)
  (defaults: N=8, kappa=2.0).

  The stimulus (gabor orientation) is converted from degrees to radians.
  Because orientation is π-periodic, the range 0–π covers the full cycle.

Output
------
  derivatives/encoding_models/vonmises/sub-<subject>/<ses_label>/func/
    sub-<subject>_<ses_label>_task-abstractvalue_space-T1w_desc-weights_pe.nii.gz
      4D image — one volume per basis function, each volume = per-voxel weight
    sub-<subject>_<ses_label>_task-abstractvalue_space-T1w_desc-r2_pe.nii.gz
      R² of the model fit

Usage
-----
  python fit_vonmises_model.py pil01 --sessions 1
  python fit_vonmises_model.py pil01 --sessions 1 --kappa 4.0
  python fit_vonmises_model.py pil01 --sessions 1 --n-basis 16
  python fit_vonmises_model.py pil01 --sessions 1 --mask /path/to/mask.nii.gz
  python fit_vonmises_model.py pil01 --sessions 1 --smoothed
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image
from nilearn.maskers import NiftiMasker

from braincoder.models import VonMisesPRF
from braincoder.optimize import WeightFitter
from braincoder.utils import get_rsq

from abstract_values.utils.data import Subject, BIDS_FOLDER


def get_gabor_paradigm(sub, sessions):
    """Return DataFrame with column 'x' (orientation in radians, float32).

    Rows are in the same order as the gabor betas written by fit_glmsingle:
    for each session → run (sorted) → event sorted by onset, gabor only.
    """
    rows = []
    for session in sessions:
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            run_ev = events.loc[run].reset_index().sort_values('onset')
            for _, row in run_ev[run_ev['event_type'] == 'gabor'].iterrows():
                rows.append(np.deg2rad(float(row['orientation'])))
    return pd.DataFrame({'x': np.array(rows, dtype=np.float32)})


def make_basis_parameters(n_basis, kappa):
    """Fixed parameters for n_basis Von Mises basis functions (amplitude=1, baseline=0)."""
    mus = np.linspace(0, np.pi, n_basis, endpoint=False).astype(np.float32)
    return pd.DataFrame({
        'mu':        mus,
        'kappa':     np.full(n_basis, kappa, dtype=np.float32),
        'amplitude': np.ones(n_basis,  dtype=np.float32),
        'baseline':  np.zeros(n_basis, dtype=np.float32),
    })


def main(subject, sessions=None, n_basis=8, kappa=2.0, mask=None,
         bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep-flair',
         smoothed=False):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)

    if sessions is None:
        sessions = sub.get_sessions()

    ses_label = f'ses-{sessions[0]}' if len(sessions) == 1 else 'ses-all'
    print(f'sub-{subject}  {ses_label}  n_basis={n_basis}  kappa={kappa}')

    # ── paradigm ─────────────────────────────────────────────────────────────
    paradigm = get_gabor_paradigm(sub, sessions)
    print(f'  {len(paradigm)} gabor trials')

    # ── betas ─────────────────────────────────────────────────────────────────
    betas_img = sub.get_single_trial_estimates(sessions, desc='gabor',
                                               smoothed=smoothed)
    assert betas_img.shape[3] == len(paradigm), (
        f'Beta count mismatch: {betas_img.shape[3]} betas vs {len(paradigm)} trials')

    # ── masker ────────────────────────────────────────────────────────────────
    if mask is None:
        mask = sub.get_brain_mask(sessions[0])
    masker = NiftiMasker(mask_img=mask).fit()
    data = pd.DataFrame(masker.transform(betas_img).astype(np.float32))
    print(f'  {data.shape[1]} voxels in mask')

    # ── basis parameters ──────────────────────────────────────────────────────
    basis_pars = make_basis_parameters(n_basis, kappa)
    print(f'  basis mus (deg): {np.rad2deg(basis_pars["mu"].values).round(1).tolist()}')

    # ── fit weights ───────────────────────────────────────────────────────────
    model = VonMisesPRF()
    weights = WeightFitter(model, basis_pars, data, paradigm).fit()
    # weights: DataFrame (n_basis, n_voxels)

    # ── R² ────────────────────────────────────────────────────────────────────
    basis_pred = model.basis_predictions(paradigm, basis_pars)  # (n_trials, n_basis)
    pred = pd.DataFrame(basis_pred @ weights.values,
                        index=data.index, columns=data.columns)
    r2 = get_rsq(data, pred)
    print(f'  mean R²={float(r2.mean()):.4f}')

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = (bids_folder / 'derivatives' / 'encoding_models' / 'vonmises'
               / f'sub-{subject}' / ses_label / 'func')
    out_dir.mkdir(parents=True, exist_ok=True)

    fn = (f'sub-{subject}_{ses_label}_task-abstractvalue'
          f'_space-T1w_desc-{{desc}}_pe.nii.gz')

    # 4D weights image: volume i = weights for basis function i
    weights_img = image.concat_imgs(
        [masker.inverse_transform(weights.loc[i]) for i in range(n_basis)])
    weights_img.to_filename(str(out_dir / fn.format(desc='weights')))

    masker.inverse_transform(r2).to_filename(str(out_dir / fn.format(desc='r2')))

    print(f'  saved to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-'")
    parser.add_argument('--sessions', type=int, nargs='+', default=None)
    parser.add_argument('--n-basis', type=int, default=8,
                        help='Number of Von Mises basis functions (default: 8)')
    parser.add_argument('--kappa', type=float, default=2.0,
                        help='Von Mises concentration parameter (default: 2.0)')
    parser.add_argument('--mask', default=None,
                        help='Brain mask NIfTI (default: fmriprep brain mask)')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep-flair',
                        choices=['fmriprep', 'fmriprep-flair', 'fmriprep-noflair'])
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()

    main(args.subject, sessions=args.sessions, n_basis=args.n_basis,
         kappa=args.kappa, mask=args.mask, bids_folder=args.bids_folder,
         fmriprep_deriv=args.fmriprep_deriv, smoothed=args.smoothed)
