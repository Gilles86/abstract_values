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
  Always fitted jointly across all of a subject's sessions; no per-session
  output path.

  derivatives/encoding_models/vonmises/sub-<subject>/func/
    sub-<subject>_task-abstractvalue_space-T1w_desc-weights_pe.nii.gz
      4D image — one volume per basis function, each volume = per-voxel weight
    sub-<subject>_task-abstractvalue_space-T1w_desc-r2_pe.nii.gz
      R² of the model fit

Usage
-----
  python fit_vonmises_model.py pil01
  python fit_vonmises_model.py pil01 --kappa 4.0
  python fit_vonmises_model.py pil01 --n-basis 16
  python fit_vonmises_model.py pil01 --mask /path/to/mask.nii.gz
  python fit_vonmises_model.py pil01 --smoothed
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image
from nilearn.maskers import NiftiMasker

from braincoder.models import AxialVonMisesPRF
from braincoder.optimize import WeightFitter
from braincoder.utils import get_rsq

from abstract_values.utils.data import Subject, BIDS_FOLDER


def get_gabor_paradigm(sub, sessions):
    """Return DataFrame with columns ['x', 'session'] (orientation in
    radians, session index matching the position in `sessions`).

    Rows are in the same order as the gabor betas written by fit_glmsingle:
    for each session → run (sorted) → event sorted by onset, gabor only.
    The session column lets downstream code fit per-session weights
    without re-globbing events.
    """
    rows = []
    for ses_idx, session in enumerate(sessions):
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            run_ev = events.loc[run].reset_index().sort_values('onset')
            for _, row in run_ev[run_ev['event_type'] == 'gabor'].iterrows():
                rows.append((np.deg2rad(float(row['orientation'])), ses_idx))
    arr = np.asarray(rows, dtype=np.float32)
    return pd.DataFrame({'x': arr[:, 0], 'session': arr[:, 1]})


def make_basis_parameters(n_basis, kappa):
    """Fixed parameters for n_basis Von Mises basis functions (amplitude=1, baseline=0)."""
    mus = np.linspace(0, np.pi, n_basis, endpoint=False).astype(np.float32)
    return pd.DataFrame({
        'mu':        mus,
        'kappa':     np.full(n_basis, kappa, dtype=np.float32),
        'amplitude': np.ones(n_basis,  dtype=np.float32),
        'baseline':  np.zeros(n_basis, dtype=np.float32),
    })


def _fit_weights_one_session(model, basis_pars, data_ses, paradigm_ses):
    """Closed-form lstsq for the 8 basis weights on a single session's
    trials. Returns weights DataFrame (n_basis × n_voxels) AND R²."""
    weights = WeightFitter(model, basis_pars, data_ses, paradigm_ses).fit()
    basis_pred = model.basis_predictions(paradigm_ses, basis_pars)
    pred = pd.DataFrame(basis_pred @ weights.values,
                         index=data_ses.index, columns=data_ses.columns)
    r2 = get_rsq(data_ses, pred)
    return weights, r2


def main(subject, n_basis=8, kappa=2.0, mask=None,
         bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep',
         smoothed=False, session_shift=False):
    """When ``session_shift=True``, fit the 8 basis weights *separately*
    per session via closed-form lstsq, write them to
    ``derivatives/encoding_models/vonmises-session-shift/`` along with
    a per-session R² map. The default (joint) fit pools all sessions
    and writes to ``vonmises/`` as before."""
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)

    sessions = sorted(sub.get_sessions())

    if session_shift and len(sessions) < 2:
        raise ValueError("--session-shift requires at least 2 sessions")

    mode_label = "session-shift" if session_shift else "joint"
    print(f'sub-{subject}  all-sessions ({sessions})  '
          f'n_basis={n_basis}  kappa={kappa}  mode={mode_label}')

    # ── paradigm ─────────────────────────────────────────────────────────────
    paradigm = get_gabor_paradigm(sub, sessions)
    print(f'  {len(paradigm)} gabor trials  '
          f'(per-session: {paradigm.groupby("session").size().to_dict()})')

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

    model = AxialVonMisesPRF()
    smooth_label = '_smoothed' if smoothed else ''

    if not session_shift:
        # ── joint fit (legacy behaviour) ─────────────────────────────────────
        # Drop the session column from the paradigm — the joint model
        # doesn't read it.
        paradigm_joint = paradigm[['x']].reset_index(drop=True)
        weights, r2 = _fit_weights_one_session(
            model, basis_pars, data, paradigm_joint)
        print(f'  mean R²={float(r2.mean()):.4f}')

        out_dir = (bids_folder / 'derivatives' / 'encoding_models' / 'vonmises'
                   / f'sub-{subject}' / 'func')
        out_dir.mkdir(parents=True, exist_ok=True)
        fn = (f'sub-{subject}_task-abstractvalue'
              f'_space-T1w_desc-{{desc}}{smooth_label}_pe.nii.gz')
        weights_img = image.concat_imgs(
            [masker.inverse_transform(weights.loc[i]) for i in range(n_basis)])
        weights_img.to_filename(str(out_dir / fn.format(desc='weights')))
        masker.inverse_transform(r2).to_filename(
            str(out_dir / fn.format(desc='r2')))
        print(f'  saved to {out_dir}')
        return

    # ── session-shift: per-session weights via per-session lstsq ──────────────
    # Each session gets its own 8-channel weight vector per voxel.
    # Output layout mirrors aprf-session-shift: outputs in the
    # 'vonmises-session-shift' sibling dir with per-session weight files.
    out_dir = (bids_folder / 'derivatives' / 'encoding_models'
               / 'vonmises-session-shift' / f'sub-{subject}' / 'func')
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = (f'sub-{subject}_task-abstractvalue'
          f'_space-T1w_desc-{{desc}}{smooth_label}_pe.nii.gz')

    # We also report a joint R² computed by stitching per-session preds
    # back together, so it's directly comparable to the joint-fit R².
    per_session_pred = pd.DataFrame(np.nan,
                                    index=data.index, columns=data.columns)
    for ses_idx, ses in enumerate(sessions):
        ses_mask = paradigm['session'].values == ses_idx
        data_ses     = data.iloc[ses_mask].reset_index(drop=True)
        paradigm_ses = paradigm.loc[ses_mask, ['x']].reset_index(drop=True)
        if len(data_ses) == 0:
            print(f'  session {ses}: no trials, skipping')
            continue
        weights, r2 = _fit_weights_one_session(
            model, basis_pars, data_ses, paradigm_ses)
        print(f'  session {ses}: mean R²={float(r2.mean()):.4f}  '
              f'({len(data_ses)} trials)')

        # Save per-session weights as 4-D volume + per-session R²
        weights_img = image.concat_imgs(
            [masker.inverse_transform(weights.loc[i]) for i in range(n_basis)])
        weights_img.to_filename(
            str(out_dir / fn.format(desc=f'weights_{ses_idx + 1}')))
        masker.inverse_transform(r2).to_filename(
            str(out_dir / fn.format(desc=f'r2_{ses_idx + 1}')))

        # Stitch this session's predictions back into the joint design
        basis_pred = model.basis_predictions(paradigm_ses, basis_pars)
        ses_pred = pd.DataFrame(basis_pred @ weights.values,
                                  index=data_ses.index,
                                  columns=data_ses.columns)
        per_session_pred.iloc[ses_mask] = ses_pred.values

    # Joint R² (stitched) — what you'd report as "this model's overall fit"
    # for nested-model cvR² comparison against the joint vonmises fit.
    valid = ~per_session_pred.isna().any(axis=1)
    r2_joint = get_rsq(data[valid], per_session_pred[valid])
    print(f'  joint R² (stitched) ={float(r2_joint.mean()):.4f}')
    masker.inverse_transform(r2_joint).to_filename(
        str(out_dir / fn.format(desc='r2')))
    print(f'  saved to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-'")
    parser.add_argument('--n-basis', type=int, default=8,
                        help='Number of Von Mises basis functions (default: 8)')
    parser.add_argument('--kappa', type=float, default=2.0,
                        help='Von Mises concentration parameter (default: 2.0)')
    parser.add_argument('--mask', default=None,
                        help='Brain mask NIfTI (default: fmriprep brain mask)')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep',
                        choices=['fmriprep', 'fmriprep-t2w'])
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--session-shift', action='store_true',
                        help="Fit per-session basis weights (output: "
                             "vonmises-session-shift). Default: joint fit.")
    args = parser.parse_args()

    main(args.subject, n_basis=args.n_basis,
         kappa=args.kappa, mask=args.mask, bids_folder=args.bids_folder,
         fmriprep_deriv=args.fmriprep_deriv, smoothed=args.smoothed,
         session_shift=args.session_shift)
