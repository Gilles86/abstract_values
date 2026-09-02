#!/usr/bin/env python3
"""Leave-one-run-out cross-validation for aPRF encoding models.

Thin CLI wrapper around :func:`fit_pipeline.fit_one_model` — the
fitting logic itself lives in the model registry (see
``model_specs.py``).

For each fold (held-out run):
  1. Fit the chosen model on the remaining runs via
     :func:`fit_one_model` (which transparently handles warm-starts
     for fwhm-shift / fully-shifted).
  2. Predict the held-out trials, compute their cv-R² per voxel.
  3. Save the per-fold cv-R² NIfTI.

After all folds: save the cohort mean cv-R² NIfTI under
``derivatives/encoding_models/<spec.cv_out_subdir>/sub-<S>/func/``.

Examples:
    python fit_aprf_cv.py pil01
    python fit_aprf_cv.py pil01 --model session-shift
    python fit_aprf_cv.py pil01 --model fwhm-only-shift
    python fit_aprf_cv.py pil01 --model fwhm-shift
    python fit_aprf_cv.py pil01 --model fully-shifted --debug
    python fit_aprf_cv.py pil01 --model null
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.utils import get_rsq

from abstract_values.encoding_models.fit_pipeline import fit_one_model
from abstract_values.encoding_models.model_specs import (
    get_spec, list_model_types,
)
from abstract_values.utils.data import Subject, BIDS_FOLDER


def save_f32(img, path):
    nib.Nifti1Image(img.get_fdata().astype(np.float32),
                    img.affine).to_filename(str(path))


def get_paradigm(sub, sessions, needs_session: bool,
                 space: str = 'value'):
    """Paradigm DataFrame indexed by (session, run, trial).

    'x' is the objective CHF value, or — with ``space='orientation'`` — the
    gabor orientation in radians wrapped to [0, pi), for models fitted in
    orientation space. 'session' (when needed) is the 0-based session index
    float for SessionShifted* models.

    The MultiIndex carries the ACTUAL session number (e.g. 1, 2) so
    leave-one-run-out folds can be selected by session+run."""
    rows = []
    for ses_idx, session in enumerate(sorted(sessions)):
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            run_ev = events.loc[run].reset_index().sort_values('onset')
            for _, row in run_ev[run_ev['event_type'] == 'gabor'].iterrows():
                stim = (np.deg2rad(float(row['orientation'])) % np.pi
                        if space == 'orientation' else float(row['value']))
                rows.append({'session': session, 'run': run,
                              'x': stim,
                              'session_idx': float(ses_idx)})
    df = pd.DataFrame(rows).astype({'x': np.float32,
                                       'session_idx': np.float32})
    df.index = pd.MultiIndex.from_frame(
        df[['session', 'run']].assign(
            trial=df.groupby(['session', 'run']).cumcount()),
        names=['session', 'run', 'trial'])
    if needs_session:
        return df[['x', 'session_idx']].rename(
            columns={'session_idx': 'session'})
    return df[['x']]


def _predict_test_fold(spec, pars, test_paradigm, test_data):
    """Construct the model from `spec`, set its parameters to the
    fold's fitted values, predict the test paradigm, return cv-R²."""
    if spec.is_null:
        # Null model in cv mode: predict the *training* mean for each
        # voxel on the test fold. (For training R² we used the same
        # voxel mean and got 0; on held-out runs it will be ≤ 0 unless
        # train and test means happen to coincide.)
        train_mean = test_paradigm  # dummy; actual mean computed by caller
        raise NotImplementedError('handled in main()')
    model = spec.cls(**spec.cls_kwargs)
    pred_arr = model.predict(parameters=pars, paradigm=test_paradigm)
    pred = pd.DataFrame(pred_arr.values if hasattr(pred_arr, 'values')
                          else np.asarray(pred_arr),
                          index=test_data.index, columns=test_data.columns)
    return get_rsq(test_data, pred)


def main(subject, n_iterations=1000, mask=None,
         bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep',
         smoothed=False, debug=False, model_type='standard'):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder,
                   fmriprep_deriv=fmriprep_deriv)
    sessions = sorted(sub.get_sessions())
    spec = get_spec(model_type)

    if spec.needs_session and len(sessions) < 2:
        raise ValueError(f"--model {spec.name} requires at least 2 sessions.")

    print(f"sub-{subject}  all-sessions ({sessions})  "
          f"[abstract pRF CV  model={spec.name}]")
    if debug:
        n_iterations = 50

    # ── paradigm + betas + masker ────────────────────────────────────────────
    paradigm = get_paradigm(sub, sessions, spec.needs_session,
                            space=spec.stimulus_space)
    value_min = float(paradigm['x'].min())
    value_max = float(paradigm['x'].max())
    print(f"  {len(paradigm)} trials  value range: "
          f"{value_min:.1f}–{value_max:.1f} CHF")

    betas_img = sub.get_single_trial_estimates(sessions, desc='gabor',
                                                smoothed=smoothed)
    if mask is None:
        mask = sub.get_brain_mask(sessions[0])
    masker = NiftiMasker(mask_img=mask,
                          target_affine=betas_img.affine,
                          target_shape=betas_img.shape[:3]).fit()
    data = pd.DataFrame(masker.transform(betas_img).astype(np.float32),
                          index=paradigm.index)
    print(f"  {data.shape[1]} voxels in mask")

    smooth_label = '_smoothed' if smoothed else ''
    out_dir = (bids_folder / 'derivatives' / 'encoding_models'
                / spec.cv_out_subdir / f'sub-{subject}' / 'func')
    out_dir.mkdir(parents=True, exist_ok=True)
    fn_run  = (f"sub-{subject}_ses-{{ses}}_task-abstractvalue"
                f"_space-T1w_run-{{run}}_desc-cvr2{smooth_label}_pe.nii.gz")
    fn_mean = (f"sub-{subject}_task-abstractvalue"
                f"_space-T1w_desc-cvr2{smooth_label}_pe.nii.gz")

    folds = sorted(set(zip(
        paradigm.index.get_level_values('session'),
        paradigm.index.get_level_values('run'))))
    all_cvr2 = []

    paradigm_cols = ['x', 'session'] if spec.needs_session else ['x']

    for test_session, test_run in folds:
        print(f"\n  fold ses-{test_session} run-{test_run}:")
        test_mask = (paradigm.index.get_level_values('session') == test_session) & \
                    (paradigm.index.get_level_values('run')     == test_run)
        train_mask = ~test_mask

        train_paradigm = paradigm.loc[train_mask].reset_index(drop=True)[paradigm_cols]
        train_data     = data.loc[train_mask].reset_index(drop=True)
        test_paradigm  = paradigm.loc[test_mask].reset_index(drop=True)[paradigm_cols]
        test_data      = data.loc[test_mask].reset_index(drop=True)

        if spec.is_null:
            # Predict the training mean per voxel for the held-out trials.
            train_mean = train_data.mean(axis=0)
            test_pred = pd.DataFrame(
                np.tile(train_mean.values, (len(test_data), 1)),
                index=test_data.index, columns=test_data.columns)
            cv_r2 = get_rsq(test_data, test_pred)
        else:
            pars, _ = fit_one_model(
                spec, train_data, train_paradigm,
                value_min=value_min, value_max=value_max,
                n_iterations=n_iterations, debug=debug,
                log_prefix="    ")
            cv_r2 = _predict_test_fold(spec, pars, test_paradigm, test_data)
        print(f"    mean cv R² = {float(cv_r2.mean()):.4f}")

        masker.inverse_transform(cv_r2).to_filename(
            str(out_dir / fn_run.format(ses=test_session, run=test_run)))
        all_cvr2.append(cv_r2)

    mean_cvr2 = pd.concat(all_cvr2, axis=1).mean(axis=1)
    print(f"\n  mean cv R² (all folds) = {float(mean_cvr2.mean()):.4f}")
    masker.inverse_transform(mean_cvr2).to_filename(str(out_dir / fn_mean))
    print(f"  saved to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-'")
    parser.add_argument('--model', default='standard',
                         choices=list_model_types(),
                         help='Model variant (see module docstring).')
    parser.add_argument('--n-iterations', type=int, default=1000)
    parser.add_argument('--mask', default=None)
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep',
                         choices=['fmriprep', 'fmriprep-t2w'])
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args.subject, n_iterations=args.n_iterations, mask=args.mask,
         bids_folder=args.bids_folder, fmriprep_deriv=args.fmriprep_deriv,
         smoothed=args.smoothed, debug=args.debug, model_type=args.model)
