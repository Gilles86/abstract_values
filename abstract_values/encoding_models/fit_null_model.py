#!/usr/bin/env python3
"""
Fit a null (mean) encoding model and compute leave-one-run-out CV R².

The null model predicts every test trial as the mean response of the
training set for each voxel. This provides a baseline CV R² — any
encoding model should beat this to be considered informative.

Output
------
  derivatives/encoding_models/null/sub-<subject>/<ses_dir>/func/
    sub-<subject>[_ses-<N>]_task-abstractvalue_space-T1w_desc-cvr2[_smoothed]_pe.nii.gz

Usage
-----
  python fit_null_model.py pil01
  python fit_null_model.py pil01 --sessions 1
  python fit_null_model.py pil01 --smoothed
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.utils import get_rsq

from abstract_values.utils.data import Subject, BIDS_FOLDER


def main(subject, sessions=None, bids_folder=BIDS_FOLDER,
         fmriprep_deriv='fmriprep', smoothed=False):

    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)

    if sessions is None:
        sessions = sub.get_sessions()
    sessions = sorted(sessions)

    ses_dir    = f'ses-{sessions[0]}' if len(sessions) == 1 else ''
    ses_entity = f'_ses-{sessions[0]}' if len(sessions) == 1 else ''
    smooth_label = '_smoothed' if smoothed else ''
    print(f'sub-{subject}  {ses_dir or "all-sessions"}  [null model CV R²]')

    # ── load betas ───────────────────────────────────────────────────────────
    betas_img = sub.get_single_trial_estimates(sessions, desc='gabor',
                                               smoothed=smoothed)
    masker = NiftiMasker(mask_img=sub.get_brain_mask(sessions[0])).fit()
    data = pd.DataFrame(masker.transform(betas_img).astype(np.float32))
    print(f'  {data.shape[1]} voxels, {data.shape[0]} trials')

    # ── build run labels ─────────────────────────────────────────────────────
    run_labels = []
    for ses in sessions:
        runs = sub.get_runs(ses)
        events = sub.get_events(ses, runs)
        for run in runs:
            n = len(events.loc[run].reset_index()
                    .query("event_type == 'gabor'"))
            run_labels.extend([(ses, run)] * n)
    run_labels = pd.MultiIndex.from_tuples(run_labels, names=['session', 'run'])
    data.index = run_labels

    # ── leave-one-run-out CV ─────────────────────────────────────────────────
    all_runs = sorted(set(run_labels))
    fold_r2s = []

    for test_ses, test_run in all_runs:
        test_mask = ((data.index.get_level_values('session') == test_ses) &
                     (data.index.get_level_values('run') == test_run))
        train_data = data.loc[~test_mask]
        test_data  = data.loc[test_mask]

        # Predict training mean for every test trial
        train_mean = train_data.mean(axis=0)
        pred = pd.DataFrame(
            np.tile(train_mean.values, (len(test_data), 1)),
            index=test_data.index, columns=test_data.columns)

        fold_r2s.append(get_rsq(test_data, pred))
        print(f'  fold ses-{test_ses} run-{test_run}: '
              f'mean CV R² = {fold_r2s[-1].mean():.4f}')

    mean_cvr2 = pd.concat(fold_r2s, axis=1).mean(axis=1)
    print(f'\n  mean CV R² across folds: {mean_cvr2.mean():.4f}  '
          f'(median: {mean_cvr2.median():.4f})')

    # ── save ─────────────────────────────────────────────────────────────────
    out_dir = (bids_folder / 'derivatives' / 'encoding_models' / 'null'
               / f'sub-{subject}')
    if ses_dir:
        out_dir = out_dir / ses_dir
    out_dir = out_dir / 'func'
    out_dir.mkdir(parents=True, exist_ok=True)

    out_fn = (out_dir /
              f'sub-{subject}{ses_entity}_task-abstractvalue'
              f'_space-T1w_desc-cvr2{smooth_label}_pe.nii.gz')

    out_img = masker.inverse_transform(mean_cvr2.values)
    out_img.to_filename(str(out_fn))
    print(f'  saved to {out_fn}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-'")
    parser.add_argument('--sessions', type=int, nargs='+', default=None)
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep',
                        choices=['fmriprep', 'fmriprep-t2w'])
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()

    main(args.subject, sessions=args.sessions,
         bids_folder=args.bids_folder, fmriprep_deriv=args.fmriprep_deriv,
         smoothed=args.smoothed)
