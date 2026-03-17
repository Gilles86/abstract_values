#!/usr/bin/env python3
"""
Fit GLMsingle single-trial betas for the abstract values fMRI task.

Design matrix
-------------
Two events are modelled per trial:

  orientation_<angle>   — TR nearest the Gabor stimulus onset; labelled by the
                          orientation of the Gabor patch (e.g. orientation_90.0)
  response_<bid>        — TR nearest the response-bar onset; labelled by the
                          participant's bid on that trial (e.g. response_22.5),
                          taken from the feedback event. For non-responses
                          (NaN bid) the label falls back to response_<angle>.

Each unique orientation and each unique response value gets its own column in
the design matrix, shared consistently across all runs and sessions.  This is
critical for GLMsingle's fracridge cross-validation: GLMsingle identifies
conditions by design-matrix column index, so assigning the same orientation the
same column in every run ensures that same-orientation trials are grouped
together when selecting the ridge regularisation parameter alpha via
leave-one-run-out cross-validation.

GLMsingle then expands the condition-level design matrix internally to a
single-trial design, yielding one beta per trial presentation per voxel.

Output
------
Three 4-D NIfTI images are saved (x × y × z × n_trials):

  desc-gabor_pe.nii.gz    — single-trial betas for the Gabor stimulus phase
  desc-response_pe.nii.gz — single-trial betas for the response phase
  desc-R2_pe.nii.gz       — cross-validated R² of the final GLMsingle model

Output is written to:
  derivatives/glmsingle/<fmriprep_deriv>/sub-<subject>/<ses_label>/func/

where <ses_label> is ses-<N> for a single session or ses-all for a joint fit.

Sessions
--------
By default all available sessions are fitted jointly (recommended: GLMsingle
uses the session indicator to handle session-to-session scaling differences).
Pass --sessions to restrict to specific session numbers.

Usage:
    python fit_glmsingle.py pil01
    python fit_glmsingle.py pil01 --sessions 1
    python fit_glmsingle.py pil01 --sessions 1 2
    python fit_glmsingle.py 01 --sessions 1 --fmriprep-deriv fmriprep-noflair
    python fit_glmsingle.py 01 --bids-folder /data/ds-abstractvalue
    python fit_glmsingle.py pil01 --debug   # write all 4 model steps + figures
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from nilearn import image

from abstract_values.utils.data import Subject, BIDS_FOLDER

warnings.filterwarnings('ignore')

TR = 0.996


def make_condition_label(row):
    """Map a single event row to its condition label.

    Gabor events are labelled by orientation (orientation_<angle>).
    Response-bar events are labelled by the participant's bid (response_<bid>),
    which is joined in from the feedback event by get_events(). For non-
    responses (bid is NaN) we fall back to orientation so there are no
    degenerate response_nan conditions.
    """
    if row['event_type'] == 'gabor':
        return f'orientation_{row["orientation"]}'
    bid = row['bid']
    if pd.isna(bid):
        return f'response_{row["orientation"]}'   # fallback for non-responses
    return f'response_{bid}'


def build_condition_index(all_events):
    """Return a global {condition_label: column_index} mapping.

    Must be shared across all runs so that the same orientation/response value
    always activates the same design-matrix column — which is what GLMsingle
    uses as the condition ID for fracridge cross-validation.

    Parameters
    ----------
    all_events : iterable of per-run DataFrames (reset index, event_type column)
    """
    conditions = set()
    for ev in all_events:
        for _, row in ev.iterrows():
            conditions.add(make_condition_label(row))

    def _key(c):
        prefix, val = c.split('_', 1)
        return (prefix, float(val))

    return {c: i for i, c in enumerate(sorted(conditions, key=_key))}


def build_design_matrix(events_run, n_vols, condition_to_idx):
    """Return (dm, trial_order) for one run.

    dm          : binary (n_vols × n_conditions) array, one 1 per row at the nearest TR
    trial_order : condition labels in onset-time order — one entry per event,
                  matching the order GLMsingle assigns single-trial betas
    """
    ev = events_run.reset_index().copy()
    ev['condition'] = ev.apply(make_condition_label, axis=1)
    ev = ev.sort_values('onset')

    dm = np.zeros((n_vols, len(condition_to_idx)))
    trial_order = []
    for _, row in ev.iterrows():
        onset_tr = int(np.round(row['onset'] / TR))
        col = condition_to_idx[row['condition']]
        dm[min(onset_tr, n_vols - 1), col] = 1.0
        trial_order.append(row['condition'])
    return dm, trial_order


def main(subject, sessions=None, bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep-flair',
         debug=False, smoothed=False):
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)

    if sessions is None:
        sessions = sub.get_sessions()

    ses_label = f'ses-{sessions[0]}' if len(sessions) == 1 else 'ses-all'
    print(f'sub-{subject}  {ses_label}  [{fmriprep_deriv}]')

    # First pass: collect all events to build a globally consistent condition map.
    # GLMsingle uses column index as condition ID, so the same orientation/response
    # value must always occupy the same column across all runs and sessions.
    session_run_events = {}
    for session in sessions:
        runs   = sub.get_runs(session)
        events = sub.get_events(session, runs)
        session_run_events[session] = (runs, events)

    condition_to_idx = build_condition_index(
        events.loc[run].reset_index()
        for session, (runs, events) in session_run_events.items()
        for run in runs
    )
    n_orientations = sum(1 for c in condition_to_idx if c.startswith('orientation'))
    n_responses    = sum(1 for c in condition_to_idx if c.startswith('response'))
    print(f'  {len(condition_to_idx)} conditions: '
          f'{n_orientations} orientations, {n_responses} response values')

    # Second pass: load BOLD and build design matrices.
    data = []
    X = []
    all_trial_types = []
    session_indicators = []
    ref_bold = None

    for session in sessions:
        runs, events = session_run_events[session]
        bold = sub.get_preprocessed_bold(session, runs)
        print(f'  ses-{session}: {len(runs)} runs: {runs}')

        for run, bold_path in zip(runs, bold):
            if ref_bold is None:
                ref_bold = bold_path
            img = image.smooth_img(str(bold_path), fwhm=5.0).get_fdata() if smoothed \
                else image.load_img(str(bold_path)).get_fdata()
            n_vols = img.shape[3]
            dm, trial_order = build_design_matrix(events.loc[run], n_vols, condition_to_idx)
            data.append(img)
            X.append(dm)
            all_trial_types.extend(trial_order)
            session_indicators.append(session)
            print(f'    run-{run}: {n_vols} volumes, {len(trial_order)} trials')

    opt = dict(
        wantlibrary=1,
        wantglmdenoise=1,
        wantfracridge=1,
        wantfileoutputs=[1, 1, 1, 1] if debug else [0, 0, 0, 1],
        sessionindicator=np.array(session_indicators)[np.newaxis, :],
    )

    from pathlib import Path
    glmsingle_deriv = 'glmsingle.smoothed' if smoothed else 'glmsingle'
    out_dir = (Path(bids_folder) / 'derivatives' / glmsingle_deriv / fmriprep_deriv
               / f'sub-{subject}' / ses_label / 'func')
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = out_dir.parent / 'figures'

    from glmsingle.glmsingle import GLM_single
    results = GLM_single(opt).fit(X, data, TR, TR,
                                  outputdir=str(out_dir),
                                  figuredir=str(fig_dir))

    betas = results['typed']['betasmd']  # (x, y, z, n_total_trials)
    gabor_idx    = [i for i, t in enumerate(all_trial_types) if t.startswith('orientation')]
    response_idx = [i for i, t in enumerate(all_trial_types) if t.startswith('response')]

    fn = (f'sub-{subject}_{ses_label}_task-abstractvalue'
          f'_space-T1w_desc-{{desc}}_pe.nii.gz')
    image.new_img_like(ref_bold, betas[..., gabor_idx]).to_filename(
        str(out_dir / fn.format(desc='gabor')))
    image.new_img_like(ref_bold, betas[..., response_idx]).to_filename(
        str(out_dir / fn.format(desc='response')))
    image.new_img_like(ref_bold, results['typed']['R2']).to_filename(
        str(out_dir / fn.format(desc='R2')))

    print(f'Saved to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-', e.g. pil01 or 01")
    parser.add_argument('--sessions', type=int, nargs='+', default=None,
                        help='Session number(s) to fit. Default: all available sessions.')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep-flair',
                        choices=['fmriprep', 'fmriprep-flair', 'fmriprep-noflair',
                                 'fmriprep-t2w'])
    parser.add_argument('--debug', action='store_true',
                        help='Write outputs and diagnostic figures for all 4 GLMsingle steps.')
    parser.add_argument('--smoothed', action='store_true',
                        help='Spatially smooth BOLD data with a 5 mm FWHM Gaussian kernel '
                             'before fitting. Outputs go to derivatives/glmsingle.smoothed/.')
    args = parser.parse_args()

    main(args.subject, sessions=args.sessions,
         bids_folder=args.bids_folder,
         fmriprep_deriv=args.fmriprep_deriv,
         debug=args.debug,
         smoothed=args.smoothed)
