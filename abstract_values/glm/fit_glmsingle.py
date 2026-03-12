#!/usr/bin/env python3
"""
Fit GLMsingle single-trial betas for the abstract values fMRI task.

Models two events per trial:
  gabor_{trial_nr}    — onset of the oriented Gabor stimulus
  response_{trial_nr} — onset of the response slider (response_bar phase)

Usage:
    python fit_glmsingle.py pil01 1
    python fit_glmsingle.py 01 1 --fmriprep-deriv fmriprep-noflair
    python fit_glmsingle.py 01 1 --bids-folder /data/ds-abstractvalue
"""

import argparse
import warnings

import numpy as np
from nilearn import image

from abstract_values.utils.data import Subject, BIDS_FOLDER

warnings.filterwarnings('ignore')

TR = 0.996


def build_design_matrix(events_run, n_vols):
    """Return (dm, trial_types) for one run.

    dm          : binary (n_vols × n_events) array, one 1 per column at the nearest TR
    trial_types : list of str labels, e.g. ['gabor_1', 'response_1', ...]
    """
    ev = events_run.reset_index().copy()
    ev['trial_type'] = ev.apply(
        lambda row: f'gabor_{int(row["trial_nr"])}'
                    if row['event_type'] == 'gabor'
                    else f'response_{int(row["trial_nr"])}',
        axis=1,
    )
    dm = np.zeros((n_vols, len(ev)))
    for col, (_, row) in enumerate(ev.iterrows()):
        onset_tr = int(np.round(row['onset'] / TR))
        dm[min(onset_tr, n_vols - 1), col] = 1.0
    return dm, ev['trial_type'].tolist()


def main(subject, session, bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep-flair'):
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)

    runs  = sub.get_runs(session)
    bold  = sub.get_preprocessed_bold(session, runs)
    events = sub.get_events(session, runs)

    print(f'sub-{subject}  ses-{session}  [{fmriprep_deriv}]')
    print(f'  {len(runs)} runs: {runs}')

    data = [image.load_img(str(f)).get_fdata() for f in bold]

    X = []
    all_trial_types = []
    for run, d in zip(runs, data):
        n_vols = d.shape[3]
        dm, trial_types = build_design_matrix(events.loc[run], n_vols)
        X.append(dm)
        all_trial_types.extend(trial_types)
        print(f'  run-{run}: {n_vols} volumes, {dm.shape[1]} regressors')

    opt = dict(
        wantlibrary=1,
        wantglmdenoise=1,
        wantfracridge=1,
        wantfileoutputs=[0, 0, 0, 1],
        sessionindicator=np.array([session] * len(runs))[np.newaxis, :],
    )

    from pathlib import Path
    out_dir = (Path(bids_folder) / 'derivatives' / 'glmsingle' / fmriprep_deriv
               / f'sub-{subject}' / f'ses-{session}' / 'func')
    out_dir.mkdir(parents=True, exist_ok=True)

    from glmsingle.glmsingle import GLM_single
    results = GLM_single(opt).fit(X, data, TR, TR, outputdir=str(out_dir))

    betas = results['typed']['betasmd']  # (x, y, z, n_total_trials)
    gabor_idx    = [i for i, t in enumerate(all_trial_types) if t.startswith('gabor')]
    response_idx = [i for i, t in enumerate(all_trial_types) if t.startswith('response')]

    fn = (f'sub-{subject}_ses-{session}_task-abstractvalue'
          f'_space-T1w_desc-{{desc}}_pe.nii.gz')
    image.new_img_like(bold[0], betas[..., gabor_idx]).to_filename(
        str(out_dir / fn.format(desc='gabor')))
    image.new_img_like(bold[0], betas[..., response_idx]).to_filename(
        str(out_dir / fn.format(desc='response')))
    image.new_img_like(bold[0], results['typed']['R2']).to_filename(
        str(out_dir / fn.format(desc='R2')))

    print(f'Saved to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-', e.g. pil01 or 01")
    parser.add_argument('session', type=int)
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep-flair',
                        choices=['fmriprep', 'fmriprep-flair', 'fmriprep-noflair'])
    args = parser.parse_args()

    main(args.subject, args.session,
         bids_folder=args.bids_folder,
         fmriprep_deriv=args.fmriprep_deriv)
