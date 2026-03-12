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
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image
from nilearn.glm.first_level import make_first_level_design_matrix
from glmsingle.glmsingle import GLM_single

warnings.filterwarnings('ignore')

TR = 0.996
BIDS_FOLDER = Path('/data/ds-abstractvalue')


def get_bold_files(subject, session, derivatives):
    func_dir = derivatives / f'sub-{subject}' / f'ses-{session}' / 'func'
    pattern = f'sub-{subject}_ses-{session}_task-abstractvalue_run-*_space-T1w_*desc-preproc_bold.nii.gz'
    files = sorted(func_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No preprocessed BOLD found in {func_dir}')
    return files


def get_run_number(path):
    return int(re.search(r'run-(\d+)', path.name).group(1))


def load_events(subject, session, run, bids_folder):
    behavior_dir = bids_folder / 'sourcedata' / 'behavior' / f'sub-{subject}' / f'ses-{session}'
    candidates = sorted(behavior_dir.glob(f'*_run-{run:02d}_task-estimate.*_events.tsv'))
    if not candidates:
        raise FileNotFoundError(f'No events file for sub-{subject} ses-{session} run-{run:02d}')
    return pd.read_csv(candidates[0], sep='\t')


def build_design_matrix(events_run, n_vols):
    """Return a (n_vols x n_trials*2) binary design matrix for one run."""
    ev = events_run[events_run['event_type'].isin(['gabor', 'response_bar'])].copy()
    ev['trial_type'] = ev.apply(
        lambda row: f'gabor_{int(row["trial_nr"])}'
                    if row['event_type'] == 'gabor'
                    else f'response_{int(row["trial_nr"])}',
        axis=1,
    )
    ev['duration'] = 0.0
    # Snap onsets to TR grid
    ev['onset'] = ((ev['onset'] + TR / 2.) // TR) * TR

    frametimes = np.linspace(TR / 2., (n_vols - .5) * TR, n_vols)
    dm = make_first_level_design_matrix(
        frametimes,
        ev[['onset', 'trial_type', 'duration']],
        hrf_model='fir',
        drift_model=None,
        drift_order=0,
    ).drop('constant', axis=1)
    dm.columns = [c.replace('_delay_0', '') for c in dm.columns]
    dm /= dm.max()
    dm = np.round(dm)
    return dm


def main(subject, session, bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep-flair'):
    bids_folder = Path(bids_folder)
    derivatives = bids_folder / 'derivatives'

    bold_files = get_bold_files(subject, session, derivatives / fmriprep_deriv)
    runs = [get_run_number(f) for f in bold_files]

    print(f'Found {len(bold_files)} BOLD runs: {runs}')

    data = [image.load_img(f).get_fdata() for f in bold_files]

    design_matrices = []
    for run, d in zip(runs, data):
        n_vols = d.shape[3]
        ev = load_events(subject, session, run, bids_folder)
        dm = build_design_matrix(ev, n_vols)
        design_matrices.append(dm)
        print(f'  run-{run}: {n_vols} volumes, {dm.shape[1]} regressors')

    X = [dm.values for dm in design_matrices]

    opt = dict(
        wantlibrary=1,
        wantglmdenoise=1,
        wantfracridge=1,
        wantfileoutputs=[0, 0, 0, 1],
        sessionindicator=np.array([session] * len(runs))[np.newaxis, :],
    )

    out_dir = (derivatives / 'glmsingle' / fmriprep_deriv
               / f'sub-{subject}' / f'ses-{session}' / 'func')
    out_dir.mkdir(parents=True, exist_ok=True)

    glmsingle_obj = GLM_single(opt)
    results = glmsingle_obj.fit(X, data, TR, TR, outputdir=str(out_dir))

    betas = image.new_img_like(bold_files[0], results['typed']['betasmd'])
    r2    = image.new_img_like(bold_files[0], results['typed']['R2'])

    fn = f'sub-{subject}_ses-{session}_task-abstractvalue_space-T1w_desc-{{desc}}_pe.nii.gz'
    betas.to_filename(str(out_dir / fn.format(desc='betas')))
    r2.to_filename(str(out_dir / fn.format(desc='R2')))

    print(f'Saved to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-', e.g. pil01 or 01")
    parser.add_argument('session', type=int)
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep-flair',
                        choices=['fmriprep', 'fmriprep-flair', 'fmriprep-noflair'],
                        help='Which fmriprep derivative to use')
    args = parser.parse_args()

    main(args.subject, args.session,
         bids_folder=args.bids_folder,
         fmriprep_deriv=args.fmriprep_deriv)
