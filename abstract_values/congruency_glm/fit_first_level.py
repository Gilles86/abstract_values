#!/usr/bin/env python3
"""Congruent/incongruent value-mapping first-level GLM (nilearn).

Fits one of three model variants per subject, across all sessions/runs
jointly (design matrices differ per run, all fed to one FirstLevelModel.fit()
call so the contrast is estimated once across the whole subject):

  congruent    -- value regressor under each session's TRUE/active mapping only
  incongruent  -- value regressor under the OTHER (counterfactual) mapping only
  both         -- both regressors together; congruent-minus-incongruent and
                  their sum (mapping-agnostic value) as extra contrasts

`congruent`/`incongruent` alone are the well-powered tests (no collinear
partner regressor in the design matrix); `both`'s congruent-minus-incongruent
contrast is intentionally weaker (the two value regressors are highly
correlated, r~0.93, at matched orientations) but is the direct congruency
test. See abstract_values/congruency_glm/build_events.py for regressor
construction.

Confounds: 6 motion params + top 5 aCompCor components (a_comp_cor_00-04).
Drift: nilearn's own cosine high-pass (0.01 Hz) -- NOT fmriprep's cosine
confounds, to avoid double-modeling near-identical low-frequency regressors.

Usage
-----
  python -m abstract_values.congruency_glm.fit_first_level 06 --model congruent
  python -m abstract_values.congruency_glm.fit_first_level 06 --model all
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix

from abstract_values.congruency_glm.build_events import build_subject_events
from abstract_values.utils.data import BIDS_FOLDER, Subject

TR = 0.996
MOTION_COLUMNS = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
ACOMPCOR_COLUMNS = [f'a_comp_cor_{i:02d}' for i in range(5)]

CONTRASTS = {
    'congruent': {'value_congruent': 'value_congruent'},
    'incongruent': {'value_incongruent': 'value_incongruent'},
    'both': {
        'value_congruent': 'value_congruent',
        'value_incongruent': 'value_incongruent',
        'congruent_minus_incongruent': 'value_congruent - value_incongruent',
        'congruent_plus_incongruent': 'value_congruent + value_incongruent',
    },
}


def get_confounds_for_run(sub, session, run):
    conf = sub.get_confounds(session, [run],
                             columns=tuple(MOTION_COLUMNS + ACOMPCOR_COLUMNS))
    conf = conf.loc[(run,)].reset_index(drop=True)
    return conf.fillna(0.0)


def fit_subject(subject, model, bids_folder=BIDS_FOLDER, out_root=None,
                smoothing_fwhm=None):
    sub = Subject(subject, bids_folder=bids_folder)
    sub.require_complete_sessions()

    events_list, keys = build_subject_events(sub, model)

    bold_paths, design_matrices = [], []
    # Same subject, T1w space grid -> one mask (from the first run) for the
    # whole FirstLevelModel fit rather than per-run masks.
    mask_img = sub.get_brain_mask(keys[0][0], keys[0][1])

    for (session, run), events in zip(keys, events_list):
        bold = sub.get_preprocessed_bold(session, [run])[0]
        n_vols = int(nib.load(str(bold)).shape[-1])
        frame_times = np.arange(n_vols) * TR
        confounds = get_confounds_for_run(sub, session, run).iloc[:n_vols]
        dm = make_first_level_design_matrix(
            frame_times, events, hrf_model='spm', drift_model='cosine',
            high_pass=0.01, add_regs=confounds.values,
            add_reg_names=list(confounds.columns))
        design_matrices.append(dm)
        bold_paths.append(str(bold))
        print(f'sub-{subject} ses-{session} run-{run}: {n_vols} vols, '
              f'{dm.shape[1]} design columns')

    glm = FirstLevelModel(t_r=TR, mask_img=mask_img, smoothing_fwhm=smoothing_fwhm,
                          minimize_memory=True, n_jobs=1)
    glm.fit(bold_paths, design_matrices=design_matrices)

    out_root = Path(out_root or Path(bids_folder) / 'derivatives' / 'congruency_glm')
    out_dir = out_root / f'sub-{subject}' / 'func'
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, contrast_def in CONTRASTS[model].items():
        for stat in ('z_score', 'effect_size'):
            img = glm.compute_contrast(contrast_def, output_type=stat)
            stat_tag = 'z' if stat == 'z_score' else 'effect'
            fn = (out_dir / f'sub-{subject}_task-abstractvalue_model-{model}'
                            f'_contrast-{name}_stat-{stat_tag}_statmap.nii.gz')
            img.to_filename(str(fn))
        print(f'sub-{subject} model={model} contrast={name}: saved')


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('subject', help="Subject label without 'sub-', e.g. 06")
    p.add_argument('--model', choices=['congruent', 'incongruent', 'both', 'all'],
                   required=True)
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    p.add_argument('--smoothing-fwhm', type=float, default=None)
    args = p.parse_args()

    models = ['congruent', 'incongruent', 'both'] if args.model == 'all' else [args.model]
    for model in models:
        fit_subject(args.subject, model, bids_folder=args.bids_folder,
                   smoothing_fwhm=args.smoothing_fwhm)


if __name__ == '__main__':
    main()
