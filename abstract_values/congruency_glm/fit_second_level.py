#!/usr/bin/env python3
"""Group (second-level) one-sample t-test on MNI-normalized congruency_glm
effect-size maps.

For each (model, contrast) produced by fit_first_level.py, pools every
subject's MNI-normalized effect-size map (see normalize_to_mni.py) into a
one-sample SecondLevelModel and saves an uncorrected z-map plus an FDR
thresholded map. (Nilearn has no RFT cluster-FWE method -- see the
project's /pycortex-adjacent thresholding discussion; permutation/TFCE via
non_parametric_inference would be the more rigorous option and is a
reasonable follow-up once the first pass looks sane.)

Usage
-----
  python -m abstract_values.congruency_glm.fit_second_level --subjects 03 04 05 ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel

from abstract_values.congruency_glm.fit_first_level import CONTRASTS
from abstract_values.utils.data import BIDS_FOLDER

MODELS = ['congruent', 'incongruent', 'both']


def find_subject_maps(subjects, model, contrast_name, bids_folder=BIDS_FOLDER):
    root = Path(bids_folder) / 'derivatives' / 'congruency_glm'
    maps, used = [], []
    for subject in subjects:
        fn = (root / f'sub-{subject}' / 'func'
              / f'sub-{subject}_task-abstractvalue_space-MNI152NLin2009cAsym'
                f'_model-{model}_contrast-{contrast_name}_stat-effect_statmap.nii.gz')
        if fn.exists():
            maps.append(str(fn))
            used.append(subject)
        else:
            print(f'  sub-{subject}: missing {fn.name} -- skipping')
    return maps, used


def fit_group(model, contrast_name, subjects, bids_folder=BIDS_FOLDER, out_root=None):
    maps, used = find_subject_maps(subjects, model, contrast_name, bids_folder)
    if len(maps) < 3:
        print(f'model={model} contrast={contrast_name}: only {len(maps)} subjects, skipping')
        return
    print(f'model={model} contrast={contrast_name}: n={len(maps)} subjects ({used})')

    design_matrix = pd.DataFrame({'intercept': [1] * len(maps)})
    glm = SecondLevelModel(smoothing_fwhm=None, minimize_memory=False)
    glm.fit(maps, design_matrix=design_matrix)
    z_map = glm.compute_contrast('intercept', output_type='z_score')

    out_root = Path(out_root or Path(bids_folder) / 'derivatives' / 'congruency_glm' / 'group')
    out_root.mkdir(parents=True, exist_ok=True)
    base = out_root / f'group_n-{len(maps)}_task-abstractvalue_model-{model}_contrast-{contrast_name}'
    z_map.to_filename(str(base) + '_stat-z_statmap.nii.gz')

    fdr_map, thr = threshold_stats_img(z_map, alpha=0.05, height_control='fdr',
                                       cluster_threshold=10, two_sided=True)
    fdr_map.to_filename(str(base) + '_stat-zFDR_statmap.nii.gz')
    print(f'  saved (FDR |z|>{thr:.2f})')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--subjects', nargs='+', required=True)
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    args = p.parse_args()

    for model in MODELS:
        for contrast_name in CONTRASTS[model]:
            fit_group(model, contrast_name, args.subjects, bids_folder=args.bids_folder)


if __name__ == '__main__':
    main()
