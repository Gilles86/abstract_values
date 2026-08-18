#!/usr/bin/env python3
"""Warp first-level congruency_glm effect-size maps to MNI152NLin2009cAsym.

Each subject has exactly one T1w anatomical reconstruction (from ses-1 --
ses-2 has only a T2w and its BOLD is coregistered directly into ses-1's T1w
space, see abstract_values/congruency_glm/fit_first_level.py), so ses-1's
`from-T1w_to-MNI152NLin2009cAsym` transform is always the right one to use,
regardless of which sessions contributed to the first-level fit.

Warps the EFFECT SIZE map (not z-score) -- SecondLevelModel expects
per-subject contrast/effect maps and computes its own group-level stats.

Usage
-----
  python -m abstract_values.congruency_glm.normalize_to_mni 06 --model congruent
  python -m abstract_values.congruency_glm.normalize_to_mni 06 --model all
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from abstract_values.congruency_glm.fit_first_level import CONTRASTS
from abstract_values.utils.data import BIDS_FOLDER

ANTS_APPLY_TRANSFORMS = (
    '/shares/zne.uzh/containers/fmriprep-25.2.3/app/.pixi/envs/fmriprep/bin/antsApplyTransforms')
# Already-cached (by fmriprep's own runs) 2mm MNI152NLin2009cAsym reference --
# avoids a network fetch on a compute node with no internet access.
MNI_REFERENCE = str(Path('~/.cache/templateflow/tpl-MNI152NLin2009cAsym/'
                         'tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz').expanduser())


def get_t1w_to_mni_xfm(subject, bids_folder=BIDS_FOLDER):
    xfm = (Path(bids_folder) / 'derivatives' / 'fmriprep' / f'sub-{subject}'
           / 'ses-1' / 'anat'
           / f'sub-{subject}_ses-1_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5')
    if not xfm.exists():
        raise FileNotFoundError(f'No T1w->MNI transform: {xfm}')
    return xfm


def normalize_contrast(subject, model, contrast_name, bids_folder=BIDS_FOLDER,
                       reference=None, out_root=None):
    in_dir = Path(bids_folder) / 'derivatives' / 'congruency_glm' / f'sub-{subject}' / 'func'
    src = (in_dir / f'sub-{subject}_task-abstractvalue_model-{model}'
                    f'_contrast-{contrast_name}_stat-effect_statmap.nii.gz')
    if not src.exists():
        raise FileNotFoundError(f'No first-level effect map: {src}')

    out_root = Path(out_root or Path(bids_folder) / 'derivatives' / 'congruency_glm')
    out_dir = out_root / f'sub-{subject}' / 'func'
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = (out_dir / f'sub-{subject}_task-abstractvalue_space-MNI152NLin2009cAsym'
                     f'_model-{model}_contrast-{contrast_name}_stat-effect_statmap.nii.gz')

    xfm = get_t1w_to_mni_xfm(subject, bids_folder)
    ref = reference or MNI_REFERENCE
    if not Path(ref).exists():
        raise FileNotFoundError(f'MNI reference not found: {ref}')

    cmd = [ANTS_APPLY_TRANSFORMS, '-d', '3', '-i', str(src), '-r', ref,
           '-t', str(xfm), '-o', str(dst), '--interpolation', 'LanczosWindowedSinc']
    print('running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    print('wrote', dst)
    return dst


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('subject')
    p.add_argument('--model', choices=['congruent', 'incongruent', 'both', 'all'],
                   required=True)
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    args = p.parse_args()

    models = ['congruent', 'incongruent', 'both'] if args.model == 'all' else [args.model]
    for model in models:
        for contrast_name in CONTRASTS[model]:
            normalize_contrast(args.subject, model, contrast_name,
                               bids_folder=args.bids_folder)


if __name__ == '__main__':
    main()
