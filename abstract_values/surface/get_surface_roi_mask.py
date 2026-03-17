#!/usr/bin/env python3
"""
Transform a surface ROI from fsaverage space to a T1w volumetric mask.

Workflow (mirrors neural_priors/surface/get_npc_mask.py):
  1. Transform GIfTI surface labels from fsaverage → fsnative using
     FreeSurfer's surface registration (nipype SurfaceTransform).
  2. Project the fsnative surface mask to T1w-space volume using
     neuropythy's cortex_to_image.
  3. Write bilateral, left-only, and right-only masks.

Input labels
------------
  GIfTI files (.label.gii or .func.gii) in fsaverage space.
  Any non-zero value is treated as "in ROI".

Output
------
  derivatives/masks/sub-<subject>/ses-<session>/anat/
    sub-<subject>_ses-<session>_space-T1w_desc-<roi>_mask.nii.gz   (bilateral)
    sub-<subject>_ses-<session>_space-T1w_desc-<roi>l_mask.nii.gz  (left)
    sub-<subject>_ses-<session>_space-T1w_desc-<roi>r_mask.nii.gz  (right)

Usage
-----
  python get_surface_roi_mask.py pil01 1 \\
      --lh /path/to/lh.myROI_space-fsaverage.label.gii \\
      --rh /path/to/rh.myROI_space-fsaverage.label.gii \\
      --roi myROI
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
from nilearn import surface
from nipype.interfaces.freesurfer import SurfaceTransform

from neuropythy.freesurfer import subject as fs_subject_fn
from neuropythy.io import load, save
from neuropythy.mri import image_clear, to_image

from abstract_values.utils.data import BIDS_FOLDER


def transform_to_fsnative(in_file, out_file, fs_hemi, target_subject, subjects_dir):
    """Transform a GIfTI surface file from fsaverage to fsnative space."""
    import os
    os.environ.setdefault('SUBJECTS_DIR', str(subjects_dir))
    sxfm = SurfaceTransform(subjects_dir=str(subjects_dir))
    sxfm.inputs.source_file = str(in_file)
    sxfm.inputs.out_file = str(out_file)
    sxfm.inputs.source_subject = 'fsaverage'
    sxfm.inputs.target_subject = target_subject
    sxfm.inputs.hemi = fs_hemi
    sxfm.run()


def main(subject, session, lh_label, rh_label, roi='roi',
         bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep-flair'):
    bids_folder = Path(bids_folder)
    fmriprep_dir = bids_folder / 'derivatives' / fmriprep_deriv
    subjects_dir = fmriprep_dir / 'sourcedata' / 'freesurfer'
    fs_subject_name = f'sub-{subject}_ses-{session}'
    fs_dir = subjects_dir / fs_subject_name

    anat_dir = fmriprep_dir / f'sub-{subject}' / f'ses-{session}' / 'anat'
    t1w = anat_dir / f'sub-{subject}_ses-{session}_desc-preproc_T1w.nii.gz'

    for p in [fs_dir, t1w]:
        if not Path(p).exists():
            raise FileNotFoundError(f'Required path not found: {p}')

    out_dir = (bids_folder / 'derivatives' / 'masks'
               / f'sub-{subject}' / f'ses-{session}' / 'anat')
    out_dir.mkdir(parents=True, exist_ok=True)

    fn_prefix = f'sub-{subject}_ses-{session}_space-T1w_desc-{roi}'

    mask_data = []  # [lh_data_or_None, rh_data_or_None]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        inputs = [
            (lh_label, 'lh', 'L'),
            (rh_label, 'rh', 'R'),
        ]
        for in_file, fs_hemi, hemi in inputs:
            if in_file is None:
                mask_data.append(None)
                continue

            fsnative_mgz = tmpdir / f'{fs_hemi}.{roi}_fsnative.mgz'
            print(f'Transforming {hemi} {roi}: fsaverage → {fs_subject_name} ...', flush=True)
            transform_to_fsnative(in_file, fsnative_mgz, fs_hemi,
                                  fs_subject_name, subjects_dir)

            data = surface.load_surf_data(str(fsnative_mgz))
            mask = (data > 0).astype(np.float32)
            mask_data.append(mask)
            print(f'  {hemi}: {int(mask.sum())} vertices in ROI')

    # Project to T1w volume via neuropythy
    sub = fs_subject_fn(str(fs_dir))
    im = load(str(t1w))
    im = to_image(image_clear(im, fill=0.0), dtype=np.int32)

    lh_mask, rh_mask = mask_data

    # Bilateral
    if lh_mask is not None and rh_mask is not None:
        print('Projecting bilateral mask to volume...', flush=True)
        vol = sub.cortex_to_image(
            (lh_mask, rh_mask), im, hemi=None, method='nearest', fill=0.0)
        out_fn = out_dir / f'{fn_prefix}_mask.nii.gz'
        save(str(out_fn), vol)
        print(f'  saved: {out_fn.name}')

    # Left only
    if lh_mask is not None:
        print('Projecting left-hemisphere mask to volume...', flush=True)
        vol = sub.cortex_to_image(
            lh_mask, im, hemi='lh', method='nearest', fill=0.0)
        out_fn = out_dir / f'{fn_prefix}l_mask.nii.gz'
        save(str(out_fn), vol)
        print(f'  saved: {out_fn.name}')

    # Right only
    if rh_mask is not None:
        print('Projecting right-hemisphere mask to volume...', flush=True)
        zero_lh = np.zeros_like(lh_mask) if lh_mask is not None else np.zeros(sub.lh.vertex_count)
        vol = sub.cortex_to_image(
            (zero_lh, rh_mask), im, hemi=None, method='nearest', fill=0.0)
        out_fn = out_dir / f'{fn_prefix}r_mask.nii.gz'
        save(str(out_fn), vol)
        print(f'  saved: {out_fn.name}')

    print(f'Done. Masks saved to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-', e.g. pil01")
    parser.add_argument('session', type=int)
    parser.add_argument('--lh', required=True,
                        help='Left hemisphere fsaverage surface label (GIfTI)')
    parser.add_argument('--rh', required=True,
                        help='Right hemisphere fsaverage surface label (GIfTI)')
    parser.add_argument('--roi', default='roi',
                        help='ROI name used in output filenames (default: roi)')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep-flair',
                        choices=['fmriprep', 'fmriprep-flair', 'fmriprep-noflair',
                                 'fmriprep-t2w'])
    args = parser.parse_args()

    main(args.subject, args.session,
         lh_label=args.lh, rh_label=args.rh,
         roi=args.roi,
         bids_folder=args.bids_folder,
         fmriprep_deriv=args.fmriprep_deriv)
