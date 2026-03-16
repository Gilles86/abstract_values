#!/usr/bin/env python3
"""
Create volumetric ROI masks in T1w space from FreeSurfer surface labels.

For each ROI, both hemispheres are projected from the FreeSurfer surface to a
volume in fsnative space (using mri_label2vol), then resampled to T1w space using
mri_vol2vol --regheader (derives the registration from image vox2ras headers,
equivalent to the ITK transform stored alongside the T1w by fmriprep), and
finally combined into a bilateral mask.

ROIs created
------------
  V1exvivo        — V1 from the cytoarchitectonic exvivo atlas (all probability)
  V1exvivoThresh  — V1 from the cytoarchitectonic exvivo atlas (thresholded, recommended)
  V2exvivo        — V2 from the cytoarchitectonic exvivo atlas
  V2exvivoThresh  — V2 from the cytoarchitectonic exvivo atlas (thresholded)
  hOc1            — V1 from the MPM VPNL atlas (hOc1)
  hOc2            — V2 from the MPM VPNL atlas (hOc2)
  hOc3v           — V3v from the MPM VPNL atlas (hOc3v)
  hOc4v           — V4 from the MPM VPNL atlas (hOc4v)

Output
------
  derivatives/masks/sub-<subject>/ses-<session>/anat/
    sub-<subject>_ses-<session>_space-T1w_hemi-L_desc-<roi>_mask.nii.gz
    sub-<subject>_ses-<session>_space-T1w_hemi-R_desc-<roi>_mask.nii.gz
    sub-<subject>_ses-<session>_space-T1w_hemi-LR_desc-<roi>_mask.nii.gz  ← bilateral

Usage
-----
  python create_roi_masks.py pil01 1
  python create_roi_masks.py pil01 1 --rois V1exvivoThresh hOc1
  python create_roi_masks.py pil01 1 --fmriprep-deriv fmriprep-noflair
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from abstract_values.utils.data import BIDS_FOLDER

FREESURFER_BIN = Path(os.environ.get('FREESURFER_HOME', '/Users/gdehol/freesurfer')) / 'bin'

# ROI name → label file stem (hemi prefix added automatically: lh.<stem>.label)
ROIS = {
    'V1exvivo':       'V1_exvivo',
    'V1exvivoThresh': 'V1_exvivo.thresh',
    'V2exvivo':       'V2_exvivo',
    'V2exvivoThresh': 'V2_exvivo.thresh',
    'hOc1':           'hOc1.mpm.vpnl',
    'hOc2':           'hOc2.mpm.vpnl',
    'hOc3v':          'hOc3v.mpm.vpnl',
    'hOc4v':          'hOc4v.mpm.vpnl',
}


def run(cmd, env=None):
    """Run a shell command, printing it first."""
    print('  $', ' '.join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, env=env)


def label_to_fsnative_vol(label_file, orig_mgz, subject_name, hemi, out_mgz, env):
    """Project a surface label file to a volume in FreeSurfer native space.

    Uses mri_label2vol with --proj frac 0 1 0.1 to fill the cortical ribbon
    between white and pial surfaces. env must have SUBJECTS_DIR set.
    """
    run([
        FREESURFER_BIN / 'mri_label2vol',
        '--label',     label_file,
        '--temp',      orig_mgz,
        '--subject',   subject_name,
        '--regheader', orig_mgz,
        '--hemi',      hemi,
        '--fillthresh', '0.5',
        '--proj', 'frac', '0', '1', '0.1',
        '--o', out_mgz,
    ], env=env)


def resample_to_t1w(src_mgz, t1w_nii, out_nii, env):
    """Resample src volume to T1w space using nearest-neighbour interpolation.

    Uses --regheader to derive the registration from image vox2ras headers,
    which is equivalent to the ITK transform fmriprep stores alongside the T1w.
    """
    run([
        FREESURFER_BIN / 'mri_vol2vol',
        '--mov',       src_mgz,
        '--targ',      t1w_nii,
        '--regheader',
        '--nearest',
        '--o',         out_nii,
    ], env=env)


def combine_hemispheres(lh_path, rh_path, out_path):
    """Logical-OR two hemisphere masks and save as bilateral NIfTI."""
    lh_img = nib.load(str(lh_path))
    rh_img = nib.load(str(rh_path))
    bilateral = (lh_img.get_fdata() > 0) | (rh_img.get_fdata() > 0)
    out_img = nib.Nifti1Image(bilateral.astype(np.uint8),
                               lh_img.affine, lh_img.header)
    out_img.to_filename(str(out_path))
    print(f'    bilateral: {int(bilateral.sum())} voxels → {out_path.name}')


def make_roi_masks(subject_id, session, bids_folder=BIDS_FOLDER,
                   fmriprep_deriv='fmriprep-flair', rois=None):
    bids_folder = Path(bids_folder)
    fmriprep_dir = bids_folder / 'derivatives' / fmriprep_deriv

    fs_subject_name = f'sub-{subject_id}_ses-{session}'
    fs_subjects_dir = fmriprep_dir / 'sourcedata' / 'freesurfer'
    fs_dir = fs_subjects_dir / fs_subject_name
    anat_dir = fmriprep_dir / f'sub-{subject_id}' / f'ses-{session}' / 'anat'

    orig_mgz = fs_dir / 'mri' / 'orig.mgz'
    t1w = anat_dir / f'sub-{subject_id}_ses-{session}_desc-preproc_T1w.nii.gz'

    for f in [orig_mgz, t1w]:
        if not f.exists():
            raise FileNotFoundError(f'Required file not found: {f}')

    out_dir = (bids_folder / 'derivatives' / 'masks'
               / f'sub-{subject_id}' / f'ses-{session}' / 'anat')
    out_dir.mkdir(parents=True, exist_ok=True)

    if rois is None:
        rois = list(ROIS.keys())

    fs_env = dict(os.environ)
    fs_env['SUBJECTS_DIR'] = str(fs_subjects_dir)

    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)

        fn_prefix = (f'sub-{subject_id}_ses-{session}'
                     f'_space-T1w')

        for roi_name in rois:
            label_stem = ROIS[roi_name]
            print(f'\n[{roi_name}]')

            hemi_paths = {}
            for hemi, hemi_label in [('lh', 'L'), ('rh', 'R')]:
                label_file = fs_dir / 'label' / f'{hemi}.{label_stem}.label'
                if not label_file.exists():
                    print(f'  WARNING: label not found: {label_file.name}')
                    continue

                # 1. Label → volume in fsnative space
                fsnative_mgz = tmpdir / f'{hemi}_{roi_name}_fsnative.mgz'
                label_to_fsnative_vol(label_file, orig_mgz,
                                       fs_subject_name, hemi, fsnative_mgz, fs_env)

                # 2. Fsnative → T1w space
                out_nii = out_dir / f'{fn_prefix}_hemi-{hemi_label}_desc-{roi_name}_mask.nii.gz'
                resample_to_t1w(fsnative_mgz, t1w, out_nii, fs_env)

                n_voxels = int((nib.load(str(out_nii)).get_fdata() > 0).sum())
                print(f'    {hemi}: {n_voxels} voxels → {out_nii.name}')
                hemi_paths[hemi_label] = out_nii

            # 3. Combine hemispheres into bilateral mask
            if 'L' in hemi_paths and 'R' in hemi_paths:
                bilateral = out_dir / f'{fn_prefix}_hemi-LR_desc-{roi_name}_mask.nii.gz'
                combine_hemispheres(hemi_paths['L'], hemi_paths['R'], bilateral)

    print(f'\nAll masks saved to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-', e.g. pil01")
    parser.add_argument('session', type=int, help='Session number')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep-flair',
                        choices=['fmriprep', 'fmriprep-flair', 'fmriprep-noflair'])
    parser.add_argument('--rois', nargs='+', choices=list(ROIS.keys()),
                        default=None,
                        help='ROIs to create (default: all)')
    args = parser.parse_args()

    make_roi_masks(args.subject, args.session,
                   bids_folder=args.bids_folder,
                   fmriprep_deriv=args.fmriprep_deriv,
                   rois=args.rois)
