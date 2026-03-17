#!/usr/bin/env python3
"""
Create volumetric V1/V2/V3 masks in T1w space from FreeSurfer surface labels.

For each ROI, both hemispheres are projected from the FreeSurfer surface to a
volume in fsnative space (using mri_label2vol), then resampled to T1w space using
mri_vol2vol --regheader (derives the registration from image vox2ras headers,
equivalent to the ITK transform stored alongside the T1w by fmriprep), and
finally combined into a bilateral mask.

Three atlas sources are supported
----------------------------------
  exvivo   — FreeSurfer cytoarchitectonic exvivo atlas (thresholded .label files)
             Provides V1exvivoThresh, V2exvivoThresh.
  vpnl     — MPM VPNL atlas (Benson et al.) distributed with FreeSurfer
             Provides hOc1 (V1), hOc2 (V2).
  benson   — Anatomical retinotopy atlas (Benson & Winawer, requires neuropythy)
             Predicts V1/V2/V3 from cortical anatomy — no fMRI data needed.
             Provides BensonV1, BensonV2, BensonV3.

Default: all three sources.

Output
------
  derivatives/masks/sub-<subject>/ses-<session>/anat/
    sub-<subject>_ses-<session>_space-T1w_hemi-L_desc-<roi>_mask.nii.gz
    sub-<subject>_ses-<session>_space-T1w_hemi-R_desc-<roi>_mask.nii.gz
    sub-<subject>_ses-<session>_space-T1w_hemi-LR_desc-<roi>_mask.nii.gz  ← bilateral

Usage
-----
  python create_roi_masks.py pil01 1
  python create_roi_masks.py pil01 1 --atlases benson
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

# Exvivo and VPNL atlas: ROI name → label file stem (lh.<stem>.label)
LABEL_ROIS = {
    'V1exvivoThresh': 'V1_exvivo.thresh',
    'V2exvivoThresh': 'V2_exvivo.thresh',
    'hOc1':           'hOc1.mpm.vpnl',   # V1
    'hOc2':           'hOc2.mpm.vpnl',   # V2
}

# Benson atlas: ROI name → integer area label from predict_retinotopy()
BENSON_ROIS = {
    'BensonV1': 1,
    'BensonV2': 2,
    'BensonV3': 3,
}


def run(cmd, env=None):
    print('  $', ' '.join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, env=env)


def label_to_fsnative_vol(label_file, orig_mgz, subject_name, hemi, out_mgz, env):
    """Project a .label file to a binary volume in FreeSurfer native space."""
    run([
        FREESURFER_BIN / 'mri_label2vol',
        '--label',      label_file,
        '--temp',       orig_mgz,
        '--subject',    subject_name,
        '--regheader',  orig_mgz,
        '--hemi',       hemi,
        '--fillthresh', '0.5',
        '--proj', 'frac', '0', '1', '0.1',
        '--o', out_mgz,
    ], env=env)


def resample_to_t1w(src_mgz, t1w_nii, out_nii, env):
    """Resample src to T1w space using nearest-neighbour (header-based registration)."""
    run([
        FREESURFER_BIN / 'mri_vol2vol',
        '--mov',       src_mgz,
        '--targ',      t1w_nii,
        '--regheader',
        '--nearest',
        '--o',         out_nii,
    ], env=env)


def write_label_file(path, vertex_indices, coords):
    """Write a FreeSurfer ASCII .label file.

    coords : (N, 3) array of RAS surface coordinates for the selected vertices.
    """
    with open(path, 'w') as f:
        f.write('#!ascii label\n')
        f.write(f'{len(vertex_indices)}\n')
        for idx, (x, y, z) in zip(vertex_indices, coords):
            f.write(f'{idx}  {x:.6f}  {y:.6f}  {z:.6f}  0.000000\n')


def combine_hemispheres(lh_path, rh_path, out_path):
    lh_img = nib.load(str(lh_path))
    rh_img = nib.load(str(rh_path))
    bilateral = (lh_img.get_fdata() > 0) | (rh_img.get_fdata() > 0)
    out_img = nib.Nifti1Image(bilateral.astype(np.uint8),
                               lh_img.affine, lh_img.header)
    out_img.to_filename(str(out_path))
    print(f'    bilateral: {int(bilateral.sum())} voxels → {out_path.name}')


def _process_label_roi(roi_name, label_file_lh, label_file_rh,
                        orig_mgz, t1w, fs_subject_name, fn_prefix,
                        out_dir, tmpdir, fs_env):
    """Shared: run mri_label2vol + mri_vol2vol for lh and rh, then combine."""
    print(f'\n[{roi_name}]')
    hemi_paths = {}
    for hemi, hemi_label, label_file in [('lh', 'L', label_file_lh),
                                           ('rh', 'R', label_file_rh)]:
        if label_file is None or not Path(label_file).exists():
            print(f'  WARNING: label not found for {hemi}')
            continue
        fsnative_mgz = tmpdir / f'{hemi}_{roi_name}_fsnative.mgz'
        label_to_fsnative_vol(label_file, orig_mgz, fs_subject_name,
                               hemi, fsnative_mgz, fs_env)
        out_nii = out_dir / f'{fn_prefix}_hemi-{hemi_label}_desc-{roi_name}_mask.nii.gz'
        resample_to_t1w(fsnative_mgz, t1w, out_nii, fs_env)
        n_voxels = int((nib.load(str(out_nii)).get_fdata() > 0).sum())
        print(f'    {hemi}: {n_voxels} voxels → {out_nii.name}')
        hemi_paths[hemi_label] = out_nii

    if 'L' in hemi_paths and 'R' in hemi_paths:
        bilateral = out_dir / f'{fn_prefix}_hemi-LR_desc-{roi_name}_mask.nii.gz'
        combine_hemispheres(hemi_paths['L'], hemi_paths['R'], bilateral)


def make_roi_masks(subject_id, session, bids_folder=BIDS_FOLDER,
                   fmriprep_deriv='fmriprep-flair', atlases=None):
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

    if atlases is None:
        atlases = ['exvivo', 'vpnl', 'benson']

    fs_env = dict(os.environ)
    fs_env['SUBJECTS_DIR'] = str(fs_subjects_dir)

    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        fn_prefix = f'sub-{subject_id}_ses-{session}_space-T1w'

        # ── exvivo + VPNL atlas (pre-existing .label files) ──────────────────
        label_roi_names = []
        if 'exvivo' in atlases:
            label_roi_names += ['V1exvivoThresh', 'V2exvivoThresh']
        if 'vpnl' in atlases:
            label_roi_names += ['hOc1', 'hOc2']

        for roi_name in label_roi_names:
            stem = LABEL_ROIS[roi_name]
            lh_label = fs_dir / 'label' / f'lh.{stem}.label'
            rh_label = fs_dir / 'label' / f'rh.{stem}.label'
            _process_label_roi(roi_name, lh_label, rh_label,
                                orig_mgz, t1w, fs_subject_name, fn_prefix,
                                out_dir, tmpdir, fs_env)

        # ── Benson anatomical atlas (neuropythy) ──────────────────────────────
        if 'benson' in atlases:
            import neuropythy as ny
            print('\n[Benson atlas] loading neuropythy predictions...')
            sub = ny.freesurfer_subject(str(fs_dir))
            lh_pred, rh_pred = ny.vision.predict_retinotopy(sub)
            lh_varea = np.array(lh_pred['varea'])
            rh_varea = np.array(rh_pred['varea'])
            lh_coords = sub.lh.white_surface.coordinates.T  # (N, 3)
            rh_coords = sub.rh.white_surface.coordinates.T

            for roi_name, area_id in BENSON_ROIS.items():
                lh_idx = np.where(lh_varea == area_id)[0]
                rh_idx = np.where(rh_varea == area_id)[0]

                lh_label = tmpdir / f'lh_{roi_name}.label'
                rh_label = tmpdir / f'rh_{roi_name}.label'
                write_label_file(lh_label, lh_idx, lh_coords[lh_idx])
                write_label_file(rh_label, rh_idx, rh_coords[rh_idx])

                _process_label_roi(roi_name, lh_label, rh_label,
                                    orig_mgz, t1w, fs_subject_name, fn_prefix,
                                    out_dir, tmpdir, fs_env)

    print(f'\nAll masks saved to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-', e.g. pil01")
    parser.add_argument('session', type=int, help='Session number')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep-flair',
                        choices=['fmriprep', 'fmriprep-flair', 'fmriprep-noflair',
                                 'fmriprep-t2w'])
    parser.add_argument('--atlases', nargs='+',
                        choices=['exvivo', 'vpnl', 'benson'],
                        default=None,
                        help='Atlas sources to use (default: all three)')
    args = parser.parse_args()

    make_roi_masks(args.subject, args.session,
                   bids_folder=args.bids_folder,
                   fmriprep_deriv=args.fmriprep_deriv,
                   atlases=args.atlases)
