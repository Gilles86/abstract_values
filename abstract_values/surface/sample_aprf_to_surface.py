#!/usr/bin/env python3
"""
Sample aPRF parameters (and gabor-model R²) from T1w volume to cortical surface.

Uses nilearn.surface.vol_to_surf with the fmriprep T1w-space pial and white
surfaces for accurate grey-matter sampling.  Optionally transforms the
fsnative GIfTI to fsaverage via FreeSurfer's surface registration (nipype).

Parameters sampled
------------------
  aPRF model   : mu, sd, amplitude, baseline, r2, fwhm
  Gabor model  : r2  (from vonmises encoding model, written as desc-gabor-r2)

Output
------
  aPRF volumes are always loaded from the joint (all-sessions) fit path.
  Surface outputs are written next to those volumes:

  derivatives/encoding_models/aprf/sub-<subject>/func/
    sub-<subject>_task-abstractvalue_hemi-{L,R}_space-fsnative_desc-<par>[_smoothed]_pe.func.gii
    sub-<subject>_task-abstractvalue_hemi-{L,R}_space-fsaverage_desc-<par>[_smoothed]_pe.func.gii

The ``--session`` argument is only used to locate the per-session
FreeSurfer/fmriprep surfaces; it does not select an aPRF fit variant.

Usage
-----
  python sample_aprf_to_surface.py pil01 --session 1
  python sample_aprf_to_surface.py pil01 --session 1 --smoothed
"""

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import surface
from nipype.interfaces.freesurfer import SurfaceTransform

from abstract_values.utils.data import BIDS_FOLDER


def transform_fsaverage(in_file, fs_hemi, source_subject, subjects_dir):
    """Transform a fsnative GIfTI surface file to fsaverage space."""
    out_file = str(in_file).replace('space-fsnative', 'space-fsaverage')
    os.environ['SUBJECTS_DIR'] = str(subjects_dir)
    sxfm = SurfaceTransform(subjects_dir=str(subjects_dir))
    sxfm.inputs.source_file = str(in_file)
    sxfm.inputs.out_file = out_file
    sxfm.inputs.source_subject = source_subject
    sxfm.inputs.target_subject = 'fsaverage'
    sxfm.inputs.hemi = fs_hemi
    sxfm.run()
    return out_file


def main(subject, session, bids_folder=BIDS_FOLDER,
         fmriprep_deriv='fmriprep', smoothed=False):
    bids_folder = Path(bids_folder)
    fmriprep_dir = bids_folder / 'derivatives' / fmriprep_deriv
    subjects_dir = fmriprep_dir / 'sourcedata' / 'freesurfer'
    fs_subject = f'sub-{subject}_ses-{session}'

    smooth_tag = '_smoothed' if smoothed else ''

    # aPRF / vonmises volumes always live at the joint (all-sessions) path —
    # no ses- entity. The `--session` argument is used only for locating the
    # FreeSurfer subject + per-session anat surfaces below.
    aprf_dir = (bids_folder / 'derivatives' / 'encoding_models' / 'aprf'
                / f'sub-{subject}' / 'func')

    # aPRF parameter volumes; (desc, to_fsaverage)
    aprf_params = [
        ('mu',        True),
        ('sd',        True),
        ('amplitude', False),
        ('baseline',  False),
        ('r2',        True),
        ('fwhm',      True),
    ]

    # Gabor (vonmises) R² volume
    vm_dir = (bids_folder / 'derivatives' / 'encoding_models' / 'vonmises'
              / f'sub-{subject}' / 'func')
    vm_r2 = (vm_dir / f'sub-{subject}_task-abstractvalue'
                       f'_space-T1w_desc-r2_pe.nii.gz')

    # Build list of (volume_path, surface_desc_label, to_fsaverage)
    volumes = []
    for par, to_fsav in aprf_params:
        fn = (aprf_dir / f'sub-{subject}_task-abstractvalue'
                         f'_space-T1w_desc-{par}{smooth_tag}_pe.nii.gz')
        if fn.exists():
            volumes.append((fn, par, to_fsav))
        else:
            print(f'  WARNING: {fn.name} not found, skipping')

    if vm_r2.exists():
        volumes.append((vm_r2, 'gabor-r2', True))
    else:
        print(f'  WARNING: vonmises r2 not found at {vm_r2}, skipping')

    if not volumes:
        raise RuntimeError('No volumes found to sample — check that aPRF fitting has been run.')

    # Surfaces from fmriprep anat dir for this session
    anat_dir = fmriprep_dir / f'sub-{subject}' / f'ses-{session}' / 'anat'

    print(f'Sampling to surface: sub-{subject}  all-sessions  (FreeSurfer: {fs_subject})')
    print(f'Output directory:    {aprf_dir}')

    for hemi, fs_hemi in [('L', 'lh'), ('R', 'rh')]:
        pial  = anat_dir / f'sub-{subject}_ses-{session}_hemi-{hemi}_pial.surf.gii'
        white = anat_dir / f'sub-{subject}_ses-{session}_hemi-{hemi}_white.surf.gii'

        if not pial.exists() or not white.exists():
            raise FileNotFoundError(
                f'Surfaces not found — expected:\n  {pial}\n  {white}\n'
                f'Check --fmriprep-deriv and --session.')

        for vol_path, desc, to_fsav in volumes:
            print(f'  [{hemi}] {desc} ...', flush=True)
            data = surface.vol_to_surf(str(vol_path), str(pial),
                                       inner_mesh=str(white))
            data = data.astype(np.float32)

            out_fn = (aprf_dir
                      / f'sub-{subject}_task-abstractvalue'
                        f'_hemi-{hemi}_space-fsnative_desc-{desc}{smooth_tag}_pe.func.gii')
            im = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(data)])
            nib.save(im, str(out_fn))

            if to_fsav:
                print(f'    → transforming to fsaverage', flush=True)
                transform_fsaverage(out_fn, fs_hemi, fs_subject, subjects_dir)

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-', e.g. pil01")
    parser.add_argument('--session', type=int, required=True,
                        help='Session number (used to find surfaces and FreeSurfer subject; '
                             'does NOT select an aPRF fit variant — those are always joint)')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep',
                        choices=['fmriprep', 'fmriprep-t2w'])
    parser.add_argument('--smoothed', action='store_true',
                        help='Load smoothed aPRF volumes (desc-<par>_smoothed)')
    args = parser.parse_args()

    main(args.subject, args.session,
         bids_folder=args.bids_folder, fmriprep_deriv=args.fmriprep_deriv,
         smoothed=args.smoothed)
