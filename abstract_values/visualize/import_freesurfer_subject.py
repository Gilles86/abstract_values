#!/usr/bin/env python3
"""
Import a FreeSurfer subject into the pycortex filestore.

Imports the fmriprep-flair FreeSurfer reconstruction and registers
the T1w→EPI transform so pycortex can map volumes to the flatmap.

The FreeSurfer subject is session-specific (sub-<subject>_ses-<session>)
because fmriprep runs per-session anatomy.  The pycortex subject is stored
as 'abstractvalue.sub-<subject>' and is session-agnostic (all sessions share
the same cortical surface).

Usage
-----
  python import_freesurfer_subject.py pil01 1
  python import_freesurfer_subject.py pil01 1 --fmriprep-deriv fmriprep-noflair
"""

import argparse
import os.path as op

import numpy as np
from cortex import freesurfer
from cortex.xfm import Transform
from nitransforms.linear import Affine

from abstract_values.utils.data import BIDS_FOLDER


def main(subject, session, bids_folder=BIDS_FOLDER,
         fmriprep_deriv='fmriprep-flair'):
    bids_folder = str(bids_folder)
    fmriprep_dir = op.join(bids_folder, 'derivatives', fmriprep_deriv)
    fs_subjects_dir = op.join(fmriprep_dir, 'sourcedata', 'freesurfer')
    fs_subject_name = f'sub-{subject}_ses-{session}'
    cx_subject = f'abstractvalue.sub-{subject}'

    anat_dir = op.join(fmriprep_dir, f'sub-{subject}', f'ses-{session}', 'anat')
    func_dir = op.join(fmriprep_dir, f'sub-{subject}', f'ses-{session}', 'func')

    t1w = op.join(anat_dir, f'sub-{subject}_ses-{session}_desc-preproc_T1w.nii.gz')
    fsnative2t1w_itk = op.join(anat_dir,
        f'sub-{subject}_ses-{session}_from-fsnative_to-T1w_mode-image_xfm.txt')
    epi = op.join(func_dir,
        f'sub-{subject}_ses-{session}_task-abstractvalue_run-1_space-T1w_boldref.nii.gz')

    for f in [t1w, fsnative2t1w_itk, epi]:
        if not op.exists(f):
            raise FileNotFoundError(f'Required file not found: {f}')

    # 1. Import FreeSurfer surfaces into pycortex filestore
    print(f'Importing {fs_subject_name} → pycortex subject {cx_subject} ...')
    freesurfer.import_subj(
        fs_subject_name,
        pycortex_subject=cx_subject,
        freesurfer_subject_dir=fs_subjects_dir)

    # 2. Convert fsnative→T1w from ITK (fmriprep) to FSL format
    fsnative2t1w_fsl = op.join(anat_dir,
        f'sub-{subject}_ses-{session}_from-fsnative_to-T1w_mode-image_xfm.fsl')
    print('Converting fsnative→T1w transform ITK → FSL ...')
    xfm = Affine.from_filename(fsnative2t1w_itk, fmt='itk', reference=t1w)
    xfm.to_filename(fsnative2t1w_fsl, fmt='fsl')

    # 3. Build pycortex transform from the FSL matrix
    print('Saving pycortex transform ...')
    pycortex_xfm = Transform.from_fsl(fsnative2t1w_fsl, epi, t1w)
    pycortex_xfm.save(cx_subject, 'epi', xfmtype='coord')

    # 4. Also save an identity transform (useful for T1w-space volumes)
    Transform(np.identity(4), epi).save(cx_subject, 'epi.identity')

    print(f'Done. Pycortex subject: {cx_subject}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-', e.g. pil01")
    parser.add_argument('session', type=int,
                        help='Session number (determines FreeSurfer subject name)')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep-flair',
                        choices=['fmriprep', 'fmriprep-flair', 'fmriprep-noflair', 'fmriprep-t2w'])
    args = parser.parse_args()

    main(args.subject, args.session,
         bids_folder=args.bids_folder,
         fmriprep_deriv=args.fmriprep_deriv)
