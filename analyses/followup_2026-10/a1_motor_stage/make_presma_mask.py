#!/usr/bin/env python3
"""Pre-SMA control ROI: fsaverage definition + projection to each subject's T1w.

No HCP-MMP (6ma) annotation is available offline here, so pre-SMA is defined
anatomically on fsaverage (MNI305 coordinates of the white surface):

    FreeSurfer BA6_exvivo.thresh  (Brodmann area 6, ex-vivo, thresholded)
  ∩ aparc 'superiorfrontal'
  ∩ medial wall:          |x| < 15 mm
  ∩ anterior to the VCA:  0 < y < 30 mm   (VCA line ≈ y = 0; SMA proper lies behind it)
  ∩ dorsal:               z > 35 mm       (excludes cingulate)

This follows the usual landmark definition (Picard & Strick 2001): pre-SMA is
the medial BA6 rostral to the vertical line through the anterior commissure.
The fsaverage label is then mapped to fsnative by nearest neighbour on
?h.sphere.reg and to T1w with neuropythy, exactly as in
``abstract_values/surface/project_fsaverage_rois.py``. Output (never touching
existing derivatives):

  derivatives/followup_2026-10/a1_motor_stage/masks/sub-<s>/anat/
      sub-<s>_space-T1w_desc-preSMA[l|r]_mask.nii.gz
  derivatives/followup_2026-10/a1_motor_stage/masks/fsaverage_preSMA_{lh,rh}.npy

Run on the cluster (abstract_values env):
    python make_presma_mask.py            # all subjects
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

BIDS = Path('/shares/zne.uzh/gdehol/ds-abstractvalue')
SD = BIDS / 'derivatives' / 'fmriprep' / 'sourcedata' / 'freesurfer'
OUT = BIDS / 'derivatives' / 'followup_2026-10' / 'a1_motor_stage' / 'masks'


def fsaverage_presma(h):
    fsa = SD / 'fsaverage'
    xyz = nib.freesurfer.read_geometry(str(fsa / 'surf' / f'{h}.white'))[0]
    lab = np.zeros(len(xyz), bool)
    lab[nib.freesurfer.read_label(str(fsa / 'label' / f'{h}.BA6_exvivo.thresh.label'))] = True
    codes, _, names = nib.freesurfer.read_annot(str(fsa / 'label' / f'{h}.aparc.annot'))
    sfg = codes == [n.decode() for n in names].index('superiorfrontal')
    x, y, z = xyz.T
    return lab & sfg & (np.abs(x) < 15) & (y > 0) & (y < 30) & (z > 35)


def project(subject, fsa_labels):
    import neuropythy as ny
    from neuropythy.mri import image_clear, to_image
    fs_dir = SD / f'sub-{subject}_ses-1'
    t1w = (BIDS / 'derivatives' / 'fmriprep' / f'sub-{subject}' / 'ses-1' / 'anat'
           / f'sub-{subject}_ses-1_desc-preproc_T1w.nii.gz')
    if not (fs_dir.exists() and t1w.exists()):
        print(f'sub-{subject}: missing freesurfer/T1w, skipped')
        return
    codes = []
    for hi, h in enumerate(('lh', 'rh')):
        fsa = nib.freesurfer.read_geometry(str(SD / 'fsaverage' / 'surf' / f'{h}.sphere.reg'))[0]
        nat = nib.freesurfer.read_geometry(str(fs_dir / 'surf' / f'{h}.sphere.reg'))[0]
        nn = cKDTree(fsa).query(nat)[1]
        code = np.full(len(nat), 1 | (hi << 1), np.int32)
        code |= fsa_labels[hi][nn].astype(np.int32) << 2
        codes.append(code)
    sub = ny.freesurfer_subject(str(fs_dir))
    im = to_image(image_clear(ny.load(str(t1w)), fill=0.0), dtype=np.int32)
    vol = sub.cortex_to_image(tuple(codes), im, hemi=None, method='nearest', fill=0)
    data = np.asarray(vol.dataobj).astype(np.int64)
    m = ((data & 1) > 0) & (((data >> 2) & 1) > 0)
    right = (data & 2) > 0
    out = OUT / f'sub-{subject}' / 'anat'
    out.mkdir(parents=True, exist_ok=True)
    for sfx, mm in (('', m), ('l', m & ~right), ('r', m & right)):
        nib.Nifti1Image(mm.astype(np.uint8), vol.affine).to_filename(
            str(out / f'sub-{subject}_space-T1w_desc-preSMA{sfx}_mask.nii.gz'))
    print(f'sub-{subject}: preSMA {int(m.sum())} mm3 (l {int((m & ~right).sum())}, '
          f'r {int((m & right).sum())})', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('subjects', nargs='*')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    labels = []
    for h in ('lh', 'rh'):
        lab = fsaverage_presma(h)
        np.save(OUT / f'fsaverage_preSMA_{h}.npy', lab)
        print(f'fsaverage {h}: {int(lab.sum())} vertices')
        labels.append(lab)
    subs = a.subjects or [p.name.removeprefix('sub-').removesuffix('_ses-1')
                          for p in sorted(SD.glob('sub-*_ses-1'))]
    for s in subs:
        project(s, labels)
