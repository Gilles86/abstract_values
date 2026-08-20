#!/usr/bin/env python3
"""Build a group NPCr mask in MNI152NLin2009cAsym space, for small-volume
FDR correction on the congruency_glm group maps.

Warps each subject's own T1w-space NPCr mask (derivatives/masks/sub-XX/anat/)
to MNI via the same ses-1 T1w->MNI transform used for the contrast maps
(NearestNeighbor interpolation -- it's a binary mask), then keeps voxels
present in at least `min_overlap` fraction of subjects.

Usage
-----
  python -m abstract_values.congruency_glm.build_group_npc_mask --subjects 03 04 ...
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from abstract_values.congruency_glm.normalize_to_mni import (
    ANTS_APPLY_TRANSFORMS, MNI_REFERENCE, get_t1w_to_mni_xfm)
from abstract_values.utils.data import BIDS_FOLDER


def warp_npc_mask(subject, bids_folder=BIDS_FOLDER, out_dir=None):
    src = (Path(bids_folder) / 'derivatives' / 'masks' / f'sub-{subject}' / 'anat'
           / f'sub-{subject}_space-T1w_desc-NPCr_mask.nii.gz')
    if not src.exists():
        print(f'  sub-{subject}: no NPCr mask ({src}) -- skipping')
        return None

    out_dir = Path(out_dir or Path(bids_folder) / 'derivatives' / 'congruency_glm' / 'group')
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f'sub-{subject}_space-MNI152NLin2009cAsym_desc-NPCr_mask.nii.gz'

    xfm = get_t1w_to_mni_xfm(subject, bids_folder)
    cmd = [ANTS_APPLY_TRANSFORMS, '-d', '3', '-i', str(src), '-r', MNI_REFERENCE,
           '-t', str(xfm), '-o', str(dst), '--interpolation', 'NearestNeighbor']
    subprocess.run(cmd, check=True)
    return dst


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--subjects', nargs='+', required=True)
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    p.add_argument('--min-overlap', type=float, default=0.25,
                   help='Keep voxels present in at least this fraction of subjects (default 0.25)')
    args = p.parse_args()

    warped = []
    for subject in args.subjects:
        dst = warp_npc_mask(subject, bids_folder=args.bids_folder)
        if dst is not None:
            warped.append(dst)

    print(f'warped {len(warped)}/{len(args.subjects)} subject NPCr masks')
    ref_img = nib.load(str(warped[0]))
    stack = np.zeros(ref_img.shape, dtype=np.float32)
    for fn in warped:
        stack += (nib.load(str(fn)).get_fdata() > 0).astype(np.float32)
    overlap = stack / len(warped)

    group_mask = (overlap >= args.min_overlap).astype(np.uint8)
    print(f'group NPCr mask (>= {args.min_overlap:.0%} overlap): {group_mask.sum()} voxels')

    out = (Path(args.bids_folder) / 'derivatives' / 'congruency_glm' / 'group'
           / f'group_n-{len(warped)}_space-MNI152NLin2009cAsym_desc-NPCr_mask.nii.gz')
    img = nib.Nifti1Image(group_mask, ref_img.affine)
    img.set_data_dtype(np.uint8)
    img.to_filename(str(out))
    print('wrote', out)


if __name__ == '__main__':
    main()
