#!/usr/bin/env python3
"""Subject-native vmPFC/OFC mask from FreeSurfer's Desikan-Killiany parcellation.

The older ``desc-vmPFCOFC`` mask (made 2026-08-18, 25 subjects) has no
generator in the repo, is right-lateralised (~70 % right hemisphere) and is
~25 % white matter / outside the brain, so it cannot be extended to the
later subjects. This one is reproducible and defined in each participant's
own anatomy: the fmriprep ``desc-aparcaseg_dseg`` labels

  medialorbitofrontal       1014 / 2014   (vmPFC / mOFC)
  lateralorbitofrontal      1012 / 2012   (lOFC)
  rostralanteriorcingulate  1026 / 2026   (pregenual ACC, the vmPFC
                                           "valuation" cluster of
                                           Bartra et al. 2013)

both hemispheres, cortical ribbon only, on the T1w grid of the aparcaseg.
decode_value.py resamples it to the BOLD grid (nearest neighbour).

Output: derivatives/masks/sub-<s>/anat/sub-<s>_space-T1w_desc-vmPFCOFCaparc_mask.nii.gz

Usage (cluster):
    python -m abstract_values.prepare.make_vmpfc_aparc_mask 03 04 pil01
    python -m abstract_values.prepare.make_vmpfc_aparc_mask --all
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from abstract_values.utils.data import BIDS_FOLDER

LABELS = {'medialorbitofrontal': (1014, 2014),
          'lateralorbitofrontal': (1012, 2012),
          'rostralanteriorcingulate': (1026, 2026)}
DESC = 'vmPFCOFCaparc'


def aparcaseg_path(bids_folder, subject):
    root = Path(bids_folder) / 'derivatives' / 'fmriprep' / f'sub-{subject}'
    hits = sorted(p for p in root.glob('**/anat/*desc-aparcaseg_dseg.nii.gz')
                  if 'space-' not in p.name)
    if not hits:
        raise FileNotFoundError(f'No aparcaseg for sub-{subject} under {root}')
    return hits[0]


def make_mask(subject, bids_folder=BIDS_FOLDER):
    seg = nib.load(str(aparcaseg_path(bids_folder, subject)))
    lab = np.asarray(seg.dataobj)
    ids = [i for pair in LABELS.values() for i in pair]
    mask = np.isin(lab, ids).astype(np.uint8)
    out = (Path(bids_folder) / 'derivatives' / 'masks' / f'sub-{subject}' / 'anat'
           / f'sub-{subject}_space-T1w_desc-{DESC}_mask.nii.gz')
    out.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(mask, seg.affine)
    img.set_data_dtype(np.uint8)
    img.to_filename(str(out))
    print(f'sub-{subject}: {int(mask.sum())} mm^3 -> {out}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('subjects', nargs='*')
    p.add_argument('--all', action='store_true')
    p.add_argument('--bids-folder', default=BIDS_FOLDER)
    a = p.parse_args()
    subs = a.subjects
    if a.all:
        subs = sorted(d.name[4:] for d in (Path(a.bids_folder) / 'derivatives'
                                           / 'glmsingle').glob('sub-*'))
    for s in subs:
        make_mask(s, a.bids_folder)


if __name__ == '__main__':
    main()
