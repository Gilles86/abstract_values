"""Preferred values of the log-Gaussian ('standard', aprf) and Gaussian
('gaussian', aprf-gauss) value tuning models, per voxel, for every participant.

Per voxel: both full-fit modes and FWHMs, both models' mean leave-one-run-out
cvR^2 and the null cvR^2, ROI membership. One row per (subject, smoothed,
voxel in any ROI). Run on the cluster; plot locally.

    python -m abstract_values.visualize.collect_mode_distribution \
        --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue \
        --out notes/data/mode_distribution_voxels.tsv.gz
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image as nimage

from abstract_values.encoding_models.aggregate_cvr2 import find_per_run_cvr2
from abstract_values.encoding_models.permnull_select import roi_masks
from abstract_values.utils.data import BIDS_FOLDER, Subject

MODELS = {'standard': ('aprf', 'aprf.cv'), 'gaussian': ('aprf-gauss', 'aprf-gauss.cv')}


def subjects(bids):
    d = bids / 'derivatives' / 'encoding_models' / 'aprf'
    return sorted(p.name[4:] for p in d.glob('sub-*'))


def collect(subject, smoothed, bids):
    sub = Subject(subject, bids_folder=bids)
    tag = '_smoothed' if smoothed else ''
    fn = lambda m, p: (bids / 'derivatives' / 'encoding_models' / MODELS[m][0]
                       / f'sub-{subject}' / 'func'
                       / f'sub-{subject}_task-abstractvalue_space-T1w_desc-{p}{tag}_pe.nii.gz')
    if not all(fn(m, 'mode').exists() for m in MODELS):
        return None
    ref = nib.load(str(fn('standard', 'mode')))
    rois = roi_masks(sub, ref)
    union = np.any(list(rois.values()), 0)
    cols = {}
    for m in MODELS:
        for p in ('mode', 'fwhm'):
            img = nimage.resample_to_img(nib.load(str(fn(m, p))), ref,
                                         interpolation='nearest', force_resample=True,
                                         copy_header=True)
            cols[f'{p}_{m}'] = img.get_fdata()[union]
    for m, cvdir in list(MODELS.items()) + [('null', (None, 'aprf-null.cv'))]:
        files = find_per_run_cvr2(subject, cvdir[1], smoothed, bids)
        if not files:
            return None
        arr = np.mean([nimage.resample_to_img(nib.load(str(f)), ref, interpolation='nearest',
                                              force_resample=True, copy_header=True)
                       .get_fdata()[union] for f in files], 0)
        cols[f'cvr2_{m}'] = arr
    df = pd.DataFrame(cols)
    for r, msk in rois.items():
        df[f'in_{r}'] = msk[union]
    return df.assign(subject=subject, smoothed=smoothed)


def main(bids, out):
    bids = Path(bids)
    rows = []
    for s in subjects(bids):
        for sm in (False, True):
            try:
                d = collect(s, sm, bids)
            except Exception as e:
                print(f'  {s} {sm}: {e}')
                continue
            if d is not None:
                rows.append(d)
                print(f'  {s} smoothed={sm}: {len(d)} voxels', flush=True)
    pd.concat(rows).to_csv(out, sep='\t', index=False, float_format='%.4g')
    print(f'wrote {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    ap.add_argument('--out', default='notes/data/mode_distribution_voxels.tsv.gz')
    a = ap.parse_args()
    main(a.bids_folder, a.out)
