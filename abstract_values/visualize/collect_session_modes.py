"""Per-voxel preferred value in each mapping, for the hierarchical lambda model.

From the session-shift value-tuning fits (preferred value free per session,
width/amplitude/baseline shared): log-Gaussian ('lognormal',
aprf-session-shift) and Gaussian ('gauss', aprf-gauss-session-shift). One row
per (subject, smoothed, voxel in any ROI): both models' preferred value under
the Misaligned (cdf) and Aligned (inverse cdf) mapping, the standard aPRF and
null mean leave-one-run-out cvR^2 (the gate), plus the session-shift (cv_shift)
and quantile-axis (cv_quant) aPRF cvR^2 for selection that does not favour
either code, the voxel's world x coordinate
and hemisphere (x relative to the brain-mask centroid), ROI membership.

    python -m abstract_values.visualize.collect_session_modes \
        --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue \
        --out notes/data/session_modes_voxels.tsv.gz

With --split odd|even: the log-Gaussian session-shift fit on half the runs
(``aprf-session-shift.runs-<half>``, unsmoothed), with the full-data cvR^2
columns as the gate, for the split-half reliability test.
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

MODELS = {'lognormal': 'aprf-session-shift', 'gauss': 'aprf-gauss-session-shift'}


def on(img, ref):
    return nimage.resample_to_img(img, ref, interpolation='nearest', force_resample=True,
                                  copy_header=True).get_fdata()


def collect(subject, smoothed, bids, MODELS=MODELS):
    sub = Subject(subject, bids_folder=bids)
    tag = '_smoothed' if smoothed else ''
    em = bids / 'derivatives' / 'encoding_models'
    fn = lambda m, p: (em / MODELS[m] / f'sub-{subject}' / 'func'          # noqa: E731
                       / f'sub-{subject}_task-abstractvalue_space-T1w_desc-{p}{tag}_pe.nii.gz')
    if not all(fn(m, f'mode_{k}').exists() for m in MODELS for k in (1, 2)):
        return None
    first = sub.get_mapping(1)
    ref = nib.load(str(fn('lognormal', 'mode_1')))
    rois = roi_masks(sub, ref)
    union = np.any(list(rois.values()), 0)
    cols = {}
    for m in MODELS:
        m1, m2 = (on(nib.load(str(fn(m, f'mode_{k}'))), ref)[union] for k in (1, 2))
        cols[f'{m}_cdf'], cols[f'{m}_inv'] = (m1, m2) if first == 'cdf' else (m2, m1)
    for key, cvdir in (('cv_std', 'aprf.cv'), ('cv_null', 'aprf-null.cv'),
                       ('cv_shift', 'aprf-shift.cv'), ('cv_quant', 'aprf-quantile.cv')):
        files = find_per_run_cvr2(subject, cvdir, smoothed, bids)
        if not files:
            return None
        cols[key] = np.mean([on(nib.load(str(f)), ref)[union] for f in files], 0)
    brain = on(sub.get_brain_mask(sorted(sub.get_sessions())[0]), ref) > 0
    ijk = np.argwhere(union)
    xyz = nib.affines.apply_affine(ref.affine, ijk)
    x0 = nib.affines.apply_affine(ref.affine, np.argwhere(brain).mean(0))[0]
    df = pd.DataFrame(cols).assign(x=xyz[:, 0], hemi=np.where(xyz[:, 0] < x0, 'L', 'R'))
    for r, msk in rois.items():
        df[f'in_{r}'] = msk[union]
    return df.assign(subject=subject, smoothed=smoothed, first=first)


def main(bids, out, split=None):
    bids = Path(bids)
    models = MODELS if split is None else {'lognormal': f'{MODELS["lognormal"]}.runs-{split}'}
    subs = sorted(p.name[4:] for p in (bids / 'derivatives' / 'encoding_models'
                                       / models['lognormal']).glob('sub-*'))
    rows = []
    for s in subs:
        for sm in ((False, True) if split is None else (False,)):
            try:
                d = collect(s, sm, bids, models)
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
    ap.add_argument('--out', default='notes/data/session_modes_voxels.tsv.gz')
    ap.add_argument('--split', default=None, choices=['odd', 'even'])
    a = ap.parse_args()
    main(a.bids_folder, a.out, a.split)
