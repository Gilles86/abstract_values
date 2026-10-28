#!/usr/bin/env python3
"""Leave-one-run-out cvR² of the single-PRF models in motor cortex (M1l + M1r, or pre-SMA).

The quantile / quantile-wrapped CV fits were run on the visual/parietal ROI
union only, so motor cortex has no fits for them. This runs the same
``fit_aprf_cv.main`` on a bilateral M1 mask, for both GLMsingle epochs
(gabor, response), and adds bid-space twins so value can be compared with the
reported bid on identical trials:

    all trials      null, gaussian (fixed CHF), linear, quantile (unwrapped),
                    quantile-wrapped, vonmises-prf (orientation)
    bid trials      null-bidmatched, gaussian-bid, gaussian-bidmatched,
                    linear-bid, linear-bidmatched,
                    quantile-bidmatched, quantile-wrapped-bidmatched

Outputs go to ``derivatives/followup_2026-10/a1_motor_stage/encoding_models/
<cv_out_subdir>[-response]/`` so no existing derivative is touched. (The first
run, 2026-10-28, wrote to ``derivatives/encoding_models/motor/`` and was moved
there afterwards.) The mask is
written as float32 so ``masker.inverse_transform`` does not quantise cvR² to a
uint8 mask's 256 levels.

    python -m abstract_values.encoding_models.fit_motor_cv 10 --desc response
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np

from abstract_values.encoding_models import fit_aprf_cv
from abstract_values.encoding_models.model_specs import MODEL_SPECS, _twin
from abstract_values.utils.data import BIDS_FOLDER

ALL_TRIALS = ['null', 'gaussian', 'linear', 'quantile', 'quantile-wrapped',
              'vonmises-prf']
BID_TRIALS = ['null-bidmatched', 'gaussian-bid', 'gaussian-bidmatched',
              'linear-bid', 'linear-bidmatched', 'quantile-bidmatched',
              'quantile-wrapped-bidmatched']
A1 = Path('derivatives') / 'followup_2026-10' / 'a1_motor_stage'
# roi set -> (mask files relative to the BIDS root, output prefix relative to encoding_models/)
ROI_SETS = {
    'M1': (['derivatives/masks/sub-{s}/anat/sub-{s}_space-T1w_desc-M1l_mask.nii.gz',
            'derivatives/masks/sub-{s}/anat/sub-{s}_space-T1w_desc-M1r_mask.nii.gz'],
           '../followup_2026-10/a1_motor_stage/encoding_models/'),
    'preSMA': ([str(A1 / 'masks/sub-{s}/anat/sub-{s}_space-T1w_desc-preSMA_mask.nii.gz')],
               '../followup_2026-10/a1_motor_stage/encoding_models_preSMA/'),
}


def register_specs(prefix):
    """Bid twins missing from the registry, then redirect every used spec
    to the follow-up output tree."""
    if 'gaussian-bid' not in MODEL_SPECS:
        MODEL_SPECS['gaussian-bid'] = _twin('gaussian', 'bid', stimulus_space='bid')
    for base in ('gaussian', 'quantile', 'quantile-wrapped'):
        if f'{base}-bidmatched' not in MODEL_SPECS:
            MODEL_SPECS[f'{base}-bidmatched'] = _twin(base, 'bidmatched')
    for name in ALL_TRIALS + BID_TRIALS:
        spec = MODEL_SPECS[name]
        if not spec.cv_out_subdir.startswith('../'):
            MODEL_SPECS[name] = replace(spec, cv_out_subdir=prefix + spec.cv_out_subdir)


def motor_mask(subject, bids_folder, roi_set='M1', tag=''):
    imgs = [nib.load(str(Path(bids_folder) / f.format(s=subject)))
            for f in ROI_SETS[roi_set][0]]
    data = np.zeros(imgs[0].shape, np.float32)
    for im in imgs:
        data[np.asarray(im.dataobj) > 0] = 1.0
    out = (Path(bids_folder) / 'derivatives' / 'followup_2026-10' / 'a1_motor_stage' / 'masks'
           / f'sub-{subject}_space-T1w_desc-{roi_set}fit{tag}_mask.nii.gz')
    out.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data, imgs[0].affine)
    img.set_data_dtype(np.float32)
    img.header.set_slope_inter(1, 0)
    img.to_filename(str(out))
    return str(out)


def main(subject, desc, models, bids_folder, n_iterations, skip_existing, roi_set='M1'):
    register_specs(ROI_SETS[roi_set][1])
    mask = motor_mask(subject, bids_folder, roi_set, tag=desc)  # one file per epoch: no write race
    for name in models:
        spec = MODEL_SPECS[name]
        sub_dir = spec.cv_out_subdir if desc == 'gabor' else f'{spec.cv_out_subdir}-{desc}'
        fn = (Path(bids_folder) / 'derivatives' / 'encoding_models' / sub_dir
              / f'sub-{subject}' / 'func'
              / f'sub-{subject}_task-abstractvalue_space-T1w_desc-cvr2_pe.nii.gz')
        if skip_existing and fn.exists():
            print(f'skip {name} ({desc}): {fn} exists')
            continue
        print(f'\n===== {name}  desc={desc} =====', flush=True)
        fit_aprf_cv.main(subject, n_iterations=n_iterations, mask=mask,
                         bids_folder=bids_folder, model_type=name, desc=desc)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject')
    ap.add_argument('--desc', default='gabor', choices=['gabor', 'response'])
    ap.add_argument('--models', nargs='+', default=ALL_TRIALS + BID_TRIALS)
    ap.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    ap.add_argument('--n-iterations', type=int, default=1000)
    ap.add_argument('--skip-existing', action='store_true')
    ap.add_argument('--roi-set', default='M1', choices=list(ROI_SETS))
    a = ap.parse_args()
    main(a.subject, a.desc, a.models, a.bids_folder, a.n_iterations, a.skip_existing,
         a.roi_set)
