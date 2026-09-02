#!/usr/bin/env python3
"""Compute per-voxel preferred orientation from the von Mises basis weights.

The vonmises encoding model fits, per voxel, weights over a fixed bank of
von Mises basis functions (``mus = linspace(0, π, n_basis)``, shared
``kappa``). The preferred orientation is the argmax of the reconstructed
tuning curve — exactly the recipe in
``visualize/preferred_tuning.py:_voxel_preferred_ori``.

Orientation is π-periodic, so a scalar preferred-orientation map cannot be
interpolated or averaged directly (175° and 5° must average to 0°, not 90°).
We therefore store the preferred orientation as its **doubled-angle unit
vector** components:

    desc-orient_cos = cos(2θ)
    desc-orient_sin = sin(2θ)

These interpolate and average correctly (sample to fsaverage with the usual
tools, take the circular mean, recover θ = ½·atan2(S̄, C̄)). The resultant
length √(C̄²+S̄²) doubles as an orientation-consistency measure.

Output (T1w, next to the weights):
  derivatives/encoding_models/<model>/sub-<s>/func/
    sub-<s>_task-abstractvalue_space-T1w_desc-orient_cos[_smoothed]_pe.nii.gz
    sub-<s>_task-abstractvalue_space-T1w_desc-orient_sin[_smoothed]_pe.nii.gz

Usage
-----
  python -m abstract_values.encoding_models.compute_preferred_orientation 07
  python -m abstract_values.encoding_models.compute_preferred_orientation 07 --smoothed
  # session-shift model: one (cos,sin) pair per condition (weights_1 / weights_2)
  python -m abstract_values.encoding_models.compute_preferred_orientation 07 \
      --model vonmises-session-shift --weights-descs weights_1 weights_2
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from abstract_values.utils.data import BIDS_FOLDER

# Match preferred_tuning.py exactly.
N_BASIS = 8
KAPPA = 2.0
ORI_GRID = np.deg2rad(np.linspace(7.5, 172.5, 200, dtype=np.float32))


def _basis():
    mus = np.linspace(0, np.pi, N_BASIS, endpoint=False, dtype=np.float32)
    b = np.exp(KAPPA * np.cos(2 * (ORI_GRID[:, None] - mus[None, :])))
    b /= b.sum(axis=0, keepdims=True)
    return b                                          # (200, n_basis)


def _save_f32(arr, affine, header, path):
    """Write a float32 NIfTI with slope=1 / inter=0 (avoids the uint8/scl
    quantisation trap; see CLAUDE.md)."""
    img = nib.Nifti1Image(arr.astype(np.float32), affine, header)
    img.set_data_dtype(np.float32)
    img.header.set_slope_inter(slope=1, inter=0)
    img.to_filename(str(path))


def compute_one(weights_path, out_cos, out_sin):
    """Reconstruct preferred orientation per voxel and write cos2θ / sin2θ."""
    basis = _basis()
    wimg = nib.load(str(weights_path))
    w = wimg.get_fdata().astype(np.float32)           # (X, Y, Z, n_basis)
    shape3 = w.shape[:3]
    w2 = w.reshape(-1, w.shape[-1])                    # (V, n_basis)

    # Voxels with all-zero / non-finite weights aren't fit — leave as 0 vector.
    valid = np.isfinite(w2).all(axis=1) & (np.abs(w2).sum(axis=1) > 0)
    theta = np.zeros(w2.shape[0], dtype=np.float32)
    if valid.any():
        curves = basis @ w2[valid].T                  # (200, n_valid)
        theta[valid] = ORI_GRID[np.argmax(curves, axis=0)]

    cos2 = np.where(valid, np.cos(2 * theta), np.nan).reshape(shape3)
    sin2 = np.where(valid, np.sin(2 * theta), np.nan).reshape(shape3)
    _save_f32(cos2, wimg.affine, wimg.header, out_cos)
    _save_f32(sin2, wimg.affine, wimg.header, out_sin)
    print(f"  {out_cos.name}  ({int(valid.sum())} fit voxels)")
    print(f"  {out_sin.name}")


def main(subject, model='vonmises', weights_descs=('weights',), smoothed=False,
         bids_folder=BIDS_FOLDER):
    bids_folder = Path(bids_folder)
    func = (bids_folder / 'derivatives' / 'encoding_models' / model
            / f'sub-{subject}' / 'func')
    smooth = '_smoothed' if smoothed else ''
    base = f'sub-{subject}_task-abstractvalue_space-T1w_desc-'

    for wdesc in weights_descs:
        wpath = func / f'{base}{wdesc}{smooth}_pe.nii.gz'
        if not wpath.exists():
            print(f"  WARNING: {wpath.name} not found, skipping")
            continue
        # weights -> orient ; weights_1 -> orient_1 ; weights_2 -> orient_2
        suffix = wdesc.replace('weights', '').lstrip('_')
        tag = f'orient_{suffix}' if suffix else 'orient'
        out_cos = func / f'{base}{tag}_cos{smooth}_pe.nii.gz'
        out_sin = func / f'{base}{tag}_sin{smooth}_pe.nii.gz'
        print(f"sub-{subject} {model} {wdesc}{smooth}:")
        compute_one(wpath, out_cos, out_sin)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('subject')
    p.add_argument('--model', default='vonmises',
                   help="Encoding-model dir holding desc-weights (default vonmises).")
    p.add_argument('--weights-descs', nargs='+', default=['weights'],
                   help="Weight desc-entities to reconstruct (default: weights; "
                        "for session-shift use: weights_1 weights_2).")
    p.add_argument('--smoothed', action='store_true')
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    args = p.parse_args()
    main(args.subject, model=args.model, weights_descs=tuple(args.weights_descs),
         smoothed=args.smoothed, bids_folder=args.bids_folder)
