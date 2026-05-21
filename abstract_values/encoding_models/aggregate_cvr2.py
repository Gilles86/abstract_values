#!/usr/bin/env python3
"""Aggregate per-run leave-one-run-out cvR² NIfTIs into a mean cvR² volume.

Each ``fit_*_cv.py`` script writes one NIfTI per held-out (session, run)
fold and a mean image alongside. That mean image is convenient, but it
inherits whatever dtype the masker emits (typically int8 via the
fmriprep BOLD mask — see CLAUDE.md "NIfTI dtype trap"), which quantises
the values to <300 distinct bins across the brain.

This script rebuilds the mean cleanly by:
  1. Globbing all per-fold ``..._run-*_desc-cvr2[_smoothed]_pe.nii.gz`` files
     for one (subject, model_dir, smoothing) combination.
  2. Stacking + averaging in float32.
  3. Writing the result back with ``set_data_dtype(np.float32)`` and
     ``set_slope_inter(slope=1, inter=0)`` so downstream surface sampling
     sees real values, not 16 quantisation bins.

Output path matches the existing mean image written by the fitters:
    derivatives/encoding_models/<model_dir>/sub-<subject>/func/
        sub-<subject>_task-abstractvalue_space-T1w_desc-cvr2[_smoothed]_pe.nii.gz

so existing readers (e.g. surface sampling) pick it up without changes.

The expected ``model_dir`` strings match the on-disk encoder dirs (note
the historical hyphen/dot drift):

    aprf.cv | aprf-shift.cv | aprf-weighted.cv | vonmises.cv

Usage
-----
    python -m abstract_values.encoding_models.aggregate_cvr2 \\
        --subject 08 --model-dir aprf.cv
    python -m abstract_values.encoding_models.aggregate_cvr2 \\
        --subject 08 --model-dir vonmises.cv --smoothed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from abstract_values.utils.data import BIDS_FOLDER


def find_per_run_cvr2(subject: str, model_dir: str, smoothed: bool,
                      bids_folder: Path) -> list[Path]:
    """All per-fold cvR² NIfTIs for one (subject, model_dir, smoothing)."""
    func = (Path(bids_folder) / "derivatives" / "encoding_models" / model_dir
            / f"sub-{subject}" / "func")
    suffix = "_smoothed" if smoothed else ""
    # Per-fold files always carry both ses- and run- entities (see fit_*_cv.py).
    pattern = (f"sub-{subject}_ses-*_task-abstractvalue"
               f"_space-T1w_run-*_desc-cvr2{suffix}_pe.nii.gz")
    return sorted(func.glob(pattern))


def aggregate(subject: str, model_dir: str, smoothed: bool,
              bids_folder: Path) -> Path:
    """Mean-aggregate per-run cvR² → one NIfTI; return the written path."""
    files = find_per_run_cvr2(subject, model_dir, smoothed, bids_folder)
    if not files:
        suffix = "_smoothed" if smoothed else ""
        raise FileNotFoundError(
            f"No per-run cvR² NIfTIs found for sub-{subject} in "
            f"{model_dir} (suffix='{suffix}'). Run the corresponding "
            f"fit_*_cv.py first."
        )
    print(f"sub-{subject} {model_dir}{' (smoothed)' if smoothed else ''}: "
          f"averaging {len(files)} per-fold cvR² volumes")

    first = nib.load(str(files[0]))
    affine = first.affine
    header = first.header
    shape = first.shape

    stack = np.empty((len(files),) + shape, dtype=np.float32)
    for i, p in enumerate(files):
        img = nib.load(str(p))
        if img.shape != shape:
            raise ValueError(
                f"Shape mismatch: {p.name} is {img.shape}, expected {shape}"
            )
        stack[i] = np.asarray(img.dataobj, dtype=np.float32)

    mean = np.nanmean(stack, axis=0).astype(np.float32)

    suffix = "_smoothed" if smoothed else ""
    out_path = (files[0].parent
                / f"sub-{subject}_task-abstractvalue_space-T1w"
                  f"_desc-cvr2{suffix}_pe.nii.gz")
    img = nib.Nifti1Image(mean, affine, header)
    # Guard against the fmriprep-mask dtype trap (see CLAUDE.md).
    img.set_data_dtype(np.float32)
    img.header.set_slope_inter(slope=1, inter=0)
    img.to_filename(str(out_path))
    print(f"  -> {out_path}")
    print(f"  mean cvR² range: [{float(mean.min()):.4f}, "
          f"{float(mean.max()):.4f}],  "
          f"unique values (rounded 1e-3): "
          f"{len(np.unique(np.round(mean, 3)))}")
    return out_path


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--subject", required=True,
                   help="Subject label without 'sub-' (e.g. 08)")
    p.add_argument("--model-dir", required=True,
                   help="On-disk encoder CV dir name "
                        "(aprf.cv, aprf-shift.cv, aprf-weighted.cv, vonmises.cv)")
    p.add_argument("--smoothed", action="store_true",
                   help="Aggregate the _smoothed cvR² files (default: unsmoothed)")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    args = p.parse_args()

    aggregate(args.subject, args.model_dir, args.smoothed,
              Path(args.bids_folder))


if __name__ == "__main__":
    main()
