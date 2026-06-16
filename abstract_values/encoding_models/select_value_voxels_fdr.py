#!/usr/bin/env python3
"""Model-neutral NPCr voxel selection for the value model comparison.

For an unbiased single-vs-weighted comparison the voxel set must not be
picked by either model's own fit. This computes the **union** of the two
families' FDR-controlled sets: a voxel is selected if it passes FDR under
the whole-brain ``aprf`` (single) R2 mixture OR the ``aprf-weighted`` R2
mixture. Symmetric -> no relative bias; one shared set applied to every
model/condition by the plotter.

Voxel indices match the sweep's per-voxel cvR2 files (same NPCr mask +
betas-affine masker), so the plotter can join on ``voxel``.

Writes:
  derivatives/experiments/npc_value_sweep/sub-<S>/func/
    sub-<S>_task-abstractvalue_mask-NPCr_desc-voxelselect{_smoothed}.tsv

Usage:
  python -m abstract_values.encoding_models.select_value_voxels_fdr 03
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from abstract_values.utils.data import Subject, BIDS_FOLDER
from abstract_values.encoding_models.compute_r2_mixture import (
    get_brain_fdr_threshold)

FALLBACK_N = 100


def _roi_r2(masker, bids_folder, subject, model, smooth_label):
    p = (bids_folder / "derivatives" / "encoding_models" / model
         / f"sub-{subject}" / "func"
         / f"sub-{subject}_task-abstractvalue_space-T1w"
           f"_desc-r2{smooth_label}_pe.nii.gz")
    if not p.exists():
        return None
    return pd.Series(masker.transform(nib.load(str(p))).ravel()
                     .astype(np.float32))


def _passing(r2, model, subject, bids_folder, alpha, smoothed):
    """Boolean Series: voxels passing FDR under this model's whole-brain
    mixture. Degenerate/missing mixture -> top-N by this R2."""
    if r2 is None:
        return None
    res = get_brain_fdr_threshold(subject, model=model,
                                  bids_folder=bids_folder, alpha=alpha,
                                  smoothed=smoothed)
    if res is not None and not res["degenerate"] and np.isfinite(res["threshold"]):
        return r2 > res["threshold"], float(res["threshold"]), "fdr"
    top = r2.sort_values(ascending=False).index[:FALLBACK_N]
    mask = pd.Series(False, index=r2.index); mask.loc[top] = True
    return mask, float("nan"), "fallback-top%d" % FALLBACK_N


def run_one(subject, alpha=0.05, smoothed=False, bids_folder=BIDS_FOLDER):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    smooth_label = "_smoothed" if smoothed else ""

    mask_img = sub.get_roi_mask("NPCr", hemi=None)
    betas_img = sub.get_single_trial_estimates(sessions, desc="gabor",
                                               smoothed=smoothed)
    masker = NiftiMasker(mask_img=mask_img, target_affine=betas_img.affine,
                         target_shape=betas_img.shape[:3]).fit()

    aprf_r2 = _roi_r2(masker, bids_folder, subject, "aprf", smooth_label)
    wtd_r2 = _roi_r2(masker, bids_folder, subject, "aprf-weighted", smooth_label)
    n = (aprf_r2 if aprf_r2 is not None else wtd_r2).shape[0]

    a_pass = _passing(aprf_r2, "aprf", subject, bids_folder, alpha, smoothed)
    w_pass = _passing(wtd_r2, "aprf-weighted", subject, bids_folder, alpha, smoothed)

    aprf_pass = a_pass[0] if a_pass else pd.Series(False, index=range(n))
    wtd_pass = w_pass[0] if w_pass else pd.Series(False, index=range(n))
    union = aprf_pass.values | wtd_pass.values

    print(f"sub-{subject}: aprf pass={int(aprf_pass.sum())} "
          f"({a_pass[2] if a_pass else 'NA'}, thr={a_pass[1] if a_pass else float('nan'):.3f}) · "
          f"weighted pass={int(wtd_pass.sum())} "
          f"({w_pass[2] if w_pass else 'NA'}, thr={w_pass[1] if w_pass else float('nan'):.3f}) · "
          f"UNION={int(union.sum())}/{n}")

    out_dir = (bids_folder / "derivatives" / "experiments" / "npc_value_sweep"
               / f"sub-{subject}" / "func")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = (out_dir / f"sub-{subject}_task-abstractvalue_mask-NPCr"
                     f"_desc-voxelselect{smooth_label}.tsv")
    pd.DataFrame({
        "subject": subject, "voxel": np.arange(n),
        "aprf_r2": aprf_r2.values if aprf_r2 is not None else np.nan,
        "weighted_r2": wtd_r2.values if wtd_r2 is not None else np.nan,
        "aprf_pass": aprf_pass.values.astype(int),
        "weighted_pass": wtd_pass.values.astype(int),
        "in_union": union.astype(int),
    }).to_csv(out, sep="\t", index=False)
    print(f"  wrote {out.name}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("subject")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    args = p.parse_args()
    run_one(args.subject, alpha=args.alpha, smoothed=args.smoothed,
            bids_folder=args.bids_folder)


if __name__ == "__main__":
    main()
