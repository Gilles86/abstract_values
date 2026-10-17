"""Combine per-subject dropout-survey outputs into small summary files -- cluster-side.

Reads /scratch/gdehol/dropout_survey/<dataset>/sub-*/ and writes
  dropout_survey_rois.tsv         one row per (dataset, subject, ROI)
  dropout_survey_meta.tsv         one row per (dataset, subject): runs, aparcaseg source, ...
  dropout_survey_acquisition.tsv  one row per dataset (from acquisition.tsv)
  dropout_survey_maps.npz         group-mean normalised tSNR / mean EPI / FOV coverage,
                                  sagittal x=-5 and axial z=-14 MNI slices + template
plus the group 3-D maps as float32 NIfTIs in <scratch>/group/.

    python -m abstract_values.dropout_survey.combine --out-dir ~/git/abstract_values/notes/data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from abstract_values.dropout_survey.datasets import DATASETS, SCRATCH

TPL = Path("/shares/zne.uzh/containers/templateflow/tpl-MNI152NLin2009cAsym/"
           "tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz")
TPL_MASK = TPL.parent / "tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
SAG_X, AX_Z, COR_Y = -5.0, -14.0, 40.0
MIN_FOV = 0.5


def _mode(x):
    x = x.dropna().astype(str)
    if not len(x):
        return None
    vc = x.value_counts()
    return vc.index[0] if len(vc) == 1 else f"{vc.index[0]} ({vc.iloc[0]}/{len(x)})"


def acquisition_summary(path):
    a = pd.read_csv(path, sep="\t", dtype={"subject": str})
    rows = []
    for ds, g in a.groupby("dataset", sort=False):
        r = dict(dataset=ds, n_checked=len(g))
        for k in ["Manufacturer", "ManufacturersModelName", "MagneticFieldStrength",
                  "RepetitionTime", "EchoTime", "FlipAngle", "SliceThickness",
                  "SpacingBetweenSlices", "MultibandAccelerationFactor",
                  "ParallelReductionFactorInPlane", "PartialFourier", "PhaseEncodingDirection",
                  "SliceEncodingDirection", "TotalReadoutTime", "EffectiveEchoSpacing",
                  "matrix", "n_slices", "fmap_files", "sdc", "SequenceName",
                  "PulseSequenceDetails", "WaterFatShift", "header_source", "descrip"]:
            if k in g:
                r[k] = _mode(g[k])
        for k in ["vox_x", "vox_y", "vox_z", "tilt_scanner_deg", "pitch_scanner_deg", "tilt_mni_deg", "pitch_mni_deg"]:
            if k in g:
                v = pd.to_numeric(g[k], errors="coerce")
                r[k] = float(v.median()) if v.notna().any() else np.nan
                if k.startswith(("tilt", "pitch")) and v.notna().sum() > 1:
                    r[k + "_range"] = f"{v.min():.1f}-{v.max():.1f}"
        rows.append(r)
    return pd.DataFrame(rows)


def ijk(img, xyz):
    return np.round(np.linalg.solve(img.affine, np.r_[xyz, 1])[:3]).astype(int)


def main(out_dir, scratch):
    out_dir, scratch = Path(out_dir), Path(scratch)
    out_dir.mkdir(parents=True, exist_ok=True)
    (scratch / "group").mkdir(exist_ok=True)
    rois, metas = [], []
    for ds in DATASETS:
        for f in sorted((scratch / ds).glob(f"sub-*/{ds}_sub-*_rois.tsv")):
            rois.append(pd.read_csv(f, sep="\t", dtype={"subject": str}))
        for f in sorted((scratch / ds).glob("sub-*/sub-*_meta.json")):
            m = json.loads(f.read_text())
            metas.append(dict(dataset=ds, subject=m["subject"], n_runs=m["n_runs"],
                              runs=";".join(Path(r["bold"]).name for r in m["runs"]),
                              tr=m["runs"][0]["tr"], n_vols=m["runs"][0]["n_vols"],
                              drift=m["runs"][0]["drift"], aparcaseg_source=m["aparcaseg_source"],
                              cortex_tsnr=m["cortex_tsnr"], bold_masked=m["bold_masked"],
                              bold_grid_mm="x".join(f"{z:.2f}" for z in m["bold_grid_mm"]),
                              mni=(f.parent / f"sub-{m['subject']}_space-MNI152NLin2009cAsym"
                                   f"_res-2_desc-tsnrnorm_map.nii.gz").exists()))
    rois = pd.concat(rois, ignore_index=True)
    metas = pd.DataFrame(metas)
    rois.to_csv(out_dir / "dropout_survey_rois.tsv", sep="\t", index=False)
    metas.to_csv(out_dir / "dropout_survey_meta.tsv", sep="\t", index=False)
    print(metas.groupby("dataset").agg(n=("subject", "size"), mni=("mni", "sum"),
                                       masked=("bold_masked", "sum")), flush=True)

    tpl = nib.load(str(TPL))
    tmask = np.asarray(nib.load(str(TPL_MASK)).dataobj) > 0
    i, j, k = ijk(tpl, [SAG_X, COR_Y, AX_Z])
    slices = dict(tpl_sag=np.asarray(tpl.dataobj)[i], tpl_ax=np.asarray(tpl.dataobj)[:, :, k],
                  tpl_cor=np.asarray(tpl.dataobj)[:, j], mask_sag=tmask[i], mask_ax=tmask[:, :, k],
                  mask_cor=tmask[:, j], affine=tpl.affine, ijk=np.array([i, j, k]),
                  xyz=np.array([SAG_X, COR_Y, AX_Z]))
    names = []
    for ds in DATASETS:
        subs = sorted((scratch / ds).glob("sub-*/sub-*_space-MNI152NLin2009cAsym_res-2_desc-fov_map.nii.gz"))
        if not subs:
            continue
        names.append(ds)
        acc = {d: [] for d in ("tsnrnorm", "meanepi", "fov")}
        for f in subs:
            for d in acc:
                acc[d].append(np.asarray(nib.load(str(f).replace("desc-fov", f"desc-{d}")).dataobj,
                                         np.float32))
        fov = np.stack(acc["fov"]) > MIN_FOV
        cov = fov.mean(0)
        with np.errstate(invalid="ignore"):
            maps = {d: np.where(fov, np.stack(acc[d]), np.nan) for d in ("tsnrnorm", "meanepi")}
            maps = {d: np.nanmean(v, 0) for d, v in maps.items()}
        maps["coverage"] = cov
        for d, v in maps.items():
            v = np.where(tmask, v, np.nan).astype(np.float32)
            im = nib.Nifti1Image(np.nan_to_num(v), tpl.affine)
            im.set_data_dtype(np.float32)
            im.header.set_slope_inter(slope=1, inter=0)
            im.to_filename(str(scratch / "group" / f"{ds}_desc-{d}_group.nii.gz"))
            slices[f"{ds}_{d}_sag"] = v[i]
            slices[f"{ds}_{d}_ax"] = v[:, :, k]
            slices[f"{ds}_{d}_cor"] = v[:, j]
        slices[f"{ds}_n"] = len(subs)
        print(ds, len(subs), flush=True)
    slices["datasets"] = np.array(names)
    np.savez_compressed(out_dir / "dropout_survey_maps.npz", **slices)
    if (scratch / "acquisition.tsv").exists():
        acquisition_summary(scratch / "acquisition.tsv").to_csv(
            out_dir / "dropout_survey_acquisition.tsv", sep="\t", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=str(SCRATCH / "summary"))
    p.add_argument("--scratch", default=str(SCRATCH))
    a = p.parse_args()
    main(a.out_dir, a.scratch)
