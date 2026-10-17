"""Task-response detectability in the survey ROIs from GLMsingle outputs -- cluster-side.

For one (dataset, subject), on the GLM's own grid (T1w space; survey ROI labels
resampled with nearest neighbour):

  R2        GLMsingle typeD R2 (%, from the R2 NIfTI, else from
            TYPED_FITHRF_GLMDENOISE_RR.npy). Per ROI: mean R2, mean R2 minus the
            subject's eroded-white-matter mean (floor), and the fraction of ROI
            voxels above the subject's white-matter 95th percentile.
  betas     single-trial betas per event type (% signal change: GLMsingle
            betasmd, scaled by the voxel mean). Per ROI: the ROI-mean beta per
            trial, its mean across trials and the t across trials, and the
            fraction of voxels whose own across-trial t exceeds |3|.
  cortex    the same over all aparc cortical labels, to normalise across tasks.

Voxels count when the GLM produced finite, non-constant betas.

    python -m abstract_values.dropout_survey.glm numrisk 10
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image as nimage

from abstract_values.dropout_survey.collect import CORTEX, load_seg, roi_masks, to_grid
from abstract_values.dropout_survey.datasets import DATASETS, SCRATCH, SHARES, find_anat, find_runs
from abstract_values.visualize.collect_vmpfc_dropout import label_volume


@dataclass
class GLM:
    root: str            # derivatives folder
    sub_dir: str         # relative to root, {s} = subject label
    r2: str | None       # glob for the R2 NIfTI (None: read R2 from the npy)
    events: dict         # event name -> glob of the 4-D single-trial beta NIfTI
    note: str = ""


def _d(owner, ds, glm):
    return str(SHARES / owner / ds / "derivatives" / glm)


GLMS = {
    "abstractvalue": GLM(_d("gdehol", "ds-abstractvalue", "glmsingle"), "sub-{s}/func",
                         "*desc-R2_pe.nii.gz", {"gabor": "*desc-gabor_pe.nii.gz",
                                                "response": "*desc-response_pe.nii.gz"},
                         "glmsingle, both sessions"),
    "numloss": GLM(_d("aldavy", "ds-numloss", "glm_stim1.denoise.smoothed"), "sub-{s}/func",
                   "*desc-R2_pe.nii.gz", {"stim1": "*desc-stim1_pe.nii.gz",
                                          "stim2": "*desc-stim2_pe.nii.gz"},
                   "glm_stim1.denoise.smoothed (only GLM available; SMOOTHED)"),
    "foodvalue": GLM(_d("gdehol", "ds-foodvalue", "glmsingle"), "sub-{s}", "*desc-R2_pe.nii.gz",
                     {"view": "*desc-view_pe.nii.gz"}, "glmsingle"),
    "numrisk": GLM(_d("gdehol", "ds-numrisk", "glm_stim1.denoise"), "sub-{s}/func",
                   "*desc-r2_par.nii.gz", {"stim1": "*desc-stim1_pe.nii.gz",
                                           "stim2": "*desc-stim2_pe.nii.gz"}, "glm_stim1.denoise"),
    "dnumrisk": GLM(_d("mrenke", "ds-dnumrisk", "glm_stim1.denoise"), "sub-{s}/ses-1/func", None,
                    {"stims1": "*desc-stims1_pe.nii.gz"}, "glm_stim1.denoise, ses-1"),
    "stressrisk": GLM(_d("mrenke", "ds-stressrisk", "glm_stim1.denoise"), "sub-{s}/ses-1/func",
                      None, {"stims1": "*desc-stims1_pe.nii.gz"}, "glm_stim1.denoise, ses-1"),
    "neuralpriors": GLM(_d("gdehol", "ds-neuralpriors", "glm_stim1.denoise"), "sub-{s}/func",
                        "*desc-R2_pe.nii.gz", {"stim": "*desc-stim_pe.nii.gz",
                                               "response": "*desc-response_pe.nii.gz"},
                        "glm_stim1.denoise, both sessions"),
    "tmsrisk": GLM(_d("gdehol", "ds-tmsrisk", "glm_stim1.denoise"), "sub-{s}/ses-1/func", None,
                   {"stims1": "*desc-stims1_pe.nii.gz"}, "glm_stim1.denoise, ses-1"),
    "risk3t": GLM(str(SCRATCH / "ds-risk/derivatives/glm_stim1.denoise"), "sub-{s}/ses-3t2/func",
                  None, {"stims1": "*desc-stims1_pe.nii.gz"},
                  "glm_stim1.denoise ses-3t2, fetched from the department share"),
    "risk7t": GLM(str(SCRATCH / "ds-risk/derivatives/glm_stim1.denoise"), "sub-{s}/ses-7t2/func",
                  None, {"stims1": "*desc-stims1_pe.nii.gz"},
                  "glm_stim1.denoise ses-7t2, fetched from the department share"),
    "retsupp": GLM(_d("gdehol", "ds-retsupp", "glmsingle"), "sub-{s}/func", "*desc-R2_pe.nii.gz",
                   {"distractor": "*desc-distractor_pe.nii.gz"}, "glmsingle"),
}
ROIS = ["mOFC", "lOFC", "pgACC", "dmPFC", "dACC", "PCC", "pericalcarine", "superiorparietal",
        "WM", "cortex"]


def _one(d, pattern):
    hits = sorted(p for p in d.glob(pattern) if not p.name.startswith("._"))
    return hits[0] if hits else None


def r2_from_npy(d, ref):
    f = d / "TYPED_FITHRF_GLMDENOISE_RR.npy"
    if not f.exists():
        return None, "missing"
    r2 = np.asarray(np.load(f, allow_pickle=True).item()["R2"], np.float32)
    if r2.shape != ref.shape[:3]:
        return None, f"npy R2 shape {r2.shape} != grid {ref.shape[:3]}"
    return nib.Nifti1Image(r2, ref.affine), "npy"


def main(dataset, s, out_root):
    g = GLMS[dataset]
    ds = DATASETS[dataset]
    d = Path(g.root) / g.sub_dir.format(s=s)
    out = Path(out_root) / dataset / f"sub-{s}"
    out.mkdir(parents=True, exist_ok=True)
    betas = {e: _one(d, p) for e, p in g.events.items()}
    betas = {e: p for e, p in betas.items() if p is not None}
    if not betas:
        raise SystemExit(f"{dataset} sub-{s}: no beta files in {d}")
    first = nib.load(str(next(iter(betas.values()))))
    ref = nimage.index_img(first, 0)
    r2p = _one(d, g.r2) if g.r2 else None
    if r2p is not None:
        r2img, r2src = nib.load(str(r2p)), str(r2p)
    else:
        r2img, r2src = r2_from_npy(d, first)
    r2 = None
    if r2img is not None:
        r2 = np.asarray(to_grid(r2img, ref, "nearest").dataobj, np.float32) \
            if r2img.shape[:3] != ref.shape else np.asarray(r2img.dataobj, np.float32)

    runs = find_runs(ds, s, 2)
    anat = find_anat(ds, s, runs)
    seg = load_seg(anat)
    seg_g = np.asarray(to_grid(seg, ref, "nearest").dataobj).astype(np.int32)
    lab = np.asarray(to_grid(label_volume(seg), ref, "nearest").dataobj)
    masks = roi_masks(seg_g, lab)
    masks["cortex"] = np.isin(seg_g, CORTEX)

    rows, info = [], dict(dataset=dataset, subject=s, glm_dir=str(d), r2_source=r2src,
                          note=g.note, grid_mm=[float(z) for z in ref.header.get_zooms()[:3]],
                          events={})
    for e, p in betas.items():
        B = np.asarray(nib.load(str(p)).dataobj, np.float32)          # X, Y, Z, trials
        valid = np.isfinite(B).all(-1) & (B.std(-1) > 0)
        if r2 is not None:
            valid &= np.isfinite(r2)
        wm = masks["WM"] & valid
        wm_r2 = r2[wm] if r2 is not None else np.array([np.nan])
        floor, p95 = float(np.mean(wm_r2)), float(np.percentile(wm_r2, 95))
        info["events"][e] = dict(file=str(p), n_trials=int(B.shape[-1]),
                                 cortex_abs_beta_median=float(np.median(np.abs(
                                     B[masks["cortex"] & valid]))))
        for roi in ROIS:
            m = masks[roi] & valid
            n_all = int(masks[roi].sum())
            row = dict(dataset=dataset, subject=s, roi=roi, event=e, n_vox=int(m.sum()),
                       frac_vox_no_glm=float(1 - m.sum() / n_all) if n_all else np.nan,
                       n_trials=int(B.shape[-1]), wm_r2_mean=floor, wm_r2_p95=p95)
            if m.sum():
                if r2 is not None:
                    rv = r2[m]
                    row.update(r2_mean=float(rv.mean()), r2_above_floor=float(rv.mean() - floor),
                               frac_r2_above_wm95=float(np.mean(rv > p95)))
                X = B[m]                                               # voxels, trials
                y = X.mean(0)
                row.update(beta_mean=float(y.mean()),
                           beta_t=float(y.mean() / (y.std(ddof=1) / np.sqrt(len(y)))))
                tv = X.mean(1) / (X.std(1, ddof=1) / np.sqrt(X.shape[1]))
                row["frac_vox_abs_t3"] = float(np.mean(np.abs(tv) > 3))
            rows.append(row)
        del B
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{dataset}_sub-{s}_glm.tsv", sep="\t", index=False)
    (out / f"sub-{s}_glm_meta.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    print(df.round(3).to_string(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=list(GLMS))
    p.add_argument("subject")
    p.add_argument("--out-dir", default=str(SCRATCH))
    a = p.parse_args()
    main(a.dataset, a.subject, a.out_dir)
