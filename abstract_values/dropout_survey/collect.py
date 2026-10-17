"""Dropout metrics for one (dataset, subject) -- cluster-side reduce.

Same recipe for every dataset (abstractvalue included, recomputed here so the
numbers are comparable):

  runs      the first <= 2 T1w-space fmriprep runs (sorted by session, run)
  tSNR      voxel mean / SD of the residual after fmriprep's cosine drifts +
            6 motion parameters (DCT drifts with a 128 s cut-off when the
            confounds have no cosines), averaged over runs
  mean EPI  temporal mean / median of the temporal mean inside fmriprep's
            BOLD brain mask, averaged over runs
  voxels    in the FOV (non-zero in > 90 % of volumes in every run) and inside
            the BOLD brain mask or any aparcaseg label -- so dropout voxels
            that fmriprep's mask clipped are still counted
  cortex    the subject's whole-cortex median tSNR (all aparc cortical labels,
            1000-1035 / 2000-2035): ROI tSNR / cortex median is the protocol-
            normalised tSNR
  ROIs      mOFC, lOFC, pgACC (rostral ACC), dmPFC, dACC, PCC (labels as in
            visualize/collect_vmpfc_dropout.label_volume), pericalcarine,
            superior parietal, eroded white matter
  per ROI   median tSNR, normalised median tSNR, fraction tSNR < 20,
            fraction signal void (mean EPI < 0.5 x brain median), fraction of
            voxels outside the FOV (coverage) and outside the BOLD mask

Writes float32 maps on the BOLD grid (T1w space) to <out>/<dataset>/sub-<s>/
(tsnr, tsnrnorm, meanepi, fov), which the SLURM wrapper warps to MNI, plus a
per-ROI TSV and a JSON with what was used.

    python -m abstract_values.dropout_survey.collect numrisk 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image as nimage
from scipy.ndimage import binary_erosion

from abstract_values.dropout_survey.datasets import DATASETS, SCRATCH, find_anat, find_runs
from abstract_values.visualize.collect_vmpfc_dropout import LABEL_ID, label_volume, residualise

MOTION = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
EXTRA_APARC = {"pericalcarine": (1021, 2021), "superiorparietal": (1029, 2029)}
ROIS = ["mOFC", "lOFC", "pgACC", "dmPFC", "dACC", "PCC", "pericalcarine",
        "superiorparietal", "WM"]
CORTEX = list(range(1001, 1036)) + list(range(2001, 2036))
VOID = 0.5


def to_grid(img, ref, interp):
    return nimage.resample_to_img(img, ref, interpolation=interp, force_resample=True,
                                  copy_header=True)


def write(arr, affine, path):
    im = nib.Nifti1Image(np.nan_to_num(arr).astype(np.float32), affine)
    im.set_data_dtype(np.float32)
    im.header.set_slope_inter(slope=1, inter=0)
    im.to_filename(str(path))


def check_fs2t1w(path):
    """Max deviation of the fsnative->T1w ITK affine from identity (mm / unitless)."""
    if path is None:
        return None
    for line in Path(path).read_text().splitlines():
        if line.startswith("Parameters:"):
            p = np.array(line.split()[1:], float)
            return float(max(np.abs(p[:9] - np.eye(3).ravel()).max(), np.abs(p[9:]).max() / 10))
    return None


def load_seg(anat):
    """aparcaseg as a NIfTI on the fmriprep T1w grid."""
    seg = nib.load(str(anat["aparcaseg"]))
    if str(anat["aparcaseg"]).endswith(".mgz"):
        dev = check_fs2t1w(anat["fs2t1w"])
        if dev is not None and dev > 0.01:
            raise ValueError(f"fsnative->T1w is not identity (max dev {dev}); handle explicitly")
        seg = nib.Nifti1Image(np.asarray(seg.dataobj).astype(np.int32), seg.affine)
        seg = to_grid(seg, nib.load(str(anat["t1w"])), "nearest")
    return seg


def dct_drifts(n, tr, cutoff=128.0):
    k = int(np.floor(2 * n * tr / cutoff))
    t = np.arange(n)
    return np.column_stack([np.cos(np.pi * (t + 0.5) * j / n) for j in range(1, k + 1)]) \
        if k else np.zeros((n, 0))


def confounds(path, n, tr):
    c = pd.read_csv(path, sep="\t")
    cos = [k for k in c.columns if k.lower().startswith("cosine")]
    mot = [k for k in MOTION if k in c.columns]
    if len(mot) < 6:   # fmriprep < 1.2 naming
        alt = ["X", "Y", "Z", "RotX", "RotY", "RotZ"]
        mot = [k for k in alt if k in c.columns]
    assert len(mot) == 6, c.columns
    X = c[cos + mot].fillna(0).to_numpy()
    drift = "fmriprep cosines"
    if not cos:
        X, drift = np.column_stack([dct_drifts(n, tr), X]), "DCT 128 s"
    return X, drift, len(cos)


def run_maps(run, ref3, seg_bold):
    img = nib.load(str(run["bold"]))
    if img.shape[:3] != ref3.shape or not np.allclose(img.affine, ref3.affine, atol=1e-3):
        img = to_grid(img, ref3, "continuous")
    bm = np.asarray(to_grid(nib.load(str(run["mask"])), ref3, "nearest").dataobj) > 0
    data = np.asarray(img.dataobj)
    fov = (data != 0).mean(-1) > 0.9
    vox = fov & (bm | (seg_bold > 0))
    Y = data[vox].T.astype(np.float32)
    nonzero_out = float(fov[~bm].mean()) if (~bm).any() else 0.0
    del data
    meta_path = Path(str(run["bold"]).replace(".nii.gz", ".json"))
    tr = img.header.get_zooms()[3]
    if meta_path.exists():
        tr = float(json.loads(meta_path.read_text()).get("RepetitionTime", tr))
    X, drift, n_cos = confounds(run["confounds"], len(Y), tr)
    assert len(X) == len(Y), (len(X), len(Y))
    mu, res = residualise(Y, X)
    sd = res.std(0)
    tsnr = np.where(sd > 0, mu / np.maximum(sd, 1e-6), 0)
    out_t = np.full(vox.shape, np.nan, np.float32)
    out_m = np.full(vox.shape, np.nan, np.float32)
    out_t[vox] = tsnr
    out_m[vox] = mu / np.median(mu[bm[vox]])
    info = dict(bold=str(run["bold"]), confounds=str(run["confounds"]), tr=float(tr),
                n_vols=int(len(Y)), drift=drift, n_cosines=n_cos,
                frac_fov_outside_mask=nonzero_out)
    return out_t, out_m, fov, bm, info


def main(dataset, s, out_root, n_runs=2):
    ds = DATASETS[dataset]
    out = Path(out_root) / dataset / f"sub-{s}"
    out.mkdir(parents=True, exist_ok=True)
    runs = find_runs(ds, s, n_runs)
    anat = find_anat(ds, s, runs)
    if not runs or anat["aparcaseg"] is None or anat["t1w"] is None:
        raise SystemExit(f"{dataset} sub-{s}: missing inputs runs={len(runs)} anat={anat}")
    ref3 = nimage.index_img(nib.load(str(runs[0]["bold"])), 0)
    seg = load_seg(anat)
    seg_bold = np.asarray(to_grid(seg, ref3, "nearest").dataobj).astype(np.int32)
    lab = np.asarray(to_grid(label_volume(seg), ref3, "nearest").dataobj)

    T, M, F, B, infos = [], [], [], [], []
    for run in runs:
        print(f"{dataset} sub-{s} {run['bold'].name}", flush=True)
        t, m, f, b, info = run_maps(run, ref3, seg_bold)
        T.append(t), M.append(m), F.append(f), B.append(b), infos.append(info)
    fov = np.all(F, 0)
    bm = np.any(B, 0)
    with np.errstate(invalid="ignore"):
        tsnr = np.nanmean(T, 0)
        mean = np.nanmean(M, 0)
    tsnr[~fov], mean[~fov] = np.nan, np.nan

    cortex = np.isin(seg_bold, CORTEX) & np.isfinite(tsnr)
    cortex_med = float(np.median(tsnr[cortex]))
    masks = {"mOFC": lab == LABEL_ID["mOFC"], "lOFC": lab == LABEL_ID["lOFC"],
             "pgACC": lab == LABEL_ID["rACC"], "dmPFC": lab == LABEL_ID["dmPFC"],
             "dACC": lab == LABEL_ID["dACC"], "PCC": lab == LABEL_ID["PCC"],
             "WM": binary_erosion(np.isin(seg_bold, [2, 41]), iterations=1)}
    for k, v in EXTRA_APARC.items():
        masks[k] = np.isin(seg_bold, v)

    rows = []
    for r in ROIS:
        m = masks[r]
        n = int(m.sum())
        if not n:
            continue
        inf = m & np.isfinite(tsnr)
        tv, ev = tsnr[inf], mean[inf]
        rows.append(dict(dataset=dataset, subject=s, roi=r, n_vox=n,
                         frac_outside_fov=float(1 - inf.sum() / n),
                         frac_outside_mask=float(1 - (m & bm).sum() / n),
                         tsnr_median=float(np.median(tv)) if len(tv) else np.nan,
                         tsnr_norm=float(np.median(tv) / cortex_med) if len(tv) else np.nan,
                         frac_tsnr_below20=float(np.mean(tv < 20)) if len(tv) else np.nan,
                         epi_rel_median=float(np.median(ev)) if len(ev) else np.nan,
                         frac_void=float(np.mean(ev < VOID)) if len(ev) else np.nan,
                         cortex_tsnr=cortex_med))
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{dataset}_sub-{s}_rois.tsv", sep="\t", index=False)

    write(tsnr, ref3.affine, out / f"sub-{s}_space-T1w_desc-tsnr_map.nii.gz")
    write(tsnr / cortex_med, ref3.affine, out / f"sub-{s}_space-T1w_desc-tsnrnorm_map.nii.gz")
    write(mean, ref3.affine, out / f"sub-{s}_space-T1w_desc-meanepi_map.nii.gz")
    write(fov.astype(np.float32), ref3.affine, out / f"sub-{s}_space-T1w_desc-fov_map.nii.gz")
    zooms = ref3.header.get_zooms()[:3]
    meta = dict(dataset=dataset, subject=s, runs=infos, n_runs=len(runs),
                aparcaseg=str(anat["aparcaseg"]), aparcaseg_source=anat["aparcaseg_source"],
                fs2t1w_dev=check_fs2t1w(anat["fs2t1w"]),
                xfm=str(anat["xfm"]) if anat["xfm"] else None, cortex_tsnr=cortex_med,
                bold_grid_mm=[float(z) for z in zooms],
                # preproc BOLD that is zero outside fmriprep's mask cannot separate
                # FOV from mask: flag it
                bold_masked=bool(max(i["frac_fov_outside_mask"] for i in infos) < 0.01))
    (out / f"sub-{s}_meta.json").write_text(json.dumps(meta, indent=1))
    print(df.round(3).to_string(), flush=True)
    print(json.dumps({k: v for k, v in meta.items() if k != "runs"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=list(DATASETS))
    p.add_argument("subject")
    p.add_argument("--out-dir", default=str(SCRATCH))
    p.add_argument("--n-runs", type=int, default=2)
    a = p.parse_args()
    main(a.dataset, a.subject, a.out_dir, a.n_runs)
