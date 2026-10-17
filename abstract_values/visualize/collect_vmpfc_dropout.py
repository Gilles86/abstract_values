"""Signal dropout and evoked responses in vmPFC/OFC -- cluster-side reduce, one subject.

For one participant, over every run of both sessions:

  maps      temporal SNR (voxel mean / SD of the residual after fmriprep's
            cosine drifts + 6 motion parameters), averaged over runs, and the
            mean EPI divided by its brain median (susceptibility dropout).
            Written as float32 NIfTIs to derivatives/qa/tsnr/sub-<s>/ (T1w
            space, BOLD grid); the SLURM wrapper warps them to MNI.
  ROIs      vmPFC/OFC split into its three Desikan labels (medial OFC, lateral
            OFC, rostral ACC), the dorsal medial PFC (medial superior frontal
            gyrus anterior to the caudal ACC) and dorsal ACC (caudal ACC),
            with posterior cingulate (good-signal DMN node),
            left precentral (motor: the bid is given with the right hand), V1,
            IPS0, IPS1-2, IPS3-5 and eroded white matter as references:
            voxel tSNR histograms, relative EPI intensity, GLMsingle R2 and
            single-trial gabor / response-bar betas.
  epochs    ROI-mean %-signal time courses (confounds regressed out) sampled
            every 0.5 s around each gabor onset and each response-bar onset,
            one row per trial, with value, bid and RT.
  slices    sagittal / coronal / axial slices through the OFC of the T1w, the
            mean EPI and the tSNR map (both resampled onto the T1w grid) and
            the label map, for the figure.

    python -m abstract_values.visualize.collect_vmpfc_dropout 03 \\
        --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue \\
        --out-dir /scratch/gdehol/vmpfc_dropout
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

from abstract_values.utils.data import BIDS_FOLDER, Subject
from abstract_values.visualize.collect_vmpfc_value import trial_table

APARC = {"mOFC": (1014, 2014), "lOFC": (1012, 2012), "rACC": (1026, 2026),
         "dACC": (1002, 2002), "PCC": (1023, 2023, 1010, 2010), "Lprecentral": (1024,)}
# dmPFC has no Desikan label: medial superior frontal gyrus (superiorfrontal
# 1028/2028 voxels within DMPFC_MM of the other hemisphere's cortex), anterior
# to the centroid of the caudal ACC, which leaves out SMA / pre-SMA.
DMPFC_MM = 12.0
ATLAS = {"V1": ("BensonV1", "LR"), "IPS0": ("IPS0", None), "IPS12": ("IPS12", None),
         "IPS345": ("IPS345", None)}
ROIS = ["mOFC", "lOFC", "rACC", "dmPFC", "dACC", "PCC", "Lprecentral"] + list(ATLAS) + ["WM"]
LABEL_ID = {"mOFC": 1, "lOFC": 2, "rACC": 3, "PCC": 4, "dmPFC": 5, "dACC": 6,
            "Lprecentral": 7}
T_GABOR = np.arange(-4, 20.01, 0.5)
T_BAR = np.arange(-10, 16.01, 0.5)
TSNR_BINS = np.arange(0, 201, 4)
# sub-18 ses-1 runs 1-2 dropped scanner triggers: event timing is not usable
BAD_TIMING = {("18", 1, 1), ("18", 1, 2)}


def anat_file(bids, s, pattern):
    root = Path(bids) / "derivatives" / "fmriprep" / f"sub-{s}"
    hits = sorted(p for p in root.glob(f"**/anat/*{pattern}") if "space-" not in p.name
                  and "task-" not in p.name)
    if not hits:
        raise FileNotFoundError(f"No {pattern} under {root}")
    return hits[0]


def residualise(Y, conf):
    """Y (T, V) minus its projection on [1, confounds]; returns (mean, resid)."""
    X = np.column_stack([np.ones(len(conf)), conf - conf.mean(0)])
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return b[0], Y - X @ b


def label_volume(seg):
    """Integer ROI labels (LABEL_ID) on the aparcaseg's own T1w grid, incl. dmPFC."""
    from scipy.ndimage import distance_transform_edt
    a = np.asarray(seg.dataobj)
    out = np.zeros(a.shape, np.int8)
    for k, v in APARC.items():
        out[np.isin(a, v)] = LABEL_ID[k]
    vox = np.abs(np.diag(seg.affine)[:3])
    sf = {}
    for h, other in ((1028, (a >= 2000) & (a < 2100)), (2028, (a >= 1000) & (a < 1100))):
        d = distance_transform_edt(~other, sampling=vox)
        sf[h] = (a == h) & (d < DMPFC_MM)
    med = sf[1028] | sf[2028]
    # anterior to the caudal-ACC centroid, in world y (RAS)
    ijk = np.argwhere(med)
    y = nib.affines.apply_affine(seg.affine, ijk)[:, 1]
    cacc = nib.affines.apply_affine(seg.affine, np.argwhere(np.isin(a, APARC["dACC"])))[:, 1].mean()
    keep = ijk[y > cacc]
    out[tuple(keep.T)] = LABEL_ID["dmPFC"]
    return nib.Nifti1Image(out, seg.affine)


def roi_masks(sub, ref, labvol):
    lab = np.asarray(nimage.resample_to_img(labvol, ref, interpolation="nearest",
                                            force_resample=True, copy_header=True).dataobj)
    seg = nimage.resample_to_img(str(anat_file(sub.bids_folder, sub.subject_id,
                                               "desc-aparcaseg_dseg.nii.gz")),
                                 ref, interpolation="nearest", force_resample=True,
                                 copy_header=True)
    out = {k: lab == LABEL_ID[k] for k in LABEL_ID}
    lab = np.asarray(seg.dataobj)
    for k, (roi, hemi) in ATLAS.items():
        m = nimage.resample_to_img(sub.get_roi_mask(roi, hemi=hemi), ref,
                                   interpolation="nearest", force_resample=True,
                                   copy_header=True)
        out[k] = np.asarray(m.dataobj) > 0
    wm = binary_erosion(np.isin(lab, [2, 41]), iterations=1)
    idx = np.flatnonzero(wm)
    keep = np.random.default_rng(0).choice(idx, min(1500, len(idx)), replace=False)
    out["WM"] = np.zeros(wm.size, bool)
    out["WM"][keep] = True
    out["WM"] = out["WM"].reshape(wm.shape)
    return out


def confounds(func_dir, s, ses, run):
    c = pd.read_csv(func_dir / f"sub-{s}_ses-{ses}_task-abstractvalue_run-{run}"
                    f"_desc-confounds_timeseries.tsv", sep="\t")
    cols = [k for k in c.columns if k.startswith("cosine")] + \
        ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    return c[cols].fillna(0).to_numpy()


def slices(t1, epi, tsnr, lab):
    """Three orthogonal slices through the OFC, all on the (canonical) T1w grid."""
    vols = {"t1": t1, "epi": epi, "tsnr": tsnr, "lab": lab}
    vols = {k: np.asarray(nib.as_closest_canonical(v).dataobj) for k, v in vols.items()}
    L = vols["lab"]
    ofc = np.isin(L, [1, 2])
    lm = np.argwhere(L == 1)
    x = int(np.round(lm[lm[:, 0] < L.shape[0] / 2, 0].mean()))   # left mOFC, RAS: low x = left
    y = int(np.round(np.argwhere(ofc)[:, 1].mean()))
    z = int(np.round(lm[:, 2].mean()))
    out = {}
    for k, v in vols.items():
        dt = np.int8 if k == "lab" else np.float16
        out[f"{k}_sag"] = v[x].astype(dt)
        out[f"{k}_cor"] = v[:, y].astype(dt)
        out[f"{k}_ax"] = v[:, :, z].astype(dt)
    out["slice_xyz"] = np.array([x, y, z])
    out["voxsize"] = np.array(nib.as_closest_canonical(t1).header.get_zooms()[:3])
    return out


def main(s, bids, out_dir):
    bids, out_dir = Path(bids), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = Subject(s, bids_folder=bids)
    sessions = sorted(sub.get_sessions())
    first = sub.get_preprocessed_bold(sessions[0])[0]
    ref = nib.load(str(first))
    ref3 = nimage.index_img(ref, 0)
    seg_img = nib.load(str(anat_file(bids, s, "desc-aparcaseg_dseg.nii.gz")))
    labvol = label_volume(seg_img)
    masks = roi_masks(sub, ref3, labvol)

    tsnr_sum = mean_sum = None
    n_runs = 0
    ts_rows, epochs_g, epochs_b = [], [], []
    for ses in sessions:
        runs = sorted(sub.get_runs(ses))
        ev = sub.get_events(ses, runs)
        func_dir = Path(first).parent.parent / f"ses-{ses}" / "func"
        for run, path in zip(runs, sub.get_preprocessed_bold(ses, runs)):
            print(f"sub-{s} ses-{ses} run-{run}", flush=True)
            img = nib.load(str(path))
            tr = float(json.loads(Path(str(path).replace(".nii.gz", ".json")).read_text())
                       .get("RepetitionTime", img.header.get_zooms()[3]))
            bm = nimage.resample_to_img(
                str(path).replace("desc-preproc_bold", "desc-brain_mask"), ref3,
                interpolation="nearest", force_resample=True, copy_header=True)
            bm = np.asarray(bm.dataobj) > 0
            assert img.shape[:3] == bm.shape, (img.shape, bm.shape)
            Y = np.asarray(img.dataobj, dtype=np.float32)[bm].T        # (T, V)
            conf = confounds(func_dir, s, ses, run)
            assert len(conf) == len(Y), (len(conf), len(Y))
            mu, res = residualise(Y, conf)
            sd = res.std(0)
            tsnr = np.where(sd > 0, mu / np.maximum(sd, 1e-6), 0)
            full_t, full_m = np.zeros(bm.shape, np.float32), np.zeros(bm.shape, np.float32)
            full_t[bm], full_m[bm] = tsnr, mu / np.median(mu)
            tsnr_sum = full_t if tsnr_sum is None else tsnr_sum + full_t
            mean_sum = full_m if mean_sum is None else mean_sum + full_m
            n_runs += 1
            if (s, ses, run) in BAD_TIMING:
                continue
            psc = res / np.where(mu > 0, mu, np.inf) * 100                 # (T, V)
            vol_idx = np.full(bm.shape, -1)
            vol_idx[bm] = np.arange(bm.sum())
            roi_ts = []
            for r in ROIS:
                ii = vol_idx[masks[r] & bm]
                roi_ts.append(psc[:, ii].mean(1) if len(ii) else np.full(len(psc), np.nan))
            roi_ts = np.stack(roi_ts, 1)                                   # (T, R)
            t = (np.arange(len(psc)) + 0.5) * tr                           # slice-time ref 0.5
            e = ev.loc[run].reset_index()
            g = e[e.event_type == "gabor"].set_index("trial_nr")
            b = e[e.event_type == "response_bar"].set_index("trial_nr")
            for tn, row in g.iterrows():
                if tn not in b.index:
                    continue
                on_g, on_b = row.onset, b.loc[tn, "onset"]
                epochs_g.append(np.stack([np.interp(on_g + T_GABOR, t, roi_ts[:, k],
                                                    left=np.nan, right=np.nan)
                                          for k in range(len(ROIS))], 1))
                epochs_b.append(np.stack([np.interp(on_b + T_BAR, t, roi_ts[:, k],
                                                    left=np.nan, right=np.nan)
                                          for k in range(len(ROIS))], 1))
                ts_rows.append(dict(subject=s, session=ses, mapping=sub.get_mapping(ses),
                                    run=run, trial_nr=tn, orientation=row.orientation,
                                    value=row.value, bid=b.loc[tn, "bid"],
                                    rt=b.loc[tn, "duration"],
                                    bar_delay=on_b - on_g))
            del Y, res, psc

    tsnr_map = tsnr_sum / n_runs
    mean_map = mean_sum / n_runs
    qa = bids / "derivatives" / "qa" / "tsnr" / f"sub-{s}"
    qa.mkdir(parents=True, exist_ok=True)
    for name, arr in (("tsnr", tsnr_map), ("meanepi", mean_map)):
        im = nib.Nifti1Image(arr.astype(np.float32), ref3.affine)
        im.set_data_dtype(np.float32)
        im.header.set_slope_inter(slope=1, inter=0)
        im.to_filename(str(qa / f"sub-{s}_task-abstractvalue_space-T1w_desc-{name}_map.nii.gz"))

    # ROI summaries + tSNR histograms
    r2 = np.asarray(nib.load(str(bids / "derivatives" / "glmsingle" / f"sub-{s}" / "func"
                                 / f"sub-{s}_task-abstractvalue_space-T1w_desc-R2_pe.nii.gz")).dataobj)
    betas = np.asarray(sub.get_single_trial_estimates(sessions, desc="gabor").dataobj)
    try:
        rbetas = np.asarray(sub.get_single_trial_estimates(sessions, desc="response").dataobj)
    except FileNotFoundError:
        rbetas = None
    tt = trial_table(sub)
    assert betas.shape[-1] == len(tt)
    rows, hist = [], []
    for r in ROIS:
        m = masks[r]
        if not m.any():
            continue
        tv = tsnr_map[m]
        row = dict(subject=s, roi=r, n_vox=int(m.sum()),
                   tsnr_median=float(np.median(tv)), tsnr_mean=float(np.mean(tv)),
                   frac_tsnr_below20=float(np.mean(tv < 20)),
                   epi_rel=float(np.mean(mean_map[m])),
                   frac_epi_below50=float(np.mean(mean_map[m] < 0.5)),
                   glm_r2=float(np.nanmean(r2[m])))
        yg = np.nanmean(betas[m], 0)
        row["beta_gabor"] = float(np.nanmean(yg))
        row["beta_gabor_t"] = float(np.nanmean(yg) / (np.nanstd(yg) / np.sqrt(np.isfinite(yg).sum())))
        if rbetas is not None and rbetas.shape[-1] == len(tt):
            yr = np.nanmean(rbetas[m], 0)
            row["beta_resp"] = float(np.nanmean(yr))
            row["beta_resp_t"] = float(np.nanmean(yr) / (np.nanstd(yr) / np.sqrt(np.isfinite(yr).sum())))
        rows.append(row)
        h, _ = np.histogram(np.clip(tv, 0, TSNR_BINS[-1] - 1e-3), TSNR_BINS)
        hist.append(dict(subject=s, roi=r, **{f"b{int(lo)}": int(c) for lo, c in zip(TSNR_BINS[:-1], h)}))
    pd.DataFrame(rows).to_csv(out_dir / f"sub-{s}_rois.tsv", sep="\t", index=False)
    pd.DataFrame(hist).to_csv(out_dir / f"sub-{s}_tsnr_hist.tsv", sep="\t", index=False)
    pd.DataFrame(ts_rows).to_csv(out_dir / f"sub-{s}_trials.tsv", sep="\t", index=False)

    # display slices on the T1w grid
    t1 = nib.load(str(anat_file(bids, s, "desc-preproc_T1w.nii.gz")))
    lab_t1 = nimage.resample_to_img(labvol, t1, interpolation="nearest",
                                    force_resample=True, copy_header=True)
    labvol.to_filename(str(qa / f"sub-{s}_space-T1w_desc-mpfclabels_dseg.nii.gz"))
    to_t1 = lambda arr: nimage.resample_to_img(nib.Nifti1Image(arr.astype(np.float32), ref3.affine),
                                               t1, interpolation="continuous",
                                               force_resample=True, copy_header=True)
    sl = slices(t1, to_t1(mean_map), to_t1(tsnr_map), lab_t1)
    np.savez_compressed(out_dir / f"sub-{s}_data.npz", epochs_gabor=np.array(epochs_g, np.float32),
                        epochs_bar=np.array(epochs_b, np.float32), t_gabor=T_GABOR, t_bar=T_BAR,
                        rois=np.array(ROIS), tsnr_bins=TSNR_BINS, **sl)
    print(pd.DataFrame(rows).round(3).to_string(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subject")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--out-dir", default="/scratch/gdehol/vmpfc_dropout")
    a = p.parse_args()
    main(a.subject, a.bids_folder, a.out_dir)
