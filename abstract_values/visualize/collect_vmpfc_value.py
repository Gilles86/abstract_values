"""Is there a value code in vmPFC/OFC? -- cluster-side reduce.

Per participant and ROI (vmPFC/OFC next to V1 and IPS0/IPS1-2/IPS3-5), writes
one row per (subject, roi) with:

  signal quality   n_vox (BOLD grid), mean boldref intensity relative to the
                   brain median (susceptibility dropout), temporal SNR of
                   ses-1 run-1 (preproc, T1w space), GLMsingle R2.
  encoding models  per model: median and mean of cvR2 - cvR2_null across ROI
                   voxels, fraction of voxels with cvR2 > cvR2_null, and the
                   mean over the ROI's top-100 voxels (ranked by the model's
                   own cvR2 - null, the same selection the decoder uses).
  univariate       the ROI-mean single-trial gabor beta, demeaned within run,
                   correlated with presented value and with the bid (classic
                   parametric "value signal"); plus the standardised value
                   coefficient with orientation covariates (theta, cos/sin 2
                   theta) -- the two mappings differ in shape, so a value term
                   that survives a shared orientation basis follows value, not
                   orientation. Same for the response-bar beta vs the bid.

Run on the cluster:

    srun -A zne.uzh -c4 --mem 32G --time 120 python -m \\
        abstract_values.visualize.collect_vmpfc_value \\
        --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue \\
        --out /scratch/gdehol/vmpfc_value_summary.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image as nimage

from abstract_values.utils.data import BIDS_FOLDER, Subject

ROIS = [("vmPFCOFCaparc", None, "vmPFC/OFC"),
        ("vmPFCOFC", None, "vmPFC/OFC (legacy)"),
        ("BensonV1", "LR", "V1"), ("IPS0", None, "IPS0"),
        ("IPS12", None, "IPS1-2"), ("IPS345", None, "IPS3-5")]
MODELS = {"linear": "aprf-linear.cv", "aprf": "aprf.cv",
          "wprf": "aprf-weighted.cv", "orientation": "vonmises-prf.cv",
          "orientation_basis": "vonmises.cv", "aprf_bid": "aprf-bid.cv"}
NULL = "aprf-null.cv"
TOPN = 100


def cvr2_path(bids, s, model):
    return (Path(bids) / "derivatives" / "encoding_models" / model / f"sub-{s}"
            / "func" / f"sub-{s}_task-abstractvalue_space-T1w_desc-cvr2_pe.nii.gz")


def trial_table(sub):
    """One row per GLMsingle gabor beta (session -> run -> onset), with bid."""
    rows = []
    for ses in sorted(sub.get_sessions()):
        runs = sub.get_runs(ses)
        ev = sub.get_events(ses, runs)
        for run in runs:
            e = ev.loc[run].reset_index()
            bids = (e[e.event_type == "response_bar"].set_index("trial_nr")["bid"])
            g = e[e.event_type == "gabor"].sort_values("onset")
            for r in g.itertuples():
                rows.append(dict(session=ses, run=run, trial_nr=r.trial_nr,
                                 orientation=float(r.orientation),
                                 value=float(r.value),
                                 bid=float(bids.get(r.trial_nr, np.nan))))
    return pd.DataFrame(rows)


def run_demean(x, groups):
    return x - pd.Series(x).groupby(groups).transform("mean").to_numpy()


def corr(y, x, groups):
    ok = np.isfinite(y) & np.isfinite(x)
    yd, xd = run_demean(y[ok], groups[ok]), run_demean(x[ok], groups[ok])
    return float(np.corrcoef(yd, xd)[0, 1])


def std_coef(y, x, covs, groups):
    """Standardised coefficient of x in y ~ x + covs, all run-demeaned."""
    ok = np.isfinite(y) & np.isfinite(x)
    cols = [x] + covs
    X = np.column_stack([run_demean(c[ok], groups[ok]) for c in cols])
    X = (X - X.mean(0)) / X.std(0)
    yd = run_demean(y[ok], groups[ok])
    yd = (yd - yd.mean()) / yd.std()
    b, *_ = np.linalg.lstsq(X, yd, rcond=None)
    resid = yd - X @ b
    dof = len(yd) - X.shape[1]
    se = np.sqrt(resid @ resid / dof * np.diag(np.linalg.inv(X.T @ X)))
    return float(b[0]), float(b[0] / se[0])


def roi_mask(sub, roi, hemi, ref):
    try:
        m = sub.get_roi_mask(roi, hemi=hemi)
    except FileNotFoundError:
        return None
    m = nimage.resample_to_img(m, ref, interpolation="nearest",
                               force_resample=True, copy_header=True)
    return np.asarray(m.dataobj) > 0


def main(bids, out):
    bids = Path(bids)
    subjects = sorted(d.name[4:] for d in (bids / "derivatives" / "glmsingle").glob("sub-*"))
    rows = []
    for s in subjects:
        sub = Subject(s, bids_folder=bids)
        print(f"sub-{s}", flush=True)
        ref = nib.load(str(cvr2_path(bids, s, NULL)))
        null = ref.get_fdata()
        cv = {k: nib.load(str(cvr2_path(bids, s, m))).get_fdata()
              for k, m in MODELS.items() if cvr2_path(bids, s, m).exists()}

        sessions = sorted(sub.get_sessions())
        betas = np.asarray(sub.get_single_trial_estimates(sessions, desc="gabor").dataobj)
        try:
            rbetas = np.asarray(sub.get_single_trial_estimates(sessions, desc="response").dataobj)
        except FileNotFoundError:
            rbetas = None
        r2 = nib.load(str(bids / "derivatives" / "glmsingle" / f"sub-{s}" / "func"
                         / f"sub-{s}_task-abstractvalue_space-T1w_desc-R2_pe.nii.gz")).get_fdata()
        tt = trial_table(sub)
        assert len(tt) == betas.shape[-1], (len(tt), betas.shape)
        if rbetas is not None and rbetas.shape[-1] != len(tt):
            rbetas = None
        groups = (tt.session.astype(str) + "_" + tt.run.astype(str)).to_numpy()
        th = np.deg2rad(tt.orientation.to_numpy())
        covs = [th, np.cos(2 * th), np.sin(2 * th)]

        # signal quality: boldref (T1w) and tSNR of ses-1 run-1
        fdir = bids / "derivatives" / "fmriprep" / f"sub-{s}" / f"ses-{sessions[0]}" / "func"
        run1 = sorted(sub.get_runs(sessions[0]))[0]
        stem = f"sub-{s}_ses-{sessions[0]}_task-abstractvalue_run-{run1}_space-T1w"
        bref = nimage.resample_to_img(str(fdir / f"{stem}_boldref.nii.gz"), ref,
                                      force_resample=True, copy_header=True).get_fdata()
        bmask = nimage.resample_to_img(str(fdir / f"{stem}_desc-brain_mask.nii.gz"), ref,
                                       interpolation="nearest", force_resample=True,
                                       copy_header=True).get_fdata() > 0
        brain_med = np.median(bref[bmask])
        bold = nib.load(str(fdir / f"{stem}_desc-preproc_bold.nii.gz"))
        same_grid = np.allclose(bold.affine, ref.affine) and bold.shape[:3] == ref.shape

        for roi, hemi, label in ROIS:
            m = roi_mask(sub, roi, hemi, ref)
            if m is None or m.sum() == 0:
                continue
            row = dict(subject=s, roi=roi, label=label, n_vox=int(m.sum()),
                       boldref_rel=float(np.mean(bref[m]) / brain_med),
                       glm_r2=float(np.nanmean(r2[m])))
            if same_grid:
                ts = np.asarray(bold.dataobj)[m].astype(np.float32)
                row["tsnr"] = float(np.nanmedian(ts.mean(1) / np.maximum(ts.std(1), 1e-6)))
            for k, c in cv.items():
                d = (c - null)[m]
                d = d[np.isfinite(d)]
                row[f"{k}_dmed"] = float(np.median(d))
                row[f"{k}_dmean"] = float(np.mean(d))
                row[f"{k}_frac"] = float(np.mean(d > 0))
                row[f"{k}_top"] = float(np.mean(np.sort(d)[-TOPN:]))
            y = np.nanmean(betas[m], 0)
            row["r_value"] = corr(y, tt.value.to_numpy(), groups)
            row["r_bid"] = corr(y, tt.bid.to_numpy(), groups)
            row["r_orientation"] = corr(y, tt.orientation.to_numpy(), groups)
            row["b_value_ori"], row["t_value_ori"] = std_coef(
                y, tt.value.to_numpy(), covs, groups)
            row["b_bid_ori"], row["t_bid_ori"] = std_coef(
                y, tt.bid.to_numpy(), covs, groups)
            if rbetas is not None:
                yr = np.nanmean(rbetas[m], 0)
                row["r_resp_bid"] = corr(yr, tt.bid.to_numpy(), groups)
                row["r_resp_value"] = corr(yr, tt.value.to_numpy(), groups)
            rows.append(row)
        del bold
    df = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(df.groupby("label").mean(numeric_only=True).T.round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--out", default="notes/data/vmpfc_value_summary.tsv")
    a = p.parse_args()
    main(a.bids_folder, a.out)
