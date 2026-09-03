#!/usr/bin/env python3
"""
NPC value model comparison: non-linear single pRF vs weighted basis,
crossed with how the two mapping conditions are handled.

Question
--------
For the abstract *value* (CHF) representation in NPCr, two things are
being compared at once, by leave-one-run-out cvR2:

  Model family
    * single   -- one non-linear log-Gaussian pRF per voxel (mode =
                  preferred value, fwhm = width); the constrained
                  "preferred-value" model (grid + Adam).
    * weighted -- a fixed bank of k log-Gaussian basis pRFs spanning the
                  value range, OLS weights per voxel; flexible, swept
                  over k x fwhm.

  Condition handling (condition == session here: each session is one
  mapping, counterbalanced across subjects)
    * joint    -- pool both conditions (assumes value tuning is
                  mapping-invariant / abstract).
    * shift    -- preferred value shifts per condition, tuning shape
                  shared (single-pRF: session-shift model). The project's
                  hypothesis.
    * separate -- fit each condition independently (max flexibility;
                  halves data and confounds with session/time).

If joint wins, the value code is abstract; if shift/separate wins, it is
mapping-dependent. The held-out fit decides, rather than assuming.

Voxel selection is fixed (independent of the swept model): the joint
`aprf` (standard LogGaussian) R2 map, top-N + R2>thr. cvR2 is reported
over all NPCr / R2>thr / top-N, against the true null (predict train
mean). Per-voxel cvR2 + null + value-R2 are saved for the plotter.

Outputs (one job per subject; loops the grid internally)
--------------------------------------------------------
  derivatives/experiments/npc_value_sweep/sub-<S>/func/
    sub-<S>_task-abstractvalue_mask-NPCr_desc-cvr2summary{_smoothed}.tsv
    sub-<S>_..._desc-cvr2voxels{_smoothed}.tsv
    sub-<S>_..._desc-nullcvr2{_smoothed}.tsv

Usage
-----
  python -m abstract_values.encoding_models.sweep_npc_value 03
  python -m abstract_values.encoding_models.sweep_npc_value 03 \
      --n-basis 4 6 8 12 16 20 --fwhm 2 4 6 10
"""
from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import LogGaussianPRF
from braincoder.optimize import WeightFitter
from braincoder.utils import get_rsq

from abstract_values.utils.data import Subject, BIDS_FOLDER
from abstract_values.encoding_models.model_specs import get_spec
from abstract_values.encoding_models.fit_pipeline import fit_one_model

DEFAULT_N_BASIS = (4, 6, 8, 12, 16, 20)
DEFAULT_FWHM = (2.0, 4.0, 6.0, 10.0)          # absolute CHF
# Ridge penalty for the closed-form weights. This axis was missing entirely,
# so every weighted-basis number this script has ever produced came from an
# unregularised solve. The orientation-space sweep found alpha=10 optimal by a
# wide margin (29/29 subjects), but the value basis is a different problem --
# bounded 2-42 CHF range, skewed stimulus distribution, log-Gaussian bumps
# rather than wrapped von Mises -- so it gets its own decade grid rather than
# inheriting the answer.
DEFAULT_ALPHA = (0.01, 0.1, 1.0, 10.0, 100.0)
SINGLE_COND_MODES = ("joint", "shift", "separate")
BASIS_COND_MODES = ("joint", "separate")      # 'shift' undefined for fixed basis


def get_value_paradigm(sub, sessions):
    """Index (session, run, trial); cols x (CHF), session_idx (0/1),
    condition (cdf/inverse_cdf). Row order matches gabor betas."""
    rows = []
    for sidx, session in enumerate(sorted(sessions)):
        cond = sub.get_mapping(session)
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            run_ev = events.loc[run].reset_index().sort_values("onset")
            for _, row in run_ev[run_ev["event_type"] == "gabor"].iterrows():
                rows.append({"session": session, "run": run,
                             "x": float(row["value"]),
                             "session_idx": float(sidx), "condition": cond})
    df = pd.DataFrame(rows)
    df.index = pd.MultiIndex.from_frame(
        df[["session", "run"]].assign(
            trial=df.groupby(["session", "run"]).cumcount()),
        names=["session", "run", "trial"])
    return df


def _folds(paradigm):
    s = paradigm.index.get_level_values("session")
    r = paradigm.index.get_level_values("run")
    return sorted(set(zip(s, r))), s, r


def _null_cvr2(data, paradigm):
    """Per-voxel LOO cvR2 of the true null (predict training mean)."""
    folds, s, r = _folds(paradigm)
    per_fold = []
    for ts, tr in folds:
        test_mask = (s == ts) & (r == tr)
        tmean = data.loc[~test_mask].mean(axis=0)
        td = data.loc[test_mask]
        pred = pd.DataFrame(np.broadcast_to(tmean.values, td.shape),
                            index=td.index, columns=td.columns)
        per_fold.append(get_rsq(td, pred))
    return pd.concat(per_fold, axis=1).mean(axis=1)


def _cv_single(data, paradigm, cond_mode, value_min, value_max, n_iter):
    """LOO cvR2 for the non-linear single log-Gaussian pRF.
    cond_mode: joint | shift | separate."""
    folds, s, r = _folds(paradigm)
    spec = get_spec("session-shift" if cond_mode == "shift" else "standard")
    per_fold = []
    for ts, tr in folds:
        test_mask = (s == ts) & (r == tr)
        if cond_mode == "separate":           # condition == session
            train_mask = (~test_mask) & (s == ts)
        else:
            train_mask = ~test_mask
        train_data = data.loc[train_mask].reset_index(drop=True)
        test_data = data.loc[test_mask].reset_index(drop=True)
        if cond_mode == "shift":
            cols = ["x", "session_idx"]
            train_par = (paradigm.loc[train_mask, cols]
                         .rename(columns={"session_idx": "session"})
                         .reset_index(drop=True))
            test_par = (paradigm.loc[test_mask, cols]
                        .rename(columns={"session_idx": "session"})
                        .reset_index(drop=True))
        else:
            train_par = paradigm.loc[train_mask, ["x"]].reset_index(drop=True)
            test_par = paradigm.loc[test_mask, ["x"]].reset_index(drop=True)
        pars, _ = fit_one_model(spec, train_data, train_par,
                                value_min=value_min, value_max=value_max,
                                n_iterations=n_iter, log_prefix="      ")
        model = spec.cls(**spec.cls_kwargs)
        pred = model.predict(parameters=pars, paradigm=test_par)
        pred.index = test_data.index
        per_fold.append(get_rsq(test_data, pred))
    return pd.concat(per_fold, axis=1).mean(axis=1)


def _cv_weighted(data, paradigm, n_basis, fwhm, cond_mode, alpha,
                 value_min, value_max):
    """LOO cvR2 for the weighted log-Gaussian basis (OLS weights)."""
    modes = np.linspace(value_min, value_max, n_basis).astype(np.float32)
    basis_pars = pd.DataFrame({
        "mode": modes,
        "fwhm": np.full(n_basis, fwhm, dtype=np.float32),
        "amplitude": np.ones(n_basis, dtype=np.float32),
        "baseline": np.zeros(n_basis, dtype=np.float32)})
    model = LogGaussianPRF(parameterisation="mode_fwhm_natural")
    folds, s, r = _folds(paradigm)
    per_fold = []
    for ts, tr in folds:
        test_mask = (s == ts) & (r == tr)
        train_mask = ((~test_mask) & (s == ts)) if cond_mode == "separate" \
            else ~test_mask
        train_data = data.loc[train_mask].reset_index(drop=True)
        train_par = paradigm.loc[train_mask, ["x"]].reset_index(drop=True)
        test_data = data.loc[test_mask].reset_index(drop=True)
        test_par = paradigm.loc[test_mask, ["x"]].reset_index(drop=True)
        w = WeightFitter(model, basis_pars, train_data,
                         train_par).fit(alpha=alpha)
        bp = model.basis_predictions(test_par, basis_pars)
        pred = pd.DataFrame(bp @ w.values, index=test_data.index,
                            columns=test_data.columns)
        per_fold.append(get_rsq(test_data, pred))
    return pd.concat(per_fold, axis=1).mean(axis=1)


def run_one(subject, n_basis_list, fwhm_list, alpha_list=DEFAULT_ALPHA,
            roi='NPCr', roi_hemi=None, r2_thr=0.05, top_n=100,
            n_iter=500, smoothed=False, bids_folder=BIDS_FOLDER):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    smooth_label = "_smoothed" if smoothed else ""
    print(f"sub-{subject}  sessions={sessions}  roi={roi}  "
          f"n_basis={list(n_basis_list)}  fwhm={list(fwhm_list)}  "
          f"alpha={list(alpha_list)}  n_iter={n_iter}")

    mask_img = sub.get_roi_mask(roi, hemi=roi_hemi)
    betas_img = sub.get_single_trial_estimates(sessions, desc="gabor",
                                               smoothed=smoothed)
    masker = NiftiMasker(mask_img=mask_img, target_affine=betas_img.affine,
                         target_shape=betas_img.shape[:3]).fit()
    paradigm = get_value_paradigm(sub, sessions)
    data = pd.DataFrame(masker.transform(betas_img).astype(np.float32),
                        index=paradigm.index)
    n_roi = data.shape[1]
    value_min, value_max = float(paradigm["x"].min()), float(paradigm["x"].max())
    assert betas_img.shape[3] == len(paradigm), "beta/paradigm mismatch"
    print(f"  {n_roi} {roi} voxels · {len(paradigm)} trials · "
          f"value {value_min:.0f}-{value_max:.0f} CHF")

    # independent voxel R2 from the joint aprf (standard) fit
    r2_path = (bids_folder / "derivatives" / "encoding_models" / "aprf"
               / f"sub-{subject}" / "func"
               / f"sub-{subject}_task-abstractvalue_space-T1w"
                 f"_desc-r2{smooth_label}_pe.nii.gz")
    if r2_path.exists():
        value_r2 = pd.Series(masker.transform(nib.load(str(r2_path))).ravel()
                             .astype(np.float32), index=data.columns)
    else:
        value_r2 = pd.Series(np.nan, index=data.columns)
        print("  (joint aprf R2 missing -> top-N / R2>thr fall back to all)")

    null = _null_cvr2(data, paradigm)

    out_dir = (bids_folder / "derivatives" / "experiments" / "npc_value_sweep"
               / f"sub-{subject}" / "func")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"sub-{subject}_task-abstractvalue_mask-{roi}"
    p_cv = out_dir / f"{base}_desc-cvr2summary{smooth_label}.tsv"
    p_vox = out_dir / f"{base}_desc-cvr2voxels{smooth_label}.tsv"
    p_null = out_dir / f"{base}_desc-nullcvr2{smooth_label}.tsv"
    pd.DataFrame({"subject": subject, "voxel": null.index,
                  "null_cvr2": null.values,
                  "value_r2": value_r2.reindex(null.index).values}
                 ).to_csv(p_null, sep="\t", index=False)

    if value_r2.notna().any():
        sel = value_r2[value_r2 > r2_thr].index
        top = value_r2.sort_values(ascending=False).index[:top_n]
    else:
        sel = top = data.columns
    print(f"  {len(sel)} voxels R2>{r2_thr} · top-{top_n} for the figure")

    cv_cols = ["subject", "model", "cond", "n_basis", "fwhm", "alpha",
               "mean_cvr2_top", "mean_cvr2_sel", "mean_cvr2_all", "n_top",
               "frac_beats_null_sel", "frac_beats_null_all",
               "median_margin_sel"]

    def _summary_row(model, cond, cvr2, n_basis="", fwhm="", alpha=""):
        # Null-relative metrics, not raw cvR2: most ROI voxels carry no value
        # response, so a mean over all of them mostly reports how badly the
        # worst voxels overfit. "Does this voxel beat its own train-mean
        # baseline" is the project's per-voxel signal test.
        margin = cvr2 - null.reindex(cvr2.index)
        return {"subject": subject, "model": model, "cond": cond,
                "n_basis": n_basis, "fwhm": fwhm, "alpha": alpha,
                "frac_beats_null_sel": float((margin.loc[sel] > 0).mean())
                    if len(sel) else float("nan"),
                "frac_beats_null_all": float((margin > 0).mean()),
                "median_margin_sel": float(margin.loc[sel].median())
                    if len(sel) else float("nan"),
                "mean_cvr2_top": float(cvr2.loc[top].mean()),
                "mean_cvr2_sel": float(cvr2.loc[sel].mean()),
                "mean_cvr2_all": float(cvr2.mean()),
                "n_top": int(len(top))}

    with ExitStack() as stack:
        f_cv = stack.enter_context(open(p_cv, "w", newline=""))
        f_vox = stack.enter_context(open(p_vox, "w", newline=""))
        w_cv = csv.DictWriter(f_cv, fieldnames=cv_cols, delimiter="\t")
        w_vox = csv.writer(f_vox, delimiter="\t")
        w_cv.writeheader()
        w_vox.writerow(["subject", "model", "cond", "n_basis", "fwhm",
                        "alpha", "voxel", "cvr2"])
        f_cv.flush(); f_vox.flush()

        def _emit(model, cond, cvr2, n_basis="", fwhm="", alpha=""):
            row = _summary_row(model, cond, cvr2, n_basis, fwhm, alpha)
            w_cv.writerow(row)
            for vox, val in cvr2.items():
                w_vox.writerow([subject, model, cond, n_basis, fwhm, alpha,
                                vox, float(val)])
            f_cv.flush(); f_vox.flush()
            tag = (f"k={n_basis} fwhm={fwhm} a={alpha}"
                   if model == "weighted" else "")
            print(f"  {model:<8} {cond:<8} {tag:<22} "
                  f"cvR2(top)={cvr2.loc[top].mean():+.4f}  "
                  f"beats-null={100 * row['frac_beats_null_sel']:.0f}%",
                  flush=True)

        # ── non-linear single pRF: joint / shift / separate ───────────────────
        for cond in SINGLE_COND_MODES:
            cvr2 = _cv_single(data, paradigm, cond, value_min, value_max, n_iter)
            _emit("single", cond, cvr2)

        # ── weighted basis: (k x fwhm) x {joint, separate} ────────────────────
        for cond in BASIS_COND_MODES:
            for n_basis in n_basis_list:
                for fwhm in fwhm_list:
                    for alpha in alpha_list:
                        cvr2 = _cv_weighted(data, paradigm, n_basis, fwhm,
                                            cond, alpha, value_min, value_max)
                        _emit("weighted", cond, cvr2, n_basis=n_basis,
                              fwhm=fwhm, alpha=alpha)

    print(f"  wrote {p_cv.name}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subject", help="Subject label without 'sub-'")
    p.add_argument("--n-basis", type=int, nargs="+", default=list(DEFAULT_N_BASIS))
    p.add_argument("--fwhm", type=float, nargs="+", default=list(DEFAULT_FWHM))
    p.add_argument("--alpha", type=float, nargs="+", default=list(DEFAULT_ALPHA),
                   help="Ridge penalties to sweep for the weighted basis.")
    p.add_argument("--roi", default="NPCr",
                   help="ROI desc for get_roi_mask (e.g. NPCr, "
                        "BensonV1ecc075-375).")
    p.add_argument("--roi-hemi", default="none",
                   help="'none' omits the hemi entity (NPCr/NPCl and the "
                        "eccentricity masks already encode hemisphere).")
    p.add_argument("--r2-thr", type=float, default=0.05)
    p.add_argument("--n-iter", type=int, default=500,
                   help="Adam iterations for the single-pRF fits (default 500)")
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    args = p.parse_args()
    run_one(args.subject, args.n_basis, args.fwhm, alpha_list=args.alpha,
            roi=args.roi,
            roi_hemi=None if args.roi_hemi.lower() == "none" else args.roi_hemi,
            r2_thr=args.r2_thr,
            n_iter=args.n_iter, smoothed=args.smoothed,
            bids_folder=args.bids_folder)


if __name__ == "__main__":
    main()
