#!/usr/bin/env python3
"""
V1 Von Mises model comparison: sweep n_basis (k) x kappa (dispersion).

Question
--------
How does the cross-validated R2 of the V1 Von Mises orientation model
depend on the number of basis functions (k) and their concentration
(kappa), and how does the preferred-orientation distribution
(the top panel of ``preferred_tuning.pdf``) change with those choices?
This lets k/kappa be picked by cvR2 rather than by hand.

What it does, per subject
-------------------------
Restricted to the BensonV1 (LR) ROI. For every (n_basis, kappa) in the
grid:
  * leave-one-run-out CV (per-session basis weights, predicting each
    held-out run with its own session's weights -- the session-shift
    scheme used by the production V1 pipeline) -> per-voxel mean cvR2.
  * full-data per-session weights -> per-voxel preferred orientation
    (argmax of the fitted tuning curve over a fine grid) -> binned
    histogram, restricted to a *fixed* voxel set so the distribution is
    comparable across k/kappa.

Voxel selection for the summaries is fixed across the whole sweep (it is
a property of the joint Von Mises R2, not of k): voxels with
joint-vonmises R2 > ``--r2-thr`` (default 0.05). cvR2 is also summarised
over *all* V1 voxels for reference.

With ``--decode`` an out-of-sample orientation decoding pass is added per
(n_basis, kappa): leave-one-run-out, FDR voxel selection on nested-CV R2
(whole-brain Von Mises mixture, ``--fdr-alpha``), spherical noise model,
posterior via ``get_stimulus_pdf``; reports the mean/median absolute
circular decoding error and circular SD. This pass is CPU-only but not
OLS (it fits a noise model + posterior per fold), so it is slower than
the encoding sweep. Decoding columns are added to the cvr2summary TSV.

Outputs (one job per subject; loops the grid internally)
--------------------------------------------------------
  derivatives/experiments/v1_k_kappa_sweep/sub-<S>/func/
    sub-<S>_task-abstractvalue_mask-{roi}_desc-cvr2summary{_smoothed}.tsv
    sub-<S>_task-abstractvalue_mask-{roi}_desc-preferredhist{_smoothed}.tsv

Usage
-----
  python -m abstract_values.encoding_models.sweep_v1_k_kappa 04
  python -m abstract_values.encoding_models.sweep_v1_k_kappa 04 \
      --n-basis 4 6 8 12 16 20 24 --kappa 1 2 4 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import AxialVonMisesPRF
from braincoder.optimize import WeightFitter, ResidualFitter
from braincoder.utils import get_rsq

from abstract_values.utils.data import Subject, BIDS_FOLDER
from abstract_values.encoding_models.fit_vonmises_cv import (
    get_gabor_paradigm_with_runs, make_basis_parameters)
from abstract_values.utils.circstats import (circular_distance, circular_sd,
                                             circular_corr)

DEFAULT_N_BASIS = (4, 6, 8, 12, 16, 20, 24)
DEFAULT_KAPPA = (1.0, 2.0, 4.0, 8.0)

OUT_ROOT = Path(BIDS_FOLDER) / "derivatives" / "experiments" / "v1_k_kappa_sweep"

# Fine orientation grid (radians, period pi) for reading off preferred tuning.
PREF_GRID_RAD = np.linspace(0, np.pi, 200, endpoint=False, dtype=np.float32)
HIST_BINS = np.linspace(0, 180, 31)            # 30 bins over 0-180 deg
HIST_CENTERS = 0.5 * (HIST_BINS[:-1] + HIST_BINS[1:])


def _fit_weights_per_session(model, basis_pars, data, paradigm):
    """Closed-form per-session weights. Returns {session: weights_df}."""
    sessions = paradigm.index.get_level_values("session")
    out = {}
    for ses in sorted(set(sessions)):
        m = (sessions == ses)
        out[ses] = WeightFitter(
            model, basis_pars,
            data.iloc[m].reset_index(drop=True),
            paradigm.loc[m].reset_index(drop=True)[["x"]],
        ).fit()
    return out


def _loo_cv_joint(model, basis_pars, data, paradigm):
    """Leave-one-run-out CV with a SINGLE joint weight set fit on all
    training runs (both sessions pooled). Correct for orientation, whose
    tuning is condition-invariant; uses 2x the training data and half the
    parameters of the session-shift variant. Returns per-voxel mean cvR2."""
    sess = paradigm.index.get_level_values("session")
    runs = paradigm.index.get_level_values("run")
    per_fold = []
    for test_ses, test_run in sorted(set(zip(sess, runs))):
        test_mask = (sess == test_ses) & (runs == test_run)
        w = WeightFitter(model, basis_pars,
                         data.loc[~test_mask].reset_index(drop=True),
                         paradigm.loc[~test_mask].reset_index(drop=True)[["x"]]).fit()
        test_data = data.loc[test_mask].reset_index(drop=True)
        test_par = paradigm.loc[test_mask].reset_index(drop=True)[["x"]]
        bp = model.basis_predictions(test_par, basis_pars)
        pred = pd.DataFrame(bp @ w.values, index=test_data.index,
                            columns=test_data.columns)
        per_fold.append(get_rsq(test_data, pred))
    return pd.concat(per_fold, axis=1).mean(axis=1)


def _loo_cv_session_shift(model, basis_pars, data, paradigm):
    """Leave-one-run-out CV with per-session weights. Returns per-voxel
    mean cvR2 (Series indexed like data columns)."""
    sess = paradigm.index.get_level_values("session")
    runs = paradigm.index.get_level_values("run")
    folds = sorted(set(zip(sess, runs)))
    per_fold = []
    for test_ses, test_run in folds:
        test_mask = (sess == test_ses) & (runs == test_run)
        train_mask = ~test_mask

        train_data = data.loc[train_mask]
        train_par = paradigm.loc[train_mask]
        test_data = data.loc[test_mask].reset_index(drop=True)
        test_par = paradigm.loc[test_mask].reset_index(drop=True)[["x"]]

        wts_by_ses = _fit_weights_per_session(model, basis_pars,
                                              train_data, train_par)
        wts = wts_by_ses.get(test_ses, next(iter(wts_by_ses.values())))
        basis_pred = model.basis_predictions(test_par, basis_pars)
        test_pred = pd.DataFrame(basis_pred @ wts.values,
                                 index=test_data.index,
                                 columns=test_data.columns)
        per_fold.append(get_rsq(test_data, test_pred))
    return pd.concat(per_fold, axis=1).mean(axis=1)


def _preferred_orientation_deg(model, basis_pars, weights):
    """Argmax-of-tuning preferred orientation (deg, 0-180) per voxel."""
    fine = pd.DataFrame({"x": PREF_GRID_RAD})
    basis_pred = model.basis_predictions(fine, basis_pars)   # (200, n_basis)
    curves = np.asarray(basis_pred) @ weights.values         # (200, n_voxels)
    pref_rad = PREF_GRID_RAD[np.argmax(curves, axis=0)]
    return np.rad2deg(pref_rad)


def _null_cvr2(data, paradigm):
    """Per-voxel leave-one-run-out cvR2 of the TRUE null model: predict the
    *training-set* mean on each held-out run. This is the honest baseline --
    it is slightly negative with finite data (train mean != test mean) and
    only ->0 in the limit, so 'has signal' means model cvR2 > this, NOT >0."""
    sess = paradigm.index.get_level_values("session")
    runs = paradigm.index.get_level_values("run")
    per_fold = []
    for test_ses, test_run in sorted(set(zip(sess, runs))):
        test_mask = (sess == test_ses) & (runs == test_run)
        train_mean = data.loc[~test_mask].mean(axis=0)
        test_data = data.loc[test_mask]
        pred = pd.DataFrame(np.broadcast_to(train_mean.values,
                                            test_data.shape),
                            index=test_data.index, columns=test_data.columns)
        per_fold.append(get_rsq(test_data, pred))
    return pd.concat(per_fold, axis=1).mean(axis=1)


def _nested_cv_r2(model, basis_pars, train_data, train_par):
    """Inner leave-one-run-out CV R2 within the training set (unbiased,
    no circularity) -- used for voxel selection."""
    sess = train_par.index.get_level_values("session")
    runs = train_par.index.get_level_values("run")
    inner = []
    for ses, run in sorted(set(zip(sess, runs))):
        itest = (sess == ses) & (runs == run)
        w = WeightFitter(model, basis_pars,
                         train_data.loc[~itest], train_par.loc[~itest]).fit()
        bp = model.basis_predictions(train_par.loc[itest], basis_pars)
        pred = pd.DataFrame(bp @ w.values,
                            index=train_data.loc[itest].index,
                            columns=train_data.columns)
        inner.append(get_rsq(train_data.loc[itest], pred))
    return pd.concat(inner, axis=1).mean(axis=1)


def _decode_oos(data, paradigm, basis_pars, fdr_thr, fallback_n=100,
                spherical=True, noise_iter=1000):
    """Leave-one-run-out out-of-sample decoding of orientation, FDR voxel
    selection on nested-CV R2. Returns (true_rad, decoded_rad, mean_n_sel)
    pooled over all held-out trials (signed error and circular correlation
    are both derived downstream from true vs decoded)."""
    model = AxialVonMisesPRF(allow_neg_amplitudes=True)
    stim_range = np.sort(paradigm["x"].unique()).astype(np.float32)
    sess = paradigm.index.get_level_values("session")
    runs = paradigm.index.get_level_values("run")
    folds = sorted(set(zip(sess, runs)))

    trues, decodeds, n_sels = [], [], []
    for test_ses, test_run in folds:
        test_mask = (sess == test_ses) & (runs == test_run)
        train_data, test_data = data.loc[~test_mask], data.loc[test_mask]
        train_par, test_par = paradigm.loc[~test_mask], paradigm.loc[test_mask]

        weights = WeightFitter(model, basis_pars, train_data, train_par).fit()
        cv_r2 = _nested_cv_r2(model, basis_pars, train_data, train_par)
        if fdr_thr is not None and np.isfinite(fdr_thr):
            sel = cv_r2[cv_r2 > fdr_thr].index
        else:
            sel = pd.Index([])
        if len(sel) < 10:                       # degenerate -> top-N by cv-R2
            sel = cv_r2.sort_values(ascending=False).index[:fallback_n]
        n_sels.append(len(sel))

        resid = ResidualFitter(model, train_data[sel], train_par,
                               parameters=basis_pars, weights=weights[sel])
        omega, dof = resid.fit(init_sigma2=0.1, init_dof=10.0, method="t",
                               learning_rate=0.05, spherical=spherical,
                               max_n_iterations=noise_iter)
        pdf = model.get_stimulus_pdf(test_data[sel], stim_range,
                                     parameters=basis_pars, weights=weights[sel],
                                     omega=omega, dof=dof, normalize=True)
        # posterior circular mean (period pi, doubled-angle) -> point estimate
        w = pdf.values
        z = (w * np.exp(1j * 2 * stim_range[None, :])).sum(axis=1)
        decodeds.append(0.5 * np.angle(z))
        trues.append(test_par["x"].values)

    return (np.concatenate(trues), np.concatenate(decodeds),
            float(np.mean(n_sels)))


def compare_cv(subject, n_basis, kappa, r2_thr=0.05, top_n=100,
               smoothed=False, bids_folder=BIDS_FOLDER, roi='BensonV1', roi_hemi='LR'):
    """Print-only diagnostic: joint vs session-shift cvR2 for one (k, kappa),
    summarised over all V1 / R2>thr / top-N (by independent joint R2), with
    the true-null reference. Writes nothing (safe to run alongside a sweep)."""
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    smooth_label = "_smoothed" if smoothed else ""
    mask_img = sub.get_roi_mask(roi, hemi=roi_hemi)
    betas_img = sub.get_single_trial_estimates(sessions, desc="gabor",
                                               smoothed=smoothed)
    masker = NiftiMasker(mask_img=mask_img, target_affine=betas_img.affine,
                         target_shape=betas_img.shape[:3]).fit()
    paradigm = get_gabor_paradigm_with_runs(sub, sessions)
    data = pd.DataFrame(masker.transform(betas_img).astype(np.float32),
                        index=paradigm.index)
    r2_path = (bids_folder / "derivatives" / "encoding_models" / "vonmises"
               / f"sub-{subject}" / "func"
               / f"sub-{subject}_task-abstractvalue_space-T1w"
                 f"_desc-r2{smooth_label}_pe.nii.gz")
    joint_r2 = pd.Series(masker.transform(nib.load(str(r2_path))).ravel()
                         .astype(np.float32), index=data.columns)
    sel = joint_r2[joint_r2 > r2_thr].index
    top = joint_r2.sort_values(ascending=False).index[:top_n]

    model = AxialVonMisesPRF()
    basis_pars = make_basis_parameters(n_basis, kappa)
    null = _null_cvr2(data, paradigm)
    joint = _loo_cv_joint(model, basis_pars, data, paradigm)
    sshift = _loo_cv_session_shift(model, basis_pars, data, paradigm)

    print(f"\nsub-{subject}  k={n_basis} kappa={kappa}  "
          f"({data.shape[1]} V1 voxels, {len(sel)} with R2>{r2_thr}, top-{top_n})")
    print(f"  {'set':<16}{'JOINT':>10}{'SESS-SHIFT':>12}{'NULL':>10}")
    for label, idx in [("all V1", data.columns), (f"R2>{r2_thr}", sel),
                       (f"top-{top_n}", top)]:
        print(f"  {label:<16}{joint.loc[idx].mean():>10.4f}"
              f"{sshift.loc[idx].mean():>12.4f}{null.loc[idx].mean():>10.4f}")
    print(f"  frac voxels beating null (JOINT, top-{top_n}): "
          f"{float((joint.loc[top] > null.loc[top]).mean()):.2f}")


def kappa_for_fwhm_ratio(n_basis, ratio):
    """Concentration giving a tuning width of ``ratio`` x the basis spacing.

    Sweeping n_basis and kappa independently is the wrong parameterisation:
    what controls whether a basis set can represent an arbitrary tuning curve
    is its FWHM *relative to the spacing between basis functions*, and the
    same kappa means very different things at n=4 and n=24. On the existing
    grid the corners are degenerate — (kappa=1, n=24) is 24 near-identical
    functions whose weights are unidentified, and (kappa=8, n=4) leaves gaps
    the basis cannot represent — so most of a 29-subject sweep is spent on
    combinations that fail for numerical rather than biological reasons.

    For an axial von Mises over [0, pi), f(t) ~ exp(kappa*cos(2*(t-mu))),
    half-max is at cos(FWHM) = 1 + ln(0.5)/kappa, so

        kappa = ln(2) / (1 - cos(FWHM))

    with FWHM in radians on the doubled angle. Well-tiled is ratio ~ 1-2.
    """
    spacing = np.pi / n_basis                  # radians, pi-periodic axis
    fwhm = ratio * spacing
    fwhm = float(np.clip(fwhm, 1e-3, np.pi - 1e-3))
    return float(np.log(2.0) / (1.0 - np.cos(fwhm)))


def run_one(subject, n_basis_list, kappa_list, r2_thr=0.05,
            fwhm_ratios=None,
            decode=False, fdr_alpha=0.05, spherical=True, noise_iter=1000,
            fallback_n=100, cv_mode="joint", smoothed=False, bids_folder=BIDS_FOLDER, roi='BensonV1', roi_hemi='LR'):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    smooth_label = "_smoothed" if smoothed else ""
    # Orientation tuning is condition-invariant, so the joint fit (pool both
    # sessions) is the correct, stronger model; session-shift halves the
    # training data and depresses cvR2. Joint is the default.
    cv_fn = _loo_cv_joint if cv_mode == "joint" else _loo_cv_session_shift

    print(f"sub-{subject}  sessions={sessions}  "
          f"n_basis={list(n_basis_list)}  kappa={list(kappa_list)}  "
          f"r2_thr={r2_thr}  cv={cv_mode}  decode={decode}  smoothed={smoothed}")

    # ── shared inputs: V1 mask, betas, paradigm ───────────────────────────────
    mask_img = sub.get_roi_mask(roi, hemi=roi_hemi)
    betas_img = sub.get_single_trial_estimates(sessions, desc="gabor",
                                               smoothed=smoothed)
    masker = NiftiMasker(mask_img=mask_img,
                         target_affine=betas_img.affine,
                         target_shape=betas_img.shape[:3]).fit()
    paradigm = get_gabor_paradigm_with_runs(sub, sessions)
    data = pd.DataFrame(masker.transform(betas_img).astype(np.float32),
                        index=paradigm.index)
    n_v1 = data.shape[1]
    print(f"  {n_v1} V1 voxels  ·  {len(paradigm)} gabor trials")
    assert betas_img.shape[3] == len(paradigm), (
        f"Beta count mismatch: {betas_img.shape[3]} vs {len(paradigm)}")

    # ── fixed voxel selection from the joint vonmises R2 (k-independent) ──────
    r2_path = (bids_folder / "derivatives" / "encoding_models" / "vonmises"
               / f"sub-{subject}" / "func"
               / f"sub-{subject}_task-abstractvalue_space-T1w"
                 f"_desc-r2{smooth_label}_pe.nii.gz")
    if r2_path.exists():
        joint_r2 = pd.Series(masker.transform(nib.load(str(r2_path)))
                             .ravel().astype(np.float32), index=data.columns)
        sel = joint_r2[joint_r2 > r2_thr].index
        sel_label = f"joint vonmises R2 > {r2_thr}"
        if len(sel) < 10:                       # degenerate -> top 100 by R2
            sel = joint_r2.sort_values(ascending=False).index[:100]
            sel_label = "fallback top-100 by joint R2"
    else:
        joint_r2 = pd.Series(np.nan, index=data.columns)
        sel = data.columns                      # no joint fit -> use all V1
        sel_label = "all V1 (joint R2 missing)"
    print(f"  selected {len(sel)} voxels  ({sel_label})")

    # ── FDR threshold for the decoding voxel selection (k-independent) ────────
    fdr_thr = None
    if decode:
        from abstract_values.encoding_models.compute_r2_mixture \
            import get_brain_fdr_threshold
        res = get_brain_fdr_threshold(subject, model="vonmises",
                                      bids_folder=bids_folder, alpha=fdr_alpha,
                                      smoothed=smoothed)
        if res is not None and not res["degenerate"] and np.isfinite(res["threshold"]):
            fdr_thr = float(res["threshold"])
            print(f"  decode: FDR<={fdr_alpha} -> nested-CV R2 > {fdr_thr:.3f}")
        else:
            print(f"  decode: vonmises mixture degenerate/missing -> "
                  f"top-{fallback_n} by nested-CV R2 per fold")

    # ── output files: written + flushed per (k, kappa) so partial results ─────
    # survive a wall-time kill and the TSV row-count doubles as a progress
    # meter (independent of any stdout buffering by the job wrapper).
    out_dir = (bids_folder / "derivatives" / "experiments"
               / "v1_k_kappa_sweep" / f"sub-{subject}" / "func")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"sub-{subject}_task-abstractvalue_mask-{roi}"
    p_cv = out_dir / f"{base}_desc-cvr2summary{smooth_label}.tsv"
    p_hist = out_dir / f"{base}_desc-preferredhist{smooth_label}.tsv"
    p_vox = out_dir / f"{base}_desc-cvr2voxels{smooth_label}.tsv"
    p_null = out_dir / f"{base}_desc-nullcvr2{smooth_label}.tsv"

    # True null model (predict training mean) -- per-voxel reference; computed
    # once (k/kappa-independent). 'Signal' = model cvR2 > this, not > 0.
    null_cvr2 = _null_cvr2(data, paradigm)
    pd.DataFrame({"subject": subject, "voxel": null_cvr2.index,
                  "null_cvr2": null_cvr2.values,
                  "joint_r2": joint_r2.reindex(null_cvr2.index).values}
                 ).to_csv(p_null, sep="\t", index=False)
    print(f"  null model: mean cvR2(all V1)={float(null_cvr2.mean()):+.4f} "
          f"(predict train mean; the true 0-signal baseline)")

    cv_cols = ["subject", "n_basis", "kappa", "n_v1", "n_sel", "mean_cvr2_sel",
               "median_cvr2_sel", "mean_cvr2_all", "frac_pos_sel"]
    if decode:
        cv_cols += ["decode_mean_abs_err_deg", "decode_median_abs_err_deg",
                    "decode_circ_sd_deg", "decode_circ_corr", "decode_mean_n_sel"]
    p_dec = out_dir / f"{base}_desc-decodetrials{smooth_label}.tsv"
    hist_cols = ["subject", "n_basis", "kappa", "session", "condition",
                 "orientation_deg", "count"]
    # per-voxel cvR2 over ALL V1 voxels (every config) -- lets the local
    # plotter define "signal voxels" (e.g. cvR2>0 in >=1 config) and compute
    # signal-restricted means, since most V1 voxels are untuned noise whose
    # negative cvR2 drags the all-voxel mean below zero.
    vox_cols = ["subject", "n_basis", "kappa", "voxel", "cvr2"]

    import csv
    from contextlib import ExitStack
    n_total = len(n_basis_list) * (len(fwhm_ratios) if fwhm_ratios
                                   else len(kappa_list))
    model = AxialVonMisesPRF()

    with ExitStack() as stack:
        f_cv = stack.enter_context(open(p_cv, "w", newline=""))
        f_hist = stack.enter_context(open(p_hist, "w", newline=""))
        f_vox = stack.enter_context(open(p_vox, "w", newline=""))
        w_cv = csv.DictWriter(f_cv, fieldnames=cv_cols, delimiter="\t",
                              extrasaction="ignore")
        w_hist = csv.DictWriter(f_hist, fieldnames=hist_cols, delimiter="\t")
        w_vox = csv.writer(f_vox, delimiter="\t")
        w_cv.writeheader(); w_hist.writeheader()
        w_vox.writerow(vox_cols)
        f_cv.flush(); f_hist.flush(); f_vox.flush()

        w_dec = None
        if decode:                              # per-trial decoded vs true
            f_dec = stack.enter_context(open(p_dec, "w", newline=""))
            w_dec = csv.writer(f_dec, delimiter="\t")
            w_dec.writerow(["subject", "n_basis", "kappa",
                            "true_deg", "decoded_deg"])
            f_dec.flush()

        done = 0
        for n_basis in n_basis_list:
            # When sweeping ratios, kappa is derived per n so that every cell
            # has the same FWHM-to-spacing relationship (see
            # kappa_for_fwhm_ratio); otherwise sweep kappa directly.
            kappas = ([kappa_for_fwhm_ratio(n_basis, r) for r in fwhm_ratios]
                      if fwhm_ratios else list(kappa_list))
            for kappa in kappas:
                basis_pars = make_basis_parameters(n_basis, kappa)

                # cvR2 (per-voxel mean over folds)
                cvr2 = cv_fn(model, basis_pars, data, paradigm)
                cvr2_sel = cvr2.loc[sel]
                row = {
                    "subject": subject, "n_basis": n_basis, "kappa": kappa,
                    "n_v1": n_v1, "n_sel": int(len(sel)),
                    "mean_cvr2_sel": float(cvr2_sel.mean()),
                    "median_cvr2_sel": float(cvr2_sel.median()),
                    "mean_cvr2_all": float(cvr2.mean()),
                    "frac_pos_sel": float((cvr2_sel > 0).mean()),
                }

                # out-of-sample decoding (FDR voxel selection, spherical noise)
                if decode:
                    true, decoded, mean_n_sel = _decode_oos(
                        data, paradigm, basis_pars, fdr_thr, fallback_n=fallback_n,
                        spherical=spherical, noise_iter=noise_iter)
                    err = circular_distance(decoded, true)
                    err_deg = np.rad2deg(np.abs(err))
                    row.update({
                        "decode_mean_abs_err_deg": float(np.mean(err_deg)),
                        "decode_median_abs_err_deg": float(np.median(err_deg)),
                        "decode_circ_sd_deg": float(np.rad2deg(circular_sd(err))),
                        "decode_circ_corr": float(circular_corr(true, decoded)),
                        "decode_mean_n_sel": float(mean_n_sel),
                    })
                    for t, d in zip(np.rad2deg(true), np.rad2deg(decoded) % 180):
                        w_dec.writerow([subject, n_basis, kappa,
                                        f"{t:.2f}", f"{d:.2f}"])
                w_cv.writerow(row)

                # per-voxel cvR2 (all V1 voxels) for post-hoc signal selection
                for vox, val in cvr2.items():
                    w_vox.writerow([subject, n_basis, kappa, vox, float(val)])

                # preferred-orientation histogram, full-data per-session weights,
                # over the fixed selected voxel set
                wts_by_ses = _fit_weights_per_session(
                    model, basis_pars, data.loc[:, sel], paradigm)
                for ses in sessions:
                    if ses not in wts_by_ses:
                        continue
                    pref_deg = _preferred_orientation_deg(
                        model, basis_pars, wts_by_ses[ses])
                    counts, _ = np.histogram(pref_deg, bins=HIST_BINS)
                    for c, cnt in zip(HIST_CENTERS, counts):
                        w_hist.writerow({
                            "subject": subject, "n_basis": n_basis, "kappa": kappa,
                            "session": ses, "condition": sub.get_mapping(ses),
                            "orientation_deg": float(c), "count": int(cnt),
                        })
                f_cv.flush(); f_hist.flush(); f_vox.flush()   # crash-safe + monitorable
                if decode:
                    f_dec.flush()

                done += 1
                msg = (f"  [{done}/{n_total}] n_basis={n_basis:2d} "
                       f"kappa={kappa:<4g} mean cvR2(sel)={row['mean_cvr2_sel']:+.4f}")
                if decode:
                    msg += (f"  decode |err|={row['decode_mean_abs_err_deg']:.1f}deg"
                            f" (n_sel~{row['decode_mean_n_sel']:.0f})")
                print(msg, flush=True)

    print(f"  wrote {p_cv.name}\n  wrote {p_hist.name}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subject", help="Subject label without 'sub-'")
    p.add_argument("--n-basis", type=int, nargs="+", default=list(DEFAULT_N_BASIS))
    p.add_argument("--kappa", type=float, nargs="+", default=list(DEFAULT_KAPPA))
    p.add_argument("--r2-thr", type=float, default=0.05,
                   help="Joint-vonmises R2 voxel-selection threshold for the "
                        "encoding/preferred-tuning summaries (default 0.05)")
    p.add_argument("--decode", action="store_true",
                   help="Also run FDR-selected out-of-sample orientation decoding")
    p.add_argument("--fdr-alpha", type=float, default=0.05,
                   help="FDR alpha for the decoding voxel selection (default 0.05)")
    p.add_argument("--full-noise", action="store_true",
                   help="Full covariance noise model for decoding "
                        "(default: spherical, which is preferred here)")
    p.add_argument("--noise-iter", type=int, default=1000,
                   help="Noise-model fit iterations for decoding (default 1000)")
    p.add_argument("--cv-mode", choices=["joint", "sessionshift"], default="joint",
                   help="LOO-CV weight fit: joint (pool sessions; correct for "
                        "orientation, default) or sessionshift (per-session).")
    p.add_argument("--compare-cv", action="store_true",
                   help="Print-only diagnostic: joint vs session-shift cvR2 for "
                        "the first (n_basis, kappa); writes nothing.")
    p.add_argument("--fwhm-ratio", type=float, nargs="+", default=None,
                   help="Sweep tuning width as a MULTIPLE OF THE BASIS "
                        "SPACING instead of raw kappa (e.g. 0.75 1.0 1.5 2.0 "
                        "3.0). kappa is then derived per n_basis, so every "
                        "cell is equally well tiled. Sweeping raw kappa "
                        "confounds width with n: the same kappa is a gap at "
                        "n=4 and near-collinear duplication at n=24.")
    p.add_argument("--roi", default="BensonV1",
                   help="ROI desc for the mask. Use "
                        "BensonV1ecc075-375 for V1 restricted to the "
                        "eccentricity band the gabor annulus actually drives "
                        "(0.75-3.75 deg) — only ~20%% of V1, the rest is "
                        "unstimulated cortex that dilutes every estimate.")
    p.add_argument("--roi-hemi", default="LR",
                   help="hemi entity for the mask ('LR', 'L', 'R', or 'none')")
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    args = p.parse_args()
    # 'none' means omit the hemi entity entirely (NPCr-style masks)
    roi_hemi = None if args.roi_hemi.lower() == 'none' else args.roi_hemi
    if args.compare_cv:
        compare_cv(args.subject, args.n_basis[0], args.kappa[0],
                   r2_thr=args.r2_thr, smoothed=args.smoothed,
                   bids_folder=args.bids_folder,
                   roi=args.roi, roi_hemi=roi_hemi)
        return
    run_one(args.subject, args.n_basis, args.kappa, r2_thr=args.r2_thr,
            decode=args.decode, fdr_alpha=args.fdr_alpha,
            spherical=not args.full_noise, noise_iter=args.noise_iter,
            cv_mode=args.cv_mode, smoothed=args.smoothed,
            bids_folder=args.bids_folder,
            roi=args.roi, roi_hemi=roi_hemi,
            fwhm_ratios=args.fwhm_ratio)


if __name__ == "__main__":
    main()
