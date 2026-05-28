"""V1 basis-count sweep: does increasing n_basis bring out the
anti-cardinal SD pattern (dips at 0°/90°/180°) in the expected
uncertainty?

For each subject × session, fit per-session vonmises basis weights at
multiple basis counts (closed-form lstsq, cheap), then run the same
EU simulation pipeline as
``compute_expected_decoded_orientation_vonmises.py`` with spherical
noise + fdr05 voxel selection. Outputs go to
``derivatives/experiments/v1_basis_sweep/sub-<S>/ses-<i>/func/`` with
``_nbasis-<N>_`` in the filename so the basis count is preserved.

Self-contained — does not touch the production fit_vonmises_model
pipeline. Designed to run as one SLURM job per subject, looping over
basis counts internally.

Usage:
    python -m abstract_values.experiments.v1_basis_sweep 04
    python -m abstract_values.experiments.v1_basis_sweep 04 --n-basis 8 16 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import AxialVonMisesPRF
from braincoder.optimize import ResidualFitter, WeightFitter
from braincoder.utils import get_rsq

from abstract_values.utils.data import Subject, BIDS_FOLDER
from abstract_values.encoding_models.compute_expected_decoded_orientation_vonmises \
    import (get_gabor_paradigm, make_basis_parameters,
            simulate_decode_session, aggregate_per_stimulus)
from abstract_values.encoding_models.compute_r2_mixture \
    import get_brain_fdr_threshold


DEFAULT_N_BASIS = (8, 16, 32)
KAPPA = 2.0
N_SIMULATIONS = 1000
N_NOISE_ITERATIONS = 1000
BATCH_STIMULI = 25
FDR_ALPHA = 0.05
SPHERICAL = True

OUT_ROOT = Path(BIDS_FOLDER) / "derivatives" / "experiments" / "v1_basis_sweep"
TRAINED_GRID_RAD = np.deg2rad(np.arange(7.5, 173, 7.5)).astype(np.float32)


def run_one(subject, n_basis_list,
             smoothed=False, bids_folder=BIDS_FOLDER, kappa=KAPPA):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    smooth_label = "_smoothed" if smoothed else ""

    print(f"sub-{subject}  sessions={sessions}  "
          f"n_basis_list={list(n_basis_list)}  kappa={kappa}  "
          f"spherical={SPHERICAL}  fdr_alpha={FDR_ALPHA}")

    # ── shared inputs (mask + betas + paradigm) ───────────────────────────────
    mask_img = sub.get_roi_mask("BensonV1", hemi="LR")
    betas_img = sub.get_single_trial_estimates(sessions, desc="gabor",
                                                smoothed=smoothed)
    masker = NiftiMasker(mask_img=mask_img,
                         target_affine=betas_img.affine,
                         target_shape=betas_img.shape[:3]).fit()
    data_all = pd.DataFrame(masker.transform(betas_img).astype(np.float32))
    paradigm_all = get_gabor_paradigm(sub, sessions)
    print(f"  {data_all.shape[1]} V1 voxels  ·  "
          f"{len(paradigm_all)} gabor trials")

    # ── voxel selection: FDR-α on the whole-brain joint vonmises mixture ──
    # Done once for the WHOLE basis-sweep — selection threshold should not
    # depend on n_basis (it's a property of the original vonmises R²
    # mixture). Falls back to top-100 by joint R² if mixture is degenerate.
    res = get_brain_fdr_threshold(
        subject, model="vonmises", bids_folder=bids_folder,
        alpha=FDR_ALPHA, smoothed=smoothed)
    if res is None:
        raise RuntimeError("Whole-brain vonmises R² mixture missing for "
                            f"sub-{subject}.")
    thr = res["threshold"]
    # We need the per-voxel joint R² inside the mask to apply the threshold.
    # The standard vonmises joint R² nifti is already on disk.
    r2_img = nib.load(str(bids_folder / "derivatives" / "encoding_models"
                           / "vonmises" / f"sub-{subject}" / "func"
                           / (f"sub-{subject}_task-abstractvalue"
                              f"_space-T1w_desc-r2{smooth_label}_pe.nii.gz")))
    r2 = pd.Series(masker.transform(r2_img).ravel().astype(np.float32))
    if res["degenerate"] or not np.isfinite(thr):
        sel = r2.sort_values(ascending=False).index[:100]
        sel_label = "fdr05 → fallback top-100"
    else:
        sel = r2[r2 > thr].index
        if len(sel) < 10:
            sel = r2.sort_values(ascending=False).index[:100]
            sel_label = f"fdr05 → fallback top-100 (only {(r2 > thr).sum()} passed)"
        else:
            sel_label = f"fdr05 (R² > {thr:.3f})"
    print(f"  {len(sel)} voxels  ({sel_label})")

    # ── loop over basis counts ────────────────────────────────────────────────
    model = AxialVonMisesPRF()
    for n_basis in n_basis_list:
        print(f"\n  ── n_basis = {n_basis} ──")
        basis_pars = make_basis_parameters(n_basis=n_basis, kappa=kappa)

        for ses_i in sessions:
            cond = sub.get_mapping(ses_i)
            ses_mask = paradigm_all["session"].values == sessions.index(ses_i)
            data_ses = data_all.iloc[ses_mask].reset_index(drop=True)
            paradigm_ses = paradigm_all.loc[ses_mask, ["x"]].reset_index(drop=True)

            data_ses_sel = data_ses[sel]
            # per-session weights (closed-form lstsq)
            weights = WeightFitter(model, basis_pars,
                                    data_ses_sel, paradigm_ses).fit()

            # quick fit-R² sanity check
            basis_pred = model.basis_predictions(paradigm_ses, basis_pars)
            pred = pd.DataFrame(basis_pred @ weights.values,
                                 index=data_ses_sel.index,
                                 columns=data_ses_sel.columns)
            r2_ses = get_rsq(data_ses_sel, pred)
            print(f"    ses-{ses_i} ({cond}): {len(paradigm_ses)} trials, "
                  f"mean fit R² (selected voxels) = {float(r2_ses.mean()):.4f}")

            # noise model
            residfit = ResidualFitter(model, data_ses_sel, paradigm_ses,
                                       parameters=basis_pars,
                                       weights=weights)
            omega, dof = residfit.fit(
                init_sigma2=1e-2, init_dof=10.0,
                learning_rate=0.05, spherical=SPHERICAL,
                max_n_iterations=N_NOISE_ITERATIONS)
            dof_str = f"{float(dof):.1f}" if dof is not None else "None (Gaussian)"
            print(f"    noise model: dof={dof_str}")

            # simulate-decode-aggregate on trained 23-orientation grid
            true_arr, decoded_arr = simulate_decode_session(
                model, basis_pars, weights, omega, dof,
                TRAINED_GRID_RAD, N_SIMULATIONS,
                batch_stimuli=BATCH_STIMULI)
            agg = aggregate_per_stimulus(true_arr, decoded_arr,
                                          TRAINED_GRID_RAD)
            # Tag rows with the basis count + session/condition so the
            # downstream viewer can pool by (n_basis, condition).
            agg["n_basis"]  = n_basis
            agg["subject"]  = subject
            agg["session"]  = ses_i
            agg["condition"] = cond

            out_dir = OUT_ROOT / f"sub-{subject}" / f"ses-{ses_i}" / "func"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_fn = (out_dir /
                       f"sub-{subject}_ses-{ses_i}_task-abstractvalue"
                       f"_mask-BensonV1_hemi-LR_nvoxels-fdr05_nsims-{N_SIMULATIONS}"
                       f"_noise-spherical_nbasis-{n_basis}{smooth_label}"
                       f"_desc-expected_decoded_orientation_pe.tsv")
            agg.to_csv(out_fn, sep="\t", index=False)
            print(f"    saved {out_fn.name}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("subject")
    p.add_argument("--n-basis", type=int, nargs="+", default=list(DEFAULT_N_BASIS))
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--kappa", type=float, default=KAPPA)
    args = p.parse_args()
    run_one(args.subject, args.n_basis, smoothed=args.smoothed,
            bids_folder=args.bids_folder, kappa=args.kappa)


if __name__ == "__main__":
    main()
