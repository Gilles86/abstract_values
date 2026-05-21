"""Whole-brain 2-component R² mixture per (subject, encoding model).

The mixture (logit-Gaussian noise + signal) is fit to the R² values of all
brain voxels for one encoding-model variant (aprf / vonmises / aprf-gauss
/ aprf-weighted). The cached fit is then used by the decoders' ``--fdr-alpha``
mode to derive an FDR-controlled R² threshold — applied within any ROI
without the small-n instability you get when fitting the mixture inside
the ROI itself.

Adapted from retsupp's ``compute_r2_mixture`` (same shape of cache file,
same use of ``braincoder.utils.stats``).

Outputs (per subject × model):
    derivatives/encoding_models/<model>/sub-XX/sub-XX_desc-p_signal.json
    derivatives/encoding_models/<model>/sub-XX/sub-XX_desc-p_signal.nii.gz
    derivatives/qa/r2_mixture/<model>/sub-XX_r2_mixture.pdf   (diagnostic)

Usage:
    python -m abstract_values.encoding_models.compute_r2_mixture --subject 08
    python -m abstract_values.encoding_models.compute_r2_mixture --subject 08 --model vonmises
    python -m abstract_values.encoding_models.compute_r2_mixture --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import image as nl_image
from nilearn.maskers import NiftiMasker

from abstract_values.utils.data import BIDS_FOLDER

from braincoder.utils.stats import (
    fit_r2_mixture,
    r2_fdr_threshold as r2_fdr_threshold_from_fit,
    r2_posterior_signal,
    plot_r2_mixture,
)

# Models for which the whole-brain mixture is meaningful (decoder consumers).
# All write desc-r2_pe.nii.gz at the project-wide joint-fit path.
DEFAULT_MODELS = ["aprf", "vonmises", "aprf-weighted", "aprf-gauss"]


def _r2_path(bids_folder: Path, subject: str, model: str,
             smoothed: bool = False) -> Path:
    smtag = "_smoothed" if smoothed else ""
    return (Path(bids_folder) / "derivatives" / "encoding_models" / model
            / f"sub-{subject}" / "func"
            / f"sub-{subject}_task-abstractvalue_space-T1w_desc-r2{smtag}_pe.nii.gz")


def _brain_mask_path(bids_folder: Path, subject: str,
                     fmriprep_deriv: str = "fmriprep") -> Path | None:
    """Pick the first fmriprep BOLD brain-mask we can find for this subject."""
    fp = Path(bids_folder) / "derivatives" / fmriprep_deriv / f"sub-{subject}"
    for cand in sorted(fp.glob("ses-*/func/*_space-T1w_desc-brain_mask.nii.gz")):
        return cand
    return None


def fit_subject_model(subject: str, model: str, bids_folder: Path, *,
                      smoothed: bool = False,
                      fmriprep_deriv: str = "fmriprep",
                      diagnostics_dir: Path | None = None,
                      ) -> dict | None:
    """Fit the whole-brain R² mixture for one (subject, model).

    Writes the sidecar JSON + p_signal NIfTI under
    ``encoding_models/<model>/sub-XX/`` and (optionally) a diagnostic PDF
    under ``diagnostics_dir``. Returns the mixture dict (or None on
    failure).
    """
    r2_path = _r2_path(bids_folder, subject, model, smoothed=smoothed)
    if not r2_path.exists():
        print(f"sub-{subject} {model}: R² NIfTI missing ({r2_path.name}) — skipping")
        return None

    mask_path = _brain_mask_path(bids_folder, subject, fmriprep_deriv)
    if mask_path is None:
        print(f"sub-{subject}: no fmriprep T1w-space brain mask found — skipping")
        return None

    masker = NiftiMasker(mask_img=str(mask_path)).fit()
    r2_all = masker.transform(str(r2_path)).flatten().astype(np.float32)

    # Mixture is fit on logit(R²); guard the open interval (0, 0.99).
    finite = np.isfinite(r2_all) & (r2_all > 0) & (r2_all < 0.99)
    if finite.sum() < 200:
        print(f"sub-{subject} {model}: only {finite.sum()} usable brain voxels — "
              "mixture not fit")
        return None

    print(f"sub-{subject} {model}: fitting mixture on {finite.sum()} brain voxels "
          f"(of {r2_all.size} total in mask)")
    fit = fit_r2_mixture(r2_all[finite])
    p_signal = np.full(r2_all.size, np.nan, dtype=np.float32)
    p_signal[finite] = r2_posterior_signal(r2_all[finite], fit)

    # Write outputs alongside the model's func/ dir, one level up so the
    # sidecar applies to the whole subject (not a per-run file).
    out_dir = (Path(bids_folder) / "derivatives" / "encoding_models"
               / model / f"sub-{subject}")
    out_dir.mkdir(parents=True, exist_ok=True)
    smtag = "_smoothed" if smoothed else ""
    json_fn = out_dir / f"sub-{subject}_desc-p_signal{smtag}.json"
    nii_fn = out_dir / f"sub-{subject}_desc-p_signal{smtag}.nii.gz"

    summary = {"model": model, "smoothed": smoothed,
               "n_voxels_total": int(r2_all.size),
               "n_voxels_used": int(finite.sum()),
               "BRAIN": fit}
    with open(json_fn, "w") as fh:
        json.dump(summary, fh, indent=2)

    p_signal_img = masker.inverse_transform(p_signal)
    p_signal_img.set_data_dtype(np.float32)
    p_signal_img.header.set_slope_inter(slope=1, inter=0)
    nib.save(p_signal_img, str(nii_fn))

    print(f"  → wrote {json_fn.name}")
    print(f"  → wrote {nii_fn.name}")
    print(f"  noise μ_R²={fit['noise_mean_r2']:.3f} sd_noise={fit['noise_sigma']:.2f}  "
          f"signal μ_R²={fit['signal_mean_r2']:.3f}  w_signal={fit['signal_weight']:.2f}")

    # Diagnostic PDF
    if diagnostics_dir is not None:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        pdf_fn = diagnostics_dir / f"sub-{subject}_r2_mixture{smtag}.pdf"
        fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)
        plot_r2_mixture(fit, r2=r2_all[finite], alpha=0.05, ax=ax)
        ax.set_title(f"sub-{subject}  ·  {model}{smtag}  ·  whole-brain R² mixture",
                     fontsize=9)
        fig.savefig(str(pdf_fn), dpi=200)
        plt.close(fig)
        print(f"  → wrote diagnostic {pdf_fn}")

    return fit


def load_brain_mixture(subject: str, model: str, bids_folder: Path,
                       smoothed: bool = False) -> dict | None:
    """Return the cached BRAIN mixture for (subject, model) or None."""
    smtag = "_smoothed" if smoothed else ""
    json_fn = (Path(bids_folder) / "derivatives" / "encoding_models" / model
               / f"sub-{subject}" / f"sub-{subject}_desc-p_signal{smtag}.json")
    if not json_fn.exists():
        return None
    with open(json_fn) as fh:
        return json.load(fh).get("BRAIN")


def get_brain_fdr_threshold(subject: str, model: str, bids_folder: Path,
                             alpha: float = 0.05, smoothed: bool = False,
                             auto_fit: bool = True) -> float | None:
    """Return the FDR-controlled R² threshold for (subject, model) using the
    whole-brain mixture cache. Fits the mixture on cache miss when
    ``auto_fit=True`` (default).
    """
    info = load_brain_mixture(subject, model, bids_folder, smoothed=smoothed)
    if info is None and auto_fit:
        diag = Path(bids_folder) / "derivatives" / "qa" / "r2_mixture" / model
        info = fit_subject_model(subject, model, bids_folder,
                                  smoothed=smoothed, diagnostics_dir=diag)
    if info is None:
        return None
    return r2_fdr_threshold_from_fit(info, alpha=alpha)


def _discover_subjects(bids_folder: Path, model: str,
                        smoothed: bool = False) -> list[str]:
    """All subjects with a T1w-space R² NIfTI for this model."""
    base = Path(bids_folder) / "derivatives" / "encoding_models" / model
    out = []
    for p in sorted(base.glob("sub-*")):
        if _r2_path(bids_folder, p.name.removeprefix("sub-"), model,
                     smoothed=smoothed).exists():
            out.append(p.name.removeprefix("sub-"))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", help="Subject label (e.g. 08). "
                                       "Omit + pass --all to fit every subject.")
    p.add_argument("--all", action="store_true",
                   help="Iterate over all subjects with R² on disk")
    p.add_argument("--model", default="aprf",
                   choices=DEFAULT_MODELS,
                   help="Encoding model whose R² to mix (default: aprf)")
    p.add_argument("--smoothed", action="store_true",
                   help="Use the BOLD-smoothed variant's R² NIfTI")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    args = p.parse_args()

    bids = Path(args.bids_folder)
    diag = bids / "derivatives" / "qa" / "r2_mixture" / args.model

    if args.all:
        subjects = _discover_subjects(bids, args.model, smoothed=args.smoothed)
    elif args.subject:
        subjects = [args.subject]
    else:
        raise SystemExit("Pass --subject or --all")

    if not subjects:
        raise SystemExit(f"No subjects with desc-r2 NIfTIs for model {args.model}")

    for sub in subjects:
        fit_subject_model(sub, args.model, bids, smoothed=args.smoothed,
                           diagnostics_dir=diag)


if __name__ == "__main__":
    main()
