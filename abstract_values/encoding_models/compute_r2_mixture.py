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
    r2_p_signal_threshold,
    r2_posterior_signal,
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

    # Degeneracy flag — the mixture sometimes fails to identify a clean
    # signal component (very small Δμ in logit space, or signal σ runs
    # away). Downstream FDR uses the threshold as-is when not degenerate,
    # else falls back to a fixed top-N by R².
    delta_mu_logit = fit["signal_mu"] - fit["noise_mu"]
    sigma_ratio = fit["signal_sigma"] / max(fit["noise_sigma"], 1e-6)
    is_degenerate = (
        delta_mu_logit < 0.5
        or sigma_ratio > 1.4
        or fit["signal_weight"] < 0.3
        or fit["signal_weight"] > 0.85
    )
    fit["degenerate"] = bool(is_degenerate)
    fit["degenerate_reasons"] = {
        "delta_mu_logit_lt_0.5":    bool(delta_mu_logit < 0.5),
        "sigma_signal_over_noise_gt_1.4": bool(sigma_ratio > 1.4),
        "signal_weight_outside_0.3_0.85": bool(
            fit["signal_weight"] < 0.3 or fit["signal_weight"] > 0.85),
    }

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
          f"signal μ_R²={fit['signal_mean_r2']:.3f}  w_signal={fit['signal_weight']:.2f}"
          + ("  ⚠ DEGENERATE" if fit['degenerate'] else ""))

    # Diagnostic PDF — histogram + projected mixture components + sum +
    # vertical threshold markers (FDR α=0.05, p_signal ≥ 0.5, p_signal ≥ 0.95).
    if diagnostics_dir is not None:
        from scipy.stats import norm
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        pdf_fn = diagnostics_dir / f"sub-{subject}_r2_mixture{smtag}.pdf"
        r2_used = r2_all[finite]
        p995 = float(np.percentile(r2_used, 99.5))
        x = np.linspace(1e-4, p995, 600)
        z_x = np.log(x / (1.0 - x))
        jac = 1.0 / (x * (1.0 - x))
        n_pdf = norm.pdf(z_x, fit["noise_mu"],  fit["noise_sigma"]) \
                * fit["noise_weight"] * jac
        s_pdf = norm.pdf(z_x, fit["signal_mu"], fit["signal_sigma"]) \
                * fit["signal_weight"] * jac
        sum_pdf = n_pdf + s_pdf

        thr = {
            ("FDR ≤ 0.05",        "#3B5BA5"): r2_fdr_threshold_from_fit(fit, alpha=0.05),
            ("P(signal) ≥ 0.5",   "#5D8C3F"): r2_p_signal_threshold(fit, p=0.5),
            ("P(signal) ≥ 0.95",  "#C44E52"): r2_p_signal_threshold(fit, p=0.95),
        }

        fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
        ax.hist(r2_used[r2_used <= p995], bins=80, density=True,
                color="0.78", lw=0, label="Brain R² (data)")
        ax.plot(x, n_pdf, color="0.45", lw=1.0, label="Noise component")
        ax.plot(x, s_pdf, color="0.15", lw=1.0, ls="--", label="Signal component")
        ax.plot(x, sum_pdf, color="0.0", lw=1.5, label="Mixture (sum)")
        for (label, color), t in thr.items():
            if not np.isfinite(t):
                continue
            ax.axvline(t, color=color, lw=1.0, ls=":", zorder=2)
            ax.text(t, ax.get_ylim()[1] * 0.97,
                    f" {label}\n R²={t:.3f}",
                    color=color, fontsize=7, va="top", ha="left")
        ax.set_xlabel("R²"); ax.set_ylabel("Density")
        ax.set_xlim(0, p995); ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=7, loc="center right")
        ax.set_title(f"sub-{subject}  ·  {model}{smtag}  ·  whole-brain R² mixture\n"
                     f"noise μ={fit['noise_mean_r2']:.3f}  "
                     f"signal μ={fit['signal_mean_r2']:.3f}  "
                     f"w_signal={fit['signal_weight']:.2f}",
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
                             auto_fit: bool = True) -> dict | None:
    """Look up the whole-brain mixture for (subject, model) and return the
    FDR-controlled R² threshold along with a ``degenerate`` flag.

    Returns ``{'threshold': float, 'degenerate': bool, 'info': dict}`` or
    ``None`` if no mixture exists and auto-fit failed. Callers should fall
    back to a deterministic voxel-count selection when ``degenerate`` is
    True.
    """
    info = load_brain_mixture(subject, model, bids_folder, smoothed=smoothed)
    if info is None and auto_fit:
        diag = Path(bids_folder) / "derivatives" / "qa" / "r2_mixture" / model
        info = fit_subject_model(subject, model, bids_folder,
                                  smoothed=smoothed, diagnostics_dir=diag)
    if info is None:
        return None
    return {
        "threshold": r2_fdr_threshold_from_fit(info, alpha=alpha),
        "degenerate": bool(info.get("degenerate", False)),
        "info": info,
    }


def get_brain_p_signal_threshold(subject: str, model: str, bids_folder: Path,
                                  p: float = 0.95, smoothed: bool = False,
                                  auto_fit: bool = True) -> dict | None:
    """Look up the whole-brain mixture for (subject, model) and return the
    R² threshold above which the posterior probability of being a signal
    voxel is ≥ ``p``, along with a ``degenerate`` flag.

    Sibling of :func:`get_brain_fdr_threshold` — same lookup / auto-fit /
    return shape, but uses ``r2_p_signal_threshold`` (posterior on the
    signal component) rather than the FDR control. Callers should fall
    back to a deterministic voxel-count selection when ``degenerate`` is
    True.

    Returns ``{'threshold': float, 'degenerate': bool, 'info': dict}`` or
    ``None`` if no mixture exists and auto-fit failed.
    """
    info = load_brain_mixture(subject, model, bids_folder, smoothed=smoothed)
    if info is None and auto_fit:
        diag = Path(bids_folder) / "derivatives" / "qa" / "r2_mixture" / model
        info = fit_subject_model(subject, model, bids_folder,
                                  smoothed=smoothed, diagnostics_dir=diag)
    if info is None:
        return None
    return {
        "threshold": r2_p_signal_threshold(info, p=p),
        "degenerate": bool(info.get("degenerate", False)),
        "info": info,
    }


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
    p.add_argument("--models", nargs="+", default=["aprf"],
                   choices=DEFAULT_MODELS,
                   help="Encoding models whose R² to mix (default: aprf). "
                        "Pass multiple to do them all in one go.")
    p.add_argument("--smoothed", action="store_true",
                   help="Use the BOLD-smoothed variant's R² NIfTI")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    args = p.parse_args()

    bids = Path(args.bids_folder)

    for model in args.models:
        diag = bids / "derivatives" / "qa" / "r2_mixture" / model
        if args.all:
            subjects = _discover_subjects(bids, model, smoothed=args.smoothed)
        elif args.subject:
            subjects = [args.subject]
        else:
            raise SystemExit("Pass --subject or --all")
        if not subjects:
            print(f"({model}: no subjects with desc-r2 NIfTIs — skipping)")
            continue
        print(f"\n=== Fitting {model} mixture for {len(subjects)} subjects ===")
        for sub in subjects:
            fit_subject_model(sub, model, bids, smoothed=args.smoothed,
                               diagnostics_dir=diag)


if __name__ == "__main__":
    main()
