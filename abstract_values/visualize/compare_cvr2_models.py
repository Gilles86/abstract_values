"""Cross-model cv-R² comparison in V1 and NPCr.

For the gabor (V1) and value (NPCr) decoding paths we use structurally
different encoding models. This script answers: how do the cv-R²
distributions compare voxel-wise within each ROI?

  V1 (BensonV1):
    aprf            — log-Gaussian PRF, 1 RF / voxel (per-voxel parametric)
    vonmises        — basis set, 8 fixed RFs + closed-form WeightFitter

  NPCr:
    aprf            — log-Gaussian PRF, 1 RF / voxel
    aprf-weighted   — basis set, 8 fixed log-Gaussian RFs + WeightFitter

The aprf-weighted model is the *value-side analog* of vonmises (same
structural family, same fitter), so the aprf vs aprf-weighted contrast
in NPCr mirrors the aprf vs vonmises contrast in V1.

Three pages per ROI:
  1. Per-subject paired histogram (overlaid KDE of cv-R²) for each model.
  2. Per-subject voxel-wise scatter (basis vs per-voxel) with y=x reference.
  3. Group summary: mean cv-R² per (subject, model) as a paired strip.

Usage:
    python -m abstract_values.visualize.compare_cvr2_models
    python -m abstract_values.visualize.compare_cvr2_models --subjects 08 09
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from nilearn import image as nimage

from abstract_values.utils.data import BIDS_FOLDER, Subject

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelpad": 4,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2, "legend.frameon": False, "legend.handlelength": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})
sns.set_context("paper")

ENC_DIR = Path(BIDS_FOLDER) / "derivatives" / "encoding_models"
DEFAULT_OUT = Path(BIDS_FOLDER) / "derivatives" / "qa" / "compare_cvr2_models.pdf"

# ROI → (mask name, hemi, list of (label, model_dir) pairs to compare)
ROI_SPECS = [
    ("V1 (BensonV1)", "BensonV1", "LR", [
        ("aprf",     "aprf.cv"),
        ("vonmises", "vonmises.cv"),
    ]),
    ("NPCr", "NPCr", None, [
        ("aprf",          "aprf.cv"),
        ("aprf-weighted", "aprf-weighted.cv"),
    ]),
]

# Per-model colour. Per-voxel parametric models in coral, basis sets in teal.
MODEL_COLOURS = {
    "aprf":          "#C44E52",
    "aprf-weighted": "#2A9D8F",
    "vonmises":      "#2A9D8F",
}


def _cvr2_path(subject: str, model_dir: str) -> Path:
    return (ENC_DIR / model_dir / f"sub-{subject}" / "func"
            / f"sub-{subject}_task-abstractvalue_space-T1w_desc-cvr2_pe.nii.gz")


def _load_cvr2_img(subject: str, model_dir: str) -> nib.Nifti1Image | None:
    p = _cvr2_path(subject, model_dir)
    if not p.exists():
        return None
    return nib.load(str(p))


def _resample_mask(mask_img: nib.Nifti1Image,
                   ref_img: nib.Nifti1Image) -> np.ndarray:
    """Nearest-neighbour resample the T1w-res ROI mask onto the BOLD grid."""
    # Some ROI masks are stored with a singleton time dim (192,256,256,1) —
    # collapse it so resample_to_img sees a 3D image.
    if mask_img.ndim == 4 and mask_img.shape[-1] == 1:
        mask_img = nib.Nifti1Image(np.squeeze(mask_img.get_fdata()),
                                   mask_img.affine, mask_img.header)
    if mask_img.shape == ref_img.shape and np.allclose(
            mask_img.affine, ref_img.affine):
        return np.asarray(mask_img.get_fdata() > 0)
    resampled = nimage.resample_to_img(mask_img, ref_img,
                                       interpolation="nearest",
                                       force_resample=True,
                                       copy_header=True)
    return np.asarray(resampled.get_fdata() > 0)


def collect_roi(subjects: list[str], mask_name: str, hemi: str | None,
                model_specs: list[tuple[str, str]]) -> dict:
    """Returns dict[subject] = dict[label] = (n_voxels,) ndarray of cv-R²."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for sub in subjects:
        try:
            mask_img = Subject(sub, bids_folder=Path(BIDS_FOLDER)).get_roi_mask(
                mask_name, hemi=hemi)
        except FileNotFoundError as exc:
            print(f"  sub-{sub}: skip ({exc})")
            continue
        # All cv-R² files for a given subject share the same affine (BOLD grid)
        # — resample the mask once on the first model we encounter.
        mask_arr = None
        per_model = {}
        for label, model_dir in model_specs:
            img = _load_cvr2_img(sub, model_dir)
            if img is None:
                print(f"  sub-{sub} / {label}: missing")
                continue
            if mask_arr is None:
                mask_arr = _resample_mask(mask_img, img)
            arr = img.get_fdata().astype(np.float32)[mask_arr]
            # Clip catastrophically negative values (a few voxels with cv-R² ≪ 0
            # blow up the plot extents).
            per_model[label] = np.clip(arr, -0.5, None)
        if len(per_model) == len(model_specs):
            out[sub] = per_model
    return out


def page_histograms(roi_label: str, model_specs, data: dict, pdf: PdfPages):
    """One small panel per subject; KDE per model, shared x-axis."""
    subjects = sorted(data.keys(),
                      key=lambda s: (0 if s[0].isdigit() else 1, s))
    if not subjects:
        return
    n = len(subjects)
    cols = 4
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(7.5, 2.0 * rows + 0.6),
                             constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()
    # Shared x range across the panel: use 1st & 99th percentile across subjects.
    all_vals = np.concatenate(
        [data[s][lab] for s in subjects for lab, _ in model_specs])
    xlo, xhi = np.percentile(all_vals, [1, 99])
    xhi = max(xhi, 0.05)
    for i, sub in enumerate(subjects):
        ax = axes[i]
        for label, _ in model_specs:
            arr = data[sub][label]
            sns.kdeplot(arr, ax=ax, color=MODEL_COLOURS[label], lw=1.1,
                        clip=(xlo, xhi), fill=True, alpha=0.18,
                        cut=0, label=label)
        ax.axvline(0, color="0.7", lw=0.6, ls="--", zorder=0)
        ax.set_title(f"sub-{sub}  n={len(arr)} vox", fontsize=8, color="0.2")
        ax.set_xlabel("")
        ax.set_ylabel("")
    for j in range(len(subjects), len(axes)):
        axes[j].set_axis_off()
    # Common labels
    fig.supxlabel("cv-R²", fontsize=9)
    fig.supylabel("Density", fontsize=9)
    fig.suptitle(f"{roi_label}: cv-R² distribution per model",
                 fontsize=10, y=1.02)
    # One legend on the first panel
    axes[0].legend(loc="upper left", fontsize=8)
    sns.despine(fig=fig, offset=3, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_scatter(roi_label: str, model_specs, data: dict, pdf: PdfPages):
    """Per-subject voxel-wise scatter: basis-set on y, per-voxel on x."""
    subjects = sorted(data.keys(),
                      key=lambda s: (0 if s[0].isdigit() else 1, s))
    if not subjects:
        return
    label_x, label_y = model_specs[0][0], model_specs[1][0]   # aprf, basis
    n = len(subjects)
    cols = 4
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.5, 2.0 * rows + 0.6),
                             constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()
    # Shared limits.
    all_vals = np.concatenate(
        [data[s][lab] for s in subjects for lab, _ in model_specs])
    lo, hi = np.percentile(all_vals, [1, 99])
    hi = max(hi, 0.05)
    for i, sub in enumerate(subjects):
        ax = axes[i]
        x = data[sub][label_x]
        y = data[sub][label_y]
        ax.scatter(x, y, s=2, color="0.25", alpha=0.25, linewidth=0)
        ax.plot([lo, hi], [lo, hi], color="0.7", lw=0.6, ls="--", zorder=0)
        # Quadrant fractions: how often is basis > per-voxel?
        n_total = len(x)
        n_basis_wins = int((y > x).sum())
        frac = n_basis_wins / n_total
        ax.text(0.04, 0.96, f"sub-{sub}\n{frac:.0%} {label_y} > {label_x}",
                transform=ax.transAxes, fontsize=7, va="top",
                color="0.2")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    for j in range(len(subjects), len(axes)):
        axes[j].set_axis_off()
    fig.supxlabel(f"cv-R² · {label_x}", fontsize=9)
    fig.supylabel(f"cv-R² · {label_y}", fontsize=9)
    fig.suptitle(f"{roi_label}: voxel-wise paired cv-R²", fontsize=10, y=1.02)
    sns.despine(fig=fig, offset=3, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_summary(roi_label: str, model_specs, data: dict, pdf: PdfPages):
    """Group summary: paired subject means + voxel-wise fraction-better."""
    subjects = sorted(data.keys(),
                      key=lambda s: (0 if s[0].isdigit() else 1, s))
    if not subjects:
        return
    rows = []
    for sub in subjects:
        for label, _ in model_specs:
            arr = data[sub][label]
            rows.append(dict(subject=sub, model=label,
                             mean_cvr2=float(np.mean(arr)),
                             median_cvr2=float(np.median(arr))))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0),
                             constrained_layout=True)
    # Left: paired stripplot of subject means
    ax = axes[0]
    order = [m[0] for m in model_specs]
    palette = [MODEL_COLOURS[m] for m in order]
    sns.stripplot(data=df, x="model", y="mean_cvr2", order=order,
                  palette=palette, size=6, alpha=0.85, jitter=False, ax=ax)
    # Pair lines
    for sub in subjects:
        sub_df = df[df.subject == sub].set_index("model").reindex(order)
        ax.plot([0, 1], sub_df["mean_cvr2"].values, "-", color="0.7",
                lw=0.8, zorder=0)
    # Group mean
    grp = df.groupby("model")["mean_cvr2"].mean().reindex(order)
    for i, (m, v) in enumerate(grp.items()):
        ax.scatter(i, v, s=110, marker="D", facecolor=MODEL_COLOURS[m],
                   edgecolor="black", linewidth=1.5, zorder=5)
    ax.axhline(0, color="0.7", lw=0.6, ls="--", zorder=0)
    ax.set_ylabel("Mean cv-R²")
    ax.set_xlabel("")
    ax.set_title(f"{roi_label}: subject-mean cv-R²", fontsize=9, color="0.2")

    # Right: voxel-wise fraction of voxels where basis-model > per-voxel-model
    ax = axes[1]
    label_x, label_y = order[0], order[1]
    fracs = []
    for sub in subjects:
        x = data[sub][label_x]
        y = data[sub][label_y]
        fracs.append((sub, (y > x).mean()))
    fdf = pd.DataFrame(fracs, columns=["subject", "frac"])
    sns.stripplot(data=fdf, x=["all"] * len(fdf), y="frac",
                  size=6, color=MODEL_COLOURS[label_y], jitter=0.15, ax=ax)
    ax.axhline(0.5, color="0.7", lw=0.6, ls="--", zorder=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"P(voxel-wise: {label_y} > {label_x})")
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_title(f"{roi_label}: voxel-wise win rate", fontsize=9, color="0.2")

    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return df


def discover_subjects() -> list[str]:
    subs = set()
    for d in ("aprf.cv", "aprf-weighted.cv", "vonmises.cv"):
        for p in (ENC_DIR / d).glob("sub-*/func/sub-*_task-abstractvalue_space-T1w_desc-cvr2_pe.nii.gz"):
            subs.add(p.parent.parent.name.removeprefix("sub-"))
    return sorted(subs)


def run(subjects: list[str] | None, out: Path):
    if subjects is None:
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No mean cv-R² NIfTIs found.")
    print(f"Subjects: {subjects}")

    out.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    with PdfPages(out) as pdf:
        for roi_label, mask_name, hemi, model_specs in ROI_SPECS:
            print(f"\n--- {roi_label} ---")
            data = collect_roi(subjects, mask_name, hemi, model_specs)
            if not data:
                print(f"  no subjects with full data — skipping")
                continue
            page_histograms(roi_label, model_specs, data, pdf)
            page_scatter(roi_label, model_specs, data, pdf)
            df = page_summary(roi_label, model_specs, data, pdf)
            df["roi"] = roi_label
            summaries.append(df)
    if summaries:
        tsv = out.with_suffix(".tsv")
        pd.concat(summaries, ignore_index=True).to_csv(tsv, sep="\t", index=False)
        print(f"Sidecar TSV: {tsv}")
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+",
                   help="Subject labels (default: discover from cv-R² NIfTIs)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, Path(args.out))


if __name__ == "__main__":
    main()
