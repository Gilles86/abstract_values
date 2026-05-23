"""Per-condition preferred-value shift in NPCr.

Uses the SessionShiftedLogGaussianPRF fits in
``derivatives/encoding_models/aprf-session-shift/sub-XX/func/``:
  * ``desc-mode_1_pe.nii.gz`` — pRF mode in session 1
  * ``desc-mode_2_pe.nii.gz`` — pRF mode in session 2
  * ``desc-r2_pe.nii.gz``     — explained variance

The CHF↔orientation mapping flips between sessions:
  even subject id → ses-1 = cdf,         ses-2 = inverse_cdf
  odd  subject id → ses-1 = inverse_cdf, ses-2 = cdf

We remap each voxel's (mode_1, mode_2) into (mode_cdf, mode_invcdf)
so the figures compare *conditions* rather than *session order*.

Three pages:
  1. Per-subject scatter mode_cdf vs mode_invcdf in NPCr (voxels filtered
     by cv-R²); y=x reference and a sign-of-shift coding.
  2. Per-subject histogram of (mode_invcdf − mode_cdf) shift in CHF.
  3. Group summary: per-subject mean shift ± per-voxel SEM.

Usage:
    python -m abstract_values.visualize.shifted_preferred_value
    python -m abstract_values.visualize.shifted_preferred_value --r2-thr 0.05
    python -m abstract_values.visualize.shifted_preferred_value --subjects 04 06
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

DERIV = Path(BIDS_FOLDER) / "derivatives" / "encoding_models" / "aprf-session-shift"
DEFAULT_OUT = Path(BIDS_FOLDER) / "derivatives" / "qa" / "shifted_preferred_value.pdf"

# Condition palette — matches behavior notebook
COND_COLOUR = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}


def _path(subject: str, desc: str) -> Path:
    return (DERIV / f"sub-{subject}" / "func"
            / f"sub-{subject}_task-abstractvalue_space-T1w_desc-{desc}_pe.nii.gz")


def _load_masked(subject: str, desc: str, mask_arr: np.ndarray,
                 mask_img_for_resample: nib.Nifti1Image) -> np.ndarray | None:
    p = _path(subject, desc)
    if not p.exists():
        return None
    img = nib.load(str(p))
    if img.shape != mask_arr.shape:
        # mask is in T1w space; mode files are in BOLD space
        return None
    return img.get_fdata().astype(np.float32)[mask_arr]


def _resampled_mask(roi_img: nib.Nifti1Image,
                    ref_img: nib.Nifti1Image) -> np.ndarray:
    if roi_img.ndim == 4 and roi_img.shape[-1] == 1:
        roi_img = nib.Nifti1Image(np.squeeze(roi_img.get_fdata()),
                                  roi_img.affine, roi_img.header)
    if roi_img.shape == ref_img.shape and np.allclose(roi_img.affine, ref_img.affine):
        return np.asarray(roi_img.get_fdata() > 0)
    resampled = nimage.resample_to_img(roi_img, ref_img,
                                       interpolation="nearest",
                                       force_resample=True,
                                       copy_header=True)
    return np.asarray(resampled.get_fdata() > 0)


def collect_subject(subject: str, roi: str = "NPCr",
                    r2_thr: float = 0.05) -> pd.DataFrame:
    """Per-voxel (mode_cdf, mode_invcdf, r2) for one subject in `roi`."""
    sub = Subject(subject, bids_folder=Path(BIDS_FOLDER))
    mode1_img = nib.load(str(_path(subject, "mode_1")))
    mode2_img = nib.load(str(_path(subject, "mode_2")))
    r2_img    = nib.load(str(_path(subject, "r2")))

    roi_img = sub.get_roi_mask(roi, hemi=None)
    mask_arr = _resampled_mask(roi_img, mode1_img)

    m1 = mode1_img.get_fdata().astype(np.float32)[mask_arr]
    m2 = mode2_img.get_fdata().astype(np.float32)[mask_arr]
    r2 = r2_img.get_fdata().astype(np.float32)[mask_arr]

    # Per-subject session→condition mapping
    s1_cond = sub.get_mapping(1)
    if s1_cond == "cdf":
        mode_cdf, mode_invcdf = m1, m2
    else:
        mode_cdf, mode_invcdf = m2, m1

    df = pd.DataFrame({
        "subject": subject,
        "voxel": np.arange(len(r2)),
        "mode_cdf": mode_cdf,
        "mode_invcdf": mode_invcdf,
        "r2": r2,
    })
    df = df[df.r2 > r2_thr].reset_index(drop=True)
    return df


def discover_subjects() -> list[str]:
    return sorted(p.name.removeprefix("sub-")
                  for p in DERIV.glob("sub-*")
                  if _path(p.name.removeprefix("sub-"), "mode_1").exists())


def page_scatter(by_sub: dict, value_lim: tuple[float, float], pdf: PdfPages):
    subjects = sorted(by_sub.keys(),
                      key=lambda s: (0 if s[0].isdigit() else 1, s))
    n = len(subjects)
    cols = 3
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.5, 2.4 * rows + 0.4),
                             constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()
    lo, hi = value_lim
    for i, sub in enumerate(subjects):
        ax = axes[i]
        df = by_sub[sub]
        # Colour by r²
        sc = ax.scatter(df.mode_cdf, df.mode_invcdf, c=df.r2,
                        cmap="viridis", vmin=0.05, vmax=0.3,
                        s=8, alpha=0.7, linewidth=0)
        ax.plot([lo, hi], [lo, hi], "--", color="0.6", lw=0.7, zorder=0)
        # Per-subject median shift
        med = float((df.mode_invcdf - df.mode_cdf).median())
        ax.text(0.04, 0.96, f"sub-{sub}\nn={len(df)}  Δ̃={med:+.2f} CHF",
                transform=ax.transAxes, fontsize=7.5, va="top",
                color="0.2")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    for j in range(len(subjects), len(axes)):
        axes[j].set_axis_off()
    fig.supxlabel("Preferred value — CDF condition  (CHF)", fontsize=9)
    fig.supylabel("Preferred value — Inverse-CDF condition  (CHF)", fontsize=9)
    fig.suptitle("Per-voxel preferred-value shift in NPCr",
                 fontsize=10, y=1.02)
    cbar = fig.colorbar(sc, ax=axes[: len(subjects)].tolist(), shrink=0.5,
                        pad=0.02, aspect=14)
    cbar.set_label("Session-shift fit R²", fontsize=8)
    sns.despine(fig=fig, offset=3, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_shift_hist(by_sub: dict, pdf: PdfPages):
    subjects = sorted(by_sub.keys(),
                      key=lambda s: (0 if s[0].isdigit() else 1, s))
    n = len(subjects)
    cols = 3
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.5, 1.9 * rows + 0.4),
                             constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()
    # Common x range
    shifts_all = np.concatenate(
        [(d.mode_invcdf - d.mode_cdf).values for d in by_sub.values()])
    xlo, xhi = np.percentile(shifts_all, [2, 98])
    for i, sub in enumerate(subjects):
        ax = axes[i]
        shift = (by_sub[sub].mode_invcdf - by_sub[sub].mode_cdf).values
        ax.hist(shift, bins=40, color="0.5", alpha=0.7,
                range=(xlo, xhi), linewidth=0)
        med = float(np.median(shift))
        ax.axvline(med, color="#C44E52", lw=1.3, zorder=4)
        ax.axvline(0, color="0.4", lw=0.6, ls="--", zorder=0)
        ax.text(0.04, 0.96, f"sub-{sub}\nmedian {med:+.2f} CHF",
                transform=ax.transAxes, fontsize=7.5, va="top", color="0.2")
        ax.set_xlim(xlo, xhi)
    for j in range(len(subjects), len(axes)):
        axes[j].set_axis_off()
    fig.supxlabel("Inverse-CDF − CDF  (CHF)", fontsize=9)
    fig.supylabel("Voxel count", fontsize=9)
    fig.suptitle("Per-voxel preferred-value shift (Inverse-CDF − CDF)",
                 fontsize=10, y=1.02)
    sns.despine(fig=fig, offset=3, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_group(by_sub: dict, pdf: PdfPages):
    subjects = sorted(by_sub.keys(),
                      key=lambda s: (0 if s[0].isdigit() else 1, s))
    rows = []
    for sub in subjects:
        df = by_sub[sub]
        rows.append(dict(
            subject=sub,
            mean_cdf=float(df.mode_cdf.mean()),
            mean_invcdf=float(df.mode_invcdf.mean()),
            median_shift=float((df.mode_invcdf - df.mode_cdf).median()),
            sem_shift=float((df.mode_invcdf - df.mode_cdf).std(ddof=1) /
                            np.sqrt(len(df))),
            n_voxels=len(df),
        ))
    g = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), constrained_layout=True)

    # Left: paired mean preferred value per condition
    ax = axes[0]
    for _, r in g.iterrows():
        ax.plot([0, 1], [r.mean_cdf, r.mean_invcdf], "-o",
                color="0.7", lw=0.8, ms=4, zorder=1)
    ax.scatter([0]*len(g), g.mean_cdf, s=60, color=COND_COLOUR["cdf"],
               edgecolor="black", linewidth=1.2, zorder=3)
    ax.scatter([1]*len(g), g.mean_invcdf, s=60, color=COND_COLOUR["inverse_cdf"],
               edgecolor="black", linewidth=1.2, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CDF", "Inverse-CDF"])
    ax.set_ylabel("Mean preferred value (CHF)")
    ax.set_title("Per-subject mean preferred value", fontsize=9, color="0.2")

    # Right: per-subject median shift with SEM bars
    ax = axes[1]
    ys = np.arange(len(g))
    ax.errorbar(g.median_shift.values, ys, xerr=g.sem_shift.values,
                fmt="o", color="0.2", ecolor="0.5", elinewidth=0.9, capsize=2.5,
                markersize=6)
    ax.axvline(0, color="0.4", lw=0.6, ls="--", zorder=0)
    grand_med = float(g.median_shift.median())
    ax.axvline(grand_med, color="#C44E52", lw=1.3, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"sub-{s}" for s in g.subject])
    ax.set_xlabel("Median shift  Inverse-CDF − CDF  (CHF)")
    ax.set_title(f"Per-subject median shift (grand median {grand_med:+.2f})",
                 fontsize=9, color="0.2")

    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return g


def run(subjects, r2_thr, out, value_lim):
    if subjects is None:
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No session-shift fits found.")
    print(f"Subjects: {subjects}  r²>{r2_thr}")
    by_sub = {}
    for sub in subjects:
        df = collect_subject(sub, r2_thr=r2_thr)
        if df.empty:
            print(f"  sub-{sub}: 0 voxels above threshold — skipping")
            continue
        print(f"  sub-{sub}: {len(df)} voxels above r²>{r2_thr}")
        by_sub[sub] = df

    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        page_scatter(by_sub, value_lim, pdf)
        page_shift_hist(by_sub, pdf)
        g = page_group(by_sub, pdf)
    tsv = out.with_suffix(".tsv")
    g.to_csv(tsv, sep="\t", index=False)
    print(f"Wrote {out}\nSidecar: {tsv}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--r2-thr", type=float, default=0.05,
                   help="Voxel selection threshold on session-shift R²")
    p.add_argument("--value-min", type=float, default=0.5)
    p.add_argument("--value-max", type=float, default=20.0)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, args.r2_thr, Path(args.out),
        (args.value_min, args.value_max))


if __name__ == "__main__":
    main()
