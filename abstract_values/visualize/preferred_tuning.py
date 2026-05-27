"""Population-level preferred tuning per ROI × condition.

For each ROI:
  - **V1**: per-voxel preferred orientation = argmax over a fine grid of
    the linear combination of fixed Von Mises basis functions weighted
    by that voxel's basis weights. When session-shift weights are
    available, computed separately for each session; else single
    histogram across all voxels.
  - **NPCr**: per-voxel preferred CHF = the ``mode_1`` / ``mode_2``
    parameters from the SessionShiftedLogGaussianPRF fit — already
    session-specific by construction.

Each voxel contributes one (preferred-value, condition) point per
session. Voxels are filtered by an R² threshold (default 0.05) so the
histogram reflects only voxels that are actually orientation- /
value-tuned at all.

Across-subject pooling: simply concatenate per-subject voxels — they're
not corresponded between brains, so a per-voxel ANOVA would be wrong.
What the histogram reports is 'across the cohort, where do the tuned
voxels of each ROI prefer to sit'.

Usage:
    python -m abstract_values.visualize.preferred_tuning
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

DERIV = Path(BIDS_FOLDER) / "derivatives" / "encoding_models"
DEFAULT_OUT = (Path(BIDS_FOLDER) / "derivatives" / "qa"
               / "preferred_tuning.pdf")

COND_COLOUR = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}

# V1 trained orientation range; matches mapping_invariance.
V1_ORI_GRID = np.deg2rad(np.linspace(7.5, 172.5, 200, dtype=np.float32))

# CHF axis for NPCr — covers full union of the two conditions with headroom.
CHF_LO, CHF_HI = 0.0, 45.0


def _load_nifti_voxels(path, mask_arr):
    """Load 3D NIfTI and return its values at masked voxels."""
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    return data[mask_arr > 0]


def _load_npcr_modes(subject, smoothed, r2_thr, bids_folder=BIDS_FOLDER):
    """Per-voxel (mode_cdf, mode_invcdf) for NPCr voxels that pass R² > thr.
    The session-to-condition mapping flips per subject so we remap
    mode_1 / mode_2 into condition-keyed columns."""
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    ssdir = (bids_folder / "derivatives" / "encoding_models"
             / "aprf-session-shift" / f"sub-{subject}" / "func")
    smooth = "_smoothed" if smoothed else ""
    p_m1 = ssdir / f"sub-{subject}_task-abstractvalue_space-T1w_desc-mode_1{smooth}_pe.nii.gz"
    p_m2 = ssdir / f"sub-{subject}_task-abstractvalue_space-T1w_desc-mode_2{smooth}_pe.nii.gz"
    p_r2 = ssdir / f"sub-{subject}_task-abstractvalue_space-T1w_desc-r2{smooth}_pe.nii.gz"
    mask_img = sub.get_roi_mask("NPCr", hemi=None)
    # Reslice the modes/r2 onto the mask grid if needed
    from nilearn import image as nli
    mask_arr = np.squeeze(mask_img.get_fdata()) > 0.5
    if not (p_m1.exists() and p_m2.exists() and p_r2.exists()):
        return pd.DataFrame()
    m1 = nli.resample_to_img(nib.load(str(p_m1)), mask_img,
                              interpolation="nearest").get_fdata()[mask_arr]
    m2 = nli.resample_to_img(nib.load(str(p_m2)), mask_img,
                              interpolation="nearest").get_fdata()[mask_arr]
    r2 = nli.resample_to_img(nib.load(str(p_r2)), mask_img,
                              interpolation="nearest").get_fdata()[mask_arr]
    keep = (r2 > r2_thr) & np.isfinite(m1) & np.isfinite(m2)
    if keep.sum() == 0:
        return pd.DataFrame()
    # Map session number → condition. Session 1 → mode_1, session 2 → mode_2.
    cond_s1 = sub.get_mapping(1)
    cond_s2 = sub.get_mapping(2)
    rows = pd.DataFrame({
        "mode_cdf":    np.where(cond_s1 == "cdf",         m1[keep], m2[keep]),
        "mode_invcdf": np.where(cond_s1 == "cdf",         m2[keep], m1[keep]),
    })
    rows["subject"] = subject
    return rows


def _load_v1_preferred(subject, smoothed, r2_thr,
                        n_basis=8, kappa=2.0,
                        bids_folder=BIDS_FOLDER, session_shift=True):
    """Per-voxel preferred orientation (radians, 0–π) for BensonV1 voxels.

    With ``session_shift=True``, returns per-condition preferred
    orientations from the vonmises-session-shift per-session weight
    files; otherwise falls back to the joint vonmises fit (preferred
    orientation per voxel is the same in both conditions by construction).
    """
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    smooth = "_smoothed" if smoothed else ""
    mask_img = sub.get_roi_mask("BensonV1", hemi="LR")
    mask_arr = np.squeeze(mask_img.get_fdata()) > 0.5
    from nilearn import image as nli

    # Basis grid
    mus = np.linspace(0, np.pi, n_basis, endpoint=False, dtype=np.float32)
    # Axial Von Mises: cos(2(x − μ)). Same as braincoder's AxialVonMisesPRF
    # basis with kappa applied.
    basis = np.exp(kappa * np.cos(2 * (V1_ORI_GRID[:, None] - mus[None, :])))
    basis /= basis.sum(axis=0, keepdims=True)  # normalize per channel

    ss_dir = (bids_folder / "derivatives" / "encoding_models"
              / "vonmises-session-shift" / f"sub-{subject}" / "func")
    joint_dir = (bids_folder / "derivatives" / "encoding_models"
                 / "vonmises" / f"sub-{subject}" / "func")

    def _voxel_preferred_ori(weights_img_path, r2_img_path):
        """Given a 4D weights file (n_basis volumes) and an R² file,
        return preferred_orientation_rad per masked voxel that passes
        R² > thr."""
        w_img = nli.resample_to_img(nib.load(str(weights_img_path)),
                                      mask_img, interpolation="nearest")
        r2_img = nli.resample_to_img(nib.load(str(r2_img_path)),
                                       mask_img, interpolation="nearest")
        w = w_img.get_fdata()[mask_arr]      # (n_voxels, n_basis)
        r2 = r2_img.get_fdata()[mask_arr]
        keep = (r2 > r2_thr) & np.isfinite(r2)
        if keep.sum() == 0:
            return np.empty(0), np.empty(0)
        # Tuning curve per voxel = basis · w (200, n_basis) · (n_voxels, n_basis)
        # Vectorise: (200, n_voxels) = basis @ w.T
        curves = basis @ w[keep].T
        pref_idx = np.argmax(curves, axis=0)
        return V1_ORI_GRID[pref_idx], keep

    if session_shift:
        p_w1 = ss_dir / f"sub-{subject}_task-abstractvalue_space-T1w_desc-weights_1{smooth}_pe.nii.gz"
        p_w2 = ss_dir / f"sub-{subject}_task-abstractvalue_space-T1w_desc-weights_2{smooth}_pe.nii.gz"
        p_r1 = ss_dir / f"sub-{subject}_task-abstractvalue_space-T1w_desc-r2_1{smooth}_pe.nii.gz"
        p_r2 = ss_dir / f"sub-{subject}_task-abstractvalue_space-T1w_desc-r2_2{smooth}_pe.nii.gz"
        if not all(p.exists() for p in (p_w1, p_w2, p_r1, p_r2)):
            return pd.DataFrame()
        pref1, _ = _voxel_preferred_ori(p_w1, p_r1)
        pref2, _ = _voxel_preferred_ori(p_w2, p_r2)
        # Map session → condition
        cond_s1 = sub.get_mapping(1)
        # Both session preferred lists are filtered independently — for a
        # joint per-condition histogram we just concatenate. (Voxel
        # identity isn't preserved across sessions because the keep masks
        # differ; that's fine for the cohort distribution we're showing.)
        cdf_pref    = pref1 if cond_s1 == "cdf"         else pref2
        invcdf_pref = pref2 if cond_s1 == "cdf"         else pref1
        return pd.DataFrame({
            "subject": subject,
            "preferred_rad": np.concatenate([cdf_pref, invcdf_pref]),
            "condition": (["cdf"] * len(cdf_pref)
                           + ["inverse_cdf"] * len(invcdf_pref)),
        })

    # Fall-back: joint fit. Single preferred orientation per voxel,
    # replicate across conditions for plotting consistency.
    p_w = joint_dir / f"sub-{subject}_task-abstractvalue_space-T1w_desc-weights{smooth}_pe.nii.gz"
    p_r2 = joint_dir / f"sub-{subject}_task-abstractvalue_space-T1w_desc-r2{smooth}_pe.nii.gz"
    if not (p_w.exists() and p_r2.exists()):
        return pd.DataFrame()
    pref, _ = _voxel_preferred_ori(p_w, p_r2)
    if len(pref) == 0:
        return pd.DataFrame()
    return pd.DataFrame({
        "subject": subject,
        "preferred_rad": np.concatenate([pref, pref]),
        "condition": ["joint"] * len(pref) * 2,
    })


def discover_subjects():
    seen = set()
    for p in DERIV.glob("aprf-session-shift/sub-*"):
        seen.add(p.name.removeprefix("sub-"))
    return sorted(seen, key=lambda s: (0 if s[0].isdigit() else 1, s))


def page(subjects, r2_thr, smoothed, pdf):
    # ── collect ─────────────────────────────────────────────────────────────
    npcr_rows = []
    for s in subjects:
        try:
            df = _load_npcr_modes(s, smoothed, r2_thr)
            if not df.empty: npcr_rows.append(df)
        except Exception as exc:
            print(f"  NPCr sub-{s}: skip ({exc})")
    df_npcr = (pd.concat(npcr_rows, ignore_index=True)
                if npcr_rows else pd.DataFrame())

    # V1: try session-shift; fall back to joint if any subject is missing
    v1_rows = []
    used_ss = True
    for s in subjects:
        try:
            df = _load_v1_preferred(s, smoothed, r2_thr, session_shift=True)
            if df.empty:
                used_ss = False
                df = _load_v1_preferred(s, smoothed, r2_thr, session_shift=False)
            if not df.empty: v1_rows.append(df)
        except Exception as exc:
            print(f"  V1 sub-{s}: skip ({exc})")
    df_v1 = (pd.concat(v1_rows, ignore_index=True)
              if v1_rows else pd.DataFrame())
    if not df_v1.empty:
        df_v1["preferred_deg"] = np.rad2deg(df_v1["preferred_rad"])

    # ── plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(7.25, 5.6),
                              constrained_layout=True)
    smooth_lbl = "smoothed" if smoothed else "unsmoothed"

    # ─── V1 preferred orientation ──────────────────────────────────────────
    ax = axes[0]
    if df_v1.empty:
        ax.text(0.5, 0.5, "No V1 data", transform=ax.transAxes,
                ha="center", va="center", color="0.5")
        ax.set_xticks([]); ax.set_yticks([])
    else:
        bins = np.linspace(0, 180, 31)
        if used_ss and "joint" not in df_v1["condition"].unique():
            for cond in ("cdf", "inverse_cdf"):
                vals = df_v1[df_v1["condition"] == cond]["preferred_deg"]
                if len(vals) == 0: continue
                ax.hist(vals, bins=bins, color=COND_COLOUR[cond],
                        alpha=0.55, edgecolor="white", linewidth=0.3,
                        label=f"{'CDF' if cond=='cdf' else 'InvCDF'}  "
                              f"(n={len(vals):,} voxels)")
        else:
            vals = df_v1["preferred_deg"]
            # Single joint distribution
            ax.hist(vals, bins=bins, color="#3B5BA5",
                    alpha=0.7, edgecolor="white", linewidth=0.3,
                    label=f"Joint  (n={len(vals):,} voxels)")
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xlabel("V1 preferred orientation (deg)")
        ax.set_ylabel("Voxel count")
        ax.legend(loc="upper right", fontsize=7.5,
                   title=f"{smooth_lbl}, R² > {r2_thr:.2f}",
                   title_fontsize=7.5)

    # ─── NPCr preferred CHF ────────────────────────────────────────────────
    ax = axes[1]
    if df_npcr.empty:
        ax.text(0.5, 0.5, "No NPCr data", transform=ax.transAxes,
                ha="center", va="center", color="0.5")
        ax.set_xticks([]); ax.set_yticks([])
    else:
        bins = np.linspace(CHF_LO, CHF_HI, 31)
        ax.hist(df_npcr["mode_cdf"], bins=bins,
                color=COND_COLOUR["cdf"], alpha=0.55,
                edgecolor="white", linewidth=0.3,
                label=f"CDF  (n={len(df_npcr):,} voxels)")
        ax.hist(df_npcr["mode_invcdf"], bins=bins,
                color=COND_COLOUR["inverse_cdf"], alpha=0.55,
                edgecolor="white", linewidth=0.3,
                label=f"InvCDF  (n={len(df_npcr):,} voxels)")
        ax.set_xlim(CHF_LO, CHF_HI)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_xlabel("NPCr preferred value (CHF)")
        ax.set_ylabel("Voxel count")
        ax.legend(loc="upper right", fontsize=7.5,
                   title=f"{smooth_lbl}, R² > {r2_thr:.2f}",
                   title_fontsize=7.5)

    sns.despine(fig=fig, offset=4, trim=True)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def run(subjects, out, r2_thr, smoothings):
    from matplotlib.backends.backend_pdf import PdfPages
    if subjects is None:
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No subjects found with aprf-session-shift fits.")
    print(f"Subjects: {subjects}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        for smoothed in smoothings:
            print(f"\n=== {'smoothed' if smoothed else 'unsmoothed'} ===")
            page(subjects, r2_thr, smoothed, pdf)
    print(f"\nWrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--r2-thr", type=float, default=0.05,
                    help="R² threshold for voxel inclusion (default 0.05)")
    p.add_argument("--smoothings", nargs="+", type=int, default=[0],
                    help="0=unsmoothed, 1=smoothed (default: just unsmoothed)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, Path(args.out), args.r2_thr,
        [bool(s) for s in args.smoothings])


if __name__ == "__main__":
    main()
