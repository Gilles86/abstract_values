"""Does preferred ORIENTATION shift across the two value mappings? — V1 vs NPCr.

The decisive dissociation between a retinotopic/orientation code and an
abstract value code, with no learning axis needed (participants arrive
already trained):

  - **Retinotopy / orientation (V1)**: a voxel is anchored to a physical
    orientation, shown identically in both conditions, so its preferred
    orientation is the SAME across cdf and inverse_cdf →
    θ_invcdf ≈ θ_cdf  (the identity line).

  - **Abstract value (NPCr)**: a voxel coding value v responds to whichever
    orientation maps to v, and the two mappings differ, so its preferred
    orientation SHIFTS across conditions along the value-preserving curve
    θ_invcdf = invcdf⁻¹( cdf(θ_cdf) ).

Crucially, the population efficient-coding signatures (preferred-value density
∝ value density, width ∝ 1/density) are confounded with "static orientation
tuning seen through the nonlinear mapping" ONLY IF voxels have fixed
orientation tuning. So showing that NPCr voxels move OFF the identity line
(toward the value-preserving curve) is what licenses reading the value
reallocation as genuine.

Per-voxel preferred orientation per condition comes from the
``vonmises-session-shift`` basis weights (weights_1 / weights_2), reconstructed
as the argmax of the weighted von Mises basis (same recipe as
``compute_preferred_orientation`` / ``preferred_tuning``). Voxels kept where
BOTH conditions fit (r2_1 and r2_2 > --r2-thr).

Output: a 2-panel figure (V1, NPCr) of θ_cdf vs θ_invcdf with the identity and
value-preserving reference curves, plus a per-voxel TSV.

Usage:
    python -m abstract_values.visualize.orientation_shift_v1_vs_npc
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
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from nilearn import image as nli

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
    "lines.linewidth": 1.2, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 150, "savefig.dpi": 300,
})
sns.set_context("paper")

C_IDENTITY = "0.45"          # neutral gray — reference (retinotopy)
C_VALUEPRES = "#C44E52"      # red — value-preserving prediction

# von Mises basis (must match compute_preferred_orientation / preferred_tuning).
N_BASIS = 8
KAPPA = 2.0
ORI_GRID = np.deg2rad(np.linspace(7.5, 172.5, 200, dtype=np.float32))
MAPPINGS_YML = (Path(__file__).resolve().parents[2]
                / "experiment" / "settings" / "sns_multisubject.yml")
ROIS = [("BensonV1", "LR"), ("NPCr", None)]
DEFAULT_OUT = Path(BIDS_FOLDER) / "derivatives" / "qa" / "orientation_shift_v1_vs_npc.pdf"


def load_mappings(yml_path=MAPPINGS_YML):
    m = yaml.safe_load(Path(yml_path).read_text())["mappings"]
    ori = np.asarray(m["orientations"], float)
    return ori, np.asarray(m["cdf"], float), np.asarray(m["inverse_cdf"], float)


def value_preserving_curve(theta_cdf_deg, ori, cdf, invcdf):
    """Predicted θ_invcdf for a value-coding voxel: the orientation that, under
    inverse_cdf, yields the same value that θ_cdf yields under cdf. Both
    mappings are monotonic in orientation, so np.interp inverts them."""
    v = np.interp(theta_cdf_deg, ori, cdf)            # value at θ under cdf
    return np.interp(v, invcdf, ori)                  # orientation at value under invcdf


def _basis():
    mus = np.linspace(0, np.pi, N_BASIS, endpoint=False, dtype=np.float32)
    b = np.exp(KAPPA * np.cos(2 * (ORI_GRID[:, None] - mus[None, :])))
    return b / b.sum(axis=0, keepdims=True)


def _preferred_deg(weights_voxels, basis):
    """weights_voxels: (n_vox, n_basis) → preferred orientation in degrees [7.5,172.5]."""
    curves = basis @ weights_voxels.T                 # (200, n_vox)
    return np.rad2deg(ORI_GRID[np.argmax(curves, axis=0)])


def _joint_cvr2_path(subject, model, bids_folder):
    return (Path(bids_folder) / "derivatives" / "encoding_models" / model
            / f"sub-{subject}" / "func"
            / f"sub-{subject}_task-abstractvalue_space-T1w_desc-cvr2_pe.nii.gz")


def load_subject_roi(subject, roi, hemi, bids_folder, signal="cvr2_null",
                     r2_thr=0.05, cv_model="vonmises.cv",
                     null_model="aprf-null.cv", cvr2_margin=0.0,
                     cvr2_floor=0.0):
    """Per-voxel (θ_cdf, θ_invcdf) for one subject × ROI, kept at clear-signal
    voxels. Returns a DataFrame (possibly empty).

    ``signal``:
      - ``"cvr2_null"`` (default): keep voxels where the orientation model's
        cross-validated R² both beats the null model's cvR²
        (``cvr2_vonmises > cvr2_null + cvr2_margin``) AND clears an absolute
        floor (``cvr2_vonmises > cvr2_floor``). The floor is what makes this a
        *clear*-signal gate: ``cvr2_floor=0`` keeps only voxels that genuinely
        predict held-out data better than the mean (~8% of V1/NPCr voxels),
        whereas beating the null alone is lenient (null ≈ −0.03, keeps ~35%).
      - ``"r2"``: legacy — keep where both per-condition fits exceed ``r2_thr``.
    """
    sub = Subject(subject, bids_folder=bids_folder)
    sdir = (Path(bids_folder) / "derivatives" / "encoding_models"
            / "vonmises-session-shift" / f"sub-{subject}" / "func")
    fn = (lambda d: sdir / f"sub-{subject}_task-abstractvalue_space-T1w"
                           f"_desc-{d}_pe.nii.gz")
    need = {d: fn(d) for d in ("weights_1", "weights_2", "r2_1", "r2_2")}
    if not all(p.exists() for p in need.values()):
        return pd.DataFrame()

    mask_img = sub.get_roi_mask(roi, hemi=hemi)
    mask = np.squeeze(mask_img.get_fdata()) > 0.5

    def _resample(p):
        img = nli.resample_to_img(nib.load(str(p)), mask_img,
                                  interpolation="nearest")
        return img.get_fdata()[mask]

    w1 = _resample(need["weights_1"])
    w2 = _resample(need["weights_2"])
    r1 = _resample(need["r2_1"]);  r2 = _resample(need["r2_2"])

    basis = _basis()
    th1 = _preferred_deg(w1, basis)                   # session-1 orientation
    th2 = _preferred_deg(w2, basis)                   # session-2 orientation

    # Map session → condition (which session was cdf for this subject).
    cond1 = sub.get_mapping(1)
    th_cdf = np.where(cond1 == "cdf", th1, th2)
    th_invcdf = np.where(cond1 == "cdf", th2, th1)

    if signal == "cvr2_null":
        cvp = _joint_cvr2_path(subject, cv_model, bids_folder)
        nullp = _joint_cvr2_path(subject, null_model, bids_folder)
        if not (cvp.exists() and nullp.exists()):
            return pd.DataFrame()
        cvr2 = _resample(cvp);  null = _resample(nullp)
        keep = (np.isfinite(cvr2) & np.isfinite(null)
                & (cvr2 > null + cvr2_margin) & (cvr2 > cvr2_floor))
    else:                                              # legacy r2 gate
        keep = np.isfinite(r1) & np.isfinite(r2) & (r1 > r2_thr) & (r2 > r2_thr)

    return pd.DataFrame({"subject": subject, "roi": roi,
                         "theta_cdf": th_cdf[keep],
                         "theta_invcdf": th_invcdf[keep]})


def _circ_shift_deg(a, b):
    """Signed circular difference b-a on the 180° orientation circle, in (-90,90]."""
    return (b - a + 90) % 180 - 90


def panel(ax, df, roi, ori, cdf, invcdf, label_axes=True):
    x = df["theta_cdf"].values
    y = df["theta_invcdf"].values
    grid = np.linspace(0, 180, 200)

    # Smooth density (hexbin) — no checkerboard, empty bins stay white.
    hb = ax.hexbin(x, y, gridsize=22, extent=(0, 180, 0, 180),
                   cmap="mako_r", mincnt=1, linewidths=0.0)

    # Reference curves: identity = retinotopy (gray dashed), value-preserving (red).
    ax.plot(grid, grid, color=C_IDENTITY, lw=1.0, ls=(0, (4, 3)), zorder=4)
    ax.plot(grid, value_preserving_curve(grid, ori, cdf, invcdf),
            color=C_VALUEPRES, lw=1.8, zorder=5)

    # Direct labels instead of a legend.
    ax.text(176, 168, "Retinotopy", color=C_IDENTITY, fontsize=7.5,
            ha="right", va="center", rotation=38, rotation_mode="anchor")
    ax.text(150, 96, "Value-\npreserving", color=C_VALUEPRES, fontsize=7.5,
            ha="center", va="center", linespacing=0.95)

    # Which prediction fits better (mean abs circular residual, degrees)?
    res_id = np.abs(_circ_shift_deg(x, y)).mean()
    res_vp = np.abs(_circ_shift_deg(value_preserving_curve(x, ori, cdf, invcdf),
                                    y)).mean()
    winner = "retinotopy" if res_id < res_vp else "value-preserving"
    ax.text(0.04, 0.96,
            f"{roi}\nn = {len(df):,} vox · {df['subject'].nunique()} subj\n"
            f"mean |shift| to identity {res_id:.0f}°\n"
            f"to value-pred. {res_vp:.0f}°\n"
            f"closer to {winner}",
            transform=ax.transAxes, fontsize=7, va="top", ha="left", color="0.15",
            linespacing=1.35)

    ax.set_xlabel("Preferred orientation, cdf (°)")
    if label_axes:
        ax.set_ylabel("Preferred orientation, inverse_cdf (°)")
    ax.set_xlim(0, 180); ax.set_ylim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180]); ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_aspect("equal")
    return hb, {"roi": roi, "n_vox": len(df), "n_subj": df["subject"].nunique(),
                "resid_identity_deg": res_id, "resid_valuepreserving_deg": res_vp}


def run(subjects, out, bids_folder=BIDS_FOLDER, signal="cvr2_null", r2_thr=0.05,
        cvr2_floor=0.0):
    ori, cdf, invcdf = load_mappings()
    rows, summary = [], []
    for roi, hemi in ROIS:
        per = []
        for s in subjects:
            try:
                d = load_subject_roi(s, roi, hemi, bids_folder,
                                     signal=signal, r2_thr=r2_thr,
                                     cvr2_floor=cvr2_floor)
                if not d.empty:
                    per.append(d)
            except Exception as exc:
                print(f"  {roi} sub-{s}: skip ({exc})")
        if per:
            rows.append(pd.concat(per, ignore_index=True))

    if not rows:
        raise SystemExit("No voxels — check vonmises-session-shift fits / signal gate.")

    gate = (f"von Mises cvR² > {cvr2_floor:g}" if signal == "cvr2_null"
            else f"R² > {r2_thr:.2f}")
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        fig, axes = plt.subplots(1, len(rows), figsize=(3.4 * len(rows) + 0.6, 3.5),
                                 constrained_layout=True)
        if len(rows) == 1:
            axes = [axes]
        hb = None
        for i, (ax, df) in enumerate(zip(axes, rows)):
            hb, summ = panel(ax, df, df["roi"].iloc[0], ori, cdf, invcdf,
                             label_axes=(i == 0))
            summary.append(summ)
            ax.text(-0.18, 1.02, "ab"[i], transform=ax.transAxes,
                    fontsize=12, fontweight="bold", va="bottom", ha="right")
        sns.despine(fig=fig, offset=4, trim=True)
        # Slim shared density colorbar.
        cb = fig.colorbar(hb, ax=axes, fraction=0.045, pad=0.02, aspect=30)
        cb.set_label("Voxels", fontsize=8)
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=7, width=0.5, length=2)
        fig.suptitle(f"Preferred-orientation consistency across value mappings  "
                     f"(clear-signal voxels: {gate})", fontsize=9.5, y=1.02)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    big = pd.concat(rows, ignore_index=True)
    big.to_csv(out.with_suffix(".tsv"), sep="\t", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nWrote {out}\n      {out.with_suffix('.tsv')}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+",
                   default=["03", "04", "05", "06", "07", "08", "09", "10",
                            "13", "14"])
    p.add_argument("--signal", default="cvr2_null", choices=["cvr2_null", "r2"],
                   help="Clear-signal voxel gate: 'cvr2_null' (default, "
                        "vonmises cvR² beats the null model per voxel) or "
                        "'r2' (both per-condition fits > --r2-thr).")
    p.add_argument("--r2-thr", type=float, default=0.05,
                   help="R² threshold (only used with --signal r2).")
    p.add_argument("--cvr2-floor", type=float, default=0.0,
                   help="Absolute cvR² floor for the clear-signal gate "
                        "(default 0.0 = genuinely predictive on held-out data; "
                        "raise to e.g. 0.05 for an even stricter set).")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    args = p.parse_args()
    run(args.subjects, Path(args.out), bids_folder=args.bids_folder,
        signal=args.signal, r2_thr=args.r2_thr, cvr2_floor=args.cvr2_floor)


if __name__ == "__main__":
    main()
