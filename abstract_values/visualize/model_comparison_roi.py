"""Orientation vs value encoding — cross-validated model comparison per ROI.

The clean, confound-free test of whether a region codes ORIENTATION (a
retinotopic/stimulus variable, identical across the two value mappings) or
VALUE (abstract, remapped between mappings). Within a single condition the two
are monotonically linked and indistinguishable; the **joint** cross-validated
fits break the tie because they force the tuning to be *shared across both
conditions* — and the same orientation maps to different values in cdf vs
inverse_cdf, so only the truly stable latent can be shared:

  - Orientation       — ``vonmises.cv``    : single orientation tuning, shared
  - Value (shared)    — ``aprf.cv``        : single value tuning, shared
  - Value (shift)     — ``aprf-shift.cv``  : value tuning reallocated per condition
                                              (the efficient-coding variant)
  - Null              — ``aprf-null.cv``   : predict the training-set mean

All compared on cross-validated R² (cvR²), which is parameter-count-fair.
Prediction: V1 → orientation wins; NPCr → value wins (and value-shift ≥
value-shared there if the code reallocates efficiently).

Two pages:
  1. Mean cvR² per model × ROI (one dot per subject, mean ± SEM), null as a
     reference line.
  2. Winning-model fraction per ROI (per responsive voxel, which model has the
     highest cvR²).

Responsive voxels = where the best non-null model has cvR² > --cvr2-floor
(default 0); the same voxel set scores every model (fair comparison).

Usage:
    python -m abstract_values.visualize.model_comparison_roi
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

# (label, model dir, colour). Order = x-axis order.
# Value (shift) uses the mode+FWHM-shift model: efficient coding reallocates
# tuning *width* across conditions, so a mode-only shift cannot capture it.
MODELS = [
    ("Orientation",     "vonmises.cv",       "#3B5BA5"),   # blue
    ("Value (shared)",  "aprf.cv",           "#C44E52"),   # red
    ("Value (μ+FWHM)",  "aprf-fwhm-shift.cv", "#8C2D33"),  # dark red
]
NULL_MODEL = "aprf-null.cv"
# The single best value model per voxel, for the orientation-vs-value contrast.
VALUE_LABELS = ["Value (shared)", "Value (μ+FWHM)"]
ORI_LABEL = "Orientation"
REAL_LABELS = [m[0] for m in MODELS]
PALETTE = {m[0]: m[2] for m in MODELS}
ROIS = [("BensonV1", "LR", "V1"), ("NPCr", None, "NPCr")]
DEFAULT_OUT = Path(BIDS_FOLDER) / "derivatives" / "qa" / "model_comparison_roi.pdf"


def _cvr2_path(subject, model, bids_folder):
    return (Path(bids_folder) / "derivatives" / "encoding_models" / model
            / f"sub-{subject}" / "func"
            / f"sub-{subject}_task-abstractvalue_space-T1w_desc-cvr2_pe.nii.gz")


def load_subject_roi(subject, roi, hemi, bids_folder):
    """Per-voxel cvR² for every model in one subject × ROI (all finite voxels).
    Returns a wide DataFrame; responsive filtering is applied per-page so the
    signal threshold can be swept."""
    sub = Subject(subject, bids_folder=bids_folder)
    mask_img = sub.get_roi_mask(roi, hemi=hemi)
    mask = np.squeeze(mask_img.get_fdata()) > 0.5

    def _load(model):
        p = _cvr2_path(subject, model, bids_folder)
        if not p.exists():
            return None
        return nli.resample_to_img(nib.load(str(p)), mask_img,
                                   interpolation="nearest").get_fdata()[mask]

    cols = {}
    for label, model, _ in MODELS:
        d = _load(model)
        if d is None:
            return None
        cols[label] = d
    wide = pd.DataFrame(cols)

    # Keep every voxel with finite cvR² across all models; the per-page
    # responsive filter (best real model > floor) is applied downstream so the
    # threshold can be swept without re-loading.
    real = wide[REAL_LABELS].values
    wide = wide[np.isfinite(real).all(axis=1)].reset_index(drop=True)
    if wide.empty:
        return None
    wide["subject"] = subject
    wide["roi"] = roi
    wide["best_real"] = wide[REAL_LABELS].max(axis=1)
    wide["winner"] = wide[REAL_LABELS].idxmax(axis=1)
    return wide


def collect(subjects, bids_folder):
    rows = []
    for roi, hemi, _ in ROIS:
        for s in subjects:
            try:
                w = load_subject_roi(s, roi, hemi, bids_folder)
                if w is not None:
                    rows.append(w)
            except Exception as exc:
                print(f"  {roi} sub-{s}: skip ({exc})")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _responsive(wide, floor):
    return wide[wide["best_real"] > floor]


def roi_label(roi):
    return next(lbl for r, _, lbl in ROIS if r == roi)


def page_cvr2(wide, pdf, floor=0.0):
    """Mean cvR² per model × ROI; one dot per subject, mean ± SEM."""
    wide = _responsive(wide, floor)
    # Per-subject mean cvR² per model per ROI.
    per_sub = (wide.melt(id_vars=["subject", "roi"], value_vars=REAL_LABELS,
                         var_name="Model", value_name="cvr2")
               .groupby(["roi", "subject", "Model"], observed=True)["cvr2"]
               .mean().reset_index())

    rois = [r for r, _, _ in ROIS if r in wide["roi"].unique()]
    fig, axes = plt.subplots(1, len(rois), figsize=(2.6 * len(rois) + 0.4, 3.2),
                             sharey=True, constrained_layout=True)
    if len(rois) == 1:
        axes = [axes]
    for i, (ax, roi) in enumerate(zip(axes, rois)):
        d = per_sub[per_sub["roi"] == roi]
        sns.stripplot(data=d, x="Model", y="cvr2", order=REAL_LABELS, ax=ax,
                      palette=PALETTE, size=3.5, alpha=0.5, jitter=0.12, zorder=1)
        sns.pointplot(data=d, x="Model", y="cvr2", order=REAL_LABELS, ax=ax,
                      color="0.15", errorbar=("se", 1), markers="D",
                      markersize=5, linestyle="none", capsize=0, zorder=3)
        ax.axhline(0, color="0.7", lw=0.6, ls="--", zorder=0)
        ax.set_title(roi_label(roi), fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Cross-validated R²" if i == 0 else "")
        ax.tick_params(axis="x", rotation=20)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    fig.suptitle(f"Encoding-model comparison  (n={wide['subject'].nunique()}, "
                 f"responsive voxels, mean ± SEM)", fontsize=9.5, y=1.04)
    sns.despine(fig=fig, offset=4, trim=True)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_winfraction(wide, pdf, floor=0.0):
    """Fraction of responsive voxels best-fit by each model, per ROI."""
    wide = _responsive(wide, floor)
    frac = (wide.groupby(["roi", "subject"], observed=True)["winner"]
            .value_counts(normalize=True).rename("frac").reset_index())
    rois = [r for r, _, _ in ROIS if r in wide["roi"].unique()]
    fig, ax = plt.subplots(figsize=(1.5 * len(rois) + 1.6, 3.2),
                           constrained_layout=True)
    order_roi = [roi_label(r) for r in rois]
    frac["ROI"] = frac["roi"].map(roi_label)
    sns.barplot(data=frac, x="ROI", y="frac", hue="winner", order=order_roi,
                hue_order=REAL_LABELS, palette=PALETTE, ax=ax,
                errorbar=("se", 1), err_kws={"linewidth": 0.8}, capsize=0)
    ax.set_ylabel("Fraction of voxels best-fit")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, 1.18),
              ncol=3, fontsize=7.5)
    fig.suptitle(f"Winning model per responsive voxel  "
                 f"(n={wide['subject'].nunique()}, mean ± SEM)",
                 fontsize=9.5, y=1.05)
    sns.despine(fig=fig, offset=4, trim=True)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_two_populations(wide, pdf, value_kind="shared"):
    """Per-voxel orientation vs value cvR² — is NPCr one population or two?

    ``value_kind``:
      - ``"shared"`` (default, the clean test): value = ``aprf`` (shared across
        conditions). No flexibility confound — a shared-value model can't
        launder per-condition orientation signal, so a bimodal Δ here is real
        two-population evidence.
      - ``"best"``: value = max over value models (incl. the flexible μ+FWHM
        shift). More sensitive but the flexible model can mimic per-condition
        orientation, so wins are not clean evidence of value coding.

    Top row: hexbin of orientation cvR² (x) vs value cvR² (y) per ROI, identity
    line. Bottom row: histogram of Δ = value − orientation. A bimodal Δ (a
    negative 'orientation' mode and a positive 'value' mode) is the signature
    of two populations.
    """
    wide = _responsive(wide, 0.0).copy()
    if value_kind == "shared":
        wide["value_use"] = wide["Value (shared)"]
        vtag = "shared value (aprf)"
    else:
        wide["value_use"] = wide[VALUE_LABELS].max(axis=1)
        vtag = "best value model"
    wide["delta"] = wide["value_use"] - wide[ORI_LABEL]
    rois = [r for r, _, _ in ROIS if r in wide["roi"].unique()]

    fig, axes = plt.subplots(2, len(rois), figsize=(2.7 * len(rois) + 0.4, 5.2),
                             constrained_layout=True)
    lim = float(np.nanpercentile(
        np.abs(wide[[ORI_LABEL, "value_use"]].values), 99))
    for j, roi in enumerate(rois):
        d = wide[wide["roi"] == roi]
        ax = axes[0, j]
        ax.hexbin(d[ORI_LABEL], d["value_use"], gridsize=30,
                  extent=(-lim, lim, -lim, lim), cmap="mako_r", mincnt=1)
        ax.plot([-lim, lim], [-lim, lim], color="0.45", lw=1.0, ls=(0, (4, 3)))
        ax.axhline(0, color="0.8", lw=0.5); ax.axvline(0, color="0.8", lw=0.5)
        ax.set_title(roi_label(roi), fontsize=10)
        ax.set_xlabel("Orientation cvR²")
        ax.set_ylabel(f"Value cvR²  ({vtag})" if j == 0 else "")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")

        ax = axes[1, j]
        dl = float(np.nanpercentile(np.abs(wide["delta"]), 99))
        ax.hist(d["delta"], bins=np.linspace(-dl, dl, 51), color="0.5",
                edgecolor="white", linewidth=0.2)
        ax.axvline(0, color="0.2", lw=0.8)
        ax.set_xlabel("Δ cvR²  (value − orientation)")
        ax.set_ylabel("Voxels" if j == 0 else "")
        frac_val = float((d["delta"] > 0).mean()) * 100
        ax.text(0.96, 0.95, f"{frac_val:.0f}% value-preferring",
                transform=ax.transAxes, fontsize=7, ha="right", va="top",
                color="0.2")
    fig.suptitle(f"Two populations in NPCr? Orientation vs {vtag}, per voxel",
                 fontsize=9.5, y=1.02)
    sns.despine(fig=fig, offset=4, trim=True)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_winfraction_sweep(wide, pdf, thresholds=None):
    """Win fraction per model as the signal threshold is swept.

    For each cvR² floor, restrict to voxels whose best model exceeds it and
    plot the fraction of those voxels won by each model — per subject, shown as
    mean ± SEM across subjects. Answers: does the orientation/value balance
    shift as you keep only clearer-signal voxels?
    """
    if thresholds is None:
        thresholds = np.round(np.linspace(-0.02, 0.10, 25), 4)
    rois = [r for r, _, _ in ROIS if r in wide["roi"].unique()]

    rows = []
    for roi in rois:
        wr = wide[wide["roi"] == roi]
        for subj, ws in wr.groupby("subject", observed=True):
            best = ws["best_real"].values
            win = ws["winner"].values
            for thr in thresholds:
                sel = best > thr
                n = int(sel.sum())
                if n < 20:                       # too few voxels to trust a fraction
                    continue
                for lbl in REAL_LABELS:
                    rows.append({"roi": roi, "subject": subj, "thr": thr,
                                 "Model": lbl, "n": n,
                                 "frac": float((win[sel] == lbl).mean())})
    sweep = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, len(rois), figsize=(2.9 * len(rois) + 0.4, 3.2),
                             sharey=True, constrained_layout=True)
    if len(rois) == 1:
        axes = [axes]
    for i, (ax, roi) in enumerate(zip(axes, rois)):
        d = sweep[sweep["roi"] == roi]
        for lbl in REAL_LABELS:
            sns.lineplot(data=d[d["Model"] == lbl], x="thr", y="frac",
                         ax=ax, color=PALETTE[lbl], errorbar=("se", 1),
                         label=lbl)
        ax.axvline(0, color="0.7", lw=0.6, ls="--", zorder=0)
        ax.set_title(roi_label(roi), fontsize=10)
        ax.set_xlabel("Signal threshold (cvR² floor)")
        ax.set_ylabel("Win fraction" if i == 0 else "")
        ax.set_ylim(0, 1)
        if i == 0:
            ax.legend(loc="upper left", fontsize=7)
        else:
            ax.get_legend().remove() if ax.get_legend() else None
    fig.suptitle(f"Winning-model fraction vs signal threshold  "
                 f"(n={wide['subject'].nunique()}, mean ± SEM; ≥20 vox/subj)",
                 fontsize=9.5, y=1.04)
    sns.despine(fig=fig, offset=4, trim=True)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def run(subjects, out, bids_folder=BIDS_FOLDER, cvr2_floor=0.0):
    wide = collect(subjects, Path(bids_folder))
    if wide.empty:
        raise SystemExit("No voxels — check cv fits.")
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        page_cvr2(wide, pdf, floor=cvr2_floor)
        page_winfraction(wide, pdf, floor=cvr2_floor)
        page_winfraction_sweep(wide, pdf)
        page_two_populations(wide, pdf, value_kind="shared")   # clean test
        page_two_populations(wide, pdf, value_kind="best")     # flexible (for ref)
    _responsive(wide, cvr2_floor).to_csv(out.with_suffix(".tsv"), sep="\t",
                                         index=False)

    # Console summary (on the responsive set at cvr2_floor).
    resp = _responsive(wide, cvr2_floor)
    summ = (resp.melt(id_vars=["roi"], value_vars=REAL_LABELS,
                      var_name="model", value_name="cvr2")
            .groupby(["roi", "model"], observed=True)["cvr2"].mean().unstack())
    print(summ.to_string())
    print("\nWin fraction per ROI:")
    print(resp.groupby("roi", observed=True)["winner"]
          .value_counts(normalize=True).unstack().to_string())
    print(f"\nWrote {out}\n      {out.with_suffix('.tsv')}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+",
                   default=["03", "04", "05", "06", "07", "08", "09", "10",
                            "13", "14"])
    p.add_argument("--cvr2-floor", type=float, default=0.0,
                   help="Responsive-voxel threshold: best real model cvR² > this "
                        "(default 0.0 = genuinely predictive on held-out data).")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    args = p.parse_args()
    run(args.subjects, Path(args.out), bids_folder=args.bids_folder,
        cvr2_floor=args.cvr2_floor)


if __name__ == "__main__":
    main()
