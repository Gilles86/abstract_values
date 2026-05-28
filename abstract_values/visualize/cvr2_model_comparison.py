"""Cross-validated R² comparison across the nested-model ladder.

For NPCr (value coding):
  null  →  standard  →  session-shift  →  fwhm-shift  →  fully-shifted

For V1 (orientation coding):
  vonmises (joint, 'fixed tuning')  vs  vonmises-shift (per-session weights,
                                                          'flexible tuning')

Reads ``cvr2`` maps from each model's ``.cv`` output directory (one
per subject × smoothing variant), masks to an ROI, and aggregates
per-voxel cvR² across subjects.

Per page (one (ROI, smoothing) cell):
  - Bar plot of median cvR² across voxels, per model. Per-subject
    points overlaid (thin gray + connecting lines).
  - Side annotation: each model's median cvR², the Δ vs the simpler
    nested model, and whether that Δ is significantly > 0 across
    subjects (paired Wilcoxon).

Usage:
    python -m abstract_values.visualize.cvr2_model_comparison
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
from scipy import stats

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
               / "cvr2_model_comparison.pdf")

# Two ROI-specific model ladders (nested in flexibility, left → right).
NPCR_LADDER = [
    ("null",          "aprf-null.cv",          "null"),
    ("standard",      "aprf.cv",                "standard"),
    ("session-shift", "aprf-shift.cv",          "mode shifts"),
    ("fwhm-shift",    "aprf-fwhm-shift.cv",     "mode + fwhm"),
    ("fully-shifted", "aprf-fully-shifted.cv",  "all 4 shift"),
]
V1_LADDER = [
    # The null model is paradigm-independent (just per-voxel training
    # mean over the same gabor-trial betas), so the aprf-null.cv map
    # is the correct null baseline for V1 too.
    ("null",           "aprf-null.cv",       "null"),
    ("vonmises",       "vonmises.cv",        "fixed tuning"),
    ("vonmises-shift", "vonmises-shift.cv",  "flexible tuning"),
]
PALETTE = ["#9C9C9C", "#3B5BA5", "#5D8C3F", "#E76F51", "#C44E52", "#8172B2"]


def _cvr2_path(model_subdir, subject, smoothed):
    smooth = "_smoothed" if smoothed else ""
    return (DERIV / model_subdir / f"sub-{subject}" / "func"
            / f"sub-{subject}_task-abstractvalue_space-T1w"
              f"_desc-cvr2{smooth}_pe.nii.gz")


def _load_roi_mask(subject, roi, hemi):
    sub = Subject(subject, bids_folder=Path(BIDS_FOLDER))
    return sub.get_roi_mask(roi, hemi=hemi)


def _collect(ladder, subjects, roi, hemi, smoothed, *,
              filter_null_loses=True):
    """For each (subject, model), load cvR² masked to ROI. Returns long
    DataFrame: subject × model × median_cvr2 + n_voxels, plus
    per-voxel arrays indexed by (subject, model).

    With ``filter_null_loses=True`` AND the ladder including ``"null"``,
    voxels where the null model is the per-voxel argmax across the
    ladder are excluded — keeps only 'real-signal voxels' for which
    at least one encoding model beats predicting the training mean.
    The filter is applied per-subject (each subject's voxel selection
    depends on that subject's own argmax across loaded models). If a
    subject is missing the null map, that subject's voxels are kept
    unfiltered with a warning.
    """
    # ── First pass: load all per-voxel maps, aligned to a per-subject ROI mask
    raw = {}                                       # (subject, model) → array
    n_voxels_per_subj = {}                         # subject → n_voxels in ROI
    for model_name, subdir, _ in ladder:
        for s in subjects:
            p = _cvr2_path(subdir, s, smoothed)
            if not p.exists():
                continue
            try:
                mask_img = _load_roi_mask(s, roi, hemi)
            except Exception:
                continue
            mask_arr = np.squeeze(mask_img.get_fdata()) > 0.5
            cv_img = nli.resample_to_img(nib.load(str(p)), mask_img,
                                           interpolation="nearest")
            vals = cv_img.get_fdata()[mask_arr].astype(np.float32)
            raw[(s, model_name)] = vals
            n_voxels_per_subj.setdefault(s, vals.size)

    # ── Per-subject voxel filter: drop voxels where null cvR² is the max
    ladder_models = [m for m, _, _ in ladder]
    keep_masks = {}                                # subject → bool array
    if filter_null_loses and "null" in ladder_models:
        for s in subjects:
            arrs = {m: raw.get((s, m)) for m in ladder_models}
            if any(a is None for a in arrs.values()):
                # Missing null or other model → can't apply argmax filter.
                # Keep all voxels but flag in the figure caption (n_voxels
                # will reflect this).
                continue
            stack = np.column_stack([arrs[m] for m in ladder_models])
            stack = np.where(np.isfinite(stack), stack, -np.inf)
            argmax = np.argmax(stack, axis=1)
            null_idx = ladder_models.index("null")
            keep = argmax != null_idx
            keep_masks[s] = keep

    # ── Second pass: build the long DataFrame using the filter where avail
    rows = []
    per_voxel = {}
    for (s, model_name), vals in raw.items():
        mask = keep_masks.get(s)
        v = vals[mask] if mask is not None else vals
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        rows.append({"subject": s, "model": model_name,
                      "median_cvr2": float(np.median(v)),
                      "mean_cvr2":   float(np.mean(v)),
                      "n_voxels":    int(v.size),
                      "n_voxels_total": int(vals.size),
                      "filtered":   mask is not None})
        per_voxel[(s, model_name)] = v
    return pd.DataFrame(rows), per_voxel


def _paired_test(df, model_a, model_b):
    """Paired Wilcoxon across subjects: returns (n_paired, median_delta, p)."""
    sub_a = df[df.model == model_a].set_index("subject")["median_cvr2"]
    sub_b = df[df.model == model_b].set_index("subject")["median_cvr2"]
    paired = pd.concat({"a": sub_a, "b": sub_b}, axis=1).dropna()
    if len(paired) < 3:
        return len(paired), float("nan"), float("nan")
    delta = paired["b"] - paired["a"]
    try:
        _, p = stats.wilcoxon(delta, alternative="greater")
    except ValueError:                            # all-zero deltas etc.
        p = float("nan")
    return len(paired), float(delta.median()), float(p)


def page(ladder, subjects, roi_label, roi, hemi, smoothed, pdf, *,
          title_prefix="", filter_null_loses=True):
    df, _ = _collect(ladder, subjects, roi, hemi, smoothed,
                       filter_null_loses=filter_null_loses)
    if df.empty:
        return
    model_order = [m for m, _, _ in ladder]

    fig, (ax, ax_frac) = plt.subplots(
        1, 2, figsize=(9.0, 4.5), constrained_layout=True,
        gridspec_kw={"width_ratios": [3, 1]})

    # Per-subject points + connecting lines so paired structure is visible
    for s in df.subject.unique():
        sub_df = df[df.subject == s].set_index("model").reindex(model_order)
        xs = list(range(len(model_order)))
        ax.plot(xs, sub_df["median_cvr2"].values, "-o",
                 color="0.7", lw=0.6, ms=3, alpha=0.6, zorder=1)

    # Group median + IQR diamond per model
    for i, (m, _, _) in enumerate(ladder):
        vals = df[df.model == m]["median_cvr2"].dropna().values
        if vals.size == 0: continue
        med = float(np.median(vals))
        q25, q75 = np.percentile(vals, [25, 75])
        c = PALETTE[i % len(PALETTE)]
        ax.errorbar([i], [med], yerr=[[med - q25], [q75 - med]],
                     fmt="D", ms=11, mec="black", mew=1.4, lw=0,
                     color=c, ecolor="black", elinewidth=0.8,
                     capsize=0, zorder=5)
        # Median value label
        ax.text(i, q75 + 0.003,
                 f"{med:.3f}", ha="center", va="bottom",
                 fontsize=7.5, color="0.2")
    ax.axhline(0, color="0.6", lw=0.5, ls=":", zorder=0)

    # Pairwise nested model deltas — only between adjacent ladder rungs
    annot_lines = []
    for (m_a, _, _), (m_b, _, _) in zip(ladder[:-1], ladder[1:]):
        n, d, p = _paired_test(df, m_a, m_b)
        if not np.isfinite(p):
            annot_lines.append(f"{m_b} − {m_a}: n={n}, p=n/a")
        else:
            sig = " *" if p < 0.05 else ""
            annot_lines.append(
                f"{m_b} − {m_a}: Δ={d:+.4f}  p={p:.1e}{sig}  (n={n})")
    ax.text(1.02, 0.98, "\n".join(annot_lines),
             transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
             color="0.25", family="monospace")

    ax.set_xticks(list(range(len(model_order))))
    ax.set_xticklabels([f"{m}\n({lbl})" for m, _, lbl in ladder],
                        fontsize=8)
    ax.set_ylabel("Per-subject median voxel cv-R²  (signal voxels only)")
    smooth_lbl = "smoothed" if smoothed else "unsmoothed"
    # Voxel-filter status: report the cohort median fraction of ROI
    # voxels kept after the null-loses filter (per subject the kept
    # fraction can vary).
    if filter_null_loses and "filtered" in df.columns and df["filtered"].any():
        keep_frac = (df.groupby("subject")
                       .apply(lambda g: g["n_voxels"].iloc[0]
                                          / max(g["n_voxels_total"].iloc[0], 1)))
        filt_note = (f"  ·  signal voxels (median {100*keep_frac.median():.0f}% "
                      f"of ROI; null-loses filter)")
    else:
        filt_note = "  ·  all ROI voxels (no null filter)"
    ax.set_title(f"{title_prefix}{roi_label}  ·  cvR² across nested models  "
                  f"·  {smooth_lbl}  ·  n_subjects={df.subject.nunique()}"
                  f"{filt_note}",
                  fontsize=10, color="0.15")

    # ── Right panel: per-subject proportion of ROI voxels where any
    # non-null model beats null. This is the "signal voxel" prevalence
    # — the population that's actually informative for the cvR² ladder
    # comparison on the left.
    if filter_null_loses and "filtered" in df.columns and df["filtered"].any():
        # df has duplicate (subject, n_voxels, n_voxels_total) across
        # models; collapse to one row per subject.
        per_sub = (df.drop_duplicates("subject")
                     .set_index("subject")[["n_voxels", "n_voxels_total"]])
        frac = (100 * per_sub["n_voxels"] / per_sub["n_voxels_total"].clip(lower=1))
        frac = frac.sort_values(ascending=False)
        ax_frac.bar(range(len(frac)), frac.values, color="#3B5BA5",
                     alpha=0.7, edgecolor="white", linewidth=0.5)
        ax_frac.set_xticks(range(len(frac)))
        ax_frac.set_xticklabels([f"sub-{s}" for s in frac.index],
                                  rotation=90, fontsize=7)
        ax_frac.axhline(float(frac.median()), color="0.2", lw=0.8, ls="--",
                          zorder=4)
        ax_frac.text(0.97, float(frac.median()) + 1, f"median {frac.median():.0f}%",
                      transform=ax_frac.get_yaxis_transform(),
                      ha="right", va="bottom", fontsize=7.5, color="0.2")
        ax_frac.set_ylabel("% of ROI voxels where any model > null")
        ax_frac.set_title("Signal-voxel prevalence",
                            fontsize=9, color="0.2")
        ax_frac.set_ylim(0, max(100, float(frac.max()) * 1.05))
    else:
        ax_frac.axis("off")
        ax_frac.text(0.5, 0.5, "Null filter not applied",
                       transform=ax_frac.transAxes,
                       ha="center", va="center", color="0.5", fontsize=8)
    sns.despine(fig=fig, offset=4, trim=False)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def discover_subjects():
    """Subjects appearing in any aprf*.cv directory or vonmises*.cv."""
    seen = set()
    for sub_glob in ("aprf*.cv/sub-*", "vonmises*.cv/sub-*"):
        for p in DERIV.glob(sub_glob):
            seen.add(p.name.removeprefix("sub-"))
    return sorted(seen, key=lambda s: (0 if s[0].isdigit() else 1, s))


def run(subjects, out):
    if subjects is None:
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No subjects with CV outputs found.")
    print(f"Subjects: {subjects}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        for smoothed in (False, True):
            print(f"\n=== {'smoothed' if smoothed else 'unsmoothed'} ===")
            print("  NPCr (value coding)…")
            page(NPCR_LADDER, subjects, "NPCr (value)",
                 "NPCr", None, smoothed, pdf, title_prefix="")
            print("  V1 (orientation coding)…")
            page(V1_LADDER, subjects, "V1 (orientation)",
                 "BensonV1", "LR", smoothed, pdf, title_prefix="")
    print(f"\nWrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, Path(args.out))


if __name__ == "__main__":
    main()
