"""Compare two voxel-selection criteria for the session-shift aPRF fits.

The "preferred value" figures (``shifted_preferred_value.py``) select
NPCr voxels by **in-sample R²** from the point-estimate session-shift fit
(``aprf-session-shift/desc-r2 > r2_thr``). That criterion is optimistic —
it's the fit's own training R², not held out.

This script builds the alternative, and puts both side by side:

  Criterion A (current):  r2_insample > r2_thr           (aprf-session-shift)
  Criterion B (stricter):  cvR2_shift > cvR2_null          (aprf-shift.cv vs.
                                                             aprf-null.cv,
                                                             leave-one-run-out)

Pages:
  1. Voxel agreement per subject (both / A-only / B-only / neither).
  2. Pooled r2_insample vs. delta_cvr2 scatter — where do the criteria
     actually disagree.
  3. Preferred-value marginal distributions per condition, per criterion.
  4. Density-difference curve (preferred − presented), criterion B,
     pooled across subjects — signed, single-curve version of #3.
  5. Q-Q plots (criterion B) of preferred- vs. presented-value quantiles,
     per subject, one page per condition — colour = local presented
     density, so density-matching vs. density-avoidance is directly
     readable per subject rather than only in the pooled marginal.
  6. Per-subject Pearson/Spearman r (mode_cdf vs mode_invcdf), criterion A
     vs criterion B, paired.
  7. Per-subject mean preferred value & median shift, criterion A vs B.

Usage:
    python -m abstract_values.visualize.compare_voxel_selection
    python -m abstract_values.visualize.compare_voxel_selection --r2-thr 0.05
    python -m abstract_values.visualize.compare_voxel_selection --subjects 04 06
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from abstract_values.utils.data import BIDS_FOLDER, Subject
from abstract_values.visualize.shifted_preferred_value import (
    COND_COLOUR, _density_peaks, _pool_value_distributions, _resampled_mask,
)

DERIV_ROOT = Path(BIDS_FOLDER) / "derivatives" / "encoding_models"
DERIV_SHIFT = DERIV_ROOT / "aprf-session-shift"
DERIV_SHIFT_CV = DERIV_ROOT / "aprf-shift.cv"
DERIV_NULL_CV = DERIV_ROOT / "aprf-null.cv"
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "notes" / "figures" / "compare_voxel_selection.pdf"

CRIT_COLOUR = {"A": "#6C757D", "B": "#8E44AD"}
CRIT_LABEL = {"A": "A: in-sample R² > thr", "B": "B: cvR²(shift) > cvR²(null)"}


def _path(base: Path, subject: str, desc: str, smoothed: bool = False) -> Path:
    smooth = "_smoothed" if smoothed else ""
    return (base / f"sub-{subject}" / "func"
            / f"sub-{subject}_task-abstractvalue_space-T1w_desc-{desc}{smooth}_pe.nii.gz")


def discover_subjects(smoothed: bool = False) -> list[str]:
    found = []
    for p in DERIV_SHIFT.glob("sub-*"):
        s = p.name.removeprefix("sub-")
        paths = [
            _path(DERIV_SHIFT, s, "mode_1", smoothed),
            _path(DERIV_SHIFT, s, "mode_2", smoothed),
            _path(DERIV_SHIFT, s, "r2", smoothed),
            _path(DERIV_SHIFT_CV, s, "cvr2", smoothed),
            _path(DERIV_NULL_CV, s, "cvr2", smoothed),
        ]
        if all(p.exists() for p in paths):
            found.append(s)
    return sorted(found, key=lambda x: (0 if x[0].isdigit() else 1, x))


def collect_subject(subject: str, roi: str = "NPCr", smoothed: bool = False) -> pd.DataFrame:
    sub = Subject(subject, bids_folder=Path(BIDS_FOLDER))
    p_mode1 = _path(DERIV_SHIFT, subject, "mode_1", smoothed)
    p_mode2 = _path(DERIV_SHIFT, subject, "mode_2", smoothed)
    p_r2 = _path(DERIV_SHIFT, subject, "r2", smoothed)
    p_cv_shift = _path(DERIV_SHIFT_CV, subject, "cvr2", smoothed)
    p_cv_null = _path(DERIV_NULL_CV, subject, "cvr2", smoothed)
    if not all(p.exists() for p in [p_mode1, p_mode2, p_r2, p_cv_shift, p_cv_null]):
        return pd.DataFrame()

    mode1_img = nib.load(str(p_mode1))
    mode2_img = nib.load(str(p_mode2))
    r2_img = nib.load(str(p_r2))
    cv_shift_img = nib.load(str(p_cv_shift))
    cv_null_img = nib.load(str(p_cv_null))

    roi_img = sub.get_roi_mask(roi, hemi=None)
    mask_arr = _resampled_mask(roi_img, mode1_img)

    m1 = mode1_img.get_fdata().astype(np.float32)[mask_arr]
    m2 = mode2_img.get_fdata().astype(np.float32)[mask_arr]
    r2 = r2_img.get_fdata().astype(np.float32)[mask_arr]
    cv_shift = cv_shift_img.get_fdata().astype(np.float32)[mask_arr]
    cv_null = cv_null_img.get_fdata().astype(np.float32)[mask_arr]

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
        "r2_insample": r2,
        "cvr2_shift": cv_shift,
        "cvr2_null": cv_null,
    })
    df["delta_cvr2"] = df.cvr2_shift - df.cvr2_null
    df = df[np.isfinite(df.r2_insample) & np.isfinite(df.delta_cvr2)].reset_index(drop=True)
    return df


def _collect_all(subjects, r2_thr, smoothed):
    by_sub = {}
    for sub in subjects:
        df = collect_subject(sub, smoothed=smoothed)
        if df.empty:
            continue
        df["sel_A"] = df.r2_insample > r2_thr
        df["sel_B"] = df.delta_cvr2 > 0
        by_sub[sub] = df
    return by_sub


def page_agreement(by_sub: dict, r2_thr: float, pdf: PdfPages):
    subjects = sorted(by_sub.keys(), key=lambda s: (0 if s[0].isdigit() else 1, s))
    rows = []
    for sub in subjects:
        df = by_sub[sub]
        a, b = df.sel_A, df.sel_B
        rows.append(dict(subject=sub,
                         both=int((a & b).sum()), a_only=int((a & ~b).sum()),
                         b_only=int((~a & b).sum()), neither=int((~a & ~b).sum()),
                         n_total=len(df)))
    g = pd.DataFrame(rows)
    g["pct_both"] = g.both / g.n_total
    g["pct_a_only"] = g.a_only / g.n_total
    g["pct_b_only"] = g.b_only / g.n_total
    g["pct_neither"] = g.neither / g.n_total

    pooled_a = pd.concat(by_sub.values()).sel_A
    pooled_b = pd.concat(by_sub.values()).sel_B
    jaccard = float((pooled_a & pooled_b).sum() / (pooled_a | pooled_b).sum())

    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    ys = np.arange(len(g))
    left = np.zeros(len(g))
    for key, colour, label in [
        ("pct_both", "#2A9D8F", "Both A & B"),
        ("pct_a_only", "#E76F51", "A only"),
        ("pct_b_only", "#8E44AD", "B only"),
        ("pct_neither", "0.85", "Neither"),
    ]:
        ax.barh(ys, g[key], left=left, color=colour, label=label, height=0.7)
        left += g[key].to_numpy()
    ax.set_yticks(ys)
    ax.set_yticklabels([f"sub-{s}  (n={n})" for s, n in zip(g.subject, g.n_total)], fontsize=7.5)
    ax.set_xlabel("Fraction of NPCr voxels")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4,
             fontsize=8, frameon=False)
    ax.set_title(f"Voxel-selection agreement  ·  pooled Jaccard(A,B) = {jaccard:.2f}  "
                f"·  A: r²_insample>{r2_thr}  ·  B: cvR²(shift)>cvR²(null)",
                fontsize=9, color="0.2", y=1.16)
    sns.despine(fig=fig, offset=4, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return g


def page_scatter_criteria(by_sub: dict, r2_thr: float, pdf: PdfPages):
    pool = pd.concat(by_sub.values(), ignore_index=True)
    quad = np.where(pool.sel_A & pool.sel_B, "both",
            np.where(pool.sel_A & ~pool.sel_B, "A only",
            np.where(~pool.sel_A & pool.sel_B, "B only", "neither")))
    pool = pool.assign(quadrant=quad)
    palette = {"both": "#2A9D8F", "A only": "#E76F51", "B only": "#8E44AD", "neither": "0.8"}

    fig, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    for q in ["neither", "A only", "B only", "both"]:
        d = pool[pool.quadrant == q]
        ax.scatter(d.r2_insample, d.delta_cvr2, s=6, alpha=0.5,
                  color=palette[q], linewidth=0,
                  label=f"{q}  (n={len(d)})")
    ax.axvline(r2_thr, color="0.3", lw=0.8, ls="--", zorder=0)
    ax.axhline(0, color="0.3", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("r²  (in-sample, session-shift fit)")
    ax.set_ylabel("Δ cvR²  =  cvR²(shift) − cvR²(null)")
    ax.set_title("Where the two criteria disagree  (pooled voxels, all subjects)",
                fontsize=9.5, color="0.2")
    ax.legend(loc="lower right", fontsize=7.5, frameon=False, markerscale=2.2)
    sns.despine(fig=fig, offset=4, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_marginals_by_criterion(by_sub: dict, value_lim: tuple[float, float], pdf: PdfPages):
    pool = pd.concat(by_sub.values(), ignore_index=True)
    lo, hi = value_lim
    val_dists = _pool_value_distributions()

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.8), constrained_layout=True, sharex=True)
    for ax, cond, col in zip(axes, ["cdf", "inverse_cdf"], ["mode_cdf", "mode_invcdf"]):
        sns.kdeplot(val_dists[cond], ax=ax, color="0.4", lw=1.2, ls=":",
                   clip=(lo, hi), cut=0, label="Presented values")
        for crit, sel_col in [("A", "sel_A"), ("B", "sel_B")]:
            vals = pool.loc[pool[sel_col], col].to_numpy()
            if len(vals) < 2:
                continue
            ls = "--" if crit == "A" else "-"
            sns.kdeplot(vals, ax=ax, color=CRIT_COLOUR[crit], lw=1.8, ls=ls,
                       clip=(lo, hi), cut=0,
                       label=f"{CRIT_LABEL[crit]}  (n={len(vals)})")
        ax.set_xlim(lo, hi)
        ax.set_ylabel("Density")
        ax.set_title(cond, fontsize=8.5, color=COND_COLOUR[cond], loc="left")
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    axes[-1].set_xlabel("CHF value")
    fig.suptitle("Preferred-value distributions: criterion A vs. B", fontsize=10, y=1.02)
    sns.despine(fig=fig, offset=4, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_density_difference(by_sub: dict, value_lim: tuple[float, float],
                            pdf: PdfPages, sel_col: str = "sel_B"):
    """Signed difference curve: density(preferred) − density(presented),
    per condition, pooled across subjects (criterion B). Positive regions
    = preferred values over-represent that part of the value range
    relative to the stimulus set; negative = under-represent. Direct
    single-curve version of the two-curve overlay in
    ``page_marginals_by_criterion`` / ``shifted_preferred_value.py``'s
    ``page_preferred_vs_presented`` — zero-crossings mark exactly where
    over- flips to under-representation.
    """
    pool = pd.concat(by_sub.values(), ignore_index=True)
    lo, hi = value_lim
    val_dists = _pool_value_distributions()
    xx = np.linspace(lo, hi, 400)

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.6), constrained_layout=True, sharex=True)
    for ax, cond, col in zip(axes, ["cdf", "inverse_cdf"], ["mode_cdf", "mode_invcdf"]):
        presented = val_dists[cond]
        preferred = pool.loc[pool[sel_col], col].to_numpy()
        d_presented = stats.gaussian_kde(presented)(xx)
        d_preferred = stats.gaussian_kde(preferred)(xx)
        diff = d_preferred - d_presented

        ax.axhline(0, color="0.4", lw=0.7, ls="--", zorder=0)
        ax.fill_between(xx, diff, 0, where=(diff >= 0), color="#2A9D8F",
                        alpha=0.55, lw=0, interpolate=True,
                        label="Preferred over-represents")
        ax.fill_between(xx, diff, 0, where=(diff < 0), color="#E76F51",
                        alpha=0.55, lw=0, interpolate=True,
                        label="Preferred under-represents")
        ax.plot(xx, diff, color="0.15", lw=1.0)
        # Mark presented-density peaks/troughs for reference
        for p in _density_peaks(presented):
            ax.axvline(p, color="0.5", lw=0.6, ls=":", zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylabel("Density(preferred) − Density(presented)")
        ax.set_title(cond, fontsize=8.5, color=COND_COLOUR[cond], loc="left")
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    axes[-1].set_xlabel("CHF value")
    fig.suptitle("Preferred- minus presented-value density  (criterion B, "
                "pooled voxels)\ndotted lines = presented-value density "
                "peaks (16.7/50/83.3 pct)", fontsize=9.5, y=1.03)
    sns.despine(fig=fig, offset=4, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_qq_per_subject(by_sub: dict, cond: str, mode_col: str,
                        value_lim: tuple[float, float], pdf: PdfPages,
                        sel_col: str = "sel_B", min_voxels: int = 5):
    """Per-subject Q-Q plot: preferred-value quantiles (criterion-B voxels)
    vs. the pooled presented-value quantiles, for one condition.

    Point colour = local presented-value density at that quantile (KDE).
    If preferred values simply tracked presented density (efficient
    coding / density matching), the curve should hug y=x regardless of
    colour. Systematic bowing away from y=x that co-varies with colour
    (e.g. consistently below the line at the brightest/densest points)
    would indicate preferred values under- or over-representing exactly
    the densest stimulus regions — the direction distinguishes "prefers
    dense regions" from "prefers sparse regions".
    """
    lo, hi = value_lim
    val_dists = _pool_value_distributions()
    presented = val_dists[cond]
    kde = stats.gaussian_kde(presented)
    qs = np.linspace(0.02, 0.98, 49)
    x_q = np.quantile(presented, qs)
    dens = kde(x_q)
    dens_norm = (dens - dens.min()) / (dens.max() - dens.min() + 1e-12)

    subjects = sorted(by_sub.keys(), key=lambda s: (0 if s[0].isdigit() else 1, s))
    n = len(subjects)
    cols = 3
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.5, 2.3 * rows + 0.4),
                             constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()
    cmap = plt.get_cmap("viridis")
    sc = None
    for i, sub in enumerate(subjects):
        ax = axes[i]
        d = by_sub[sub]
        vals = d.loc[d[sel_col], mode_col].to_numpy()
        if len(vals) < min_voxels:
            ax.set_axis_off()
            continue
        y_q = np.quantile(vals, qs)
        ax.plot([lo, hi], [lo, hi], "--", color="0.6", lw=0.8, zorder=0)
        ax.plot(x_q, y_q, "-", color="0.3", lw=0.6, alpha=0.5, zorder=1)
        sc = ax.scatter(x_q, y_q, c=dens_norm, cmap=cmap, s=12,
                        linewidth=0, zorder=2)
        ax.text(0.04, 0.96, f"sub-{sub}\nn={len(vals)}",
                transform=ax.transAxes, fontsize=7.5, va="top", color="0.2")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    for j in range(len(subjects), len(axes)):
        axes[j].set_axis_off()
    fig.supxlabel(f"Presented-value quantile — {cond}  (CHF)", fontsize=9)
    fig.supylabel("Preferred-value quantile  (CHF)", fontsize=9)
    fig.suptitle(f"Q-Q: preferred vs. presented value per subject — {cond}  "
                f"(criterion B)\ncolour = local presented density at that "
                f"quantile", fontsize=9.5, y=1.03)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes[:n].tolist(), shrink=0.5,
                            pad=0.02, aspect=14)
        cbar.set_label("Presented density (relative)", fontsize=8)
    sns.despine(fig=fig, offset=3, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _corr_by_criterion(by_sub: dict, min_voxels: int = 5) -> pd.DataFrame:
    rows = []
    for sub, df in by_sub.items():
        row = dict(subject=sub, n_A=int(df.sel_A.sum()), n_B=int(df.sel_B.sum()))
        for crit, sel_col in [("A", "sel_A"), ("B", "sel_B")]:
            d = df[df[sel_col]]
            if len(d) >= min_voxels:
                r_p, _ = stats.pearsonr(d.mode_cdf, d.mode_invcdf)
                r_s, _ = stats.spearmanr(d.mode_cdf, d.mode_invcdf)
                med_shift = float((d.mode_invcdf - d.mode_cdf).median())
                mean_cdf, mean_inv = float(d.mode_cdf.mean()), float(d.mode_invcdf.mean())
            else:
                r_p = r_s = med_shift = mean_cdf = mean_inv = np.nan
            row[f"r_pearson_{crit}"] = r_p
            row[f"r_spearman_{crit}"] = r_s
            row[f"median_shift_{crit}"] = med_shift
            row[f"mean_cdf_{crit}"] = mean_cdf
            row[f"mean_invcdf_{crit}"] = mean_inv
        rows.append(row)
    return pd.DataFrame(rows)


def _dumbbell(ax, g, col_a, col_b, ylabels):
    d = g.dropna(subset=[col_a, col_b])
    ys = np.arange(len(d))
    for y, (_, r) in zip(ys, d.iterrows()):
        ax.plot([r[col_a], r[col_b]], [y, y], "-", color="0.75", lw=0.8, zorder=1)
    ax.scatter(d[col_a], ys, s=42, color=CRIT_COLOUR["A"], edgecolor="black",
              linewidth=0.7, zorder=3, label="A")
    ax.scatter(d[col_b], ys, s=42, marker="D", color=CRIT_COLOUR["B"],
              edgecolor="black", linewidth=0.7, zorder=3, label="B")
    ax.axvline(0, color="0.4", lw=0.6, ls="--", zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"sub-{s}" for s in d.subject], fontsize=7)
    return d


def page_correlation_comparison(corr_df: pd.DataFrame, pdf: PdfPages):
    if corr_df.empty:
        return
    g = corr_df.sort_values("subject",
                            key=lambda s: s.map(lambda x: (0 if x[0].isdigit() else 1, x)))
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.4), constrained_layout=True)

    d = _dumbbell(axes[0], g, "r_pearson_A", "r_pearson_B", g.subject)
    axes[0].set_xlabel("r  (mode_CDF vs. mode_InvCDF)")
    axes[0].set_title(f"Pearson  (mean A={d.r_pearson_A.mean():.2f}, "
                      f"B={d.r_pearson_B.mean():.2f})", fontsize=8.5, color="0.2")
    axes[0].legend(loc="best", fontsize=7.5, frameon=False)

    d2 = _dumbbell(axes[1], g, "r_spearman_A", "r_spearman_B", g.subject)
    axes[1].set_xlabel("r  (mode_CDF vs. mode_InvCDF)")
    axes[1].set_title(f"Spearman  (mean A={d2.r_spearman_A.mean():.2f}, "
                      f"B={d2.r_spearman_B.mean():.2f})", fontsize=8.5, color="0.2")

    fig.suptitle("Per-subject preferred-value correlation: criterion A vs. B",
                fontsize=10, y=1.03)
    sns.despine(fig=fig, offset=4, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_shift_comparison(corr_df: pd.DataFrame, pdf: PdfPages):
    if corr_df.empty:
        return
    g = corr_df.sort_values("subject",
                            key=lambda s: s.map(lambda x: (0 if x[0].isdigit() else 1, x)))
    fig, ax = plt.subplots(figsize=(5.0, 4.6), constrained_layout=True)
    d = _dumbbell(ax, g, "median_shift_A", "median_shift_B", g.subject)
    ax.set_xlabel("Median shift  Inverse-CDF − CDF  (CHF)")
    ax.set_title(f"Preferred-value shift: criterion A vs. B  "
                f"(grand median A={d.median_shift_A.median():+.2f}, "
                f"B={d.median_shift_B.median():+.2f})",
                fontsize=8.5, color="0.2")
    ax.legend(loc="best", fontsize=7.5, frameon=False)
    sns.despine(fig=fig, offset=4, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_variant(by_sub, r2_thr, value_lim, pdf, banner):
    if not by_sub:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 1.2), constrained_layout=True)
    ax.text(0.5, 0.5, banner, ha="center", va="center", fontsize=18,
           color="0.1", transform=ax.transAxes)
    ax.set_axis_off()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    page_agreement(by_sub, r2_thr, pdf)
    page_scatter_criteria(by_sub, r2_thr, pdf)
    page_marginals_by_criterion(by_sub, value_lim, pdf)
    page_density_difference(by_sub, value_lim, pdf)
    page_qq_per_subject(by_sub, "cdf", "mode_cdf", value_lim, pdf)
    page_qq_per_subject(by_sub, "inverse_cdf", "mode_invcdf", value_lim, pdf)
    corr_df = _corr_by_criterion(by_sub)
    page_correlation_comparison(corr_df, pdf)
    page_shift_comparison(corr_df, pdf)
    return corr_df


def run(subjects, r2_thr, out, value_lim, variants=("unsmoothed", "smoothed")):
    if subjects is None:
        subjects_set = set()
        for v in variants:
            subjects_set.update(discover_subjects(smoothed=(v == "smoothed")))
        subjects = sorted(subjects_set, key=lambda s: (0 if s[0].isdigit() else 1, s))
    if not subjects:
        raise SystemExit("No subjects with both session-shift point-estimate and cv fits found.")

    out.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    with PdfPages(out) as pdf:
        for v in variants:
            smoothed = (v == "smoothed")
            by_sub = _collect_all(subjects, r2_thr, smoothed)
            print(f"\n=== {v} ({len(by_sub)} subjects) ===")
            for sub in subjects:
                n = len(by_sub.get(sub, []))
                if n == 0:
                    print(f"  sub-{sub}: skipped (missing fit)")
                    continue
                df = by_sub[sub]
                print(f"  sub-{sub}: n={n}  A={int(df.sel_A.sum())}  "
                      f"B={int(df.sel_B.sum())}  both={int((df.sel_A & df.sel_B).sum())}")
            banner = f"{v.upper()}  ·  criterion A vs. B  ·  r²_thr={r2_thr}"
            corr_df = _render_variant(by_sub, r2_thr, value_lim, pdf, banner)
            if corr_df is not None and not corr_df.empty:
                corr_df["variant"] = v
                summaries.append(corr_df)
    if summaries:
        agg = pd.concat(summaries, ignore_index=True)
        tsv = out.with_suffix(".tsv")
        agg.to_csv(tsv, sep="\t", index=False)
        print(f"\nWrote {out}\nSidecar: {tsv}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--r2-thr", type=float, default=0.05)
    p.add_argument("--value-min", type=float, default=0.0)
    p.add_argument("--value-max", type=float, default=45.0)
    p.add_argument("--variants", nargs="+", default=["unsmoothed", "smoothed"],
                  choices=["unsmoothed", "smoothed"])
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, args.r2_thr, Path(args.out),
       (args.value_min, args.value_max), variants=tuple(args.variants))


if __name__ == "__main__":
    main()
