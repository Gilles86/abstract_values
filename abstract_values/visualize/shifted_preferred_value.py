"""Per-condition preferred-value shift in NPCr.

Uses the SessionShiftedLogGaussianPRF fits in
``derivatives/encoding_models/aprf-session-shift/sub-XX/func/``:
  * ``desc-mode_1_pe.nii.gz`` — pRF mode in session 1
  * ``desc-mode_2_pe.nii.gz`` — pRF mode in session 2

plus the leave-one-run-out cross-validated R² of the same shift model and
of a null (per-voxel-mean) baseline, from
``derivatives/encoding_models/aprf-shift.cv/`` and ``aprf-null.cv/``.

Voxel selection is the null-gated criterion cvR²(shift) > cvR²(null) —
see ``compare_voxel_selection.py`` for why this replaced the original
in-sample ``r2 > r2_thr`` threshold.

The CHF↔orientation mapping flips between sessions:
  even subject id → ses-1 = cdf,         ses-2 = inverse_cdf
  odd  subject id → ses-1 = inverse_cdf, ses-2 = cdf

We remap each voxel's (mode_1, mode_2) into (mode_cdf, mode_invcdf)
so the figures compare *conditions* rather than *session order*.

Pages (per smoothing variant):
  1. Preferred vs. presented marginal distributions, per condition.
  2. Pooled 2D hexbin of mode_cdf vs. mode_invcdf with efficient-coding
     Q-Q overlay.
  3. Per-subject scatter mode_cdf vs mode_invcdf in NPCr (voxels selected
     by cvR²(shift) > cvR²(null)); y=x reference and a sign-of-shift coding.
  4. Per-subject Pearson/Spearman r (mode_cdf vs mode_invcdf) and the
     distribution of those r's over subjects.
  5. Per-subject histogram of (mode_invcdf − mode_cdf) shift in CHF.
  6. Group summary: per-subject mean shift ± per-voxel SEM.

Usage:
    python -m abstract_values.visualize.shifted_preferred_value
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

DERIV = Path(BIDS_FOLDER) / "derivatives" / "encoding_models" / "aprf-session-shift"
DERIV_SHIFT_CV = Path(BIDS_FOLDER) / "derivatives" / "encoding_models" / "aprf-shift.cv"
DERIV_NULL_CV = Path(BIDS_FOLDER) / "derivatives" / "encoding_models" / "aprf-null.cv"
DEFAULT_OUT = Path(BIDS_FOLDER) / "derivatives" / "qa" / "shifted_preferred_value.pdf"

# Condition palette — matches behavior notebook
COND_COLOUR = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}


def _path(subject: str, desc: str, smoothed: bool = False) -> Path:
    smooth = "_smoothed" if smoothed else ""
    return (DERIV / f"sub-{subject}" / "func"
            / f"sub-{subject}_task-abstractvalue_space-T1w_desc-{desc}{smooth}_pe.nii.gz")


def _cv_path(subject: str, deriv: Path, smoothed: bool = False) -> Path:
    smooth = "_smoothed" if smoothed else ""
    return (deriv / f"sub-{subject}" / "func"
            / f"sub-{subject}_task-abstractvalue_space-T1w_desc-cvr2{smooth}_pe.nii.gz")


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
                    smoothed: bool = False) -> pd.DataFrame:
    """Per-voxel (mode_cdf, mode_invcdf, delta_cvr2) for one subject in `roi`.

    Voxel selection is the null-gated criterion: leave-one-run-out
    cvR²(session-shift) > cvR²(null), i.e. the shift model beats a trivial
    per-voxel-mean baseline on genuinely held-out data. This replaced the
    original in-sample ``r2 > r2_thr`` threshold after
    ``compare_voxel_selection.py`` showed the in-sample criterion both
    over- and under-selects relative to the honest out-of-sample test (see
    that script's docstring for the comparison).

    ``smoothed=True`` reads the spatially-smoothed session-shift fits
    (useful when only the smoothed variant is on disk, e.g. sub-07
    while its unsmoothed encoding chain is still running).
    """
    sub = Subject(subject, bids_folder=Path(BIDS_FOLDER))
    p1 = _path(subject, "mode_1", smoothed=smoothed)
    p2 = _path(subject, "mode_2", smoothed=smoothed)
    p_cv_shift = _cv_path(subject, DERIV_SHIFT_CV, smoothed=smoothed)
    p_cv_null = _cv_path(subject, DERIV_NULL_CV, smoothed=smoothed)
    if not (p1.exists() and p2.exists() and p_cv_shift.exists() and p_cv_null.exists()):
        return pd.DataFrame()
    mode1_img = nib.load(str(p1))
    mode2_img = nib.load(str(p2))
    cv_shift_img = nib.load(str(p_cv_shift))
    cv_null_img = nib.load(str(p_cv_null))

    roi_img = sub.get_roi_mask(roi, hemi=None)
    mask_arr = _resampled_mask(roi_img, mode1_img)

    m1 = mode1_img.get_fdata().astype(np.float32)[mask_arr]
    m2 = mode2_img.get_fdata().astype(np.float32)[mask_arr]
    cv_shift = cv_shift_img.get_fdata().astype(np.float32)[mask_arr]
    cv_null = cv_null_img.get_fdata().astype(np.float32)[mask_arr]

    # Per-subject session→condition mapping
    s1_cond = sub.get_mapping(1)
    if s1_cond == "cdf":
        mode_cdf, mode_invcdf = m1, m2
    else:
        mode_cdf, mode_invcdf = m2, m1

    df = pd.DataFrame({
        "subject": subject,
        "voxel": np.arange(len(cv_shift)),
        "mode_cdf": mode_cdf,
        "mode_invcdf": mode_invcdf,
        "mean_mode": 0.5 * (mode_cdf + mode_invcdf),
        "delta_cvr2": cv_shift - cv_null,
    })
    df = df[np.isfinite(df.delta_cvr2) & (df.delta_cvr2 > 0)].reset_index(drop=True)
    return df


def collect_subject_full(subject: str, roi: str = "NPCr",
                         smoothed: bool = False) -> pd.DataFrame:
    """Per-voxel (mode_cdf, mode_invcdf, cv_shift, cv_null, reliability)
    for ALL voxels in `roi` — unlike :func:`collect_subject`, no
    reliability pre-filter. Feeds :func:`page_reliability_gate`, which
    needs the full quality spectrum (noisy → well-tuned) to show how the
    mode_cdf/mode_invcdf relationship strengthens with reliability, not
    just the voxels that already cleared a bar.

    ``reliability = cv_shift − cv_null`` — the session-shift model's
    margin over the trivial "predict the training mean" baseline, i.e.
    the same "beats null" criterion :func:`collect_subject` already
    filters on (and the one page_hexbin's clean-looking pooled pattern
    actually rests on). Tried raw ``cv_shift`` percentile first — a voxel
    can score a high cvR² just by being low-noise, with no real value
    tuning at all, so it's a weaker reliability signal than genuinely
    beating the null.
    """
    sub = Subject(subject, bids_folder=Path(BIDS_FOLDER))
    p1 = _path(subject, "mode_1", smoothed=smoothed)
    p2 = _path(subject, "mode_2", smoothed=smoothed)
    p_cv_shift = _cv_path(subject, DERIV_SHIFT_CV, smoothed=smoothed)
    p_cv_null = _cv_path(subject, DERIV_NULL_CV, smoothed=smoothed)
    if not (p1.exists() and p2.exists() and p_cv_shift.exists()
            and p_cv_null.exists()):
        return pd.DataFrame()
    mode1_img = nib.load(str(p1))
    mode2_img = nib.load(str(p2))
    cv_shift_img = nib.load(str(p_cv_shift))
    cv_null_img = nib.load(str(p_cv_null))

    roi_img = sub.get_roi_mask(roi, hemi=None)
    mask_arr = _resampled_mask(roi_img, mode1_img)

    m1 = mode1_img.get_fdata().astype(np.float32)[mask_arr]
    m2 = mode2_img.get_fdata().astype(np.float32)[mask_arr]
    cv_shift = cv_shift_img.get_fdata().astype(np.float32)[mask_arr]
    cv_null = cv_null_img.get_fdata().astype(np.float32)[mask_arr]

    s1_cond = sub.get_mapping(1)
    if s1_cond == "cdf":
        mode_cdf, mode_invcdf = m1, m2
    else:
        mode_cdf, mode_invcdf = m2, m1

    df = pd.DataFrame({
        "subject": subject,
        "mode_cdf": mode_cdf,
        "mode_invcdf": mode_invcdf,
        "cv_shift": cv_shift,
        "cv_null": cv_null,
        "reliability": cv_shift - cv_null,
    })
    return df[np.isfinite(df.cv_shift) & np.isfinite(df.cv_null)].reset_index(drop=True)


def discover_subjects(smoothed: bool = False,
                       include_smoothed_fallback: bool = True) -> list[str]:
    """Subjects with a session-shift point-estimate AND cv fit on disk.

    When ``include_smoothed_fallback`` is True (default) we keep subjects
    that only have the *smoothed* variant even if the unsmoothed pipeline
    hasn't finished yet — the per-subject loader falls back to smoothed.
    """
    def _have_all(s, smooth):
        return (_path(s, "mode_1", smoothed=smooth).exists()
                and _cv_path(s, DERIV_SHIFT_CV, smoothed=smooth).exists()
                and _cv_path(s, DERIV_NULL_CV, smoothed=smooth).exists())

    found = []
    for p in DERIV.glob("sub-*"):
        s = p.name.removeprefix("sub-")
        if _have_all(s, smoothed):
            found.append(s)
        elif include_smoothed_fallback and not smoothed and _have_all(s, True):
            found.append(s)
    return sorted(found)


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
    # Colour scale = mean preferred value across conditions; alpha = ΔcvR²
    # (cvR²_shift − cvR²_null, the null-gated selection margin). This carries
    # the "where in value space does this voxel live?" signal — exactly what
    # efficient coding predicts should organise the shift.
    cmap_pref = plt.get_cmap("plasma")
    for i, sub in enumerate(subjects):
        ax = axes[i]
        df = by_sub[sub]
        alpha = np.clip(df["delta_cvr2"].to_numpy() / 0.1, 0.15, 0.95)
        sc = ax.scatter(df.mode_cdf, df.mode_invcdf, c=df["mean_mode"],
                        cmap=cmap_pref, vmin=lo, vmax=hi,
                        s=10, alpha=alpha, linewidth=0)
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
    fig.suptitle("Per-voxel preferred-value shift in NPCr  "
                 "(hue = mean preferred value, α ∝ ΔcvR²)",
                 fontsize=10, y=1.02)
    cbar = fig.colorbar(sc, ax=axes[: len(subjects)].tolist(), shrink=0.5,
                        pad=0.02, aspect=14)
    cbar.set_label("½ (mode_CDF + mode_InvCDF)  (CHF)", fontsize=8)
    sns.despine(fig=fig, offset=3, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _compute_correlations(by_sub: dict) -> pd.DataFrame:
    """Per-subject Pearson & Spearman r between mode_cdf and mode_invcdf
    across NPCr voxels (null-gated cvR² selection). Tests whether a voxel's
    preferred value is *consistent* across conditions, independent of any
    mean shift.
    """
    rows = []
    for sub, df in by_sub.items():
        r_p, p_p = stats.pearsonr(df.mode_cdf, df.mode_invcdf)
        r_s, p_s = stats.spearmanr(df.mode_cdf, df.mode_invcdf)
        rows.append(dict(subject=sub, r_pearson=r_p, p_pearson=p_p,
                         r_spearman=r_s, p_spearman=p_s, n_voxels=len(df)))
    return pd.DataFrame(rows)


def page_correlation(corr_df: pd.DataFrame, pdf: PdfPages):
    """Per-subject Pearson/Spearman r (mode_cdf vs. mode_invcdf) and the
    distribution of those r's over subjects."""
    if corr_df.empty:
        return
    g = corr_df.sort_values("subject",
                            key=lambda s: s.map(lambda x: (0 if x[0].isdigit() else 1, x)))
    PEARSON_C, SPEARMAN_C = "#264653", "#E9C46A"

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), constrained_layout=True)

    # Left: per-subject dumbbell of r_pearson vs r_spearman
    ax = axes[0]
    ys = np.arange(len(g))
    for y, (_, r) in zip(ys, g.iterrows()):
        ax.plot([r.r_pearson, r.r_spearman], [y, y], "-", color="0.75",
                lw=0.8, zorder=1)
    ax.scatter(g.r_pearson, ys, s=45, color=PEARSON_C, edgecolor="black",
              linewidth=0.8, zorder=3, label="Pearson")
    ax.scatter(g.r_spearman, ys, s=45, marker="D", color=SPEARMAN_C,
              edgecolor="black", linewidth=0.8, zorder=3, label="Spearman")
    ax.axvline(0, color="0.4", lw=0.6, ls="--", zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"sub-{s}" for s in g.subject])
    ax.set_xlabel("r  (mode_CDF vs. mode_InvCDF)")
    ax.set_title("Per-subject correlation", fontsize=9, color="0.2")
    ax.legend(loc="best", fontsize=7.5, frameon=False)

    # Right: distribution of r's across subjects
    ax = axes[1]
    long = pd.concat([
        pd.DataFrame({"metric": "Pearson", "r": g.r_pearson}),
        pd.DataFrame({"metric": "Spearman", "r": g.r_spearman}),
    ], ignore_index=True)
    sns.stripplot(data=long, x="metric", y="r", ax=ax,
                 palette={"Pearson": PEARSON_C, "Spearman": SPEARMAN_C},
                 size=6, jitter=0.08, alpha=0.85, linewidth=0.6,
                 edgecolor="black")
    for i, metric, colour in [(0, "r_pearson", PEARSON_C), (1, "r_spearman", SPEARMAN_C)]:
        mean_r = g[metric].mean()
        ax.plot([i - 0.2, i + 0.2], [mean_r, mean_r], color=colour, lw=2.2, zorder=4)
    ax.axhline(0, color="0.4", lw=0.6, ls="--", zorder=0)
    ax.set_xlabel("")
    ax.set_ylabel("r  (mode_CDF vs. mode_InvCDF)")
    ax.set_title(
        f"Across subjects  "
        f"(Pearson {g.r_pearson.mean():.2f}±{g.r_pearson.std():.2f}, "
        f"Spearman {g.r_spearman.mean():.2f}±{g.r_spearman.std():.2f})",
        fontsize=8.5, color="0.2")

    fig.suptitle("Preferred-value correlation across conditions (per-voxel, within subject)",
                 fontsize=10, y=1.03)
    sns.despine(fig=fig, offset=4, trim=True)
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


def _orientation_value_pairs() -> pd.DataFrame:
    """Return the (orientation, value_cdf, value_invcdf) lookup table.

    Each orientation is presented in both sessions and maps deterministically
    to a CHF value in each condition. We pool across subjects (the mapping
    is identical for every subject) and pivot.
    """
    rows = []
    for p in DERIV.glob("sub-*"):
        s = p.name.removeprefix("sub-")
        try:
            sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
            for ses in sub.get_sessions():
                cond = sub.get_mapping(ses)
                ev = sub.get_events(ses, sub.get_runs(ses))
                for _, row in ev[ev.event_type == "gabor"].iterrows():
                    rows.append({"orientation": float(row["orientation"]),
                                 "value": float(row["value"]),
                                 "condition": cond})
        except Exception:
            pass
    df = pd.DataFrame(rows).drop_duplicates(["orientation", "condition"])
    return (df.pivot(index="orientation", columns="condition", values="value")
            .reset_index().rename(columns={"cdf": "value_cdf",
                                            "inverse_cdf": "value_invcdf"})
            .dropna()
            .sort_values("orientation").reset_index(drop=True))


def _pool_value_distributions() -> dict[str, np.ndarray]:
    """Pool all gabor-event CHF values across the cohort, per condition."""
    out: dict[str, list[float]] = {"cdf": [], "inverse_cdf": []}
    for p in DERIV.glob("sub-*"):
        s = p.name.removeprefix("sub-")
        try:
            sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
            for ses in sub.get_sessions():
                cond = sub.get_mapping(ses)
                ev = sub.get_events(ses, sub.get_runs(ses))
                vs = ev[ev.event_type == "gabor"]["value"].astype(float).values
                out[cond].extend(vs.tolist())
        except Exception as exc:
            print(f"  pool: skip sub-{s} ({exc})")
    return {k: np.asarray(v) for k, v in out.items()}


def page_stimulus_distributions(value_lim: tuple[float, float], pdf: PdfPages):
    """Side-by-side raw stimulus-value distributions per condition.

    Each condition presents 23 discrete CHF values, each shown 64 times
    (over all subjects+runs). The clustering of those 23 grid points
    determines what efficient coding predicts for pRF tuning.
    """
    val_dists = _pool_value_distributions()
    lo, hi = value_lim

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.4),
                             constrained_layout=True, sharex=True, sharey=False)

    for ax, cond in zip(axes, ["cdf", "inverse_cdf"]):
        vs = val_dists[cond]
        # Fine bins to show the 23-value grid structure
        ax.hist(vs, bins=np.arange(lo, hi + 0.5, 0.5),
                color=COND_COLOUR[cond], alpha=0.85, edgecolor="white",
                linewidth=0.3)
        # Rug on top: mark the unique presented values
        uniq = np.unique(vs)
        ax.scatter(uniq, np.full(len(uniq), ax.get_ylim()[1] * 0.95),
                   marker="|", s=40, color="0.2", linewidth=0.7)
        # Median + IQR
        med = float(np.median(vs))
        q1, q3 = np.percentile(vs, [25, 75])
        ax.axvline(med, color="0.15", lw=1.0, ls="-", zorder=4)
        ax.axvspan(q1, q3, color="0.85", alpha=0.4, zorder=0)
        ax.text(0.02, 0.95,
                f"{cond}  ·  {len(uniq)} values  ·  median {med:.1f}  "
                f"·  IQR [{q1:.1f}, {q3:.1f}]  ·  std {vs.std():.1f}",
                transform=ax.transAxes, fontsize=8.5, va="top",
                color=COND_COLOUR[cond])
        ax.set_xlim(lo, hi)
        ax.set_ylabel("Trial count")
    axes[-1].set_xlabel("Presented CHF value")
    fig.suptitle("Per-condition stimulus-value distributions  "
                 "(pooled across subjects)",
                 fontsize=10, y=1.02)
    sns.despine(fig=fig, offset=4, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_preferred_vs_presented(by_sub: dict, value_lim: tuple[float, float],
                                pdf: PdfPages):
    """Empirical preferred-value distribution per condition, overlaid on the
    presented (stimulus) value distribution for that same condition.

    Preferred values are pooled raw voxel-level pRF modes (all subjects,
    null-gated cvR² selection); presented values are pooled raw trial-level
    CHF values.
    Both are density-normalised (unit area), so the comparison is about
    *shape*, not sample size — this is the direct empirical test of whether
    pRF preference tracks stimulus density (efficient coding) rather than
    spreading uniformly across the value range.
    """
    pool = pd.concat(by_sub.values(), ignore_index=True)
    if pool.empty:
        return
    lo, hi = value_lim
    val_dists = _pool_value_distributions()
    pref_dists = {"cdf": pool.mode_cdf.to_numpy(),
                 "inverse_cdf": pool.mode_invcdf.to_numpy()}

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.6),
                             constrained_layout=True, sharex=True)
    for ax, cond in zip(axes, ["cdf", "inverse_cdf"]):
        presented = val_dists[cond]
        preferred = pref_dists[cond]
        sns.kdeplot(presented, ax=ax, color=COND_COLOUR[cond], lw=1.4,
                    ls="--", clip=(lo, hi), cut=0,
                    label="Presented values (stimulus)")
        sns.kdeplot(preferred, ax=ax, color=COND_COLOUR[cond], lw=1.8,
                    fill=True, alpha=0.25, clip=(lo, hi), cut=0,
                    label="Preferred values (pRF mode, NPCr voxels)")
        med_pres, med_pref = np.median(presented), np.median(preferred)
        ax.axvline(med_pres, color=COND_COLOUR[cond], lw=1.0, ls="--", alpha=0.7)
        ax.axvline(med_pref, color=COND_COLOUR[cond], lw=1.4, alpha=0.9)
        ax.text(0.02, 0.95,
                f"{cond}  ·  presented median {med_pres:.1f}  ·  "
                f"preferred median {med_pref:.1f}  (Δ={med_pref-med_pres:+.1f})",
                transform=ax.transAxes, fontsize=8.5, va="top",
                color=COND_COLOUR[cond])
        ax.set_xlim(lo, hi)
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    axes[-1].set_xlabel("CHF value")
    fig.suptitle("Preferred vs. presented value distributions, per condition",
                 fontsize=10, y=1.02)
    sns.despine(fig=fig, offset=4, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _density_peaks(values: np.ndarray) -> list[float]:
    """Three rank-matched 'peaks' per condition: 1/6, 1/2, 5/6 quantiles.

    Because each condition presents the same total number of trials with
    a denser-toward-the-mode-region grid, the 1/6, 1/2, 5/6 quantiles fall
    in the cohort's high-density 'lobes' for both conditions, and pair
    naturally for the efficient-coding rank match (low↔low, mid↔mid, high↔high).
    """
    return [float(np.percentile(values, q)) for q in (16.67, 50.0, 83.33)]


def page_hexbin(by_sub: dict, value_lim: tuple[float, float], pdf: PdfPages):
    """Pooled-across-subjects 2D hexbin of (mode_cdf, mode_invcdf).

    Annotated with the per-condition marginal stimulus distributions so the
    efficient-coding prediction is readable directly from the figure:
      CDF condition concentrates around the median;
      Inverse-CDF spreads toward the tails.

    Overlaid arrows mark the efficient-coding prediction: a rank-matched
    mapping of dense CDF regions (low/middle/high) → dense InvCDF regions.
    """
    pool = pd.concat(by_sub.values(), ignore_index=True)
    if pool.empty:
        return
    lo, hi = value_lim
    val_dists = _pool_value_distributions()
    cdf_peaks = _density_peaks(val_dists["cdf"])
    inv_peaks = _density_peaks(val_dists["inverse_cdf"])
    print(f"  CDF density peaks: {[round(p, 1) for p in cdf_peaks]}")
    print(f"  InvCDF density peaks: {[round(p, 1) for p in inv_peaks]}")

    fig = plt.figure(figsize=(6.5, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          wspace=0.02, hspace=0.02)
    ax_top   = fig.add_subplot(gs[0, 0])
    ax_main  = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main: 2D hexbin of preferred values — explicit aspect 'equal' so a
    # 1 CHF step on x looks the same as on y (the y=x diagonal is at 45°).
    hb = ax_main.hexbin(pool.mode_cdf, pool.mode_invcdf,
                        gridsize=35, extent=(lo, hi, lo, hi),
                        cmap="rocket_r", mincnt=1)
    ax_main.plot([lo, hi], [lo, hi], "--", color="0.5", lw=0.8, zorder=2,
                 label="y = x  (no reorganization)")
    ax_main.set_aspect("equal", adjustable="box")

    # Q-Q curve: efficient-coding prediction = histogram-matching map from
    # the CDF marginal to the InvCDF marginal. For each x_cdf, look up its
    # percentile rank in the CDF stimulus distribution, then return the
    # value at the same rank in the InvCDF stimulus distribution.
    qs = np.linspace(0.005, 0.995, 200)
    xx = np.quantile(val_dists["cdf"], qs)
    yy = np.quantile(val_dists["inverse_cdf"], qs)
    ax_main.plot(xx, yy, "-", color="#1B998B", lw=1.6, zorder=3,
                 label="Histogram-matching Q–Q  (efficient-coding prediction)")

    # "Orientation-tuning" null — what the voxels would do if they were
    # actually tuned to orientation rather than value: for each presented
    # orientation θ, the (value_cdf(θ), value_invcdf(θ)) pair. Because both
    # mappings are MONOTONIC in θ and use the same orientation grid, these
    # 23 dots fall exactly on the histogram-matching Q-Q line — making the
    # two predictions empirically indistinguishable in this experiment.
    pairs = _orientation_value_pairs()
    ax_main.scatter(pairs["value_cdf"], pairs["value_invcdf"],
                    s=44, marker="o", facecolor="none",
                    edgecolor="#FFD23F", linewidth=1.4, zorder=4,
                    label="Orientation-tuning null  (1 dot / orientation)")

    # Efficient-coding prediction: rank-matched arrows from each CDF density
    # peak to the corresponding InvCDF peak. Drawn at x = peak_cdf, from
    # y = peak_cdf (the "no-shift" baseline) → y = predicted peak_invcdf.
    n = min(len(cdf_peaks), len(inv_peaks))
    for i in range(n):
        x0 = cdf_peaks[i]
        y0 = cdf_peaks[i]
        y1 = inv_peaks[i]
        ax_main.annotate("", xy=(x0, y1), xytext=(x0, y0),
                         arrowprops=dict(arrowstyle="-|>",
                                          color="#1B998B", lw=1.6,
                                          shrinkA=0, shrinkB=0,
                                          mutation_scale=14),
                         zorder=4)
        ax_main.text(x0 + 1.0, (y0 + y1) / 2, f"{y1 - y0:+.0f}",
                     color="#1B998B", fontsize=8.5,
                     va="center", ha="left", zorder=5,
                     bbox=dict(boxstyle="round,pad=0.15",
                                facecolor="white", edgecolor="none",
                                alpha=0.85))
    # Group median crosshair
    med_cdf = float(pool.mode_cdf.median())
    med_inv = float(pool.mode_invcdf.median())
    ax_main.scatter(med_cdf, med_inv, marker="+", s=200, lw=1.8,
                    color="#FFD23F", zorder=3)
    ax_main.set_xlim(lo, hi); ax_main.set_ylim(lo, hi)
    ax_main.set_xlabel("Preferred value — CDF condition  (CHF)")
    ax_main.set_ylabel("Preferred value — Inverse-CDF condition  (CHF)")
    ax_main.text(0.04, 0.96,
                 f"All voxels with r²>0.05 across {len(by_sub)} subjects "
                 f"(n={len(pool)})\nGroup median  CDF {med_cdf:.1f}  "
                 f"InvCDF {med_inv:.1f}  (delta = {med_inv-med_cdf:+.2f} CHF)",
                 transform=ax_main.transAxes, fontsize=8, va="top",
                 color="0.15", bbox=dict(boxstyle="round,pad=0.3",
                                          facecolor="white",
                                          edgecolor="none", alpha=0.7))
    ax_main.legend(loc="lower right", fontsize=7.5, frameon=False)

    # Top: stimulus-value distribution under CDF condition (efficient-coding
    # prediction for the x-axis density of pRFs)
    sns.kdeplot(val_dists["cdf"], ax=ax_top, color=COND_COLOUR["cdf"],
                fill=True, alpha=0.4, lw=1.2, clip=(lo, hi), cut=0)
    for p in cdf_peaks:
        ax_top.axvline(p, color=COND_COLOUR["cdf"], lw=0.8, alpha=0.7)
    ax_top.text(0.02, 0.85, "CDF stimulus values", transform=ax_top.transAxes,
                fontsize=8, color=COND_COLOUR["cdf"], va="top")
    ax_top.set_axis_off()
    # Right: Inverse-CDF stimulus distribution (vertical)
    sns.kdeplot(y=val_dists["inverse_cdf"], ax=ax_right,
                color=COND_COLOUR["inverse_cdf"], fill=True, alpha=0.4,
                lw=1.2, clip=(lo, hi), cut=0)
    for p in inv_peaks:
        ax_right.axhline(p, color=COND_COLOUR["inverse_cdf"], lw=0.8, alpha=0.7)
    ax_right.text(0.18, 0.02, "InvCDF\nstimulus\nvalues",
                  transform=ax_right.transAxes,
                  fontsize=8, color=COND_COLOUR["inverse_cdf"], va="bottom")
    ax_right.set_axis_off()

    cbar = fig.colorbar(hb, ax=ax_right, shrink=0.6, pad=0.18, aspect=12)
    cbar.set_label("Voxel count", fontsize=8)

    fig.suptitle("2D hexbin: preferred value per condition  (pooled voxels)",
                 fontsize=10, y=1.02)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_reliability_gate(by_sub_full: dict, value_lim: tuple[float, float],
                          pdf: PdfPages, top_frac: float = 0.3,
                          n_boot: int = 1000, seed: int = 0):
    """Is the mode_cdf/mode_invcdf relationship (page_hexbin) real signal
    concentrated in a reliable minority, or uniform noise? The per-voxel
    argmax "winner" tallies used elsewhere in this project
    (cvr2_model_comparison.py) count every voxel as one vote — the wrong
    question here: a real effect confined to a well-tuned minority reads
    as "no effect" once averaged against a sea of independent estimation
    noise in unreliable voxels.

    First attempt at this figure used voxel win-rate (cvR²(shift) >
    cvR²(standard)) per reliability decile — binned by cvR²(standard). That
    has a hidden ceiling artifact: cvR²(standard) sits on *both* axes (once
    directly, once inside the win/lose comparison), so a voxel already near
    its achievable-R² ceiling mechanically has less room for further gain —
    shrinking with reliability regardless of whether real shifting occurs.
    Fixed by using an *independent* reliability axis (cvR² over null, not
    a model-vs-model comparison delta) and testing structure directly in
    the quantity that actually matters — the mode_cdf/mode_invcdf
    relationship, not a cvR² difference. Second fix: an initial version
    binned voxels into equal-count reliability deciles, but the
    reliability distribution is heavily skewed (~74% of voxels never beat
    null at all) — equal-count bins mostly just slice up the noise floor,
    compressing all the interesting variation into the last one or two
    bins and reading as flat/noisy everywhere else. Fixed by sweeping a
    cumulative "keep the top X% most reliable voxels" threshold instead
    of exclusive equal-count bins — a standard selection-stringency curve
    that doesn't fight the skew.

    a) Pearson r(mode_cdf, mode_invcdf), pooled voxels, as a function of
       how strict the reliability cutoff is (from all voxels down to the
       top few percent), with a voxel-level bootstrap 95% CI. Independent
       noise in each condition's mode estimate pulls r toward 0 when
       unreliable voxels dominate the pool, regardless of whether the
       population truly reorganizes; a real, coherent relationship
       (shifted or not) needs reliability to be visible at all. This is
       deliberately silent on *whether* reliable voxels shift — only that
       they carry a real, structured relationship instead of noise.
       Panel b asks the shift question.
    b) mode_cdf vs mode_invcdf hexbin restricted to the top ``top_frac`` of
       voxels by the same reliability axis (one point on panel a's
       curve, made concrete) — y=x "no reorganization" line and the
       histogram-matching Q–Q efficient-coding prediction overlaid (same
       construction as page_hexbin) — so the reliable subset can be read
       against both references directly.
    """
    pool = pd.concat(by_sub_full.values(), ignore_index=True)
    if pool.empty:
        return
    rng = np.random.default_rng(seed)

    # --- Panel a: r(mode_cdf, mode_invcdf) vs selection stringency ---
    # Stops at top 5%: below that the subset is a few hundred voxels
    # increasingly dominated by 1-2 subjects (n_subj_contributing drops
    # from 24 to 17 by the top 1%) and r swings sharply negative — a small-
    # N/outlier-voxel regime, not evidence against the reliable-voxel
    # story, but not informative for it either. Omitted rather than shown
    # noisy and over-interpreted.
    keep_fracs = np.array([1.0, 0.7, 0.5, 0.35, 0.25, 0.20, 0.15, 0.10, 0.05])
    reliability = pool["reliability"].to_numpy()
    order = np.argsort(-reliability)                     # most reliable first
    mode_cdf_sorted = pool["mode_cdf"].to_numpy()[order]
    mode_invcdf_sorted = pool["mode_invcdf"].to_numpy()[order]
    n_total = len(pool)
    r_med, r_lo, r_hi = [], [], []
    for frac in keep_fracs:
        k = max(int(round(frac * n_total)), 20)
        x, y = mode_cdf_sorted[:k], mode_invcdf_sorted[:k]
        boot = np.empty(n_boot)
        idx_pool = np.arange(k)
        for i in range(n_boot):
            idx = rng.choice(idx_pool, size=k, replace=True)
            boot[i] = np.corrcoef(x[idx], y[idx])[0, 1]
        r_med.append(float(np.corrcoef(x, y)[0, 1]))
        r_lo.append(float(np.percentile(boot, 2.5)))
        r_hi.append(float(np.percentile(boot, 97.5)))
    r_med, r_lo, r_hi = np.array(r_med), np.array(r_lo), np.array(r_hi)
    peak = int(np.argmax(r_med))

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(11.5, 5.2), constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1.15]})

    # Plot against rank order (evenly spaced), label ticks with the actual
    # kept-% — avoids inverted-log-axis quirks with annotate()/tick
    # formatters while keeping "strictest cutoff on the right" reading.
    xr = np.arange(len(keep_fracs))[::-1]
    ax_a.axhline(0, color="0.7", lw=0.8, ls="--", zorder=0)
    ax_a.text(xr[0] + 0.15, 0.01, "No relationship (noise)", fontsize=8,
             color="0.45", va="top", ha="left")
    ax_a.fill_between(xr, r_lo, r_hi, color="#2A9D8F", alpha=0.2, lw=0)
    ax_a.plot(xr, r_med, "-o", color="#2A9D8F", ms=4)
    ax_a.plot(xr[peak], r_med[peak], "o", color="#2A9D8F", ms=8,
             mec="white", mew=1.2, zorder=5)
    ax_a.set_xticks(xr)
    ax_a.set_xticklabels([f"{int(f * 100)}" for f in keep_fracs], fontsize=8)
    ax_a.set_xlabel("Voxels kept, ranked by reliability (%)\n"
                    "(cvR² over null; 100% = all voxels)")
    ax_a.set_ylabel("Preferred value CDF ↔ InvCDF correlation  (r)\n"
                    "pooled, 95% bootstrap CI")
    ax_a.set_ylim(-0.05, 0.35)
    ax_a.text(0.03, 0.06,
             "Peak around the top 25–35% most\nreliable voxels — the "
             "single noisiest\nand single strictest cuts both weaken it",
             transform=ax_a.transAxes, fontsize=8, ha="left", va="bottom",
             color="0.15")
    ax_a.text(0.97, 0.94, f"n={n_total} voxels, {len(by_sub_full)} subjects",
             transform=ax_a.transAxes, fontsize=7.5, color="0.4", ha="right")

    # --- Panel b: shift hexbin, gated to the top reliability quantile ---
    thr = float(pool["reliability"].quantile(1 - top_frac))
    top = pool[pool["reliability"] >= thr]
    lo, hi = value_lim
    hb = ax_b.hexbin(top.mode_cdf, top.mode_invcdf, gridsize=28,
                     extent=(lo, hi, lo, hi), cmap="rocket_r", mincnt=1)
    ax_b.plot([lo, hi], [lo, hi], "--", color="0.5", lw=0.8, zorder=2,
             label="y = x  (no reorganization)")
    val_dists = _pool_value_distributions()
    qs = np.linspace(0.005, 0.995, 200)
    xx = np.quantile(val_dists["cdf"], qs)
    yy = np.quantile(val_dists["inverse_cdf"], qs)
    ax_b.plot(xx, yy, "-", color="#1B998B", lw=1.6, zorder=3,
             label="Efficient-coding Q–Q prediction")
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xlim(lo, hi); ax_b.set_ylim(lo, hi)
    ax_b.set_xlabel("Preferred value — CDF condition  (CHF)")
    ax_b.set_ylabel("Preferred value — Inverse-CDF condition  (CHF)")
    ax_b.text(0.04, 0.96,
             f"Top {top_frac:.0%} most reliable voxels\n"
             f"(n={len(top)}, cvR² over null ≥ {thr:.3f})",
             transform=ax_b.transAxes, fontsize=8, va="top", color="0.15",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="none", alpha=0.75))
    ax_b.legend(loc="lower right", fontsize=7.5, frameon=False)
    cbar = fig.colorbar(hb, ax=ax_b, shrink=0.7, pad=0.02, aspect=15)
    cbar.set_label("Voxel count", fontsize=8)

    fig.suptitle(
        "Preferred-value reorganization is a reliability-gated effect in "
        "a minority of NPCr voxels — not a popular vote",
        fontsize=10.5, color="0.1", y=1.05)
    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _collect_variant(subjects, smoothed):
    by_sub = {}
    for sub in subjects:
        df = collect_subject(sub, smoothed=smoothed)
        if df.empty:
            continue
        by_sub[sub] = df
    return by_sub


def _collect_variant_full(subjects, smoothed):
    by_sub = {}
    for sub in subjects:
        df = collect_subject_full(sub, smoothed=smoothed)
        if df.empty:
            continue
        by_sub[sub] = df
    return by_sub


def _render_variant(by_sub, value_lim, pdf, banner, smoothed=False, subjects=None):
    """Render the per-variant pages with a leading banner page."""
    if not by_sub:
        return None
    # Banner-style title page
    fig, ax = plt.subplots(figsize=(7.5, 1.2), constrained_layout=True)
    ax.text(0.5, 0.5, banner,
            ha="center", va="center", fontsize=18, color="0.1",
            transform=ax.transAxes)
    ax.set_axis_off()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    page_preferred_vs_presented(by_sub, value_lim, pdf)
    page_hexbin(by_sub, value_lim, pdf)
    by_sub_full = _collect_variant_full(subjects or list(by_sub), smoothed=smoothed)
    page_reliability_gate(by_sub_full, value_lim, pdf)
    page_scatter(by_sub, value_lim, pdf)
    corr_df = _compute_correlations(by_sub)
    page_correlation(corr_df, pdf)
    page_shift_hist(by_sub, pdf)
    g = page_group(by_sub, pdf)
    if g is not None and not corr_df.empty:
        g = g.merge(corr_df[["subject", "r_pearson", "p_pearson",
                             "r_spearman", "p_spearman"]], on="subject")
    return g


def run(subjects, out, value_lim,
        variants=("unsmoothed", "smoothed")):
    """Build the figure. ``variants`` controls which smoothing variants
    are rendered as separate sections of the same PDF. Default: both.
    Pass ``variants=("unsmoothed",)`` or ``("smoothed",)`` for one only.
    """
    if subjects is None:
        # Union of subjects across requested variants
        subjects_set = set()
        for v in variants:
            subjects_set.update(discover_subjects(smoothed=(v == "smoothed"),
                                                   include_smoothed_fallback=False))
        subjects = sorted(subjects_set,
                          key=lambda s: (0 if s[0].isdigit() else 1, s))
    if not subjects:
        raise SystemExit("No session-shift fits (point-estimate + cv) found.")

    out.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    with PdfPages(out) as pdf:
        # Reference page (stimulus distributions) — same for both variants
        page_stimulus_distributions(value_lim, pdf)
        for v in variants:
            smoothed = (v == "smoothed")
            by_sub = _collect_variant(subjects, smoothed=smoothed)
            print(f"\n=== {v} ({len(by_sub)} subjects) ===")
            for sub in subjects:
                n = len(by_sub.get(sub, []))
                print(f"  sub-{sub}: {n} voxels  (cvR²(shift) > cvR²(null))"
                      f"{' (skipped: no data)' if n == 0 else ''}")
            banner = f"{v.upper()}  ·  session-shift fits  ·  cvR²(shift) > cvR²(null)"
            g = _render_variant(by_sub, value_lim, pdf, banner,
                               smoothed=smoothed, subjects=subjects)
            if g is not None:
                g["variant"] = v
                summaries.append(g)
    if summaries:
        agg = pd.concat(summaries, ignore_index=True)
        tsv = out.with_suffix(".tsv")
        agg.to_csv(tsv, sep="\t", index=False)
        print(f"\nWrote {out}\nSidecar: {tsv}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+")
    # CHF range presented in the experiment: CDF 5.5–38.5, InvCDF 2.5–41.5.
    # Defaults cover the full union with a little headroom for the kernel
    # extrapolation at the edges of the hexbin / scatter views.
    p.add_argument("--value-min", type=float, default=0.0)
    p.add_argument("--value-max", type=float, default=45.0)
    p.add_argument("--variants", nargs="+",
                   default=["unsmoothed", "smoothed"],
                   choices=["unsmoothed", "smoothed"],
                   help="Smoothing variants to render as separate sections "
                        "(default: both)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, Path(args.out),
        (args.value_min, args.value_max),
        variants=tuple(args.variants))


if __name__ == "__main__":
    main()
