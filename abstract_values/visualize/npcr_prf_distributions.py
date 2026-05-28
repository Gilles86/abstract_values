"""NPCr aprf-session-shift parameter distributions across the cohort.

What this answers:
  - How dispersed are NPCr PRFs (fwhm distribution)?
  - Where are voxels tuned in CHF space (mode_cdf, mode_invcdf)?
  - Does dispersion (fwhm) scale with preferred value (mode)?
  - What does the amplitude / baseline distribution look like?

Reads per-voxel maps from
``derivatives/encoding_models/aprf-session-shift/sub-XX/func/`` and
the NPCr ROI mask. Voxels filtered by ``--r2-thr`` so the histogram
reflects the population that's actually well-fit. Across-subject
pooling = plain voxel concatenation (voxels aren't corresponded
between brains; the histogram describes 'across the cohort, what
NPCr PRFs look like').

Single-page 2×3 figure:
  (0,0) — mode_cdf      histogram (per condition)
  (0,1) — mode_invcdf   histogram (per condition)
  (0,2) — fwhm          histogram (single — fwhm shared across sessions
          in the SessionShiftedLogGaussianPRF model)
  (1,0) — fwhm vs mode_avg  scatter (test whether dispersion scales
          with preferred value — Weber-like / efficient-coding signature)
  (1,1) — amplitude     histogram
  (1,2) — baseline      histogram

Usage:
    python -m abstract_values.visualize.npcr_prf_distributions
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
    "lines.linewidth": 1.2, "legend.frameon": False, "legend.handlelength": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})
sns.set_context("paper")

DERIV = Path(BIDS_FOLDER) / "derivatives" / "encoding_models" / "aprf-session-shift"
DEFAULT_OUT = (Path(BIDS_FOLDER) / "derivatives" / "qa"
               / "npcr_prf_distributions.pdf")
COND_COLOUR = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}
NEUTRAL = "#3B5BA5"
CHF_LO, CHF_HI = 0.0, 45.0


def _load_subject_params(subject, smoothed, r2_thr, bids_folder=BIDS_FOLDER):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    smooth = "_smoothed" if smoothed else ""
    sdir = (Path(bids_folder) / "derivatives" / "encoding_models"
            / "aprf-session-shift" / f"sub-{subject}" / "func")
    fns = {p: (sdir / f"sub-{subject}_task-abstractvalue_space-T1w"
                       f"_desc-{p}{smooth}_pe.nii.gz")
            for p in ("mode_1", "mode_2", "fwhm", "amplitude",
                      "baseline", "r2")}
    if not all(fn.exists() for fn in fns.values()):
        return pd.DataFrame()
    mask_img = sub.get_roi_mask("NPCr", hemi=None)
    mask_arr = np.squeeze(mask_img.get_fdata()) > 0.5
    arrs = {}
    for p, fn in fns.items():
        arrs[p] = nli.resample_to_img(nib.load(str(fn)), mask_img,
                                        interpolation="nearest"
                                       ).get_fdata()[mask_arr]
    df = pd.DataFrame(arrs)
    df["subject"] = subject
    # Remap session-shift modes into condition-keyed columns
    cond_s1 = sub.get_mapping(1)
    df["mode_cdf"]    = np.where(cond_s1 == "cdf", df["mode_1"], df["mode_2"])
    df["mode_invcdf"] = np.where(cond_s1 == "cdf", df["mode_2"], df["mode_1"])
    df["mode_avg"]    = 0.5 * (df["mode_cdf"] + df["mode_invcdf"])
    df = df[np.isfinite(df["r2"]) & (df["r2"] > r2_thr)]
    df = df[(df["fwhm"] > 0) & (df["mode_cdf"] > 0) & (df["mode_invcdf"] > 0)]
    return df.reset_index(drop=True)


def discover_subjects():
    return sorted({p.name.removeprefix("sub-") for p in DERIV.glob("sub-*")},
                   key=lambda s: (0 if s[0].isdigit() else 1, s))


def page(df, smoothed, r2_thr, pdf):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.5),
                              constrained_layout=True)
    smooth_lbl = "smoothed" if smoothed else "unsmoothed"
    fig.suptitle(
        f"NPCr aprf-session-shift parameters  "
        f"(n_subjects={df['subject'].nunique()}, n_voxels={len(df):,}, "
        f"R² > {r2_thr:.2f}, {smooth_lbl})",
        fontsize=10, y=1.03, color="0.15")

    chf_bins = np.linspace(CHF_LO, CHF_HI, 36)

    # ── (0,0) mode_cdf ────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.hist(df["mode_cdf"], bins=chf_bins, color=COND_COLOUR["cdf"],
            alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(float(df["mode_cdf"].median()), color="0.15", lw=1.0, zorder=4)
    ax.set_xlim(CHF_LO, CHF_HI); ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel("Preferred value (CHF)")
    ax.set_ylabel("Voxels")
    ax.set_title("mode_cdf  (preferred CHF in CDF condition)",
                  fontsize=9, color="0.2")

    # ── (0,1) mode_invcdf ────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.hist(df["mode_invcdf"], bins=chf_bins, color=COND_COLOUR["inverse_cdf"],
            alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(float(df["mode_invcdf"].median()), color="0.15", lw=1.0, zorder=4)
    ax.set_xlim(CHF_LO, CHF_HI); ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel("Preferred value (CHF)")
    ax.set_title("mode_invcdf  (preferred CHF in InvCDF condition)",
                  fontsize=9, color="0.2")

    # ── (0,2) FWHM distribution (shared across conditions) ──────────────
    ax = axes[0, 2]
    fwhm = df["fwhm"].values
    # Long upper tail typical for FWHM — clip the view at the 99th
    # percentile so the bulk is readable but don't drop rows.
    hi = float(np.percentile(fwhm, 99))
    bins = np.linspace(0, hi, 36)
    ax.hist(np.clip(fwhm, 0, hi), bins=bins, color=NEUTRAL,
            alpha=0.85, edgecolor="white", linewidth=0.3)
    med = float(np.median(fwhm))
    q1, q3 = np.percentile(fwhm, [25, 75])
    ax.axvline(med, color="0.15", lw=1.0, zorder=4)
    ax.axvspan(q1, q3, color="0.85", alpha=0.4, zorder=0)
    ax.text(0.97, 0.95,
             f"median {med:.1f} CHF\nIQR [{q1:.1f}, {q3:.1f}]\n"
             f"99th-pct cap shown",
             transform=ax.transAxes, fontsize=8, va="top", ha="right",
             color="0.2")
    ax.set_xlim(0, hi); ax.set_xlabel("FWHM (CHF)  —  dispersion")
    ax.set_title("FWHM  (shared across sessions in SS model)",
                  fontsize=9, color="0.2")

    # ── (1,0) FWHM vs mode scatter — does dispersion scale with mode? ─
    ax = axes[1, 0]
    sub_n = min(8000, len(df))
    samp = df.sample(sub_n, random_state=0)
    ax.scatter(samp["mode_avg"], np.clip(samp["fwhm"], 0, hi),
                s=4, alpha=0.25, color=NEUTRAL, edgecolor="none")
    # Median FWHM per mode bin (10 quantile bins) for visual trend
    bins_x = np.quantile(df["mode_avg"], np.linspace(0, 1, 11))
    df_bin = df.assign(b=pd.cut(df["mode_avg"], bins_x,
                                 include_lowest=True, duplicates="drop"))
    agg = df_bin.groupby("b", observed=True).agg(
        mode=("mode_avg", "median"),
        fwhm_med=("fwhm", "median"),
        fwhm_iqr=("fwhm", lambda x: np.percentile(x, 75)
                                       - np.percentile(x, 25))
    )
    ax.plot(agg["mode"], np.clip(agg["fwhm_med"], 0, hi),
            "o-", color="0.15", lw=1.3, ms=5,
            mec="white", mew=0.8, zorder=5,
            label="Median FWHM per mode bin")
    ax.set_xlim(CHF_LO, CHF_HI); ax.set_ylim(0, hi)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel("Preferred CHF  (avg of mode_cdf, mode_invcdf)")
    ax.set_ylabel("FWHM (CHF)")
    ax.set_title("FWHM vs preferred CHF",
                  fontsize=9, color="0.2")
    ax.legend(loc="upper right", fontsize=7)

    # ── (1,1) amplitude ────────────────────────────────────────────────
    ax = axes[1, 1]
    amp = df["amplitude"].values
    # Amplitude can be very heavy-tailed; show middle 99% range
    lo_a, hi_a = np.percentile(amp, [0.5, 99.5])
    bins = np.linspace(lo_a, hi_a, 36)
    ax.hist(np.clip(amp, lo_a, hi_a), bins=bins, color=NEUTRAL,
            alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(float(np.median(amp)), color="0.15", lw=1.0, zorder=4)
    ax.axvline(0, color="0.6", lw=0.5, ls="--", zorder=0)
    ax.set_xlim(lo_a, hi_a); ax.set_xlabel("Amplitude (a.u.)")
    ax.set_title("Amplitude (PRF peak response)",
                  fontsize=9, color="0.2")

    # ── (1,2) baseline ─────────────────────────────────────────────────
    ax = axes[1, 2]
    base = df["baseline"].values
    lo_b, hi_b = np.percentile(base, [0.5, 99.5])
    bins = np.linspace(lo_b, hi_b, 36)
    ax.hist(np.clip(base, lo_b, hi_b), bins=bins, color=NEUTRAL,
            alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(float(np.median(base)), color="0.15", lw=1.0, zorder=4)
    ax.axvline(0, color="0.6", lw=0.5, ls="--", zorder=0)
    ax.set_xlim(lo_b, hi_b); ax.set_xlabel("Baseline (a.u.)")
    ax.set_title("Baseline (additive offset)",
                  fontsize=9, color="0.2")

    sns.despine(fig=fig, offset=4, trim=False)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def run(subjects, r2_thr, smoothings, out):
    if subjects is None:
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No aprf-session-shift fits found.")
    print(f"Subjects: {subjects}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        for smoothed in smoothings:
            rows = []
            for s in subjects:
                try:
                    d = _load_subject_params(s, smoothed, r2_thr)
                    if not d.empty: rows.append(d)
                    print(f"  sub-{s} {'sm' if smoothed else 'unsm'}: "
                           f"{len(d)} voxels")
                except Exception as exc:
                    print(f"  sub-{s}: skip ({exc})")
            if not rows: continue
            df = pd.concat(rows, ignore_index=True)
            page(df, smoothed, r2_thr, pdf)
            # Also persist the per-voxel TSV for downstream stats
            tsv = out.with_suffix(
                f".{'smoothed' if smoothed else 'unsmoothed'}.tsv")
            df.to_csv(tsv, sep="\t", index=False)
            print(f"  → {tsv.name}: {len(df):,} voxels")
    print(f"\nWrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--r2-thr", type=float, default=0.05,
                    help="R² threshold for voxel inclusion (default 0.05)")
    p.add_argument("--smoothings", nargs="+", type=int, default=[0],
                    help="0=unsmoothed, 1=smoothed (default: 0)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, args.r2_thr,
        [bool(s) for s in args.smoothings],
        Path(args.out))


if __name__ == "__main__":
    main()
