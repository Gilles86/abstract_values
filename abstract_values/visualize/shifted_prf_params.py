#!/usr/bin/env python3
"""
CDF vs inverse-CDF scatter of NPCr value-pRF parameters from the
**fwhm-shift** model, which lets BOTH the preferred value (mode) and the
tuning width (fwhm) differ per condition. (session-shift only shifts the
mode with a shared fwhm, so it can't show a width change — this is the
fwhm-shift counterpart of `shifted_preferred_value.py`.)

For each NPCr voxel passing R² > thr, remaps (param_1, param_2) [per
session] into (param_cdf, param_invcdf) [per condition] using the
per-subject mapping flip, pools voxels across subjects, and plots:

  Page 1: scatter mode_cdf vs mode_invcdf  AND  fwhm_cdf vs fwhm_invcdf,
          each with y=x; on-axis = abstract/invariant code, systematic
          off-axis = mapping-dependent. Annotated with Pearson r and the
          median (invcdf − cdf) shift.
  Page 2: per-subject median shift (mode & fwhm), Inverse-CDF − CDF.

Reads ``derivatives/encoding_models/aprf-fwhm-shift/sub-XX/func/`` —
needs the fwhm-shift FULL fit (fit_aprf.py --model fwhm-shift), not just
its .cv cvR².

Usage:
  python -m abstract_values.visualize.shifted_prf_params
  python -m abstract_values.visualize.shifted_prf_params --r2-thr 0.05
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

from abstract_values.utils.data import Subject, BIDS_FOLDER

mpl.rcParams.update({"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.dpi": 300})
MODEL_DIR = "aprf-fwhm-shift"
DEFAULT_OUT = Path(BIDS_FOLDER) / "derivatives" / "qa" / "shifted_prf_params_fwhmshift.pdf"


def load_subject(subject, r2_thr, smoothed, bids_folder):
    sub = Subject(subject, bids_folder=bids_folder)
    smooth = "_smoothed" if smoothed else ""
    d = (Path(bids_folder) / "derivatives" / "encoding_models" / MODEL_DIR
         / f"sub-{subject}" / "func")

    def p(desc):
        return d / f"sub-{subject}_task-abstractvalue_space-T1w_desc-{desc}{smooth}_pe.nii.gz"

    needed = ["mode_1", "mode_2", "fwhm_1", "fwhm_2", "r2"]
    if not all(p(x).exists() for x in needed):
        return None
    mask_img = sub.get_roi_mask("NPCr", hemi=None)
    mask_arr = np.squeeze(mask_img.get_fdata()) > 0.5

    def load(desc):
        return nli.resample_to_img(nib.load(str(p(desc))), mask_img,
                                   interpolation="nearest").get_fdata()[mask_arr]
    m1, m2, f1, f2, r2 = (load(x) for x in needed)
    keep = ((r2 > r2_thr) & np.isfinite(m1) & np.isfinite(m2)
            & np.isfinite(f1) & np.isfinite(f2))
    if keep.sum() == 0:
        return None
    cdf_is_1 = (sub.get_mapping(1) == "cdf")
    return pd.DataFrame({
        "subject": subject,
        "mode_cdf":    np.where(cdf_is_1, m1, m2)[keep],
        "mode_invcdf": np.where(cdf_is_1, m2, m1)[keep],
        "fwhm_cdf":    np.where(cdf_is_1, f1, f2)[keep],
        "fwhm_invcdf": np.where(cdf_is_1, f2, f1)[keep],
    })


def discover_subjects(bids_folder):
    base = Path(bids_folder) / "derivatives" / "encoding_models" / MODEL_DIR
    return sorted(p.name.removeprefix("sub-") for p in base.glob("sub-*"))


def _scatter(ax, df, par, lim, label):
    x, y = df[f"{par}_cdf"], df[f"{par}_invcdf"]
    ax.plot(lim, lim, color="k", ls=":", lw=0.8, alpha=0.6, zorder=0)
    ax.scatter(x, y, s=4, alpha=0.15, color="#3B5BA5", edgecolor="none")
    r = np.corrcoef(x, y)[0, 1]
    shift = float(np.median(y - x))
    ax.set(xlim=lim, ylim=lim, xlabel=f"{label} — CDF", ylabel=f"{label} — Inverse-CDF")
    ax.set_aspect("equal")
    ax.set_title(f"{label}: r={r:.2f}, median shift {shift:+.1f}  "
                 f"(n={len(x):,} vox)", fontsize=9)


def run(subjects, r2_thr, out, bids_folder, smoothed=False):
    dfs = [d for s in subjects
           if (d := load_subject(s, r2_thr, smoothed, bids_folder)) is not None]
    if not dfs:
        raise SystemExit(f"No fwhm-shift param maps found (run fit_aprf.py "
                         f"--model fwhm-shift first). Looked under {MODEL_DIR}/.")
    df = pd.concat(dfs, ignore_index=True)
    n_sub = df["subject"].nunique()
    print(f"{n_sub} subjects · {len(df):,} NPCr voxels (R²>{r2_thr})")
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.4), constrained_layout=True)
        _scatter(axes[0], df, "mode", (0, 45), "Preferred value (CHF)")
        vmax = float(np.nanpercentile(
            np.r_[df["fwhm_cdf"], df["fwhm_invcdf"]], 99))
        _scatter(axes[1], df, "fwhm", (0, vmax), "Tuning width FWHM (CHF)")
        fig.suptitle(f"NPCr value pRF: CDF vs Inverse-CDF (fwhm-shift model, "
                     f"n={n_sub})  ·  on y=x ⇒ abstract; off ⇒ mapping-dependent",
                     y=1.04, fontsize=10)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # per-subject median shift summary
        g = df.groupby("subject").apply(
            lambda s: pd.Series({"d_mode": np.median(s.mode_invcdf - s.mode_cdf),
                                 "d_fwhm": np.median(s.fwhm_invcdf - s.fwhm_cdf)}),
            include_groups=False).reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), constrained_layout=True)
        for ax, col, lab in zip(axes, ["d_mode", "d_fwhm"],
                                ["Δ mode (CHF)", "Δ fwhm (CHF)"]):
            ax.axvline(0, color="k", ls=":", lw=0.8)
            ax.scatter(g[col], range(len(g)), color="#3B5BA5")
            ax.set_yticks(range(len(g))); ax.set_yticklabels(g["subject"], fontsize=7)
            ax.set_xlabel(f"{lab}  Inverse-CDF − CDF")
            ax.axvline(float(g[col].median()), color="#E76F51", lw=1.2,
                       label=f"median {g[col].median():+.1f}")
            ax.legend(fontsize=7)
        fig.suptitle(f"Per-subject median per-condition shift (fwhm-shift)", y=1.03)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--r2-thr", type=float, default=0.05)
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    subs = args.subjects or discover_subjects(args.bids_folder)
    run(subs, args.r2_thr, Path(args.out), args.bids_folder, smoothed=args.smoothed)


if __name__ == "__main__":
    main()
