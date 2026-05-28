"""V1 vs NPCr Fisher information per condition, mirroring the structure
of compare_v1_npcr_uncertainty.pdf.

  - Refresher page first (stimulus distributions + mapping curves)
  - SPHERICAL section: 4 pages (fdr05, nvoxels-100 × unsmoothed, smoothed)
  - RESIDUAL section : 4 pages, same layout
  - Per page: V1 (vonmises, per-session FI) | NPCr (aprf-session-shift FI)

V1 FI comes from per-session compute_fisher_information.py runs (sessions
loop the script externally so each session gets its own FI curve). NPCr
FI comes from aprf-session-shift's internal per-session loop.

Usage:
    python -m abstract_values.visualize.fisher_information_v1_vs_npcr
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

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
               / "fisher_information_v1_vs_npcr.pdf")
COND_COLOUR = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}
TRAINED_MIN, TRAINED_MAX = 7.5, 172.5
SELECTIONS = ("nvoxels-fdr05", "nvoxels-100")
SMOOTHINGS = (False, True)


def _v1_path(subject, session, sel_tag, smoothed, noise):
    """Per-session V1 FI TSV. The compute_fisher_information.py outputs
    to vonmises/sub-X/ses-Y/func/ when SESSION env is set."""
    smooth = "_smoothed" if smoothed else ""
    noise_tag = f"_noise-{noise}" if noise else ""
    return (DERIV / "vonmises" / f"sub-{subject}" / f"ses-{session}" / "func"
            / f"sub-{subject}_ses-{session}_task-abstractvalue"
              f"_mask-BensonV1_hemi-LR_{sel_tag}{noise_tag}{smooth}"
              f"_desc-fisherinfo_pe.tsv")


def _npcr_path(subject, session, sel_tag, smoothed, noise):
    """Per-session NPCr FI TSV from aprf-session-shift."""
    smooth = "_smoothed" if smoothed else ""
    noise_tag = f"_noise-{noise}" if noise else ""
    return (DERIV / "aprf-session-shift" / f"sub-{subject}"
            / f"ses-{session}" / "func"
            / f"sub-{subject}_ses-{session}_task-abstractvalue"
              f"_mask-NPCr_{sel_tag}{noise_tag}{smooth}"
              f"_desc-fisherinfo_pe.tsv")


def _orientation_lookup(subjects):
    """Per-condition (orientation_deg → value_chf) lookup from gabor events."""
    pairs = {"cdf": set(), "inverse_cdf": set()}
    for s in subjects:
        try:
            sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
            for ses in sub.get_sessions():
                cond = sub.get_mapping(ses)
                ev = sub.get_events(ses, sub.get_runs(ses))
                for _, row in ev[ev.event_type == "gabor"].iterrows():
                    pairs[cond].add((float(row["orientation"]),
                                      float(row["value"])))
        except Exception:
            pass
    out = {}
    for c, ps in pairs.items():
        if not ps:
            out[c] = pd.DataFrame(columns=["orientation_deg", "value"])
            continue
        out[c] = (pd.DataFrame(sorted(ps),
                                 columns=["orientation_deg", "value"])
                   .drop_duplicates("orientation_deg")
                   .sort_values("orientation_deg")
                   .reset_index(drop=True))
    return out


def discover_subjects():
    seen = set()
    for p in DERIV.glob("aprf-session-shift/sub-*"):
        seen.add(p.name.removeprefix("sub-"))
    return sorted(seen, key=lambda s: (0 if s[0].isdigit() else 1, s))


def _load_fi(path_fn, subjects, sel_tag, smoothed, noise,
              x_label_in_file=None):
    """Load FI TSVs across (subject, session). The TSV is indexed by
    stimulus value (orientation in radians for V1, CHF for NPCr); we
    re-index it as a numeric column and tag it with subject/condition."""
    rows = []
    for s in subjects:
        try:
            sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
        except Exception:
            continue
        for ses in sub.get_sessions():
            p = path_fn(s, ses, sel_tag, smoothed, noise)
            if not p.exists():
                continue
            df = pd.read_csv(p, sep="\t")
            # First column = stimulus (whatever it's called), second = FI
            cols = list(df.columns)
            df.columns = ["stim", "fi"] + cols[2:]
            df["subject"] = s
            df["session"] = ses
            df["condition"] = sub.get_mapping(ses)
            rows.append(df)
    return (pd.concat(rows, ignore_index=True)
              if rows else pd.DataFrame())


def _aggregate(df, x_col, y_col, x_grid):
    per_sub = []
    for _, g in df.groupby("subject"):
        g = g.sort_values(x_col)
        if g[x_col].nunique() < 3:
            continue
        per_sub.append(np.interp(x_grid, g[x_col].values, g[y_col].values,
                                  left=np.nan, right=np.nan))
    if not per_sub:
        return None, None, 0
    arr = np.asarray(per_sub)
    n_eff = np.maximum(np.sum(~np.isnan(arr), axis=0), 1)
    return (np.nanmean(arr, axis=0),
            np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n_eff),
            arr.shape[0])


def page_stimulus_refresher(subjects, lookup, pdf):
    """Identical to compare_v1_npcr_uncertainty: CHF + orientation refresher."""
    rows = []
    for s in subjects:
        try:
            sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
            for ses in sub.get_sessions():
                cond = sub.get_mapping(ses)
                ev = sub.get_events(ses, sub.get_runs(ses))
                for _, row in ev[ev.event_type == "gabor"].iterrows():
                    rows.append({"condition": cond,
                                  "orientation_deg": float(row["orientation"]),
                                  "value_chf": float(row["value"])})
        except Exception:
            pass
    if not rows:
        return
    df = pd.DataFrame(rows)
    CHF_LO, CHF_HI = 0.0, 45.0
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.0),
                              constrained_layout=True)
    fig.suptitle("Stimulus refresher", fontsize=10, y=1.03, color="0.15")
    chf_bins = np.arange(CHF_LO, CHF_HI + 0.5, 0.5)
    for ax, cond in zip(axes[0, :], ("cdf", "inverse_cdf")):
        vals = df[df.condition == cond]["value_chf"].values
        ax.hist(vals, bins=chf_bins, color=COND_COLOUR[cond],
                alpha=0.85, edgecolor="white", linewidth=0.3)
        med = float(np.median(vals))
        q1, q3 = np.percentile(vals, [25, 75])
        ax.axvline(med, color="0.15", lw=1.0, zorder=4)
        ax.axvspan(q1, q3, color="0.85", alpha=0.4, zorder=0)
        ax.text(0.02, 0.95,
                 f"{'CDF' if cond=='cdf' else 'InvCDF'}  ·  "
                 f"median {med:.1f}  ·  IQR [{q1:.1f}, {q3:.1f}]",
                 transform=ax.transAxes, fontsize=8, va="top",
                 color=COND_COLOUR[cond])
        ax.set_xlim(CHF_LO, CHF_HI)
        ax.set_xlabel("Presented CHF")
        ax.set_ylabel("Trial count")
    ax = axes[1, 0]
    for cond in ("cdf", "inverse_cdf"):
        lut = lookup.get(cond)
        if lut is None or lut.empty: continue
        ax.plot(lut["orientation_deg"], lut["value"],
                color=COND_COLOUR[cond], lw=2.0, marker="o", ms=4,
                mec="white", mew=0.4,
                label="CDF" if cond == "cdf" else "InvCDF")
    ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("Orientation (deg)"); ax.set_ylabel("CHF")
    ax.set_title("Mapping CHF(orientation)", fontsize=9, color="0.2")
    ax.legend(loc="upper right", fontsize=8)
    axes[1, 1].axis("off")
    sns.despine(fig=fig, offset=4)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_v1_vs_npcr(subjects, sel_tag, smoothed, lookup, pdf, which_noise):
    """One page of FI comparison: V1 left, NPCr right.

    V1 x-axis: orientation (degrees). NPCr x-axis: value (CHF). The two
    panels can't share a stimulus axis the way EU does (FI on NPCr is
    over CHF, not orientation), so we keep them on their native axes
    rather than forcing the orientation re-projection of compare_v1_npcr_uncertainty.
    """
    noise_token = "spherical" if which_noise == "spherical" else ""

    df_v1   = _load_fi(_v1_path,   subjects, sel_tag, smoothed, noise_token)
    df_npcr = _load_fi(_npcr_path, subjects, sel_tag, smoothed, noise_token)
    if df_v1.empty and df_npcr.empty:
        return

    # V1 stim is radians → convert to degrees for the x-axis
    if not df_v1.empty:
        df_v1["orientation_deg"] = np.rad2deg(df_v1["stim"])

    smooth_lbl = "smoothed" if smoothed else "unsmoothed"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2),
                              constrained_layout=True)
    fig.suptitle(f"Fisher information per condition  ·  "
                 f"{sel_tag}  ·  {smooth_lbl}  ·  noise: {which_noise.upper()}",
                 fontsize=10, y=1.04, color="0.15")

    # ── V1 panel ────────────────────────────────────────────────────────────
    ax = axes[0]
    if not df_v1.empty:
        ori_grid = np.linspace(0, 180, 60)
        for cond, sub_df in df_v1.groupby("condition"):
            mean, sem, n = _aggregate(sub_df, "orientation_deg", "fi",
                                        ori_grid)
            if mean is None: continue
            ax.plot(ori_grid, mean, color=COND_COLOUR[cond], lw=1.8,
                    label=f"{'CDF' if cond=='cdf' else 'InvCDF'}  (n={n})")
            ax.fill_between(ori_grid, mean - sem, mean + sem,
                             color=COND_COLOUR[cond], alpha=0.22,
                             linewidth=0)
        ax.set_xlim(TRAINED_MIN, TRAINED_MAX)
        ax.set_xticks([15, 45, 90, 135, 165])
        # Cardinal reference dotted lines
        for c in (45, 90, 135):
            ax.axvline(c, color="0.8", lw=0.5, ls=":", zorder=0)
    else:
        ax.text(0.5, 0.5, "No V1 data for this cell",
                transform=ax.transAxes, ha="center", va="center",
                color="0.5")
    ax.set_xlabel("Orientation (deg)")
    ax.set_ylabel("V1 Fisher information (a.u.)")
    ax.set_title("V1 (vonmises) FI vs orientation",
                  fontsize=9, color="0.2")
    if ax.has_data(): ax.legend(loc="upper right", fontsize=7)

    # ── NPCr panel ──────────────────────────────────────────────────────────
    ax = axes[1]
    if not df_npcr.empty:
        v_lo = float(df_npcr["stim"].min())
        v_hi = float(df_npcr["stim"].max())
        chf_grid = np.linspace(v_lo, v_hi, 80)
        for cond, sub_df in df_npcr.groupby("condition"):
            mean, sem, n = _aggregate(sub_df, "stim", "fi", chf_grid)
            if mean is None: continue
            ax.plot(chf_grid, mean, color=COND_COLOUR[cond], lw=1.8,
                    label=f"{'CDF' if cond=='cdf' else 'InvCDF'}  (n={n})")
            ax.fill_between(chf_grid, mean - sem, mean + sem,
                             color=COND_COLOUR[cond], alpha=0.22,
                             linewidth=0)
        ax.set_xlim(v_lo, v_hi + 3)
    else:
        ax.text(0.5, 0.5, "No NPCr data for this cell",
                transform=ax.transAxes, ha="center", va="center",
                color="0.5")
    ax.set_xlabel("Value (CHF)")
    ax.set_ylabel("NPCr Fisher information (a.u.)")
    ax.set_title("NPCr (aprf-session-shift) FI vs value",
                  fontsize=9, color="0.2")
    if ax.has_data(): ax.legend(loc="upper right", fontsize=7)

    sns.despine(fig=fig, offset=4)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def run(subjects, out):
    if subjects is None:
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No subjects with aprf-session-shift fits.")
    print(f"Subjects: {subjects}")
    lookup = _orientation_lookup(subjects)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        page_stimulus_refresher(subjects, lookup, pdf)
        for which_noise in ("spherical", "residual"):
            print(f"\n────── {which_noise.upper()} ──────")
            for sel_tag in SELECTIONS:
                for smoothed in SMOOTHINGS:
                    print(f"  {sel_tag}  smoothed={smoothed}")
                    page_v1_vs_npcr(subjects, sel_tag, smoothed,
                                     lookup, pdf, which_noise)
    print(f"\nWrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, Path(args.out))


if __name__ == "__main__":
    main()
