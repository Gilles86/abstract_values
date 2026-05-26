"""V1 vs NPCr expected decoding uncertainty per condition, on a common
orientation axis. The disambiguation figure:

  - V1 vonmises decoder uncertainty on the **orientation** axis directly.
  - NPCr aprf decoder uncertainty on the **value** axis, projected onto
    the **corresponding orientation** via the per-condition CHF↔orientation
    lookup from gabor events.tsv.

If V1 is orientation-tuned (the null hypothesis), its SD-vs-orientation
curves under CDF and InvCDF should overlay. If NPCr is value-tuned
with condition-dependent reorganization (efficient coding), its
SD-vs-orientation curves should diverge between conditions — because
the CHF→orientation warp differs.

The figure renders four selection/smoothing combinations as separate
pages (n_voxels=100 + FDR05 × smoothed + unsmoothed), each with V1 on
the left, NPCr on the right.

A trailing page overlays behavioural noisiness (MAE in BDM bid) per
condition for reference.

Usage:
    python -m abstract_values.visualize.compare_v1_npcr_uncertainty
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
DEFAULT_OUT = Path(BIDS_FOLDER) / "derivatives" / "qa" / "compare_v1_npcr_uncertainty.pdf"
COND_COLOUR = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}
SELECTIONS = ("nvoxels-100", "nvoxels-fdr05")
SMOOTHINGS = (False, True)


def _v1_tsv_path(subject, session, sel_tag, smoothed, nsims=1000):
    smooth = "_smoothed" if smoothed else ""
    return (DERIV / "vonmises" / f"sub-{subject}" / f"ses-{session}" / "func"
            / f"sub-{subject}_ses-{session}_task-abstractvalue"
              f"_mask-BensonV1_hemi-LR_{sel_tag}_nsims-{nsims}"
              f"{smooth}_desc-expected_decoded_orientation_pe.tsv")


def _npcr_tsv_path(subject, session, sel_tag, smoothed, nsims=1000):
    smooth = "_smoothed" if smoothed else ""
    return (DERIV / "aprf-session-shift" / f"sub-{subject}" / f"ses-{session}" / "func"
            / f"sub-{subject}_ses-{session}_task-abstractvalue"
              f"_mask-NPCr_{sel_tag}_nsims-{nsims}{smooth}_desc-expected_decoded_pe.tsv")


def discover_subjects():
    seen = set()
    for p in (DERIV / "vonmises").glob("sub-*"):
        seen.add(p.name.removeprefix("sub-"))
    for p in (DERIV / "aprf-session-shift").glob("sub-*"):
        seen.add(p.name.removeprefix("sub-"))
    return sorted(seen, key=lambda s: (0 if s[0].isdigit() else 1, s))


def load_v1(subjects, sel_tag, smoothed, nsims=1000):
    rows = []
    for s in subjects:
        sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
        for ses in (1, 2):
            p = _v1_tsv_path(s, ses, sel_tag, smoothed, nsims=nsims)
            if not p.exists():
                continue
            df = pd.read_csv(p, sep="\t")
            df["subject"] = s
            df["session"] = ses
            df["condition"] = sub.get_mapping(ses)
            # V1 TSV has value=orientation in radians + value_deg
            if "value_deg" not in df.columns:
                df["value_deg"] = np.rad2deg(df["value"])
            df["orientation_deg"] = df["value_deg"]
            if "sd_E" not in df.columns:
                df["sd_E"] = np.sqrt(df["var_E"])
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _orientation_lookup(subjects):
    """Per-condition (value → orientation_deg) lookup from gabor events."""
    table = {"cdf": set(), "inverse_cdf": set()}
    for s in subjects:
        try:
            sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
            for ses in sub.get_sessions():
                cond = sub.get_mapping(ses)
                ev = sub.get_events(ses, sub.get_runs(ses))
                for _, row in ev[ev.event_type == "gabor"].iterrows():
                    table[cond].add((float(row["orientation"]),
                                      float(row["value"])))
        except Exception:
            pass
    out = {}
    for cond, pairs in table.items():
        if not pairs:
            out[cond] = pd.DataFrame(columns=["orientation_deg", "value"])
            continue
        df = (pd.DataFrame(sorted(pairs), columns=["orientation_deg", "value"])
              .drop_duplicates("value")
              .sort_values("value")
              .reset_index(drop=True))
        out[cond] = df
    return out


def load_npcr(subjects, sel_tag, smoothed, lookup, nsims=1000):
    rows = []
    for s in subjects:
        sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
        for ses in (1, 2):
            p = _npcr_tsv_path(s, ses, sel_tag, smoothed, nsims=nsims)
            if not p.exists():
                continue
            df = pd.read_csv(p, sep="\t")
            df["subject"] = s
            df["session"] = ses
            df["condition"] = sub.get_mapping(ses)
            df["sd_E"] = np.sqrt(df["var_E"])
            # value → orientation via per-condition lookup
            df["orientation_deg"] = np.nan
            for cond in ("cdf", "inverse_cdf"):
                lut = lookup.get(cond)
                if lut is None or lut.empty:
                    continue
                mask = df["condition"] == cond
                df.loc[mask, "orientation_deg"] = np.interp(
                    df.loc[mask, "value"].values,
                    lut["value"].values,
                    lut["orientation_deg"].values,
                    left=np.nan, right=np.nan,
                )
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    return df.dropna(subset=["orientation_deg"]).reset_index(drop=True)


def _interp_group(df, ori_grid):
    """Per-subject SD interpolated onto ori_grid, then group mean ± SEM."""
    per_sub = []
    for s, ssub in df.groupby("subject"):
        ssub = ssub.sort_values("orientation_deg")
        sd_interp = np.interp(ori_grid,
                              ssub["orientation_deg"].values,
                              ssub["sd_E"].values,
                              left=np.nan, right=np.nan)
        per_sub.append(sd_interp)
    if not per_sub:
        return np.full_like(ori_grid, np.nan), np.full_like(ori_grid, np.nan), 0
    per_sub = np.array(per_sub)
    mean_sd = np.nanmean(per_sub, axis=0)
    sem_sd = np.nanstd(per_sub, axis=0, ddof=1) / np.sqrt(
        np.maximum(np.sum(~np.isnan(per_sub), axis=0), 1))
    return mean_sd, sem_sd, per_sub.shape[0]


def page_v1_vs_npcr(subjects, sel_tag, smoothed, lookup, pdf):
    df_v1 = load_v1(subjects, sel_tag, smoothed)
    df_n = load_npcr(subjects, sel_tag, smoothed, lookup)

    if df_v1.empty and df_n.empty:
        return

    ori_grid = np.linspace(0, 180, 60, dtype=np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2),
                             constrained_layout=True, sharey=False)
    smooth_lbl = "smoothed" if smoothed else "unsmoothed"
    fig.suptitle(f"Expected decoder SD vs orientation  ·  "
                 f"{sel_tag}  ·  {smooth_lbl}",
                 fontsize=10, y=1.03, color="0.15")

    # V1 panel
    ax = axes[0]
    for cond, sub_df in df_v1.groupby("condition"):
        mean_sd, sem_sd, n = _interp_group(sub_df, ori_grid)
        ax.plot(ori_grid, mean_sd, color=COND_COLOUR[cond], lw=1.8,
                label=f"{cond}  (n={n})")
        ax.fill_between(ori_grid, mean_sd - sem_sd, mean_sd + sem_sd,
                        color=COND_COLOUR[cond], alpha=0.22, linewidth=0)
    ax.set_xlabel("Orientation (deg)")
    ax.set_ylabel(r"V1 decoder SD  $\sqrt{\mathrm{Var}[\hat{\theta}]}$  (deg)")
    ax.set_title("V1 (vonmises)  —  decoded ORIENTATION", fontsize=9, color="0.2")
    ax.legend(loc="upper right", fontsize=7.5)

    # NPCr panel
    ax = axes[1]
    for cond, sub_df in df_n.groupby("condition"):
        mean_sd, sem_sd, n = _interp_group(sub_df, ori_grid)
        ax.plot(ori_grid, mean_sd, color=COND_COLOUR[cond], lw=1.8,
                label=f"{cond}  (n={n})")
        ax.fill_between(ori_grid, mean_sd - sem_sd, mean_sd + sem_sd,
                        color=COND_COLOUR[cond], alpha=0.22, linewidth=0)
    ax.set_xlabel("Corresponding orientation (deg)")
    ax.set_ylabel(r"NPCr decoder SD  $\sqrt{\mathrm{Var}[\hat{V}]}$  (CHF)")
    ax.set_title("NPCr (aprf, value)  —  projected via CHF→orientation",
                 fontsize=9, color="0.2")
    ax.legend(loc="upper right", fontsize=7.5)

    for ax in axes:
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 45, 90, 135, 180])
    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return df_v1, df_n


def page_behavior_overlay(pdf):
    """Behavioural noisiness: per-condition summary + per-orientation curve."""
    p_agg = Path("/Users/gdehol/git/abstract_values/notes/figures/behavior_noisiness.tsv")
    p_ori = Path("/Users/gdehol/git/abstract_values/notes/figures/behavior_noisiness_per_orientation.tsv")
    if not p_agg.exists() or not p_ori.exists():
        return
    df_agg = pd.read_csv(p_agg, sep="\t")
    df_ori = pd.read_csv(p_ori, sep="\t")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6),
                             constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.2, 1.2, 2.4]))
    metrics = [("mean_abs_error", "Mean abs error (CHF)"),
               ("sd_error",       "SD of error (CHF)")]
    for ax, (m, lbl) in zip(axes[:2], metrics):
        for sub, g in df_agg.groupby("subject"):
            if g["condition"].nunique() < 2:
                continue
            row_c = g[g.condition == "cdf"]
            row_i = g[g.condition == "inverse_cdf"]
            if row_c.empty or row_i.empty:
                continue
            ax.plot([0, 1], [row_c[m].iloc[0], row_i[m].iloc[0]],
                    "-o", color="0.7", lw=0.8, ms=4, zorder=1)
        for i, cond in enumerate(("cdf", "inverse_cdf")):
            vals = df_agg[df_agg.condition == cond][m]
            ax.scatter([i], [vals.mean()], s=140, marker="D",
                       facecolor=COND_COLOUR[cond], edgecolor="black",
                       linewidth=1.4, zorder=5)
            ax.text(i, vals.mean(), f"  {vals.mean():.2f}",
                    fontsize=8, color="0.2", va="center")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["CDF", "InvCDF"])
        ax.set_ylabel(lbl)

    # Per-orientation behavioural SD curve — matches the V1/NPCr panels
    ax = axes[2]
    for cond, sub_df in df_ori.groupby("condition"):
        g = sub_df.groupby("orientation_deg")["sd_error"].agg(
            ["mean", "sem", "count"]).reset_index()
        ax.plot(g["orientation_deg"], g["mean"],
                "-o", color=COND_COLOUR[cond], lw=1.6, ms=4,
                label=f"{cond}  (n={int(g['count'].max())})")
        ax.fill_between(g["orientation_deg"],
                        g["mean"] - g["sem"],
                        g["mean"] + g["sem"],
                        color=COND_COLOUR[cond], alpha=0.22, linewidth=0)
    # Mark the three behavioural-precision anchor points: orientation
    # extremes (min/max CHF in both mappings) + the shared median (90° →
    # 22 CHF under both). Subjects show SD dips at all three.
    for anchor in (7.5, 90.0, 172.5):
        ax.axvline(anchor, color="0.5", lw=0.6, ls=":", zorder=0)
    ymax = ax.get_ylim()[1]
    ax.text(90, ymax * 0.97, "shared\nmedian (22 CHF)",
            ha="center", va="top", fontsize=7.5, color="0.4")
    ax.text(7.5, ymax * 0.97, "min\nCHF",
            ha="left", va="top", fontsize=7.5, color="0.4")
    ax.text(172.5, ymax * 0.97, "max\nCHF",
            ha="right", va="top", fontsize=7.5, color="0.4")
    ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("Orientation (deg)")
    ax.set_ylabel("Behavioural SD of bid error (CHF)")
    ax.set_title("Behaviour vs orientation  —  SD dips at 0°, 90°, 180°  "
                 "(min, shared median, max)",
                 fontsize=9, color="0.2")
    ax.legend(loc="upper right", fontsize=7.5)

    fig.suptitle("Behavioural noisiness — BDM bid error", fontsize=10,
                 y=1.04, color="0.15")
    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def run(subjects, out, nsims=1000):
    if subjects is None:
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No subjects discovered.")
    print(f"Subjects: {subjects}")
    print("Building value↔orientation lookup…")
    lookup = _orientation_lookup(subjects)
    for c, l in lookup.items():
        print(f"  {c}: {len(l)} pairs")

    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        # 4 pages: each (selection × smoothing) combo
        for sel_tag in SELECTIONS:
            for smoothed in SMOOTHINGS:
                print(f"\n=== {sel_tag}  smoothed={smoothed} ===")
                page_v1_vs_npcr(subjects, sel_tag, smoothed, lookup, pdf)
        # Behavior reference
        page_behavior_overlay(pdf)

    print(f"\nWrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--nsims", type=int, default=1000)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, Path(args.out), nsims=args.nsims)


if __name__ == "__main__":
    main()
