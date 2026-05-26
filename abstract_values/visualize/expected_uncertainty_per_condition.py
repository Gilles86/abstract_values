"""Per-condition expected decoding uncertainty (bias + variance) on NPCr.

Loads the per-(subject, session) TSVs written by
``compute_expected_decoded_value_aprf.py`` (which simulates noisy
responses from each session's session-shift PRF + Student-t noise and
decodes them through the same model), remaps each session to its
CDF/InvCDF condition via ``Subject.get_mapping``, and plots:

  1. Per-condition decoder **bias** (mean_E vs true value) — should sit
     on y=x for an unbiased decoder.
  2. Per-condition decoder **uncertainty** (√var_E across simulations)
     vs true value — the actual SD of the posterior expected value, not
     the posterior width.
  3. Per-subject panels with both curves.
  4. Per-value difference (CDF − InvCDF) of var_E.

Smoothed and unsmoothed variants are rendered as separate sections of
the same PDF (mirrors fisher_information_per_condition.py).
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

DERIV = Path(BIDS_FOLDER) / "derivatives" / "encoding_models" / "aprf-session-shift"
DEFAULT_OUT = Path(BIDS_FOLDER) / "derivatives" / "qa" / "expected_uncertainty_per_condition.pdf"

COND_COLOUR = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}


def _tsv_path(subject, session, mask, nvoxels, nsims, smoothed):
    smooth = "_smoothed" if smoothed else ""
    return (DERIV / f"sub-{subject}" / f"ses-{session}" / "func"
            / f"sub-{subject}_ses-{session}_task-abstractvalue_mask-{mask}"
              f"_nvoxels-{nvoxels}_nsims-{nsims}{smooth}_desc-expected_decoded_pe.tsv")


def discover_subjects(mask, nvoxels, nsims, smoothed):
    subs = set()
    for p in DERIV.glob("sub-*"):
        s = p.name.removeprefix("sub-")
        if (_tsv_path(s, 1, mask, nvoxels, nsims, smoothed).exists()
                and _tsv_path(s, 2, mask, nvoxels, nsims, smoothed).exists()):
            subs.add(s)
    return sorted(subs, key=lambda s: (0 if s[0].isdigit() else 1, s))


def collect(subjects, mask, nvoxels, nsims, smoothed):
    rows = []
    for s in subjects:
        sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
        for ses in (1, 2):
            p = _tsv_path(s, ses, mask, nvoxels, nsims, smoothed)
            if not p.exists():
                continue
            df = pd.read_csv(p, sep="\t")
            df["subject"] = s
            df["session"] = ses
            df["condition"] = sub.get_mapping(ses)
            df["sd_E"] = np.sqrt(df["var_E"])
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _banner_page(text, pdf):
    fig, ax = plt.subplots(figsize=(7.5, 1.2), constrained_layout=True)
    ax.text(0.5, 0.5, text, ha="center", va="center",
            fontsize=18, color="0.1", transform=ax.transAxes)
    ax.set_axis_off()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_group_bias_uncertainty(df, subjects, pdf):
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             constrained_layout=True)

    # Left: bias (mean_E vs true value)
    ax = axes[0]
    for cond, sub_df in df.groupby("condition"):
        g = sub_df.groupby("value")["mean_E"].agg(["mean", "sem"]).reset_index()
        g = g.sort_values("value")
        ax.plot(g["value"], g["mean"], color=COND_COLOUR[cond], lw=1.8, label=cond)
        ax.fill_between(g["value"], g["mean"] - g["sem"], g["mean"] + g["sem"],
                        color=COND_COLOUR[cond], alpha=0.22, linewidth=0)
    lim = (df["value"].min(), df["value"].max())
    ax.plot(lim, lim, "--", color="0.5", lw=0.8, label="Identity")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True value (CHF)")
    ax.set_ylabel(r"Mean decoded $\hat{V}$ (CHF)")
    ax.set_title("Decoder bias per condition", fontsize=10, color="0.2")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    # Right: uncertainty (SD of decoded across simulations)
    ax = axes[1]
    for cond, sub_df in df.groupby("condition"):
        g = sub_df.groupby("value")["sd_E"].agg(["mean", "sem"]).reset_index()
        g = g.sort_values("value")
        ax.plot(g["value"], g["mean"], color=COND_COLOUR[cond], lw=1.8, label=cond)
        ax.fill_between(g["value"], g["mean"] - g["sem"], g["mean"] + g["sem"],
                        color=COND_COLOUR[cond], alpha=0.22, linewidth=0)
    ax.set_xlabel("True value (CHF)")
    ax.set_ylabel(r"$\sqrt{\mathrm{Var}[\hat{V}]}$ across simulations (CHF)")
    ax.set_title(f"Expected uncertainty per condition  (n={df['subject'].nunique()})",
                 fontsize=10, color="0.2")
    ax.legend(loc="upper right", fontsize=8)

    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_per_subject(df, subjects, pdf):
    if df.empty:
        return
    n = len(subjects)
    cols = 3
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.5, 1.9 * rows + 0.4),
                             constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()
    for i, s in enumerate(subjects):
        ax = axes[i]
        sub_df = df[df.subject == s]
        for cond in ("cdf", "inverse_cdf"):
            d = sub_df[sub_df.condition == cond].sort_values("value")
            if d.empty:
                continue
            ax.plot(d["value"], d["sd_E"], color=COND_COLOUR[cond], lw=1.0,
                    label=cond)
        ax.set_title(f"sub-{s}", fontsize=8.5, color="0.2")
        ax.axhline(0, color="0.7", lw=0.4, ls="--", zorder=0)
    for j in range(len(subjects), len(axes)):
        axes[j].set_axis_off()
    axes[0].legend(loc="upper right", fontsize=7.5)
    fig.supxlabel("True value (CHF)", fontsize=9)
    fig.supylabel(r"$\sqrt{\mathrm{Var}[\hat{V}]}$", fontsize=9)
    fig.suptitle("Per-subject expected uncertainty (decoder SD across simulations)",
                 fontsize=10, y=1.02)
    sns.despine(fig=fig, offset=3, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_per_value_diff(df, subjects, pdf):
    """Per-value difference (CDF − InvCDF) of decoder SD."""
    if df.empty:
        return
    rows = []
    for s in subjects:
        a = (df[(df.subject == s) & (df.condition == "cdf")]
             .set_index("value")["sd_E"])
        b = (df[(df.subject == s) & (df.condition == "inverse_cdf")]
             .set_index("value")["sd_E"])
        if a.empty or b.empty:
            continue
        diff = (a - b).rename("diff").reset_index()
        diff["subject"] = s
        rows.append(diff)
    if not rows:
        return
    long = pd.concat(rows, ignore_index=True)
    grp = long.groupby("value")["diff"].agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 3.2), constrained_layout=True)
    for s, d in long.groupby("subject"):
        d = d.sort_values("value")
        ax.plot(d["value"], d["diff"], color="0.7", lw=0.7, zorder=1)
    ax.plot(grp["value"], grp["mean"], color="#3B5BA5", lw=1.8, zorder=3,
            label="Mean")
    ax.fill_between(grp["value"], grp["mean"] - grp["sem"],
                    grp["mean"] + grp["sem"],
                    color="#3B5BA5", alpha=0.22, linewidth=0, zorder=2)
    ax.axhline(0, color="0.7", lw=0.5, ls="--", zorder=0)
    ax.set_xlabel("True value (CHF)")
    ax.set_ylabel(r"SD(CDF) − SD(InvCDF)  (CHF)")
    ax.set_title("Per-value decoder-SD difference between conditions",
                 fontsize=10, color="0.2")
    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def run(subjects, mask, nvoxels, nsims, out,
        variants=("unsmoothed", "smoothed")):
    if subjects is None:
        subjects_set = set()
        for v in variants:
            subjects_set.update(discover_subjects(mask, nvoxels, nsims,
                                                   smoothed=(v == "smoothed")))
        subjects = sorted(subjects_set,
                          key=lambda s: (0 if s[0].isdigit() else 1, s))
    if not subjects:
        raise SystemExit(f"No expected_decoded TSVs found for "
                          f"mask={mask}, nvoxels={nvoxels}, nsims={nsims}")

    out.parent.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    with PdfPages(out) as pdf:
        for v in variants:
            smoothed = (v == "smoothed")
            sub_v = [s for s in subjects
                     if _tsv_path(s, 1, mask, nvoxels, nsims, smoothed).exists()
                     and _tsv_path(s, 2, mask, nvoxels, nsims, smoothed).exists()]
            print(f"\n=== {v} ({len(sub_v)} subjects) ===  {sub_v}")
            if not sub_v:
                continue
            df = collect(sub_v, mask, nvoxels, nsims, smoothed)
            _banner_page(f"{v.upper()}  ·  NPCr expected uncertainty  ·  "
                          f"n_voxels={nvoxels}  ·  n_sims={nsims}", pdf)
            page_group_bias_uncertainty(df, sub_v, pdf)
            page_per_subject(df, sub_v, pdf)
            page_per_value_diff(df, sub_v, pdf)
            df["variant"] = v
            all_dfs.append(df)
    if all_dfs:
        tsv = out.with_suffix(".tsv")
        pd.concat(all_dfs, ignore_index=True).to_csv(tsv, sep="\t", index=False)
        print(f"\nWrote {out}\nSidecar: {tsv}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--mask", default="NPCr")
    p.add_argument("--nvoxels", type=int, default=100)
    p.add_argument("--nsims", type=int, default=1000)
    p.add_argument("--variants", nargs="+",
                   default=["unsmoothed", "smoothed"],
                   choices=["unsmoothed", "smoothed"])
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, args.mask, args.nvoxels, args.nsims, Path(args.out),
        variants=tuple(args.variants))


if __name__ == "__main__":
    main()
