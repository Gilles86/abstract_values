"""Population Fisher information across the value axis, split by condition.

Loads per-(subject, session) FI TSVs written by
``compute_fisher_information_aprf.py --model session-shift`` from
``derivatives/encoding_models/aprf-session-shift/sub-XX/ses-N/func/``.

Each TSV is one column of (value, fisher_information) over a value grid
(default 200 points 0.5–50 CHF), summed over the top-N voxels by R².

We remap (session 1, session 2) → (CDF, InvCDF) per subject via
``Subject.get_mapping``, then overlay group mean ± SEM per condition.

Three pages:
  1. Group FI curve per condition + per-subject light overlays. Stimulus
     density per condition shaded in the background.
  2. Per-subject 2-panel (CDF / InvCDF) FI traces.
  3. Difference: (FI_CDF − FI_InvCDF) per value, per-subject + mean.

Usage:
    python -m abstract_values.visualize.fisher_information_per_condition
    python -m abstract_values.visualize.fisher_information_per_condition --nvoxels 100
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
DEFAULT_OUT = Path(BIDS_FOLDER) / "derivatives" / "qa" / "fisher_information_per_condition.pdf"

COND_COLOUR = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}


def _fi_path(subject: str, session: int, nvoxels: int, mask: str) -> Path:
    return (DERIV / f"sub-{subject}" / f"ses-{session}" / "func"
            / f"sub-{subject}_ses-{session}_task-abstractvalue_mask-{mask}"
              f"_nvoxels-{nvoxels}_desc-fisherinfo_pe.tsv")


def discover_subjects(nvoxels: int, mask: str) -> list[str]:
    subs = set()
    for p in DERIV.glob("sub-*"):
        s = p.name.removeprefix("sub-")
        if _fi_path(s, 1, nvoxels, mask).exists() and _fi_path(s, 2, nvoxels, mask).exists():
            subs.add(s)
    return sorted(subs, key=lambda s: (0 if s[0].isdigit() else 1, s))


def collect(subjects: list[str], nvoxels: int, mask: str) -> pd.DataFrame:
    rows = []
    for sub_id in subjects:
        sub = Subject(sub_id, bids_folder=Path(BIDS_FOLDER))
        for ses in (1, 2):
            p = _fi_path(sub_id, ses, nvoxels, mask)
            if not p.exists():
                continue
            df = pd.read_csv(p, sep="\t", index_col=0)
            cond = sub.get_mapping(ses)
            df = df.reset_index().rename(columns={df.index.name or "": "value"})
            df["subject"] = sub_id; df["session"] = ses; df["condition"] = cond
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out.columns = ["value", "fi", "subject", "session", "condition"]
    return out


def _pool_value_distributions(subjects: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, list[float]] = {"cdf": [], "inverse_cdf": []}
    for sub_id in subjects:
        try:
            sub = Subject(sub_id, bids_folder=Path(BIDS_FOLDER))
            for ses in sub.get_sessions():
                cond = sub.get_mapping(ses)
                ev = sub.get_events(ses, sub.get_runs(ses))
                vs = ev[ev.event_type == "gabor"]["value"].astype(float).values
                out[cond].extend(vs.tolist())
        except Exception:
            pass
    return {k: np.asarray(v) for k, v in out.items()}


def page_group(df: pd.DataFrame, subjects: list[str], pdf: PdfPages):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.0), constrained_layout=True)

    # Stimulus density bands (shaded) per condition — for visual reference
    val_dists = _pool_value_distributions(subjects)
    val_ax2 = ax.twinx()
    for cond, vals in val_dists.items():
        sns.kdeplot(vals, ax=val_ax2, color=COND_COLOUR[cond], fill=True,
                    alpha=0.08, lw=0.6, clip=(df["value"].min(),
                                              df["value"].max()),
                    cut=0)
    val_ax2.set_yticks([]); val_ax2.set_ylabel("")
    val_ax2.spines["right"].set_visible(False)
    val_ax2.spines["top"].set_visible(False)

    # Per-subject light traces
    for sub_id in subjects:
        for cond in ("cdf", "inverse_cdf"):
            sub_df = df[(df.subject == sub_id) & (df.condition == cond)].sort_values("value")
            if sub_df.empty:
                continue
            ax.plot(sub_df["value"], sub_df["fi"],
                    color=COND_COLOUR[cond], alpha=0.25, lw=0.7, zorder=1)

    # Group mean ± SEM per condition
    grp = (df.groupby(["condition", "value"])["fi"]
           .agg(["mean", "sem"]).reset_index())
    for cond, sub_df in grp.groupby("condition"):
        sub_df = sub_df.sort_values("value")
        ax.plot(sub_df["value"], sub_df["mean"],
                color=COND_COLOUR[cond], lw=1.8, zorder=3)
        ax.fill_between(sub_df["value"],
                        sub_df["mean"] - sub_df["sem"],
                        sub_df["mean"] + sub_df["sem"],
                        color=COND_COLOUR[cond], alpha=0.22,
                        linewidth=0, zorder=2)
        # Direct label at right edge
        last = sub_df.iloc[-1]
        ax.text(last["value"] + 0.5, last["mean"], cond,
                color=COND_COLOUR[cond], fontsize=8.5, va="center")

    ax.set_xlabel("Value (CHF)")
    ax.set_ylabel("Population Fisher information  (a.u.)")
    ax.set_title(f"NPCr Fisher information per condition  "
                 f"(n={len(subjects)} subjects, mean ± SEM)",
                 fontsize=10, color="0.2")
    ax.set_xlim(df["value"].min(), df["value"].max() + 4)
    ax.axhline(0, color="0.7", lw=0.5, ls="--", zorder=0)
    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_per_subject(df: pd.DataFrame, subjects: list[str], pdf: PdfPages):
    if df.empty:
        return
    n = len(subjects)
    cols = 3
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.5, 2.1 * rows + 0.4),
                             constrained_layout=True, sharex=True, sharey=False)
    axes = np.atleast_2d(axes).ravel()
    for i, sub_id in enumerate(subjects):
        ax = axes[i]
        for cond in ("cdf", "inverse_cdf"):
            sub_df = df[(df.subject == sub_id) & (df.condition == cond)].sort_values("value")
            if sub_df.empty:
                continue
            ax.plot(sub_df["value"], sub_df["fi"],
                    color=COND_COLOUR[cond], lw=1.2, label=cond)
        ax.set_title(f"sub-{sub_id}", fontsize=8.5, color="0.2")
        ax.axhline(0, color="0.7", lw=0.4, ls="--", zorder=0)
    for j in range(len(subjects), len(axes)):
        axes[j].set_axis_off()
    axes[0].legend(loc="upper right", fontsize=7.5)
    fig.supxlabel("Value (CHF)", fontsize=9)
    fig.supylabel("FI", fontsize=9)
    fig.suptitle("Per-subject FI per condition", fontsize=10, y=1.02)
    sns.despine(fig=fig, offset=3, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_difference(df: pd.DataFrame, subjects: list[str], pdf: PdfPages):
    if df.empty:
        return
    rows = []
    for sub_id in subjects:
        a = df[(df.subject == sub_id) & (df.condition == "cdf")].set_index("value")["fi"]
        b = df[(df.subject == sub_id) & (df.condition == "inverse_cdf")].set_index("value")["fi"]
        if a.empty or b.empty:
            continue
        diff = (a - b).rename("diff").reset_index()
        diff["subject"] = sub_id
        rows.append(diff)
    if not rows:
        return
    long = pd.concat(rows, ignore_index=True)
    grp = long.groupby("value")["diff"].agg(["mean", "sem"]).reset_index()

    fig, ax = plt.subplots(figsize=(7.5, 3.2), constrained_layout=True)
    for sub_id, sub_df in long.groupby("subject"):
        sub_df = sub_df.sort_values("value")
        ax.plot(sub_df["value"], sub_df["diff"], color="0.7", lw=0.7, zorder=1)
    ax.plot(grp["value"], grp["mean"], color="#3B5BA5", lw=1.8, zorder=3,
            label="Mean")
    ax.fill_between(grp["value"], grp["mean"] - grp["sem"],
                    grp["mean"] + grp["sem"],
                    color="#3B5BA5", alpha=0.22, linewidth=0, zorder=2)
    ax.axhline(0, color="0.7", lw=0.5, ls="--", zorder=0)
    ax.set_xlabel("Value (CHF)")
    ax.set_ylabel("FI(CDF) − FI(InvCDF)")
    ax.set_title("Per-value FI difference between conditions", fontsize=10, color="0.2")
    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def run(subjects, nvoxels, mask, out):
    if subjects is None:
        subjects = discover_subjects(nvoxels, mask)
    if not subjects:
        raise SystemExit(f"No FI TSVs found for nvoxels={nvoxels}, mask={mask}")
    print(f"Subjects: {subjects}  ({len(subjects)} with both sessions)")

    df = collect(subjects, nvoxels, mask)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        page_group(df, subjects, pdf)
        page_per_subject(df, subjects, pdf)
        page_difference(df, subjects, pdf)
    tsv = out.with_suffix(".tsv")
    df.to_csv(tsv, sep="\t", index=False)
    print(f"Wrote {out}\nSidecar: {tsv}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--nvoxels", type=int, default=100)
    p.add_argument("--mask", default="NPCr")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, args.nvoxels, args.mask, Path(args.out))


if __name__ == "__main__":
    main()
