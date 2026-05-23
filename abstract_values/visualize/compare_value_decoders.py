"""Matched comparison: per-voxel parametric vs basis-set value decoder.

Both decoders run leave-one-run-out with the same noise model + voxel
selection criteria. Only the encoding family differs:

  loggauss : 1 free LogGaussianPRF per voxel  (4 params, non-convex fit)
  weighted : 8 fixed log-Gaussian basis RFs + WeightFitter (8 closed-form
             weights per voxel) — value-axis analog of vonmises gabor decoder

We compare across {nv=50, nv=100, fdr05, psig95} on NPCr.

Three pages:
  1. Per-subject paired strip: loggauss vs weighted at each criterion.
  2. Group mean ± SEM by (criterion × decoder).
  3. Per-criterion swarm with paired lines.

Metric: within-run Pearson r between true CHF and posterior-mean
decoded CHF, averaged across runs per subject.
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

from abstract_values.utils.data import BIDS_FOLDER

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

DERIV = Path(BIDS_FOLDER) / "derivatives" / "decoding"
DEFAULT_OUT = Path(BIDS_FOLDER) / "derivatives" / "qa" / "compare_value_decoders.pdf"

CRITERIA = ["50", "100", "fdr05", "psig95"]
DECODERS = [
    ("loggauss", "value"),           # subdir
    ("weighted", "value-weighted"),
]
DECODER_COLOURS = {
    "loggauss": "#C44E52",  # per-voxel parametric — coral
    "weighted": "#2A9D8F",  # basis-set — teal (matches vonmises in V1)
}


def _crit_label(c: str) -> str:
    return {"50": "n=50", "100": "n=100",
            "fdr05": "FDR≤0.05", "psig95": "P(sig)≥0.95"}[c]


def load_pars(subdir: str, subject: str, crit: str, lambd: float
              ) -> pd.DataFrame | None:
    lam_tag = f"_lambda-{lambd}" if lambd != 0.0 else ""
    fn = (DERIV / subdir / f"sub-{subject}" / "func"
          / f"sub-{subject}_mask-NPCr_nvoxels-{crit}"
            f"_noise-full{lam_tag}_pars.tsv")
    if not fn.exists():
        return None
    return pd.read_csv(fn, sep="\t", index_col=[0, 1, 2])


def within_run_pearson(df: pd.DataFrame) -> float:
    truth = df["true_value_chf"].to_numpy(np.float32)
    grid = np.asarray(df.columns.drop("true_value_chf"), dtype=np.float32)
    posteriors = df.drop(columns="true_value_chf").to_numpy(np.float32)
    posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)
    decoded = (posteriors * grid[None, :]).sum(axis=1)
    rs = []
    for (_, _), idx in df.groupby(level=["session", "run"]).indices.items():
        if len(idx) < 3:
            continue
        t, p = truth[idx], decoded[idx]
        if t.std() == 0 or p.std() == 0:
            continue
        rs.append(float(np.corrcoef(t, p)[0, 1]))
    return float(np.nanmean(rs)) if rs else float("nan")


def discover_subjects() -> list[str]:
    subs = set()
    for _, sub in DECODERS:
        if not (DERIV / sub).exists():
            continue
        for p in (DERIV / sub).glob("sub-*"):
            subs.add(p.name.removeprefix("sub-"))
    return sorted(subs)


def collect(subjects: list[str], lambd: float) -> pd.DataFrame:
    rows = []
    for sub in subjects:
        for dec_label, subdir in DECODERS:
            for crit in CRITERIA:
                df = load_pars(subdir, sub, crit, lambd)
                if df is None:
                    continue
                rows.append({
                    "subject": sub, "decoder": dec_label, "criterion": crit,
                    "r": within_run_pearson(df),
                })
    return pd.DataFrame(rows)


def page_paired(df: pd.DataFrame, pdf: PdfPages):
    """Per-subject paired strip with horizontal connector lines."""
    subjects = sorted(df["subject"].unique(),
                      key=lambda s: (0 if s[0].isdigit() else 1, s))
    if not subjects:
        return

    fig, ax = plt.subplots(figsize=(7.5, 3.4), constrained_layout=True)

    # x positions: criterion group × decoder offset
    x_base = {c: i for i, c in enumerate(CRITERIA)}
    dx = 0.18
    offsets = {"loggauss": -dx, "weighted": +dx}

    for sub in subjects:
        # Connector lines: same subject, same criterion, two decoders
        for crit in CRITERIA:
            row_lg = df[(df.subject == sub) & (df.decoder == "loggauss")
                        & (df.criterion == crit)]
            row_wt = df[(df.subject == sub) & (df.decoder == "weighted")
                        & (df.criterion == crit)]
            if row_lg.empty or row_wt.empty:
                continue
            ax.plot([x_base[crit] + offsets["loggauss"],
                     x_base[crit] + offsets["weighted"]],
                    [row_lg["r"].iloc[0], row_wt["r"].iloc[0]],
                    "-", color="0.85", lw=0.7, zorder=1)

    for dec, colour in DECODER_COLOURS.items():
        sub_df = df[df.decoder == dec]
        for _, row in sub_df.iterrows():
            ax.scatter(x_base[row["criterion"]] + offsets[dec], row["r"],
                       s=22, color=colour, alpha=0.8, edgecolor="white",
                       linewidth=0.5, zorder=3)

    # Group mean ± SEM as larger diamond
    grp = df.groupby(["decoder", "criterion"])["r"].agg(["mean", "sem", "count"])
    for (dec, crit), row in grp.iterrows():
        ax.errorbar(x_base[crit] + offsets[dec], row["mean"], yerr=row["sem"],
                    fmt="D", color=DECODER_COLOURS[dec], mec="black", mew=1.3,
                    markersize=8, elinewidth=1.1, capsize=2.5, zorder=5)

    ax.set_xticks(list(x_base.values()))
    ax.set_xticklabels([_crit_label(c) for c in CRITERIA])
    ax.set_xlabel("Voxel-selection criterion")
    ax.set_ylabel("Within-run Pearson r")
    ax.axhline(0, color="0.7", lw=0.6, ls="--", zorder=0)
    ax.set_title("NPCr value decoding: per-voxel parametric vs basis-set",
                 fontsize=10, color="0.2")

    # In-panel direct labels (no legend frame)
    ax.text(0.02, 0.96, "● loggauss (per-voxel)", transform=ax.transAxes,
            color=DECODER_COLOURS["loggauss"], fontsize=8.5, va="top")
    ax.text(0.02, 0.91, "● weighted (basis-set)", transform=ax.transAxes,
            color=DECODER_COLOURS["weighted"], fontsize=8.5, va="top")
    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_summary(df: pd.DataFrame, pdf: PdfPages):
    """Group mean ± SEM by (criterion × decoder), bar form."""
    if df.empty:
        return
    grp = df.groupby(["decoder", "criterion"])["r"].agg(["mean", "sem", "count"])

    fig, ax = plt.subplots(figsize=(7.5, 3.0), constrained_layout=True)
    width = 0.36
    xs = np.arange(len(CRITERIA))
    for i, dec in enumerate(["loggauss", "weighted"]):
        means = [grp.loc[(dec, c), "mean"] if (dec, c) in grp.index else np.nan
                 for c in CRITERIA]
        sems = [grp.loc[(dec, c), "sem"] if (dec, c) in grp.index else 0
                for c in CRITERIA]
        ax.bar(xs + (i - 0.5) * width, means, width,
               color=DECODER_COLOURS[dec], alpha=0.85, edgecolor="white",
               linewidth=0.7, label=dec)
        ax.errorbar(xs + (i - 0.5) * width, means, yerr=sems,
                    fmt="none", ecolor="0.2", elinewidth=0.9, capsize=2.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([_crit_label(c) for c in CRITERIA])
    ax.set_xlabel("Voxel-selection criterion")
    ax.set_ylabel("Group mean Pearson r ± SEM")
    ax.axhline(0, color="0.7", lw=0.6, ls="--", zorder=0)
    ax.set_title("NPCr value decoding: group means", fontsize=10, color="0.2")

    # In-panel labels
    ax.text(0.02, 0.96, "■ loggauss", transform=ax.transAxes,
            color=DECODER_COLOURS["loggauss"], fontsize=8.5, va="top")
    ax.text(0.02, 0.91, "■ weighted", transform=ax.transAxes,
            color=DECODER_COLOURS["weighted"], fontsize=8.5, va="top")
    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_swarm_paired(df: pd.DataFrame, pdf: PdfPages):
    """Per-criterion: swarm of subjects with paired connector lines."""
    if df.empty:
        return
    fig, axes = plt.subplots(1, len(CRITERIA), figsize=(8.0, 3.0),
                             constrained_layout=True, sharey=True)
    for ax, crit in zip(axes, CRITERIA):
        sub_df = df[df.criterion == crit]
        # Pair lines
        for sub in sub_df.subject.unique():
            row_lg = sub_df[(sub_df.subject == sub) & (sub_df.decoder == "loggauss")]
            row_wt = sub_df[(sub_df.subject == sub) & (sub_df.decoder == "weighted")]
            if row_lg.empty or row_wt.empty:
                continue
            ax.plot([0, 1], [row_lg["r"].iloc[0], row_wt["r"].iloc[0]],
                    "-", color="0.8", lw=0.7, zorder=1)
        for i, dec in enumerate(["loggauss", "weighted"]):
            vals = sub_df[sub_df.decoder == dec]["r"].values
            ax.scatter([i] * len(vals), vals, s=22, alpha=0.8,
                       color=DECODER_COLOURS[dec],
                       edgecolor="white", linewidth=0.5, zorder=3)
            if len(vals) > 0:
                ax.scatter(i, np.nanmean(vals), s=120, marker="D",
                           facecolor=DECODER_COLOURS[dec],
                           edgecolor="black", linewidth=1.4, zorder=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["loggauss", "weighted"], fontsize=8,
                           rotation=15, ha="right")
        ax.set_title(_crit_label(crit), fontsize=9, color="0.2")
        ax.axhline(0, color="0.7", lw=0.6, ls="--", zorder=0)
    axes[0].set_ylabel("Within-run Pearson r")
    fig.suptitle("Paired subject comparison per criterion",
                 fontsize=10, y=1.02)
    sns.despine(fig=fig, offset=4, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def run(subjects: list[str] | None, lambd: float, out: Path):
    if subjects is None:
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No decode TSVs found.")
    print(f"Subjects: {subjects}")

    df = collect(subjects, lambd)
    if df.empty:
        raise SystemExit("No matching (decoder, criterion, subject) combinations.")
    print(df.groupby(["decoder", "criterion"])["r"]
          .agg(["mean", "sem", "count"]).round(3))

    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        page_paired(df, pdf)
        page_summary(df, pdf)
        page_swarm_paired(df, pdf)
    tsv = out.with_suffix(".tsv")
    df.to_csv(tsv, sep="\t", index=False)
    print(f"Wrote {out}\nSidecar: {tsv}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+",
                   help="Subject labels (default: discover from decode TSVs)")
    p.add_argument("--lambd", type=float, default=0.1)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, args.lambd, Path(args.out))


if __name__ == "__main__":
    main()
