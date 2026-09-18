"""Decoding error by condition, and the aggregate decoded posterior, for the
general (standard leave-one-run-out, cross-validated) value decoder.

"General" = the regular decode_value.py LORO output — PRF (loggauss) and
Linear, NOT the full-data or session-shift variants (see
compare_fullfit_decoding.py for those). n_voxels=100, spherical noise,
lambda=0.1, unsmoothed, BensonV1 + NPCr.

Condition = the counterbalanced orientation->value mapping ('cdf' vs
'inverse_cdf'), alternating by subject parity and session
(Subject.get_mapping) — the two experimental conditions.

Pages:
  1. Decoding error (mean |decoded - true| CHF) per subject, split by
     model x condition, per ROI.
  2. Aggregate decoded posterior: each trial's posterior circularly
     aligned to its true stimulus (deviation from truth on x), then
     averaged across trials/subjects — one curve per model x condition,
     per ROI. Shows whether the decoder produces a sensible unimodal
     peak at zero deviation, and whether the two conditions differ.

Usage:
    python -m abstract_values.visualize.decoding_error_and_posterior
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

DERIV = Path(BIDS_FOLDER) / "derivatives" / "decoding"
DEFAULT_OUT = (Path(__file__).resolve().parents[2] / "notes" / "figures"
               / "decoding_error_and_posterior.pdf")

MODELS = [("PRF", "value"), ("Linear", "value-linear")]
MODEL_COLOUR = {"PRF": "#3B5BA5", "Linear": "#C97B2E"}
MASKS = [("BensonV1", "V1"), ("NPCr", "NPCr")]
CONDITIONS = ["cdf", "inverse_cdf"]
COND_LINESTYLE = {"cdf": "-", "inverse_cdf": "--"}
N_VOXELS = "100"
LAMBD = 0.1


def load_pars(subdir: str, subject: str, mask: str) -> pd.DataFrame | None:
    fn = (DERIV / subdir / f"sub-{subject}" / "func"
          / f"sub-{subject}_mask-{mask}_nvoxels-{N_VOXELS}_noise-spherical"
            f"_lambda-{LAMBD}_pars.tsv")
    if not fn.exists():
        return None
    return pd.read_csv(fn, sep="\t", index_col=[0, 1, 2])


def discover_subjects() -> list[str]:
    subs = set()
    for _, subdir in MODELS:
        d = DERIV / subdir
        if not d.exists():
            continue
        for p in d.glob("sub-*"):
            subs.add(p.name.removeprefix("sub-"))
    return sorted(subs)


def decorate(df: pd.DataFrame, sub: Subject) -> pd.DataFrame:
    """Add decoded_mean/error/condition columns; return per-trial DataFrame."""
    grid_cols = df.columns.drop("true_value_chf")
    grid = grid_cols.values.astype(np.float64)
    probs = df[grid_cols].to_numpy(np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    decoded = probs @ grid

    out = df[["true_value_chf"]].copy()
    out["decoded_mean"] = decoded
    out["error"] = out["decoded_mean"] - out["true_value_chf"]
    sessions = out.index.get_level_values("session")
    out["condition"] = [sub.get_mapping(int(s)) for s in sessions]
    out["probs"] = list(probs)
    out["grid"] = [grid] * len(out)
    return out


def collect(subjects: list[str]) -> pd.DataFrame:
    rows = []
    for subject in subjects:
        sub = Subject(subject, bids_folder=Path(BIDS_FOLDER))
        for model_label, subdir in MODELS:
            for mask, mask_label in MASKS:
                df = load_pars(subdir, subject, mask)
                if df is None:
                    continue
                dec = decorate(df, sub)
                dec["subject"] = subject
                dec["model"] = model_label
                dec["mask"] = mask_label
                rows.append(dec)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=False)


def page_error(trials: pd.DataFrame, pdf: PdfPages):
    per_sub = (trials.groupby(["subject", "model", "mask", "condition"])["error"]
              .apply(lambda e: e.abs().mean()).reset_index(name="mae"))

    fig, axes = plt.subplots(1, len(MASKS), figsize=(6.6, 3.8),
                             constrained_layout=True, sharey=True)
    rng = np.random.default_rng(0)
    dx = 0.17

    for ax, (_, mask_label) in zip(axes, MASKS):
        for i, cond in enumerate(CONDITIONS):
            for model, off in [("PRF", -dx), ("Linear", dx)]:
                vals = per_sub[(per_sub.model == model) & (per_sub["mask"] == mask_label)
                               & (per_sub.condition == cond)]["mae"].dropna().to_numpy()
                if len(vals) == 0:
                    continue
                x = i + off + rng.uniform(-0.045, 0.045, size=len(vals))
                ax.scatter(x, vals, s=15, color=MODEL_COLOUR[model], alpha=0.5,
                          linewidth=0, zorder=2)
                mean = np.nanmean(vals)
                sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                ax.errorbar([i + off], [mean], yerr=sem, fmt="D",
                          markersize=7, markerfacecolor=MODEL_COLOUR[model],
                          markeredgecolor="black", markeredgewidth=1.3,
                          ecolor=MODEL_COLOUR[model], elinewidth=1.3, capsize=0,
                          zorder=5)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(CONDITIONS)
        ax.set_xlabel("Condition")
        ax.set_title(mask_label, fontsize=10, color="0.2")

    axes[0].set_ylabel("Decoding error\nmean |decoded − true| (CHF)")
    axes[0].text(0.02, 0.97, "PRF", transform=axes[0].transAxes,
                color=MODEL_COLOUR["PRF"], fontsize=9, fontweight="medium", va="top")
    axes[0].text(0.02, 0.895, "Linear", transform=axes[0].transAxes,
                color=MODEL_COLOUR["Linear"], fontsize=9, fontweight="medium", va="top")
    fig.suptitle("Decoding error by condition  (general LORO decoder, "
                f"n_voxels={N_VOXELS}, λ={LAMBD}, n_subjects={per_sub.subject.nunique()})",
                fontsize=9.2, y=1.05)
    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _aggregate_posterior(sub_df: pd.DataFrame):
    """Circularly align each trial's posterior to its true stimulus
    (deviation from truth on x), average across trials. Mirrors
    analyze_decoding.ipynb's aggregate_posterior."""
    grid = sub_df["grid"].iloc[0]
    n = len(grid)
    aligned = np.zeros((len(sub_df), n))
    for i, (probs, tv) in enumerate(zip(sub_df["probs"], sub_df["true_value_chf"])):
        ci = int(np.argmin(np.abs(grid - tv)))
        aligned[i] = np.roll(probs, n // 2 - ci)
    step = grid[1] - grid[0]
    deviation = (np.arange(n) - n // 2) * step
    return deviation, aligned.mean(axis=0)


def page_posterior(trials: pd.DataFrame, pdf: PdfPages):
    fig, axes = plt.subplots(1, len(MASKS), figsize=(6.6, 3.8),
                             constrained_layout=True, sharey=True)

    for ax, (_, mask_label) in zip(axes, MASKS):
        for model, _ in MODELS:
            for cond in CONDITIONS:
                sub_df = trials[(trials.model == model) & (trials["mask"] == mask_label)
                                & (trials.condition == cond)]
                if sub_df.empty:
                    continue
                dev, post = _aggregate_posterior(sub_df)
                ax.plot(dev, post, color=MODEL_COLOUR[model],
                       linestyle=COND_LINESTYLE[cond], lw=1.4,
                       label=f"{model} · {cond}")
        ax.axvline(0, color="0.7", lw=0.6, ls=":", zorder=0)
        ax.set_xlabel("Deviation from true value (CHF)")
        ax.set_title(mask_label, fontsize=10, color="0.2")

    axes[0].set_ylabel("Mean posterior\n(trial-aligned, unnormalized)")
    axes[1].legend(loc="upper right", fontsize=6.5, frameon=False)
    fig.suptitle("Aggregate decoded posterior, aligned to truth  "
                "(general LORO decoder, n_voxels=100)",
                fontsize=9.2, y=1.05)
    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def run(subjects: list[str] | None, out: Path):
    if subjects is None:
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No decode TSVs found under derivatives/decoding/{value,value-linear}.")
    print(f"Subjects: {subjects}")

    trials = collect(subjects)
    if trials.empty:
        raise SystemExit("No matching (model, mask, subject) combinations.")

    per_sub_mae = (trials.groupby(["subject", "model", "mask", "condition"])["error"]
                   .apply(lambda e: e.abs().mean()))
    print(per_sub_mae.groupby(["mask", "model", "condition"]).agg(["mean", "count"]).round(3))

    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        page_error(trials, pdf)
        page_posterior(trials, pdf)
    tsv = out.with_suffix(".tsv")
    trials.drop(columns=["probs", "grid"]).to_csv(tsv, sep="\t")
    print(f"\nWrote {out}\nSidecar (trial-level, no posterior arrays): {tsv}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+",
                   help="Subject labels (default: discover from decode TSVs)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, Path(args.out))


if __name__ == "__main__":
    main()
