"""Full-data (non-cross-validated) decoding accuracy: loggauss vs linear vs
session-shift (only the preferred value shifts between sessions).

Uses decode_value_fullfit.py's output — the encoding model is fit on ALL
trials (no LORO), one noise model is fit on all trials, and every trial is
decoded by that single fixed, full-data model. This is circular/optimistic
by construction (every trial's own run contributed to the model decoding
it) — a ceiling/best-case comparison, not a generalization estimate (see
compare_linear_vs_prf_decoding.py for the properly cross-validated version).

n_voxels=100 (top-N by the full-fit model's own R²), spherical noise,
lambda=0, unsmoothed, BensonV1 + NPCr.

Metric: within-run Pearson r (true vs. posterior-mean decoded CHF value),
averaged across runs per subject.

Usage:
    python -m abstract_values.visualize.compare_fullfit_decoding
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
from abstract_values.visualize.check_voxel_count_sweep import metric_per_run

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

DERIV = Path(BIDS_FOLDER) / "derivatives" / "decoding" / "value-fullfit"
DEFAULT_OUT = (Path(__file__).resolve().parents[2] / "notes" / "figures"
               / "compare_fullfit_decoding.pdf")

MODELS = ["loggauss", "linear", "session-shift"]
MODEL_LABEL = {"loggauss": "PRF", "linear": "Linear", "session-shift": "Session-shift\n(mu only)"}
MODEL_COLOUR = {"loggauss": "#3B5BA5", "linear": "#C97B2E", "session-shift": "#5D8C3F"}
MASKS = [("BensonV1", "V1"), ("NPCr", "NPCr")]
N_VOXELS = "100"


def load_pars(model: str, subject: str, mask: str) -> pd.DataFrame | None:
    fn = (DERIV / model / f"sub-{subject}" / "func"
          / f"sub-{subject}_mask-{mask}_nvoxels-{N_VOXELS}_noise-spherical_pars.tsv")
    if not fn.exists():
        return None
    return pd.read_csv(fn, sep="\t", index_col=[0, 1, 2, 3])


def discover_subjects() -> list[str]:
    subs = set()
    for model in MODELS:
        d = DERIV / model
        if not d.exists():
            continue
        for p in d.glob("sub-*"):
            subs.add(p.name.removeprefix("sub-"))
    return sorted(subs)


def collect(subjects: list[str]) -> pd.DataFrame:
    rows = []
    for sub in subjects:
        for model in MODELS:
            for mask, mask_label in MASKS:
                df = load_pars(model, sub, mask)
                if df is None:
                    continue
                df2 = df.reset_index(level=3)
                r = metric_per_run(df2, "value")
                rows.append(dict(subject=sub, model=model, mask=mask_label, r=r))
    return pd.DataFrame(rows)


def page_swarm(df: pd.DataFrame, pdf: PdfPages):
    fig, axes = plt.subplots(1, len(MASKS), figsize=(6.4, 3.8),
                             constrained_layout=True, sharey=True)
    rng = np.random.default_rng(0)

    for ax, (_, mask_label) in zip(axes, MASKS):
        for i, model in enumerate(MODELS):
            vals = df[(df.model == model) & (df["mask"] == mask_label)]["r"].dropna().to_numpy()
            if len(vals) == 0:
                continue
            x = i + rng.uniform(-0.09, 0.09, size=len(vals))
            ax.scatter(x, vals, s=15, color=MODEL_COLOUR[model], alpha=0.5,
                      linewidth=0, zorder=2)
            mean = np.nanmean(vals)
            sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            ax.errorbar([i], [mean], yerr=sem, fmt="D",
                      markersize=7, markerfacecolor=MODEL_COLOUR[model],
                      markeredgecolor="black", markeredgewidth=1.3,
                      ecolor=MODEL_COLOUR[model], elinewidth=1.3, capsize=0,
                      zorder=5)
        ax.axhline(0, color="0.7", lw=0.6, ls="--", zorder=0)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], fontsize=8)
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        ax.set_title(mask_label, fontsize=10, color="0.2")

    axes[0].set_ylabel("Decoding accuracy (Pearson r)\nfull-data fit, all trials decoded")
    fig.suptitle("Full-data (non-CV) decoding accuracy: PRF vs. Linear vs. session-shift  "
                f"(n_voxels=100, spherical noise, unsmoothed, n_subjects={df.subject.nunique()})",
                fontsize=8.8, y=1.05)
    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def run(subjects: list[str] | None, out: Path):
    if subjects is None:
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No decode TSVs found under derivatives/decoding/value-fullfit.")
    print(f"Subjects: {subjects}")

    df = collect(subjects)
    if df.empty:
        raise SystemExit("No matching (model, mask, subject) combinations.")
    print(df.groupby(["mask", "model"])["r"].agg(["mean", "median", "count"]).round(3))

    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        page_swarm(df, pdf)
    tsv = out.with_suffix(".tsv")
    df.to_csv(tsv, sep="\t", index=False)
    print(f"\nWrote {out}\nSidecar: {tsv}")


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
