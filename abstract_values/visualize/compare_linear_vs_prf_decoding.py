"""Matched comparison: linear (ramp) vs log-Gaussian PRF (bump) value decoder.

Both decoders run leave-one-run-out with the same noise model (spherical),
lambda (0.1), and voxel-selection sweep. Only the encoding family differs:

  PRF (loggauss) : 1 free LogGaussianPRF per voxel (4 params) — derivatives/decoding/value
  Linear         : signed slope + baseline (2 params, closed-form) — derivatives/decoding/value-linear

Compared across the full n_voxels sweep {0 (nested cvR²>0), 50, 100, 250, 500}
on both BensonV1 ("V1") and NPCr, and under two priors over the stimulus grid:

  uniform : flat prior (decode_value.py's default — the saved pars.tsv store
            the *unnormalized* likelihood, so this is just row-sum
            normalization, no reweighting).
  matched : prior = each subject's own empirical CHF-value distribution
            (Gaussian KDE over their true_value_chf trials, pooled across
            all runs/sessions), evaluated on the stimulus grid. No new
            decode runs needed — this is a post-hoc reweighting of the
            already-saved raw likelihoods.

One page, 2x2: rows = prior (uniform, matched), columns = ROI (V1, NPCr);
per-subject swarm with a group mean ± SEM diamond, across the n_voxels sweep.

Metric: within-run Pearson r between true CHF value and posterior-mean
decoded value, averaged across runs per subject (matches
check_voxel_count_sweep.metric_per_run's aggregation, generalized here to
support the prior reweighting).

Usage:
    python -m abstract_values.visualize.compare_linear_vs_prf_decoding
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
from scipy.stats import gaussian_kde

from abstract_values.utils.data import BIDS_FOLDER
from abstract_values.visualize.check_voxel_count_sweep import _nvox_label, _sort_nvox

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
               / "compare_linear_vs_prf_decoding.pdf")

MODELS = [("PRF", "value"), ("Linear", "value-linear")]
MODEL_COLOUR = {"PRF": "#3B5BA5", "Linear": "#C97B2E"}   # blue / amber — colourblind-safe
MASKS = [("BensonV1", "V1"), ("NPCr", "NPCr"), ("vmPFCOFC", "vmPFC+OFC")]
NVOX_GRID = ["0", "50", "100", "250", "500"]
PRIORS = ["uniform", "matched"]


def _decode_metric(df: pd.DataFrame, prior: str) -> float:
    """Within-run Pearson r (true vs. posterior-mean decoded CHF value),
    averaged across runs — same aggregation as
    check_voxel_count_sweep.metric_per_run, generalized with a prior over
    the stimulus grid. ``df`` has a 'true_value_chf' column plus one
    column per stimulus-grid value holding the raw (unnormalized)
    likelihood decode_value.py saved (normalize=False)."""
    truth_col = "true_value_chf"
    grid = np.asarray(df.columns.drop(truth_col), dtype=np.float64)
    truth = df[truth_col].to_numpy(np.float64)
    lik = df.drop(columns=truth_col).to_numpy(np.float64)

    if prior == "matched":
        prior_w = gaussian_kde(truth)(grid)
        lik = lik * prior_w[None, :]
    elif prior != "uniform":
        raise ValueError(f"Unknown prior {prior!r}")

    post = lik / lik.sum(axis=1, keepdims=True)
    decoded = post @ grid

    rs = []
    for (_, _), idx in df.groupby(level=["session", "run"]).indices.items():
        if len(idx) < 3:
            continue
        t, p = truth[idx], decoded[idx]
        if t.std() == 0 or p.std() == 0:
            continue
        rs.append(float(np.corrcoef(t, p)[0, 1]))
    return float(np.nanmean(rs)) if rs else float("nan")


def load_pars(subdir: str, subject: str, mask: str, nv: str, lambd: float
              ) -> pd.DataFrame | None:
    lam_tag = f"_lambda-{lambd}" if lambd != 0.0 else ""
    fn = (DERIV / subdir / f"sub-{subject}" / "func"
          / f"sub-{subject}_mask-{mask}_nvoxels-{nv}_noise-spherical{lam_tag}_pars.tsv")
    if not fn.exists():
        return None
    return pd.read_csv(fn, sep="\t", index_col=[0, 1, 2, 3])


def discover_subjects() -> list[str]:
    subs = set()
    for _, subdir in MODELS:
        d = DERIV / subdir
        if not d.exists():
            continue
        for p in d.glob("sub-*"):
            subs.add(p.name.removeprefix("sub-"))
    return sorted(subs)


def collect(subjects: list[str], lambd: float) -> pd.DataFrame:
    rows = []
    for sub in subjects:
        for model_label, subdir in MODELS:
            for mask, mask_label in MASKS:
                for nv in NVOX_GRID:
                    df = load_pars(subdir, sub, mask, nv, lambd)
                    if df is None:
                        continue
                    df2 = df.reset_index(level=3)  # true_value_chf -> column
                    for prior in PRIORS:
                        r = _decode_metric(df2, prior)
                        rows.append(dict(subject=sub, model=model_label, mask=mask_label,
                                         nvoxels=nv, prior=prior, r=r))
    return pd.DataFrame(rows)


PRIOR_LABEL = {"uniform": "Uniform prior", "matched": "Matched prior (subject's own value distribution)"}


def page_swarm(df: pd.DataFrame, pdf: PdfPages):
    order = _sort_nvox(NVOX_GRID)
    fig, axes = plt.subplots(len(PRIORS), len(MASKS), figsize=(9.8, 6.8),
                             constrained_layout=True, sharey=True)
    rng = np.random.default_rng(0)
    dx = 0.17

    for row, prior in enumerate(PRIORS):
        for col, (_, mask_label) in enumerate(MASKS):
            ax = axes[row, col]
            for i, nv in enumerate(order):
                for model, off in [("PRF", -dx), ("Linear", dx)]:
                    vals = df[(df.model == model) & (df["mask"] == mask_label)
                              & (df.nvoxels == nv) & (df.prior == prior)]["r"].dropna().to_numpy()
                    if len(vals) == 0:
                        continue
                    x = i + off + rng.uniform(-0.045, 0.045, size=len(vals))
                    ax.scatter(x, vals, s=13, color=MODEL_COLOUR[model], alpha=0.5,
                              linewidth=0, zorder=2)
                    mean = np.nanmean(vals)
                    sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                    ax.errorbar([i + off], [mean], yerr=sem, fmt="D",
                              markersize=6.5, markerfacecolor=MODEL_COLOUR[model],
                              markeredgecolor="black", markeredgewidth=1.3,
                              ecolor=MODEL_COLOUR[model], elinewidth=1.3, capsize=0,
                              zorder=5)
            ax.axhline(0, color="0.7", lw=0.6, ls="--", zorder=0)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([_nvox_label(nv) for nv in order], fontsize=7.5)
            if row == len(PRIORS) - 1:
                ax.set_xlabel("Voxels selected")
            if row == 0:
                ax.set_title(mask_label, fontsize=10, color="0.2")
            if col == 0:
                ax.set_ylabel(f"{PRIOR_LABEL[prior]}\nDecoding accuracy (Pearson r)", fontsize=8.5)

    axes[0, 0].text(0.02, 0.97, "PRF", transform=axes[0, 0].transAxes,
                    color=MODEL_COLOUR["PRF"], fontsize=9, fontweight="medium", va="top")
    axes[0, 0].text(0.02, 0.895, "Linear", transform=axes[0, 0].transAxes,
                    color=MODEL_COLOUR["Linear"], fontsize=9, fontweight="medium", va="top")
    fig.suptitle("Value decoding accuracy vs. voxel-selection criterion  "
                f"(spherical noise, λ=0.1, unsmoothed, n_subjects={df.subject.nunique()})",
                fontsize=9.5, y=1.02)
    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def run(subjects: list[str] | None, lambd: float, out: Path):
    if subjects is None:
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No decode TSVs found under derivatives/decoding/{value,value-linear}.")
    print(f"Subjects: {subjects}")

    df = collect(subjects, lambd)
    if df.empty:
        raise SystemExit("No matching (model, mask, nvoxels, subject) combinations.")
    print(df.groupby(["prior", "mask", "nvoxels", "model"])["r"]
          .agg(["mean", "median", "count"]).round(3))

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
    p.add_argument("--lambd", type=float, default=0.1)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, args.lambd, Path(args.out))


if __name__ == "__main__":
    main()
