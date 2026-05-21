"""Compare R²-mixture voxel-selection thresholds: FDR α vs p_signal cuts.

For each subject (and model) with a cached whole-brain mixture from
``compute_r2_mixture.py``, show the R² distribution + mixture overlay
and three competing thresholds:

    1. ``FDR ≤ α`` (default α=0.05) — tail-FDR-controlled cut.
    2. ``P(signal | R²) ≥ 0.5`` — voxels more likely signal than noise.
    3. ``P(signal | R²) ≥ 0.95`` — strict posterior cut (~ "high-confidence
       signal voxels"). Yields fewer voxels but lower expected FP rate.

Page layout:
    - One per-subject page: R² histogram + mixture overlay + 3 vertical
      threshold lines, with the voxel count above each.
    - One group summary page: per-subject voxel counts under each
      criterion (paired lines connect a subject across criteria).

Usage:
    python -m abstract_values.visualize.check_r2_mixture
    python -m abstract_values.visualize.check_r2_mixture --model vonmises
    python -m abstract_values.visualize.check_r2_mixture --alpha 0.01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

from abstract_values.utils.data import BIDS_FOLDER
from braincoder.utils.stats import (
    r2_fdr_threshold as r2_fdr_threshold_from_fit,
    r2_posterior_signal,
)

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

# Hand-picked palette for the three criteria.
CRITERIA = [
    ("fdr",  "FDR ≤ α",        "#3B5BA5"),   # blue
    ("p50",  "P(signal) ≥ 0.5",  "#5D8C3F"),   # green
    ("p95",  "P(signal) ≥ 0.95", "#C44E52"),   # red
]


def _inv_logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(r):
    return np.log(r / (1.0 - r))


def r2_at_p_signal(fit: dict, target_p: float, n_grid: int = 4000) -> float:
    """Smallest R² at which posterior P(signal | R²) ≥ target_p."""
    z = np.linspace(fit["noise_mu"] - 5 * fit["noise_sigma"],
                    fit["signal_mu"] + 6 * fit["signal_sigma"], n_grid)
    r2 = _inv_logit(z)
    p = r2_posterior_signal(r2, fit)
    hits = np.where(p >= target_p)[0]
    return float(r2[hits[0]]) if len(hits) else float("inf")


def _load_mixture(subject: str, model: str, bids_folder: Path,
                   smoothed: bool = False) -> dict | None:
    smtag = "_smoothed" if smoothed else ""
    json_fn = (bids_folder / "derivatives" / "encoding_models" / model
               / f"sub-{subject}" / f"sub-{subject}_desc-p_signal{smtag}.json")
    if not json_fn.exists():
        return None
    with open(json_fn) as fh:
        return json.load(fh)


def _load_r2(subject: str, model: str, bids_folder: Path,
              smoothed: bool = False) -> np.ndarray | None:
    smtag = "_smoothed" if smoothed else ""
    nii = (bids_folder / "derivatives" / "encoding_models" / model
           / f"sub-{subject}" / "func"
           / f"sub-{subject}_task-abstractvalue_space-T1w_desc-r2{smtag}_pe.nii.gz")
    if not nii.exists():
        return None
    return nib.load(str(nii)).get_fdata().astype(np.float32).flatten()


def _mixture_pdf_in_logit(z, fit):
    """Two component densities on the logit-R² axis (in mixture-weight scale)."""
    n = norm.pdf(z, fit["noise_mu"], fit["noise_sigma"]) * fit["noise_weight"]
    s = norm.pdf(z, fit["signal_mu"], fit["signal_sigma"]) * fit["signal_weight"]
    return n, s


def plot_subject_page(subject: str, model: str, sidecar: dict,
                       r2_all: np.ndarray, alpha: float,
                       x_max: float | None = None,
                       voxel_count_max: int | None = None):
    fit = sidecar["BRAIN"]
    r2 = r2_all[np.isfinite(r2_all) & (r2_all > 0) & (r2_all < 0.99)]

    fig = plt.figure(figsize=(7.25, 3.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0])
    ax_h = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    fig.suptitle(f"sub-{subject}  ·  {model}  ·  whole-brain R² mixture",
                 fontsize=10, y=1.02)

    # Histogram of R² (linear x; clip at the shared x_max so subject pages
    # are visually comparable). Mixture components are projected back from
    # logit-space via Jacobian so they integrate to the right density.
    p995 = float(np.percentile(r2, 99.5))
    x_lim = x_max if x_max is not None else p995
    x = np.linspace(1e-4, x_lim, 600)
    z_x = _logit(x)
    n_pdf, s_pdf = _mixture_pdf_in_logit(z_x, fit)
    jac = 1.0 / (x * (1.0 - x))               # |dz/dx|
    n_pdf, s_pdf = n_pdf * jac, s_pdf * jac

    ax_h.hist(r2[r2 <= x_lim], bins=80, density=True, color="0.78", lw=0)
    sum_pdf = n_pdf + s_pdf
    ax_h.plot(x, n_pdf, color="0.45", lw=1.0, label="Noise component")
    ax_h.plot(x, s_pdf, color="0.15", lw=1.0, ls="--", label="Signal component")
    ax_h.plot(x, sum_pdf, color="0.0", lw=1.5, label="Mixture (sum)")
    ax_h.set_xlabel("R²")
    ax_h.set_ylabel("Density")
    ax_h.set_xlim(0, x_lim)
    ax_h.set_ylim(bottom=0)

    thr = {
        "fdr": r2_fdr_threshold_from_fit(fit, alpha=alpha),
        "p50": r2_at_p_signal(fit, 0.5),
        "p95": r2_at_p_signal(fit, 0.95),
    }
    counts = {k: int(np.sum(r2_all > t)) if np.isfinite(t) else 0
              for k, t in thr.items()}

    for key, label, color in CRITERIA:
        t = thr[key]
        if not np.isfinite(t):
            continue
        ax_h.axvline(t, color=color, lw=1.2, ls=":", zorder=2)
        ax_h.text(t, ax_h.get_ylim()[1] * 0.97,
                  f" {label}\n R²={t:.3f}\n n={counts[key]}",
                  color=color, fontsize=7, va="top", ha="left")
    ax_h.legend(loc="upper right", frameon=False, fontsize=7)

    # Right panel: bar of voxel counts per criterion
    keys = [k for k, _, _ in CRITERIA]
    labels = [l for _, l, _ in CRITERIA]
    colors = [c for _, _, c in CRITERIA]
    ax_b.bar(range(len(keys)),
             [counts[k] for k in keys],
             color=colors, width=0.6)
    ax_b.set_xticks(range(len(keys)))
    ax_b.set_xticklabels(labels, fontsize=8)
    ax_b.set_ylabel("Voxels above threshold")
    if voxel_count_max is not None:
        ax_b.set_ylim(0, voxel_count_max * 1.1)
    else:
        ax_b.set_ylim(bottom=0)
    for i, k in enumerate(keys):
        ax_b.text(i, counts[k], f"{counts[k]:,}", ha="center", va="bottom",
                  fontsize=7)

    sns.despine(fig=fig, offset=5, trim=True)
    return fig, counts


def plot_thresholds_and_means_page(per_subject_fits: dict[str, dict],
                                    model: str, alpha: float):
    """Per-subject overview: 3 thresholds + signal/noise mean R²s, one
    horizontal swarm per quantity, subjects connected by faint lines.
    """
    if not per_subject_fits:
        return None
    subjects = sorted(per_subject_fits.keys())
    series: dict[str, list[float]] = {
        "Noise μ":         [],
        "Signal μ":        [],
        "FDR ≤ α":         [],
        "P(signal) ≥ 0.5": [],
        "P(signal) ≥ 0.95":[],
    }
    palette = {
        "Noise μ":          "0.55",
        "Signal μ":         "0.15",
        "FDR ≤ α":          "#3B5BA5",
        "P(signal) ≥ 0.5":  "#5D8C3F",
        "P(signal) ≥ 0.95": "#C44E52",
    }
    for sub in subjects:
        info = per_subject_fits[sub]
        series["Noise μ"].append(info["noise_mean_r2"])
        series["Signal μ"].append(info["signal_mean_r2"])
        series["FDR ≤ α"].append(r2_fdr_threshold_from_fit(info, alpha=alpha))
        series["P(signal) ≥ 0.5"].append(r2_at_p_signal(info, 0.5))
        series["P(signal) ≥ 0.95"].append(r2_at_p_signal(info, 0.95))

    keys = list(series.keys())
    n_subj = len(subjects)
    fig, ax = plt.subplots(figsize=(6.5, 3.4), constrained_layout=True)
    fig.suptitle(f"Whole-brain R² mixture summary  ·  {model}  (α={alpha:.2f}, "
                 f"n_subj={n_subj})", fontsize=10, y=1.02)
    x = np.arange(len(keys))
    # Subject-by-subject line connecting all five quantities (light grey)
    for i in range(n_subj):
        ys = [series[k][i] for k in keys]
        ax.plot(x, ys, color="0.85", lw=0.5, zorder=1)
    # Coloured dots per quantity
    for j, k in enumerate(keys):
        ys = [v for v in series[k] if np.isfinite(v)]
        ax.plot([j] * len(ys), ys, "o", color=palette[k], ms=5, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=8)
    ax.set_ylabel("R²")
    ax.set_ylim(bottom=0)
    sns.despine(fig=fig, offset=5, trim=True)
    return fig


def plot_summary_page(per_subject_counts: dict[str, dict[str, int]], model: str,
                       alpha: float):
    fig, ax = plt.subplots(figsize=(5.5, 3.5), constrained_layout=True)
    fig.suptitle(f"Voxel count per threshold criterion  ·  {model}  (α={alpha:.2f})",
                 fontsize=10, y=1.02)

    subjects = sorted(per_subject_counts.keys())
    keys = [k for k, _, _ in CRITERIA]
    colors = {k: c for k, _, c in CRITERIA}
    labels = {k: l for k, l, _ in CRITERIA}
    x = np.arange(len(keys))

    for sub in subjects:
        ys = [per_subject_counts[sub].get(k, 0) for k in keys]
        ax.plot(x, ys, color="0.7", lw=0.6, zorder=1)
        for i, k in enumerate(keys):
            ax.plot(i, ys[i], "o", color=colors[k], ms=4, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in keys])
    ax.set_ylabel("Voxels above threshold (whole brain)")
    ax.set_ylim(bottom=0)
    sns.despine(fig=fig, offset=5, trim=True)
    return fig


def run(model: str, smoothed: bool, alpha: float, out: Path):
    bids = Path(BIDS_FOLDER)
    base = bids / "derivatives" / "encoding_models" / model
    subjects = []
    for p in sorted(base.glob("sub-*")):
        if _load_mixture(p.name.removeprefix("sub-"), model, bids, smoothed):
            subjects.append(p.name.removeprefix("sub-"))
    if not subjects:
        raise SystemExit(f"No cached mixtures under {base}. Run "
                         f"compute_r2_mixture first.")

    # First pass: collect r2 distributions to compute a SHARED x-axis
    # (max 99.5th percentile across subjects), so per-subject pages can be
    # visually compared.
    loaded = {}
    for sub in subjects:
        sidecar = _load_mixture(sub, model, bids, smoothed)
        r2_all = _load_r2(sub, model, bids, smoothed)
        if sidecar is None or r2_all is None or "BRAIN" not in sidecar:
            continue
        r2 = r2_all[np.isfinite(r2_all) & (r2_all > 0) & (r2_all < 0.99)]
        loaded[sub] = (sidecar, r2_all, r2)
    if not loaded:
        raise SystemExit("No mixtures loaded.")
    x_max = float(max(np.percentile(r2, 99.5) for _, _, r2 in loaded.values()))

    # Pre-compute voxel counts so we can set a shared y-axis on the bar
    # panels — biggest count across subjects defines the cap.
    pre_counts = {}
    for sub, (sidecar, r2_all, _) in loaded.items():
        info = sidecar["BRAIN"]
        thr = {
            "fdr": r2_fdr_threshold_from_fit(info, alpha=alpha),
            "p50": r2_at_p_signal(info, 0.5),
            "p95": r2_at_p_signal(info, 0.95),
        }
        pre_counts[sub] = {k: int(np.sum(r2_all > t)) if np.isfinite(t) else 0
                            for k, t in thr.items()}
    voxel_max = max(max(c.values()) for c in pre_counts.values())

    out.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    fits: dict[str, dict] = {}
    with PdfPages(out) as pdf:
        for sub in subjects:
            if sub not in loaded:
                continue
            sidecar, r2_all, _ = loaded[sub]
            print(f"sub-{sub}: rendering")
            fig, c = plot_subject_page(sub, model, sidecar, r2_all, alpha,
                                        x_max=x_max, voxel_count_max=voxel_max)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            counts[sub] = c
            fits[sub] = sidecar["BRAIN"]
        if counts:
            for fn in (lambda: plot_summary_page(counts, model, alpha),
                       lambda: plot_thresholds_and_means_page(fits, model, alpha)):
                fig = fn()
                if fig is None:
                    continue
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="aprf",
                   choices=["aprf", "vonmises", "aprf-weighted", "aprf-gauss"])
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="FDR α for the FDR threshold criterion (default 0.05)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out = (Path(args.out) if args.out
           else Path(BIDS_FOLDER) / "derivatives" / "qa" / "r2_mixture"
                / f"compare_thresholds_{args.model}{'_smoothed' if args.smoothed else ''}.pdf")
    run(args.model, args.smoothed, args.alpha, out)


if __name__ == "__main__":
    main()
