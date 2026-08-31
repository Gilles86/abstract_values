"""Compare two voxel-selection criteria for Bayesian value decoding.

Mirrors ``compare_voxel_selection.py``'s encoding-model comparison, but for
``decode_value.py``'s per-fold nested-CV voxel selection:

  Criterion 0  (existing):  nested cvR² > 0            (--n-voxels 0)
  Criterion NG (new):       nested cvR² > cvR²_null      (--null-gate)

Both already run for mask-NPCr, spherical noise, lambd=0.1, unsmoothed,
across the same subject set used throughout this project's session-shift
analyses (sub-22 excluded — missing the criterion-0 NPCr baseline).

Pages:
  1. Per-fold voxel counts selected, criterion 0 vs NG (from meta.tsv).
  2. Decoding accuracy (Pearson r, decoded posterior mean vs true value)
     per subject per condition, criterion 0 vs NG, paired.
  3. Mean absolute decoding error, same comparison.

Usage:
    python -m abstract_values.visualize.compare_decoding_selection
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from abstract_values.encoding_models.voxel_selection import usable_folds
from abstract_values.utils.data import BIDS_FOLDER, Subject
from abstract_values.visualize.shifted_preferred_value import COND_COLOUR

DECODE_ROOT = Path(BIDS_FOLDER) / "derivatives" / "decoding" / "value"
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "notes" / "figures" / "compare_decoding_selection.pdf"

CRIT_COLOUR = {"0": "#6C757D", "NG": "#8E44AD"}
CRIT_LABEL = {"0": "criterion 0: nested cvR² > 0",
             "NG": "criterion NG: nested cvR² > cvR²(null)"}
NVOX_TAG = {"0": "0", "NG": "nullgated"}

SUBJECTS = ["03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13",
           "14", "15", "16", "17", "19", "20", "21", "pil01", "pil02"]


def _path(subject: str, nvox_tag: str, desc: str, smoothed: bool = False,
         lambd: float = 0.1) -> Path:
    smooth = "_smoothed" if smoothed else ""
    lam = f"_lambda-{lambd}" if lambd != 0.0 else ""
    return (DECODE_ROOT / f"sub-{subject}" / "func"
            / f"sub-{subject}_mask-NPCr_nvoxels-{nvox_tag}_noise-spherical{smooth}{lam}_{desc}.tsv")


def load_meta(subject: str, crit: str, smoothed: bool = False) -> pd.DataFrame | None:
    p = _path(subject, NVOX_TAG[crit], "meta", smoothed=smoothed)
    if not p.exists():
        return None
    df = pd.read_csv(p, sep="\t")
    df["subject"] = subject
    df["criterion"] = crit
    return df


def load_decoded(subject: str, crit: str, smoothed: bool = False) -> pd.DataFrame | None:
    p = _path(subject, NVOX_TAG[crit], "pars", smoothed=smoothed)
    if not p.exists():
        return None
    df = pd.read_csv(p, sep="\t")
    if df.empty:
        # Every fold was undecodable (no voxel cleared the nested-CV floor).
        # Drop the cell rather than let a zero-trial result become a data point.
        return None
    grid_cols = df.columns[4:]
    grid = grid_cols.astype(float).to_numpy()
    probs = df[grid_cols].to_numpy(dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    decoded_mean = (probs * grid[None, :]).sum(axis=1)

    sub = Subject(subject, bids_folder=Path(BIDS_FOLDER))
    cond = df["session"].map(lambda s: sub.get_mapping(int(s)))

    out = df[["session", "run", "trial_nr", "true_value_chf"]].copy()
    out["decoded_mean"] = decoded_mean
    out["error"] = out["decoded_mean"] - out["true_value_chf"]
    out["condition"] = cond
    out["subject"] = subject
    out["criterion"] = crit
    return out


def collect_all(subjects, smoothed=False):
    meta_rows, decoded_rows = [], []
    for s in subjects:
        for crit in ["0", "NG"]:
            m = load_meta(s, crit, smoothed=smoothed)
            if m is not None:
                meta_rows.append(m)
            d = load_decoded(s, crit, smoothed=smoothed)
            if d is not None:
                decoded_rows.append(d)
    meta = pd.concat(meta_rows, ignore_index=True) if meta_rows else pd.DataFrame()
    decoded = pd.concat(decoded_rows, ignore_index=True) if decoded_rows else pd.DataFrame()
    return meta, decoded


def page_voxel_counts(meta: pd.DataFrame, pdf: PdfPages):
    subjects = sorted(meta.subject.unique(), key=lambda s: (0 if s[0].isdigit() else 1, s))
    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    ys = np.arange(len(subjects))
    for crit, colour, dy in [("0", CRIT_COLOUR["0"], -0.15), ("NG", CRIT_COLOUR["NG"], 0.15)]:
        means, sds = [], []
        for s in subjects:
            d = usable_folds(meta[(meta.subject == s) & (meta.criterion == crit)])
            means.append(d.n_voxels_selected.mean() if len(d) else np.nan)
            sds.append(d.n_voxels_selected.std() if len(d) else np.nan)
        ax.errorbar(means, ys + dy, xerr=sds, fmt="o", color=colour, ecolor=colour,
                   elinewidth=1.0, capsize=2.5, markersize=5, alpha=0.9,
                   label=CRIT_LABEL[crit])
    ax.set_yticks(ys)
    ax.set_yticklabels([f"sub-{s}" for s in subjects], fontsize=7.5)
    ax.set_xlabel("Voxels selected per fold (mean ± SD across folds)")
    ax.set_title("NPCr decoding voxel counts: criterion 0 vs. NG", fontsize=9.5, color="0.2")
    ax.legend(loc="best", fontsize=7.5, frameon=False)
    ax.invert_yaxis()
    sns.despine(fig=fig, offset=4, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _per_subject_metrics(decoded: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, crit, cond), d in decoded.groupby(["subject", "criterion", "condition"]):
        if len(d) < 5:
            continue
        r, _ = stats.pearsonr(d.true_value_chf, d.decoded_mean)
        rows.append(dict(subject=s, criterion=crit, condition=cond, n=len(d),
                         r=r, mae=d.error.abs().mean()))
    return pd.DataFrame(rows)


def _dumbbell(ax, g, col_a, col_b, label_a, label_b, colour_a, colour_b, ylabels):
    d = g.dropna(subset=[col_a, col_b])
    ys = np.arange(len(d))
    for y, (_, row) in zip(ys, d.iterrows()):
        ax.plot([row[col_a], row[col_b]], [y, y], "-", color="0.75", lw=0.8, zorder=1)
    ax.scatter(d[col_a], ys, s=42, color=colour_a, edgecolor="black",
              linewidth=0.7, zorder=3, label=label_a)
    ax.scatter(d[col_b], ys, s=42, marker="D", color=colour_b, edgecolor="black",
              linewidth=0.7, zorder=3, label=label_b)
    ax.set_yticks(ys)
    ax.set_yticklabels(ylabels, fontsize=7)
    return d


def page_accuracy_comparison(metrics: pd.DataFrame, pdf: PdfPages):
    for cond in ["cdf", "inverse_cdf"]:
        m = metrics[metrics.condition == cond]
        piv_r = m.pivot(index="subject", columns="criterion", values="r").reset_index()
        piv_r = piv_r.sort_values("subject",
                                  key=lambda x: x.map(lambda s: (0 if s[0].isdigit() else 1, s)))

        fig, ax = plt.subplots(figsize=(5.5, 4.6), constrained_layout=True)
        d = _dumbbell(ax, piv_r, "0", "NG", CRIT_LABEL["0"], CRIT_LABEL["NG"],
                     CRIT_COLOUR["0"], CRIT_COLOUR["NG"],
                     [f"sub-{s}" for s in piv_r.subject])
        ax.axvline(0, color="0.4", lw=0.6, ls="--", zorder=0)
        w = stats.wilcoxon(d["0"], d["NG"]) if len(d) > 3 else None
        title = (f"Decoding accuracy (r, decoded vs. true value) — {cond}\n"
                f"mean 0={d['0'].mean():.2f}  NG={d['NG'].mean():.2f}"
                + (f"  (Wilcoxon p={w.pvalue:.3f})" if w is not None else ""))
        ax.set_title(title, fontsize=8.5, color=COND_COLOUR[cond])
        ax.set_xlabel("Pearson r (decoded posterior mean vs. true CHF value)")
        ax.legend(loc="best", fontsize=7, frameon=False)
        sns.despine(fig=fig, offset=4, trim=True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def page_mae_comparison(metrics: pd.DataFrame, pdf: PdfPages):
    for cond in ["cdf", "inverse_cdf"]:
        m = metrics[metrics.condition == cond]
        piv = m.pivot(index="subject", columns="criterion", values="mae").reset_index()
        piv = piv.sort_values("subject",
                              key=lambda x: x.map(lambda s: (0 if s[0].isdigit() else 1, s)))

        fig, ax = plt.subplots(figsize=(5.5, 4.6), constrained_layout=True)
        d = _dumbbell(ax, piv, "0", "NG", CRIT_LABEL["0"], CRIT_LABEL["NG"],
                     CRIT_COLOUR["0"], CRIT_COLOUR["NG"],
                     [f"sub-{s}" for s in piv.subject])
        w = stats.wilcoxon(d["0"], d["NG"]) if len(d) > 3 else None
        title = (f"Mean absolute decoding error — {cond}\n"
                f"mean 0={d['0'].mean():.2f}  NG={d['NG'].mean():.2f} CHF"
                + (f"  (Wilcoxon p={w.pvalue:.3f})" if w is not None else ""))
        ax.set_title(title, fontsize=8.5, color=COND_COLOUR[cond])
        ax.set_xlabel("Mean |decoded − true| (CHF)")
        ax.legend(loc="best", fontsize=7, frameon=False)
        sns.despine(fig=fig, offset=4, trim=True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def run(subjects, out, smoothed_variants=(False,)):
    out.parent.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    with PdfPages(out) as pdf:
        for smoothed in smoothed_variants:
            label = "smoothed" if smoothed else "unsmoothed"
            fig, ax = plt.subplots(figsize=(7.5, 1.2), constrained_layout=True)
            ax.text(0.5, 0.5, f"{label.upper()}  ·  NPCr  ·  spherical  ·  λ=0.1",
                   ha="center", va="center", fontsize=18, color="0.1",
                   transform=ax.transAxes)
            ax.set_axis_off()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            meta, decoded = collect_all(subjects, smoothed=smoothed)
            print(f"\n=== {label} ===")
            print(f"  meta rows: {len(meta)}  decoded trial rows: {len(decoded)}")
            for s in subjects:
                n0 = len(decoded[(decoded.subject == s) & (decoded.criterion == "0")])
                nng = len(decoded[(decoded.subject == s) & (decoded.criterion == "NG")])
                print(f"  sub-{s}: n_trials 0={n0}  NG={nng}")

            if meta.empty or decoded.empty:
                continue
            page_voxel_counts(meta, pdf)
            metrics = _per_subject_metrics(decoded)
            metrics["variant"] = label
            all_metrics.append(metrics)
            page_accuracy_comparison(metrics, pdf)
            page_mae_comparison(metrics, pdf)

    if all_metrics:
        agg = pd.concat(all_metrics, ignore_index=True)
        tsv = out.with_suffix(".tsv")
        agg.to_csv(tsv, sep="\t", index=False)
        print(f"\nWrote {out}\nSidecar: {tsv}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+", default=SUBJECTS)
    p.add_argument("--smoothed", action="store_true", help="Also render the smoothed variant")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    variants = (False, True) if args.smoothed else (False,)
    run(args.subjects, Path(args.out), smoothed_variants=variants)


if __name__ == "__main__":
    main()
