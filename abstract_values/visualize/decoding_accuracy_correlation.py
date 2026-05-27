"""Real-trial decoding accuracy as Pearson r and Spearman ρ between the
posterior-mean decoded value (or orientation) and the true value/orientation.

This complements compare_noise_models.py — that one looks at *simulated*
expected-decoded MAE/SD from the encoding-model simulator, this one looks
at the *actual* held-out trial decodes from the BOLD data using each
encoding model as the basis for the decoder.

Reads pars.tsv files written by the decode_value / decode_gabor pipelines:
each file is one (subject, mask, nvoxels, noise, [smoothed], [lambda])
config; each row is one trial with `true_value_chf` (or true_orientation)
plus a 50-bin posterior PDF over the stimulus grid. We compute decoded =
sum(prob_i * value_i), then Pearson r and Spearman ρ on the (true,
decoded) pairs pooled across all trials of that subject.

Two ROIs:
  - NPCr (value decoder) — true_value_chf
  - BensonV1 (gabor decoder) — true orientation

For each (ROI, smoothing, nvoxels, lambda) cell, show per-subject r and
ρ as a swarm + group mean ± SEM. Best (encoder × selection × smoothing)
combination across the cohort wins on these plots.

Usage:
    python -m abstract_values.visualize.decoding_accuracy_correlation
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

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

DECODING_ROOT = Path(BIDS_FOLDER) / "derivatives" / "decoding"
DEFAULT_OUT   = (Path(BIDS_FOLDER) / "derivatives" / "qa"
                 / "decoding_accuracy_correlation.pdf")

# (folder, kind, true_col, value_unit). 'value-weighted' uses the
# aprf-weighted encoder; 'value' uses aprf-session-shift.
DECODERS = [
    ("value",          "NPCr",     "value", "true_value_chf",   "CHF"),
    ("value-weighted", "NPCr",     "value", "true_value_chf",   "CHF"),
    # Gabor decoder writes orientation in radians. Pearson r is
    # rotation-invariant under linear rescaling so unit choice doesn't
    # affect the correlation; we just need to match the column name.
    ("gabor",          "BensonV1", "gabor", "true_orientation_rad", "rad"),
]

# Parse filenames like:
#   sub-XX_mask-NPCr_nvoxels-100_noise-full_lambda-0.1_pars.tsv
#   sub-XX_mask-BensonV1_nvoxels-0_noise-full_smoothed_pars.tsv
_FN_RE = re.compile(
    r"sub-(?P<subject>[A-Za-z0-9]+)"
    r"_mask-(?P<mask>[A-Za-z0-9]+)"
    r"_nvoxels-(?P<nv>[A-Za-z0-9]+)"
    r"_noise-(?P<noise>[A-Za-z0-9]+)"
    r"(?P<smooth>_smoothed)?"
    r"(?:_lambda-(?P<lam>[0-9.]+))?"
    r"_pars\.tsv$"
)


def _posterior_mean(df, value_bins):
    """Compute sum(p_i * value_i) per row. value_bins is the 1-D array of
    bin centres (extracted from column names)."""
    probs = df[[str(v) for v in value_bins]].to_numpy(dtype=np.float64)
    # Normalize each row to sum to 1 (pars.tsv writes already-normalized
    # posteriors but be safe — some early runs had un-normalized output).
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs @ value_bins


def _collect(folder, mask_filter, true_col):
    """Walk derivatives/decoding/<folder> and return one row per
    (subject, mask, nv, smooth, lambda) cell with r and ρ."""
    base = DECODING_ROOT / folder
    if not base.exists():
        return pd.DataFrame()
    rows = []
    for p in sorted(base.glob("sub-*/func/*_pars.tsv")):
        m = _FN_RE.match(p.name)
        if not m:
            continue
        meta = m.groupdict()
        if meta["mask"] != mask_filter:
            continue
        df = pd.read_csv(p, sep="\t")
        if df.empty or true_col not in df.columns:
            continue
        # The PDF bin columns are numeric strings. Filter to ones that
        # parse as floats — sidesteps any future metadata columns.
        value_bins = []
        for c in df.columns:
            try:
                value_bins.append(float(c))
            except ValueError:
                pass
        value_bins = np.array(value_bins, dtype=np.float64)
        decoded = _posterior_mean(df, value_bins)
        true_v  = df[true_col].to_numpy(dtype=np.float64)
        mask = np.isfinite(decoded) & np.isfinite(true_v)
        if mask.sum() < 5:
            continue
        r, _   = stats.pearsonr (true_v[mask], decoded[mask])
        rho, _ = stats.spearmanr(true_v[mask], decoded[mask])
        rows.append({
            "subject":  meta["subject"],
            "mask":     meta["mask"],
            "nvoxels":  meta["nv"],
            "noise":    meta["noise"],
            "smoothed": bool(meta["smooth"]),
            "lambda":   meta["lam"] if meta["lam"] else "0",
            "decoder":  folder,
            "n_trials": int(mask.sum()),
            "pearson_r":   float(r),
            "spearman_rho": float(rho),
        })
    return pd.DataFrame(rows)


def _swarm_with_mean(ax, df, x_col, y_col, hue_col=None, ylabel="", title=""):
    """Per-cell swarm + group mean ± SEM diamond. The diamond layer is
    what the eye lands on first (see scientific-figures skill)."""
    if df.empty:
        ax.set_title(f"{title}  —  no data", fontsize=9, color="0.5")
        ax.set_xticks([]); ax.set_yticks([])
        return
    # Stable x ordering: numeric nvoxels first, then 'fdr05' / 'psig*'.
    def _xkey(v):
        try:
            return (0, int(v))
        except ValueError:
            return (1, v)
    order = sorted(df[x_col].unique(), key=_xkey)

    palette = sns.color_palette("colorblind", n_colors=max(2,
                                df[hue_col].nunique() if hue_col else 1))
    sns.stripplot(data=df, x=x_col, y=y_col, hue=hue_col, order=order,
                  ax=ax, size=3.2, alpha=0.5, dodge=True, palette=palette,
                  jitter=0.18, linewidth=0)
    # Aggregate diamonds
    group_cols = [x_col] + ([hue_col] if hue_col else [])
    g = (df.groupby(group_cols)[y_col]
            .agg(["mean", "sem", "count"]).reset_index())
    if hue_col:
        hue_order = sorted(df[hue_col].unique())
        ndodge = len(hue_order)
        for i, h in enumerate(hue_order):
            sub = g[g[hue_col] == h]
            x = [order.index(v) + (i - (ndodge - 1) / 2) * 0.18 / ndodge * 2
                 for v in sub[x_col]]
            ax.errorbar(x, sub["mean"], yerr=sub["sem"],
                        fmt="D", ms=7, mec="black", mew=1.2, lw=0,
                        color=palette[i % len(palette)],
                        ecolor="black", elinewidth=0.8, capsize=0, zorder=5)
    else:
        x = [order.index(v) for v in g[x_col]]
        ax.errorbar(x, g["mean"], yerr=g["sem"],
                    fmt="D", ms=7, mec="black", mew=1.2, lw=0,
                    color=palette[0], ecolor="black",
                    elinewidth=0.8, capsize=0, zorder=5)

    ax.axhline(0, color="0.7", lw=0.6, ls=":", zorder=0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x_col)
    ax.set_title(title, fontsize=9, color="0.2")
    if hue_col and ax.get_legend() is not None:
        # Replace stripplot's tiny legend with the diamond-aware one
        handles, labels = ax.get_legend_handles_labels()
        keep = len(handles) // 2  # stripplot legend is duplicated by dodge
        ax.legend(handles[:keep], labels[:keep], title=hue_col,
                  loc="best", fontsize=7, title_fontsize=7)


def page_decoder(df, decoder_label, value_unit, pdf):
    if df.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.axis("off")
        ax.text(0.5, 0.5,
                f"{decoder_label}: no pars.tsv files yet.",
                ha="center", va="center", fontsize=10, color="0.3")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        return

    # One row per (smoothing × lambda) so the panels stay simple; columns
    # are r and ρ. Within each panel, the `noise` hue contrasts the two
    # residual models (full = legacy residual covariance vs spherical =
    # iid Gaussian). Lambda=0 vs 0.1 makes a small difference; leaving
    # the contrast visible lets the user pick the better regularizer.
    df = df.copy()
    df["cfg"] = df.apply(
        lambda r: f"{'smoothed' if r['smoothed'] else 'unsmoothed'}, "
                   f"λ={r['lambda']}", axis=1)
    cfgs = sorted(df["cfg"].unique())
    n = len(cfgs)
    fig, axes = plt.subplots(n, 2, figsize=(11.0, 2.6 * n + 0.8),
                              constrained_layout=True, squeeze=False)
    fig.suptitle(f"Real-trial decoding accuracy — {decoder_label}  "
                 f"(noise: full ‖ spherical)",
                 fontsize=10, y=1.02, color="0.15")
    hue_arg = "noise" if df["noise"].nunique() > 1 else None
    for i, cfg in enumerate(cfgs):
        sub = df[df["cfg"] == cfg]
        _swarm_with_mean(axes[i, 0], sub, "nvoxels", "pearson_r",
                          hue_col=hue_arg,
                          ylabel=f"Pearson r (true vs decoded {value_unit})",
                          title=f"{cfg}  ·  Pearson r")
        _swarm_with_mean(axes[i, 1], sub, "nvoxels", "spearman_rho",
                          hue_col=hue_arg,
                          ylabel=f"Spearman ρ (true vs decoded {value_unit})",
                          title=f"{cfg}  ·  Spearman ρ")
        # Match y across the row (r and ρ live on the same scale)
        ylo = min(axes[i, 0].get_ylim()[0], axes[i, 1].get_ylim()[0])
        yhi = max(axes[i, 0].get_ylim()[1], axes[i, 1].get_ylim()[1])
        axes[i, 0].set_ylim(ylo, yhi); axes[i, 1].set_ylim(ylo, yhi)

    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def run(out):
    out.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary = []
    for folder, mask, kind, true_col, unit in DECODERS:
        df = _collect(folder, mask, true_col)
        label = (f"NPCr value (aprf-session-shift, "
                 f"{folder.split('-')[-1] if '-' in folder else 'standard'})"
                 if kind == "value" else "V1 orientation (vonmises)")
        print(f"  {label}: {len(df)} cells "
              f"({df['subject'].nunique() if not df.empty else 0} subjects)")
        all_rows.append(df.assign(label=label, unit=unit))
        if not df.empty:
            summary.append((label, df))

    if not summary:
        raise SystemExit("No pars.tsv files found in any decoding subdir.")

    big = pd.concat(all_rows, ignore_index=True)
    tsv = out.with_suffix(".tsv")
    big.to_csv(tsv, sep="\t", index=False)
    print(f"Wrote {tsv}")

    with PdfPages(out) as pdf:
        for label, df in summary:
            page_decoder(df, label, df["unit"].iloc[0]
                          if "unit" in df.columns else "", pdf)
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(Path(args.out))


if __name__ == "__main__":
    main()
