"""V1 Von Mises model comparison: cvR2 and out-of-sample decoding as a
function of basis count (k) and concentration (kappa), plus the
preferred-orientation distribution at each setting.

Reads the per-subject TSVs written by
``abstract_values.encoding_models.sweep_v1_k_kappa`` under
``derivatives/experiments/v1_k_kappa_sweep/`` and produces a multi-page
PDF:

  1. Mean cvR2 (selected V1 voxels) vs k, one line per kappa (mean +/- SEM
     across subjects).
  2. If decoding was run: mean absolute out-of-sample decoding error vs k,
     one line per kappa.
  3. Preferred-orientation histograms, grid of (k x kappa), pooled across
     subjects -- how the tuning distribution sharpens/spreads.

Run the sweep on the cluster, rsync the TSVs back, then run this locally:
    python -m abstract_values.visualize.v1_k_kappa_modelcomp
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
    "xtick.direction": "out", "ytick.direction": "out",
    "lines.linewidth": 1.4, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})
sns.set_context("paper")

COND_COLOUR = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}
SWEEP_DIR = Path(BIDS_FOLDER) / "derivatives" / "experiments" / "v1_k_kappa_sweep"
DEFAULT_OUT = Path(BIDS_FOLDER) / "derivatives" / "qa" / "v1_k_kappa_modelcomp.pdf"


def _load(sweep_dir, smoothed):
    smooth = "_smoothed" if smoothed else ""
    cv = [pd.read_csv(p, sep="\t") for p in
          sweep_dir.glob(f"sub-*/func/*_desc-cvr2summary{smooth}.tsv")]
    hist = [pd.read_csv(p, sep="\t") for p in
            sweep_dir.glob(f"sub-*/func/*_desc-preferredhist{smooth}.tsv")]
    vox = [pd.read_csv(p, sep="\t") for p in
           sweep_dir.glob(f"sub-*/func/*_desc-cvr2voxels{smooth}.tsv")]
    null = [pd.read_csv(p, sep="\t") for p in
            sweep_dir.glob(f"sub-*/func/*_desc-nullcvr2{smooth}.tsv")]
    cv = pd.concat(cv, ignore_index=True) if cv else pd.DataFrame()
    hist = pd.concat(hist, ignore_index=True) if hist else pd.DataFrame()
    vox = pd.concat(vox, ignore_index=True) if vox else pd.DataFrame()
    null = pd.concat(null, ignore_index=True) if null else pd.DataFrame()
    return cv, hist, vox, null


def _signal_restricted_mean(vox, null):
    """Per (subject, n_basis, kappa) mean cvR2 over SIGNAL voxels -- voxels
    that BEAT THE TRUE NULL (model cvR2 > predict-train-mean cvR2) in >=1
    (k, kappa) config. Most V1 voxels are untuned noise sitting at the null;
    this isolates voxels carrying orientation signal. Returns the signal df
    plus the mean null cvR2 over the signal set (for the reference line)."""
    if vox.empty or null.empty:
        return pd.DataFrame(), float("nan")
    v = vox.merge(null, on=["subject", "voxel"], how="left")
    peak = v.groupby(["subject", "voxel"])["cvr2"].transform("max")
    sig = v[peak > v["null_cvr2"]]
    if sig.empty:
        return pd.DataFrame(), float("nan")
    out = (sig.groupby(["subject", "n_basis", "kappa"])["cvr2"]
           .mean().reset_index().rename(columns={"cvr2": "mean_cvr2_signal"}))
    n_sig = (sig.groupby("subject")["voxel"].nunique()
             .rename("n_signal").reset_index())
    out = out.merge(n_sig, on="subject")
    # mean null over the signal voxels, averaged across subjects
    signal_null = (sig.drop_duplicates(["subject", "voxel"])
                   .groupby("subject")["null_cvr2"].mean().mean())
    return out, float(signal_null)


def _topn_mean(vox, null, top_n=100):
    """Per (subject, n_basis, kappa) mean cvR2 over the top-N voxels by an
    INDEPENDENT criterion (joint Von Mises R2) -- non-circular, the honest
    'tuned voxels' view. Returns the df + mean null over those voxels."""
    if vox.empty or null.empty or null["joint_r2"].isna().all():
        return pd.DataFrame(), float("nan")
    top = (null.dropna(subset=["joint_r2"])
           .sort_values("joint_r2", ascending=False)
           .groupby("subject").head(top_n)[["subject", "voxel", "null_cvr2"]])
    v = vox.merge(top, on=["subject", "voxel"], how="inner")
    out = (v.groupby(["subject", "n_basis", "kappa"])["cvr2"]
           .mean().reset_index().rename(columns={"cvr2": "mean_cvr2_top"}))
    top_null = (top.groupby("subject")["null_cvr2"].mean().mean())
    return out, float(top_null)


def _basis_page(pdf, ks, kappas):
    """Reference page: the actual axial Von Mises basis sets, so 'k' and
    'kappa' are visually concrete. Rows = a few representative k (number of
    bumps), columns = kappa (concentration). Higher kappa = narrower/sharper
    tuning; more k = finer tiling of the 0-180 deg orientation axis."""
    ks_show = [k for k in (4, 8, 16, 24) if k in ks] or sorted(ks)[:4]
    x = np.linspace(0, np.pi, 360)
    xd = np.degrees(x)
    fig, axes = plt.subplots(len(ks_show), len(kappas),
                             figsize=(1.7 * len(kappas) + 0.6,
                                      1.3 * len(ks_show) + 0.8),
                             sharex=True, sharey=True, squeeze=False)
    for i, k in enumerate(ks_show):
        mus = np.linspace(0, np.pi, k, endpoint=False)
        colors = sns.color_palette("husl", k)
        for j, kap in enumerate(kappas):
            ax = axes[i][j]
            for mu, c in zip(mus, colors):
                b = np.exp(kap * np.cos(2 * (x - mu)))   # axial Von Mises
                ax.plot(xd, b / b.max(), lw=0.8, color=c)
            ax.set_xlim(0, 180); ax.set_xticks([0, 90, 180]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"κ = {kap:g}"
                             + ("  (broad)" if j == 0 else
                                "  (sharp)" if j == len(kappas) - 1 else ""),
                             fontsize=8)
            if j == 0:
                ax.set_ylabel(f"k = {k}", fontsize=9)
    fig.suptitle("Von Mises basis sets  -  rows: k (number of bumps tiling "
                 "orientation) · cols: κ (concentration).\n"
                 "Higher κ = narrower/sharper tuning curves; more k = "
                 "finer tiling. The model fits a weighted sum of these per voxel.",
                 fontsize=9, y=1.0)
    fig.supxlabel("Orientation (deg)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _line_vs_k(ax, df, ycol, ylabel, null_val=None):
    kappas = sorted(df["kappa"].unique())
    pal = sns.color_palette("viridis", len(kappas))
    for kappa, c in zip(kappas, pal):
        g = df[df["kappa"] == kappa]
        stat = g.groupby("n_basis")[ycol].agg(["mean", "sem"]).reset_index()
        ax.errorbar(stat["n_basis"], stat["mean"], yerr=stat["sem"],
                    color=c, marker="o", ms=3, capsize=2, label=f"{kappa:g}")
    if null_val is not None:    # true null = predict TRAINING mean (<0, not 0)
        ax.axhline(null_val, color="k", ls=":", lw=0.9, alpha=0.7, zorder=-1,
                   label="Null (train mean)")
    ax.set_xlabel("Number of basis functions  k")
    ax.set_ylabel(ylabel)
    ax.legend(title="κ (↑ = sharper)", fontsize=7.5, title_fontsize=7.5,
              loc="best", ncol=1)


def run(sweep_dir, out, smoothed):
    cv, hist, vox, null = _load(sweep_dir, smoothed)
    if cv.empty:
        raise SystemExit(f"No cvr2summary TSVs under {sweep_dir}")
    n_sub = cv["subject"].nunique()
    print(f"{n_sub} subjects · {sorted(cv['n_basis'].unique())} k · "
          f"{sorted(cv['kappa'].unique())} kappa")
    has_decode = "decode_mean_abs_err_deg" in cv.columns
    sig, signal_null = _signal_restricted_mean(vox, null)
    topn, top_null = _topn_mean(vox, null, top_n=100)
    null_all = float(null["null_cvr2"].mean()) if not null.empty else None

    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        # ── page 0: reference — what k and kappa look like ────────────────────
        _basis_page(pdf, sorted(cv["n_basis"].unique()),
                    sorted(cv["kappa"].unique()))

        # ── page 1: cvR2 vs k by kappa (tuned-voxel views + all V1) ───────────
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
        if not topn.empty:
            _line_vs_k(axes[0], topn, "mean_cvr2_top",
                       "Mean cvR2 (top-100 voxels)", null_val=top_null)
            axes[0].set_title("Top-100 by independent VM R2 (non-circular)",
                              fontsize=8)
        else:
            axes[0].text(0.5, 0.5, "No joint-R2 per-voxel info\n(rerun sweep)",
                         transform=axes[0].transAxes, ha="center", va="center",
                         color="0.5")
        if not sig.empty:
            n_sig = int(sig.groupby("subject")["n_signal"].first().mean())
            _line_vs_k(axes[1], sig, "mean_cvr2_signal",
                       "Mean cvR2 (signal voxels)", null_val=signal_null)
            axes[1].set_title(f"Signal voxels (beat null in >=1 config; "
                              f"~{n_sig}/subj; mildly circular)", fontsize=8)
        _line_vs_k(axes[2], cv, "mean_cvr2_all",
                   "Mean cvR2 (all V1 voxels)", null_val=null_all)
        axes[2].set_title("All V1 voxels (noise-dominated)", fontsize=8)
        fig.suptitle(f"V1 Von Mises encoding cvR2 vs k x kappa  "
                     f"(n={n_sub}, {'smoothed' if smoothed else 'unsmoothed'}, "
                     f"joint CV; dotted = true null / predict train mean)", y=1.06)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── page 2: out-of-sample decoding (error + circular correlation) ─────
        if has_decode:
            has_corr = "decode_circ_corr" in cv.columns
            ncol = 3 if has_corr else 2
            fig, axes = plt.subplots(1, ncol, figsize=(3.7 * ncol, 3.2),
                                     constrained_layout=True)
            _line_vs_k(axes[0], cv, "decode_mean_abs_err_deg",
                       "Mean abs decoding error (deg)")
            _line_vs_k(axes[1], cv, "decode_circ_sd_deg",
                       "Circular SD of decoding error (deg)")
            if has_corr:
                _line_vs_k(axes[2], cv, "decode_circ_corr",
                           "Circular corr (true vs decoded)")
                axes[2].axhline(0, color="k", ls=":", lw=0.8, alpha=0.5,
                                zorder=-1)
                axes[2].set_title("higher = better", fontsize=8)
            fig.suptitle("V1 out-of-sample orientation decoding (FDR voxels) "
                         "vs k x kappa", y=1.06)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── page 3: preferred-orientation histograms grid ─────────────────────
        if not hist.empty:
            ks = sorted(hist["n_basis"].unique())
            kappas = sorted(hist["kappa"].unique())
            agg = (hist.groupby(["n_basis", "kappa", "condition",
                                 "orientation_deg"])["count"]
                   .sum().reset_index())
            fig, axes = plt.subplots(len(ks), len(kappas),
                                     figsize=(1.5 * len(kappas) + 1,
                                              1.3 * len(ks) + 1),
                                     sharex=True, squeeze=False)
            for i, k in enumerate(ks):
                for j, kap in enumerate(kappas):
                    ax = axes[i][j]
                    sub = agg[(agg["n_basis"] == k) & (agg["kappa"] == kap)]
                    for cond, gc in sub.groupby("condition"):
                        gc = gc.sort_values("orientation_deg")
                        ax.plot(gc["orientation_deg"], gc["count"],
                                color=COND_COLOUR.get(cond, "0.4"), lw=1.0)
                    ax.set_xlim(0, 180); ax.set_xticks([0, 90, 180])
                    ax.set_yticks([])
                    if i == 0:
                        ax.set_title(f"kappa={kap:g}", fontsize=8)
                    if j == 0:
                        ax.set_ylabel(f"k={k}", fontsize=8)
            fig.suptitle("V1 preferred-orientation distribution (pooled voxels) "
                         "across k x kappa", y=1.0)
            fig.supxlabel("Preferred orientation (deg)", fontsize=9)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-dir", default=str(SWEEP_DIR))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--smoothed", action="store_true")
    args = p.parse_args()
    run(Path(args.sweep_dir), Path(args.out), args.smoothed)


if __name__ == "__main__":
    main()
