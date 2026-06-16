"""NPC value model comparison: non-linear single pRF vs weighted basis,
across condition handling (joint / shift / separate).

Reads the per-subject TSVs from
``abstract_values.encoding_models.sweep_npc_value`` under
``derivatives/experiments/npc_value_sweep/`` and produces:

  1. Headline: single-pRF cvR2 (top voxels) by condition, and weighted vs
     single for the joint condition (does flexibility help?). Dotted line
     = true null (predict train mean) over the same voxels.
  2. Weighted basis k x fwhm sweep, joint vs separate panels.

cvR2 is summarised over the top-N NPCr voxels by the INDEPENDENT joint
`aprf` value-R2 (non-circular). Run the sweep on the cluster, rsync the
TSVs back, then run locally:
    python -m abstract_values.visualize.npc_value_modelcomp
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
    "font.sans-serif": ["Helvetica", "Arial"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})
sns.set_context("paper")

SWEEP_DIR = Path(BIDS_FOLDER) / "derivatives" / "experiments" / "npc_value_sweep"
DEFAULT_OUT = Path(BIDS_FOLDER) / "derivatives" / "qa" / "npc_value_modelcomp.pdf"
COND_ORDER = ["joint", "shift", "separate"]
COND_COLOUR = {"joint": "#264653", "shift": "#2A9D8F", "separate": "#E9C46A"}


def _load(sweep_dir, smoothed):
    sm = "_smoothed" if smoothed else ""
    cv = [pd.read_csv(p, sep="\t") for p in
          sweep_dir.glob(f"sub-*/func/*_desc-cvr2summary{sm}.tsv")]
    null = [pd.read_csv(p, sep="\t") for p in
            sweep_dir.glob(f"sub-*/func/*_desc-nullcvr2{sm}.tsv")]
    cv = pd.concat(cv, ignore_index=True) if cv else pd.DataFrame()
    null = pd.concat(null, ignore_index=True) if null else pd.DataFrame()
    return cv, null


def _top_null(null, top_n=100):
    """Mean null cvR2 over the top-N voxels by independent value-R2,
    averaged across subjects."""
    if null.empty or null["value_r2"].isna().all():
        return float("nan")
    top = (null.dropna(subset=["value_r2"])
           .sort_values("value_r2", ascending=False)
           .groupby("subject").head(top_n))
    return float(top.groupby("subject")["null_cvr2"].mean().mean())


def _agg(df, ycol="mean_cvr2_top"):
    """mean +/- SEM across subjects."""
    return df.groupby(["model", "cond", "n_basis", "fwhm"], dropna=False)[ycol] \
             .agg(["mean", "sem"]).reset_index()


def run(sweep_dir, out, smoothed):
    cv, null = _load(sweep_dir, smoothed)
    if cv.empty:
        raise SystemExit(f"No cvr2summary TSVs under {sweep_dir}")
    n_sub = cv["subject"].nunique()
    print(f"{n_sub} subjects · models={sorted(cv['model'].unique())} · "
          f"conds={sorted(cv['cond'].unique())}")
    null_top = _top_null(null)

    single = cv[cv["model"] == "single"]
    weighted = cv[cv["model"] == "weighted"].copy()
    out.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out) as pdf:
        # ── page 1: headline ──────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), constrained_layout=True)

        # A: single-pRF by condition
        ax = axes[0]
        s = (single.groupby(["subject", "cond"])["mean_cvr2_top"].mean()
             .reset_index())
        order = [c for c in COND_ORDER if c in s["cond"].unique()]
        sns.barplot(data=s, x="cond", y="mean_cvr2_top", order=order,
                    hue="cond", palette=COND_COLOUR, legend=False,
                    errorbar="se", ax=ax)
        sns.stripplot(data=s, x="cond", y="mean_cvr2_top", order=order,
                      color="0.25", size=3, alpha=0.6, ax=ax)
        if np.isfinite(null_top):
            ax.axhline(null_top, color="k", ls=":", lw=0.9,
                       label="Null (train mean)")
            ax.legend(fontsize=7.5)
        ax.set_xlabel(""); ax.set_ylabel("Mean cvR2 (top-100 voxels)")
        ax.set_title("Single pRF: condition handling", fontsize=9)

        # B: weighted vs single (joint condition)
        ax = axes[1]
        wj = _agg(weighted[weighted["cond"] == "joint"])
        for fwhm, c in zip(sorted(wj["fwhm"].unique()),
                           sns.color_palette("flare", wj["fwhm"].nunique())):
            g = wj[wj["fwhm"] == fwhm].sort_values("n_basis")
            ax.errorbar(g["n_basis"], g["mean"], yerr=g["sem"], color=c,
                        marker="o", ms=3, capsize=2, label=f"{fwhm:g}")
        sj = single[single["cond"] == "joint"]["mean_cvr2_top"]
        ax.axhline(sj.mean(), color="#264653", ls="--", lw=1.2,
                   label="single (joint)")
        if np.isfinite(null_top):
            ax.axhline(null_top, color="k", ls=":", lw=0.9, label="null")
        ax.set_xlabel("Number of basis functions  k")
        ax.set_ylabel("Mean cvR2 (top-100 voxels)")
        ax.set_title("Weighted basis vs single (joint cond.)", fontsize=9)
        ax.legend(title="fwhm (CHF)", fontsize=7, title_fontsize=7.5, ncol=1)

        fig.suptitle(f"NPCr value encoding cvR2 — model family x condition  "
                     f"(n={n_sub}, {'smoothed' if smoothed else 'unsmoothed'})",
                     y=1.05)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── page 2: weighted k x fwhm, joint vs separate ──────────────────────
        conds = [c for c in ("joint", "separate") if c in weighted["cond"].unique()]
        fig, axes = plt.subplots(1, len(conds), figsize=(4.3 * len(conds), 3.4),
                                 constrained_layout=True, squeeze=False)
        for ax, cond in zip(axes[0], conds):
            wc = _agg(weighted[weighted["cond"] == cond])
            for fwhm, c in zip(sorted(wc["fwhm"].unique()),
                               sns.color_palette("flare", wc["fwhm"].nunique())):
                g = wc[wc["fwhm"] == fwhm].sort_values("n_basis")
                ax.errorbar(g["n_basis"], g["mean"], yerr=g["sem"], color=c,
                            marker="o", ms=3, capsize=2, label=f"{fwhm:g}")
            if np.isfinite(null_top):
                ax.axhline(null_top, color="k", ls=":", lw=0.9, zorder=-1)
            ax.set_xlabel("Number of basis functions  k")
            ax.set_ylabel("Mean cvR2 (top-100 voxels)")
            ax.set_title(f"Weighted basis — {cond}", fontsize=9)
            ax.legend(title="fwhm (CHF)", fontsize=7, title_fontsize=7.5)
        fig.suptitle("NPCr weighted-basis value model: k x fwhm", y=1.05)
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
