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

    def _cat(pattern):
        fs = list(sweep_dir.glob(pattern))
        return pd.concat([pd.read_csv(p, sep="\t") for p in fs],
                         ignore_index=True) if fs else pd.DataFrame()

    vox = _cat(f"sub-*/func/*_desc-cvr2voxels{sm}.tsv")
    null = _cat(f"sub-*/func/*_desc-nullcvr2{sm}.tsv")
    sel = _cat(f"sub-*/func/*_desc-voxelselect{sm}.tsv")
    return vox, null, sel


def _union_per_subject(vox, sel):
    """Per (subject, model, cond, n_basis, fwhm) mean cvR2 over the UNION
    voxel set (passes FDR under aprf OR aprf-weighted) -- the model-neutral
    shared set. Same voxels for every model. Returns the per-subject frame."""
    if vox.empty or sel.empty:
        return pd.DataFrame()
    union = sel.loc[sel["in_union"] == 1, ["subject", "voxel"]]
    v = vox.merge(union, on=["subject", "voxel"], how="inner")
    return (v.groupby(["subject", "model", "cond", "n_basis", "fwhm"],
                      dropna=False)["cvr2"].mean().reset_index())


def _union_null(null, sel):
    if null.empty or sel.empty:
        return float("nan"), float("nan")
    union = sel.loc[sel["in_union"] == 1, ["subject", "voxel"]]
    n = null.merge(union, on=["subject", "voxel"], how="inner")
    per_sub = n.groupby("subject")["null_cvr2"].mean()
    n_union = union.groupby("subject").size().mean()
    return float(per_sub.mean()), float(n_union)


def _agg(df, ycol="cvr2"):
    """mean +/- SEM across subjects."""
    return df.groupby(["model", "cond", "n_basis", "fwhm"], dropna=False)[ycol] \
             .agg(["mean", "sem"]).reset_index()


def run(sweep_dir, out, smoothed):
    vox, null, sel = _load(sweep_dir, smoothed)
    per_sub = _union_per_subject(vox, sel)
    if per_sub.empty:
        raise SystemExit(f"No per-voxel + voxelselect TSVs under {sweep_dir}")
    n_sub = per_sub["subject"].nunique()
    null_top, n_union = _union_null(null, sel)
    print(f"{n_sub} subjects · models={sorted(per_sub['model'].unique())} · "
          f"conds={sorted(per_sub['cond'].unique())} · "
          f"~{n_union:.0f} union voxels/subj · null={null_top:+.4f}")

    single = per_sub[per_sub["model"] == "single"]
    weighted = per_sub[per_sub["model"] == "weighted"].copy()
    ylab = f"Mean cvR2 (union FDR voxels, ~{n_union:.0f}/subj)"
    out.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out) as pdf:
        # ── page 1: headline ──────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), constrained_layout=True)

        # A: single-pRF by condition
        ax = axes[0]
        s = single.groupby(["subject", "cond"])["cvr2"].mean().reset_index()
        order = [c for c in COND_ORDER if c in s["cond"].unique()]
        sns.barplot(data=s, x="cond", y="cvr2", order=order,
                    hue="cond", palette=COND_COLOUR, legend=False,
                    errorbar="se", ax=ax)
        sns.stripplot(data=s, x="cond", y="cvr2", order=order,
                      color="0.25", size=3, alpha=0.6, ax=ax)
        if np.isfinite(null_top):
            ax.axhline(null_top, color="k", ls=":", lw=0.9,
                       label="Null (train mean)")
            ax.legend(fontsize=7.5)
        ax.set_xlabel(""); ax.set_ylabel(ylab)
        ax.set_title("Single pRF: condition handling", fontsize=9)

        # B: weighted vs single (joint condition)
        ax = axes[1]
        wj = _agg(weighted[weighted["cond"] == "joint"])
        for fwhm, c in zip(sorted(wj["fwhm"].unique()),
                           sns.color_palette("flare", wj["fwhm"].nunique())):
            g = wj[wj["fwhm"] == fwhm].sort_values("n_basis")
            ax.errorbar(g["n_basis"], g["mean"], yerr=g["sem"], color=c,
                        marker="o", ms=3, capsize=2, label=f"{fwhm:g}")
        sj = single[single["cond"] == "joint"]["cvr2"]
        ax.axhline(sj.mean(), color="#264653", ls="--", lw=1.2,
                   label="single (joint)")
        if np.isfinite(null_top):
            ax.axhline(null_top, color="k", ls=":", lw=0.9, label="null")
        ax.set_xlabel("Number of basis functions  k")
        ax.set_ylabel("Mean cvR2 (union FDR voxels)")
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
            ax.set_ylabel("Mean cvR2 (union FDR voxels)")
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
