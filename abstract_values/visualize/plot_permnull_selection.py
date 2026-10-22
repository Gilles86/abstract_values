"""Permutation-null voxel selection vs the cvR² > null rule, per ROI.

Two rules, applied to the same session-shift aPRF fit:
  * Permutation — full-data R² against K within-run label shuffles of the same
    fit (``permnull_select.py``), pooled p, BH q ≤ .05 within the ROI
  * cvR² > null — leave-one-run-out cvR²(aprf-shift.cv) > cvR²(aprf-null.cv),
    the current rule (``project_cvr2_null_baseline``)

Pages:
  1. Per ROI and smoothing: fraction of voxels selected; cross-session
     consistency of the preferred value (per-subject Spearman ρ between the
     cdf-mapping and inverse-cdf-mapping mode — the two modes are fitted on
     different sessions, so higher ρ means more reliable estimates, unless the
     code genuinely remaps); out-of-sample gain cvR²(shift) − cvR²(null) of the
     selected voxels (biased towards the cv rule, which selects on it).
  2. Calibration of the permutation rule: leave-one-shuffle-out rate of any
     discovery (should be ≲ .05) and the exchangeability checks (split-half r
     ≈ 0, variance ratio ≈ 1).
  3. NPCr: preferred value under the two mappings, voxels from each rule.

Input: ``notes/data/permnull_voxels.tsv.gz`` + ``permnull_summary.tsv``
(``collect_permnull.py``).

Usage:
    python -m abstract_values.visualize.plot_permnull_selection
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

from abstract_values.visualize import qc_value_shift as q   # house rcParams
from abstract_values.visualize.collect_permnull import swap_selection
from abstract_values.visualize import shifted_preferred_value as spv

REPO = Path(__file__).resolve().parents[2]
ROIS = [("BensonV1", "V1"), ("BensonV2", "V2"), ("BensonV3", "V3"),
        ("hV4", "hV4"), ("V3ab", "V3a/b"), ("IPS0", "IPS0"), ("IPS12", "IPS1–2"),
        ("IPS345", "IPS3–5"), ("NPCl", "NPC left"), ("NPCr", "NPC right")]
RULE_COL = {"Permutation": "#3B5BA5", "cvR² > null": "#D08C2E"}
Q = 0.05


def per_roi(vox):
    """One row per (subject, smoothed, roi, rule)."""
    rows = []
    for (s, sm), d in vox.groupby(["subject", "smoothed"]):
        for roi, _ in ROIS:
            if f"in_{roi}" not in d:
                continue
            r = d[d[f"in_{roi}"].astype(bool)]
            if r.empty:
                continue
            rules = {"Permutation": r[f"q_{roi}"].fillna(1) <= Q,
                     "cvR² > null": r.cvr2_session_shift > r.cvr2_null}
            for rule, m in rules.items():
                x = r[m]
                rho = (stats.spearmanr(x.mode_cdf, x.mode_invcdf)[0]
                       if len(x) >= 10 else np.nan)
                rows.append(dict(
                    subject=s, smoothed=sm, roi=roi, rule=rule, n_voxels=len(r),
                    n_selected=int(m.sum()), frac_selected=float(m.mean()),
                    rho=rho,
                    gain=float((x.cvr2_session_shift - x.cvr2_null).mean())
                    if len(x) else np.nan,
                    overlap=float((rules["Permutation"] & rules["cvR² > null"]).sum()
                                  / max((rules["Permutation"] | rules["cvR² > null"]).sum(), 1))))
    return pd.DataFrame(rows)


def strip(ax, df, y, ylabel, ref=None):
    order = [r for r, _ in ROIS if r in set(df.roi)]
    sns.stripplot(data=df, x="roi", y=y, hue="rule", order=order, dodge=True,
                  palette=RULE_COL, size=2.2, alpha=0.45, jitter=0.15, ax=ax,
                  legend=False, zorder=1)
    sns.pointplot(data=df, x="roi", y=y, hue="rule", order=order, dodge=0.4,
                  palette=RULE_COL, errorbar=("se", 1), markersize=3.5,
                  linestyle="none", err_kws={"linewidth": 1.0}, ax=ax, zorder=3)
    if ref is not None:
        ax.axhline(ref, color="0.5", lw=0.6, ls=(0, (3, 2)), zorder=0)
    labels = dict(ROIS)
    sns.despine(ax=ax, offset=2)  # before ticks: offset resets tick labels
    ax.set_xticks(range(len(order)), [labels[o] for o in order])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set(xlabel="", ylabel=ylabel)
    if ax.get_legend():
        ax.get_legend().remove()


def page_rules(pdf, comp):
    fig, axes = plt.subplots(3, 2, figsize=(7.5, 8.2), constrained_layout=True,
                             sharey="row")
    for j, sm in enumerate((False, True)):
        c = comp[comp.smoothed == sm]
        strip(axes[0, j], c, "frac_selected", "Fraction of ROI selected")
        strip(axes[1, j], c, "rho", "Cross-session ρ\n(mode cdf vs inverse-cdf)", ref=0)
        strip(axes[2, j], c, "gain", "cvR²(shift) − cvR²(null)\nof selected voxels", ref=0)
        axes[0, j].set_title("Smoothed betas" if sm else "Unsmoothed betas",
                             loc="left", fontweight="bold")
    handles = [plt.Line2D([], [], marker="o", ls="none", color=c, label=r)
               for r, c in RULE_COL.items()]
    fig.legend(handles=handles, frameon=False, loc="outside upper right", ncol=2)
    fig.suptitle("Voxel selection: permutation null vs cvR² > null "
                 "(session-shift aPRF)", x=0.02, ha="left", fontweight="bold")
    pdf.savefig(fig)
    return fig


def page_calibration(pdf, summ):
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), constrained_layout=True)
    order = [r for r, _ in ROIS if r in set(summ.roi)]
    labels = dict(ROIS)
    for ax, (col, lab, ref) in zip(axes, [
            ("fpr_any", "Shuffles with any discovery\n(leave-one-shuffle-out)", Q),
            ("split_half_r", "Split-half r of\nper-voxel null mean", 0),
            ("var_ratio", "Variance ratio of\nper-voxel null mean", 1)]):
        sns.pointplot(data=summ, x="roi", y=col, hue="smoothed", order=order,
                      dodge=0.3, palette={False: "0.55", True: "0.1"},
                      errorbar=("se", 1), markersize=3.5, linestyle="none", ax=ax)
        ax.axhline(ref, color="#C44E52", lw=0.7, ls=(0, (3, 2)))
        sns.despine(ax=ax, offset=2)  # before ticks: offset resets tick labels
        ax.set_xticks(range(len(order)), [labels[o] for o in order])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set(xlabel="", ylabel=lab)
        h, l = ax.get_legend_handles_labels()
        ax.get_legend().remove()
    axes[0].legend(h, ["Unsmoothed", "Smoothed"], frameon=False, fontsize=6.5)
    fig.suptitle("Permutation null: calibration and exchangeability "
                 "(mean ± SEM over subjects)", x=0.02, ha="left", fontweight="bold")
    pdf.savefig(fig)
    return fig


def page_npcr(pdf, vox, comp, roi="NPCr"):
    pairs = spv._orientation_value_pairs().sort_values("value_cdf")
    fig, axes = plt.subplots(2, 2, figsize=(5.4, 5.6), constrained_layout=True)
    for j, sm in enumerate((False, True)):
        d = vox[(vox.smoothed == sm) & vox[f"in_{roi}"].astype(bool)]
        for i, (rule, m) in enumerate({
                "Permutation": d[f"q_{roi}"].fillna(1) <= Q,
                "cvR² > null": d.cvr2_session_shift > d.cvr2_null}.items()):
            ax = axes[i, j]
            x = d[m]
            if len(x) < 5:
                ax.set_axis_off()
                continue
            hb = q.hexpanel(ax, x, pairs)
            c = comp[(comp.smoothed == sm) & (comp.roi == roi) & (comp.rule == rule)]
            ax.set_title(f"{rule} · {'smoothed' if sm else 'unsmoothed'}",
                         loc="left", fontsize=7, color=RULE_COL[rule], pad=11)
            ax.text(0, 1.03, f"{x.subject.nunique()} subjects · {len(x)} voxels · "
                    f"median ρ {c.rho.median():.2f}", transform=ax.transAxes,
                    fontsize=6, color="0.4")
            ax.set(xlabel="Preferred value, cdf (CHF)",
                   ylabel="Preferred value,\ninverse-cdf (CHF)" if j == 0 else "")
    cb = fig.colorbar(hb, ax=axes, shrink=0.5)
    cb.set_label("Density (× uniform)")
    fig.suptitle("NPC right: preferred value under the two mappings, by "
                 "selection rule", x=0.02, ha="left", fontweight="bold")
    pdf.savefig(fig)
    return fig


def main(vox_fn, summ_fn, out, selection_from=None, tag=""):
    vox = pd.read_csv(vox_fn, sep="\t", dtype={"subject": str})
    if selection_from:
        vox = swap_selection(vox, selection_from)
    summ = pd.read_csv(summ_fn, sep="\t", dtype={"subject": str})
    comp = per_roi(vox)
    comp.to_csv(REPO / "notes" / "data" / f"permnull_selection_comparison{tag}.tsv",
                sep="\t", index=False)
    agg = (comp.groupby(["smoothed", "roi", "rule"])
           [["frac_selected", "rho", "gain", "n_selected"]]
           .agg(["mean", "median"]).round(3))
    print(agg.to_string())
    for (sm, roi), c in comp.groupby(["smoothed", "roi"]):
        w = c.pivot(index="subject", columns="rule", values="rho").dropna()
        if len(w) >= 5:
            t = stats.wilcoxon(w["Permutation"], w["cvR² > null"])
            print(f"  rho perm vs cv  {roi:8s} sm={sm}: n={len(w)}  "
                  f"median diff {np.median(w['Permutation'] - w['cvR² > null']):+.3f}"
                  f"  p={t.pvalue:.3g}")
    print(summ.groupby(["smoothed", "roi"])[["fpr_any", "split_half_r",
                                            "var_ratio"]].mean().round(3).to_string())
    out = Path(out)
    with PdfPages(out) as pdf:
        for i, f in enumerate((page_rules(pdf, comp), page_calibration(pdf, summ),
                               page_npcr(pdf, vox, comp))):
            f.savefig(str(out).replace(".pdf", f"_p{i + 1}.png"), dpi=200)
            plt.close(f)
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxels", default=str(REPO / "notes/data/permnull_voxels.tsv.gz"))
    ap.add_argument("--summary", default=str(REPO / "notes/data/permnull_summary.tsv"))
    ap.add_argument("--out", default=str(REPO / "notes/figures/permnull_selection.pdf"))
    ap.add_argument("--selection-from", default=None,
                    help="Collected voxels table of another permnull model whose "
                         "q columns define the permutation selection")
    ap.add_argument("--tag", default="", help="Suffix for the side TSVs")
    a = ap.parse_args()
    main(a.voxels, a.summary, a.out, a.selection_from, a.tag)
