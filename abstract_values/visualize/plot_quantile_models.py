"""Quantile coding vs fixed-value vs flexible value tuning, per ROI.

Cross-validated (leave-one-run-out) comparison of aPRF variants that differ
only in what the tuning curve is laid out on:

  Fixed value       'gaussian'            Gaussian on CHF, same in both sessions
  Quantile          'quantile'            Gaussian on the within-session value
                                          quantile (efficient / quantile coding)
  Quantile, wrapped 'quantile-wrapped'    von Mises on the quantile (circular)
  Orientation       'vonmises-prf'        von Mises on orientation (circular)
  Flexible          'gauss-session-shift' Gaussian on CHF, free mode per session
  (+ 'standard' log-Gaussian, 'session-shift' log-Gaussian, 'linear')

In this design both mappings are monotone in orientation and each of the 23
orientations is shown equally often, so a value's quantile IS the rank of its
orientation: 'quantile' is an unwrapped Gaussian on orientation and
'quantile-wrapped' is 'vonmises-prf' up to a 23/24 rescaling. The informative
contrast is therefore wrapped vs unwrapped — circular (orientation-like) vs
ordered (value-like) — next to the fixed-value and flexible models.

Statistic: per subject and ROI, the mean over voxels of cvR²(model) − cvR²(null)
(percentage points), for all ROI voxels and for the permutation-selected ones
(``permnull_select.py`` on session-shift; selection is on a different, full-data
statistic but favours voxels the flexible model fits).

Usage:
    python -m abstract_values.visualize.plot_quantile_models
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

from abstract_values.visualize import qc_value_shift as q   # noqa: F401 house rcParams
from abstract_values.visualize.plot_permnull_selection import ROIS, Q

REPO = Path(__file__).resolve().parents[2]
MODELS = [  # (column, label, colour)
    ("gaussian", "Fixed value", "#8C8C8C"),
    ("standard", "Fixed value (log)", "#BDBDBD"),
    ("linear", "Linear", "#E0E0E0"),
    ("quantile", "Quantile", "#3B5BA5"),
    ("quantile_wrapped", "Quantile, wrapped", "#8172B2"),
    ("vonmises_prf", "Orientation (wrapped)", "#5D8C3F"),
    ("gauss_session_shift", "Flexible", "#C44E52"),
    ("session_shift", "Flexible (log)", "#E39C9E"),
]
CONTRASTS = [  # (a, b, label)
    ("quantile", "gaussian", "Quantile − fixed value"),
    ("quantile", "gauss_session_shift", "Quantile − flexible"),
    ("quantile_wrapped", "quantile", "Wrapped − unwrapped quantile"),
    ("gauss_session_shift", "gaussian", "Flexible − fixed value"),
]


def per_subject(vox):
    rows = []
    cols = [m for m, _, _ in MODELS if f"cvr2_{m}" in vox]
    for (s, sm), d in vox.groupby(["subject", "smoothed"]):
        for roi, _ in ROIS:
            if f"in_{roi}" not in d:
                continue
            r = d[d[f"in_{roi}"].astype(bool)]
            for sel, m in (("All voxels", np.ones(len(r), bool)),
                           ("Permutation-selected", r[f"q_{roi}"].fillna(1) <= Q)):
                x = r[np.asarray(m)]
                if len(x) < 5:
                    continue
                row = dict(subject=s, smoothed=sm, roi=roi, selection=sel,
                           n=len(x))
                for c in cols:
                    row[c] = 100 * float((x[f"cvr2_{c}"] - x.cvr2_null).mean())
                win = x[[f"cvr2_{c}" for c in
                         ("gaussian", "quantile", "quantile_wrapped",
                          "gauss_session_shift") if f"cvr2_{c}" in x]]
                if win.shape[1] == 4 and win.notna().all(1).any():
                    w = win.dropna().idxmax(1).str.removeprefix("cvr2_").value_counts(
                        normalize=True)
                    for c in ("gaussian", "quantile", "quantile_wrapped",
                              "gauss_session_shift"):
                        row[f"win_{c}"] = float(w.get(c, 0.0))
                rows.append(row)
    return pd.DataFrame(rows)


def page_models(pdf, ps, sel, sm):
    d = ps[(ps.selection == sel) & (ps.smoothed == sm)]
    roi_order = [r for r, _ in ROIS if r in set(d.roi)]
    labels = dict(ROIS)
    models = [(m, l, c) for m, l, c in MODELS if m in d]
    fig, axes = plt.subplots(2, 5, figsize=(8.27, 4.6), constrained_layout=True)
    for ax, roi in zip(axes.flat, roi_order):
        r = d[d.roi == roi]
        long = r.melt(id_vars="subject", value_vars=[m for m, _, _ in models],
                      var_name="model", value_name="gain")
        pal = {m: c for m, _, c in models}
        sns.pointplot(data=long, x="model", y="gain", hue="model", palette=pal,
                      order=[m for m, _, _ in models], errorbar=("se", 1),
                      markersize=3, linestyle="none", ax=ax, legend=False)
        ax.axhline(0, color="0.5", lw=0.6, ls=(0, (3, 2)))
        sns.despine(ax=ax, offset=2)  # before ticks: offset resets tick labels
        ax.set_xticks(range(len(models)), [l for _, l, _ in models])
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor", fontsize=5.5)
        ax.set(xlabel="", ylabel="cvR² − null (pp)" if ax in axes[:, 0] else "")
        ax.set_title(f"{labels[roi]} (n = {r.subject.nunique()})", loc="left",
                     fontsize=7)
    for ax in axes.flat[len(roi_order):]:
        ax.set_axis_off()
    fig.suptitle(f"Cross-validated fit by model · {sel.lower()} · "
                 f"{'smoothed' if sm else 'unsmoothed'} betas (mean ± SEM)",
                 x=0.02, ha="left", fontweight="bold")
    pdf.savefig(fig)
    return fig


def page_contrasts(pdf, ps, sel):
    d = ps[ps.selection == sel]
    roi_order = [r for r, _ in ROIS if r in set(d.roi)]
    labels = dict(ROIS)
    fig, axes = plt.subplots(len(CONTRASTS), 2, figsize=(7.5, 8.6),
                             constrained_layout=True, sharey="row")
    rows = []
    for i, (a, b, lab) in enumerate(CONTRASTS):
        for j, sm in enumerate((False, True)):
            ax = axes[i, j]
            x = d[d.smoothed == sm].assign(diff=lambda t: t[a] - t[b])
            sns.stripplot(data=x, x="roi", y="diff", order=roi_order, color="0.6",
                          size=2, alpha=0.5, jitter=0.15, ax=ax, zorder=1)
            sns.pointplot(data=x, x="roi", y="diff", order=roi_order, color="0.1",
                          errorbar=("se", 1), markersize=3.5, linestyle="none",
                          ax=ax, zorder=3)
            ax.axhline(0, color="#C44E52", lw=0.6, ls=(0, (3, 2)))
            for k, roi in enumerate(roi_order):
                v = x.loc[x.roi == roi, "diff"].dropna()
                if len(v) < 5:
                    continue
                t, p = stats.ttest_1samp(v, 0)
                rows.append(dict(contrast=lab, smoothed=sm, roi=roi, n=len(v),
                                 mean=v.mean(), sem=v.sem(), t=t, p=p))
                if p < 0.05:
                    ax.text(k, 1.0, "*" if p >= 0.001 else "**",
                            transform=ax.get_xaxis_transform(), ha="center",
                            va="top", fontsize=8)
            sns.despine(ax=ax, offset=2)  # before ticks: offset resets tick labels
            ax.set_xticks(range(len(roi_order)), [labels[r] for r in roi_order])
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
            ax.set(xlabel="", ylabel=f"{lab}\n(pp cvR²)" if j == 0 else "")
            if i == 0:
                ax.set_title("Smoothed betas" if sm else "Unsmoothed betas",
                             loc="left", fontweight="bold")
    fig.suptitle(f"Model contrasts per ROI · {sel.lower()} · one dot per subject "
                 "(* p < .05, ** p < .001, uncorrected)", x=0.02, ha="left",
                 fontweight="bold")
    pdf.savefig(fig)
    return fig, pd.DataFrame(rows)


def main(vox_fn, out):
    vox = pd.read_csv(vox_fn, sep="\t", dtype={"subject": str})
    ps = per_subject(vox)
    ps.to_csv(REPO / "notes" / "data" / "quantile_models_per_subject.tsv",
              sep="\t", index=False)
    out = Path(out)
    figs, stats_rows = [], []
    with PdfPages(out) as pdf:
        for sel in ("Permutation-selected", "All voxels"):
            f, st = page_contrasts(pdf, ps, sel)
            figs.append(f)
            stats_rows.append(st.assign(selection=sel))
            for sm in (False, True):
                figs.append(page_models(pdf, ps, sel, sm))
    st = pd.concat(stats_rows)
    st.to_csv(REPO / "notes" / "data" / "quantile_models_contrasts.tsv", sep="\t",
              index=False)
    with pd.option_context("display.width", 200):
        print(st.round(3).to_string(index=False))
    win = [c for c in ps if c.startswith("win_")]
    print(ps[ps.selection == "Permutation-selected"]
          .groupby(["smoothed", "roi"])[win].mean().round(3).to_string())
    for i, f in enumerate(figs):
        f.savefig(str(out).replace(".pdf", f"_p{i + 1}.png"), dpi=200)
        plt.close(f)
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxels", default=str(REPO / "notes/data/permnull_voxels.tsv.gz"))
    ap.add_argument("--out", default=str(REPO / "notes/figures/quantile_models.pdf"))
    a = ap.parse_args()
    main(a.voxels, a.out)
