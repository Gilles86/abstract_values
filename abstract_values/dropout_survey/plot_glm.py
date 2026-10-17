"""Is the mPFC task response still detectable, and does that track the dropout? -- local.

Reads notes/data/dropout_survey_glm.tsv (glm.py via combine --glm-only) and
dropout_survey_rois.tsv (the tSNR / void survey) and draws, in the dataset
order and grouping of fig_dropout_survey:

  a  dataset, GLM and the event used (the first single-trial event per dataset)
  b  GLMsingle R2 minus the subject's eroded-white-matter mean R2, per ROI
  c  t across trials of the ROI-mean single-trial beta, per ROI
  d  per participant: mOFC |beta t| against mOFC normalised tSNR
  e  per participant: mOFC fraction of voxels with R2 above the WM 95th
     percentile against the mOFC signal-void fraction

and writes notes/data/dropout_survey_glm_stats.tsv (per-dataset medians,
within-dataset Spearman correlations with the dropout metrics, and the
across-dataset correlation of the dataset medians).

    python -m abstract_values.dropout_survey.plot_glm
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from abstract_values.dropout_survey.datasets import DATASETS
from abstract_values.dropout_survey.plot import COL as ROI_COL, DATA, FIG, GROUPS, HILITE, style

ROIS = ["mOFC", "lOFC", "pgACC", "dmPFC", "cortex"]
COL = dict(ROI_COL, cortex="#4A4A4A")
GCOL = {"Siemens 3T": "#C44E52", "Philips 3T": "#3B5BA5", "7T": "#5D8C3F"}
TCLIP = 20


def load():
    g = pd.read_csv(DATA / "dropout_survey_glm.tsv", sep="\t", dtype={"subject": str})
    meta = pd.read_csv(DATA / "dropout_survey_glm_meta.tsv", sep="\t", dtype={"subject": str})
    r = pd.read_csv(DATA / "dropout_survey_rois.tsv", sep="\t", dtype={"subject": str})
    # primary event = first event of each dataset's GLM definition
    from abstract_values.dropout_survey.glm import GLMS
    prim = {ds: next(iter(v.events)) for ds, v in GLMS.items()}
    g["primary"] = g.apply(lambda x: x.event == prim.get(x.dataset), axis=1)
    return g, meta, r, prim


def subject_table(g, r):
    """One row per (dataset, subject, event): mOFC GLM metrics + survey metrics."""
    w = g[g.roi.isin(["mOFC", "cortex", "WM", "dmPFC"])].pivot_table(
        index=["dataset", "subject", "event", "primary"], columns="roi",
        values=["r2_above_floor", "frac_r2_above_wm95", "beta_t", "beta_mean",
                "frac_vox_abs_t3", "frac_vox_no_glm"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    s = r[r.roi == "mOFC"][["dataset", "subject", "tsnr_norm", "frac_void", "tsnr_median"]]
    w = w.merge(s, on=["dataset", "subject"], how="left")
    w["abs_t_mOFC"] = w["beta_t_mOFC"].abs()
    w["abs_t_cortex"] = w["beta_t_cortex"].abs()
    # detectability relative to the same subject's white matter (null-ish floor)
    w["t3_excess_mOFC"] = w["frac_vox_abs_t3_mOFC"] - w["frac_vox_abs_t3_WM"]
    w["t3_excess_cortex"] = w["frac_vox_abs_t3_cortex"] - w["frac_vox_abs_t3_WM"]
    w["group"] = w.dataset.map({d: k for k, v in GROUPS for d in v})
    return w


def relations(w):
    rows = []
    pairs = [("abs_t_mOFC", "tsnr_norm"), ("abs_t_mOFC", "frac_void"),
             ("r2_above_floor_mOFC", "tsnr_norm"), ("r2_above_floor_mOFC", "frac_void"),
             ("frac_r2_above_wm95_mOFC", "tsnr_norm"), ("frac_r2_above_wm95_mOFC", "frac_void"),
             ("t3_excess_mOFC", "tsnr_norm"), ("t3_excess_mOFC", "frac_void")]
    p = w[w.primary]
    for y, x in pairs:
        zs = []
        for ds, d in p.groupby("dataset"):
            d = d[[x, y]].dropna()
            if len(d) >= 8:
                rho, pv = stats.spearmanr(d[x], d[y])
                rows.append(dict(level="within", dataset=ds, y=y, x=x, n=len(d), rho=rho, p=pv))
                zs.append(np.arctanh(rho))
        if zs:
            t = stats.ttest_1samp(zs, 0)
            rows.append(dict(level="within_pooled", dataset=f"{len(zs)} datasets", y=y, x=x,
                             n=len(zs), rho=float(np.tanh(np.mean(zs))), p=t.pvalue))
        med = p.groupby("dataset")[[x, y]].median().dropna()
        rho, pv = stats.spearmanr(med[x], med[y])
        rows.append(dict(level="across_medians", dataset="all", y=y, x=x, n=len(med), rho=rho,
                         p=pv))
        d = p[[x, y]].dropna()
        rho, pv = stats.spearmanr(d[x], d[y])
        rows.append(dict(level="pooled_subjects", dataset="all", y=y, x=x, n=len(d), rho=rho,
                         p=pv))
    return pd.DataFrame(rows)


def group_tests(g):
    """Across participants: is the ROI-mean beta different from 0 (one-sample t), and how
    many participants have |t across trials| > 2 -- per dataset, event and ROI."""
    rows = []
    wm = g[g.roi == "WM"].set_index(["dataset", "event", "subject"]).beta_mean
    for (ds, e, roi), d in g.groupby(["dataset", "event", "roi"]):
        v = d.beta_mean.dropna()
        if len(v) < 3:
            continue
        t = stats.ttest_1samp(v, 0)
        # relative to the same participant's white matter: global / baseline shifts
        # in the single-trial betas move WM too
        dv = (d.set_index(["dataset", "event", "subject"]).beta_mean - wm).dropna()
        tw = stats.ttest_1samp(dv, 0) if roi != "WM" else None
        rows.append(dict(dataset=ds, event=e, roi=roi, n=len(v), beta_mean=v.mean(),
                         group_t=t.statistic, group_p=t.pvalue,
                         beta_minus_wm=dv.mean(),
                         group_t_vs_wm=tw.statistic if tw else np.nan,
                         group_p_vs_wm=tw.pvalue if tw else np.nan,
                         frac_subj_abs_t2=float((d.beta_t.abs() > 2).mean()),
                         frac_r2_above_wm95=d.frac_r2_above_wm95.median(),
                         r2_above_floor=d.r2_above_floor.median()))
    return pd.DataFrame(rows)


def strip(ax, d, col, xlim, clip=None):
    for k, r in enumerate(ROIS):
        v = d.loc[d.roi == r, col].dropna().to_numpy()
        if clip is not None:
            v = np.clip(v, -clip, clip)
        yy = -k + np.random.default_rng(k).uniform(-0.22, 0.22, len(v))
        ax.scatter(v, yy, s=3, color=COL[r], alpha=0.45, lw=0, clip_on=False)
        if len(v):
            ax.plot([np.median(v)] * 2, [-k - 0.4, -k + 0.4], color=COL[r], lw=1.8,
                    solid_capstyle="butt")
    ax.set_ylim(-len(ROIS) + 0.4, 0.6)
    ax.set_xlim(*xlim)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)


def main():
    style()
    g, meta, r, prim = load()
    w = subject_table(g, r)
    rel = relations(w)
    med = w.groupby(["dataset", "event"]).median(numeric_only=True).reset_index()
    med.insert(2, "n", w.groupby(["dataset", "event"]).size().values)
    out = pd.concat([med.assign(table="dataset_medians"), rel.assign(table="relations")])
    out.to_csv(DATA / "dropout_survey_glm_stats.tsv", sep="\t", index=False)
    w.to_csv(DATA / "dropout_survey_glm_subjects.tsv", sep="\t", index=False)
    gt = group_tests(g)
    gt.to_csv(DATA / "dropout_survey_glm_group.tsv", sep="\t", index=False)

    have = set(g.dataset)
    order = [d for _, grp in GROUPS for d in grp if d in have]
    firsts = {[d for d in grp if d in have][0] for _, grp in GROUPS if any(d in have for d in grp)}
    n = len(order)
    row_h, top, scat_h, bot = 0.6, 0.45, 2.2, 0.15
    H = top + n * row_h + 0.55 + scat_h + bot
    fig = plt.figure(figsize=(7.25, H))
    y_of = lambda i: 1 - (top + (i + 1) * row_h) / H
    hh = (row_h - 0.07) / H
    xs = dict(text=0.0, r2=0.33, t=0.66)
    wd = 0.27
    gp = g[g.primary]
    axes_r, axes_t = [], []
    for i, ds in enumerate(order):
        y = y_of(i)
        d = gp[gp.dataset == ds]
        nsub = d.subject.nunique()
        m = meta[(meta.dataset == ds) & (meta.event == prim[ds])]
        nt = int(m.n_trials.median()) if len(m) else 0
        fig.text(xs["text"], y + hh * 0.86, f"{DATASETS[ds].label} (n = {nsub})", fontsize=7,
                 fontweight="bold" if ds == HILITE else "normal", va="center")
        note = m.note.iloc[0].split(",")[0].split(" (")[0] if len(m) else ""
        nog = d.loc[d.roi == "mOFC", "frac_vox_no_glm"].median()
        extra = f" · {100 * nog:.0f}% of mOFC unmodelled" if nog > 0.02 else ""
        fig.text(xs["text"], y + hh * 0.55, f"{note} · {prim[ds]} ({nt} trials){extra}",
                 fontsize=5.6, color="0.3", va="center")
        q = gt[(gt.dataset == ds) & (gt.event == prim[ds]) & (gt.roi == "mOFC")]
        if len(q):
            q = q.iloc[0]
            fig.text(xs["text"], y + hh * 0.24, f"mOFC − WM β across participants: "
                     f"t({int(q.n) - 1}) = {q.group_t_vs_wm:.1f}", fontsize=5.6,
                     color="0.3", va="center")
        if ds in firsts:
            grp = [k for k, v in GROUPS if ds in v][0]
            fig.text(xs["r2"] - 0.02, y + hh * 0.8, grp, fontsize=6, color="0.45", ha="right",
                     va="center", style="italic")
            if i > 0:
                yl = y + hh + 0.04 / H
                fig.add_artist(plt.Line2D([0, 1], [yl, yl], color="0.75", lw=0.5))
        ax = fig.add_axes([xs["r2"], y, wd, hh])
        n_r2 = d.loc[d.roi == "mOFC", "r2_above_floor"].notna().sum()
        if n_r2 >= 8:
            strip(ax, d, "r2_above_floor", (-2, 6))
        else:     # npy R2 could not be matched to the beta NIfTI for most participants
            strip(ax, d.assign(r2_above_floor=np.nan), "r2_above_floor", (-2, 6))
            ax.text(3.0, -2, f"R² verifiable for {n_r2} of {nsub}", fontsize=5.8, color="0.4",
                    ha="center", va="center")
        ax.axvline(0, color="0.6", lw=0.6, ls=(0, (3, 2)), zorder=0)
        axes_r.append(ax)
        ax = fig.add_axes([xs["t"], y, wd, hh])
        strip(ax, d, "beta_t", (-TCLIP, TCLIP), clip=TCLIP)
        ax.axvline(0, color="0.6", lw=0.6, ls=(0, (3, 2)), zorder=0)
        for v in (-2, 2):
            ax.axvline(v, color="0.85", lw=0.5, zorder=0)
        axes_t.append(ax)
    for ax in axes_r[:-1] + axes_t[:-1]:
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
    for ax in (axes_r[-1], axes_t[-1]):
        sns.despine(ax=ax, left=True, offset=3)
    axes_r[-1].set_xlabel("R² − white-matter R² (%)")
    axes_r[-1].set_xticks([-2, 0, 2, 4, 6])
    axes_t[-1].set_xticks([-20, -10, 0, 10, 20])
    axes_t[-1].set_xticklabels(["≤ −20", "−10", "0", "10", "≥ 20"])
    axes_t[-1].set_xlabel("ROI-mean single-trial β, t across trials")
    for k, roi in enumerate(ROIS):
        axes_t[0].text(TCLIP * 1.04, -k, "Cortex" if roi == "cortex" else roi, color=COL[roi],
                       fontsize=6, va="center", ha="left")
    yh = 1 - (top - 0.1) / H
    for ltr, x, t in (("a", xs["text"], "Dataset, GLM and event"),
                      ("b", xs["r2"], "GLMsingle R² above the WM floor"),
                      ("c", xs["t"], "Single-trial response")):
        fig.text(x, yh, ltr, fontsize=8, fontweight="bold", va="bottom")
        fig.text(x + 0.018, yh, t, fontsize=7.5, va="bottom")

    # d, e: across participants, mOFC detectability vs dropout
    p = w[w.primary]
    yb = (bot + 0.3) / H
    hs = (scat_h - 0.45) / H
    specs = [("d", "tsnr_norm", "abs_t_mOFC", "mOFC tSNR / cortex median", "mOFC |β t|", 0.075),
             ("e", "frac_void", "frac_r2_above_wm95_mOFC", "mOFC signal-void fraction",
              "mOFC voxels with R² > WM 95th pct", 0.53)]
    for ltr, x, y, xl, yl, x0 in specs:
        ax = fig.add_axes([x0, yb, 0.33, hs])
        for grp, c in GCOL.items():
            q = p[p.group == grp]
            ax.scatter(q[x], q[y], s=4, color=c, alpha=0.35, lw=0)
        mm = p.groupby(["dataset", "group"])[[x, y]].median().reset_index()
        for _, row in mm.iterrows():
            ax.scatter(row[x], row[y], s=22, color=GCOL[row.group], edgecolor="k",
                       lw=0.8 if row.dataset == HILITE else 0.3, zorder=3)
        sns.despine(ax=ax, offset=3)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        rr = rel[(rel.x == x) & (rel.y == y)].set_index("level")
        txt = (f"Across datasets ρ = {rr.loc['across_medians', 'rho']:.2f}"
               f" (p = {rr.loc['across_medians', 'p']:.2g})\n"
               f"Within datasets mean ρ = {rr.loc['within_pooled', 'rho']:.2f}"
               f" (p = {rr.loc['within_pooled', 'p']:.2g})")
        ax.text(0.02, 1.0, txt, transform=ax.transAxes, fontsize=5.8, va="bottom", color="0.25")
        fig.text(x0 - 0.06, yb + hs + 0.12 / H, ltr, fontsize=8, fontweight="bold", va="bottom")
        if ltr == "d":
            ax.set_yscale("symlog", linthresh=2)
            ax.set_yticks([0, 2, 5, 10, 20])
            ax.set_yticklabels(["0", "2", "5", "10", "20"])
    kx = 0.885
    for k, (grp, c) in enumerate(GCOL.items()):
        fig.text(kx, yb + hs - k * 0.12 / H, grp, color=c, fontsize=6.5, ha="left", va="top")
    ky = yb + hs - 3.6 * 0.12 / H
    for k, (lab, mew, ms, a) in enumerate((("Participant", 0, 2.2, 0.5),
                                           ("Dataset median", 0.3, 4.5, 1),
                                           ("Abstract values", 0.9, 4.5, 1))):
        yy = ky - k * 0.12 / H
        fig.add_artist(plt.Line2D([kx + 0.008], [yy], marker="o", ms=ms, mfc="0.55", mec="k",
                                  mew=mew, alpha=a, transform=fig.transFigure))
        fig.text(kx + 0.022, yy, lab, color="0.35", fontsize=6, ha="left", va="center")
    fig.savefig(FIG / "fig_dropout_survey_glm.pdf")
    fig.savefig(FIG / "fig_dropout_survey_glm.png", dpi=200)
    plt.close(fig)
    pd.set_option("display.width", 250)
    cols = ["dataset", "event", "n", "r2_above_floor_mOFC", "frac_r2_above_wm95_mOFC",
            "beta_mean_mOFC", "beta_t_mOFC", "frac_vox_abs_t3_mOFC", "frac_vox_abs_t3_WM",
            "beta_t_cortex", "r2_above_floor_cortex", "frac_vox_no_glm_mOFC", "tsnr_norm",
            "frac_void"]
    print(med[[c for c in cols if c in med]].round(3).to_string())
    print(rel.round(3).to_string())
    print(gt[gt.roi.isin(["mOFC", "dmPFC", "WM", "cortex"])].pivot_table(
        index=["dataset", "event"], columns="roi", values=["group_t", "group_t_vs_wm"]).round(1)
        .to_string())


if __name__ == "__main__":
    main()
