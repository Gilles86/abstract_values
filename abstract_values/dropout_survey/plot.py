"""Is the vmPFC/OFC dropout specific to the abstract-values protocol? -- local plotting.

Reads the small summaries written by combine.py (rsynced to notes/data) and
draws one row per dataset: acquisition parameters, group-mean normalised tSNR
(voxel tSNR / the subject's whole-cortex median, MNI sagittal x = -5 and axial
z = -14, one colour scale), subject-level normalised ROI tSNR and the
signal-void fraction (mean EPI < 0.5 x brain median) of medial and lateral OFC.
Also writes the dataset-level statistics to notes/data/dropout_survey_stats.tsv.

    python -m abstract_values.dropout_survey.plot      # abstract_values env
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from abstract_values.dropout_survey.datasets import DATASETS

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "notes" / "data"
FIG = REPO / "notes" / "figures"
ROIS = ["mOFC", "lOFC", "pgACC", "dmPFC"]
COL = {"mOFC": "#D9730D", "lOFC": "#EDAA5E", "pgACC": "#A8431A", "dmPFC": "#6D4C9F"}
NORM_CMAP = LinearSegmentedColormap.from_list(
    "tsnr", ["#2b0a3d", "#7a1f5c", "#c4384a", "#ee7b30", "#fbd35a", "#fdf6c9"])
VMAX = 1.4
HILITE = "abstractvalue"


def style():
    plt.rcParams.update({
        "font.family": "Helvetica", "font.size": 7, "axes.labelsize": 7.5,
        "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3,
        "ytick.major.size": 3, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "legend.frameon": False})


def fnum(v, fmt="{:g}"):
    try:
        v = float(str(v).split()[0])
    except (TypeError, ValueError):
        return "?"
    return "?" if np.isnan(v) else fmt.format(v)


PE = {"j": "PA", "j-": "AP", "i": "LR", "i-": "RL"}   # BIDS j- = posterior->anterior? see note


def acq_lines(a):
    """Three short lines of acquisition parameters for one dataset row."""
    if a is None:
        return ["", "", ""]
    vendor = str(a.get("Manufacturer") or "").split()[0].replace("SIEMENS", "Siemens") \
        .replace("Philips", "Philips").split("(")[0]
    model = str(a.get("ManufacturersModelName") or "").split("(")[0].strip()
    field = fnum(a.get("MagneticFieldStrength"))
    tr = fnum(a.get("RepetitionTime"), "{:.2g}")
    te = fnum(1000 * float(str(a.get("EchoTime")).split()[0]) if a.get("EchoTime") not in
              (None, "nan") and str(a.get("EchoTime")) != "nan" else np.nan, "{:.0f}")
    vx = "×".join(fnum(a.get(k), "{:.2g}") for k in ("vox_x", "vox_y", "vox_z"))
    mb = fnum(a.get("MultibandAccelerationFactor"))
    pe = str(a.get("PhaseEncodingDirection") or "?").split()[0]
    tilt = fnum(a.get("tilt_mni_deg"), "{:.0f}")
    l1 = f"{field} T {vendor} {model}".replace("nan", "").strip()
    l2 = f"TR {tr} s · TE {te} ms · MB {mb}"
    l3 = f"{vx} mm · PE {pe} · tilt {tilt}°"
    return [l1, l2, l3]


def mni_view(maps, ds, view):
    v = maps[f"{ds}_tsnrnorm_{view}"]
    cov = maps[f"{ds}_coverage_{view}"]
    tpl = maps[f"tpl_{view}"]
    brain = maps[f"mask_{view}"]
    return tpl, np.where(brain & (cov >= 0.5), v, np.nan), brain


def draw_slice(ax, tpl, over, brain):
    lo, hi = np.percentile(tpl[brain], [1, 99.5])
    ax.imshow(tpl.T, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    im = ax.imshow(np.ma.masked_invalid(over).T, origin="lower", cmap=NORM_CMAP, vmin=0,
                   vmax=VMAX, interpolation="nearest")
    ii = np.argwhere(brain)
    ax.set_xlim(ii[:, 0].min() - 2, ii[:, 0].max() + 2)
    ax.set_ylim(ii[:, 1].min() - 2, ii[:, 1].max() + 2)
    ax.set_axis_off()
    return im


def dataset_stats(rois):
    """Subject-level normalised tSNR / void per dataset; abstractvalue vs each other."""
    rows = []
    for roi in ROIS + ["WM"]:
        for metric in ("tsnr_norm", "frac_void", "tsnr_median", "frac_outside_fov"):
            d = rois[rois.roi == roi]
            ref = d.loc[d.dataset == HILITE, metric].dropna()
            groups = [g[metric].dropna().to_numpy() for _, g in d.groupby("dataset")]
            kw = stats.kruskal(*[g for g in groups if len(g) > 1])
            for ds, g in d.groupby("dataset"):
                v = g[metric].dropna()
                r = dict(roi=roi, metric=metric, dataset=ds, n=len(v), median=v.median(),
                         q25=v.quantile(.25), q75=v.quantile(.75), mean=v.mean(),
                         kruskal_H=kw.statistic, kruskal_p=kw.pvalue)
                if ds != HILITE and len(v) > 1:
                    u = stats.mannwhitneyu(ref, v, alternative="two-sided")
                    r.update(mwu_vs_abstractvalue_U=u.statistic, mwu_vs_abstractvalue_p=u.pvalue,
                             # rank-biserial: >0 means abstractvalue higher
                             rank_biserial=2 * u.statistic / (len(ref) * len(v)) - 1)
                rows.append(r)
    out = pd.DataFrame(rows)
    # mOFC / dmPFC ratio per subject
    w = rois.pivot_table(index=["dataset", "subject"], columns="roi", values="tsnr_median")
    ratio = (w["mOFC"] / w["dmPFC"]).rename("value").reset_index()
    for ds, g in ratio.groupby("dataset"):
        r = dict(roi="mOFC/dmPFC", metric="tsnr_ratio", dataset=ds, n=len(g),
                 median=g.value.median(), q25=g.value.quantile(.25), q75=g.value.quantile(.75),
                 mean=g.value.mean())
        if ds != HILITE:
            u = stats.mannwhitneyu(ratio.loc[ratio.dataset == HILITE, "value"], g.value)
            r.update(mwu_vs_abstractvalue_U=u.statistic, mwu_vs_abstractvalue_p=u.pvalue,
                     rank_biserial=2 * u.statistic / ((ratio.dataset == HILITE).sum() * len(g)) - 1)
        out = pd.concat([out, pd.DataFrame([r])], ignore_index=True)
    return out


def main():
    style()
    rois = pd.read_csv(DATA / "dropout_survey_rois.tsv", sep="\t", dtype={"subject": str})
    acq = pd.read_csv(DATA / "dropout_survey_acquisition.tsv", sep="\t").set_index("dataset")
    maps = np.load(DATA / "dropout_survey_maps.npz", allow_pickle=True)
    st = dataset_stats(rois)
    st.to_csv(DATA / "dropout_survey_stats.tsv", sep="\t", index=False)

    # order: abstractvalue first, then by median normalised mOFC tSNR (low = worse)
    m = st[(st.roi == "mOFC") & (st.metric == "tsnr_norm")].set_index("dataset")["median"]
    order = [HILITE] + [d for d in m.drop(HILITE).sort_values().index]
    order = [d for d in order if f"{d}_tsnrnorm_sag" in maps.files]
    n = len(order)

    row_h = 0.62
    top, bot = 0.42, 0.42
    H = top + bot + n * row_h
    fig = plt.figure(figsize=(7.25, H))
    y_of = lambda i: 1 - (top + (i + 1) * row_h) / H     # bottom of row i (figure fraction)
    hh = (row_h - 0.06) / H

    xs = dict(text=0.005, sag=0.255, ax=0.36, strip=0.505, void=0.80)
    ws = dict(sag=0.10, ax=0.085, strip=0.255, void=0.18)
    axes_s, axes_v = [], []
    im = None
    for i, ds in enumerate(order):
        y = y_of(i)
        nsub = rois[rois.dataset == ds].subject.nunique()
        a = acq.loc[ds].to_dict() if ds in acq.index else None
        lines = acq_lines(a)
        fw = "bold" if ds == HILITE else "normal"
        fig.text(xs["text"], y + hh * 0.86, f"{DATASETS[ds].label} (n = {nsub})", fontsize=7,
                 fontweight=fw, va="center")
        for k, l in enumerate(lines):
            fig.text(xs["text"], y + hh * (0.6 - 0.24 * k), l, fontsize=5.8, color="0.3",
                     va="center")
        for view in ("sag", "ax"):
            ax = fig.add_axes([xs[view], y, ws[view], hh])
            tpl, over, brain = mni_view(maps, ds, view)
            im = draw_slice(ax, tpl, over, brain)
        # strip of normalised tSNR, four ROIs as sub-rows
        ax = fig.add_axes([xs["strip"], y, ws["strip"], hh])
        d = rois[rois.dataset == ds]
        for k, r in enumerate(ROIS):
            v = d.loc[d.roi == r, "tsnr_norm"].dropna().to_numpy()
            yy = -k + np.random.default_rng(k).uniform(-0.22, 0.22, len(v))
            ax.scatter(v, yy, s=3, color=COL[r], alpha=0.45, lw=0)
            if len(v):
                ax.plot([np.median(v)] * 2, [-k - 0.38, -k + 0.38], color=COL[r], lw=1.8,
                        solid_capstyle="butt")
        ax.axvline(1, color="0.6", lw=0.6, ls=(0, (3, 2)), zorder=0)
        ax.set_ylim(-3.6, 0.6)
        ax.set_xlim(0, 1.6)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        axes_s.append(ax)
        # void fraction, mOFC and lOFC
        ax = fig.add_axes([xs["void"], y, ws["void"], hh])
        for k, r in enumerate(["mOFC", "lOFC"]):
            v = d.loc[d.roi == r, "frac_void"].dropna().to_numpy()
            yy = -k + np.random.default_rng(k).uniform(-0.25, 0.25, len(v))
            ax.scatter(v, yy, s=3, color=COL[r], alpha=0.45, lw=0)
            if len(v):
                ax.plot([np.median(v)] * 2, [-k - 0.4, -k + 0.4], color=COL[r], lw=1.8,
                        solid_capstyle="butt")
        cov = d.loc[d.roi == "mOFC", "frac_outside_fov"].median()
        if cov > 0.02:
            ax.text(1.0, -0.5, f"mOFC {100 * cov:.0f}% outside FOV", fontsize=5.5,
                    color="0.35", ha="right", va="center")
        ax.set_ylim(-1.6, 0.6)
        ax.set_xlim(0, 0.8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        axes_v.append(ax)
    for ax in axes_s[:-1] + axes_v[:-1]:
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
    for ax in (axes_s[-1], axes_v[-1]):
        sns.despine(ax=ax, left=True, offset=3)
    axes_s[-1].set_xticks([0, 0.5, 1, 1.5])
    axes_s[-1].set_xlabel("tSNR / whole-cortex median")
    axes_v[-1].set_xticks([0, 0.4, 0.8])
    axes_v[-1].set_xticklabels(["0", "40", "80"])
    axes_v[-1].set_xlabel("Signal-void voxels (%)")
    # direct ROI labels on the first row
    for k, r in enumerate(ROIS):
        axes_s[0].text(1.62, -k, r, color=COL[r], fontsize=6, va="center", ha="left")
    for k, r in enumerate(["mOFC", "lOFC"]):
        axes_v[0].text(0.82, -k, r, color=COL[r], fontsize=6, va="center", ha="left")
    axes_s[0].text(1.0, 0.75, "Cortex median", fontsize=5.8, color="0.45", ha="center",
                   va="bottom")

    # headers
    yh = 1 - (top - 0.06) / H
    fig.text(xs["text"], yh, "Dataset and acquisition", fontsize=7.5, fontweight="bold",
             va="bottom")
    fig.text(xs["sag"] + 0.09, yh, "Group normalised tSNR (x = −5, z = −14)", fontsize=7.5,
             fontweight="bold", va="bottom", ha="center")
    fig.text(xs["strip"] + ws["strip"] / 2, yh, "Normalised tSNR per participant",
             fontsize=7.5, fontweight="bold", va="bottom", ha="center")
    fig.text(xs["void"] + ws["void"] / 2, yh, "Signal void (EPI < 50% of brain median)",
             fontsize=7.5, fontweight="bold", va="bottom", ha="center")
    for ltr, x in zip("abcd", (xs["text"], xs["sag"], xs["strip"], xs["void"])):
        fig.text(x - 0.0, yh + 0.13 / H, ltr, fontsize=8, fontweight="bold", va="bottom")
    cax = fig.add_axes([xs["sag"] + 0.015, 0.25 / H, 0.16, 0.06 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1, VMAX])
    cb.outline.set_visible(False)
    cax.tick_params(labelsize=6, length=2)
    cb.set_label("tSNR / cortex median", fontsize=6.5, labelpad=1)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig_dropout_survey.pdf")
    fig.savefig(FIG / "fig_dropout_survey.png", dpi=200)
    plt.close(fig)
    show = st[(st.metric.isin(["tsnr_norm", "frac_void", "tsnr_ratio"])) &
              (st.roi.isin(["mOFC", "mOFC/dmPFC"]))]
    print(show[["roi", "metric", "dataset", "n", "median", "q25", "q75",
                "mwu_vs_abstractvalue_p", "rank_biserial"]].round(3).to_string())


if __name__ == "__main__":
    main()
