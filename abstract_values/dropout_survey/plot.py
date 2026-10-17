"""Is the vmPFC/OFC dropout specific to the abstract-values protocol? -- local plotting.

Reads the small summaries written by combine.py (rsynced to notes/data) and
draws one row per dataset, grouped by scanner: acquisition parameters,
group-mean normalised tSNR (voxel tSNR / the subject's whole-cortex median;
MNI sagittal x = -5, frontal half, and axial z = -14; one colour scale),
subject-level normalised ROI tSNR, and the signal-void fraction (mean EPI <
0.5 x brain median) of medial and lateral OFC. Also writes the dataset-level
statistics to notes/data/dropout_survey_stats.tsv.

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
GROUPS = [("Siemens 3T", ["abstractvalue", "numloss", "foodvalue"]),
          ("Philips 3T", ["numrisk", "dnumrisk", "stressrisk", "neuralpriors", "tmsrisk",
                          "risk3t"]),
          ("7T", ["risk7t", "retsupp", "valuecapture"])]


def style():
    plt.rcParams.update({
        "font.family": "Helvetica", "font.size": 7, "axes.labelsize": 7.5,
        "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3,
        "ytick.major.size": 3, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "legend.frameon": False})


def _num(v):
    try:
        v = float(str(v).split()[0])
    except (TypeError, ValueError):
        return np.nan
    return v


def acq_values(a):
    """Field, vendor, model, TR, TE (ms), MB, in-plane, slice, gap, PE axis, pitch."""
    out = {}
    man = a.get("Manufacturer")
    out["vendor"] = "" if not isinstance(man, str) else man.split()[0].replace("SIEMENS", "Siemens")
    model = a.get("ManufacturersModelName")
    out["model"] = "" if not isinstance(model, str) else model.split("(")[0].replace(
        "MAGNETOM ", "").strip()
    field = _num(a.get("MagneticFieldStrength"))
    out["field"] = 3.0 if np.isnan(field) else field   # numloss: sidecars unreadable, SNS 3T
    te = _num(a.get("EchoTime")) * 1000
    out["te"] = _num(a.get("par_TE_ms")) if np.isnan(te) else te
    out["mb"] = _num(a.get("MultibandAccelerationFactor"))
    out["tr"] = _num(a.get("RepetitionTime"))
    thick, gap = _num(a.get("SliceThickness")), np.nan
    if not np.isnan(_num(a.get("SpacingBetweenSlices"))) and not np.isnan(thick):
        gap = _num(a.get("SpacingBetweenSlices")) - thick
    if not np.isnan(_num(a.get("par_thick"))):
        thick, gap = _num(a.get("par_thick")), _num(a.get("par_gap"))
    out["inplane"] = _num(a.get("vox_x"))
    out["slice"] = _num(a.get("vox_z")) if np.isnan(thick) else thick
    out["gap"] = gap
    pe = str(a.get("PhaseEncodingDirection") or "")
    out["pe"] = {"j": "AP", "j-": "AP", "i": "LR", "i-": "LR"}.get(pe, "?")
    out["pitch"] = _num(a.get("pitch_mni_deg"))
    return out


def acq_lines(a):
    if a is None:
        return []
    v = acq_values(a)
    f = lambda x, fmt="{:g}": "?" if x != x else fmt.format(x)
    sl = f"{f(v['inplane'], '{:.2g}')} mm in-plane, {f(v['slice'], '{:.2g}')} mm slices"
    if v["gap"] == v["gap"] and v["gap"] > 0.05:
        sl += f" + {v['gap']:.1f} gap"
    p = v["pitch"]
    pitch = "?" if p != p else f"{'+' if p >= 0 else '−'}{abs(p):.0f}°"
    return [f"{f(v['field'])} T {v['vendor']} {v['model']}".strip(),
            f"TR {f(v['tr'], '{:.2g}')} s · TE {f(v['te'], '{:.0f}')} ms · MB {f(v['mb'])}",
            sl,
            f"PE axis {v['pe']} · slab pitch {pitch}"]


def mni_view(maps, ds, view):
    v = maps[f"{ds}_tsnrnorm_{view}"]
    cov = maps[f"{ds}_coverage_{view}"]
    brain = maps[f"mask_{view}"]
    return maps[f"tpl_{view}"], np.where(brain & (cov >= 0.5), v, np.nan), brain


def draw_slice(ax, tpl, over, brain, crop=None):
    lo, hi = np.percentile(tpl[brain], [1, 99.5])
    ax.imshow(tpl.T, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    im = ax.imshow(np.ma.masked_invalid(over).T, origin="lower", cmap=NORM_CMAP, vmin=0,
                   vmax=VMAX, interpolation="nearest")
    if crop is None:
        ii = np.argwhere(brain)
        crop = ((ii[:, 0].min() - 2, ii[:, 0].max() + 2), (ii[:, 1].min() - 2, ii[:, 1].max() + 2))
    ax.set_xlim(crop[0])
    ax.set_ylim(crop[1])
    ax.set_axis_off()
    return im


def strip(ax, d, rois, col, xlim):
    for k, r in enumerate(rois):
        v = d.loc[d.roi == r, col].dropna().to_numpy()
        yy = -k + np.random.default_rng(k).uniform(-0.22, 0.22, len(v))
        ax.scatter(v, yy, s=3, color=COL[r], alpha=0.45, lw=0, clip_on=False)
        if len(v):
            ax.plot([np.median(v)] * 2, [-k - 0.4, -k + 0.4], color=COL[r], lw=1.8,
                    solid_capstyle="butt")
    ax.set_ylim(-len(rois) + 0.4, 0.6)
    ax.set_xlim(*xlim)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)


def dataset_stats(rois):
    """Subject-level metrics per dataset; abstractvalue vs each other dataset (Mann-Whitney)."""
    rows = []
    for roi in ROIS + ["WM", "PCC", "pericalcarine", "superiorparietal"]:
        for metric in ("tsnr_norm", "frac_void", "tsnr_median", "frac_outside_fov",
                       "frac_outside_mask", "frac_tsnr_below20"):
            d = rois[rois.roi == roi]
            ref = d.loc[d.dataset == HILITE, metric].dropna()
            groups = [g[metric].dropna().to_numpy() for _, g in d.groupby("dataset")]
            groups = [g for g in groups if len(g) > 1 and np.ptp(g) > 0]
            kw = stats.kruskal(*groups) if len(groups) > 1 else None
            for ds, g in d.groupby("dataset"):
                v = g[metric].dropna()
                r = dict(roi=roi, metric=metric, dataset=ds, n=len(v), median=v.median(),
                         q25=v.quantile(.25), q75=v.quantile(.75), mean=v.mean(),
                         kruskal_H=kw.statistic if kw else np.nan,
                         kruskal_p=kw.pvalue if kw else np.nan)
                if ds != HILITE and len(v) > 1:
                    u = stats.mannwhitneyu(ref, v, alternative="two-sided")
                    r.update(mwu_vs_abstractvalue_U=u.statistic, mwu_vs_abstractvalue_p=u.pvalue,
                             # rank-biserial: > 0 means abstractvalue higher
                             rank_biserial=2 * u.statistic / (len(ref) * len(v)) - 1)
                rows.append(r)
    out = pd.DataFrame(rows)
    w = rois.pivot_table(index=["dataset", "subject"], columns="roi", values="tsnr_median")
    ratio = (w["mOFC"] / w["dmPFC"]).rename("value").reset_index()
    ref = ratio.loc[ratio.dataset == HILITE, "value"]
    extra = []
    for ds, g in ratio.groupby("dataset"):
        r = dict(roi="mOFC/dmPFC", metric="tsnr_ratio", dataset=ds, n=len(g),
                 median=g.value.median(), q25=g.value.quantile(.25),
                 q75=g.value.quantile(.75), mean=g.value.mean())
        if ds != HILITE:
            u = stats.mannwhitneyu(ref, g.value, alternative="two-sided")
            r.update(mwu_vs_abstractvalue_U=u.statistic, mwu_vs_abstractvalue_p=u.pvalue,
                     rank_biserial=2 * u.statistic / (len(ref) * len(g)) - 1)
        extra.append(r)
    return pd.concat([out, pd.DataFrame(extra)], ignore_index=True)


def main():
    style()
    rois = pd.read_csv(DATA / "dropout_survey_rois.tsv", sep="\t", dtype={"subject": str})
    acq = pd.read_csv(DATA / "dropout_survey_acquisition.tsv", sep="\t").set_index("dataset")
    maps = np.load(DATA / "dropout_survey_maps.npz", allow_pickle=True)
    st = dataset_stats(rois)
    st.to_csv(DATA / "dropout_survey_stats.tsv", sep="\t", index=False)
    table = pd.DataFrame([dict(dataset=d, **acq_values(acq.loc[d].to_dict()))
                          for d in acq.index])
    table.to_csv(DATA / "dropout_survey_acquisition_compact.tsv", sep="\t", index=False)

    have = set(rois.dataset)
    order = [d for _, g in GROUPS for d in g if d in have]
    firsts = {[d for d in g if d in have][0] for _, g in GROUPS if any(d in have for d in g)}
    n = len(order)
    row_h, top, bot = 0.68, 0.48, 0.55
    H = top + bot + n * row_h
    fig = plt.figure(figsize=(7.25, H))
    y_of = lambda i: 1 - (top + (i + 1) * row_h) / H
    hh = (row_h - 0.08) / H

    xs = dict(text=0.0, sag=0.268, ax=0.392, strip=0.53, void=0.835)
    ws = dict(sag=0.118, ax=0.085, strip=0.24, void=0.125)
    aff = maps["affine"]
    ij = lambda y, z: np.linalg.solve(aff, [0, y, z, 1])[1:3]
    (j0, k0), (j1, k1) = ij(-45, -50), ij(80, 55)
    sag_crop = ((min(j0, j1), max(j0, j1)), (min(k0, k1), max(k0, k1)))
    axes_s, axes_v, im = [], [], None
    for i, ds in enumerate(order):
        y = y_of(i)
        d = rois[rois.dataset == ds]
        nsub = d.subject.nunique()
        a = acq.loc[ds].to_dict() if ds in acq.index else None
        fig.text(xs["text"], y + hh * 0.93, f"{DATASETS[ds].label} (n = {nsub})", fontsize=7,
                 fontweight="bold" if ds == HILITE else "normal", va="center")
        for k, l in enumerate(acq_lines(a)):
            fig.text(xs["text"], y + hh * (0.71 - 0.2 * k), l, fontsize=5.6, color="0.3",
                     va="center")
        if f"{ds}_tsnrnorm_sag" in maps.files:
            for view, crop in (("sag", sag_crop), ("ax", None)):
                ax = fig.add_axes([xs[view], y, ws[view], hh])
                tpl, over, brain = mni_view(maps, ds, view)
                im = draw_slice(ax, tpl, over, brain, crop)
            nm = int(maps[f"{ds}_n"])
            if nm < nsub:
                ax.text(1.03, 0.0, f"n = {nm}", transform=ax.transAxes, fontsize=5.5,
                        color="0.4", va="bottom")
        ax = fig.add_axes([xs["strip"], y, ws["strip"], hh])
        strip(ax, d, ROIS, "tsnr_norm", (0, 1.6))
        ax.axvline(1, color="0.6", lw=0.6, ls=(0, (3, 2)), zorder=0)
        axes_s.append(ax)
        ax = fig.add_axes([xs["void"], y + hh * 0.22, ws["void"], hh * 0.56])
        strip(ax, d, ["mOFC", "lOFC"], "frac_void", (0, 0.8))
        axes_v.append(ax)
        if ds in firsts:
            yl = y + hh + 0.045 / H
            grp = [g for g, m in GROUPS if ds in m][0]
            if i > 0:
                fig.add_artist(plt.Line2D([0, 1], [yl, yl], color="0.75", lw=0.5))
            fig.text(1.0, yl - 0.012 / H, grp, fontsize=6, color="0.45", ha="right", va="top")
    for ax in axes_s[:-1] + axes_v[:-1]:
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
    for ax in (axes_s[-1], axes_v[-1]):
        sns.despine(ax=ax, left=True, offset=3)
    axes_s[-1].set_xticks([0, 0.5, 1, 1.5])
    axes_s[-1].set_xlabel("tSNR / whole-cortex median")
    axes_v[-1].set_xticks([0, 0.4, 0.8])
    axes_v[-1].set_xticklabels(["0", "40", "80"])
    axes_v[-1].set_xlabel("Voxels with EPI\n< 50% of brain median (%)")
    for k, r in enumerate(ROIS):
        axes_s[0].text(1.62, -k, r, color=COL[r], fontsize=6, va="center", ha="left")
    for k, r in enumerate(["mOFC", "lOFC"]):
        axes_v[0].text(0.82, -k, r, color=COL[r], fontsize=6, va="center", ha="left")
    axes_s[0].text(1.0, 0.75, "Cortex median", fontsize=5.8, color="0.45", ha="center",
                   va="bottom")

    yh = 1 - (top - 0.12) / H
    for ltr, x, t in (("a", xs["text"], "Dataset and acquisition"),
                      ("b", xs["sag"], "Group tSNR / cortex median"),
                      ("c", xs["strip"], "Normalised tSNR per participant"),
                      ("d", xs["void"], "Signal void")):
        fig.text(x, yh, ltr, fontsize=8, fontweight="bold", va="bottom")
        fig.text(x + 0.018, yh, t, fontsize=7.5, va="bottom")
    fig.text(xs["sag"] + 0.018, yh - 0.15 / H, "MNI x = −5 (anterior two-thirds), z = −14", fontsize=6,
             color="0.4", va="bottom")
    cax = fig.add_axes([xs["sag"] + 0.02, 0.28 / H, 0.17, 0.05 / H])
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
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
