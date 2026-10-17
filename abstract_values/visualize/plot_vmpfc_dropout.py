"""Signal dropout and evoked responses in medial prefrontal cortex.

Two figures from the outputs of collect_vmpfc_dropout / combine_vmpfc_dropout:

  fig_vmpfc_dropout.pdf   where the signal is lost: group tSNR in MNI space
                          with the mPFC subregions outlined, one participant's
                          T1w next to their mean EPI, tSNR and voxel tSNR
                          distributions per ROI, GLMsingle R2 and mean
                          gabor / response-bar betas per ROI.
  fig_vmpfc_response.pdf  is there any evoked response? ROI-mean %-signal time
                          courses locked to the gabor and to the response bar,
                          and the time-resolved effect of value (gabor-locked)
                          and of the bid (bar-locked).

    python -m abstract_values.visualize.plot_vmpfc_dropout       # abstract_values env
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "notes" / "data"
FIG = REPO / "notes" / "figures"

NAMES = {"mOFC": "Medial OFC", "lOFC": "Lateral OFC", "rACC": "Pregenual ACC",
         "dmPFC": "Dorsomedial PFC", "dACC": "Dorsal ACC", "PCC": "Posterior cingulate",
         "Lprecentral": "Left M1", "V1": "V1", "IPS0": "IPS0", "IPS12": "IPS1–2",
         "IPS345": "IPS3–5", "WM": "White matter"}
COL = {"mOFC": "#D9730D", "lOFC": "#EDAA5E", "rACC": "#A8431A", "dmPFC": "#6D4C9F",
       "dACC": "#A889CF", "PCC": "#3B5BA5", "Lprecentral": "#1B8A80", "V1": "#4A4A4A",
       "IPS0": "#8C86B8", "IPS12": "#5B4A9E", "IPS345": "#3A2E6E", "WM": "#A8A8A8"}
SHORT = {"mOFC": "mOFC", "lOFC": "lOFC", "rACC": "pgACC", "dmPFC": "dmPFC", "dACC": "dACC",
         "PCC": "PCC", "Lprecentral": "Left M1", "V1": "V1", "IPS0": "IPS0", "IPS12": "IPS1–2",
         "IPS345": "IPS3–5", "WM": "White matter"}
ORDER = ["mOFC", "lOFC", "rACC", "dmPFC", "dACC", "PCC", "Lprecentral", "V1",
         "IPS0", "IPS12", "IPS345", "WM"]
MPFC = ["mOFC", "lOFC", "rACC", "dmPFC", "dACC"]
REFS = ["PCC", "Lprecentral", "V1", "IPS12", "WM"]
LABEL_ID = {"mOFC": 1, "lOFC": 2, "rACC": 3, "PCC": 4, "dmPFC": 5, "dACC": 6}
TSNR_CMAP = LinearSegmentedColormap.from_list(
    "tsnr", ["#2b0a3d", "#7a1f5c", "#c4384a", "#ee7b30", "#fbd35a", "#fdf6c9"])


def style():
    plt.rcParams.update({
        "font.family": "Helvetica", "font.size": 7, "axes.labelsize": 7.5,
        "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3,
        "ytick.major.size": 3, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "legend.frameon": False})


def letter(ax, s, dx=-0.02, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right")


# ── slices ────────────────────────────────────────────────────────────────────

def show_slice(ax, under, over=None, labels=None, vmax=None, crop=None,
               cmap=TSNR_CMAP, alpha_over=None, lw=0.8):
    """2-D slices given as (horizontal, vertical) arrays; displayed with origin lower."""
    u = np.asarray(under, np.float32)
    lo, hi = np.percentile(u[u > 0], [1, 99.5]) if (u > 0).any() else (0, 1)
    ax.imshow(u.T, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    im = None
    if over is not None:
        o = np.ma.masked_invalid(np.asarray(over, np.float32))
        if alpha_over is not None:
            o = np.ma.masked_where(alpha_over == 0, o)
            alpha_over = None
        im = ax.imshow(o.T, origin="lower", cmap=cmap, vmin=0,
                       vmax=vmax, interpolation="bilinear",
                       alpha=None if alpha_over is None else alpha_over.T)
    if labels is not None:
        for roi, i in LABEL_ID.items():
            m = labels[i - 1] >= 0.5 if np.ndim(labels) == 3 else labels == i
            if m.any():
                ax.contour(m.T.astype(float), levels=[0.5], colors=[COL[roi]],
                           linewidths=lw, origin="lower")
    if crop is not None:
        ax.set_xlim(crop[0]); ax.set_ylim(crop[1])
    ax.set_axis_off()
    return im


def bbox(mask, pad=3):
    ii = np.argwhere(mask)
    return ((ii[:, 0].min() - pad, ii[:, 0].max() + pad),
            (ii[:, 1].min() - pad, ii[:, 1].max() + pad))


def mni_slices(z, x_mm=-5, y_mm=38, z_mm=-14):
    aff = z["affine"]
    inv = np.linalg.inv(aff)
    i, j, k = (np.round(inv @ np.array([x_mm, y_mm, z_mm, 1]))[:3]).astype(int)
    flipx = aff[0, 0] < 0
    out = {}
    for name in ("template", "tsnr", "brain"):
        v = z[name]
        if flipx:
            v = v[::-1]
        out[name] = {"sag": v[i if not flipx else v.shape[0] - 1 - i], "cor": v[:, j], "ax": v[:, :, k]}
    lp = z["labprob"][:, ::-1] if flipx else z["labprob"]
    out["lab"] = {"sag": lp[:, i if not flipx else lp.shape[1] - 1 - i],
                  "cor": lp[:, :, j], "ax": lp[:, :, :, k]}
    return out, (x_mm, y_mm, z_mm)


# ── figure 1 ──────────────────────────────────────────────────────────────────

def roi_dots(ax, d, col, ylabel, rois=ORDER, ref=None):
    for i, r in enumerate(rois):
        v = d.loc[d.roi == r, col].dropna().to_numpy()
        if not len(v):
            continue
        jit = np.random.default_rng(i).uniform(-0.2, 0.2, len(v))
        ax.scatter(i + jit, v, s=4, color=COL[r], alpha=0.5, lw=0)
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=COL[r], lw=2,
                solid_capstyle="butt")
    if ref is not None:
        ax.axhline(ref, color="0.6", lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels([SHORT[r] for r in rois])
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax, offset=3)
    ax.tick_params(axis="x", labelrotation=55, labelsize=6.3)
    for t in ax.get_xticklabels():
        t.set_ha("right"); t.set_rotation_mode("anchor")
    ax.set_xlim(-0.6, len(rois) - 0.4)


def figure_dropout(rois, hist, ep, mni, example):
    fig = plt.figure(figsize=(7.25, 9.0))
    vmax = 100

    # a: group tSNR in MNI
    sl, (xm, ym, zm) = mni_slices(mni)
    fig.text(0.01, 0.985, "a", fontsize=9, fontweight="bold", va="top")
    fig.text(0.035, 0.985, f"Group temporal SNR (n = {len(mni['subjects'])}, MNI space): "
             "signal is lost over the orbitofrontal cortex, not over dorsomedial PFC",
             fontsize=7.5, fontweight="bold", va="top")
    x0, w = 0.02, 0.29
    im = None
    for c, (view, title) in enumerate((("sag", f"x = {xm} mm"), ("cor", f"y = +{ym} mm"),
                                        ("ax", f"z = {zm} mm"))):
        ax = fig.add_axes([x0 + c * (w + 0.01), 0.745, w, 0.215])
        br = sl["brain"][view] > 0
        tsnr = np.where(br, sl["tsnr"][view], np.nan)
        im = show_slice(ax, sl["template"][view], tsnr, sl["lab"][view], vmax=vmax,
                        crop=bbox(br, pad=2), alpha_over=br.astype(float), lw=1.0)
        ax.text(0.02, 0.98, title, transform=ax.transAxes, fontsize=6.5, color="white",
                va="top")
    cax = fig.add_axes([0.935, 0.765, 0.012, 0.17])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("tSNR", fontsize=6.5, labelpad=2)
    cb.outline.set_visible(False)
    cax.tick_params(labelsize=6, length=2)
    # outline key: the actual contour colours, two rows
    kx = fig.add_axes([0.02, 0.695, 0.96, 0.04]); kx.set_axis_off()
    kx.set_xlim(0, 1); kx.set_ylim(0, 1)
    for n, r in enumerate(["mOFC", "lOFC", "rACC", "dmPFC", "dACC", "PCC"]):
        xx, yy = (n % 3) * 0.24, 0.75 - (n // 3) * 0.5
        lab = f"{NAMES[r]} ({SHORT[r]})" if SHORT[r] != NAMES[r] else f"{NAMES[r]} (PCC)"
        kx.plot([xx, xx + 0.02], [yy, yy], color=COL[r], lw=1.4)
        kx.text(xx + 0.026, yy, lab, color=COL[r], va="center", fontsize=6.3)
    kx.text(1.0, 0.75, "Outline: label present in", fontsize=6, color="0.45", va="center", ha="right")
    kx.text(1.0, 0.25, "≥50% of participants", fontsize=6, color="0.45", va="center", ha="right")

    # b: example participant, T1w vs mean EPI vs tSNR, zoomed on the mPFC
    s = example
    key = lambda k: ep[f"sub{s}__{k}"]
    sub_t = rois[(rois.subject == s) & (rois.roi == "mOFC")].tsnr_median.iloc[0]
    fig.text(0.01, 0.672, "b", fontsize=9, fontweight="bold", va="top")
    fig.text(0.035, 0.672, f"One participant (sub-{s}, the median for medial-OFC tSNR "
             f"= {sub_t:.0f}): the EPI is empty where the T1w shows cortex",
             fontsize=7.5, fontweight="bold", va="top")
    w = 0.15
    for c, (view, kind) in enumerate([("sag", "t1"), ("sag", "epi"), ("sag", "tsnr"),
                                      ("ax", "t1"), ("ax", "epi"), ("ax", "tsnr")]):
        ax = fig.add_axes([0.02 + c * (w + 0.008) + (0.02 if c >= 3 else 0), 0.47, w, 0.175])
        t1 = key(f"t1_{view}").astype(np.float32)
        lab = key(f"lab_{view}")
        mp = np.isin(lab, [1, 2, 3, 5, 6])
        crop = bbox(mp, pad=18)
        if kind == "t1":
            show_slice(ax, t1, labels=lab, crop=crop, lw=0.7)
            ttl = "T1w"
        elif kind == "epi":
            show_slice(ax, key(f"epi_{view}").astype(np.float32), labels=lab, crop=crop, lw=0.7)
            ttl = "Mean EPI"
        else:
            ts = key(f"tsnr_{view}").astype(np.float32)
            show_slice(ax, t1, np.where(ts > 5, ts, np.nan), lab, vmax=vmax, crop=crop,
                       alpha_over=(ts > 5).astype(float), lw=0.7)
            ttl = "tSNR"
        ax.set_title(ttl + (" (sagittal)" if c == 0 else " (axial)" if c == 3 else ""),
                     fontsize=6.5, pad=2)

    # c: tSNR per ROI ; d: voxel tSNR distributions
    ax = fig.add_axes([0.075, 0.285, 0.5, 0.14])
    roi_dots(ax, rois, "tsnr_median", "Median voxel tSNR")
    ax.set_ylim(0, None)
    letter(ax, "c", dx=-0.08)
    ax = fig.add_axes([0.66, 0.31, 0.32, 0.115])
    bins = ep["tsnr_bins"]
    centres = (bins[:-1] + bins[1:]) / 2
    cols = [c for c in hist.columns if c.startswith("b")]
    for r in ["mOFC", "lOFC", "rACC", "dmPFC", "V1", "IPS12", "WM"]:
        h = hist[hist.roi == r][cols].to_numpy(float)
        h = h / h.sum(1, keepdims=True)
        mu = h.mean(0)
        ax.plot(centres, mu, color=COL[r], lw=1.3)
    for r, (xv, yv, ha) in {"mOFC": (7, 0.15, "left"), "lOFC": (13, 0.085, "left"),
                            "WM": (50, 0.112, "right"), "rACC": (66, 0.112, "left"),
                            "dmPFC": (84, 0.085, "left"), "V1": (98, 0.038, "left"),
                            "IPS12": (112, 0.06, "left")}.items():
        ax.text(xv, yv, SHORT[r], color=COL[r], fontsize=6, ha=ha, va="center")
    ax.set_xlim(0, 160)
    ax.set_xlabel("Voxel tSNR")
    ax.set_ylabel("Fraction of voxels")
    sns.despine(ax=ax, offset=3)
    letter(ax, "d", dx=-0.2)

    # e: GLMsingle R2 ; f, g: gabor and response-bar betas
    for j, (col, ttl, ref) in enumerate((
            ("glm_r2", "GLMsingle R² (%)", rois.loc[rois.roi == "WM", "glm_r2"].median()),
            ("beta_gabor", "Gabor β (% signal)", 0),
            ("beta_resp", "Response-bar β (% signal)", 0))):
        ax = fig.add_axes([0.075 + j * 0.335, 0.07, 0.245, 0.12])
        roi_dots(ax, rois, col, ttl, ref=ref)
        letter(ax, "efg"[j], dx=-0.3)
    fig.savefig(FIG / "fig_vmpfc_dropout.pdf")
    fig.savefig(FIG / "fig_vmpfc_dropout.png", dpi=200)
    plt.close(fig)


# ── figure 2 ──────────────────────────────────────────────────────────────────

def subject_curves(ep, trials):
    """Per subject x ROI x time: the mean epoch (relative to the run mean -- no
    pre-stimulus baseline, which would sit on the previous trial's response) and
    the slope of the response on z(value) (gabor-locked) and z(bid) (bar-locked),
    predictors demeaned within run."""
    rois = list(ep["rois"])
    eg, eb = ep["epochs_gabor"], ep["epochs_bar"]
    trials = trials.reset_index(drop=True)
    grp = (trials.subject + "_" + trials.session.astype(str) + "_" + trials.run.astype(str)).to_numpy()
    out = {k: [] for k in ("mean_g", "value", "bid")}
    subs = []
    for s, idx in trials.groupby("subject").groups.items():
        idx = np.asarray(idx)
        subs.append(s)
        out["mean_g"].append(np.nanmean(eg[idx], 0))
        for name, E in (("value", eg), ("bid", eb)):
            x = trials.loc[idx, name].to_numpy(float)
            ok = np.isfinite(x)
            g = grp[idx][ok]
            xd = x[ok] - pd.Series(x[ok]).groupby(g).transform("mean").to_numpy()
            xd = xd / xd.std()
            Y = E[idx][ok]
            Yd = Y - np.nanmean(Y, 0, keepdims=True)
            w = np.isfinite(Yd)
            out[name].append(np.nansum(Yd * xd[:, None, None], 0) /
                             np.sum(w * xd[:, None, None] ** 2, 0))
    return {k: np.stack(v) for k, v in out.items()}, rois, subs


def plot_curves(ax, t, C, rois, which, sig_y=None):
    for n, r in enumerate(which):
        k = rois.index(r)
        y = C[:, :, k]
        mu, se = np.nanmean(y, 0), stats.sem(y, 0, nan_policy="omit")
        ax.fill_between(t, mu - se, mu + se, color=COL[r], alpha=0.18, lw=0)
        ax.plot(t, mu, color=COL[r], lw=1.2)
        if sig_y is not None and len(y) > 2:
            p = stats.ttest_1samp(y, 0, nan_policy="omit").pvalue
            ax.plot(t, np.where(p < 0.01, sig_y[0] - sig_y[1] * n, np.nan), color=COL[r],
                    lw=2.0, solid_capstyle="butt")
    ax.axhline(0, color="0.6", lw=0.6, zorder=0)


def figure_response(ep, trials):
    C, rois, subs = subject_curves(ep, trials)
    tg, tb = ep["t_gabor"], ep["t_bar"]
    delay = trials.bar_delay.median()
    rt = trials.rt.median()
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.1), constrained_layout=True)
    cols = [("mean_g", tg, "Evoked response", "Signal change (%)"),
            ("value", tg, "Effect of value", "% signal per SD of value"),
            ("bid", tb, "Effect of the bid", "% signal per SD of bid")]
    ylims = {}
    for i, (group, gname) in enumerate(((MPFC, "Medial PFC"), (REFS, "Reference regions"))):
        for j, (k, t, ttl, yl) in enumerate(cols):
            ax = axes[i, j]
            gab = k != "bid"
            g0 = 0 if gab else -delay
            b0 = delay if gab else 0
            ax.axvspan(g0, g0 + 1.5, color="#1B8A80", alpha=0.12, lw=0)
            ax.axvspan(b0, b0 + rt, color="#D9730D", alpha=0.12, lw=0)
            idx = [rois.index(r) for r in group]
            mu = np.nanmean(C[k][:, :, idx], 0)
            se = stats.sem(C[k][:, :, idx], 0, nan_policy="omit")
            hi, lo = np.nanmax(mu + se), np.nanmin(mu - se)
            span = hi - lo
            sig = None if k == "mean_g" else (lo - 0.08 * span, 0.06 * span)
            plot_curves(ax, t, C[k], rois, group, sig_y=sig)
            ax.set_xlim(t[0], t[-1])
            ax.set_xlabel("Time from gabor onset (s)" if gab else "Time from response-bar onset (s)")
            ax.set_ylabel(yl)
            if i == 0:
                ax.set_title(ttl, loc="left", fontweight="bold")
            sns.despine(ax=ax, offset=3)
            if j == 0:
                yy = 0.98
                for r in group:
                    ax.text(1.0, yy, SHORT[r], color=COL[r], transform=ax.transAxes,
                            ha="right", va="top", fontsize=6.2)
                    yy -= 0.085
                ax.text(-0.36, 0.5, gname, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontweight="bold", fontsize=7.5)
            if i == 0 and j in (0, 2):
                ax.text(g0 + 0.3, 1.0, "Gabor", color="#1B8A80", fontsize=6,
                        transform=ax.get_xaxis_transform(), ha="left", va="bottom")
                ax.text(b0 + 0.3, 1.0, "Bar on screen", color="#D9730D", fontsize=6,
                        transform=ax.get_xaxis_transform(), ha="left", va="bottom")
            if i == 0 and j == 1:
                ax.text(0.99, 0.02, "Thick bars: p < .01 (uncorrected)", fontsize=5.8,
                        color="0.4", transform=ax.transAxes, ha="right", va="bottom")
    fig.savefig(FIG / "fig_vmpfc_response.pdf")
    fig.savefig(FIG / "fig_vmpfc_response.png", dpi=200)
    plt.close(fig)
    return C, rois, subs


def main():
    style()
    rois = pd.read_csv(DATA / "vmpfc_dropout_rois.tsv", sep="\t", dtype={"subject": str})
    hist = pd.read_csv(DATA / "vmpfc_dropout_tsnr_hist.tsv", sep="\t", dtype={"subject": str})
    trials = pd.read_csv(DATA / "vmpfc_dropout_trials.tsv", sep="\t", dtype={"subject": str})
    ep = np.load(DATA / "vmpfc_dropout_epochs.npz")
    mni = np.load(DATA / "vmpfc_dropout_mni.npz")
    m = rois[rois.roi == "mOFC"].set_index("subject").tsnr_median
    example = (m - m.median()).abs().idxmin()
    pd.set_option("display.width", 200)
    print(rois.groupby("roi")[["tsnr_median", "frac_tsnr_below20", "epi_rel", "frac_epi_below50",
                               "glm_r2"]].median().reindex(ORDER).round(3))
    for col in ("glm_r2", "beta_gabor", "beta_resp"):
        wm = rois[rois.roi == "WM"].set_index("subject")[col]
        for r in ORDER:
            v = rois[rois.roi == r].set_index("subject")[col]
            t0 = stats.ttest_1samp(v, 0)
            tw = stats.ttest_rel(v, wm.reindex(v.index))
            print(f"{col:10s} {r:12s} {v.mean():+.3f} ± {v.sem():.3f}  vs0 t={t0.statistic:+.2f} "
                  f"p={t0.pvalue:.2g} | vsWM t={tw.statistic:+.2f} p={tw.pvalue:.2g}")
    figure_dropout(rois, hist, ep, mni, example)
    C, rois_, subs = figure_response(ep, trials)
    tg, tb = ep["t_gabor"], ep["t_bar"]
    for k, t, win in (("mean_g", tg, (4, 8)), ("mean_g", tg, (10, 14)),
                      ("value", tg, (4, 8)), ("bid", tb, (3, 9))):
        w = (t >= win[0]) & (t <= win[1])
        print(k, win)
        for r in ORDER:
            v = np.nanmean(C[k][:, w, rois_.index(r)], 1)
            tt = stats.ttest_1samp(v, 0, nan_policy="omit")
            print(f"   {r:12s} {np.nanmean(v):+.4f} ± {stats.sem(v, nan_policy='omit'):.4f}  "
                  f"t={tt.statistic:+.2f} p={tt.pvalue:.2g}")
    print(f"Wrote {FIG / 'fig_vmpfc_dropout.pdf'} and {FIG / 'fig_vmpfc_response.pdf'}")


if __name__ == "__main__":
    main()
