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
        o = np.ma.masked_where(~np.isfinite(over) | (alpha_over is not None and False), over)
        im = ax.imshow(np.asarray(o, np.float32).T, origin="lower", cmap=cmap, vmin=0,
                       vmax=vmax, interpolation="bilinear",
                       alpha=None if alpha_over is None else alpha_over.T)
    if labels is not None:
        for roi, i in LABEL_ID.items():
            m = labels[i] if isinstance(labels, (list, np.ndarray)) and np.ndim(labels) == 3 \
                else (labels == i)
            if np.ndim(labels) == 3:
                m = labels[i - 1] >= 0.5
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

def roi_dots(ax, d, col, ylabel, rois=ORDER, ref=None, log=False):
    for i, r in enumerate(rois):
        v = d.loc[d.roi == r, col].dropna().to_numpy()
        if not len(v):
            continue
        jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(v))
        ax.scatter(i + jit, v, s=4, color=COL[r], alpha=0.45, lw=0)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=COL[r], lw=2,
                solid_capstyle="butt")
    if ref is not None:
        ax.axhline(ref, color="0.6", lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels([NAMES[r] for r in rois], rotation=50, ha="right")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    sns.despine(ax=ax, offset=3, trim=False)


def figure_dropout(rois, hist, ep, mni, example):
    fig = plt.figure(figsize=(7.25, 8.6))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.15, 1.25, 1.05, 1.05], hspace=0.5,
                          left=0.07, right=0.98, top=0.97, bottom=0.08)

    # a: group tSNR in MNI
    sl, (xm, ym, zm) = mni_slices(mni)
    ga = gs[0].subgridspec(1, 4, width_ratios=[1.25, 1.0, 1.0, 0.07], wspace=0.04)
    vmax = 100
    for c, (view, title) in enumerate((("sag", f"x = {xm} mm"), ("cor", f"y = +{ym} mm"),
                                        ("ax", f"z = {zm} mm"))):
        ax = fig.add_subplot(ga[c])
        br = sl["brain"][view] > 0
        tsnr = np.where(br, sl["tsnr"][view], np.nan)
        im = show_slice(ax, sl["template"][view], tsnr, sl["lab"][view], vmax=vmax,
                        crop=bbox(br), alpha_over=np.where(br, 0.85, 0.0))
        ax.set_title(title, fontsize=6.5, color="0.35", pad=2)
        if c == 0:
            letter(ax, "a", dx=0.0)
            ax.text(0.0, 1.13, f"Group tSNR (n = {len(mni['subjects'])}, MNI)", transform=ax.transAxes,
                    fontsize=7.5, fontweight="bold", va="bottom")
    cax = fig.add_subplot(ga[3])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Temporal SNR", fontsize=6.5)
    cb.outline.set_visible(False)
    cax.tick_params(labelsize=6)
    # outline key
    kx = fig.add_axes([0.07, 0.735, 0.9, 0.02]); kx.set_axis_off()
    for n, r in enumerate(["mOFC", "lOFC", "rACC", "dmPFC", "dACC", "PCC"]):
        kx.plot([n * 0.15, n * 0.15 + 0.025], [0.5, 0.5], color=COL[r], lw=1.4)
        kx.text(n * 0.15 + 0.032, 0.5, NAMES[r], color=COL[r], va="center", fontsize=6.5)
    kx.set_xlim(0, 1); kx.set_ylim(0, 1)
    kx.text(0.92, 0.5, "≥50% of participants", fontsize=6, color="0.45", va="center", ha="right")

    # b: example participant, T1w vs mean EPI vs tSNR
    gb = gs[1].subgridspec(1, 7, width_ratios=[1, 1, 1, 0.25, 1, 1, 1], wspace=0.03)
    s = example
    key = lambda k: ep[f"sub{s}__{k}"]
    for c, (view, kind) in enumerate([("sag", "t1"), ("sag", "epi"), ("sag", "tsnr"),
                                      None, ("ax", "t1"), ("ax", "epi"), ("ax", "tsnr")]):
        if view is None:
            continue
        ax = fig.add_subplot(gb[c] if c < 3 else gb[c])
        t1 = key(f"t1_{view}").astype(np.float32)
        lab = key(f"lab_{view}")
        br = t1 > np.percentile(t1[t1 > 0], 5)
        crop = bbox(t1 > np.percentile(t1, 60), pad=4)
        if kind == "t1":
            show_slice(ax, t1, labels=lab, crop=crop)
            ttl = "T1w"
        elif kind == "epi":
            epi = key(f"epi_{view}").astype(np.float32)
            show_slice(ax, epi, labels=lab, crop=crop)
            ttl = "Mean EPI"
        else:
            ts = key(f"tsnr_{view}").astype(np.float32)
            show_slice(ax, t1, np.where(ts > 0, ts, np.nan), lab, vmax=vmax, crop=crop,
                       alpha_over=np.where(ts > 0, 0.9, 0.0))
            ttl = "tSNR"
        ax.set_title(ttl, fontsize=6.5, pad=2)
        if c == 0:
            letter(ax, "b", dx=0.0)
            sub_t = rois[(rois.subject == s) & (rois.roi == "mOFC")].tsnr_median.iloc[0]
            ax.text(0.0, 1.13, f"One participant (sub-{s}; median medial-OFC tSNR {sub_t:.0f}): "
                    "signal is lost above the sinuses", transform=ax.transAxes, fontsize=7.5,
                    fontweight="bold", va="bottom")

    # c: tSNR per ROI ; d: voxel tSNR distributions
    gc = gs[2].subgridspec(1, 2, width_ratios=[1.35, 1], wspace=0.3)
    ax = fig.add_subplot(gc[0])
    roi_dots(ax, rois, "tsnr_median", "Median voxel tSNR")
    ax.set_ylim(0, None)
    letter(ax, "c")
    ax = fig.add_subplot(gc[1])
    bins = ep["tsnr_bins"]
    centres = (bins[:-1] + bins[1:]) / 2
    cols = [c for c in hist.columns if c.startswith("b")]
    for r in ["mOFC", "lOFC", "rACC", "dmPFC", "dACC", "V1", "IPS12", "WM"]:
        h = hist[hist.roi == r][cols].to_numpy(float)
        h = h / h.sum(1, keepdims=True)
        mu = h.mean(0) / np.diff(bins)
        ax.plot(centres, mu, color=COL[r], lw=1.3)
        k = np.argmax(mu)
        ax.text(centres[k], mu[k] * 1.04, NAMES[r], color=COL[r], fontsize=6,
                ha="center", va="bottom")
    ax.set_xlim(0, 160)
    ax.set_xlabel("Voxel tSNR")
    ax.set_ylabel("Density")
    ax.set_yticks([])
    sns.despine(ax=ax, offset=3, left=True)
    letter(ax, "d")

    # e: GLMsingle R2 ; f: gabor and response-bar betas
    ge = gs[3].subgridspec(1, 3, wspace=0.42)
    ax = fig.add_subplot(ge[0])
    roi_dots(ax, rois, "glm_r2", "GLMsingle R² (%)",
             ref=rois.loc[rois.roi == "WM", "glm_r2"].median())
    letter(ax, "e")
    for j, (col, ttl) in enumerate((("beta_gabor", "Gabor β (%)"),
                                    ("beta_resp", "Response-bar β (%)"))):
        ax = fig.add_subplot(ge[j + 1])
        roi_dots(ax, rois, col, ttl, ref=0)
        letter(ax, "f" if j == 0 else "g")
    fig.savefig(FIG / "fig_vmpfc_dropout.pdf")
    fig.savefig(FIG / "fig_vmpfc_dropout.png", dpi=200)
    plt.close(fig)


# ── figure 2 ──────────────────────────────────────────────────────────────────

def subject_curves(ep, trials):
    """Per subject x ROI x time: mean evoked (gabor-, bar-locked) and the slope of
    the response on z(value) (gabor-locked) / z(bid) (bar-locked), predictors
    demeaned within run."""
    rois = list(ep["rois"])
    eg, eb = ep["epochs_gabor"].copy(), ep["epochs_bar"].copy()
    tg = ep["t_gabor"]
    base = np.nanmean(eg[:, (tg >= -2) & (tg <= 0)], 1, keepdims=True)
    eg, eb = eg - base, eb - base
    trials = trials.reset_index(drop=True)
    grp = trials.subject + "_" + trials.session.astype(str) + "_" + trials.run.astype(str)
    out = {k: [] for k in ("mean_g", "mean_b", "value", "bid")}
    subs = []
    for s, idx in trials.groupby("subject").groups.items():
        idx = np.asarray(idx)
        subs.append(s)
        out["mean_g"].append(np.nanmean(eg[idx], 0))
        out["mean_b"].append(np.nanmean(eb[idx], 0))
        for name, E in (("value", eg), ("bid", eb)):
            x = trials.loc[idx, name].to_numpy(float)
            ok = np.isfinite(x)
            g = grp[idx][ok].to_numpy()
            xd = x[ok] - pd.Series(x[ok]).groupby(g).transform("mean").to_numpy()
            xd = xd / xd.std()
            Y = E[idx][ok]                                       # (n, T, R)
            Yd = Y - np.nanmean(Y, 0, keepdims=True)
            out[name].append(np.nansum(Yd * xd[:, None, None], 0) /
                             np.sum(np.isfinite(Yd) * xd[:, None, None] ** 2, 0))
    return {k: np.stack(v) for k, v in out.items()}, rois, subs


def plot_curves(ax, t, C, rois, which, sig=False, ybar=None):
    for r in which:
        k = rois.index(r)
        y = C[:, :, k]
        mu, se = np.nanmean(y, 0), stats.sem(y, 0, nan_policy="omit")
        ax.fill_between(t, mu - se, mu + se, color=COL[r], alpha=0.18, lw=0)
        ax.plot(t, mu, color=COL[r], lw=1.2)
        if sig:
            p = stats.ttest_1samp(y, 0, nan_policy="omit").pvalue
            on = np.where(p < 0.01, ybar[0] + ybar[1] * which.index(r), np.nan)
            ax.plot(t, on, color=COL[r], lw=2.2, solid_capstyle="butt")
    ax.axhline(0, color="0.6", lw=0.6, zorder=0)


def figure_response(ep, trials):
    C, rois, subs = subject_curves(ep, trials)
    tg, tb = ep["t_gabor"], ep["t_bar"]
    delay = trials.bar_delay.median()
    rt = trials.rt.median()
    fig, axes = plt.subplots(2, 4, figsize=(7.25, 3.9), constrained_layout=True)
    cols = [("mean_g", tg, "Gabor-locked response", "Signal change (%)"),
            ("mean_b", tb, "Response-bar-locked response", "Signal change (%)"),
            ("value", tg, "Effect of value (gabor-locked)", "% signal per SD value"),
            ("bid", tb, "Effect of bid (bar-locked)", "% signal per SD bid")]
    for i, (group, gname) in enumerate(((MPFC, "Medial PFC"), (REFS, "Reference regions"))):
        for j, (k, t, ttl, yl) in enumerate(cols):
            ax = axes[i, j]
            locked_g = k in ("mean_g", "value")
            g0, g1 = (0, 1.5) if locked_g else (-delay, -delay + 1.5)
            b0, b1 = (delay, delay + rt) if locked_g else (0, rt)
            ax.axvspan(g0, g1, color="#1B8A80", alpha=0.10, lw=0)
            ax.axvspan(b0, b1, color="#D9730D", alpha=0.10, lw=0)
            effect = k in ("value", "bid")
            lim = np.nanmax(np.abs(np.nanmean(C[k][:, :, [rois.index(r) for r in group]], 0)))
            plot_curves(ax, t, C[k], rois, group, sig=effect,
                        ybar=(-lim * 1.6, -lim * 0.18) if effect else None)
            ax.set_xlim(t[0], t[-1])
            ax.set_xlabel("Time from gabor (s)" if locked_g else "Time from response bar (s)")
            if j in (0, 2):
                ax.set_ylabel(yl)
            elif j in (1, 3):
                ax.set_ylabel(yl)
            if i == 0:
                ax.set_title(ttl, loc="left", fontweight="bold")
            sns.despine(ax=ax, offset=3)
            if j == 0:
                yy = 0.97
                for r in group:
                    ax.text(0.98, yy, NAMES[r], color=COL[r], transform=ax.transAxes,
                            ha="right", va="top", fontsize=6)
                    yy -= 0.09
                ax.text(-0.42, 0.5, gname, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontweight="bold", fontsize=7.5)
        top = axes[i, 0]
    ax = axes[0, 0]
    ax.text(0.75, 1.0, "Gabor", color="#1B8A80", fontsize=6, transform=ax.get_xaxis_transform(),
            ha="left", va="bottom")
    ax.text(delay + 0.2, 1.0, "Response bar", color="#D9730D", fontsize=6,
            transform=ax.get_xaxis_transform(), ha="left", va="bottom")
    axes[0, 2].text(0.02, 0.04, "Bar: p < .01 (uncorrected)", transform=axes[0, 2].transAxes,
                    fontsize=5.8, color="0.4")
    for a in axes.ravel():
        a.text(0, 0, "")
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
    print(rois.groupby("roi")[["tsnr_median", "frac_tsnr_below20", "epi_rel", "glm_r2",
                               "beta_gabor", "beta_resp"]].median().reindex(ORDER).round(3))
    for col in ("beta_gabor", "beta_resp"):
        for r in ORDER:
            v = rois.loc[rois.roi == r, col].dropna()
            t = stats.ttest_1samp(v, 0)
            print(f"{col:10s} {r:12s} {v.mean():+.3f} ± {v.sem():.3f}  t={t.statistic:+.2f} p={t.pvalue:.3g}")
    figure_dropout(rois, hist, ep, mni, example)
    C, rois_, subs = figure_response(ep, trials)
    tg, tb = ep["t_gabor"], ep["t_bar"]
    for k, t, win in (("mean_g", tg, (4, 8)), ("mean_b", tb, (3, 8)),
                      ("value", tg, (4, 8)), ("bid", tb, (3, 8))):
        w = (t >= win[0]) & (t <= win[1])
        print(k, win)
        for r in ORDER:
            v = np.nanmean(C[k][:, w, rois_.index(r)], 1)
            tt = stats.ttest_1samp(v, 0, nan_policy="omit")
            print(f"   {r:12s} {np.nanmean(v):+.4f} ± {stats.sem(v, nan_policy='omit'):.4f}  "
                  f"t={tt.statistic:+.2f} p={tt.pvalue:.3g}")
    print(f"Wrote {FIG / 'fig_vmpfc_dropout.pdf'} and {FIG / 'fig_vmpfc_response.pdf'}")


if __name__ == "__main__":
    main()
