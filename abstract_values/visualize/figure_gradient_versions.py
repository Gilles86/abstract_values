"""Perception -> representation gradient: several versions of the figure.

The logic, stated once. In the two-stage model a parietal value code is
built FROM the perceptual estimate, so its trial-to-trial noise is the
product of both stages: perceptual noise carried forward plus value noise
added on top (in CHF, total^2 = sigma_perc^2 + sigma_rep^2). An early visual
code carries only the first. So the prediction is not "perception early,
value late" but:

    perceptual precision  predicts neural precision EVERYWHERE (it is inherited)
    value precision       predicts it only from the stage where value is formed

i.e. the value-specific contribution grows along the hierarchy while the
perceptual one need not fade. The versions below test that in different ways.

  v1  Two links (the original): orientation code ~ kappa_r, value code ~
      -log sigma_rep, partial on motion.
  v2  One neural measure per region (value-code strength), regressed on
      perceptual AND value precision together: standardised betas along the
      hierarchy. Prediction: beta_perc everywhere, beta_value grows.
  v3  As v2, with decoding precision from the weighted aPRF (expected SD of
      decoded value, union prior) instead of code strength.
  v4  Unique variance: per region, R² explained uniquely by value precision
      and uniquely by perceptual precision (stacked), so the shift in what
      drives the region reads as a change in colour along the hierarchy.

Error bars: ±1 SE (bootstrap SD over participants). Gradient test: the
late-minus-early difference (V1-V3 vs IPS0-IPS5) of the value beta minus the
same for the perceptual beta, with a participant bootstrap.

    python -m abstract_values.visualize.figure_gradient_versions [--params TSV]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cortex
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abstract_values.utils.data import BIDS_FOLDER
from abstract_values.visualize import qc_flatmaps as qf
from abstract_values.visualize.figure_gradient import (CX_ROI, DORSAL, PARAMS, PIL,
                                                       exclusive_rois, paint, roi_verts)
from abstract_values.visualize.group_surface_maps import discover_subjects
from abstract_values.visualize.qc_tuning_maps import collect

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "notes" / "figures"
PER_C, REP_C = "#1B8A80", "#D9730D"
EARLY, LATE = ["V1", "V2", "V3"], ["IPS0", "IPS1–2", "IPS3–5"]
# Control region, drawn apart from the hierarchy (never part of any test)
CONTROL = ["M1 L"]
CTRL_LAB = {"M1 L": "Left M1"}
ALL = None                     # set after DORSAL is imported


def xpos(r):
    return DORSAL.index(r) if r in DORSAL else len(DORSAL) + 1.3
N_BOOT = int(__import__("os").environ.get("N_BOOT", 3000))
EU_DIR = Path("/data/ds-abstractvalue/derivatives/encoding_models/aprf-weighted")
EU_ROI = {"BensonV1": "V1", "BensonV2": "V2", "BensonV3": "V3", "V3ab": "V3a/b",
          "IPS0": "IPS0", "IPS12": "IPS1–2", "IPS345": "IPS3–5", "M1l": "M1 L"}


def zs(x):
    return (x - np.nanmean(x)) / np.nanstd(x)


def betas(y, X):
    """Standardised OLS betas of y on columns of X (+ intercept)."""
    ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
    A = np.column_stack([np.ones(ok.sum()), np.apply_along_axis(zs, 0, X[ok])])
    b = np.linalg.lstsq(A, zs(y[ok]), rcond=None)[0][1:]
    return b


def unique_r2(y, X):
    ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
    y, X = zs(y[ok]), np.apply_along_axis(zs, 0, X[ok])

    def r2(cols):
        A = np.column_stack([np.ones(len(y))] + [X[:, c] for c in cols])
        res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        return 1 - res.var() / y.var()
    full = r2([0, 1, 2])
    return full - r2([1, 2]), full - r2([0, 2])        # unique perc, unique value


def subject_table(params):
    par = pd.read_csv(params, sep="\t")
    par.index = par.subject.map(lambda s: f"{int(s):02d}")
    perc = par.total_noise * np.sqrt(par.perceptual_share)        # CHF
    mo = pd.read_csv(REPO / "notes/data/motion_summary.tsv", sep="\t", dtype={"subject": str})
    mo["subject"] = mo.subject.str.replace("sub-", "").replace(PIL)
    return pd.DataFrame({"perc_prec": -np.log(perc), "val_prec": -np.log(par.sigma_rep),
                         "kappa": np.log(par.kappa_r),
                         "fd": mo.groupby("subject").mean_fd.mean()}).dropna()


def strength_measures():
    deriv = Path(BIDS_FOLDER) / "derivatives"
    d = collect(deriv, discover_subjects(deriv), True)
    subs = [PIL.get(s, s) for s in d["subjects"]]
    verts = exclusive_rois(roi_verts())
    val = pd.DataFrame({r: np.nanmean(d["val_gain"][:, vx], 1) for r, vx in verts.items()},
                       index=subs)
    ori = pd.DataFrame({r: np.nanmean(d["ori_gain"][:, vx], 1) for r, vx in verts.items()},
                       index=subs)
    maps = {"val": np.nanmean(d["val_gain"], 0), "ori": np.nanmean(d["ori_gain"], 0)}
    return val, ori, verts, maps


def eu_table():
    frames = []
    for f in EU_DIR.glob("sub-*/ses-*/func/*_nvoxels-100_nsims-1000_noise-spherical_perses"
                         "_desc-expected_decoded_value_pe.tsv"):
        mask = f.name.split("_mask-")[1].split("_")[0]
        if mask in EU_ROI:
            dd = pd.read_csv(f, sep="\t", dtype={"subject": str})
            frames.append(dd.assign(roi=EU_ROI[mask]))
    eu = pd.concat(frames)
    eu["subject"] = eu.subject.replace(PIL).str.zfill(2)
    return eu[eu.value.between(6, 38)]


def eu_error():
    """How well value can be decoded at all: expected |decoded - true| (CHF)."""
    eu = eu_table()
    return eu.groupby(["subject", "roi"]).mean_abs_error.mean().unstack()


DEC_DIR = Path("/data/ds-abstractvalue/derivatives/decoding/value")
DEC_ROI = {"BensonV1": "V1", "BensonV2": "V2", "BensonV3": "V3", "V3ab": "V3a/b",
           "IPS0": "IPS0", "IPS12": "IPS1–2", "IPS345": "IPS3–5", "M1l": "M1 L"}


def decode_r():
    """How well value decodes at all: r(posterior-mean decoded value, true value),
    leave-one-run-out, top-100 voxels, spherical noise, lambda 0.1; subject x ROI."""
    rows = []
    for mask, roi in DEC_ROI.items():
        for f in DEC_DIR.glob(f"sub-*/func/*_mask-{mask}_nvoxels-100_noise-spherical_lambda-0.1_pars.tsv"):
            d = pd.read_csv(f, sep="\t")
            grid = np.array([float(c) for c in d.columns[4:]])
            pdf = d.iloc[:, 4:].to_numpy(float)
            e = (pdf * grid).sum(1) / pdf.sum(1)
            ok = np.isfinite(e) & np.isfinite(d.true_value_chf)
            sub = f.name.split("_")[0].replace("sub-", "")
            rows.append(dict(subject=PIL.get(sub, sub), roi=roi,
                             r=np.corrcoef(e[ok], d.true_value_chf[ok])[0, 1]))
    return pd.DataFrame(rows).pivot(index="subject", columns="roi", values="r")


def eu_measure():
    frames = []
    for f in EU_DIR.glob("sub-*/ses-*/func/*_nvoxels-100_nsims-1000_noise-spherical_perses"
                         "_desc-expected_decoded_value_pe.tsv"):
        mask = f.name.split("_mask-")[1].split("_")[0]
        if mask in EU_ROI:
            dd = pd.read_csv(f, sep="\t", dtype={"subject": str})
            frames.append(dd.assign(roi=EU_ROI[mask]))
    eu = pd.concat(frames)
    eu["subject"] = eu.subject.replace(PIL).str.zfill(2)
    eu = eu[eu.value.between(6, 38)]
    return (eu.assign(p=-np.log(eu.sd_E)).groupby(["subject", "roi"]).p.mean()
            .unstack())


def profile(meas, tab, regs, idx=None):
    """Per-ROI betas of measure on regressors (+ motion as covariate)."""
    subs = [s for s in tab.index if s in meas.index]
    if idx is not None:
        subs = [subs[i] for i in idx]
    X = tab.loc[subs, regs + ["fd"]].to_numpy()
    return {r: (betas(meas.loc[subs, r].to_numpy(float), X)[:len(regs)] if r in meas
                else np.full(len(regs), np.nan)) for r in DORSAL + CONTROL}


def boot(fn, n_subj):
    rng = np.random.default_rng(0)
    return [fn(rng.integers(0, n_subj, n_subj)) for _ in range(N_BOOT)]


def gradient_stat(prof):
    e = np.nanmean([prof[r] for r in EARLY if r in prof], 0)
    l_ = np.nanmean([prof[r] for r in LATE if r in prof], 0)
    d = l_ - e
    return d[1] - d[0], d


LIGHT = {PER_C: "#E3F1EF", REP_C: "#FBEBDD"}
DARK = {PER_C: "#0B4A44", REP_C: "#6E3404"}
LABEL_ROIS = {"V1": "V1", "V2": "V2", "V3": "V3", "V3a/b": "V3a/b", "IPS0": "IPS0",
              "IPS1–2": "IPS1–2", "IPS3–5": "IPS3–5"}


def stage_cmap(col):
    return sns.blend_palette([LIGHT[col], col, DARK[col]], n_colors=256, as_cmap=True)


BG_LO, BG_HI = 0.002, 0.012          # group cvR² over null: opacity ramp


def composite(m, col, lim, bg):
    """Vertex RGB, 'spotlight' style: cortex without a code fades to white.

    Where the group-mean cvR² over null is low, the curvature washes out; where
    it is high, the cortex shows at full contrast. Inside the ROIs the ROI
    colour (its weight) gets the same signal-dependent opacity, so a region
    reads as solid only where it codes. (Alternatives:
    figure_gradient_bg_variants.py.)"""
    zero = np.zeros(len(m), np.float32)
    curv = cortex.Vertex(zero, "fsaverage").blend_curvature(zero, brightness=0.7,
                                                              contrast=0.35)
    rgb = np.stack([curv.red.data, curv.green.data, curv.blue.data], 1) / 255.0
    lo, hi = np.nanpercentile(bg, [75, 95])            # top quarter of cortex fades in
    ramp = np.clip((np.nan_to_num(bg) - lo) / (hi - lo), 0, 1)
    inside = np.isfinite(m)
    fade = 0.8 * (1 - ramp)
    rgb = rgb * (1 - fade[:, None]) + fade[:, None]
    roi_rgb = stage_cmap(col)(np.clip(np.nan_to_num(m), 0, lim) / lim)[:, :3]
    a_in = np.where(inside, 0.3 + 0.7 * ramp, 0)
    rgb = rgb * (1 - a_in[:, None]) + roi_rgb * a_in[:, None]
    u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return cortex.VertexRGB(u8[:, 0], u8[:, 1], u8[:, 2], "fsaverage")


SIG_CMAP = sns.blend_palette(["#FDD49E", "#FC8D59", "#D7301F", "#7F0000"], n_colors=256,
                             as_cmap=True)


def signal_vertex(bg, cmap=None):
    """Group signal map in warm colours, fading in over the top quarter of cortex."""
    lo, hi = np.nanpercentile(bg, [75, 99])
    x = np.nan_to_num(bg).astype(np.float32)
    v = cortex.Vertex(x, "fsaverage", vmin=lo, vmax=hi, cmap=SIG_CMAP if cmap is None else cmap)
    al = np.clip((x - lo) / (np.nanpercentile(bg, 95) - lo), 0, 1).astype(np.float32)
    return v.blend_curvature(al, brightness=0.82, contrast=0.12), lo, hi


def brain_panel(ax, values, verts, col, lim=0.5, bg=None, signal=False, labels=True):
    """ROI-painted flatmap on a light, low-contrast curvature background.
    signal=True draws the group signal map ``bg`` itself instead of weights."""
    m = paint(values, verts)
    al = np.where(np.isfinite(m), 1.0, 0.0).astype(np.float32)
    if signal:
        vtx = signal_vertex(bg)[0]
    elif bg is None:
        v = cortex.Vertex(np.clip(np.nan_to_num(m), 0, lim).astype(np.float32),
                          "fsaverage", vmin=0, vmax=lim, cmap=stage_cmap(col))
        vtx = v.blend_curvature(al, brightness=0.82, contrast=0.12)
    else:
        vtx = composite(m, col, lim, bg)
    cortex.quickflat.make_figure(
        vtx, fig=ax, with_curvature=False, with_colorbar=False, with_rois=True,
        with_labels=False, with_sulci=False,
        roi_list=["V1", "V2", "V3", "V3a", "IPS0", "IPS1", "IPS2"],
        linewidth=2, linecolor=(1, 1, 1, 0.9), height=2048)
    ax.set_axis_off()
    pts = cortex.db.get_surf("fsaverage", "flat", merge=True, nudge=True)[0][:, :2]
    if not labels:
        lh = cortex.db.get_surf("fsaverage", "flat", "lh")[0].shape[0]
        x_txt = np.percentile(pts[np.concatenate([verts[r][verts[r] < lh] for r in LABEL_ROIS]), 0], 1) - 8
        ax.set_xlim(x_txt - 62, 128)
        ax.set_ylim(-88, 68)
        ax.set_aspect("equal", adjustable="box")
        return
    n_lh = cortex.db.get_surf("fsaverage", "flat", "lh")[0].shape[0]
    # Labels outside the left hemisphere, with leader lines to each region's
    # centre, so they never sit on the colour they name.
    cents = {}
    for r, txt in LABEL_ROIS.items():
        vx = verts[r][verts[r] < n_lh]
        if len(vx):
            cents[txt] = np.median(pts[vx], 0)
    lh_painted = np.concatenate([verts[r][verts[r] < n_lh] for r in LABEL_ROIS])
    x_txt = np.percentile(pts[lh_painted, 0], 1) - 8
    order = sorted(cents, key=lambda t: cents[t][1])
    ys = np.array([cents[t][1] for t in order], float)
    for i in range(1, len(ys)):                         # keep labels >= 17 units apart
        ys[i] = max(ys[i], ys[i - 1] + 17)
    ys -= max(0, ys[-1] - 62)
    for t, y in zip(order, ys):
        cx, cy = cents[t]
        ax.plot([x_txt + 3, cx], [y, cy], color="0.35", lw=0.5)
        ax.plot(cx, cy, "o", ms=1.6, color="0.2")
        ax.text(x_txt, y, t, ha="right", va="center", fontsize=6, color="0.25")
    ax.set_xlim(x_txt - 62, 128)
    ax.set_ylim(-88, 68)
    ax.set_aspect("equal", adjustable="box")   # crop, do not widen the view


def hierarchy_axis(ax, x):
    """Group brackets under the ROI tick labels (offsets in points, so they sit
    the same distance below any axis height)."""
    from matplotlib.transforms import offset_copy
    base = ax.get_xaxis_transform()
    tr_line = offset_copy(base, fig=ax.figure, y=-21, units="points")
    tr_text = offset_copy(base, fig=ax.figure, y=-23, units="points")
    xc = len(DORSAL) + 1.3
    for lo, hi, lab in ((x[0], x[2], "Early visual"), (x[4], x[6], "Parietal"),
                        (xc, xc, "Control")):
        ax.plot([lo - 0.2, hi + 0.2], [0, 0], color="0.55", lw=0.8,
                transform=tr_line, clip_on=False)
        ax.text((lo + hi) / 2, 0, lab, ha="center", va="top", fontsize=6.5,
                color="0.4", transform=tr_text)


def line_panel(ax, est, B, labels, cols, ylab):
    x = np.arange(len(DORSAL))
    for k, (lab, col) in enumerate(zip(labels, cols)):
        y = np.array([est[r][k] for r in DORSAL])
        se = np.array([np.std([b[r][k] for b in B]) for r in DORSAL])
        ax.fill_between(x, y - se, y + se, color=col, alpha=0.16, lw=0)
        ax.plot(x, y, "-", color=col, lw=1.8, solid_capstyle="round", zorder=3)
        ax.plot(x, y, "o", color=col, ms=4.5, mec="white", mew=0.8, zorder=4)
        ax.text(0.01, 0.98 - 0.1 * k, lab, color=col, fontsize=7, fontweight="bold",
                va="top", transform=ax.transAxes)
        # control region: its own point, no line joining it to the hierarchy
        for r in CONTROL:
            if r in est and np.isfinite(est[r][k]):
                xc = xpos(r) + (k - 0.5) * 0.2
                s_ = np.std([b[r][k] for b in B])
                ax.vlines(xc, est[r][k] - s_, est[r][k] + s_, color=col, lw=1.1)
                ax.plot(xc, est[r][k], "s", color=col, ms=4.2, mec="white", mew=0.8,
                        zorder=4)
    ax.axvline(len(DORSAL) + 0.15, color="0.8", lw=0.8, ls=(0, (1, 2)))
    ax.axhline(0, color="0.7", lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks([])
    ax.set_xlim(-0.3, len(DORSAL) + 2.0)
    ax.set_ylabel(ylab)
    sns.despine(ax=ax, offset=4, trim=True, bottom=True)


def signal_strip(ax, df, xlim, label, scale=1.0, chance=None, fmt="{:.1g}", points=False):
    """Per-region group mean ± 1 SE of a signal measure, aligned under panel c;
    with points=True every participant is a dot (swarm) behind the mean."""
    rois = DORSAL + CONTROL
    x = np.array([xpos(r) for r in rois])
    m = np.array([df[r].mean() if r in df else np.nan for r in rois]) * scale
    se = np.array([df[r].std() / np.sqrt(df[r].notna().sum()) if r in df else np.nan
                   for r in rois]) * scale
    if points:
        long = (df.reindex(columns=rois) * scale).melt(var_name="roi", value_name="v")
        long = long.dropna()
        long["x"] = long.roi.map(xpos)
        sns.swarmplot(data=long, x="x", y="v", native_scale=True, color="0.72", size=1.8,
                      ax=ax, zorder=1)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.hlines(m, x - 0.25, x + 0.25, color="0.15", lw=1.4, zorder=3)
        ax.vlines(x, m - se, m + se, color="0.15", lw=1.0, zorder=3)
        lo = min(0, np.nanmin(long.v))
        top = np.nanmax(long.v)
    else:
        ax.bar(x, m, width=0.5, color="0.78", lw=0)
        ax.vlines(x, m - se, m + se, color="0.35", lw=0.8)
        lo = 0
        top = np.nanmax(m + se) if chance is None else max(np.nanmax(m + se), chance)
    if chance is not None:
        ax.axhline(chance, color="0.45", lw=0.7, ls=(0, (3, 2)), zorder=0)
        ax.text(len(DORSAL) - 0.6, chance, "Chance", fontsize=6, color="0.45",
                va="bottom")
    ax.axvline(len(DORSAL) + 0.15, color="0.8", lw=0.8, ls=(0, (1, 2)))
    ax.set_xlim(xlim)
    ax.set_xticks(x)
    ax.set_xticklabels([CTRL_LAB.get(r, r) for r in rois])
    ax.set_ylim(lo - 0.05 * (top - lo), top * 1.05)
    tick = float(fmt.format(top))
    ax.set_yticks([0, tick])
    ax.set_yticklabels(["0", f"{tick:g}"])
    ax.set_ylabel(label, fontsize=6.5, color="0.3")
    ax.tick_params(axis="y", labelsize=6)
    sns.despine(ax=ax, offset=4, trim=True)
    hierarchy_axis(ax, np.arange(len(DORSAL)))


def contrast_panel(ax, est, B, cols, g, Bg):
    """Early vs parietal mean weight per stage: the interaction at a glance."""
    def grp(prof, roi_set, k):
        return np.nanmean([prof[r][k] for r in roi_set if r in prof])
    for k, col in enumerate(cols):
        y = [grp(est, EARLY, k), grp(est, LATE, k)]
        se = [np.std([grp(b, EARLY, k) for b in B]), np.std([grp(b, LATE, k) for b in B])]
        ax.plot([0, 1], y, "-", color=col, lw=1.8, zorder=3)
        ax.errorbar([0, 1], y, yerr=se, fmt="o", color=col, ms=5.5, mec="white", mew=0.8,
                    elinewidth=1.2, capsize=0, zorder=4)
    ax.axhline(0, color="0.7", lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Early\nvisual", "Parietal"])
    ax.set_xlim(-0.35, 1.35)
    sns.despine(ax=ax, offset=4, trim=True)
    p = np.mean(np.array(Bg) <= 0)
    ptxt = f"P = {p:.3f}" if p < 0.1 else f"P = {p:.2f}"
    ax.set_title(f"Interaction {g:+.2f} ± {np.std(Bg):.2f}\n{ptxt}", fontsize=7,
                 color="0.2", loc="center", pad=4)
    return p


def version_beta(meas, tab, measure, fname, verts, strip, bg=None):
    """a, b: brain maps of the two weights; c: profile; d: early-vs-parietal."""
    regs = ["perc_prec", "val_prec"]
    subs = [s for s in tab.index if s in meas.index]
    est = profile(meas, tab, regs)
    B = boot(lambda i: profile(meas, tab, regs, i), len(subs))
    g, _ = gradient_stat(est)
    Bg = [gradient_stat(b)[0] for b in B]
    fig = plt.figure(figsize=(7.25, 4.9))
    labs = ["Perceptual precision", "Value precision"]
    sig_name = "Orientation" if "orientation" in measure else "Value"
    for k, col in enumerate((PER_C, REP_C, None)):
        x0 = k / 3
        a = fig.add_axes([x0 + 0.004, 0.545, 0.326, 0.36])
        if col is None:
            brain_panel(a, {r: np.nan for r in verts}, verts, REP_C, bg=bg, signal=True,
                        labels=False)
            title, sub, cm = (f"Where {sig_name.lower()} is coded",
                              f"Group {sig_name.lower()} cvR² over null", SIG_CMAP)
            lo, hi = signal_vertex(bg)[1:]
            cb = qf.colorbar(fig, [x0 + 0.09, 0.565, 0.15, 0.01], cm, lo * 100, hi * 100,
                             "", ticks=[lo * 100, hi * 100],
                             ticklabels=[f"{lo * 100:.1f}%", f"{hi * 100:.1f}%"])
            tcol = "#B2301F"
        else:
            brain_panel(a, {r: est[r][k] if r in est and r not in CONTROL else np.nan for r in verts}, verts,
                        col, bg=bg, labels=(k == 0))
            title, sub = labs[k], f"Weight on {measure}"
            cb = qf.colorbar(fig, [x0 + 0.09, 0.565, 0.15, 0.01], stage_cmap(col), 0, 0.5,
                             "", ticks=[0, 0.25, 0.5], ticklabels=["≤ 0", "0.25", "0.5"])
            tcol = col
        cb.ax.tick_params(labelsize=6, length=1.5)
        cb.outline.set_visible(False)
        fig.text(x0 + 0.01, 0.955, "abc"[k], fontsize=8, fontweight="bold")
        fig.text(x0 + 0.03, 0.955, title, fontsize=8, color=tcol, fontweight="bold")
        fig.text(x0 + 0.03, 0.928, sub, fontsize=6.5, color="0.35")
    axc = fig.add_axes([0.075, 0.225, 0.62, 0.245])
    line_panel(axc, est, B, labs, [PER_C, REP_C], "Weight (std. β)")
    axr = fig.add_axes([0.075, 0.105, 0.62, 0.085])
    signal_strip(axr, xlim=axc.get_xlim(), **strip)
    fig.text(0.01, 0.48, "d", fontsize=8, fontweight="bold")
    axd = fig.add_axes([0.855, 0.12, 0.11, 0.26])
    p = contrast_panel(axd, est, B, [PER_C, REP_C], g, Bg)
    axd.set_ylabel("Mean weight (std. β)")
    sns.despine(ax=axd, offset=4, trim=True)
    fig.text(0.79, 0.465, "e", fontsize=8, fontweight="bold")
    fig.savefig(FIG / fname, dpi=300)
    fig.savefig((FIG / fname).with_suffix(".png"), dpi=200)
    plt.close(fig)
    return g, np.std(Bg), p, est


def version_unique(meas, tab, fname, strip, test):
    """Unique R²: how much of each region's between-participant variance in the
    value code only perceptual (teal) or only value (amber) precision explains."""
    regs = ["perc_prec", "val_prec"]
    subs = [s for s in tab.index if s in meas.index]

    def ur2(idx=None):
        ss = subs if idx is None else [subs[i] for i in idx]
        X = tab.loc[ss, regs + ["fd"]].to_numpy()
        return np.array([unique_r2(meas.loc[ss, r].to_numpy(float), X)
                         if r in meas else (np.nan, np.nan) for r in DORSAL + CONTROL])
    est = ur2()
    # Jackknife SE: the bootstrap of an R² is biased upward (resampling
    # duplicates participants and inflates fit), which puts the whole whisker
    # above the estimate; leave-one-out is the standard SE for R².
    n = len(subs)
    J = np.array([ur2(np.delete(np.arange(n), i)) for i in range(n)])
    se = np.sqrt((n - 1) / n * ((J - J.mean(0)) ** 2).sum(0))
    fig = plt.figure(figsize=(5.2, 3.0))
    ax = fig.add_axes([0.12, 0.38, 0.62, 0.54])
    x = np.array([xpos(r) for r in DORSAL + CONTROL])
    up, uv = est[:, 0], est[:, 1]
    ax.axvline(len(DORSAL) + 0.15, color="0.8", lw=0.8, ls=(0, (1, 2)))
    ax.bar(x, up, color=PER_C, width=0.6, lw=0)
    ax.bar(x, uv, bottom=up, color=REP_C, width=0.6, lw=0)
    # ±1 jackknife SE at the top of each segment, floored at the segment base
    ax.vlines(x - 0.12, np.maximum(up - se[:, 0], 0), up + se[:, 0], color=DARK[PER_C],
              lw=0.9)
    ax.vlines(x + 0.12, up + np.maximum(uv - se[:, 1], 0), up + uv + se[:, 1],
              color=DARK[REP_C], lw=0.9)
    ax.set_xticks([])
    ax.set_ylabel("Unique R² in\nvalue-code strength")
    j = len(DORSAL) - 1
    ax.text(x[j] + 0.45, up[j] / 2, "Perceptual\nprecision", color=PER_C, fontsize=7,
            fontweight="bold", va="center")
    ax.text(x[j] + 0.45, up[j] + uv[j] / 2, "Value\nprecision", color=REP_C,
            fontsize=7, fontweight="bold", va="center")
    ax.set_xlim(-0.5, len(DORSAL) + 2.0)
    sns.despine(ax=ax, offset=4, trim=True, bottom=True)
    g, gse, p = test
    ax.text(0.02, 0.98, f"Interaction (std. β): {g:+.2f} ± {gse:.2f}, P = {p:.3f}",
            transform=ax.transAxes, fontsize=6.5, color="0.25", va="top")
    axr = fig.add_axes([0.12, 0.13, 0.62, 0.15])
    signal_strip(axr, xlim=ax.get_xlim(), **strip)
    fig.savefig(FIG / fname, dpi=300)
    fig.savefig((FIG / fname).with_suffix(".png"), dpi=200)
    plt.close(fig)
    print("v4 unique R² (perc, value) ± SE:")
    for r, e, s_ in zip(DORSAL + CONTROL, est, se):
        print(f"  {r:7s} {e[0]:.3f}±{s_[0]:.3f}  {e[1]:.3f}±{s_[1]:.3f}")
    return est, se


def main(params=PARAMS):
    import matplotlib as mpl
    mpl.rcParams.update({"font.family": "Helvetica", "font.size": 7,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.linewidth": 0.8, "xtick.direction": "out",
                         "ytick.direction": "out", "pdf.fonttype": 42, "svg.fonttype": "none"})
    tab = subject_table(params)
    val, ori, verts, maps = strength_measures()
    s_val = dict(df=val, label="Value signal\n(cvR² over null, %)", scale=100)
    s_ori = dict(df=ori, label="Orientation signal\n(cvR² over null, %)", scale=100)
    dr = decode_r()
    if set(DORSAL) <= set(dr.columns):
        s_dec = dict(df=dr, label="Value decoding\n(r, decoded vs true)", chance=0,
                     fmt="{:.1f}", points=True)
    else:                                   # per-ROI decoding not in yet
        print(f"decoding r only for {list(dr.columns)}; strip shows cvR² instead")
        s_dec = dict(df=val, label="Value signal\n(cvR² over null, %)", scale=100,
                     points=True)
    s_val = dict(s_val, points=True)
    s_ori = dict(s_ori, points=True)
    out = []
    g = version_beta(val, tab, "value-code strength", "fig_gradient_v2_strength.pdf", verts,
                     s_val, bg=maps["val"])
    out.append(("v2 value-code strength (brain + betas)", *g[:3]))
    v2_test = g[:3]
    eu = eu_measure()
    g = version_beta(eu, tab, "value decoding precision", "fig_gradient_v3_decoding.pdf",
                     verts, s_dec, bg=maps["val"])
    out.append(("v3 value decoding precision", *g[:3]))
    g = version_beta(ori, tab, "orientation-code strength",
                     "fig_gradient_v2b_orientation.pdf", verts, s_ori, bg=maps["ori"])
    out.append(("v2b orientation-code strength", *g[:3]))
    version_unique(val, tab, "fig_gradient_v4_unique.pdf", s_dec, v2_test)
    for row in out:
        print(f"{row[0]:42s} gradient {row[1]:+.3f} ± {row[2]:.3f} SE, P(≤0) = {row[3]:.3f}")
    print(f"n = {len(tab)}; figures in {FIG}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", default=str(PARAMS))
    a = p.parse_args()
    main(a.params)
