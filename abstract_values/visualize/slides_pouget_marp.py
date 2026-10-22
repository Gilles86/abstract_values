"""Slide-size figures for the abstract_values section of the Pouget MARP deck.

Regenerated at slide size (not cropped from the paper-size figures): about
11.5 x 4.3 in, shown ~1150 px wide on the 1280 x 720 slide, 14-16 pt type.

  av_mappings.png    the two mappings and the value distributions they imply
  av_circle_line.png V1 a circle, IPS a line: wrapped - unwrapped cvR^2 per ROI
  av_lambda.png      fixed-value vs quantile tuning, and the cvR^2 profile over lambda
  av_wiring.png      orientation- vs value-wired V1 input, and the tuning-matched
                     connectivity gain over lambda'
  av_task.png, av_bias.png, av_maps.png   crops of the paper-size figures
                     (Bedi et al. Figs 2 and 4, fig_summary maps)

Usage:
    python -m abstract_values.visualize.slides_pouget_marp \
        --out-dir /Users/gdehol/git/presentations/2026/pouget_visit/figures
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

import abstract_values.visualize.figure_tuning_connectivity as ftc
from abstract_values.connective_fields.gates import load_mappings
from abstract_values.connective_fields.test_coupling import IEM_GRID

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "notes" / "figures"
DATA = REPO / "notes" / "data"
SES, SES_COL, SES_NAME = ftc.SES, ftc.SES_COL, {"cdf": "Misaligned", "inverse_cdf": "Aligned"}
ORI_COL, VAL_COL = ftc.ORI_COL, ftc.VAL_COL
ROI_COL, ROI_NAME = ftc.ROI_COL, ftc.ROI_NAME
DPI = 200

mpl.rcParams.update({
    "font.family": "Helvetica", "font.size": 15, "axes.labelsize": 15,
    "axes.titlesize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.linewidth": 1.1, "xtick.major.width": 1.1, "ytick.major.width": 1.1,
    "xtick.major.size": 5, "ytick.major.size": 5, "lines.linewidth": 2.4,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def despine(ax, left=True):
    for s, v in (("left", left),):
        ax.spines[s].set_visible(v)
    ax.spines["left"].set_position(("outward", 6))
    ax.spines["bottom"].set_position(("outward", 6))
    if not left:
        ax.set_yticks([])


def save(fig, out):
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


def kde(v, grid, bw=1.4):
    return np.exp(-.5 * ((grid[:, None] - v[None]) / bw) ** 2).sum(1)


# ── figures ──────────────────────────────────────────────────────────────────

def mappings(out):
    ori, maps = load_mappings()
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3),
                           gridspec_kw=dict(wspace=.35, left=.08, right=.97,
                                            bottom=.18, top=.9))
    a, b = ax
    g = np.linspace(0, 44, 400)
    for i, m in enumerate(SES):
        a.plot(ori, maps[m], color=SES_COL[m])
        v = maps[m][1:-1]                         # the 23 shown orientations
        d = kde(v, g)
        b.plot(g, d / d.max() + 1.25 * (1 - i), color=SES_COL[m])
        b.plot(v, np.full(23, 1.25 * (1 - i) - .08), "|", ms=9, mew=1.3,
               color=SES_COL[m])
        b.text((1, 30)[i], 1.25 * (1 - i) + (.6, .42)[i], SES_NAME[m], color=SES_COL[m],
               ha=("left", "center")[i], fontsize=15)
    a.set_xlim(0, 180)
    a.set_xticks([0, 45, 90, 135, 180])
    a.set_ylim(0, 44)
    a.set_yticks([2, 12, 22, 32, 42])
    a.set_xlabel("Orientation (°)")
    a.set_ylabel("Value (CHF)")
    a.set_title("23 orientations, shown equally often", loc="left", fontsize=15,
                color="0.35")
    for i, m in enumerate(SES):
        a.text(4, 40 - 5 * i, SES_NAME[m], color=SES_COL[m], fontsize=15)
    b.set_xlim(0, 44)
    b.set_xticks([2, 12, 22, 32, 42])
    b.set_xlabel("Value (CHF)")
    b.set_ylabel("Value density")
    b.set_title("Same stimuli, different value priors", loc="left", fontsize=15,
                color="0.35")
    despine(a)
    despine(b, left=False)
    save(fig, out)


def circle_line(out):
    ps = pd.read_csv(DATA / "quantile_models_per_subject.tsv", sep="\t",
                     dtype={"subject": str})
    rois = [("BensonV1", "V1"), ("BensonV2", "V2"), ("BensonV3", "V3"), ("hV4", "hV4"),
            ("V3ab", "V3a/b"), ("IPS0", "IPS0"), ("IPS12", "IPS1–2"), ("IPS345", "IPS3–5")]
    ps = ps[(ps.selection == "All voxels") & ps.smoothed]
    d = ps.set_index(["subject", "roi"])
    diff = (d["quantile_wrapped"] - d["quantile"]).rename("diff").reset_index()
    fig = plt.figure(figsize=(11.5, 4.3))
    # the two encoding models, one voxel each, preferring the far end
    th = np.linspace(7.5, 172.5, 400)
    stim = np.arange(7.5, 172.6, 7.5)
    mu = 165.
    circ = lambda x: np.exp(3 * (np.cos(np.deg2rad(2 * (x - mu))) - 1))
    line = lambda x: np.exp(-.5 * ((x - mu) / 22) ** 2)
    for k, (f, col, t, sub) in enumerate((
            (circ, ORI_COL, "Circle", "von Mises on orientation"),
            (line, VAL_COL, "Line", "Gaussian on orientation rank"))):
        s1 = fig.add_axes([.06, .6 - .43 * k, .27, .26])
        s1.plot(th, f(th), color=col)
        s1.plot(stim, f(stim), "o", ms=5, color=col, mew=0)
        s1.set_xlim(0, 180)
        s1.set_ylim(-.05, 1.1)
        s1.set_xticks([7.5, 90, 172.5])
        s1.set_xticklabels(["7.5°\n2 CHF", "90°", "172.5°\n42 CHF"] if k else [])
        s1.set_title(f"{t}: {sub}", loc="left", color=col, fontsize=14)
        despine(s1, left=False)
    fig.text(.03, .5, "One voxel's response", rotation=90, va="center", fontsize=14)
    # data
    ax = fig.add_axes([.43, .2, .55, .68])
    ax.axvspan(4.5, 7.5, color="0.94", lw=0, zorder=0)
    rng = np.random.default_rng(0)
    for i, (r, lab) in enumerate(rois):
        y = diff.loc[diff.roi == r, "diff"].dropna()
        col = ORI_COL if y.mean() > 0 else VAL_COL
        ax.plot(i + rng.uniform(-.12, .12, len(y)), y, "o", ms=3.5, color=col,
                alpha=.25, mew=0)
        ax.errorbar(i, y.mean(), y.sem(), color=col, marker="o", ms=9, capsize=0,
                    elinewidth=2.4)
    piv = diff.pivot(index="subject", columns="roi", values="diff")
    t = stats.ttest_rel(piv[["IPS0", "IPS12", "IPS345"]].mean(1),
                        piv[["BensonV1", "BensonV2", "BensonV3"]].mean(1))
    ax.axhline(0, color="0.5", lw=1, ls=":")
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels([l for _, l in rois])
    ax.set_xlim(-.5, len(rois) - .5)
    ax.set_ylim(-.8, .8)
    ax.set_ylabel("Circle − line, cvR² (pp)")
    ax.text(.01, .97, "Circle fits better", transform=ax.transAxes, color=ORI_COL,
            va="top", fontsize=14)
    ax.text(.01, .03, "Line fits better", transform=ax.transAxes, color=VAL_COL,
            fontsize=14)
    ax.set_title(f"IPS vs V1–3: t(34) = {t.statistic:.1f}, p < .001", loc="right",
                 fontsize=13, color="0.3")
    despine(ax)
    save(fig, out)


def lam(out):
    ori, maps = load_mappings()
    d = ftc.load(DATA / "tuning_connectivity_summary.tsv")
    d = d[d.smoothed & (d.model == "coding")]
    fig = plt.figure(figsize=(11.5, 4.3))
    V = ftc.V
    # a: one voxel at lambda = 1
    ax = fig.add_axes([.05, .2, .2, .62])
    for m in SES:
        ax.plot(V, ftc.gauss(ftc.x_lam(V, ori, maps[m], 1.)), color=SES_COL[m])
    ax.set_xlim(2, 42)
    ax.set_xticks([2, 22, 42])
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Value (CHF)")
    ax.set_ylabel("One voxel")
    ax.set_title("Tuning moves with p(v)", loc="left", fontsize=15)
    for i, m in enumerate(SES):
        ax.text(42, 1.0 - .13 * i, SES_NAME[m], color=SES_COL[m], ha="right", fontsize=13)
    despine(ax, left=False)
    # b: predicted preferred values, Aligned vs Misaligned
    ax = fig.add_axes([.34, .2, .24, .66])
    modes = np.linspace(2, 42, 161)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("l", [VAL_COL, ORI_COL])
    for l in (0., .5, 1.):
        pv = {m: np.array([V[np.argmax(ftc.gauss(ftc.x_lam(V, ori, maps[m], l), mu=x0))]
                           for x0 in modes]) for m in SES}
        c = cmap(l)
        ax.plot(pv["inverse_cdf"], pv["cdf"], color=c, lw=2.6 if l in (0, 1) else 1.6)
        ax.text(4, 39 - 4 * (1 - l) / .5, f"λ = {l:g}", color=c, fontsize=13)
    ax.set_xlim(2, 42)
    ax.set_ylim(2, 42)
    ax.set_xticks([2, 12, 22, 32, 42])
    ax.set_yticks([2, 12, 22, 32, 42])
    ax.set_aspect("equal")
    ax.set_xlabel("Preferred value, Aligned")
    ax.set_ylabel("Preferred value,\nMisaligned")
    ax.set_title("Prediction, every voxel", loc="left", fontsize=15)
    despine(ax)
    # c: data
    ax = fig.add_axes([.68, .2, .23, .66])
    ends = {}
    for roi in ftc.ROIS:
        p = d[d.roi == roi].pivot(index="subject", columns="lam", values="cvr2")
        p = p.sub(p[0.0], axis=0) * 1e3
        m, se = p.mean(), p.sem()
        x = p.columns.to_numpy(float)
        ip = roi.startswith("IPS")
        ax.fill_between(x, m - se, m + se, color=ROI_COL[roi], alpha=.18 if ip else .1,
                        lw=0)
        ax.plot(x, m, color=ROI_COL[roi], lw=2.6 if ip else 1.6)
        ends[roi] = m.iloc[-1]
    ax.axhline(0, color="0.6", lw=1)
    for l, c in ((0, VAL_COL), (1, ORI_COL)):
        ax.axvline(l, color=c, lw=1.2, alpha=.5, zorder=0)
    ax.set_xlim(-.5, 1.5)
    ax.set_xticks([-.5, 0, .5, 1, 1.5])
    ax.set_xlabel("λ")
    ax.set_ylabel("Δ cvR² vs λ = 0 (×10⁻³)")
    ax.set_title("Data, n = 35", loc="left", fontsize=15)
    last = -9
    for roi, y in sorted(ends.items(), key=lambda kv: kv[1]):
        y = max(y, last + .5)
        last = y
        ax.text(1.55, y, ROI_NAME[roi], color=ROI_COL[roi], va="center", fontsize=13)
    despine(ax)
    save(fig, out)


def wiring(out):
    ori, maps = load_mappings()
    d = ftc.load(DATA / "tuning_connectivity_summary.tsv")
    d = d[d.smoothed & (d.lam == 1.0)]
    fig = plt.figure(figsize=(11.5, 4.3))
    qq = (IEM_GRID / 7.5 - .5) / 23
    for k, (l, col, t) in enumerate(((1., ORI_COL, "Orientation-wired (λ′ = 1)"),
                                     (0., VAL_COL, "Value-wired (λ′ = 0)"))):
        ax = fig.add_axes([.05, .58 - .42 * k, .36, .3])
        for m in SES:
            xc = (1 - l) * np.interp(IEM_GRID, ori, maps[m]) + l * (2 + 40 * qq)
            ax.plot(IEM_GRID, ftc.gauss(xc), color=SES_COL[m])
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xticklabels([] if k == 0 else ["0", "45", "90", "135", "180"])
        ax.set_title(t, color=col, loc="left", fontsize=15)
        despine(ax, left=False)
        if k == 0:
            ax.text(178, .75, "Same in both sessions", ha="right", fontsize=13,
                    color="0.4")
        if k == 1:
            ax.set_xlabel("V1 orientation channel (°)")
            for i, m in enumerate(SES):
                ax.text(178, .85 - .3 * i, SES_NAME[m], color=SES_COL[m], ha="right",
                        fontsize=13)
    fig.text(.015, .5, "Weight on V1", rotation=90, va="center", fontsize=15)
    ax = fig.add_axes([.56, .2, .32, .66])
    ends = {}
    for roi in ftc.ROIS:
        sel = d[d.roi == roi]
        j = sel[sel.model == "joint"].pivot(index="subject", columns="lam_conn",
                                            values="cvr2")
        s = sel[sel.model == "joint_shuffled"].pivot(index="subject",
                                                     columns="lam_conn", values="cvr2")
        g = (j - s) * 1e3
        m, se = g.mean(), g.sem()
        x = g.columns.to_numpy(float)
        ip = roi.startswith("IPS")
        ax.fill_between(x, m - se, m + se, color=ROI_COL[roi], alpha=.18 if ip else .1,
                        lw=0)
        ax.plot(x, m, color=ROI_COL[roi], lw=2.6 if ip else 1.6)
        ends[roi] = m.iloc[-1]
    ax.axhline(0, color="0.6", lw=1)
    for l, c in ((0, VAL_COL), (1, ORI_COL)):
        ax.axvline(l, color=c, lw=1.2, alpha=.5, zorder=0)
    ax.set_xlim(-.5, 1.5)
    ax.set_xticks([-.5, 0, .5, 1, 1.5])
    ax.set_xlabel("Connectivity axis λ′")
    ax.set_ylabel("Gain from V1 input\nover shuffle (×10⁻³)")
    last = -9
    for roi, y in sorted(ends.items(), key=lambda kv: kv[1]):
        y = max(y, last + .8)
        last = y
        ax.text(1.55, y, ROI_NAME[roi], color=ROI_COL[roi], va="center", fontsize=13)
    despine(ax)
    save(fig, out)


def fmri_design(out):
    """Trial timeline, session structure and what goes into the encoding model."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    fig = plt.figure(figsize=(11.5, 4.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 4.3)
    ax.axis("off")
    W, H, y0 = 1.45, 1.0, 2.15
    labels = ["Gabor", "Delay", "Bid", "ITI"]
    times = ["1.5 s", "4–5.5 s", "≤ 3.5 s", "~6 s"]
    yy, xx = np.mgrid[-1:1:200j, -1:1:200j]
    th = np.deg2rad(52.5)
    grating = .5 + .12 * np.cos(2 * np.pi * 4 * (xx * np.cos(th) + yy * np.sin(th)))
    grating = np.where(xx ** 2 + yy ** 2 < .8, grating, .55)
    for i, (lab, t) in enumerate(zip(labels, times)):
        x = .3 + i * 1.7
        ax.add_patch(Rectangle((x, y0), W, H, facecolor="0.55", edgecolor="0.25",
                               lw=1.2))
        cx, cy = x + W / 2, y0 + H / 2
        if i == 0:
            ax.imshow(grating, cmap="gray", vmin=0, vmax=1, extent=(cx - .45, cx + .45,
                      cy - .45, cy + .45), zorder=3)
        elif i == 2:
            ax.add_patch(Rectangle((cx - .55, cy - .03), 1.1, .06, facecolor="w", lw=0,
                                   zorder=3))
            ax.add_patch(Rectangle((cx + .12, cy - .12), .06, .24, facecolor="#E8C547",
                                   lw=0, zorder=4))
            ax.text(cx + .15, cy + .17, "27 CHF", color="w", ha="center", fontsize=10)
        else:
            ax.plot(cx, cy, "+", color="w", ms=9, mew=1.6)
        ax.text(cx, y0 - .18, lab, ha="center", va="top", fontsize=14)
        ax.text(cx, y0 - .48, t, ha="center", va="top", fontsize=12, color="0.4")
        if i < 3:
            ax.annotate("", (x + 1.68, cy), (x + W + .02, cy),
                        arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.2))
    ax.text(.3, 3.75, "One trial, ~14 s", fontsize=15, fontweight="bold")
    x0 = 7.4
    ax.text(x0, 3.75, "Two sessions, one mapping each", fontsize=15, fontweight="bold")
    for i, (m, lab) in enumerate((("inverse_cdf", "Aligned"), ("cdf", "Misaligned"))):
        y = 2.75 - .8 * i
        ax.text(x0, y + .4, f"Session {i + 1}: {lab}", color=SES_COL[m], fontsize=13)
        for r in range(8):
            ax.add_patch(FancyBboxPatch((x0 + r * .48, y), .4, .3,
                                        boxstyle="round,pad=0.01,rounding_size=.05",
                                        facecolor=SES_COL[m], alpha=.85, lw=0))
    ax.text(x0, 1.75, "8 runs × 23 orientations, once each;\norder counterbalanced",
            fontsize=12, color="0.35", va="top")
    ax.plot([.3, 11.2], [.95, .95], color="0.85", lw=1)
    steps = ["Single-trial BOLD (GLMsingle)", "Tuning model per voxel",
             "Leave-one-run-out cvR²"]
    xs = [.3, 4.55, 8.1]
    for i, (x, t) in enumerate(zip(xs, steps)):
        ax.text(x, .48, t, fontsize=14, color="0.2", va="center")
        if i < 2:
            ax.annotate("", (xs[i + 1] - .15, .48), (xs[i + 1] - .85, .48),
                        arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.2))
    save(fig, out)


# ── crops of existing figures ────────────────────────────────────────────────

def render(pdf, page=None, dpi=250):
    tmp = Path(tempfile.mkdtemp())
    cmd = ["pdftoppm", "-r", str(dpi), "-png"]
    if page:
        cmd += ["-f", str(page), "-l", str(page)]
    subprocess.run(cmd + [str(pdf), str(tmp / "p")], check=True)
    return sorted(tmp.glob("p-*.png"))[0]


def crop(path, box, out):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    x0, y0, x1, y1 = box
    im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))).save(out)
    print(f"Wrote {out}")


def crops(od):
    paper = REPO / "notes" / "papers" / "bedi_et_al2026.pdf"
    crop(render(paper, 8), (0.18, 0.103, 0.85, 0.44), od / "av_task.png")
    crop(render(paper, 14), (0.18, 0.104, 0.85, 0.46), od / "av_bias.png")
    S = render(FIG / "fig_summary.pdf")
    crop(S, (0, 0.019, 1, 0.19), od / "av_maps.png")
    m = Image.open(od / "av_maps.png")
    a = m.crop((0, 95, 530, 335))
    cd = m.crop((975, 0, 1812, 420)).copy()
    from PIL import ImageDraw
    ImageDraw.Draw(cd).rectangle([400, 50, 440, 90], fill="white")   # panel letter d
    top = Image.new("RGB", (a.width + 60 + cd.width, cd.height), "white")
    top.paste(a, (0, (cd.height - a.height) // 2 + 20))
    top.paste(cd, (a.width + 60, 0))
    top.save(od / "av_maps_coded.png")
    m.crop((50, 655, 1725, 1205)).save(od / "av_maps_preferred.png")
    print("Wrote av_maps_coded.png, av_maps_preferred.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir",
                    default="/Users/gdehol/git/presentations/2026/pouget_visit/figures")
    a = ap.parse_args()
    od = Path(a.out_dir)
    mappings(od / "av_mappings.png")
    fmri_design(od / "av_fmri_design.png")
    circle_line(od / "av_circle_line.png")
    lam(od / "av_lambda.png")
    wiring(od / "av_wiring.png")
    crops(od)
