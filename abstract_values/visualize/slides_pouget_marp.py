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
    # schematic: circle vs line
    s1 = fig.add_axes([.02, .14, .17, .72])
    s2 = fig.add_axes([.2, .14, .12, .72])
    th = np.deg2rad(np.linspace(0, 360, 200))
    s1.plot(np.cos(th), np.sin(th), color=ORI_COL, lw=3)
    for t, lab, ha in ((7.5, "7.5°", "left"), (172.5, "172.5°", "right")):
        p = np.deg2rad(90 - 2 * t)
        s1.plot(np.cos(p), np.sin(p), "o", ms=9, color="0.2")
        s1.text(np.cos(p) + (.12 if ha == "left" else -.12), np.sin(p) + .22, lab,
                ha=ha, fontsize=13)
    s1.text(0, -1.4, "Orientation:\na circle", ha="center", va="top", color=ORI_COL,
            fontsize=15)
    s1.set_xlim(-1.6, 1.6)
    s1.set_ylim(-2.3, 1.6)
    s1.set_aspect("equal")
    s1.axis("off")
    s2.plot([0, 0], [-1, 1], color=VAL_COL, lw=3)
    for y, lab in ((-1, "2 CHF"), (1, "42 CHF")):
        s2.plot(0, y, "o", ms=9, color="0.2")
        s2.text(.12, y, lab, va="center", fontsize=13)
    s2.text(0, -1.4, "Value:\na line", ha="center", va="top", color=VAL_COL, fontsize=15)
    s2.set_xlim(-.5, 1)
    s2.set_ylim(-2.3, 1.6)
    s2.axis("off")
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
    ax.text(.99, .03, f"IPS vs V1–3: t(34) = {t.statistic:.1f}, p < .001",
            transform=ax.transAxes, ha="right", fontsize=13, color="0.3")
    despine(ax)
    save(fig, out)


def lam(out):
    ori, maps = load_mappings()
    d = ftc.load(DATA / "tuning_connectivity_summary.tsv")
    d = d[d.smoothed & (d.model == "coding")]
    fig = plt.figure(figsize=(11.5, 4.3))
    V = ftc.V
    for k, (l, col, t) in enumerate(((0., VAL_COL, "λ = 0: fixed value"),
                                     (1., ORI_COL, "λ = 1: follows p(v)"))):
        ax = fig.add_axes([.05 + .22 * k, .2, .18, .62])
        for m in SES:
            y = ftc.gauss(ftc.x_lam(V, ori, maps[m], l))
            ax.plot(V, y, color=SES_COL[m])
        ax.set_xlim(2, 42)
        ax.set_xticks([2, 22, 42])
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Value (CHF)")
        ax.set_title(t, color=col, loc="left", fontsize=16)
        despine(ax, left=False)
        if k == 0:
            ax.set_ylabel("One voxel")
            for i, m in enumerate(SES):
                ax.text(42, .95 - .14 * i, SES_NAME[m], color=SES_COL[m], ha="right",
                        fontsize=13)
    ax = fig.add_axes([.58, .2, .3, .66])
    ends = {}
    for roi in ftc.ROIS:
        p = d[d.roi == roi].pivot(index="subject", columns="lam", values="cvr2")
        p = p.sub(p[0.0], axis=0) * 1e3
        m, s = p.mean(), p.sem()
        x = p.columns.to_numpy(float)
        ip = roi.startswith("IPS")
        ax.fill_between(x, m - s, m + s, color=ROI_COL[roi], alpha=.18 if ip else .1, lw=0)
        ax.plot(x, m, color=ROI_COL[roi], lw=2.6 if ip else 1.6)
        ends[roi] = m.iloc[-1]
    ax.axhline(0, color="0.6", lw=1)
    for l, c in ((0, VAL_COL), (1, ORI_COL)):
        ax.axvline(l, color=c, lw=1.2, alpha=.5, zorder=0)
    ax.set_xlim(-.5, 1.5)
    ax.set_xticks([-.5, 0, .5, 1, 1.5])
    ax.set_xlabel("Tuning axis λ")
    ax.set_ylabel("Δ cvR² vs λ = 0 (×10⁻³)")
    ys = sorted(ends.items(), key=lambda kv: kv[1])
    last = -9
    for roi, y in ys:
        y = max(y, last + .45)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir",
                    default="/Users/gdehol/git/presentations/2026/pouget_visit/figures")
    a = ap.parse_args()
    od = Path(a.out_dir)
    mappings(od / "av_mappings.png")
    circle_line(od / "av_circle_line.png")
    lam(od / "av_lambda.png")
    wiring(od / "av_wiring.png")
    crops(od)
