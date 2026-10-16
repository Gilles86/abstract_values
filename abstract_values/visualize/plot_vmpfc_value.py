"""Is there a value code in vmPFC/OFC? One compact figure.

a  ROI (subject-native aparc mOFC + lOFC + rostral ACC) on one participant's
   T1w and on the mean EPI of the same slice: the susceptibility dropout.
b  Signal quality: temporal SNR per ROI.
c  Encoding models: fraction of ROI voxels whose cross-validated R2 beats the
   null (training-mean) model, for the linear value ramp, the tuned value
   aPRF, the 8-basis value wPRF, and the orientation von Mises PRF (same
   parameter count as the aPRF). Grey band = the same fraction in random
   eroded white matter (noise floor).
d  Bayesian decoding of presented value (100 voxels, spherical noise):
   within-run r(decoded, presented), per participant.
e  Univariate parametric "value signal": ROI-mean single-trial gabor beta
   correlated with presented value and with the bid (within run).

Inputs (cluster-side reduces, rsynced back):
  notes/data/vmpfc_value_summary.tsv   (collect_vmpfc_value.py)
  notes/data/vmpfc_decoding_r.tsv      (collect_roi_decoding_r.py)

    python -m abstract_values.visualize.plot_vmpfc_value
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn import image as nimage
from scipy import stats

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelpad": 3, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "notes" / "data"
OUT = REPO / "notes" / "figures" / "vmpfc_value_code"
BIDS = Path("/data/ds-abstractvalue")
EXAMPLE = "03"

AMBER, TEAL, GREY = "#D9730D", "#1B8A80", "#9C9C9C"
AMBER_L = "#F0B27A"
ROI_ORDER = ["BensonV1", "IPS0", "IPS12", "IPS345", "vmPFCOFCaparc"]
ROI_LABEL = {"BensonV1": "V1", "IPS0": "IPS0", "IPS12": "IPS1–2",
             "IPS345": "IPS3–5", "vmPFCOFCaparc": "vmPFC/\nOFC"}
MODELS = [("linear", "Linear value", AMBER_L), ("aprf", "Tuned value (aPRF)", AMBER),
          ("wprf", "Value wPRF", "#9A4F08"), ("orientation", "Orientation PRF", TEAL)]


def letter(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="right")


def xticks(ax):
    ax.set_xticks(range(len(ROI_ORDER)))
    ax.set_xticklabels([ROI_LABEL[r] for r in ROI_ORDER])
    ax.set_xlim(-0.6, len(ROI_ORDER) - 0.4)
    ax.axvline(len(ROI_ORDER) - 1.5, color="0.8", lw=0.6, zorder=0)


def group_points(ax, x, vals, color, marker="o", ms=5, jitter=0.0, alpha=.35):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    rng = np.random.default_rng(1)
    ax.scatter(x + rng.uniform(-jitter, jitter, len(vals)), vals, s=5, color=color,
               alpha=alpha, lw=0, zorder=2)
    m, se = vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))
    ax.errorbar(x, m, se, fmt=marker, color=color, ms=ms, mec="0.15", mew=0.8,
                elinewidth=1.0, capsize=0, zorder=3)


def panel_anatomy(ax_t1, ax_epi):
    t1 = nib.load(str(next((BIDS / "derivatives" / "fmriprep" / f"sub-{EXAMPLE}")
                           .glob("ses-1/anat/*desc-preproc_T1w.nii.gz"))))
    mask = nib.load(str(BIDS / "derivatives" / "masks" / f"sub-{EXAMPLE}" / "anat"
                        / f"sub-{EXAMPLE}_space-T1w_desc-vmPFCOFCaparc_mask.nii.gz"))
    epi = nimage.resample_to_img(
        str(next((BIDS / "derivatives" / "fmriprep" / f"sub-{EXAMPLE}")
                 .glob("ses-1/func/*run-1_space-T1w_boldref.nii.gz"))),
        t1, force_resample=True, copy_header=True)
    mask = nimage.resample_to_img(mask, t1, interpolation="nearest",
                                  force_resample=True, copy_header=True)
    md = mask.get_fdata() > 0
    # sagittal slice through the right hemisphere's mask centroid, 6 mm off midline
    ijk = np.argwhere(md)
    xyz = nib.affines.apply_affine(t1.affine, ijk)
    target = np.array([6.0, *xyz[:, 1:].mean(0)])
    i = int(round(np.linalg.solve(t1.affine[:3, :3], target - t1.affine[:3, 3])[0]))
    t1d = t1.get_fdata()
    ys, xs = np.nonzero(np.rot90(t1d[i] > 0.1 * np.percentile(t1d[t1d > 0], 99)))
    for ax, img, title in [(ax_t1, t1d, "T1w"), (ax_epi, epi.get_fdata(), "Mean EPI")]:
        sl = np.rot90(img[i])
        m = np.rot90(md[i])
        vmax = np.percentile(img[img > 0], 99)
        ax.imshow(sl, cmap="gray", vmin=0, vmax=vmax, interpolation="none")
        ax.contour(m, levels=[0.5], colors=[AMBER], linewidths=0.8)
        # same crop on both: the dropout is the missing EPI inside the contour
        ax.set_xlim(xs.min() + 5, xs.max() - 5); ax.set_ylim(ys.max() - 40, ys.min() + 5)
        ax.set_axis_off()
        ax.text(0.03, 0.97, title, transform=ax.transAxes, color="w", fontsize=7,
                va="top")


def main():
    summ = pd.read_csv(DATA / "vmpfc_value_summary.tsv", sep="\t", dtype={"subject": str})
    dec = pd.read_csv(DATA / "vmpfc_decoding_r.tsv", sep="\t", dtype={"subject": str})

    fig = plt.figure(figsize=(7.25, 5.0))
    ax_t1 = fig.add_axes([0.0, 0.60, 0.125, 0.34])
    ax_epi = fig.add_axes([0.13, 0.60, 0.125, 0.34])
    ax_q = fig.add_axes([0.33, 0.60, 0.25, 0.35])
    ax_cv = fig.add_axes([0.65, 0.60, 0.34, 0.35])
    ax_dec = fig.add_axes([0.08, 0.10, 0.40, 0.37])
    ax_uni = fig.add_axes([0.59, 0.10, 0.40, 0.37])

    # a
    panel_anatomy(ax_t1, ax_epi)
    letter(ax_t1, "a", x=-0.02, y=1.0)
    ax_t1.text(0.0, -0.03, "vmPFC/OFC: mOFC + lOFC + rostral ACC", transform=ax_t1.transAxes,
               ha="left", va="top", fontsize=6.5, color=AMBER)

    # b: tSNR
    for k, r in enumerate(ROI_ORDER):
        v = summ.loc[summ.roi == r, "tsnr"]
        group_points(ax_q, k, v, AMBER if r.startswith("vmPFC") else "0.35", jitter=.15)
    wm = summ.loc[summ.roi == "WM", "tsnr"].mean()
    ax_q.set_ylabel("Temporal SNR")
    ax_q.set_ylim(0, None)
    xticks(ax_q)
    ax_q.set_xticklabels(["V1", "IPS0", "IPS\n1–2", "IPS\n3–5", "vmPFC/\nOFC"])
    letter(ax_q, "b")

    # c: fraction of voxels beating null
    off = np.linspace(-0.27, 0.27, len(MODELS))
    for (key, lab, col), o in zip(MODELS, off):
        for k, r in enumerate(ROI_ORDER):
            group_points(ax_cv, k + o, summ.loc[summ.roi == r, f"{key}_frac"], col,
                         ms=3.8, alpha=.0)
    floor = summ[summ.roi == "WM"][[f"{k}_frac" for k, _, _ in MODELS]].mean()
    ax_cv.axhspan(floor.min(), floor.max(), color=GREY, alpha=.35, lw=0, zorder=0)
    ax_cv.text(len(ROI_ORDER) - 0.45, floor.max() + .005, "White-matter floor",
               color="0.4", fontsize=6.5, ha="right", va="bottom")
    ax_cv.set_ylabel("Voxels with cvR² > null")
    ax_cv.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    xticks(ax_cv)
    for i, (key, lab, col) in enumerate(MODELS):
        ax_cv.text(0.02, 0.99 - i * 0.075, lab, color=col, transform=ax_cv.transAxes,
                   fontsize=6.5, va="top")
    letter(ax_cv, "c")

    # d: decoding
    for dname, col, o in [("Linear", AMBER_L, -0.15), ("Tuned (aPRF)", AMBER, 0.15)]:
        for k, r in enumerate(ROI_ORDER):
            group_points(ax_dec, k + o, dec.loc[(dec.roi == r) & (dec.decoder == dname), "r"],
                         col, jitter=.07)
    ax_dec.axhline(0, color="0.6", lw=0.6, ls="--", zorder=0)
    ax_dec.set_ylabel("Decoding r (decoded, presented value)")
    xticks(ax_dec)
    vm = dec[(dec.roi == "vmPFCOFCaparc")].groupby("decoder").r.mean()
    ax_dec.text(4.1, 0.36, f"r = {vm.min():.2f}–{vm.max():.2f}", fontsize=6.5,
                ha="center", va="bottom", color="0.25")
    ax_dec.text(0.02, 0.99, "Linear decoder", color=AMBER_L, transform=ax_dec.transAxes,
                fontsize=6.5, va="top")
    ax_dec.text(0.02, 0.92, "Tuned (aPRF) decoder", color=AMBER, transform=ax_dec.transAxes,
                fontsize=6.5, va="top")
    letter(ax_dec, "d", x=-0.1)

    # e: univariate
    for key, lab, col, o in [("r_value", "Presented value", AMBER, -0.15),
                             ("r_bid", "Bid", "0.3", 0.15)]:
        for k, r in enumerate(ROI_ORDER):
            group_points(ax_uni, k + o, summ.loc[summ.roi == r, key], col, jitter=.07)
    ax_uni.axhline(0, color="0.6", lw=0.6, ls="--", zorder=0)
    wmv = summ.loc[summ.roi == "WM", ["r_value", "r_bid"]].mean()
    ax_uni.axhspan(wmv.min() - .002, wmv.max() + .002, color=GREY, alpha=.5, lw=0, zorder=0)
    ax_uni.add_patch(plt.Rectangle((0.60, 0.955), 0.06, 0.03, transform=ax_uni.transAxes,
                                   color=GREY, alpha=.5, lw=0))
    ax_uni.text(0.68, 0.97, "White-matter floor", transform=ax_uni.transAxes,
                color="0.4", fontsize=6.5, va="center")
    ax_uni.set_ylabel("r with ROI-mean beta")
    xticks(ax_uni)
    ax_uni.text(0.02, 0.99, "Presented value", color=AMBER, transform=ax_uni.transAxes,
                fontsize=6.5, va="top")
    ax_uni.text(0.02, 0.92, "Bid", color="0.3", transform=ax_uni.transAxes,
                fontsize=6.5, va="top")
    letter(ax_uni, "e", x=-0.1)

    for ax in (ax_q, ax_cv, ax_dec, ax_uni):
        sns.despine(ax=ax, offset=4, trim=False)
        ax.tick_params(axis="x", length=0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    print(OUT.with_suffix(".pdf"))


def stats_table():
    """Group statistics printed for the written answer."""
    summ = pd.read_csv(DATA / "vmpfc_value_summary.tsv", sep="\t", dtype={"subject": str})
    dec = pd.read_csv(DATA / "vmpfc_decoding_r.tsv", sep="\t", dtype={"subject": str})
    rois = ROI_ORDER + ["vmPFCOFC", "WM"]
    print("\n== signal / encoding (mean across participants)")
    cols = ["n_vox", "tsnr", "boldref_rel", "glm_r2"] + [f"{k}_frac" for k, _, _ in MODELS] \
        + [f"{k}_dmean" for k, _, _ in MODELS] + ["bid_vs_value_frac", "bid_vs_value_dmean"]
    print(summ.groupby("roi")[[c for c in cols if c in summ]].mean().reindex(rois).T.round(4).to_string())
    print("\n== encoding: ROI vs white-matter floor (paired t, frac beating null)")
    wm = summ[summ.roi == "WM"].set_index("subject")
    for r in rois[:-1]:
        d = summ[summ.roi == r].set_index("subject")
        for k, _, _ in MODELS:
            x = (d[f"{k}_frac"] - wm[f"{k}_frac"]).dropna()
            t, p = stats.ttest_1samp(x, 0)
            print(f"  {r:14s} {k:12s} diff={x.mean():+.3f} t({len(x)-1})={t:.2f} p={p:.3g}")
        x = (d["aprf_frac"] - d["orientation_frac"]).dropna()
        t, p = stats.ttest_1samp(x, 0)
        print(f"  {r:14s} aPRF - orientation frac diff={x.mean():+.3f} t={t:.2f} p={p:.3g}")
    print("\n== decoding r")
    for (r, dn), g in dec.groupby(["roi", "decoder"]):
        t, p = stats.ttest_1samp(g.r.dropna(), 0)
        print(f"  {r:14s} {dn:13s} r={g.r.mean():+.3f}±{g.r.sem():.3f} n={len(g)} t={t:.2f} p={p:.3g}")
    print("\n== univariate (Fisher z t-test across participants)")
    for r in rois:
        g = summ[summ.roi == r]
        for k in ["r_value", "r_bid", "r_orientation", "t_value_ori", "t_bid_ori",
                  "r_resp_bid", "r_resp_value"]:
            if k not in g:
                continue
            v = g[k].dropna()
            z = np.arctanh(v) if k.startswith("r_") else v
            t, p = stats.ttest_1samp(z, 0)
            print(f"  {r:14s} {k:14s} mean={v.mean():+.3f} t({len(v)-1})={t:.2f} p={p:.3g}")


if __name__ == "__main__":
    main()
    stats_table()
