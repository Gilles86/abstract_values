"""Subject-by-subject decoding data-quality scatters.

For each subject, plots decoded-vs-true scatter panels for both decoded
quantities and both ROIs:

    orientation · BensonV1 | orientation · NPCr | value · BensonV1 | value · NPCr

Each panel shows one point per trial (decoded posterior mean vs the true
stimulus), an identity line, and the relevant correlation:

  - orientation is π-periodic (axial), so we use a circular-circular
    correlation (Jammalamadaka–Sarma, doubled-angle) and the decoded
    estimate is the circular mean of the posterior.
  - value is linear, so Pearson r and a posterior-mean estimate.

Reads the per-subject ``*_pars.tsv`` written by decode_gabor / decode_value:
columns ``session, run, trial_nr, true_<q>, <grid cols...>`` where the grid
column *names* are the stimulus grid points and the values are the (unnormalised)
posterior over that grid.

Intended as a quick data-quality look — e.g. new subjects 13 & 14:

    python -m abstract_values.visualize.decoding_quality_scatter --subjects 13 14
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr

from abstract_values.utils.data import BIDS_FOLDER

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})

DECODE = Path(BIDS_FOLDER) / "derivatives" / "decoding"
DEFAULT_OUT = Path(BIDS_FOLDER) / "derivatives" / "qa" / "decoding_quality_scatter.pdf"

# (decoded quantity dir, true-column, ROI, axis label, is-circular)
PANELS = [
    ("gabor", "true_orientation_rad", "BensonV1", "Orientation (deg)", True),
    ("gabor", "true_orientation_rad", "NPCr",     "Orientation (deg)", True),
    ("value", "true_value_chf",       "BensonV1", "Value (CHF)",       False),
    ("value", "true_value_chf",       "NPCr",     "Value (CHF)",       False),
]
ROI_COLOUR = {"BensonV1": "#3B5BA5", "NPCr": "#E76F51"}


# Noise model whose decoding pars to read: full | spherical | geodesic.
# Set from --noise in main().
NOISE = "full"


def _pars_path(quantity, subject, roi, nvoxels, smoothed):
    """Locate the decode pars TSV, tolerating an optional _lambda-X suffix
    (variant runs use LAMBD=0.1; the original nvoxels=100 run had none)."""
    smooth = "_smoothed" if smoothed else ""
    d = DECODE / quantity / f"sub-{subject}" / "func"
    stem = f"sub-{subject}_mask-{roi}_nvoxels-{nvoxels}_noise-{NOISE}{smooth}"
    hits = sorted(d.glob(f"{stem}_pars.tsv")) or sorted(d.glob(f"{stem}_lambda-*_pars.tsv"))
    return hits[0] if hits else d / f"{stem}_pars.tsv"   # non-existent if no hit


def discover_subjects(nvoxels, smoothed):
    """All subjects with at least one decoding pars file at this nvoxels/noise.
    Study subjects (numeric) first, then pilots."""
    smooth = "_smoothed" if smoothed else ""
    subs = set()
    for q, _, roi, _, _ in PANELS:
        for pat in (f"*_mask-{roi}_nvoxels-{nvoxels}_noise-{NOISE}{smooth}_pars.tsv",
                    f"*_mask-{roi}_nvoxels-{nvoxels}_noise-{NOISE}{smooth}_lambda-*_pars.tsv"):
            for p in (DECODE / q).glob(f"sub-*/func/{pat}"):
                subs.add(p.name.split("_")[0].removeprefix("sub-"))
    return sorted(subs, key=lambda s: (0 if s[0].isdigit() else 1, s))


def _load(quantity, subject, true_col, roi, nvoxels, smoothed):
    """Return (true, decoded) arrays. Decoded = posterior mean over the grid
    (circular for orientation). None if the file is missing/empty."""
    p = _pars_path(quantity, subject, roi, nvoxels, smoothed)
    if not p.exists():
        return None
    df = pd.read_csv(p, sep="\t")
    meta = ["session", "run", "trial_nr", true_col]
    grid = np.array([float(c) for c in df.columns if c not in meta])
    post = df[[c for c in df.columns if c not in meta]].to_numpy(dtype=float)
    w = post / np.clip(post.sum(axis=1, keepdims=True), 1e-12, None)  # normalise rows
    true = df[true_col].to_numpy(dtype=float)
    if true_col.endswith("_rad"):                      # orientation: circular mean (π-periodic)
        ang = 2.0 * grid
        dec = 0.5 * np.arctan2((w * np.sin(ang)).sum(1),
                               (w * np.cos(ang)).sum(1)) % np.pi
    else:                                              # value: posterior mean
        dec = (w * grid).sum(1)
    ok = np.isfinite(true) & np.isfinite(dec)
    return true[ok], dec[ok]


def _circular_corr(a_rad, b_rad):
    """Jammalamadaka–Sarma circular-circular correlation on a π-periodic
    axis (doubled-angle). Returns r in [-1, 1]."""
    a, b = 2.0 * np.asarray(a_rad), 2.0 * np.asarray(b_rad)
    a0 = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    b0 = np.arctan2(np.sin(b).mean(), np.cos(b).mean())
    sa, sb = np.sin(a - a0), np.sin(b - b0)
    denom = np.sqrt((sa**2).sum() * (sb**2).sum())
    return float((sa * sb).sum() / denom) if denom > 0 else np.nan


def _panel(ax, true, dec, label, circular, colour):
    if circular:
        true, dec = np.rad2deg(true), np.rad2deg(dec)
        r = _circular_corr(np.deg2rad(true), np.deg2rad(dec))
        rtxt = f"$r_{{circ}}$ = {r:.2f}"
        lim = (0, 180); ticks = [0, 45, 90, 135, 180]
    else:
        r, _ = pearsonr(true, dec)
        rtxt = f"$r$ = {r:.2f}"
        lo, hi = min(true.min(), dec.min()), max(true.max(), dec.max())
        pad = 0.05 * (hi - lo)
        lim = (lo - pad, hi + pad); ticks = None
    ax.scatter(true, dec, s=9, color=colour, alpha=0.45, linewidths=0)
    ax.plot(lim, lim, "--", color="0.5", lw=0.8, zorder=0)
    ax.set_xlim(lim); ax.set_ylim(lim)
    if ticks: ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"True {label.lower()}")
    ax.set_ylabel(f"Decoded {label.lower()}")
    ax.text(0.05, 0.95, f"{rtxt}\nn = {len(true)}", transform=ax.transAxes,
            ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.5))
    return r


def run(subjects, nvoxels, smoothed, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    smooth_lbl = "smoothed" if smoothed else "unsmoothed"
    summary = []
    with PdfPages(out) as pdf:
        for s in subjects:
            fig, axes = plt.subplots(1, 4, figsize=(13, 3.6),
                                     constrained_layout=True)
            for ax, (q, true_col, roi, label, circ) in zip(axes, PANELS):
                loaded = _load(q, s, true_col, roi, nvoxels, smoothed)
                if loaded is None or len(loaded[0]) == 0:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, f"no data\n{q}·{roi}", ha="center",
                            va="center", transform=ax.transAxes, color="0.6")
                    continue
                true, dec = loaded
                r = _panel(ax, true, dec, label, circ, ROI_COLOUR[roi])
                ax.set_title(f"{label.split()[0]} · {roi}", fontsize=9, color="0.2")
                summary.append(dict(subject=s, quantity=q, roi=roi,
                                    r=r, n=len(true)))
            fig.suptitle(f"sub-{s}  ·  decoding quality  ·  {smooth_lbl}  ·  "
                         f"nvoxels={nvoxels}", fontsize=11, y=1.06)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # summary table page
        if summary:
            sdf = pd.DataFrame(summary)
            tbl = sdf.pivot_table(index="subject",
                                  columns=["quantity", "roi"], values="r")
            fig, ax = plt.subplots(figsize=(8.5, 1.2 + 0.4 * len(subjects)))
            ax.set_axis_off()
            ax.set_title("Decoding correlation summary (r)", fontsize=11, pad=12)
            t = ax.table(cellText=np.round(tbl.values, 2),
                         rowLabels=[f"sub-{i}" for i in tbl.index],
                         colLabels=[f"{q}\n{roi}" for q, roi in tbl.columns],
                         loc="center", cellLoc="center")
            t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.6)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(out.with_suffix(".tsv"), sep="\t", index=False)
    print(f"Wrote {out}\nSidecar: {out.with_suffix('.tsv')}")
    print(sdf.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+", default=None,
                   help="subject labels; default: all subjects with decoding data")
    p.add_argument("--nvoxels", default="100",
                   help="voxel-selection tag, e.g. 100, 250, 0, or fdr05")
    p.add_argument("--noise", default="full", choices=["full", "spherical", "geodesic"],
                   help="noise model whose decoding pars to read")
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    global NOISE
    NOISE = args.noise
    subjects = args.subjects or discover_subjects(args.nvoxels, args.smoothed)
    if not subjects:
        raise SystemExit(f"No decoding pars found for nvoxels={args.nvoxels}")
    print(f"Subjects ({len(subjects)}): {subjects}")
    run(subjects, args.nvoxels, args.smoothed, Path(args.out))


if __name__ == "__main__":
    main()
