"""Trial-level calibration: does posterior SD predict empirical |error|?

For each (quantity × ROI), pulls per-trial posterior SD and per-trial absolute
error (circular for orientation, linear for value), then:

  - groups trials per subject into posterior-SD quintiles (rank-binned) and
    plots mean |error| per quintile — subject lines + group mean ± SEM.
  - reports Spearman ρ(post_SD, |err|) per subject + group mean.

A well-calibrated decoder has positive ρ: trials it flags as uncertain
genuinely have larger errors. This is the *trial-level* analog of the
binned-by-true-stim calibration in decoding_calibration.py.
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
from scipy.stats import spearmanr

from abstract_values.utils.data import BIDS_FOLDER

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150, "savefig.dpi": 300,
})

BIDS = Path(BIDS_FOLDER)
DECODE = BIDS / "derivatives" / "decoding"

# (quantity_dir, roi, label, unit, circular)
PANELS = [
    ("gabor", "BensonV1", "Orientation", "deg", True),
    ("gabor", "NPCr",     "Orientation", "deg", True),
    ("value", "BensonV1", "Value",       "CHF", False),
    ("value", "NPCr",     "Value",       "CHF", False),
]
ROI_COLOUR = {"BensonV1": "#3B5BA5", "NPCr": "#E76F51"}


def _pars_path(quantity, subject, roi, nvoxels, smoothed, noise):
    smooth = "_smoothed" if smoothed else ""
    d = DECODE / quantity / f"sub-{subject}" / "func"
    stem = f"sub-{subject}_mask-{roi}_nvoxels-{nvoxels}_noise-{noise}{smooth}"
    hits = sorted(d.glob(f"{stem}_pars.tsv")) or sorted(d.glob(f"{stem}_lambda-*_pars.tsv"))
    return hits[0] if hits else None


def _per_trial(p, circular):
    """Return (true, dec, post_sd, abs_err) per trial; orientation in rad."""
    df = pd.read_csv(p, sep="\t")
    true_col = "true_orientation_rad" if circular else "true_value_chf"
    meta = ["session", "run", "trial_nr", true_col]
    grid = np.array([float(c) for c in df.columns if c not in meta])
    post = df[[c for c in df.columns if c not in meta]].to_numpy(float)
    w = post / np.clip(post.sum(1, keepdims=True), 1e-12, None)
    true = df[true_col].to_numpy(float)
    if circular:
        a = 2.0 * grid
        c, s = (w * np.cos(a)).sum(1), (w * np.sin(a)).sum(1)
        dec = 0.5 * np.arctan2(s, c) % np.pi
        R = np.clip(np.sqrt(c**2 + s**2), 1e-12, 1.0)
        post_sd = np.sqrt(-2.0 * np.log(R)) / 2.0
        err = (dec - true + np.pi / 2) % np.pi - np.pi / 2     # signed axial
        abs_err = np.abs(err)
    else:
        dec = (w * grid).sum(1)
        post_sd = np.sqrt((w * (grid[None, :] - dec[:, None]) ** 2).sum(1))
        abs_err = np.abs(dec - true)
    return true, dec, post_sd, abs_err


def collect(subjects, nvoxels, smoothed, noise, n_bins=5):
    """For each panel and each subject: per-bin (rank-binned post_SD) mean |err|
    and Spearman ρ(post_SD, |err|). Returns a dict {(q,roi): list of (subject,
    bin_means_df, rho, n_trials)}."""
    out = {(q, roi): [] for q, roi, *_ in PANELS}
    for q, roi, _, unit, circular in PANELS:
        for s in subjects:
            p = _pars_path(q, s, roi, nvoxels, smoothed, noise)
            if p is None:
                continue
            _, _, sd, abs_err = _per_trial(p, circular)
            keep = np.isfinite(sd) & np.isfinite(abs_err)
            sd, abs_err = sd[keep], abs_err[keep]
            if circular:
                sd, abs_err = np.rad2deg(sd), np.rad2deg(abs_err)
            if len(sd) < n_bins * 4:
                continue
            ranks = pd.Series(sd).rank(method="first") - 1
            bins = np.minimum((ranks / len(sd) * n_bins).astype(int), n_bins - 1)
            g = pd.DataFrame({"bin": bins, "sd": sd, "abs_err": abs_err}) \
                  .groupby("bin").agg(sd=("sd", "mean"), err=("abs_err", "mean")).reset_index()
            rho, _ = spearmanr(sd, abs_err)
            out[(q, roi)].append((s, g, float(rho), int(len(sd))))
    return out


def _panel(ax, q, roi, label, unit, rows, n_bins):
    """Per-subject quintile lines + group mean ± SEM."""
    if not rows:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="0.6")
        return
    # per-subject thin lines
    for _, g, _, _ in rows:
        ax.plot(g["bin"] + 1, g["err"], color="0.78", lw=0.7, zorder=1)
    # group mean ± SEM across subjects, per bin
    long = pd.concat([g.assign(subject=s) for s, g, _, _ in rows], ignore_index=True)
    grp = long.groupby("bin")["err"].agg(["mean", "sem"]).reset_index()
    col = ROI_COLOUR[roi]
    ax.plot(grp["bin"] + 1, grp["mean"], color=col, lw=2.0, zorder=3, label="Group mean")
    ax.fill_between(grp["bin"] + 1, grp["mean"] - grp["sem"], grp["mean"] + grp["sem"],
                    color=col, alpha=0.22, lw=0, zorder=2)
    rhos = [r for _, _, r, _ in rows]
    rho_mean = float(np.nanmean(rhos))
    n_pos = int(np.sum(np.array(rhos) > 0))
    ax.set_xlabel("Posterior-SD quintile  (1=lowest)")
    ax.set_ylabel(f"Mean |error|  ({unit})")
    ax.set_xticks(range(1, n_bins + 1))
    ax.set_title(f"{label} · {roi}", fontsize=10, color="0.2")
    ax.set_ylim(bottom=0)
    ax.text(0.04, 0.96,
            fr"$\overline{{\rho}}_{{Sp}}$ = {rho_mean:+.2f}" + "\n"
            f"+ρ in {n_pos}/{len(rhos)} subs",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.5))


def run(subjects, nvoxels, smoothed, noise, n_bins, out):
    data = collect(subjects, nvoxels, smoothed, noise, n_bins=n_bins)
    out.parent.mkdir(parents=True, exist_ok=True)
    smooth_lbl = "smoothed" if smoothed else "unsmoothed"
    with PdfPages(out) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), constrained_layout=True)
        for ax, (q, roi, label, unit, _) in zip(axes.ravel(), PANELS):
            _panel(ax, q, roi, label, unit, data[(q, roi)], n_bins)
        fig.suptitle("Trial-level calibration: posterior SD vs empirical |error|  "
                     f"·  {smooth_lbl}  ·  nvoxels={nvoxels}  ·  noise={noise}",
                     fontsize=11, y=1.04)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    # sidecar: per-subject Spearman correlations
    rows = []
    for (q, roi), entries in data.items():
        for s, _, rho, n in entries:
            rows.append(dict(quantity=q, roi=roi, subject=s, rho=rho, n=n))
    pd.DataFrame(rows).to_csv(out.with_suffix(".tsv"), sep="\t", index=False)
    print(f"Wrote {out}\nSidecar: {out.with_suffix('.tsv')}")
    # print group summary
    for (q, roi), entries in data.items():
        rhos = [r for _, _, r, _ in entries]
        if rhos:
            n_pos = int(np.sum(np.array(rhos) > 0))
            print(f"  {q:>6s} · {roi:<9s}  mean ρ = {np.nanmean(rhos):+.3f}  "
                  f"(+ρ in {n_pos}/{len(rhos)} subs)")


def discover(nvoxels, smoothed, noise):
    smooth = "_smoothed" if smoothed else ""
    subs = set()
    for q, roi, *_ in PANELS:
        for pat in (f"sub-*/func/*_mask-{roi}_nvoxels-{nvoxels}_noise-{noise}{smooth}_pars.tsv",
                    f"sub-*/func/*_mask-{roi}_nvoxels-{nvoxels}_noise-{noise}{smooth}_lambda-*_pars.tsv"):
            for p in (DECODE / q).glob(pat):
                subs.add(p.name.split("_")[0].removeprefix("sub-"))
    return sorted(subs, key=lambda s: (0 if s[0].isdigit() else 1, s))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--nvoxels", default="100")
    p.add_argument("--noise", default="full", choices=["full", "spherical", "geodesic"])
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--n-bins", type=int, default=5)
    p.add_argument("--out", default=str(BIDS / "derivatives" / "qa"
                                        / "decoding_uncertainty_vs_error.pdf"))
    args = p.parse_args()
    subjects = args.subjects or discover(args.nvoxels, args.smoothed, args.noise)
    if not subjects:
        raise SystemExit(f"No subjects found for nvoxels={args.nvoxels} noise={args.noise}")
    print(f"Subjects ({len(subjects)}): {subjects}")
    run(subjects, args.nvoxels, args.smoothed, args.noise, args.n_bins, Path(args.out))


if __name__ == "__main__":
    main()
