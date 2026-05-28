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
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr

from abstract_values.utils.data import BIDS_FOLDER

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelpad": 4,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2, "lines.markersize": 4,
    "legend.frameon": False, "legend.handlelength": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
sns.set_context("paper")

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


def collect(subjects, nvoxels, smoothed, noise, n_bins=5, control_stim=False):
    """For each panel and each subject: per-bin (rank-binned post_SD) mean |err|
    and Spearman ρ(post_SD, |err|). Returns a dict {(q,roi): list of (subject,
    bin_means_df, rho, n_trials)}.

    ``control_stim``: subtract the per-(subject, true-stim) mean from both SD
    and |err| before binning/correlating. Tests whether the SD↔|err| coupling
    holds *within* each stimulus level — i.e., over and above the obvious
    effect that some stimuli are intrinsically harder."""
    out = {(q, roi): [] for q, roi, *_ in PANELS}
    for q, roi, _, unit, circular in PANELS:
        for s in subjects:
            p = _pars_path(q, s, roi, nvoxels, smoothed, noise)
            if p is None:
                continue
            true, _, sd, abs_err = _per_trial(p, circular)
            keep = np.isfinite(sd) & np.isfinite(abs_err)
            true, sd, abs_err = true[keep], sd[keep], abs_err[keep]
            if circular:
                sd, abs_err = np.rad2deg(sd), np.rad2deg(abs_err)
            if control_stim:
                # Residualise within each true-stim bin (the 23-level grid).
                tdf = pd.DataFrame({"t": np.round(true, 4), "sd": sd, "err": abs_err})
                sd      = (tdf["sd"]  - tdf.groupby("t")["sd"].transform("mean")).values
                abs_err = (tdf["err"] - tdf.groupby("t")["err"].transform("mean")).values
            if len(sd) < n_bins * 4:
                continue
            ranks = pd.Series(sd).rank(method="first") - 1
            bins = np.minimum((ranks / len(sd) * n_bins).astype(int), n_bins - 1)
            g = pd.DataFrame({"bin": bins, "sd": sd, "abs_err": abs_err}) \
                  .groupby("bin").agg(sd=("sd", "mean"), err=("abs_err", "mean")).reset_index()
            rho, _ = spearmanr(sd, abs_err)
            out[(q, roi)].append((s, g, float(rho), int(len(sd))))
    return out


def _panel(ax, q, roi, label, unit, rows, n_bins, control_stim):
    """Per-subject quintile lines + group mean ± SEM."""
    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="0.55", fontsize=9)
        ax.set_axis_off()
        return None
    col = ROI_COLOUR[roi]
    # per-subject lines: pale, narrow, low alpha — the data swarm
    for _, g, _, _ in rows:
        ax.plot(g["bin"] + 1, g["err"], color="0.72", lw=0.5, alpha=0.7, zorder=1)
    # group mean ± SEM across subjects, per bin
    long = pd.concat([g.assign(subject=s) for s, g, _, _ in rows], ignore_index=True)
    grp = long.groupby("bin")["err"].agg(["mean", "sem"]).reset_index()
    ax.plot(grp["bin"] + 1, grp["mean"], color=col, lw=2.0, zorder=3)
    ax.fill_between(grp["bin"] + 1, grp["mean"] - grp["sem"], grp["mean"] + grp["sem"],
                    color=col, alpha=0.25, lw=0, zorder=2)
    # reference line at 0 under residualisation (negative residuals are the
    # informative half — low-SD trials err less than the per-stim mean)
    if control_stim or (long["err"] < 0).any():
        ax.axhline(0, color="0.65", lw=0.5, ls="--", zorder=0)
    rhos = np.array([r for _, _, r, _ in rows])
    rho_mean = float(np.nanmean(rhos))
    n_pos = int(np.sum(rhos > 0))
    ax.set_xticks(range(1, n_bins + 1))
    ax.set_xlabel("Posterior-SD quintile")
    ylab = ("Δ|Error|" if control_stim else "Mean |error|") + f" ({unit})"
    ax.set_ylabel(ylab)
    # ρ annotation: matched colour, no box, upper-left
    ax.text(0.04, 0.96,
            f"ρ̄ = {rho_mean:+.2f}\n{n_pos}/{len(rhos)} +",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=col, linespacing=1.2)
    return rho_mean


def _add_panel_letter(ax, letter):
    ax.text(-0.22, 1.04, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="right")


def run(subjects, variants, smoothed, n_bins, control_stim, out):
    """variants: list of (nvoxels, noise) tuples → one PDF page each."""
    out.parent.mkdir(parents=True, exist_ok=True)
    ctrl_note = ("Mean ± SEM across subjects; subject lines in light gray. "
                 "Within-stimulus residualisation: per (subject, true-stim) means "
                 "subtracted from both SD and |error| before quintile-binning, so "
                 "the relationship is tested *over and above* stimulus difficulty.") \
        if control_stim else \
        ("Mean ± SEM across subjects; subject lines in light gray.")
    all_rows = []
    with PdfPages(out) as pdf:
        for nvoxels, noise in variants:
            data = collect(subjects, nvoxels, smoothed, noise,
                            n_bins=n_bins, control_stim=control_stim)
            # sharey='row' so V1 and NPCr of the same quantity sit on the same scale
            fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.0),
                                     sharex=True, sharey="row",
                                     constrained_layout=True)
            for ax, letter, (q, roi, label, unit, _) in zip(
                    axes.ravel(), "ABCD", PANELS):
                rho = _panel(ax, q, roi, label, unit, data[(q, roi)],
                             n_bins, control_stim)
                _add_panel_letter(ax, letter)
            # direct-label the ROI as small text inside each panel (top-right),
            # so panel titles aren't required and the eye lands on the data
            for ax, (q, roi, label, _, _) in zip(axes.ravel(), PANELS):
                ax.text(0.97, 0.96, f"{label} · {roi}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=8.5, color="0.25")
            # (No "higher SD -> larger error" arrow — the upward slope + the
            # axhline-at-0 + the ρ̄ annotation already tell the reader the
            # story; an extra arrow would collide with the in-panel ROI
            # label and add ink without adding information.)
            # compact suptitle: only the variant identity
            fig.suptitle(f"nvoxels = {nvoxels}  ·  noise = {noise}",
                         fontsize=11, y=1.02)
            sns.despine(fig=fig, offset=5, trim=True)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"\n=== nvoxels={nvoxels}  noise={noise} ===")
            for (q, roi), entries in data.items():
                rhos = [r for _, _, r, _ in entries]
                if rhos:
                    n_pos = int(np.sum(np.array(rhos) > 0))
                    print(f"  {q:>6s} · {roi:<9s}  n_subs={len(rhos):2d}  "
                          f"mean ρ = {np.nanmean(rhos):+.3f}  "
                          f"(+ρ in {n_pos}/{len(rhos)})")
                for s, _, rho, n in entries:
                    all_rows.append(dict(nvoxels=nvoxels, noise=noise,
                                          quantity=q, roi=roi,
                                          subject=s, rho=rho, n=n,
                                          control_stim=control_stim))
        # footer note on residualisation (caption-level info as a small
        # in-PDF page; readers skimming panels still get the message via
        # the in-panel annotation + axis labels)
        # (Skip a separate caption page — keep the PDF terse; the suptitle
        # +  Δ|Error| ylab already say what's residualised.)
    pd.DataFrame(all_rows).to_csv(out.with_suffix(".tsv"), sep="\t", index=False)
    print(f"\nWrote {out}\nSidecar: {out.with_suffix('.tsv')}")


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
    p.add_argument("--variants", nargs="+",
                   default=["100:full", "100:spherical", "fdr05:full", "fdr05:spherical"],
                   help="list of nvoxels:noise tokens — one PDF page each")
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--n-bins", type=int, default=5)
    p.add_argument("--no-control-stim", action="store_true",
                   help="don't residualise within each true-stim level (raw analysis)")
    p.add_argument("--out", default=str(BIDS / "derivatives" / "qa"
                                        / "decoding_uncertainty_vs_error.pdf"))
    args = p.parse_args()
    variants = [tuple(v.split(":", 1)) for v in args.variants]
    # union of subjects across variants
    if args.subjects:
        subjects = args.subjects
    else:
        sset = set()
        for nv, noise in variants:
            sset.update(discover(nv, args.smoothed, noise))
        subjects = sorted(sset, key=lambda s: (0 if s[0].isdigit() else 1, s))
    if not subjects:
        raise SystemExit("No subjects with any of the requested variants.")
    print(f"Subjects ({len(subjects)}): {subjects}")
    print(f"Variants: {variants}")
    run(subjects, variants, args.smoothed, args.n_bins,
        not args.no_control_stim, Path(args.out))


if __name__ == "__main__":
    main()
