"""Decoding accuracy, error and posterior uncertainty per ROI and mapping.

Reads the per-trial posteriors the decoders write
(``derivatives/decoding/{value,gabor}/sub-*/func/sub-*_mask-<ROI>_*_pars.tsv``,
one row per trial, one column per stimulus-grid point) and reduces each trial
to three numbers: the posterior's point estimate, its spread, and the error
against the true stimulus. Value is a linear quantity; orientation is axial
(period pi), so its mean, spread and error are circular on 2*theta.

Everything is then split by **mapping** — ``cdf`` vs ``inverse_cdf``, the two
orientation-to-value assignments, which alternate across sessions by subject
parity (``utils.data.Subject.get_mapping``). That split is the point: the
stimulus sequence, the scanner and the subject are the same, so a difference
between the two conditions is about the mapping rather than about signal
quality.

Three things are reported per ROI, condition and stimulus space:

    accuracy    correlation between true and decoded (circular for orientation)
    |error|     mean absolute error, in CHF or degrees
    uncertainty mean posterior SD, in the same units

plus the within-subject correlation between uncertainty and |error|, which is
the check that the posterior width means anything at all: a well-calibrated
decoder is more wrong on the trials where it says it is less sure.

Whatever ROIs are on disk get picked up, so this widens by itself as more
decoding runs land.

Usage
-----
    python -m abstract_values.visualize.decoding_by_roi_condition
    python -m abstract_values.visualize.decoding_by_roi_condition \\
        --setting nvoxels-100_noise-spherical_lambda-0.1
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from abstract_values.utils.data import BIDS_FOLDER

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})

CONDITION_COLOUR = {"cdf": "#3B5BA5", "inverse_cdf": "#C44E52"}
SPACES = {"value": ("value", "CHF"), "gabor": ("orientation", "deg")}


def mapping_for(subject, session):
    """cdf / inverse_cdf, mirroring utils.data.Subject.get_mapping."""
    num = int("".join(c for c in str(subject) if c.isdigit()))
    if num % 2 == 0:
        return "cdf" if session == 1 else "inverse_cdf"
    return "inverse_cdf" if session == 1 else "cdf"


def summarise_posteriors(df, circular):
    """Point estimate, spread and signed error per trial."""
    meta = [c for c in df.columns if not re.fullmatch(r"-?\d+\.?\d*(e-?\d+)?", c)]
    grid = np.array([float(c) for c in df.columns if c not in meta])
    p = df[[c for c in df.columns if c not in meta]].to_numpy(float)
    p = np.clip(p, 0, None)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-30)
    true = df[[c for c in meta if c.startswith("true_")][0]].to_numpy(float)

    if circular:
        # Axial: everything lives on 2*theta, so a resultant vector there is
        # the right mean and its length the right concentration.
        z = (p * np.exp(1j * 2 * grid[None, :])).sum(axis=1)
        est = (np.angle(z) / 2) % np.pi
        R = np.clip(np.abs(z), 1e-12, 1 - 1e-12)
        sd = np.sqrt(-2 * np.log(R)) / 2                      # radians
        err = (est - true + np.pi / 2) % np.pi - np.pi / 2     # wrapped
        return np.degrees(est), np.degrees(sd), np.degrees(err), np.degrees(true)
    est = (p * grid[None, :]).sum(axis=1)
    sd = np.sqrt((p * (grid[None, :] - est[:, None]) ** 2).sum(axis=1))
    return est, sd, est - true, true


def accuracy(true, est, circular):
    """Pearson r, or the circular-circular correlation for orientation."""
    if len(true) < 5:
        return np.nan
    if circular:
        a, b = np.radians(true) * 2, np.radians(est) * 2
        a = a - np.angle((np.exp(1j * a)).mean())
        b = b - np.angle((np.exp(1j * b)).mean())
        num = (np.sin(a) * np.sin(b)).sum()
        den = np.sqrt((np.sin(a) ** 2).sum() * (np.sin(b) ** 2).sum())
        return float(num / den) if den else np.nan
    return float(np.corrcoef(true, est)[0, 1])


def collect(deriv, setting, spaces=("value", "gabor")):
    rows = []
    for space in spaces:
        root = deriv / "decoding" / space
        if not root.exists():
            continue
        for fn in sorted(root.glob(f"sub-*/func/sub-*_{setting}_pars.tsv")):
            m = re.search(r"sub-([^_]+)_mask-([^_]+)_", fn.name)
            if not m:
                continue
            subject, roi = m.groups()
            df = pd.read_csv(fn, sep="\t")
            if "session" not in df.columns or df.empty:
                continue
            circular = space == "gabor"
            est, sd, err, true = summarise_posteriors(df, circular)
            out = pd.DataFrame(dict(session=df.session, est=est, sd=sd,
                                    err=err, true=true))
            out["condition"] = [mapping_for(subject, s) for s in out.session]
            for cond, d in out.groupby("condition"):
                ok = np.isfinite(d.est) & np.isfinite(d.sd)
                if ok.sum() < 20:
                    continue
                d = d[ok]
                true_sd = float(np.std(d.true)) if not circular else np.nan
                # The two mappings do NOT present the same value distribution
                # (SD 10.4 CHF under cdf vs 12.3 under inverse_cdf, same mean),
                # so a raw MAE difference between them is partly mechanical:
                # more spread-out stimuli produce larger absolute errors at the
                # same correlation. Divide it out before comparing conditions.
                slope = (np.polyfit(d.true, d.est, 1)[0]
                         if not circular and len(d) > 5 else np.nan)
                rows.append(dict(
                    space=space, roi=roi, subject=subject, condition=cond,
                    n_trials=len(d),
                    accuracy=accuracy(d.true.to_numpy(), d.est.to_numpy(), circular),
                    abs_error=float(np.abs(d.err).mean()),
                    uncertainty=float(d.sd.mean()),
                    true_sd=true_sd,
                    norm_abs_error=float(np.abs(d.err).mean()) / true_sd
                    if true_sd and np.isfinite(true_sd) else np.nan,
                    norm_uncertainty=float(d.sd.mean()) / true_sd
                    if true_sd and np.isfinite(true_sd) else np.nan,
                    bias=float(np.mean(d.err)),
                    slope=slope,
                    unc_err_r=float(np.corrcoef(d.sd, np.abs(d.err))[0, 1])
                    if len(d) > 5 else np.nan))
    return pd.DataFrame(rows)


def panel(ax, df, column, rois, ylabel, chance=None):
    x = np.arange(len(rois))
    for cond, dx in (("cdf", -0.15), ("inverse_cdf", 0.15)):
        d = df[df.condition == cond]
        piv = d.pivot_table(index="subject", columns="roi", values=column)
        mean = [piv[r].mean() if r in piv else np.nan for r in rois]
        sem = [piv[r].std(ddof=1) / np.sqrt(piv[r].notna().sum())
               if r in piv else np.nan for r in rois]
        ax.errorbar(x + dx, mean, yerr=sem, fmt="o", ms=3.4, lw=0,
                    elinewidth=0.9, color=CONDITION_COLOUR[cond], zorder=3)
    if chance is not None:
        ax.axhline(chance, color="0.7", lw=0.6, ls=(0, (4, 3)), zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, len(rois) - 0.4)
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 4))


def figure(df, space, out_pdf):
    label, unit = SPACES[space]
    d = df[df.space == space]
    rois = sorted(d.roi.unique(),
                  key=lambda r: -d[d.roi == r].accuracy.mean())
    fig, axes = plt.subplots(3, 1, figsize=(max(5.0, 0.55 * len(rois) + 2.2), 6.0),
                             sharex=True)
    panel(axes[0], d, "accuracy", rois, f"Decoding accuracy\n(r, true vs decoded)",
          chance=0)
    err_col = "norm_abs_error" if space == "value" else "abs_error"
    err_lab = ("|Error| / SD of true value" if space == "value"
               else f"|Error| ({unit})")
    panel(axes[1], d, err_col, rois, err_lab)
    # Same normalisation as the error panel, for the same reason: the two
    # mappings present value distributions of different width, so a posterior
    # SD in CHF is not comparable across them.
    unc_col = "norm_uncertainty" if space == "value" else "uncertainty"
    unc_lab = ("Posterior SD / SD of true value" if space == "value"
               else f"Posterior SD ({unit})")
    panel(axes[2], d, unc_col, rois, unc_lab)
    for cond, dy in (("cdf", 0.96), ("inverse_cdf", 0.86)):
        axes[0].text(0.985, dy, cond, transform=axes[0].transAxes, ha="right",
                     va="top", fontsize=7, color=CONDITION_COLOUR[cond],
                     fontweight="bold")
    for ax, letter in zip(axes, "abc"):
        ax.text(-0.02, 1.03, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="right")
    fig.suptitle(f"{label.capitalize()} decoding per ROI, by mapping — "
                 f"n={d.subject.nunique()}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_pdf.savefig(fig)
    plt.close(fig)


def report(df):
    for space, d in df.groupby("space"):
        label, unit = SPACES[space]
        print(f"\n=== {label} ===")
        print(f"  {'ROI':10s} {'cond':12s} {'acc':>7s} {'|err|':>8s} "
              f"{'post SD':>8s} {'unc~|err| r':>11s}  n")
        for (roi, cond), g in d.groupby(["roi", "condition"]):
            print(f"  {roi:10s} {cond:12s} {g.accuracy.mean():7.3f} "
                  f"{g.abs_error.mean():8.2f} {g.uncertainty.mean():8.2f} "
                  f"{g.unc_err_r.mean():11.3f}  {g.subject.nunique()}")
        for roi, g in d.groupby("roi"):
            piv = g.pivot_table(index="subject", columns="condition",
                                values=["accuracy", "abs_error", "uncertainty",
                                        "norm_abs_error", "norm_uncertainty",
                                        "slope", "bias", "true_sd"])
            if piv.shape[1] < 6:
                continue
            print(f"  --- {roi}: cdf vs inverse_cdf (paired, "
                  f"n={len(piv.dropna())})")
            for metric in ("accuracy", "abs_error", "norm_abs_error",
                           "uncertainty", "norm_uncertainty", "slope",
                           "bias", "true_sd"):
                # Orientation has no normalised columns (its stimulus set is
                # the same under both mappings), so they drop out of the pivot.
                if (metric, "cdf") not in piv.columns:
                    continue
                a = piv[(metric, "cdf")]
                b = piv[(metric, "inverse_cdf")]
                ok = a.notna() & b.notna()
                if ok.sum() < 5:
                    continue
                t = stats.ttest_rel(a[ok], b[ok])
                print(f"      {metric:12s} {a[ok].mean():7.3f} vs "
                      f"{b[ok].mean():7.3f}  diff {a[ok].mean() - b[ok].mean():+7.3f}"
                      f"  t={t.statistic:+5.2f} p={t.pvalue:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--setting", default="nvoxels-100_noise-spherical_lambda-0.1",
                   help="Filename tail identifying the decoding run to read "
                        "(default: the setting that exists for every subject).")
    p.add_argument("--out", default="notes/figures/decoding_by_roi_condition.pdf")
    args = p.parse_args()

    deriv = Path(args.bids_folder) / "derivatives"
    df = collect(deriv, args.setting)
    if df.empty:
        raise SystemExit(f"No decoding output matching *_{args.setting}_pars.tsv")
    report(df)

    from matplotlib.backends.backend_pdf import PdfPages
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out)) as pdf:
        for space in ("value", "gabor"):
            if (df.space == space).any():
                figure(df, space, pdf)
    df.to_csv(out.with_suffix(".tsv"), sep="\t", index=False)
    print(f"\nWrote {out}\nWrote {out.with_suffix('.tsv')}")


if __name__ == "__main__":
    main()
