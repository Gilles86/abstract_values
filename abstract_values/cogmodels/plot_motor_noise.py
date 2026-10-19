"""How big is the fitted motor noise?

From a fit with heavy-tailed motor noise (--fit-motor-noise --motor-tail), the
per-participant motor kernel in CHF,
    k(d) ~ (1 + eta d^2 / sigma^2) ** (-1 / (2 eta)),
against the observed bids at 90 deg -- where the cardinal category gate leaves
motor noise as the only source of spread -- and against the bid spread overall.

  a  the kernel, linear scale: median participant, 10-90% of participants, the
     Gaussian with the same sigma, and the observed spread at 90 deg
  b  the same on a log scale, out to the tails
  c  per participant: the share of bids motor noise alone puts > 3 CHF off,
     against the share of the participant's bids that are > 3 CHF off their
     cell median

    python -m abstract_values.cogmodels.plot_motor_noise --trace <..._mtail_trace.nc>
    -> notes/figures/fig_motor_noise_<tag>.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abstract_values.visualize.plot_vmpfc_dropout import style

REPO = Path(__file__).resolve().parents[2]
COL = "#C2560A"


def kernel(d, sigma, eta):
    k = np.exp(-np.log1p(eta * (d / sigma) ** 2) / (2 * eta))
    return k / np.trapz(k, d)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True)
    p.add_argument("--paradigm-tsv", default=str(REPO / "notes/data/efficient_coding_paradigm_n35.tsv"))
    a = p.parse_args()
    post = az.from_netcdf(a.trace).posterior
    sig = post.sigma_motor.mean(("chain", "draw")).to_pandas()
    eta = post.motor_tail.mean(("chain", "draw")).to_pandas()
    sig.index = sig.index.astype(int); eta.index = eta.index.astype(int)

    d = pd.read_csv(a.paradigm_tsv, sep="\t").dropna(subset=["response"])
    d["res"] = d.response - d.groupby(["subject", "mapping", "orientation"]).response.transform("median")
    obs90 = d[d.orientation == 90].res
    big = d.assign(b=d.res.abs() > 3).groupby("subject").b.mean()

    x = np.linspace(-20, 20, 8001)
    K = np.stack([kernel(x, sig[s], eta[s]) for s in sig.index])
    lo, med, hi = np.percentile(K, [10, 50, 90], axis=0)
    s_med = float(np.median(sig))
    gauss = np.exp(-0.5 * (x / s_med) ** 2); gauss /= np.trapz(gauss, x)
    edges = np.arange(-20.25, 20.26, 0.5)
    h, _ = np.histogram(obs90.clip(-20, 20), edges)
    h = h / (len(obs90) * 0.5)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    tail = pd.Series({s: float(np.trapz(np.where(np.abs(x) > 3, K[i], 0), x))
                      for i, s in enumerate(sig.index)})
    hwhm = sig * np.sqrt((2 ** (2 * eta) - 1) / eta)

    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.5), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1, 0.85]))
    for ax, log in zip(axes[:2], (False, True)):
        ax.fill_between(x, lo, hi, color=COL, alpha=0.25, lw=0)
        ax.plot(x, med, color=COL, lw=1.4)
        ax.plot(x, gauss, color="0.35", lw=0.9, ls=(0, (3, 2)))
        ax.plot(ctr, np.where(h > 0, h, np.nan), "o", ms=2.6, color="#5B4A9E", mec="white",
                mew=0.3, zorder=3)
        ax.set_xlabel("Bid − intended bid (CHF)")
        if log:
            ax.set_yscale("log"); ax.set_ylim(1e-4, 5); ax.set_xlim(-12, 12)
            ax.set_xticks([-12, -6, -3, 0, 3, 6, 12])
        else:
            ax.set_xlim(-3, 3); ax.set_xticks([-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3])
            ax.set_xticklabels(["−3", "−2", "−1", "", "0", "", "1", "2", "3"])
            ax.set_ylabel("Density (per CHF)")
            tf = ax.transAxes
            ents = (("Motor noise, median participant", "line", COL, "-"),
                    ("10–90% of participants", "band", COL, None),
                    ("Gaussian, same σ", "line", "0.35", (0, (3, 2))),
                    ("Observed bids at 90°", "marker", "#5B4A9E", None))
            for i, (lab, kind, c, ls) in enumerate(ents):
                yy = 0.95 - i * 0.085
                if kind == "band":
                    ax.add_patch(plt.Rectangle((0.03, yy - 0.022), 0.08, 0.044, transform=tf,
                                               color=c, alpha=0.25, lw=0))
                elif kind == "marker":
                    ax.plot(0.07, yy, "o", ms=2.6, transform=tf, color=c, mec="white", mew=0.3)
                else:
                    ax.plot([0.03, 0.11], [yy, yy], transform=tf, color=c, ls=ls, lw=1.2)
                ax.text(0.14, yy, lab, transform=tf, fontsize=6, va="center", color="0.25")
            ax.text(0.97, 0.5, f"Half-width at\nhalf maximum\n{np.median(hwhm):.2f} CHF\n"
                    f"(range {hwhm.min():.2f}–{hwhm.max():.2f})", transform=tf, ha="right",
                    fontsize=6, color="0.3")
        sns.despine(ax=ax, offset=3)
    ax = axes[2]
    ax.plot(100 * big[tail.index], 100 * tail, "o", ms=3.2, color=COL, mec="white", mew=0.3)
    m = max(100 * big.max(), 100 * tail.max()) * 1.05
    ax.plot([0, m], [0, m], color="0.6", lw=0.6, ls=(0, (3, 2)))
    ax.set_xlim(0, m); ax.set_ylim(0, m)
    ax.set_xlabel("Bids > 3 CHF off, observed (%)")
    ax.set_ylabel("From motor noise alone (%)")
    ax.text(0.05, 0.95, f"Motor share of misses:\nmedian {np.median(tail / big[tail.index]) * 100:.0f}%",
            transform=ax.transAxes, va="top", fontsize=6, color="0.3")
    sns.despine(ax=ax, offset=3)
    for ax, l in zip(axes, "abc"):
        ax.text(-0.2, 1.06, l, transform=ax.transAxes, fontsize=9, fontweight="bold")
    tag = Path(a.trace).name.replace("efficient_coding_", "").replace("_trace.nc", "")
    out = REPO / f"notes/figures/fig_motor_noise_{tag}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"sigma_motor median {np.median(sig):.2f} [{sig.min():.2f}, {sig.max():.2f}]; "
          f"eta median {np.median(eta):.2f} [{eta.min():.2f}, {eta.max():.2f}]; "
          f"HWHM median {np.median(hwhm):.2f}; P(|motor|>3) median {100*np.median(tail):.2f}% "
          f"[{100*tail.min():.2f}, {100*tail.max():.2f}]; observed >3 median {100*big.median():.1f}%; "
          f"observed SD at 90 deg {obs90.std():.2f}, MAD-SD {1.4826*np.median(np.abs(obs90)):.2f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
