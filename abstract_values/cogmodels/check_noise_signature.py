"""Where does the bid noise live -- orientation space or value space?

The two efficient-coding stages are only separable if the data say something
about WHERE the noise enters.  This is the model-free version of that test.

Perceptual (orientation-space) noise reaches the bid through the mapping
slope: SD_response ~ |G'(theta)| * SD_theta.  Valuation (value-space) noise
does not depend on the mapping at all.  Because this study ran BOTH mappings
in the same subjects, the same orientation has two different |G'| values, so
the two accounts make opposite predictions for the cross-mapping SD ratio.

Writes notes/figures/noise_signature.pdf.
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

from bauer.efficient_coding import MAPPING_ORIENTATIONS_DEG as ORI, MAPPING_VALUES as G

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "legend.frameon": False,
    "pdf.fonttype": 42, "figure.dpi": 150, "savefig.dpi": 300,
})

COND_C = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}
COND_L = {"cdf": "CDF", "inverse_cdf": "Inverse CDF"}
CARDINAL = 90.0


def summarise(paradigm_tsv):
    d = pd.read_csv(paradigm_tsv, sep="\t")
    d["ori"] = d["orientation"].round(1)
    out = []
    for m in ("cdf", "inverse_cdf"):
        slope = np.abs(np.gradient(G[m], ORI))
        s = d[d.mapping == m].groupby("ori")["response"].agg(sd="std", n="count")
        s["abs_gprime"] = np.interp(s.index, ORI, slope)
        s["value"] = np.interp(s.index, ORI, G[m])
        s["implied_ori_sd"] = s["sd"] / s["abs_gprime"]
        s["mapping"] = m
        out.append(s.reset_index())
    return pd.concat(out, ignore_index=True)


def report(t):
    print(f"{'mapping':<13}{'corr(SD, |G|)':>16}{'excl. 90 deg':>15}")
    for m, g in t.groupby("mapping"):
        r_all = np.corrcoef(g["sd"], g["abs_gprime"])[0, 1]
        gx = g[g["ori"] != CARDINAL]
        r_ex = np.corrcoef(gx["sd"], gx["abs_gprime"])[0, 1]
        print(f"{COND_L[m]:<13}{r_all:>16.3f}{r_ex:>15.3f}")
    w = t.pivot(index="ori", columns="mapping", values=["sd", "abs_gprime"])
    ratio_sd = w["sd"]["cdf"] / w["sd"]["inverse_cdf"]
    ratio_gp = w["abs_gprime"]["cdf"] / w["abs_gprime"]["inverse_cdf"]
    keep = ratio_sd.index != CARDINAL
    print(f"\ncorr(SD ratio, |G'| ratio) across mappings: "
          f"{np.corrcoef(ratio_sd, ratio_gp)[0,1]:.3f} "
          f"(excl. 90 deg: {np.corrcoef(ratio_sd[keep], ratio_gp[keep])[0,1]:.3f})")
    c = t[t.ori == CARDINAL].set_index("mapping")
    o = t[t.ori != CARDINAL].groupby("mapping")["sd"].mean()
    print(f"\n90 deg: SD = {c.loc['cdf','sd']:.2f} / {c.loc['inverse_cdf','sd']:.2f} CHF "
          f"(CDF / inverse CDF) vs {o['cdf']:.2f} / {o['inverse_cdf']:.2f} elsewhere; "
          f"|G'| there differs {c.loc['cdf','abs_gprime']/c.loc['inverse_cdf','abs_gprime']:.1f}x, "
          f"so an orientation-space account needs SD_theta = "
          f"{c.loc['cdf','implied_ori_sd']:.1f} vs {c.loc['inverse_cdf','implied_ori_sd']:.1f} deg "
          f"for the same subjects at the same orientation.")


def figure(t, out):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), constrained_layout=True)
    for m, g in t.groupby("mapping"):
        g = g.sort_values("ori")
        axes[0].plot(g.ori, g.sd, "o-", ms=3, color=COND_C[m], label=COND_L[m])
        axes[1].plot(g.ori, g.abs_gprime, "o-", ms=3, color=COND_C[m], label=COND_L[m])
        axes[2].plot(g.ori, g.implied_ori_sd, "o-", ms=3, color=COND_C[m], label=COND_L[m])
    for ax, ylab, title in zip(
            axes,
            ["Response SD (CHF)", "|G'| (CHF/deg)", "Implied orientation SD (deg)"],
            ["Bid noise", "Mapping slope", "Noise read as perceptual"]):
        ax.axvline(CARDINAL, color="0.6", lw=0.8, ls=":", zorder=0)
        ax.set_xlabel("Orientation (deg)"); ax.set_ylabel(ylab); ax.set_title(title)
        ax.set_xticks([0, 45, 90, 135, 180])
    axes[2].set_yscale("log")
    axes[0].legend(title="Mapping")
    axes[0].annotate("90 deg", (CARDINAL, 0.55), xytext=(100, 0.9), fontsize=8,
                     arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nWrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paradigm-tsv", default="notes/data/efficient_coding_paradigm.tsv")
    p.add_argument("--out", default="notes/figures/noise_signature.pdf")
    a = p.parse_args()
    t = summarise(a.paradigm_tsv)
    report(t)
    figure(t, a.out)


if __name__ == "__main__":
    main()
