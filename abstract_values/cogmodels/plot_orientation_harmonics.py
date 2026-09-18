"""Orientation precision is structured the same way in behaviour and in V1.

Both the bids and the V1 encoding model allocate precision unevenly across
orientation, and in both the structure decomposes into the same two harmonics:

    cos 4θ   cardinal vs oblique   -- the classic cardinal prior
    cos 2θ   horizontal vs vertical -- an asymmetry the paper's 1 - w|sin 2θ|
                                      prior is symmetric about 45 deg and so
                                      cannot express at all

That second term is ~80% the size of the first, in both domains, which is the
empirical argument for the Fourier prior: a_2 carries the cardinal term, a_1 the
horizontal-vertical one.

Writes notes/figures/orientation_harmonics.pdf.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
from bauer.efficient_coding import MAPPING_ORIENTATIONS_DEG as ORI, MAPPING_VALUES as G

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})

BEHAV, BRAIN = "#C44E52", "#3B5BA5"
TICKS = [0, 45, 90, 135, 180]
SLOPE_FLOOR = 0.15          # CHF/deg; below this, bid error / |G'| explodes

# Artefact floor for the BEHAVIOURAL amplitudes. Converting bid error into
# orientation units divides by |G'|, and |G'| itself has harmonic structure --
# so a subject whose error is purely value-stage still produces non-zero
# amplitudes. Measured by simulating with perception switched off (kappa = 1e4,
# sigma_rep = 1.5) and running this exact pipeline; insensitive to kappa
# (kappa = 30 gives 0.177 / 0.175). The brain side has no such division, so its
# floor is zero.
NULL_FLOOR = {"cos 2θ": 0.215, "cos 4θ": 0.170}


def harmonic_fit(theta_deg, y):
    """Least-squares fit  y = a0 + [2θ terms] + [4θ terms].

    Returns the fitted curve on a dense grid plus each component separately, so
    the plot can show what the two coefficients actually describe rather than
    just their amplitudes.
    """
    th = np.deg2rad(theta_deg)
    X = np.column_stack([np.ones(len(th)), np.cos(2*th), np.sin(2*th),
                         np.cos(4*th), np.sin(4*th)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    g = np.linspace(0, 180, 361)
    gt = np.deg2rad(g)
    c2 = b[1]*np.cos(2*gt) + b[2]*np.sin(2*gt)
    c4 = b[3]*np.cos(4*gt) + b[4]*np.sin(4*gt)
    return g, b[0] + c2 + c4, b[0] + c2, b[0] + c4, b[0]


def harmonics(theta_deg, y):
    th = np.deg2rad(theta_deg)
    X = np.column_stack([np.ones(len(th)), np.cos(2 * th), np.sin(2 * th),
                         np.cos(4 * th), np.sin(4 * th)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.hypot(b[1], b[2]) / abs(b[0]), np.hypot(b[3], b[4]) / abs(b[0])


def behavioural_sd_deg(tsv):
    """Per-subject implied orientation SD in DEGREES (not normalised)."""
    d = pd.read_csv(tsv, sep="\t")
    d["value"] = [np.interp(o, ORI, G[m]) for o, m in zip(d.orientation, d.mapping)]
    d["err"] = d.response - d.value
    sl = {m: np.abs(np.gradient(G[m], ORI)) for m in ("cdf", "inverse_cdf")}
    d["gp"] = [np.interp(o, ORI, sl[m]) for o, m in zip(d.orientation, d.mapping)]
    use = d[(d.gp >= SLOPE_FLOOR) & (d.orientation != 90.0)]
    out = {}
    for s_, g in use.groupby("subject"):
        pr = g.groupby("orientation").apply(lambda x: np.std(x.err.values / x.gp.values))
        pr = pr[(pr > 1e-6) & np.isfinite(pr)]
        if len(pr) >= 10:
            out[s_] = pr
    return out


def brain_sd_deg(tsv):
    """V1 expected decoded orientation SD in DEGREES (sd_E is radians of theta)."""
    e = pd.read_csv(tsv, sep="\t")
    prof = e.groupby(["subject", "orientation"]).sd_E.mean().reset_index()
    return {s_: pd.Series(np.rad2deg(g.sd_E.values), index=g.orientation.values)
            for s_, g in prof.groupby("subject")}


def behavioural_profiles(tsv):
    """Per-subject implied orientation precision, normalised to its own mean."""
    d = pd.read_csv(tsv, sep="\t")
    d["value"] = [np.interp(o, ORI, G[m]) for o, m in zip(d.orientation, d.mapping)]
    d["err"] = d.response - d.value
    sl = {m: np.abs(np.gradient(G[m], ORI)) for m in ("cdf", "inverse_cdf")}
    d["gp"] = [np.interp(o, ORI, sl[m]) for o, m in zip(d.orientation, d.mapping)]
    use = d[(d.gp >= SLOPE_FLOOR) & (d.orientation != 90.0)]
    out = {}
    for s, g in use.groupby("subject"):
        p = g.groupby("orientation").apply(lambda x: np.std(x.err.values / x.gp.values))
        p = p[(p > 1e-6) & np.isfinite(p)]
        if len(p) >= 10:
            prec = 1.0 / p.values
            out[s] = pd.Series(prec / prec.mean(), index=p.index.values)
    return out


def brain_profiles(tsv):
    e = pd.read_csv(tsv, sep="\t")
    prof = e.groupby(["subject", "orientation"]).sd_E.mean().reset_index()
    out = {}
    for s, g in prof.groupby("subject"):
        prec = 1.0 / g.sd_E.values
        out[s] = pd.Series(prec / prec.mean(), index=g.orientation.values)
    return out


def band(ax, profiles, color, label, marker):
    grid = sorted(set(np.concatenate([p.index.values for p in profiles.values()])))
    M = np.vstack([np.interp(grid, p.index.values, p.values) for p in profiles.values()])
    m, se = M.mean(0), M.std(0) / np.sqrt(len(M))
    ax.fill_between(grid, m - se, m + se, color=color, alpha=0.22, lw=0)
    ax.plot(grid, m, color=color, lw=1.3, marker=marker,
            ms=3.5 if marker else 0, mec="white", mew=0.5)
    return grid, m


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paradigm-tsv", default="notes/data/efficient_coding_paradigm.tsv")
    p.add_argument("--brain-tsv", default="notes/data/expected_decoded_orientation_v1.tsv")
    p.add_argument("--out", default="notes/figures/orientation_harmonics.pdf")
    a = p.parse_args()

    beh_sd, bra_sd = behavioural_sd_deg(a.paradigm_tsv), brain_sd_deg(a.brain_tsv)
    beh, bra = behavioural_profiles(a.paradigm_tsv), brain_profiles(a.brain_tsv)

    def group(profs):
        grid = np.array(sorted(set(np.concatenate([q.index.values for q in profs.values()]))))
        M = np.vstack([np.interp(grid, q.index.values, q.values) for q in profs.values()])
        return grid, M.mean(0), M.std(0) / np.sqrt(len(M)), len(M)

    fig, axg = plt.subplots(2, 2, figsize=(7.25, 5.2), constrained_layout=True)

    # --- a, b: the fitted noise function, and what each harmonic contributes -
    for ax, (profs, col, name, mk) in zip(axg[0], ((beh_sd, BEHAV, "Bids", "o"),
                                                   (bra_sd, BRAIN, "V1 encoding model", None))):
        x, m, se, n = group(profs)
        for c in (0, 90, 180):
            ax.axvline(c, color="0.9", lw=0.7, ls=":", zorder=0)
        ax.fill_between(x, m - se, m + se, color=col, alpha=0.20, lw=0, zorder=1)
        ax.plot(x, m, color=col, lw=0, marker=mk or "o", ms=3.0, alpha=0.55,
                mec="white", mew=0.4, zorder=2)
        g, fit, only2, only4, mean = harmonic_fit(x, m)
        ax.plot(g, fit, color=col, lw=1.8, zorder=4)
        ax.plot(g, only2, color="0.35", lw=1.0, ls=(0, (4, 1.8)), zorder=3)
        ax.plot(g, only4, color="0.35", lw=1.0, ls=(0, (1.2, 1.4)), zorder=3)
        ax.axhline(mean, color="0.75", lw=0.7, zorder=0)
        ax.set_xticks([0, 45, 90, 135, 180]); ax.set_xlim(0, 180)
        ax.set_xlabel("Orientation θ (deg)")
        ax.set_ylabel("Implied orientation noise (deg)")
        ax.set_title(f"{name}  (n = {n})")
        a2, a4 = harmonics(x, m)
        ax.text(0.02, 0.03, f"— fitted   ---- 2θ only   ···· 4θ only\n"
                            f"2θ {a2:.2f}   4θ {a4:.2f}  (of mean)",
                transform=ax.transAxes, fontsize=6, color="0.35", va="bottom")

    # --- c: per-subject amplitudes, against a clear zero and the null floor --
    ax = axg[1, 0]
    amps = {n_: np.array([harmonics(q.index.values, q.values) for q in pr.values()])
            for n_, pr in (("Bids", beh), ("V1", bra))}
    rng = np.random.default_rng(0)
    x0 = {"Bids": 0.0, "V1": 1.5}
    ax.axhline(0, color="0.2", lw=1.1, zorder=5)
    for name, col in (("Bids", BEHAV), ("V1", BRAIN)):
        A = amps[name]
        for j, off in ((0, -0.28), (1, 0.28)):
            xs = x0[name] + off + rng.normal(0, 0.035, len(A))
            ax.plot(xs, A[:, j], "o", ms=3.6, color=col, alpha=0.45, mec="white", mew=0.4)
            ax.hlines(np.median(A[:, j]), x0[name]+off-0.17, x0[name]+off+0.17,
                      color=col, lw=2.4, zorder=4)
            if name == "Bids":
                f = NULL_FLOOR["cos 2θ" if j == 0 else "cos 4θ"]
                ax.hlines(f, x0[name]+off-0.19, x0[name]+off+0.19, color="0.3",
                          lw=1.2, ls=(0, (2.5, 1.6)), zorder=5)
    ax.set_xticks([-0.28, 0.28, 1.22, 1.78])
    ax.set_xticklabels(["2θ", "4θ", "2θ", "4θ"], fontsize=7)
    ax.set_xlim(-0.62, 2.12); ax.set_ylim(bottom=0)
    ax.set_ylabel("Amplitude (fraction of mean)")
    ax.set_title("Both harmonics, both domains")
    ax.text(0, ax.get_ylim()[1]*0.98, "Bids", color=BEHAV, fontsize=7, ha="center", va="top")
    ax.text(1.5, ax.get_ylim()[1]*0.98, "V1", color=BRAIN, fontsize=7, ha="center", va="top")
    ax.text(0.02, 0.02, "dashed = floor from the |G′| conversion\n(perception switched off)",
            transform=ax.transAxes, fontsize=6, color="0.35", va="bottom")

    # --- d: do they line up orientation by orientation? ---------------------
    ax = axg[1, 1]
    common = sorted(set(beh) & set(bra))
    grid = np.array(sorted(beh[common[0]].index.values))
    Bh = np.vstack([np.interp(grid, q.index.values, q.values) for q in beh.values()])
    Br = np.vstack([np.interp(grid, bra[s_].index.values, bra[s_].values) for s_ in common])
    from scipy import stats as _st
    r = _st.pearsonr(Br.mean(0), Bh.mean(0))
    sc = ax.scatter(Br.mean(0), Bh.mean(0), c=grid, cmap="twilight", s=30,
                    edgecolor="white", linewidth=0.4)
    cb = fig.colorbar(sc, ax=ax, ticks=[0, 45, 90, 135, 180])
    cb.set_label("Orientation (deg)", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax.set_xlabel("V1 relative precision"); ax.set_ylabel("Behavioural relative precision")
    ax.set_title(f"Same structure, different phase:  r = {r.statistic:+.2f}")
    ax.text(0.02, 0.02, "one dot per orientation, group means",
            transform=ax.transAxes, fontsize=6, color="0.35", va="bottom")

    for name, A in amps.items():
        print(f"{name:6s} n={len(A):2d}  2θ {np.median(A[:,0]):.3f}  4θ {np.median(A[:,1]):.3f}")
    print(f"profile correspondence r = {r.statistic:+.3f}")
    sns.despine(fig=fig, offset=4)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
