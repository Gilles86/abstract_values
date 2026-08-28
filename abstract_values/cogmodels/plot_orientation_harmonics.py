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

    beh, bra = behavioural_profiles(a.paradigm_tsv), brain_profiles(a.brain_tsv)
    beh_sd, bra_sd = behavioural_sd_deg(a.paradigm_tsv), brain_sd_deg(a.brain_tsv)
    fig, axg = plt.subplots(2, 2, figsize=(7.25, 5.0), constrained_layout=True)
    axes = [axg[0, 1], axg[1, 0], axg[1, 1]]

    # --- the implied noise models, in degrees, unnormalised -----------------
    ax = axg[0, 0]
    for cardinal in (0, 90, 180):
        ax.axvline(cardinal, color="0.88", lw=0.7, ls=":", zorder=0)
    for profs, col, mk, lab in ((beh_sd, BEHAV, "o", "Bids"),
                                (bra_sd, BRAIN, None, "V1")):
        grid = sorted(set(np.concatenate([q.index.values for q in profs.values()])))
        M = np.vstack([np.interp(grid, q.index.values, q.values) for q in profs.values()])
        m, se = M.mean(0), M.std(0) / np.sqrt(len(M))
        ax.fill_between(grid, m - se, m + se, color=col, alpha=0.22, lw=0)
        ax.plot(grid, m, color=col, lw=1.3, marker=mk, ms=3.5 if mk else 0,
                mec="white", mew=0.5)
        ax.text(3, m[0] * (1.18 if col == BEHAV else 0.72), lab, color=col, fontsize=7)
    ax.set_xticks([0, 45, 90, 135, 180]); ax.set_xlim(0, 180)
    ax.set_xlabel("Orientation θ (deg)")
    ax.set_ylabel("Implied orientation noise (deg)")
    ax.set_title("The two implied noise models")


    # --- a: the two precision profiles, each normalised to its own mean ------
    ax = axes[0]
    for cardinal in (0, 90, 180):
        ax.axvline(cardinal, color="0.88", lw=0.7, ls=":", zorder=0)
    ax.axhline(1.0, color="0.5", lw=0.9, ls="--", zorder=1)
    band(ax, beh, BEHAV, "Behaviour", "o")
    band(ax, bra, BRAIN, "V1", None)
    ax.set_xticks([0, 45, 90, 135, 180]); ax.set_xticklabels(["0", "45", "90", "135", "180"], fontsize=6.5)
    ax.set_xlim(0, 180)
    ax.set_xlabel("Orientation θ (deg)")
    ax.set_ylabel("Relative precision\n(1 = subject's own mean)")
    ax.set_title("Precision is not uniform across orientation")
    ax.text(3, ax.get_ylim()[1] * 0.99, f"Bids (n = {len(beh)})", color=BEHAV,
            fontsize=6.5, va="top")
    ax.text(3, ax.get_ylim()[1] * 0.90, f"V1 encoding model (n = {len(bra)})",
            color=BRAIN, fontsize=6.5, va="top")
    ax.text(3, 1.02, "no structure", color="0.45", fontsize=6, va="bottom")

    # --- b: per-subject harmonic amplitudes, against a clear zero -----------
    ax = axes[1]
    amps = {}
    for name, profs in (("Behaviour", beh), ("V1", bra)):
        amps[name] = np.array([harmonics(p.index.values, p.values) for p in profs.values()])
    rng = np.random.default_rng(0)
    x0 = {"Behaviour": 0.0, "V1": 1.5}
    ax.axhline(0, color="0.25", lw=1.0, zorder=3)          # a clear zero
    for name, col in (("Behaviour", BEHAV), ("V1", BRAIN)):
        A = amps[name]
        for j, off in ((0, -0.28), (1, 0.28)):
            x = x0[name] + off + rng.normal(0, 0.03, len(A))
            ax.plot(x, A[:, j], "o", ms=3.6, color=col, alpha=0.5,
                    mec="white", mew=0.4, zorder=2)
            ax.hlines(np.median(A[:, j]), x0[name]+off-0.16, x0[name]+off+0.16,
                      color=col, lw=2.2, zorder=4)
            if name == "Behaviour":                         # artefact floor
                f = NULL_FLOOR["cos 2θ" if j == 0 else "cos 4θ"]
                ax.hlines(f, x0[name]+off-0.18, x0[name]+off+0.18, color="0.35",
                          lw=1.1, ls=(0, (2.5, 1.6)), zorder=5)
    ax.set_xticks([-0.28, 0.28, 1.22, 1.78])
    ax.set_xticklabels(["2θ", "4θ", "2θ", "4θ"], fontsize=7)
    ax.set_xlim(-0.62, 2.12); ax.set_ylim(bottom=0)
    ax.set_ylabel("Harmonic amplitude  (fraction of mean)")
    ax.set_title("Both harmonics are present in both")
    ax.text(0, ax.get_ylim()[1]*0.99, "Bids", color=BEHAV, fontsize=7, ha="center", va="top")
    ax.text(1.5, ax.get_ylim()[1]*0.99, "V1", color=BRAIN, fontsize=7, ha="center", va="top")

    # --- c: do the two profiles agree orientation by orientation? -----------
    ax = axes[2]
    common = sorted(set(beh) & set(bra))
    grid = np.array(sorted(beh[common[0]].index.values))
    Bh = np.vstack([np.interp(grid, beh[s].index.values, beh[s].values) for s in beh.values().__iter__().__class__ and beh])
    Bh = np.vstack([np.interp(grid, v.index.values, v.values) for v in beh.values()])
    Br = np.vstack([np.interp(grid, bra[s].index.values, bra[s].values) for s in common])
    x, y = Br.mean(0), Bh.mean(0)
    from scipy import stats as _st
    r = _st.pearsonr(x, y)
    sc = ax.scatter(x, y, c=grid, cmap="twilight", s=26, edgecolor="white", linewidth=0.4)
    ax.axhline(1.0, color="0.85", lw=0.7, ls=":"); ax.axvline(1.0, color="0.85", lw=0.7, ls=":")
    cb = fig.colorbar(sc, ax=ax, ticks=[0, 45, 90, 135, 180])
    cb.set_label("Orientation (deg)", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax.set_xlabel("V1 relative precision"); ax.set_ylabel("Behavioural relative precision")
    ax.set_title(f"But they do not line up:  r = {r.statistic:+.2f}")

    for name in ("Behaviour", "V1"):
        A = amps[name]
        print(f"{name:10s} n={len(A):2d}  cos2θ {np.median(A[:,0]):.3f}   "
              f"cos4θ {np.median(A[:,1]):.3f}   ratio {np.median(A[:,1])/np.median(A[:,0]):.2f}")
    fig.text(0.5, -0.09,
             "Left: each subject's precision profile divided by their own mean, so bids "
             "and V1 are on one scale; shading is ±1 SEM.   "
             "Middle: amplitude of each harmonic per subject, medians as bars; the dashed "
             "line is the floor this analysis produces when perception is noiseless, "
             "created by dividing bid error by |G′| — bids clear it 2.8×.   "
             "Right: each dot is one orientation (group means).",
             ha="center", va="top", fontsize=6, color="0.4", wrap=True)
    sns.despine(fig=fig, offset=4)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
