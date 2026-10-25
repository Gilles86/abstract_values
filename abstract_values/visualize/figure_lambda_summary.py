"""Figures for the executive summary of the hierarchical-lambda analysis.

Does parietal value tuning stay put in CHF (lambda = 0, value code) or move with
the value distribution of the session, keeping its quantile (lambda = 1)?

  concept     a: value -> quantile under the two mappings; b: predicted preferred
              values in the two sessions for lambda = 0, .5, 1; c: observed IPS voxels.
  group       a: group mean mu per ROI (default model); b: mu across voxel gate x
              offset model; c: per-participant lambda (IPS, default model).
  checks      a-b: posterior predictive per-participant median session shift, without
              and with the session offset; c-e: parameter recovery of the offset model.
  individual  a: odd-run vs even-run lambda (split-half reliability); b-c: lambda vs
              behavioural lambda and the cognitive model's value noise.

Default model: session-shift aPRF voxel gate (cvR^2 over null), per-participant
session offset, unsmoothed, log-Gaussian. Correlations: Spearman on posterior means.

    python -m abstract_values.visualize.figure_lambda_summary [concept group checks individual]
    -> notes/figures/fig_lambda_summary_<name>.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter

from abstract_values.encoding_models import hierarchical_lambda as hl
from abstract_values.visualize import qc_value_shift as q   # noqa: F401 house rcParams
from abstract_values.visualize.figure_tuning_connectivity import ORI_COL, ROI_COL, ROI_NAME, VAL_COL

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "notes/data"
FIG = REPO / "notes/figures"
DEFAULT = "hierlambda_default/hierlambda_lognormal_sel-shift0_offset_default"
HALF = "hierlambda_default/hierlambda_lognormal_sel-shift0_offset_runs-{}"
VARIANTS = {   # label -> fit prefix (relative to notes/data)
    "Value-aPRF gate": "hierlambda_lognormal",
    "Value-aPRF gate + offset": "hierlambda_lognormal_offset",
    "Session-shift gate": "hierlambda_selection/hierlambda_lognormal_sel-shift0",
    "Session-shift gate + offset": DEFAULT}
GROUPS = ["IPS", "IPS0", "IPS12", "IPS345", "V3ab"]
COL = {**ROI_COL, "IPS": "k"}
NAME = {**ROI_NAME, "IPS": "IPS (pooled)"}
IPS_C = "#2E5E73"
MAP_COL = {"cdf": "#3B5BA5", "inv": "#C44E52"}
MAP_NAME = {"cdf": "Misaligned (cdf)", "inv": "Aligned (inverse cdf)"}


def sid(s):
    return int(str(s).replace("pil", ""))


def finish(fig, name):
    for a in fig.axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    out = FIG / f"fig_lambda_summary_{name}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=250)
    print(f"Wrote {out}")


def title(ax, s):
    ax.set_title(s, loc="left", fontweight="bold")


def rho_text(ax, x, y, where=(.03, .97)):
    r, p = stats.spearmanr(x, y)
    ax.text(*where, f"ρ = {r:.2f}, p = {p:.2g}, n = {len(x)}".replace("-", "−"),
            transform=ax.transAxes, fontsize=6.5, va="top", color="0.2")


def lam_refs(ax, axis="x"):
    line = ax.axvline if axis == "x" else ax.axhline
    line(0, color=VAL_COL, lw=.7, alpha=.7, zorder=0)
    line(1, color=ORI_COL, lw=.7, alpha=.7, zorder=0)


def pairs(lam):
    """Noise-free preferred value (cdf, inverse cdf) of voxels along X, for one lambda."""
    M = hl.mode_table()
    j = np.argmin(np.abs(hl.LAM - lam))
    return hl.V[M[0, j]], hl.V[M[1, j]]


# ── concept ───────────────────────────────────────────────────────────────────
def concept():
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.6),
                             gridspec_kw=dict(left=.07, right=.98, top=.86, bottom=.22,
                                              wspace=.42))
    ax = axes[0]
    for m in ("cdf", "inv"):
        ax.plot(hl.V, hl.quantile_fn(m), color=MAP_COL[m], lw=1.2)
        ax.plot(hl.VALUES[m], (np.arange(23) + .5) / 23, "o", ms=2.2, color=MAP_COL[m], mew=0)
    ax.text(30, .3, MAP_NAME["cdf"], color=MAP_COL["cdf"], fontsize=6.5, ha="left")
    ax.text(3, .88, MAP_NAME["inv"], color=MAP_COL["inv"], fontsize=6.5, ha="left")
    ax.set_xlabel("Value (CHF)")
    ax.set_ylabel("Quantile in session\n(= orientation rank)")
    ax.set_xlim(2, 42)
    title(ax, "a  Two value distributions")

    ax = axes[1]
    for lam, c, lab in ((0, VAL_COL, "λ = 0: same value"), (.5, "0.55", "λ = 0.5"),
                        (1, ORI_COL, "λ = 1: same quantile")):
        a, b = pairs(lam)
        ax.plot(a, b, color=c, lw=1.4)
    ax.text(41, 19, "λ = 0\nsame value", color=VAL_COL, fontsize=6.3, ha="right", va="top")
    ax.text(12, 33, "λ = 1\nsame quantile", color=ORI_COL, fontsize=6.3, ha="center")
    ax.annotate("λ = 0.5", xy=(16.5, 20.6), xytext=(22, 9), color="0.45", fontsize=6.3,
                arrowprops=dict(arrowstyle="-", color="0.6", lw=.5))
    ax.set_xlim(2, 42)
    ax.set_ylim(2, 42)
    ax.set_aspect("equal")
    ax.set_xlabel("Preferred value, Misaligned (CHF)")
    ax.set_ylabel("Preferred value, Aligned (CHF)")
    title(ax, "b  What each code predicts")

    ax = axes[2]
    d = hl.load(DATA / "session_modes_voxels.tsv.gz", "lognormal", False, "shift")
    d = d[d[[f"in_{r}" for r in hl.GROUPS["IPS"]]].any(axis=1)]
    (lc, hc), (li, hi) = (hl.RANGE if hasattr(hl, "RANGE") else
                          [(min(hl.VALUES[m]), max(hl.VALUES[m])) for m in ("cdf", "inv")])
    inside = d.m_cdf.between(lc, hc) & d.m_inv.between(li, hi)
    ex, ey = np.linspace(lc, hc, 34), np.linspace(li, hi, 40)
    h = gaussian_filter(np.histogram2d(d.m_cdf[inside], d.m_inv[inside], (ex, ey))[0], 1)
    ax.pcolormesh(ex, ey, h.T, cmap="Greys", vmin=0, vmax=np.percentile(h, 99.5),
                  rasterized=True)
    for lam, c in ((0, VAL_COL), (1, ORI_COL)):
        ax.plot(*pairs(lam), color=c, lw=1)
    ax.set_xlim(2, 42)
    ax.set_ylim(2, 42)
    ax.set_aspect("equal")
    ax.set_xlabel("Preferred value, Misaligned (CHF)")
    ax.set_ylabel("Preferred value, Aligned (CHF)")
    title(ax, "c  IPS voxels (observed)")
    ax.text(1.0, -.27, f"{inside.sum()} of {len(d)} voxels within the presented ranges; "
            f"{d.subject.nunique()} participants", transform=ax.transAxes, fontsize=5.5,
            ha="right", color="0.4")
    finish(fig, "concept")


# ── group ─────────────────────────────────────────────────────────────────────
def group():
    pop = np.load(DATA / f"{DEFAULT}_population.npz")
    grp = pd.read_csv(DATA / f"{DEFAULT}_group.tsv", sep="\t").set_index("roi")
    sub = pd.read_csv(DATA / f"{DEFAULT}_subjects.tsv", sep="\t", dtype={"subject": str})
    fig = plt.figure(figsize=(7.25, 2.7))
    gs = fig.add_gridspec(1, 3, left=.07, right=.98, top=.86, bottom=.17, wspace=.45,
                          width_ratios=[1, 1.15, 1])
    ax = fig.add_subplot(gs[0])
    lam_refs(ax)
    MU = pop["MU"]
    for g in GROUPS:
        ax.plot(MU, pop[f"{g}_mu"] / np.diff(MU)[0], color=COL[g], lw=1.6 if g == "IPS" else 1)
    for k, g in enumerate(GROUPS):
        yy = .95 - .085 * k
        ax.plot([.03, .11], [yy, yy], color=COL[g], lw=1.6 if g == "IPS" else 1,
                transform=ax.transAxes)
        ax.text(.14, yy, NAME[g], color=COL[g], fontsize=6, va="center", transform=ax.transAxes)
    ax.set_xlim(-1, 2)
    ax.set_ylim(0, None)
    ax.set_xlabel("Group mean λ (μ)")
    ax.set_ylabel("Posterior density")
    title(ax, "a  Group λ")

    ax = fig.add_subplot(gs[1])
    lam_refs(ax)
    rois = ["IPS", "IPS0", "IPS12", "IPS345"]
    vcol = ["0.65", "0.4", "#7FA7BA", IPS_C]
    for k, (lab, f) in enumerate(VARIANTS.items()):
        p = DATA / f"{f}_group.tsv"
        if not p.exists():
            continue
        g = pd.read_csv(p, sep="\t").set_index("roi")
        for i, r in enumerate(rois):
            y = i + (k - 1.5) * .17
            ax.plot([g.loc[r, "mu_lo"], g.loc[r, "mu_hi"]], [y, y], color=vcol[k], lw=1.1)
            ax.plot(g.loc[r, "mu"], y, "o", color=vcol[k], ms=3, mew=0)
    for k, lab in enumerate(VARIANTS):                       # key, above the rows
        y = -2.05 + .3 * k
        ax.plot([-.95, -.75], [y, y], color=vcol[k], lw=1.1)
        ax.plot(-.85, y, "o", color=vcol[k], ms=3, mew=0)
        ax.text(-.68, y, lab, fontsize=5.8, color=vcol[k], va="center")
    ax.set_yticks(range(len(rois)), [NAME[r] for r in rois])
    ax.set_ylim(len(rois) - .5, -2.3)
    ax.set_xlim(-1.05, 2.05)
    ax.set_xlabel("Group mean λ (μ), 95% CrI")
    title(ax, "b  Robust to gate and model")

    ax = fig.add_subplot(gs[2])
    h = sub[sub.roi == "IPS"].sort_values("mean").reset_index(drop=True)
    y = np.arange(len(h))
    lam_refs(ax)
    g = grp.loc["IPS"]
    ax.fill_betweenx([-1, len(h)], g.mu_lo, g.mu_hi, color="0.9", lw=0, zorder=-1)
    ax.hlines(y, h.lo, h.hi, color=IPS_C, lw=.7, alpha=.6)
    ax.plot(h["mean"], y, "o", ms=2.6, color=IPS_C, mew=0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-1.05, 2.05)
    ax.set_ylim(-1, len(h))
    ax.set_xlabel("λ (IPS), 95% CrI")
    ax.set_ylabel("Participants (sorted)")
    ax.text(g.mu, len(h) - .2, "Group μ", fontsize=6, color="0.45", ha="center", va="bottom")
    title(ax, "c  Per participant")
    fig.text(.07, .965, f"Default model (session-shift gate + offset): IPS μ = {g.mu:.2f} "
             f"[{g.mu_lo:.2f}, {g.mu_hi:.2f}], τ = {g.tau:.2f}, {int(g.n_subjects)} participants"
             .replace("[-", "[−"),
             fontsize=6.5, color="0.3")
    finish(fig, "group")


# ── checks ────────────────────────────────────────────────────────────────────
def median_shift(cdf, inv, subj):
    return np.sort(pd.Series(inv - cdf).groupby(subj).median().to_numpy())


def checks():
    fig = plt.figure(figsize=(7.25, 4.6))
    gs = fig.add_gridspec(2, 3, left=.08, right=.98, top=.93, bottom=.1, hspace=.62, wspace=.45)
    for k, (name, lab) in enumerate((("shift", "Without session offset"),
                                     ("offset_default", "With session offset"))):
        ax = fig.add_subplot(gs[0, k], sharey=fig.axes[0] if k else None)
        p = DATA / f"hierlambda_ppc_{name}_IPS.npz"
        title(ax, f"{'ab'[k]}  {lab}")
        if not p.exists():
            ax.text(.5, .5, "Pending", transform=ax.transAxes, ha="center")
            continue
        z = np.load(p)
        o = median_shift(z["obs_c"], z["obs_i"], z["subj"])
        sims = np.array([median_shift(r[0], r[1], z["subj"]) for r in z["rep2"]])
        lo, hi = np.percentile(sims, [2.5, 97.5], 0)
        rank = np.arange(len(o))
        ax.fill_between(rank, lo, hi, color=IPS_C, alpha=.3, lw=0)
        ax.plot(rank, o, "o", color="k", ms=2.2)
        ax.axhline(0, color="0.8", lw=.6, zorder=0)
        inside = np.mean((o >= lo) & (o <= hi))
        ax.text(.97, .04, f"{inside * len(o):.0f}/{len(o)} in band", transform=ax.transAxes,
                ha="right", fontsize=6, color="0.3")
        ax.set_xlabel("Participant (sorted)")
        ax.set_ylabel("Median voxel shift,\nAligned − Misaligned (CHF)")
        if k == 0:
            ax.add_patch(plt.Rectangle((.04, .86), .07, .06, color=IPS_C, alpha=.3, lw=0,
                                       transform=ax.transAxes))
            ax.text(.14, .89, "Model, 95% predictive", fontsize=6, va="center", color=IPS_C,
                    transform=ax.transAxes)
            ax.plot(.075, .78, "o", color="k", ms=2.2, transform=ax.transAxes)
            ax.text(.14, .78, "Observed", fontsize=6, va="center", transform=ax.transAxes)

    p = DATA / "hierlambda_offset_recovery.tsv"
    ax = fig.add_subplot(gs[0, 2])                             # offsets on the real data
    z = DATA / "hierlambda_ppc_offset_default_IPS.npz"
    sub = pd.read_csv(DATA / f"{DEFAULT}_subjects.tsv", sep="\t", dtype={"subject": str})
    sub = sub[sub.roi == "IPS"].set_index("subject")
    if z.exists() and "offset_mean" in sub:
        z = np.load(z)
        med = pd.Series(z["obs_i"] - z["obs_c"]).groupby(z["subj"]).median()
        o = sub.offset_mean.reindex(med.index)
        ax.plot([-15, 15], [-15, 15], color="0.8", lw=.6, zorder=0)
        ax.errorbar(med, o, yerr=sub.offset_sd.reindex(med.index), fmt="o", ms=2.6, color="0.3",
                    mew=0, elinewidth=.5, ecolor=(.3, .3, .3, .35))
        rho_text(ax, med, o)
    ax.set_xlabel("Observed median voxel shift (CHF)")
    ax.set_ylabel("Estimated session offset (CHF)")
    title(ax, "c  Offsets absorb the shifts")
    if not p.exists():
        finish(fig, "checks")
        return
    r = pd.read_csv(p, sep="\t")
    c0 = r[r.condition == 0]
    ax = fig.add_subplot(gs[1, 0])
    ax.plot([-1, 2], [-1, 2], color="0.8", lw=.6, zorder=0)
    ax.errorbar(c0.lam_true, c0.lam_est, yerr=c0.lam_sd, fmt="o", ms=2.6, color=IPS_C, mew=0,
                elinewidth=.5, ecolor=(*matplotlib.colors.to_rgb(IPS_C), .35))
    rr = np.corrcoef(c0.lam_true, c0.lam_est)[0, 1]
    ax.text(.03, .97, f"r = {rr:.2f}", transform=ax.transAxes, fontsize=6.5, va="top")
    ax.set_xlabel("True λ")
    ax.set_ylabel("Recovered λ")
    title(ax, "d  Recovery (model): participants")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot([-.2, 1.2], [-.2, 1.2], color="0.8", lw=.6, zorder=0)
    lab = {0: "Spread, σ 6", 1: "Spread, σ 10", 2: "All value", 3: "All quantile"}
    for c, x in r.groupby("condition"):
        x = x.iloc[0]
        jit = (c - 1.5) * .03
        ax.plot([x.mu_true + jit] * 2, [x.mu_lo, x.mu_hi], color=IPS_C, lw=1)
        ax.plot(x.mu_true + jit, x.mu_est, "o", color=IPS_C, ms=3.2, mew=0)
        left = c == 0
        ax.text(x.mu_true + jit + (-.05 if left else .05), x.mu_est, lab[c], fontsize=5.8,
                va="center", ha="right" if left else "left", color="0.35")
    ax.set_xlabel("True group μ")
    ax.set_ylabel("Recovered μ, 95% CrI")
    title(ax, "e  Recovery (model): group")

    ax = fig.add_subplot(gs[1, 2])
    ax.plot([-9, 9], [-9, 9], color="0.8", lw=.6, zorder=0)
    ax.plot(c0.delta_true, c0.delta_est, "o", ms=2.6, color="0.3", mew=0)
    err = c0.lam_est - c0.lam_true
    ax.text(.03, .97, f"r = {np.corrcoef(c0.delta_true, c0.delta_est)[0, 1]:.2f}\n"
            f"λ error vs true offset: r = {np.corrcoef(err, c0.delta_true)[0, 1]:.2f}"
            .replace("-", "−"), transform=ax.transAxes, fontsize=6.2, va="top")
    ax.set_xlabel("True session offset (CHF)")
    ax.set_ylabel("Recovered offset (CHF)")
    title(ax, "f  Recovery (model): offset")
    finish(fig, "checks")


# ── end-to-end pipeline check ──────────────────────────────────────────────────
SIM = "hierlambda_default/hierlambda_lognormal_sel-shift0_offset_sim-{}"


def density(ax, tsv, label):
    d = hl.load(DATA / tsv, "lognormal", False, "shift")
    d = d[d[[f"in_{r}" for r in hl.GROUPS["IPS"]]].any(axis=1)]
    (lc, hc), (li, hi) = [(min(hl.VALUES[m]), max(hl.VALUES[m])) for m in ("cdf", "inv")]
    inside = d.m_cdf.between(lc, hc) & d.m_inv.between(li, hi)
    ex, ey = np.linspace(lc, hc, 34), np.linspace(li, hi, 40)
    h = gaussian_filter(np.histogram2d(d.m_cdf[inside], d.m_inv[inside], (ex, ey))[0], 1)
    ax.pcolormesh(ex, ey, h.T, cmap="Greys", vmin=0, vmax=np.percentile(h, 99.5),
                  rasterized=True)
    for lam, c in ((0, VAL_COL), (1, ORI_COL)):
        ax.plot(*pairs(lam), color=c, lw=1)
    ax.set_xlim(2, 42)
    ax.set_ylim(2, 42)
    ax.set_aspect("equal")
    ax.set_xlabel("Preferred value, Misaligned (CHF)")
    title(ax, label)


def pipeline():
    fig = plt.figure(figsize=(7.25, 2.75))
    gs = fig.add_gridspec(1, 4, left=.06, right=.99, top=.86, bottom=.18, wspace=.42,
                          width_ratios=[1, 1, 1, 1.6])
    for k, (tsv, lab) in enumerate((("session_modes_voxels.tsv.gz", "a  Observed"),
                                    ("session_modes_voxels_sim-lam0.tsv.gz", "b  Simulated, λ = 0"),
                                    ("session_modes_voxels_sim-lam1.tsv.gz", "c  Simulated, λ = 1"))):
        ax = fig.add_subplot(gs[k])
        density(ax, tsv, lab)
        if k == 0:
            ax.set_ylabel("Preferred value, Aligned (CHF)")
    ax = fig.add_subplot(gs[3])
    rois = ["IPS", "IPS0", "IPS12", "IPS345"]
    for l, c in ((0, VAL_COL), (1, ORI_COL)):                  # refs over the rows only
        ax.plot([l, l], [-.6, len(rois) - .5], color=c, lw=.7, alpha=.7, zorder=0)
    rows = [("Simulated λ = 0, all runs", SIM.format("lam0"), VAL_COL, "o"),
            ("Simulated λ = 0, half the runs", SIM.format("lam0.runs-odd"), VAL_COL, "s"),
            ("Simulated λ = 1, all runs", SIM.format("lam1"), ORI_COL, "o"),
            ("Simulated λ = 1, half the runs", SIM.format("lam1.runs-odd"), ORI_COL, "s"),
            ("Observed, all runs", DEFAULT, "k", "o"),
            ("Observed, odd runs", HALF.format("odd"), "k", "s"),
            ("Observed, even runs", HALF.format("even"), "k", "D")]
    n = len(rows)
    for k, (lab, f, c, mk) in enumerate(rows):
        p = DATA / f"{f}_group.tsv"
        y0 = -n * .27 - .5 + k * .27                       # key, above the rows
        ax.plot([-.95, -.75], [y0, y0], color=c, lw=1, alpha=.5 if "half" in lab else 1)
        ax.plot(-.85, y0, mk, color=c, ms=2.8, mew=0, mfc=c if "all" in lab else "white",
                mec=c)
        ax.text(-.68, y0, lab, fontsize=5.6, color=c, va="center")
        if not p.exists():
            continue
        g = pd.read_csv(p, sep="\t").set_index("roi")
        for i, r in enumerate(rois):
            y = i + (k - (n - 1) / 2) * .1
            ax.plot([g.loc[r, "mu_lo"], g.loc[r, "mu_hi"]], [y, y], color=c, lw=.9,
                    alpha=.5 if "half" in lab else 1)
            ax.plot(g.loc[r, "mu"], y, mk, color=c, ms=2.8, mew=.6,
                    mfc=c if "all" in lab else "white", mec=c)
    ax.set_yticks(range(len(rois)), [NAME[r] for r in rois])
    ax.set_ylim(len(rois) - .5, -n * .27 - .8)
    ax.set_xlim(-1.05, 2.05)
    ax.set_xlabel("Recovered group μ, 95% CrI")
    title(ax, "d  Recovered after refitting")
    finish(fig, "pipeline")


# ── individual differences ────────────────────────────────────────────────────
def externals():
    lb = pd.read_csv(DATA / "lambda_behaviour.tsv", sep="\t", dtype={"subject": str})
    out = dict(lam_b=lb.set_index(lb.subject.map(sid)).lam_b)
    p = DATA / "cogmodel_shrink_mtail_subject_params.tsv"
    if p.exists():
        cog = pd.read_csv(p, sep="\t")
        out["sigma_rep"] = cog.set_index(cog.subject.astype(int)).sigma_rep
    return out


def individual():
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.45),
                             gridspec_kw=dict(left=.07, right=.98, top=.86, bottom=.18,
                                              wspace=.4))
    sub = pd.read_csv(DATA / f"{DEFAULT}_subjects.tsv", sep="\t", dtype={"subject": str})
    h = sub[sub.roi == "IPS"].set_index(sub[sub.roi == "IPS"].subject.map(sid))
    ax = axes[0]
    po, pe = (DATA / f"{HALF.format(k)}_subjects.tsv" for k in ("odd", "even"))
    if po.exists() and pe.exists():
        o, e = (pd.read_csv(p, sep="\t", dtype={"subject": str}).query("roi == 'IPS'")
                .assign(s=lambda d: d.subject.map(sid)).set_index("s") for p in (po, pe))
        j = o.join(e, lsuffix="_o", rsuffix="_e", how="inner")
        ax.plot([-1, 2], [-1, 2], color="0.8", lw=.6, zorder=0)
        ax.errorbar(j.mean_o, j.mean_e, xerr=j.sd_o, yerr=j.sd_e, fmt="o", ms=2.8, color=IPS_C,
                    mew=0, elinewidth=.5, ecolor=(*matplotlib.colors.to_rgb(IPS_C), .3))
        rho_text(ax, j.mean_o, j.mean_e)
        ax.set_xlim(-1.1, 2.1)
        ax.set_ylim(-1.1, 2.4)
    else:
        ax.text(.5, .5, "Pending", transform=ax.transAxes, ha="center")
    ax.set_xlabel("λ, odd runs (IPS)")
    ax.set_ylabel("λ, even runs (IPS)")
    title(ax, "a  Split-half reliability")
    ext = externals()
    for ax, k, lab, t in ((axes[1], "lam_b", "Behavioural λ (bids)", "b  vs behaviour"),
                          (axes[2], "sigma_rep", "Value noise σ_rep (cognitive model)",
                           "c  vs value noise")):
        title(ax, t)
        if k not in ext:
            continue
        x = ext[k].reindex(h.index)
        ok = x.notna()
        ax.vlines(x[ok], h.lo[ok], h.hi[ok], color=IPS_C, lw=.5, alpha=.3)
        ax.plot(x[ok], h["mean"][ok], "o", ms=2.8, color=IPS_C, mew=0)
        rho_text(ax, x[ok], h["mean"][ok])
        lam_refs(ax, "y")
        ax.set_ylim(-1.1, 2.4)
        ax.set_xlabel(lab)
        ax.set_ylabel("λ (IPS)")
    finish(fig, "individual")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["concept", "group", "checks", "pipeline", "individual"]:
        globals()[name]()
