"""Motor-cortex tuning: what left M1 codes about the bid, and how that relates to noise.

Inputs, written on the cluster by abstract_values/visualize/collect_motor_tuning.py to
derivatives/followup_2026-10/a1_motor_stage/tables/ (rsynced to the local BIDS root):
    motor_tuning_trials.tsv   ROI-mean single-trial betas (gabor / response epoch)
    motor_tuning_prf_cv.tsv   PRF-model cvR² in M1 / pre-SMA (encoding_models/fit_motor_cv.py)
plus notes/data/cogmodel_shrink_mtail_subject_params.tsv and notes/data/motion_summary.tsv.

Panels
    a  Group tuning curve of the left vs right M1 hand area over the bid (gabor-epoch betas)
    b  PRF model comparison (gabor epoch): value vs quantile codes, unwrapped vs wrapped, bid
    c  Per-subject regression: expected slider travel, bid ramp, orientation (cos 2θ), RT
    d  Lateralised extent effect vs total (perceptual + value) noise, partial on mean FD

    ~/mambaforge/bin/python make_motor_tuning_fig.py
"""
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
HERE = Path(__file__).parent
D = HERE.parents[1] / "notes/data"
TAB = Path("/data/ds-abstractvalue/derivatives/followup_2026-10/a1_motor_stage/tables")
C_L, C_R, C_V1, C_SMA = "#C44E52", "#8C8C8C", "#3B5BA5", "#5D8C3F"
C_VAL, C_UNW, C_WR, C_BID = "#8C8C8C", "#8172B2", "#2A9D8F", "#C44E52"
SLIDER = 42.0                       # fMRI slider range [0, 42]; start uniform on it

mpl.rcParams.update({
    "font.family": "Helvetica", "font.size": 12, "axes.labelsize": 12.5,
    "axes.titlesize": 12.5, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.9,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "svg.fonttype": "path", "pdf.fonttype": 42,
})


def extent(b):
    """E|b − U(0, 42)|: expected slider travel for a bid b from a random start."""
    return (b ** 2 + (SLIDER - b) ** 2) / (2 * SLIDER)


def load_trials():
    t = pd.read_csv(TAB / "motor_tuning_trials.tsv", sep="\t", dtype={"subject": str})
    t = t[t.bid.notna() & t.rt.notna()].copy()
    t["ext"] = extent(t.bid)
    t["cos"] = np.cos(np.deg2rad(t.orientation) * 2)
    return t


def demean(s, cols):
    s = s.copy()
    keys = [k for k in ("subject", "roi", "epoch") if k in s] + ["session", "run"]
    for c in cols:
        s[c] = s[c] - s.groupby(keys)[c].transform("mean")
    return s


def coefs(t, cols, rois, epoch):
    rows = []
    for (roi, sub), s in t[(t.epoch == epoch) & t.roi.isin(rois)].groupby(["roi", "subject"]):
        s = demean(s, ["beta"] + cols)
        X = s[cols].to_numpy()
        X = (X - X.mean(0)) / X.std(0)
        y = (s.beta / s.beta.std()).to_numpy()
        b = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0][1:]
        rows.append(dict(roi=roi, subject=sub, **dict(zip(cols, b))))
    return pd.DataFrame(rows)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def partial_corr(x, y, z):
    Z = np.column_stack([np.ones(len(z)), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r = np.corrcoef(rx, ry)[0, 1]
    n = len(x)
    tt = r * np.sqrt((n - 3) / (1 - r ** 2))
    return r, 2 * stats.t.sf(abs(tt), n - 3)


def panel_tuning(ax, t, epoch="gabor"):
    s = t[(t.epoch == epoch) & t.roi.isin(["M1lhand", "M1rhand"])].copy()
    s = demean(s, ["beta"])
    s["z"] = s.groupby(["subject", "roi"]).beta.transform(lambda x: x / x.std())
    edges = np.linspace(0, SLIDER, 8)
    s["bin"] = pd.cut(s.bid, edges, labels=(edges[:-1] + edges[1:]) / 2).astype(float)
    g = s.groupby(["roi", "subject", "bin"]).z.mean().groupby(["roi", "bin"])
    m, se = g.mean(), g.sem()
    for roi, col, lab in [("M1rhand", C_R, "Right M1 (ipsilateral)"),
                          ("M1lhand", C_L, "Left M1 (contralateral)")]:
        x = m[roi].index.to_numpy()
        ax.fill_between(x, m[roi] - se[roi], m[roi] + se[roi], color=col, alpha=.22, lw=0)
        ax.plot(x, m[roi], "o-", color=col, ms=5, lw=1.8)
        ax.text(x[-1] + 1.2, m[roi].iloc[-1], lab.split(" (")[0], color=col, fontsize=10.5,
                va="center")
    # expected-travel prediction, scaled to the left-M1 curve
    xb = np.linspace(0, SLIDER, 200)
    e = extent(m["M1lhand"].index.to_numpy())
    k = np.polyfit(e - e.mean(), m["M1lhand"].to_numpy(), 1)
    ax.plot(xb, k[0] * (extent(xb) - e.mean()) + k[1], color="0.25", ls="--", lw=1.1, zorder=0)
    ax.annotate("Expected slider travel\n(random start)", xy=(13, k[0] * (extent(13) - e.mean()) + k[1]),
                xytext=(10, 0.24), fontsize=9.5, color="0.25", ha="left",
                arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1, mutation_scale=8,
                                shrinkA=2, shrinkB=6))
    ax.axhline(0, color="0.75", lw=0.7, zorder=0)
    ax.set_xticks([0, 10, 21, 32, 42])
    ax.set_xlim(-1, 50)
    ax.set_xlabel("Bid (CHF)")
    ax.set_ylabel("Hand-area beta (z)")
    ax.set_title("U-shaped in the bid, left M1 only", loc="left")


def panel_prf(ax):
    fn = TAB / "motor_tuning_prf_cv.tsv"
    if fn.stat().st_size < 10:
        ax.text(.5, .5, "PRF fits pending", transform=ax.transAxes, ha="center")
        return None
    d = pd.read_csv(fn, sep="\t", dtype={"subject": str}, keep_default_na=False)
    d = d[d.roi.isin(["M1lhand", "M1rhand", "preSMA"]) & (d.epoch == "gabor")]
    d["gain"] = (d.cvr2 - d.cvr2_null) * 1e3
    models = [("gaussian", "Value (CHF)", C_VAL), ("quantile", "Quantile, unwrapped", C_UNW),
              ("quantile-wrapped", "Quantile, wrapped", C_WR),
              ("gaussian-bid", "Bid (CHF)", C_BID)]
    groups = [("gabor", "M1lhand"), ("gabor", "M1rhand"), ("gabor", "preSMA")]
    out = {}
    for j, (mod, lab, col) in enumerate(models):
        for i, (ep, roi) in enumerate(groups):
            v = d[(d.model == mod) & (d.epoch == ep) & (d.roi == roi)].gain
            if v.empty:
                continue
            x = i + (j - 1.5) * 0.18
            ax.errorbar(x, v.mean(), yerr=v.sem(), fmt="o", color=col, ms=6, lw=1.8)
            out[(mod, ep, roi)] = v
        ax.text(0.99, 0.30 - 0.075 * j, lab, color=col, transform=ax.transAxes, ha="right",
                fontsize=9.5)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(3), ["Left M1", "Right M1", "Pre-SMA"])
    ax.set_ylabel("cvR² − null (×10⁻³)")
    ax.set_title("No value or quantile tuning in M1", loc="left")
    return out


def panel_coefs(ax, t):
    cols = ["ext", "bid", "cos", "rt"]
    c = coefs(t, cols, ["M1lhand", "M1rhand", "preSMA", "V1"], "gabor")
    labels = {"ext": "Expected\ntravel", "bid": "Bid\n(ramp)", "cos": "Orientation\n(cos 2θ)",
              "rt": "RT"}
    rois = [("M1lhand", "Left M1", C_L), ("M1rhand", "Right M1", C_R),
            ("preSMA", "Pre-SMA", C_SMA), ("V1", "V1", C_V1)]
    for j, (roi, lab, col) in enumerate(rois):
        for i, k in enumerate(cols):
            v = c.loc[c.roi == roi, k]
            x = i + (j - 1.5) * 0.19
            ax.errorbar(x, v.mean(), yerr=v.sem(), fmt="o", color=col, ms=6, lw=1.8)
            if roi == "M1lhand":
                ax.text(x, v.mean() + v.sem() + 0.015, stars(stats.ttest_1samp(v, 0).pvalue),
                        ha="center", fontsize=9, color=col)
        ax.text(0.99, 0.97 - 0.08 * j, lab, color=col, transform=ax.transAxes, ha="right",
                va="top", fontsize=10)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(len(cols)), [labels[k] for k in cols])
    ax.set_ylabel("Standardised β")
    ax.set_title("Travel, not bid or orientation", loc="left", pad=14)
    return c


def panel_noise(ax, t):
    c = coefs(t, ["ext", "rt", "cos"], ["M1lhand", "M1rhand"], "gabor")
    w = c.pivot(index="subject", columns="roi", values="ext")
    lat = (w.M1lhand - w.M1rhand).rename("lat")
    lat.index = lat.index.to_series().replace({"pil01": "01", "pil02": "02"})
    p = pd.read_csv(D / "cogmodel_shrink_mtail_subject_params.tsv", sep="\t")
    p["subject"] = p.subject.astype(str).str.zfill(2)
    p = p.set_index("subject")
    fd = pd.read_csv(D / "motion_summary.tsv", sep="\t", dtype={"subject": str})
    fd = fd.groupby("subject").mean_fd.mean()
    fd.index = fd.index.to_series().replace({"pil01": "01", "pil02": "02"})
    d = pd.concat([lat, p.total_noise, p.sigma_rep, p.sigma_motor, fd], axis=1).dropna()
    x, y, z = d.lat.to_numpy(), d.total_noise.to_numpy(), d.mean_fd.to_numpy()
    r, pv = partial_corr(x, y, z)
    ax.scatter(x, y, s=22, color=C_L, alpha=.75, lw=0)
    X = np.column_stack([np.ones(len(x)), x, z])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, b[0] + b[1] * xs + b[2] * z.mean(), color=C_L, lw=1.5)
    ax.text(0.97, 0.97, f"Partial r = {r:+.2f}\np = {pv:.3f}, n = {len(d)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10)
    ax.set_xlabel("Travel effect, left − right M1")
    ax.set_ylabel("Total noise T (CHF)")
    ax.set_title("Stronger M1 travel code, less noise", loc="left")
    res = {}
    for q in ("total_noise", "sigma_rep", "sigma_motor"):
        res[q] = partial_corr(x, d[q].to_numpy(), z) + (stats.spearmanr(x, d[q]),)
    return res


def main():
    t = load_trials()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.15, 1.2, 1.05, 0.95]))
    panel_tuning(axes[0], t)
    prf = panel_prf(axes[1])
    c = panel_coefs(axes[2], t)
    res = panel_noise(axes[3], t)
    for ax, l in zip(axes, "abcd"):
        ax.text(-0.12, 1.06, l, transform=ax.transAxes, fontsize=14, fontweight="bold",
                va="bottom", ha="right")
    out = HERE / "figures" / "motor_tuning"
    fig.savefig(f"{out}.svg")
    fig.savefig(f"{out}.png", dpi=200)
    fig.savefig(f"{out}.pdf")

    print("gabor-epoch coefficients (mean, t, p):")
    for roi in ("M1lhand", "M1rhand", "preSMA", "V1"):
        for k in ("ext", "bid", "cos", "rt"):
            v = c.loc[c.roi == roi, k]
            tt = stats.ttest_1samp(v, 0)
            print(f"  {roi:8s} {k:4s} {v.mean():+.3f}  t={tt.statistic:+.2f}  p={tt.pvalue:.2g}")
    for q, (r, pv, sp) in res.items():
        print(f"lat travel vs {q}: partial r={r:+.2f} p={pv:.3f}; rho={sp.statistic:+.2f} p={sp.pvalue:.3f}")
    if prf:
        for k, v in prf.items():
            tt = stats.ttest_1samp(v, 0)
            print(f"PRF {k}: {v.mean():+.2f} ± {v.sem():.2f}  t={tt.statistic:+.2f} p={tt.pvalue:.2g} n={len(v)}")


if __name__ == "__main__":
    main()
