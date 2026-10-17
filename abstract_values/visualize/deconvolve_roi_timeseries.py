"""Deconvolved event-related responses per ROI (nideconv), and their value / bid modulation.

Per participant, all runs concatenated, one GLM per ROI (nideconv
GroupResponseFitter, OLS):

  gabor          Fourier basis (9 regressors, 0-20 s) + the same basis times
                 z(value) (z-scored within run)
  response bar   Fourier basis (9 regressors, 0-20 s) + the same basis times
                 z(bid) (within run; trials without a valid bid are left out)
  confounds      fmriprep cosines + 6 motion parameters (per run)

The gabor and the response bar are ~6.5 s apart on every trial, so the two
kernels are separated only by the jitter of that gap (and the RT) and by the
inter-trial intervals; the kernels are therefore noisier than the raw trial
averages, but they no longer mix the two events.

Input: per-subject files from collect_roi_timeseries (rsync'd to --in-dir).
    python -m abstract_values.visualize.deconvolve_roi_timeseries    # abstract_values env
    -> notes/data/vmpfc_deconv_timecourses.tsv, notes/figures/fig_vmpfc_deconvolved.pdf
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from abstract_values.visualize.plot_vmpfc_dropout import COL, DATA, FIG, MPFC, SHORT, style

REFS = ["PCC", "Lprecentral", "V1", "IPS12", "WM"]
IDX = ["subject", "session", "run"]
N_REG, INTERVAL = 9, [0, 19.92]


def fit_subject(ts, cf, ev):
    from nideconv import GroupResponseFitter
    rois = [c for c in ts.columns if c not in IDX + ["volume"]]
    T = ts.set_index(IDX)[rois]
    C = cf.set_index(IDX).drop(columns="volume").fillna(0)
    ev = ev.copy()
    z = lambda x: (x - x.mean()) / x.std()
    ev["value_z"] = ev.groupby(["session", "run"]).value.transform(z)
    ev["bid_z"] = ev.groupby(["session", "run"]).bid.transform(z)
    ons = ev.set_index(IDX)[["event_type", "onset", "value_z", "bid_z"]]
    ons = ons[~((ons.event_type == "response_bar") & ons.bid_z.isna())].fillna({"bid_z": 0})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = GroupResponseFitter(T, ons, input_sample_rate=1 / ev.tr.iloc[0], confounds=C,
                                concatenate_runs=True, oversample_design_matrix=10)
        g.add_event("gabor", basis_set="fourier", n_regressors=N_REG, interval=INTERVAL,
                    covariates="value_z")
        g.add_event("response_bar", basis_set="fourier", n_regressors=N_REG,
                    interval=INTERVAL, covariates="bid_z")
        g.fit()
        tc = g.get_timecourses()
    tc = tc.reset_index().rename(columns={"event type": "event"})
    return tc.melt(id_vars=["subject", "event", "covariate", "time"], var_name="roi",
                   value_name="psc")


def compute(in_dir, out):
    rows = []
    for f in sorted(Path(in_dir).glob("sub-*_roi_ts.tsv.gz")):
        s = f.name.split("_")[0][4:]
        rd = lambda k: pd.read_csv(Path(in_dir) / f"sub-{s}_{k}", sep="\t",
                                   dtype={"subject": str})
        print(f"sub-{s}", flush=True)
        rows.append(fit_subject(rd("roi_ts.tsv.gz"), rd("confounds.tsv.gz"), rd("events.tsv")))
    d = pd.concat(rows)
    d.to_csv(out, sep="\t", index=False)
    return d


def panel(ax, d, event, cov, rois, sig_y=None):
    for n, r in enumerate(rois):
        w = d[(d.event == event) & (d.covariate == cov) & (d.roi == r)] \
            .pivot(index="subject", columns="time", values="psc")
        t = w.columns.to_numpy()
        mu, se = w.mean().to_numpy(), w.sem().to_numpy()
        ax.fill_between(t, mu - se, mu + se, color=COL[r], alpha=0.18, lw=0)
        ax.plot(t, mu, color=COL[r], lw=1.2)
        if sig_y is not None:
            p = stats.ttest_1samp(w.to_numpy(), 0).pvalue
            ax.plot(t, np.where(p < 0.01, sig_y[0] - sig_y[1] * n, np.nan), color=COL[r],
                    lw=2.0, solid_capstyle="butt")
    ax.axhline(0, color="0.6", lw=0.6, zorder=0)
    ax.set_xlim(0, 20)
    sns.despine(ax=ax, offset=3)


def figure(d, ev_rt):
    fig, axes = plt.subplots(2, 4, figsize=(7.25, 4.0), constrained_layout=True)
    cols = [("gabor", "intercept", "Response to the gabor", "Signal change (%)"),
            ("response_bar", "intercept", "Response to the response bar", "Signal change (%)"),
            ("gabor", "value_z", "Effect of value (gabor)", "% signal per SD of value"),
            ("response_bar", "bid_z", "Effect of the bid (bar)", "% signal per SD of bid")]
    for i, (group, gname) in enumerate(((MPFC, "Medial PFC"), (REFS, "Reference regions"))):
        for j, (ev, cov, ttl, yl) in enumerate(cols):
            ax = axes[i, j]
            ax.axvspan(0, 1.5 if ev == "gabor" else ev_rt, color="#1B8A80" if ev == "gabor"
                       else "#D9730D", alpha=0.12, lw=0)
            sub = d[(d.event == ev) & (d.covariate == cov) & d.roi.isin(group)]
            m = sub.groupby(["roi", "time"]).psc.agg(["mean", "sem"])
            lo, hi = (m["mean"] - m["sem"]).min(), (m["mean"] + m["sem"]).max()
            span = hi - lo
            panel(ax, d, ev, cov, group,
                  sig_y=None if cov == "intercept" else (lo - 0.08 * span, 0.06 * span))
            ax.set_xlabel("Time from gabor (s)" if ev == "gabor" else "Time from bar onset (s)")
            ax.set_ylabel(yl)
            if i == 0:
                ax.set_title(ttl, loc="left", fontweight="bold", fontsize=7.2)
            if j == 0:
                for k, r in enumerate(group):
                    ax.text(1.0, 0.98 - k * 0.085, SHORT[r], color=COL[r],
                            transform=ax.transAxes, ha="right", va="top", fontsize=6.2)
                ax.text(-0.5, 0.5, gname, transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontweight="bold", fontsize=7.5)
            if i == 0 and j == 2:
                ax.text(0.99, 0.02, "Thick bars: p < .01 (uncorrected)", fontsize=5.8,
                        color="0.4", transform=ax.transAxes, ha="right", va="bottom")
    fig.savefig(FIG / "fig_vmpfc_deconvolved.pdf")
    fig.savefig(FIG / "fig_vmpfc_deconvolved.png", dpi=200)
    print(f"Wrote {FIG / 'fig_vmpfc_deconvolved.pdf'}")


def report(d):
    for ev, cov, win in (("gabor", "intercept", (3, 8)), ("response_bar", "intercept", (3, 8)),
                         ("gabor", "value_z", (3, 10)), ("response_bar", "bid_z", (3, 10))):
        print(f"{ev} {cov} mean over {win[0]}-{win[1]} s")
        x = d[(d.event == ev) & (d.covariate == cov) & d.time.between(*win)]
        v = x.groupby(["roi", "subject"]).psc.mean()
        for r in MPFC + REFS:
            y = v.loc[r]
            t = stats.ttest_1samp(y, 0)
            print(f"   {r:12s} {y.mean():+.4f} ± {y.sem():.4f}  t={t.statistic:+.2f} p={t.pvalue:.2g}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", default=str(DATA / "vmpfc_roi_ts"))
    p.add_argument("--recompute", action="store_true")
    a = p.parse_args()
    style()
    out = DATA / "vmpfc_deconv_timecourses.tsv"
    d = compute(a.in_dir, out) if a.recompute or not out.exists() else \
        pd.read_csv(out, sep="\t", dtype={"subject": str})
    rt = pd.concat(pd.read_csv(f, sep="\t") for f in Path(a.in_dir).glob("sub-*_events.tsv"))
    rt = rt[rt.event_type == "response_bar"].duration.median()
    report(d)
    figure(d, rt)


if __name__ == "__main__":
    main()
