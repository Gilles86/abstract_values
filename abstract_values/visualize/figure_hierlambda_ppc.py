"""Posterior predictive checks for the hierarchical lambda model.

Data: per IPS voxel, the preferred value in the Misaligned (cdf) and the Aligned
(inverse cdf) sessions, censored at each mapping's presented range. Three
models, each simulated from its own posterior: everyone at lambda = 0 (value
code), everyone at lambda = 1 (quantile code), and the hierarchical model
(lambda_s from each participant's posterior). Per replicate dataset and
participant: lambda (hierarchical: drawn from the participant's posterior),
then sigma from p(sigma | lambda, data), then each voxel's position x from
p(x | its two preferred values, lambda, sigma), then two new preferred values
(Student-t noise, censored). Summary statistics are computed per replicate and
summarised across replicates.

Top: 2-D histograms of (cdf, inverse-cdf) preferred value, observed and one
replicate per model; lines: noiseless lambda = 0 (identity) and lambda = 1
(same quantile) predictions. Bottom: median inverse-cdf preferred value in
bins of the cdf preferred value, and the sorted per-participant median shift
(inverse cdf - cdf). Bottom: distribution of the voxel shift. Bands: 95%
predictive intervals across replicate datasets.

    python -m abstract_values.visualize.figure_hierlambda_ppc [--roi IPS] [--offset]
    (cluster: --simulate-only writes notes/data/hierlambda_ppc[_offset]_<roi>.npz;
     a local run then only plots)
    -> notes/figures/fig_hierlambda_ppc.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from abstract_values.encoding_models import hierarchical_lambda as hl
ORI_COL, VAL_COL = "#8172B2", "#D08C2E"     # as figure_tuning_connectivity

REPO = Path(__file__).resolve().parents[2]
RANGE = {m: (min(v), max(v)) for m, v in hl.VALUES.items()}
MODEL_COL = {"Value (λ = 0)": VAL_COL, "Quantile (λ = 1)": ORI_COL,
             "Hierarchical": "#2E5E73"}
BINS = np.linspace(5.5, 38.5, 9)


def censor(x, m):
    return np.clip(x, *RANGE[m])


def softmax_draw(logp, rng):
    p = np.exp(logp - logsumexp(logp, -1, keepdims=True))
    c = np.cumsum(p, -1)
    u = rng.random(p.shape[:-1] + (1,))
    return np.minimum((u > c).sum(-1), p.shape[-1] - 1)


def simulate(a, b, lam_post, T, n_rep, rng, M, offsets=np.zeros(1)):
    """Replicate (n_rep, 2, n_voxels) preferred values for one participant.
    lam_post: probability over hl.LAM to draw lambda from; offsets: the grid of
    the per-participant session offset (one zero = no offset)."""
    A = np.stack([hl.obs_loglik(a, s, "cdf") for s in hl.SIG]).astype(np.float32)
    B = np.stack([hl.obs_loglik(b - d, s, "inv", shift=d) for s in hl.SIG
                  for d in offsets]).astype(np.float32)                   # SIG*OFF, n, V
    A = np.repeat(A, len(offsets), 0)
    out = np.empty((n_rep, 2, len(a)))
    js = rng.choice(len(hl.LAM), n_rep, p=lam_post)
    for r, j in enumerate(js):
        ll = A[:, :, M[0, j]] + B[:, :, M[1, j]]                          # SIG*OFF, n, X
        S = (logsumexp(ll, -1) - np.log(len(hl.X))).sum(1)
        k = softmax_draw(S / T, rng)
        x = softmax_draw(ll[k] / T, rng)                                  # n
        sig, dlt = hl.SIG[k // len(offsets)], offsets[k % len(offsets)]
        noise = sig * rng.standard_t(hl.NU, (2, len(a)))
        out[r, 0] = censor(hl.V[M[0, j, x]] + noise[0], "cdf")
        out[r, 1] = censor(hl.V[M[1, j, x]] + dlt + noise[1], "inv")
    return out


DIFF = np.linspace(-40, 40, 41)


def stats_of(cdf, inv, subj):
    """Binned median of inv given cdf; voxel shift histogram (fraction);
    sorted per-participant median shift."""
    idx = np.digitize(cdf, BINS) - 1
    med = np.array([np.median(inv[idx == i]) if (idx == i).sum() else np.nan
                    for i in range(len(BINS) - 1)])
    dh = np.histogram(inv - cdf, DIFF)[0] / len(cdf)
    shift = np.sort(pd.Series(inv - cdf).groupby(subj).median().to_numpy())
    return med, dh, shift


def quantile_line():
    """inverse-cdf value with the same quantile as each cdf value (lambda = 1)."""
    v = np.linspace(*RANGE["cdf"], 200)
    Fc = np.interp(v, np.r_[hl.LO, hl.VALUES["cdf"], hl.HI],
                   np.r_[0, (np.arange(23) + .5) / 23, 1])
    w = np.interp(Fc, np.r_[0, (np.arange(23) + .5) / 23, 1],
                  np.r_[hl.LO, hl.VALUES["inv"], hl.HI])
    return v, w


def replicate(roi, n_rep, offset, cache, fit=None, select="std"):
    """Simulate replicate datasets for the three models; save to ``cache``.
    fit: the fit's output prefix relative to notes/data (default: the std-gate fit);
    select: its voxel gate (``hierarchical_lambda.SELECT``)."""
    fit = fit or ("hierlambda_lognormal_offset" if offset else "hierlambda_lognormal")
    offsets = np.arange(-12., 12.01, 1.) if offset else np.zeros(1)
    rng = np.random.default_rng(0)
    d = hl.load(REPO / "notes/data/session_modes_voxels.tsv.gz", "lognormal", False, select)
    d = d[d[[f"in_{r}" for r in hl.GROUPS[roi]]].any(axis=1)]
    sub = pd.read_csv(REPO / f"notes/data/{fit}_subjects.tsv", sep="\t",
                      dtype={"subject": str})
    sub = sub[sub.roi == roi].set_index("subject")
    d = d[d.subject.isin(sub.index)]                       # min-voxels as in the fit
    T = pd.read_csv(REPO / f"notes/data/{fit}_group.tsv", sep="\t"
                    ).set_index("roi").loc[roi, "temperature"]
    M = hl.mode_table()
    onehot = lambda l: (hl.LAM == l).astype(float)                          # noqa: E731
    obs_c, obs_i, subj = [], [], []
    rep = {m: [] for m in MODEL_COL}
    for s, x in d.groupby("subject"):
        a, b = x.m_cdf.to_numpy(), x.m_inv.to_numpy()
        obs_c.append(censor(a, "cdf"))
        obs_i.append(censor(b, "inv"))
        subj.append(np.full(len(a), s))
        post = np.array(sub.loc[s, "posterior"].split(), float)
        for m, lp in zip(MODEL_COL, (onehot(0), onehot(1), post / post.sum())):
            rep[m].append(simulate(a, b, lp, T, n_rep, rng, M, offsets))
        print(f"  {s}", end="", flush=True)
    print()
    obs_c, obs_i, subj = map(np.concatenate, (obs_c, obs_i, subj))
    rep = {m: np.concatenate(v, -1).astype(np.float32) for m, v in rep.items()}  # R, 2, N
    np.savez_compressed(cache, obs_c=obs_c, obs_i=obs_i, subj=subj,
                        **{f"rep{k}": v for k, v in enumerate(rep.values())})
    print(f"Wrote {cache}")


def main(roi, out, cache):
    from abstract_values.visualize import qc_value_shift  # noqa: F401 house rcParams
    z = np.load(cache)
    obs_c, obs_i, subj = z["obs_c"], z["obs_i"], z["subj"]
    rep = {m: z[f"rep{k}"] for k, m in enumerate(MODEL_COL)}
    o_med, _, o_shift = stats_of(obs_c, obs_i, subj)
    r_stats = {m: [stats_of(r[0], r[1], subj) for r in v] for m, v in rep.items()}

    fig = plt.figure(figsize=(7.25, 6.4))
    gs = fig.add_gridspec(3, 4, left=.08, right=.98, top=.95, bottom=.07, hspace=.62,
                          wspace=.4)
    edges = (np.linspace(*RANGE["cdf"], 23), np.linspace(*RANGE["inv"], 23))
    H = [np.histogram2d(obs_c, obs_i, edges)[0]] + [
        np.histogram2d(v[0, 0], v[0, 1], edges)[0] for v in rep.values()]
    vmax = np.percentile(np.concatenate([h[1:-1, 1:-1].ravel() for h in H]), 99)  # interior
    qv, qw = quantile_line()
    for k, (title, h) in enumerate(zip(["Observed", *MODEL_COL], H)):
        ax = fig.add_subplot(gs[0, k])
        ax.pcolormesh(*edges, h.T, cmap="Greys", vmin=0, vmax=vmax, rasterized=True)
        ax.plot(qv, qv, color=VAL_COL, lw=1)
        ax.plot(qv, qw, color=ORI_COL, lw=1)
        ax.set_aspect("equal")
        ax.set_title(title, loc="left", fontweight="bold",
                     color=MODEL_COL.get(title, "k"))
        ax.set_xlabel("Preferred value, cdf (CHF)")
        if k == 0:
            ax.set_ylabel("Preferred value,\ninverse cdf (CHF)")
            ax.text(36, 9, "λ = 0", color=VAL_COL, fontsize=6, ha="right")
            ax.text(13, 33, "λ = 1", color=ORI_COL, fontsize=6)
    # binned medians
    centres = (BINS[:-1] + BINS[1:]) / 2
    for k, m in enumerate(MODEL_COL):
        ax = fig.add_subplot(gs[1, k])
        meds = np.array([s[0] for s in r_stats[m]])
        lo, hi = np.nanpercentile(meds, [2.5, 97.5], 0)
        ax.fill_between(centres, lo, hi, color=MODEL_COL[m], alpha=.3, lw=0)
        ax.plot(centres, o_med, "o", color="k", ms=3)
        inside = np.mean((o_med >= lo) & (o_med <= hi))
        ax.plot(qv, qv, color=VAL_COL, lw=.6, alpha=.6)
        ax.plot(qv, qw, color=ORI_COL, lw=.6, alpha=.6)
        ax.set_xlim(*RANGE["cdf"])
        ax.set_ylim(*RANGE["inv"])
        ax.set_title(m, loc="left", fontweight="bold", color=MODEL_COL[m])
        ax.text(.97, .04, f"{inside * len(centres):.0f}/{len(centres)} in band",
                transform=ax.transAxes, ha="right", fontsize=6, color="0.3")
        ax.set_xlabel("Preferred value, cdf (CHF)")
        if k == 0:
            ax.set_ylabel("Median, inverse cdf (CHF)")
    # voxel shift distribution
    o_dh = np.histogram(obs_i - obs_c, DIFF)[0] / len(obs_c)
    dc = (DIFF[:-1] + DIFF[1:]) / 2
    for k, m in enumerate(MODEL_COL):
        ax = fig.add_subplot(gs[2, k])
        lo, hi = np.percentile([s[1] for s in r_stats[m]], [2.5, 97.5], 0)
        ax.fill_between(dc, lo, hi, color=MODEL_COL[m], alpha=.35, lw=0)
        ax.step(dc, o_dh, where="mid", color="k", lw=.9)
        ax.set_xlim(-38, 38)
        ax.set_title(m, loc="left", fontweight="bold", color=MODEL_COL[m])
        ax.set_xlabel("Voxel shift, inverse cdf − cdf (CHF)")
        if k == 0:
            ax.set_ylabel("Fraction of voxels")
    # sorted per-participant median shift
    ax = fig.add_subplot(gs[1, 3])
    rank = np.arange(len(o_shift))
    for m in MODEL_COL:
        lo, hi = np.percentile([s[2] for s in r_stats[m]], [2.5, 97.5], 0)
        ax.fill_between(rank, lo, hi, color=MODEL_COL[m], alpha=.3, lw=0)
    ax.plot(rank, o_shift, "o", color="k", ms=2.2)
    ax.axhline(0, color="0.8", lw=.6, zorder=0)
    ax.set_xlabel("Participant (sorted)")
    ax.set_ylabel("Median shift (CHF)")
    ax.set_title("Per participant", loc="left", fontweight="bold")
    # key for the bands
    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    for k, (m, c) in enumerate(MODEL_COL.items()):
        y = .85 - .16 * k
        ax.add_patch(plt.Rectangle((.02, y - .04), .12, .08, color=c, alpha=.35, lw=0,
                                   transform=ax.transAxes))
        ax.text(.18, y, f"{m.split(' (')[0]}, 95% band", fontsize=6.3, va="center", color=c,
                transform=ax.transAxes)
    ax.plot([.02, .14], [.85 - .16 * 3] * 2, color="k", lw=.9, transform=ax.transAxes)
    ax.plot(.08, .85 - .16 * 4, "o", color="k", ms=3, transform=ax.transAxes)
    ax.text(.18, .85 - .16 * 3, "Observed (histogram)", fontsize=6.3, va="center",
            transform=ax.transAxes)
    ax.text(.18, .85 - .16 * 4, "Observed (summary)", fontsize=6.3, va="center",
            transform=ax.transAxes)
    fig.text(.01, .99, f"{roi}, {len(obs_c)} voxels, {len(np.unique(subj))} participants",
             fontsize=6, color="0.4", va="top")
    out = Path(out)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=250)
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roi", default="IPS")
    ap.add_argument("--n-rep", type=int, default=100)
    ap.add_argument("--offset", action="store_true",
                    help="the model with a per-participant session offset")
    ap.add_argument("--out")
    ap.add_argument("--fit", help="fit output prefix relative to notes/data")
    ap.add_argument("--select", default="std", help="that fit's voxel gate")
    ap.add_argument("--name", default="", help="cache / figure name suffix")
    ap.add_argument("--simulate-only", action="store_true",
                    help="write the replicate cache only (cluster); plot locally")
    a = ap.parse_args()
    sfx = ("_offset" if a.offset else "") + a.name
    cache = REPO / f"notes/data/hierlambda_ppc{sfx}_{a.roi}.npz"
    if a.simulate_only:
        replicate(a.roi, a.n_rep, a.offset, cache, a.fit, a.select)
    elif not cache.exists():
        raise SystemExit(f"{cache} missing: run with --simulate-only on the cluster first")
    if not a.simulate_only:
        main(a.roi, a.out or str(REPO / f"notes/figures/fig_hierlambda_ppc{sfx}.pdf"), cache)
