"""Hierarchical Bayesian estimate of lambda from per-session preferred values.

Generative model. Voxel i of participant s has a fixed preferred position x_i
on the mix axis
    x_{lambda,m}(v) = (1 - lambda) v + lambda (2 + 40 F_m(v)),
where F_m is the quantile function of mapping m (cdf = Misaligned, inverse
cdf = Aligned; piecewise linear through the 23 presented values, anchored at
F(2) = 0 and F(42) = 1). Its preferred VALUE under mapping m is therefore the
v that brings x_{lambda,m}(v) closest to x_i: lambda = 0 keeps it fixed in
CHF, lambda = 1 keeps its quantile (= its orientation). The observed
preferred values (session-shift aPRF, one per mapping) are that plus
Student-t noise (nu = 4, scale sigma_s per participant), censored at the
edges of the values PRESENTED under that mapping (cdf 5.5-38.5, inverse cdf
2.5-41.5 CHF): a fitted preferred value beyond the presented range only says
"at or beyond the edge". Without this, the narrower cdf range alone would pull
the cdf-session estimates inwards, which mimics quantile coding.

Everything is evaluated on grids (no sampler): x_i is integrated out
(uniform), sigma_s is integrated out (log-uniform on 1-15 CHF), giving each
participant a log-likelihood curve L_s(lambda). The population model
lambda_s ~ Normal(mu, tau) is then exact on a (mu, tau) grid with flat
priors on mu and a half-normal(0.5) on tau; per-participant posteriors are
the grid posteriors given (mu, tau), mixed over the (mu, tau) posterior.

Voxels are not independent (point spread, shared noise), so per-participant
curves would be over-confident. A temperature T (voxel log-likelihood / T,
roughly "voxels per independent observation") is calibrated as the smallest T
at which left- and right-hemisphere estimates (no pooling) disagree no more
than their posteriors say they should (SD of the L - R z-scores <= 1). The
hemisphere estimates are then reported separately as a reliability check.

    python -m abstract_values.encoding_models.hierarchical_lambda \
        [--tsv notes/data/session_modes_voxels.tsv.gz] [--model lognormal] [--smoothed]
    -> notes/data/hierlambda_<model>{_smoothed}_{subjects,group}.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

REPO = Path(__file__).resolve().parents[2]
VALUES = {   # the 23 presented values per mapping (identical for every participant)
    'cdf': [5.5, 8, 10, 11, 11.5, 12, 12.5, 13, 14, 16, 18.5, 22, 25.5, 28, 30, 31, 31.5,
            32, 32.5, 33, 34, 36, 38.5],
    'inv': [2.5, 3, 4, 5, 7.5, 11.5, 16, 18.5, 20, 21, 21.5, 22, 22.5, 23, 24, 25.5, 28,
            32.5, 36.5, 39, 40, 41, 41.5]}
LO, HI = 2., 42.
V = np.linspace(LO, HI, 401)
X = np.linspace(LO, HI, 161)
LAM = np.round(np.arange(-1, 2.0001, .05), 2)
SIG = np.geomspace(1, 15, 15)
NU = 4
GROUPS = {'IPS': ['IPS0', 'IPS12', 'IPS345'], 'IPS0': ['IPS0'], 'IPS12': ['IPS12'],
          'IPS345': ['IPS345'], 'V3ab': ['V3ab'], 'hV4': ['hV4']}


def quantile_fn(m):
    v = np.r_[LO, VALUES[m], HI]
    q = np.r_[0, (np.arange(23) + .5) / 23, 1]
    return np.interp(V, v, q)


def mode_table():
    """(2 mappings, n_lambda, n_x) index into V: preferred value of a voxel at x."""
    out = np.empty((2, len(LAM), len(X)), int)
    for k, m in enumerate(('cdf', 'inv')):
        F = quantile_fn(m)
        for j, lam in enumerate(LAM):
            ax = (1 - lam) * V + lam * (2 + 40 * F)
            out[k, j] = np.argmin(np.abs(ax[None, :] - X[:, None]), 1)
    return out


def obs_loglik(obs, sig, m):
    """log p(obs | pred = V, sig), censored at mapping m's presented range: (n, len(V))."""
    lo, hi = min(VALUES[m]), max(VALUES[m])
    t = stats.t(NU, loc=V[None, :], scale=sig)
    o = obs[:, None]
    return np.where(o <= lo, t.logcdf(lo), np.where(o >= hi, t.logsf(hi), t.logpdf(o)))


def subject_S(m_cdf, m_inv, M):
    """(n_sigma, n_lambda): voxel-summed log-likelihood per sigma (x integrated)."""
    out = []
    for sig in SIG:
        a, b = obs_loglik(m_cdf, sig, 'cdf'), obs_loglik(m_inv, sig, 'inv')          # n, V
        ll = a[:, M[0]] + b[:, M[1]]                                    # n, L, X
        out.append((logsumexp(ll, -1) - np.log(len(X))).sum(0))
    return np.stack(out)


def curve(S, temp=1.):
    """log-likelihood over LAM, voxel evidence tempered by T, sigma integrated."""
    return logsumexp(S / temp, 0) - np.log(len(SIG))


def grid_post(curve):
    p = np.exp(curve - curve.max())
    return p / p.sum()


def summarize(p):
    c = np.cumsum(p)
    return dict(mean=(p * LAM).sum(), sd=np.sqrt((p * LAM ** 2).sum() - (p * LAM).sum() ** 2),
                lo=LAM[np.searchsorted(c, .025)], hi=LAM[min(np.searchsorted(c, .975), len(LAM) - 1)])


MU = np.round(np.arange(-1, 2.0001, .025), 3)
TAU = np.linspace(.01, 2., 200)


def population(curves):
    """Exact grid posterior over (mu, tau), and per-participant posteriors."""
    C = np.stack(curves)                                         # S, L
    logprior_lam = stats.norm.logpdf(LAM[None, None, :], MU[:, None, None], TAU[None, :, None])
    logprior_lam -= logsumexp(logprior_lam, -1, keepdims=True)    # normalised on the grid
    # log p(data | mu, tau) = sum_s logsumexp_l (C_sl + log p(l | mu, tau))
    lp = logsumexp(C[:, None, None, :] + logprior_lam[None], -1).sum(0)        # MU, TAU
    lp += stats.halfnorm.logpdf(TAU, scale=.5)[None, :]
    post = np.exp(lp - lp.max())
    post /= post.sum()
    # per participant: sum over (mu, tau) of post * p(l | data_s, mu, tau)
    cond = C[:, None, None, :] + logprior_lam[None]
    cond = np.exp(cond - logsumexp(cond, -1, keepdims=True))                    # S, MU, TAU, L
    subj = np.einsum('mt,smtl->sl', post, cond)
    return post, subj


def calibrate_temperature(SL, SR, temps=np.geomspace(1, 128, 22)):
    """T at which L-vs-R hemisphere disagreement matches the posterior SDs.
    SL, SR: per-participant S arrays (None where a hemisphere has < 10 voxels)."""
    rows = []
    for T in temps:
        z = []
        for a, b in zip(SL, SR):
            if a is None or b is None:
                continue
            sa, sb = summarize(grid_post(curve(a, T))), summarize(grid_post(curve(b, T)))
            z.append((sa['mean'] - sb['mean']) / np.hypot(sa['sd'], sb['sd']))
        z = np.array(z)
        rows.append(dict(T=T, z_sd=z.std(), n=len(z)))
    r = pd.DataFrame(rows)
    ok = r[r.z_sd <= 1]
    T = float(ok['T'].iloc[0]) if len(ok) else float(r['T'].iloc[-1])
    print('  ' + ', '.join(f'T {x.T:.0f}: {x.z_sd:.2f}' for x in r.itertuples()
                           if x.Index % 3 == 0), flush=True)
    return T, r


def load(tsv, model, smoothed):
    d = pd.read_csv(tsv, sep='\t', dtype={'subject': str})
    d = d[(d.smoothed.astype(str) == str(smoothed)) & (d.cv_std > d.cv_null)].copy()
    d['m_cdf'], d['m_inv'] = d[f'{model}_cdf'], d[f'{model}_inv']
    return d[np.isfinite(d.m_cdf) & np.isfinite(d.m_inv)]


def main(tsv, model, smoothed, groups):
    d = load(tsv, model, smoothed)
    M = mode_table()
    tag = f'{model}{"_smoothed" if smoothed else ""}'
    subj_rows, grp_rows = [], []
    for gname in groups:
        g = d[d[[f'in_{r}' for r in GROUPS[gname]]].any(axis=1)]
        print(f'{gname}: {len(g)} voxels, {g.subject.nunique()} participants', flush=True)
        subs = sorted(g.subject.unique())
        S, SL, SR = [], [], []
        for _, x in g.groupby('subject'):
            S.append(subject_S(x.m_cdf.to_numpy(), x.m_inv.to_numpy(), M))
            for lst, k in ((SL, 'L'), (SR, 'R')):
                y = x[x.hemi == k]
                lst.append(subject_S(y.m_cdf.to_numpy(), y.m_inv.to_numpy(), M)
                           if len(y) >= 10 else None)
        T, cal = calibrate_temperature(SL, SR)
        print(f'  temperature {T:.1f}', flush=True)
        curves = [curve(x, T) for x in S]
        flat = np.zeros(len(LAM))
        hemi = {'L': [curve(x, T) if x is not None else flat for x in SL],
                'R': [curve(x, T) if x is not None else flat for x in SR]}
        post, subj = population(curves)
        pm, pt = post.sum(1), post.sum(0)
        cm, ct = np.cumsum(pm), np.cumsum(pt)
        grp_rows.append(dict(roi=gname, temperature=T, n_subjects=len(subs), n_voxels=len(g),
                             mu=(pm * MU).sum(), mu_lo=MU[np.searchsorted(cm, .025)],
                             mu_hi=MU[np.searchsorted(cm, .975)],
                             tau=(pt * TAU).sum(), tau_lo=TAU[np.searchsorted(ct, .025)],
                             tau_hi=TAU[np.searchsorted(ct, .975)],
                             p_mu_gt_0=pm[MU > 0].sum(), p_mu_gt_half=pm[MU > .5].sum()))
        _, sub_L = population(hemi['L'])
        _, sub_R = population(hemi['R'])
        for k, s in enumerate(subs):
            row = dict(roi=gname, subject=s, n_voxels=int((g.subject == s).sum()),
                       **{f'{key}': v for key, v in summarize(subj[k]).items()},
                       **{f'L_{key}': v for key, v in summarize(sub_L[k]).items()},
                       **{f'R_{key}': v for key, v in summarize(sub_R[k]).items()},
                       **{f'nopool_{key}': v for key, v in
                          summarize(grid_post(curves[k])).items()})
            row['posterior'] = ' '.join(f'{x:.4g}' for x in subj[k])
            subj_rows.append(row)
        h = pd.DataFrame([r for r in subj_rows if r['roi'] == gname])
        rLR = stats.spearmanr(h.L_mean, h.R_mean)
        print(f'  mu = {grp_rows[-1]["mu"]:.2f} [{grp_rows[-1]["mu_lo"]:.2f}, '
              f'{grp_rows[-1]["mu_hi"]:.2f}], tau = {grp_rows[-1]["tau"]:.2f} '
              f'[{grp_rows[-1]["tau_lo"]:.2f}, {grp_rows[-1]["tau_hi"]:.2f}]; '
              f'L-R rho = {rLR[0]:.2f} (p = {rLR[1]:.3f})', flush=True)
        grp_rows[-1].update(rho_LR=rLR[0], p_LR=rLR[1])
    out = REPO / 'notes' / 'data'
    pd.DataFrame(subj_rows).to_csv(out / f'hierlambda_{tag}_subjects.tsv', sep='\t',
                                   index=False, float_format='%.4g')
    pd.DataFrame(grp_rows).to_csv(out / f'hierlambda_{tag}_group.tsv', sep='\t',
                                  index=False, float_format='%.4g')
    print(pd.DataFrame(grp_rows).round(3).to_string(index=False))


def recover(n_subjects=35, n_voxels=300, mu=.8, tau=.3, sigma=6., seed=0):
    """Parameter recovery with independent voxels: simulate, refit, compare."""
    rng = np.random.default_rng(seed)
    M = mode_table()
    lam = np.clip(rng.normal(mu, tau, n_subjects), LAM[0], LAM[-1])
    S, SL, SR = [], [], []
    for l in lam:
        j = np.argmin(np.abs(LAM - l))
        k = rng.integers(0, len(X), n_voxels)
        obs = V[M[:, j, k]] + sigma * rng.standard_t(NU, (2, n_voxels))
        S.append(subject_S(obs[0], obs[1], M))
        h = rng.random(n_voxels) < .5
        SL.append(subject_S(obs[0, h], obs[1, h], M))
        SR.append(subject_S(obs[0, ~h], obs[1, ~h], M))
    T, _ = calibrate_temperature(SL, SR)
    post, subj = population([curve(x, T) for x in S])
    est = np.array([summarize(p)['mean'] for p in subj])
    pm, pt = post.sum(1), post.sum(0)
    print(f'true mu {mu}, tau {tau}; recovered mu {(pm * MU).sum():.2f}, '
          f'tau {(pt * TAU).sum():.2f}; T = {T:.1f}; r(true, est) = '
          f'{np.corrcoef(lam, est)[0, 1]:.2f}')
    return lam, est


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tsv', default=str(REPO / 'notes/data/session_modes_voxels.tsv.gz'))
    ap.add_argument('--model', default='lognormal', choices=['lognormal', 'gauss'])
    ap.add_argument('--smoothed', action='store_true')
    ap.add_argument('--groups', nargs='+', default=list(GROUPS))
    ap.add_argument('--recover', action='store_true',
                    help='parameter recovery on simulated, independent voxels')
    a = ap.parse_args()
    if a.recover:
        for nv, sg in ((300, 6.), (300, 10.), (100, 10.)):
            print(f'{nv} voxels, sigma {sg}:')
            recover(n_voxels=nv, sigma=sg)
    else:
        main(a.tsv, a.model, a.smoothed, a.groups)
