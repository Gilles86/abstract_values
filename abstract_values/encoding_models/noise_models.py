"""Prototype noise models and calibration metrics for encoding-model decoding.

Methods study (notes/decoding_noise_model_calibration.md). Nothing here is used
by the production pipeline (`decode_fdr.py`). The general pieces are written so
they could move into braincoder unchanged:

- ``recentre_runs``           per-run re-centring of betas (unsupervised)
- ``lowrank_corr_cov``        diag + k principal factors of the residual correlation
- ``lw_corr_cov``             Ledoit-Wolf shrinkage of the correlation towards I
- ``choose_rank``             k by run-grouped CV likelihood
- ``fit_scale_dof``           (c, nu) of a Student-t with scale c*S, by ML
- ``safe_cholesky_np``        numpy Cholesky with adaptive relative jitter
- ``WhitenedMVT``             grid log-likelihood via ||a||^2 + ||b||^2 - 2ab
                              (whiten data and grid predictions once)
- ``tempered_posterior``      p(s|y) propto L(s)^(1/T)
- ``choose_temperature``      T by inner held-out log score
- ``calibration_metrics``     log score, randomised PIT, HPD level, moments

Everything is float64 numpy; residual matrices are (n_trials, n_voxels).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp

NU_GRID = np.array([2.5, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100, 300], dtype=float)
# 1 .. 4096 in steps of 2^(1/4), plus inf (= flat posterior). The original
# grid stopped at 64 and the debug run hit that cap (notes 2.0).
T_GRID = np.r_[2.0 ** np.arange(0, 12.01, 0.25), np.inf]
RANKS = (1, 2, 5, 10, 20, 40, 80, 160)  # 80/160 added: k=40 was always chosen (notes 3.4)


# --------------------------------------------------------------------------
# data transforms
# --------------------------------------------------------------------------

def recentre_runs(Y, run_ids, train_mask):
    """Replace each run's per-voxel mean by the training grand mean.

    y_r <- y_r - mean_r(y_r) + mean_{train}(y). Uses no labels; the held-out
    run's own mean is estimated from its own betas. Keeps the baseline of
    parameters fitted on raw (un-centred) betas valid.
    """
    Y = np.asarray(Y, dtype=float).copy()
    grand = Y[train_mask].mean(0)
    for r in np.unique(run_ids):
        m = run_ids == r
        Y[m] = Y[m] - Y[m].mean(0) + grand
    return Y


# --------------------------------------------------------------------------
# covariance estimators
# --------------------------------------------------------------------------

def _standardise(R):
    mu = R.mean(0)
    sd = R.std(0)
    sd = np.where(sd > 0, sd, np.nanmedian(sd[sd > 0]))
    return (R - mu) / sd, sd


def lowrank_corr_cov(R, k, psi_floor=0.02, svd=None):
    """Diag + k principal factors of the correlation of R, back on the data scale.

    C_k = W_k' diag(lam_k) W_k + diag(psi), psi = 1 - diag(W_k' lam W_k),
    floored at psi_floor; Omega = D C_k D, D = diag(sd). One principal-factor
    step (no iteration) -- closed form, so k can be chosen by CV cheaply.
    ``svd`` may pass a precomputed (Z, sd, s, Vt) of the same R.
    """
    if svd is None:
        Z, sd = _standardise(R)
        _, s, Vt = np.linalg.svd(Z, full_matrices=False)
    else:
        Z, sd, s, Vt = svd
    n = Z.shape[0]
    lam = s[:k] ** 2 / n
    W = Vt[:k]
    common = (W ** 2 * lam[:, None]).sum(0)
    psi = np.clip(1.0 - common, psi_floor, None)
    C = (W.T * lam) @ W
    C[np.diag_indices_from(C)] += psi
    return C * sd[:, None] * sd[None, :]


def svd_of(R):
    Z, sd = _standardise(R)
    _, s, Vt = np.linalg.svd(Z, full_matrices=False)
    return Z, sd, s, Vt


def lw_corr_cov(R):
    """Ledoit-Wolf shrinkage of the correlation matrix towards identity.

    Equivalent to shrinking the covariance towards its diagonal. Returns
    (Omega, shrinkage).
    """
    from sklearn.covariance import ledoit_wolf
    Z, sd = _standardise(R)
    C, shrink = ledoit_wolf(Z, assume_centered=True)
    return C * sd[:, None] * sd[None, :], float(shrink)


def gauss_loglik(R, Omega):
    """Sum of N(0, Omega) log-densities of rows of R."""
    L = safe_cholesky_np(Omega)
    z = solve_triangular(L, R.T, lower=True)
    V = R.shape[1]
    logdet = 2 * np.log(np.diag(L)).sum()
    return float(-0.5 * ((z ** 2).sum() + R.shape[0] * (logdet + V * np.log(2 * np.pi))))


def choose_rank(R, groups, ranks=RANKS):
    """k for ``lowrank_corr_cov`` by run-grouped CV Gaussian likelihood.

    ``groups``: array of group labels per row of R. Returns (k_best, scores).
    """
    scores = {k: 0.0 for k in ranks}
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        svd = svd_of(R[tr])
        mu = R[tr].mean(0)
        for k in ranks:
            if k >= tr.sum():
                scores[k] = -np.inf
                continue
            scores[k] += gauss_loglik(R[te] - mu, lowrank_corr_cov(R[tr], k, svd=svd))
    k_best = max(scores, key=scores.get)
    return k_best, scores


# --------------------------------------------------------------------------
# Student-t scale and dof
# --------------------------------------------------------------------------

def safe_cholesky_np(S, max_tries=8):
    S = 0.5 * (S + S.T)
    jit = 0.0
    base = float(np.mean(np.diag(S)))
    for i in range(max_tries):
        try:
            return np.linalg.cholesky(S + jit * np.eye(len(S)))
        except np.linalg.LinAlgError:
            jit = base * (1e-6 * 10 ** i)
    raise np.linalg.LinAlgError("safe_cholesky_np failed")


def mvt_logpdf_from_d2(d2, logdet, V, nu):
    return (gammaln((nu + V) / 2) - gammaln(nu / 2) - 0.5 * V * np.log(nu * np.pi)
            - 0.5 * logdet - 0.5 * (nu + V) * np.log1p(d2 / nu))


def fit_scale_dof(R, S, nu_grid=NU_GRID):
    """ML (c, nu) for rows of R ~ MVT(0, c*S, nu). Returns (c, nu, loglik)."""
    L = safe_cholesky_np(S)
    d2 = (solve_triangular(L, R.T, lower=True) ** 2).sum(0)
    logdet = 2 * np.log(np.diag(L)).sum()
    V = R.shape[1]
    best = (None, None, -np.inf)
    for nu in nu_grid:
        f = lambda lc: -mvt_logpdf_from_d2(d2 / np.exp(lc), logdet + V * lc, V, nu).sum()
        res = minimize_scalar(f, bounds=(-6, 6), method="bounded")
        if -res.fun > best[2]:
            best = (float(np.exp(res.x)), float(nu), float(-res.fun))
    return best


# --------------------------------------------------------------------------
# likelihood on a stimulus grid
# --------------------------------------------------------------------------

class WhitenedMVT:
    """Student-t log-likelihood of data rows against grid predictions.

    d2(t, s) = ||L^-1 y_t||^2 + ||L^-1 mu_s||^2 - 2 (L^-1 y_t).(L^-1 mu_s):
    data and grid predictions are whitened once, so a full (T x S) table costs
    one matrix product instead of T*S triangular solves.
    """

    def __init__(self, Omega, nu):
        self.L = safe_cholesky_np(Omega)
        self.logdet = 2 * np.log(np.diag(self.L)).sum()
        self.V = len(Omega)
        self.nu = nu

    def whiten(self, A):
        return solve_triangular(self.L, np.asarray(A, float).T, lower=True).T

    def d2(self, Yw, Pw):
        return np.clip((Yw ** 2).sum(1)[:, None] + (Pw ** 2).sum(1)[None, :]
                       - 2 * Yw @ Pw.T, 0, None)

    def loglik(self, Yw, Pw):
        d2 = self.d2(Yw, Pw)
        if self.nu is None or not np.isfinite(self.nu):
            return -0.5 * d2 - 0.5 * self.logdet
        return mvt_logpdf_from_d2(d2, self.logdet, self.V, self.nu)


def tempered_posterior(ll, T=1.0):
    """Row-wise p(s|y) propto exp(ll/T), flat prior on the grid."""
    a = ll / T
    return np.exp(a - logsumexp(a, axis=1, keepdims=True))


def choose_temperature(ll_list, true_idx_list, t_grid=T_GRID):
    """T maximising the mean joint log score over inner held-out trials."""
    ll = np.concatenate(ll_list)
    ti = np.concatenate(true_idx_list)
    scores = []
    for T in t_grid:
        a = ll / T
        lp = a[np.arange(len(a)), ti] - logsumexp(a, axis=1)
        scores.append(lp.mean())
    scores = np.array(scores)
    return float(t_grid[int(np.argmax(scores))]), scores


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def _hpd_level(p, true_i, u):
    """Randomised HPD level of the truth for rows of p (T x K)."""
    pt = p[np.arange(len(p)), true_i][:, None]
    above = (p * (p > pt)).sum(1)
    tie = (p * (p == pt)).sum(1)
    return above + u * tie


def _pit(p, true_i, u):
    cdf = np.cumsum(p, 1)
    below = np.where(true_i > 0, cdf[np.arange(len(p)), np.maximum(true_i - 1, 0)], 0.0)
    return below + u * p[np.arange(len(p)), true_i]


def calibration_metrics(post, n1, n2, stim_log, rng, cue=None):
    """Per-trial calibration metrics of a posterior on a 2-D (log n1, log n2) grid.

    post: (T, S) normalised. stim_log: (S, 2). n1, n2: integer arrays.
    """
    vals = np.unique(np.round(np.exp(stim_log[:, 0])).astype(int))
    K = len(vals)
    i1 = np.searchsorted(vals, np.round(np.exp(stim_log[:, 0])).astype(int))
    i2 = np.searchsorted(vals, np.round(np.exp(stim_log[:, 1])).astype(int))
    Tn = len(post)
    M1 = np.zeros((Tn, K)); M2 = np.zeros((Tn, K))
    np.add.at(M1.T, i1, post.T)
    np.add.at(M2.T, i2, post.T)
    t1 = np.searchsorted(vals, n1); t2 = np.searchsorted(vals, n2)
    tj = np.array([np.where((i1 == a) & (i2 == b))[0][0] for a, b in zip(t1, t2)])
    u = rng.uniform(size=(5, Tn))
    eps = 1e-300
    out = dict(
        logp_joint=np.log(post[np.arange(Tn), tj] + eps),
        logp_n1=np.log(M1[np.arange(Tn), t1] + eps),
        logp_n2=np.log(M2[np.arange(Tn), t2] + eps),
        pit_n1=_pit(M1, t1, u[0]), pit_n2=_pit(M2, t2, u[1]),
        hpd_n1=_hpd_level(M1, t1, u[2]), hpd_n2=_hpd_level(M2, t2, u[3]),
        hpd_joint=_hpd_level(post, tj, u[4]),
        mean_n1=M1 @ vals, mean_n2=M2 @ vals,
        sd_n1=np.sqrt(np.clip(M1 @ vals ** 2 - (M1 @ vals) ** 2, 0, None)),
        sd_n2=np.sqrt(np.clip(M2 @ vals ** 2 - (M2 @ vals) ** 2, 0, None)),
    )
    df = pd.DataFrame(out)
    if cue is not None:
        left = np.asarray(cue) == "left"
        for k in ["logp", "pit", "hpd", "mean", "sd"]:
            df[f"{k}_cued"] = np.where(left, df[f"{k}_n1"], df[f"{k}_n2"])
    return df, tj


def summarise(df):
    """Group-level summary of a per-trial metrics frame (one participant x ROI)."""
    pit = np.concatenate([df.pit_n1.values, df.pit_n2.values])
    hm = np.concatenate([df.hpd_n1.values, df.hpd_n2.values])
    r1 = np.corrcoef(df.mean_n1, df.n1)[0, 1]
    r2 = np.corrcoef(df.mean_n2, df.n2)[0, 1]
    return pd.Series(dict(
        n=len(df),
        logp_joint=df.logp_joint.mean(), logp_marg=0.5 * (df.logp_n1.mean() + df.logp_n2.mean()),
        pit_sd=pit.std(), pit_tail05=np.mean((pit < .025) | (pit > .975)),
        cov50_marg=np.mean(hm < .5), cov95_marg=np.mean(hm < .95),
        cov50_joint=np.mean(df.hpd_joint < .5), cov95_joint=np.mean(df.hpd_joint < .95),
        r_mean=0.5 * (r1 + r2),
        mae=0.5 * (np.abs(df.mean_n1 - df.n1).mean() + np.abs(df.mean_n2 - df.n2).mean()),
        post_sd=0.5 * (df.sd_n1.mean() + df.sd_n2.mean()),
    ))
