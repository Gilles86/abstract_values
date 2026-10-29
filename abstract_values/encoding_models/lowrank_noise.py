"""Low-rank + diagonal decoding noise model with nested-CV rank k and temperature T.

numpy port of ``braincoder.noise.LowRankNoise`` (braincoder ``noise-models``
branch, ~/git/braincoder-noise, commit f4e5a73, 2026-10-20) -- the noise model
retinonumeral decodes with (``decode_fdr.py --noise_model lowrank_cv``). The
abstract_values env is still on the old braincoder, which has no LowRankNoise,
so the algebra is vendored here unchanged. Only ``fit`` differs: it takes
callables for residuals / grid predictions / inner refits instead of a
braincoder model, so it works with both the von Mises basis (WeightFitter) and
the per-voxel log-Gaussian PRF (ParameterFitter) decoders.

    Omega = D (F F' + Psi) D,   p(s | y) propto p(y | s)^(1/T)

F = first k principal factors of the residual *correlation*, Psi = 1 - diag(FF')
floored at psi_floor, an overall scale c and Student-t dof nu by ML. k and T are
chosen jointly by leave-one-group(run)-out CV inside the training data, on the
decoding objective (held-out log posterior of the true grid stimulus), not on
the covariance likelihood.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp

DEFAULT_K_GRID = (0, 1, 2, 5, 10, 20, 40, 80)
DEFAULT_TEMPERATURE_GRID = tuple(np.r_[2.0 ** np.arange(-2, 12.001, 0.25), np.inf])
_DOF_BOUNDS = (2.05, 1e4)


def _standardise(R):
    mu = R.mean(0)
    sd = R.std(0)
    good = sd > 0
    fill = np.median(sd[good]) if good.any() else 1.0
    sd = np.where(good, sd, fill)
    return (R - mu) / sd, sd


def mvt_logpdf(d2, logdet, V, dof):
    if dof is None or not np.isfinite(dof):
        return -0.5 * (V * np.log(2 * np.pi) + logdet + d2)
    return (gammaln((dof + V) / 2.) - gammaln(dof / 2.) - 0.5 * V * np.log(dof * np.pi)
            - 0.5 * logdet - 0.5 * (dof + V) * np.log1p(d2 / dof))


def _fit_scale_dof(d2, logdet, V, dof='fit'):
    d2 = np.asarray(d2, dtype=np.float64)
    n = len(d2)
    if dof is None:
        c = max(d2.mean() / V, 1e-300)
        return c, None, float(mvt_logpdf(d2 / c, logdet + V * np.log(c), V, None).sum())

    def best_c(nu):
        c = max(d2.mean() / V, 1e-300)
        for _ in range(200):
            w = (nu + V) / (nu + d2 / c)
            c_new = (w * d2).sum() / (n * V)
            if abs(c_new - c) <= 1e-10 * c:
                c = c_new
                break
            c = c_new
        return c, mvt_logpdf(d2 / c, logdet + V * np.log(c), V, nu).sum()

    if dof == 'fit':
        res = minimize_scalar(lambda lnu: -best_c(np.exp(lnu))[1],
                              bounds=np.log(_DOF_BOUNDS), method='bounded',
                              options=dict(xatol=1e-3))
        cands = [float(np.exp(res.x)), _DOF_BOUNDS[0], _DOF_BOUNDS[1]]
        nu = cands[int(np.argmax([best_c(c)[1] for c in cands]))]
    else:
        nu = float(dof)
    c, ll = best_c(nu)
    return float(c), nu, float(ll)


class LowRankNoise:
    """Factored Omega = D(FF' + Psi)D; Woodbury algebra, never dense."""

    def __init__(self, scale, factors, psi, dof=None, temperature=1.0):
        self.scale = np.asarray(scale, dtype=np.float64)
        V = len(self.scale)
        factors = np.asarray(factors, dtype=np.float64)
        if factors.ndim == 1:
            factors = factors[:, None]
        if factors.size == 0:
            factors = np.zeros((V, 0))
        self.factors = factors
        self.psi = np.asarray(psi, dtype=np.float64)
        self.dof = None if dof is None or not np.isfinite(dof) else float(dof)
        self.temperature = float(temperature)
        self.cv_scores_ = None
        F, psi = self.factors, self.psi
        k = F.shape[1]
        FtPinv = (F / psi[:, None]).T
        Lm = np.linalg.cholesky(np.eye(k) + FtPinv @ F)
        self._A = np.linalg.solve(Lm, FtPinv) if k > 0 else np.zeros((0, V))
        self._sqrt_inv_psi = 1.0 / np.sqrt(psi)
        self.logdet = float(2 * np.log(self.scale).sum() + np.log(psi).sum()
                            + (2 * np.log(np.diag(Lm)).sum() if k > 0 else 0.0))

    @property
    def k(self):
        return self.factors.shape[1]

    @property
    def n_voxels(self):
        return len(self.scale)

    def _components(self, X):
        Z = np.asarray(X, dtype=np.float64) / self.scale
        return Z * self._sqrt_inv_psi, Z @ self._A.T

    def mahalanobis(self, R):
        G, H = self._components(R)
        return np.clip((G ** 2).sum(1) - (H ** 2).sum(1), 0, None)

    def grid_log_likelihood(self, Y, P):
        """(n_trials, n_stimuli) log-likelihood; P = (n_stimuli, V) predictions."""
        Gy, Hy = self._components(Y)
        Gp, Hp = self._components(P)
        d2 = ((Gy ** 2).sum(1) - (Hy ** 2).sum(1))[:, None] \
            + ((Gp ** 2).sum(1) - (Hp ** 2).sum(1))[None, :] \
            - 2 * (Gy @ Gp.T - Hy @ Hp.T)
        return mvt_logpdf(np.clip(d2, 0, None), self.logdet, self.n_voxels, self.dof)

    def log_posterior(self, Y, P):
        """Tempered, normalised log posterior on the grid (flat prior)."""
        ll = self.grid_log_likelihood(Y, P)
        if np.isinf(self.temperature):
            return np.full_like(ll, -np.log(ll.shape[1]))
        a = ll / self.temperature
        return a - logsumexp(a, axis=1, keepdims=True)

    @staticmethod
    def _svd(R):
        Z, sd = _standardise(R)
        _, s, Vt = np.linalg.svd(Z, full_matrices=False)
        return dict(sd=sd, s=s, Vt=Vt, n=len(R))

    @classmethod
    def from_residuals(cls, R, k=40, dof='fit', psi_floor=0.02, temperature=1.0, _svd=None):
        R = np.asarray(R, dtype=np.float64)
        n, V = R.shape
        k = int(min(k, max(min(n, V) - 1, 0)))
        svd = cls._svd(R) if _svd is None else _svd
        lam = svd['s'][:k] ** 2 / svd['n']
        F = (svd['Vt'][:k] * np.sqrt(lam)[:, None]).T
        psi = np.clip(1.0 - (F ** 2).sum(1), psi_floor, None)
        unit = cls(svd['sd'], F, psi)
        c, nu, ll = _fit_scale_dof(unit.mahalanobis(R), unit.logdet, V, dof=dof)
        out = cls(svd['sd'] * np.sqrt(c), F, psi, dof=nu, temperature=temperature)
        out.train_loglik_ = ll
        return out


def nearest_grid_index(x, grid):
    x = np.asarray(x, dtype=np.float64).ravel()
    grid = np.asarray(grid, dtype=np.float64).ravel()
    return np.abs(x[:, None] - grid[None, :]).argmin(1)


def fit_lowrank_cv(Y, residuals_all, true_idx, groups, inner_fold,
                   k_grid=DEFAULT_K_GRID, temperature_grid=DEFAULT_TEMPERATURE_GRID,
                   dof='fit', psi_floor=0.02):
    """Choose (k, T) by leave-one-group-out CV inside the training data.

    Y               (n, V) training data of the selected voxels
    residuals_all   (n, V) training residuals under the outer-fold parameters
    true_idx        (n,) grid index of each training trial's true stimulus
    groups          (n,) inner CV group (run) label
    inner_fold(g)   -> (R_tr, P): inner-training residuals under parameters fitted
                    without group g, and the (S, V) grid predictions of those
                    parameters (retinonumeral ``--inner_params loro``)

    Returns (LowRankNoise fitted on all residuals with the chosen k and T,
    cv table DataFrame [k x T], mean log score relative to a flat posterior).
    """
    Y = np.asarray(Y, dtype=np.float64)
    n, V = Y.shape
    groups = np.asarray(groups)
    ug = pd.unique(groups)
    n_tr_min = min((groups != g).sum() for g in ug)
    ks = sorted({int(k) for k in k_grid if k < min(n_tr_min, V)})
    Ts = np.asarray(temperature_grid, dtype=float)
    scores = np.zeros((len(ks), len(Ts)))
    S = None
    for g in ug:
        te = np.where(groups == g)[0]
        R_tr, P = inner_fold(g)
        S = P.shape[0]
        svd = LowRankNoise._svd(R_tr)
        for ki, kk in enumerate(ks):
            nm = LowRankNoise.from_residuals(R_tr, k=kk, dof=dof, psi_floor=psi_floor, _svd=svd)
            ll = nm.grid_log_likelihood(Y[te], P)
            ll_true = ll[np.arange(len(te)), true_idx[te]]
            for ti, T in enumerate(Ts):
                if np.isinf(T):
                    scores[ki, ti] += -len(te) * np.log(S)
                else:
                    a = ll / T
                    scores[ki, ti] += (ll_true / T - logsumexp(a, axis=1)).sum()
    rel = scores / n + np.log(S)
    cv = pd.DataFrame(rel, index=pd.Index(ks, name='k'), columns=pd.Index(Ts, name='temperature'))
    ki, ti = np.unravel_index(np.nanargmax(rel), rel.shape)
    out = LowRankNoise.from_residuals(residuals_all, k=ks[ki], dof=dof, psi_floor=psi_floor,
                                      temperature=float(Ts[ti]))
    out.cv_scores_ = cv
    finite = Ts[np.isfinite(Ts)]
    if np.isfinite(Ts[ti]) and Ts[ti] == finite.max():
        warnings.warn('Chosen temperature is the largest finite grid value.', RuntimeWarning)
    return out, cv
