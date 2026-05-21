"""Project-local R² mixture helpers (logit-Gaussian, 2 components).

Inlined from braincoder.utils.stats to dodge the version mismatch between
the (newer) local braincoder install and the (older) cluster checkout
that ships with the abstract_values env. Pure NumPy / sklearn / scipy —
no braincoder dependency. Functions are byte-identical to the upstream
braincoder versions (commit f692f60 at time of vendoring); re-vendor
from braincoder if upstream changes.

Provides:
    fit_r2_mixture(r2)               -> dict of mixture params
    r2_fdr_threshold(fit, alpha)     -> R² threshold for tail-FDR ≤ α
    r2_posterior_signal(r2, fit)     -> P(signal | r²) per voxel
    r2_p_signal_threshold(fit, p)    -> smallest R² with P(signal) ≥ p
"""
from __future__ import annotations

import numpy as np


def _logit(r2):
    return np.log(r2 / (1.0 - r2))


def _inv_logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_r2_mixture(r2, n_init: int = 8, max_iter: int = 500, seed: int = 0) -> dict:
    """Fit a 2-component Gaussian mixture on ``logit(R²) = log(R²/(1-R²))``.

    The logit pulls apart the low-R² region so the noise and signal
    components don't overlap on a near-singular support, and Gaussians
    (unlike Betas) can't go pathologically U-shaped. Returns a dict with
    ``noise_mu``/``signal_mu`` etc. on the logit scale, plus
    ``noise_mean_r2`` / ``signal_mean_r2`` as the inv-logit of the means.

    Raises ``ValueError`` if fewer than 50 finite R² values lie in (0, 0.99).
    """
    from sklearn.mixture import GaussianMixture
    r2 = np.asarray(r2, dtype=float).ravel()
    r2 = r2[np.isfinite(r2) & (r2 > 0) & (r2 < 0.99)]
    if len(r2) < 50:
        raise ValueError(
            f"need ≥50 finite R² values in (0, 0.99); got {len(r2)}")
    r2 = np.clip(r2, 1e-6, 1 - 1e-6)
    z = _logit(r2).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, n_init=n_init,
                          max_iter=max_iter, random_state=seed,
                          reg_covar=1e-6).fit(z)
    means = gmm.means_.flatten()
    sds = np.sqrt(gmm.covariances_.flatten())
    ws = gmm.weights_
    n_idx, s_idx = int(np.argmin(means)), int(np.argmax(means))
    return {
        'mixture':        'gmm_logit',
        'noise_mu':       float(means[n_idx]),
        'noise_sigma':    float(sds[n_idx]),
        'noise_weight':   float(ws[n_idx]),
        'noise_mean_r2':  float(_inv_logit(means[n_idx])),
        'signal_mu':      float(means[s_idx]),
        'signal_sigma':   float(sds[s_idx]),
        'signal_weight':  float(ws[s_idx]),
        'signal_mean_r2': float(_inv_logit(means[s_idx])),
        'log_likelihood': float(gmm.score(z) * len(z)),
        'n_voxels':       int(len(r2)),
    }


def r2_fdr_threshold(r2_or_fit, alpha: float = 0.05, n_grid: int = 4000) -> float:
    """R² threshold at which the 2-component mixture's tail-FDR is ≤ α.

        FDR(t) = w_n · P(R² > t | noise) /
                 [w_n · P(R² > t | noise) + w_s · P(R² > t | signal)]
    """
    from scipy.stats import norm
    fit = r2_or_fit if isinstance(r2_or_fit, dict) else fit_r2_mixture(r2_or_fit)
    z_grid = np.linspace(fit['noise_mu'] - 5 * fit['noise_sigma'],
                          fit['signal_mu'] + 8 * fit['signal_sigma'],
                          n_grid)
    sf_n = 1.0 - norm.cdf(z_grid, fit['noise_mu'],  fit['noise_sigma'])
    sf_s = 1.0 - norm.cdf(z_grid, fit['signal_mu'], fit['signal_sigma'])
    denom = fit['noise_weight'] * sf_n + fit['signal_weight'] * sf_s
    fdr = np.where(denom > 1e-12,
                   fit['noise_weight'] * sf_n / np.maximum(denom, 1e-12),
                   1.0)
    hits = np.where(fdr <= alpha)[0]
    if len(hits) == 0:
        return float('inf')
    return float(_inv_logit(z_grid[hits[0]]))


def r2_posterior_signal(r2, fit: dict) -> np.ndarray:
    """Posterior P(signal | r²) per voxel from a fitted 2-component mixture."""
    from scipy.stats import norm
    r2 = np.asarray(r2, dtype=float).ravel()
    out = np.zeros_like(r2)
    valid = np.isfinite(r2) & (r2 > 0) & (r2 < 1)
    if not valid.any():
        return out
    r2_safe = np.clip(r2[valid], 1e-6, 1 - 1e-6)
    z = _logit(r2_safe)
    p_n = fit['noise_weight'] * norm.pdf(z, fit['noise_mu'], fit['noise_sigma'])
    p_s = fit['signal_weight'] * norm.pdf(z, fit['signal_mu'], fit['signal_sigma'])
    denom = p_n + p_s
    out[valid] = np.where(denom > 0, p_s / np.maximum(denom, 1e-300), 0.0)
    return out


def r2_p_signal_threshold(r2_or_fit, p: float = 0.5, n_grid: int = 4000) -> float:
    """Smallest R² with P(signal | R²) ≥ ``p``. ``inf`` if never reached."""
    fit = r2_or_fit if isinstance(r2_or_fit, dict) else fit_r2_mixture(r2_or_fit)
    z_grid = np.linspace(fit['noise_mu'] - 5 * fit['noise_sigma'],
                          fit['signal_mu'] + 8 * fit['signal_sigma'],
                          n_grid)
    r2_grid = _inv_logit(z_grid)
    p_signal = r2_posterior_signal(r2_grid, fit)
    hits = np.where(p_signal >= p)[0]
    return float('inf') if len(hits) == 0 else float(r2_grid[hits[0]])
