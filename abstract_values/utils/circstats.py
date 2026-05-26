"""Circular statistics utilities for the orientation axis.

The orientations in this project are π-periodic (axial), so when you
compute correlations or summary statistics involving orientation, use
the helpers here rather than ``np.corrcoef`` / ``np.mean`` directly.

The doubled-angle trick is used throughout: a π-periodic angle θ is
mapped to a 2π-periodic angle 2θ before taking sin/cos, so circular
mean and SD inherit the same arithmetic as standard circular statistics
on the full circle.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


PERIOD = np.pi
SCALE = 2 * np.pi / PERIOD          # 2 — doubled-angle factor


def circular_mean(theta_rad: np.ndarray) -> float:
    """Population circular mean on a π-periodic axis (radians)."""
    s = np.sin(SCALE * theta_rad).mean()
    c = np.cos(SCALE * theta_rad).mean()
    return (np.arctan2(s, c) / SCALE) % PERIOD


def circular_sd(theta_rad: np.ndarray) -> float:
    """Population circular SD on a π-periodic axis (radians)."""
    s = np.sin(SCALE * theta_rad).mean()
    c = np.cos(SCALE * theta_rad).mean()
    R = np.clip(np.sqrt(s * s + c * c), 1e-12, 1.0)
    return np.sqrt(-2.0 * np.log(R)) / SCALE


def circular_distance(a_rad: np.ndarray, b_rad: np.ndarray) -> np.ndarray:
    """Signed minimal distance from ``b`` to ``a`` on a π-periodic axis
    (returned in (-π/2, π/2])."""
    d = (a_rad - b_rad) % PERIOD
    return np.where(d > PERIOD / 2, d - PERIOD, d)


def circular_linear_r2(theta_deg: np.ndarray, x: np.ndarray, k: int = 2) -> float:
    """Mardia's circular-linear correlation squared, harmonic ``k``.

    Tests whether the linear variable ``x`` is modulated by the
    π-periodic variable ``theta_deg`` (in degrees) at the harmonic ``k``
    (i.e., period 180°/k).

        r² = (r_xc² + r_xs² − 2 r_xc r_xs r_cs) / (1 − r_cs²)

    where r_xc = corr(x, cos kθ), r_xs = corr(x, sin kθ),
    r_cs = corr(cos kθ, sin kθ). Common choices:

    - ``k=2``: one full cycle per 180° (single peak / single trough).
    - ``k=4``: two cycles per 180° (e.g., a W-shape with 3 dips at
      0°/90°/180° is well-fit here — it's the right basis for
      cardinal/oblique modulations).
    - ``k=6``, ``k=8``: higher-frequency periodicities.
    """
    theta_rad = np.deg2rad(np.asarray(theta_deg, dtype=float)) * k
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    r_xc = np.corrcoef(x, c)[0, 1]
    r_xs = np.corrcoef(x, s)[0, 1]
    r_cs = np.corrcoef(c, s)[0, 1]
    return float((r_xc**2 + r_xs**2 - 2 * r_xc * r_xs * r_cs)
                 / max(1 - r_cs**2, 1e-12))


def harmonic_table(theta_deg: np.ndarray, x: np.ndarray,
                    ks: Iterable[int] = (2, 4, 6, 8)) -> dict[int, float]:
    """Run :func:`circular_linear_r2` for several harmonics and return
    a dict ``{k: r²}``. Useful when you don't know a-priori which
    harmonic carries the signal (linear Pearson r is uninformative for
    non-monotone periodic structure)."""
    return {int(k): circular_linear_r2(theta_deg, x, k=k) for k in ks}
