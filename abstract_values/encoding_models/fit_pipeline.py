"""Generic aPRF fitting pipeline driven by :data:`MODEL_SPECS`.

One :func:`fit_one_model` function replaces the 5 large if-elif
branches that used to live in ``fit_aprf.py`` and ``fit_aprf_cv.py``.
A new model variant becomes ~5 lines in :mod:`model_specs` instead of
~70 lines of duplicated fitting boilerplate.

Pipeline (per variant):

  1. Instantiate the PRF class with the spec's kwargs.
  2. If ``spec.grid_dims == 0``: warm-start from ``spec.warm_start_from``
     (recurses through :func:`fit_one_model` to fit the warm-start
     spec first, then builds ``init_pars`` via
     :func:`build_warm_start_init`).
     Otherwise: build a (1D or 2D) parameter grid and run
     ``ParameterFitter.fit_grid`` + ``refine_baseline_and_amplitude``
     to get the initial guesses.
  3. Gradient-descent refinement via Adam.
  4. Compute R² on the training data.

Returns ``(pars_df, r2_series)`` — caller is responsible for masking
back into NIfTI and writing to disk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from braincoder.optimize import ParameterFitter
from braincoder.utils import get_rsq

from abstract_values.encoding_models.model_specs import (
    MODEL_SPECS, ModelSpec, build_warm_start_init, get_spec,
)


def fit_one_model(spec: ModelSpec,
                    data: pd.DataFrame,
                    paradigm: pd.DataFrame,
                    value_min: float,
                    value_max: float,
                    n_iterations: int = 1000,
                    debug: bool = False,
                    log_prefix: str = '  '):
    """Run the full fit for one model variant.

    Parameters
    ----------
    spec : ModelSpec
        Variant config — see :mod:`model_specs`.
    data : DataFrame (n_trials, n_voxels)
        Single-trial betas.
    paradigm : DataFrame
        Must contain ``x``; if ``spec.needs_session`` also ``session``.
    value_min, value_max : float
        Used to construct the grid-search ranges.
    n_iterations : int
        Adam iterations on the final refinement.
    debug : bool
        Smaller grid, fewer iterations for fast local smoke tests.
    log_prefix : str
        Prepended to print() lines so nested warm-start calls can be
        visually distinguished.

    Returns
    -------
    pars : DataFrame (n_voxels, n_params)
        Final fitted parameters with one column per
        ``spec.save_params``.
    r2 : Series (n_voxels,)
        Training R² per voxel.
    """
    # ── Null-model special case ─────────────────────────────────────────────
    # No parameters, no fit. Prediction = the per-voxel training mean,
    # broadcast across all trials. R² on training set is 0 by definition
    # of "mean predictor" (SS_res == SS_total). The value of this model
    # is as a cvR² baseline — see ``fit_aprf_cv.py`` for the cv path.
    if spec.is_null:
        mean_per_voxel = data.mean(axis=0)
        pred = pd.DataFrame(np.tile(mean_per_voxel.values,
                                       (len(data), 1)),
                             index=data.index, columns=data.columns)
        r2 = get_rsq(data, pred)
        print(f"{log_prefix}[null] mean training R² = "
              f"{float(r2.mean()):.4f}  (≈ 0 by construction)")
        return pd.DataFrame(index=data.columns), r2

    model = spec.cls(**spec.cls_kwargs)
    fitter = ParameterFitter(model, data, paradigm)

    n_iter_eff = 50 if debug else n_iterations

    if spec.grid_dims == 0:
        # Warm-start from another spec — fit that first, then duplicate
        # the warm-start fit's params into this spec's parameter shape
        # and run only the Adam refinement.
        if not spec.warm_start_from:
            raise ValueError(
                f"Spec {spec.name!r} has grid_dims=0 but no "
                "warm_start_from — cannot fit.")
        warm = get_spec(spec.warm_start_from)
        print(f"{log_prefix}[warm-start] fitting {warm.name} first…")
        warm_pars, _ = fit_one_model(
            warm, data, paradigm, value_min, value_max,
            n_iterations=n_iter_eff // 2,        # half iters on warm-start
            debug=debug,
            log_prefix=log_prefix + "  ",
        )
        init_pars = build_warm_start_init(spec, warm_pars)
        print(f"{log_prefix}[{spec.name}] descent ({n_iter_eff} iters)…")
        pars = fitter.fit(max_n_iterations=n_iter_eff, init_pars=init_pars)

    else:
        # Grid search + refine + Adam.
        n_mode = 8  if debug else 15
        n_fwhm = 6  if debug else 10
        # Non-shift models get a finer grid on the single mode — same
        # default as the legacy code so behaviour is preserved.
        if spec.grid_dims == 1 and not spec.needs_session:
            n_mode = 12 if debug else 20
            n_fwhm = 8  if debug else 15

        modes      = np.linspace(value_min, value_max, n_mode).astype(np.float32)
        fwhms      = np.linspace(1.0, value_max - value_min, n_fwhm).astype(np.float32)
        amplitudes = np.array([1.0], dtype=np.float32)
        baselines  = np.array([0.0], dtype=np.float32)

        if spec.grid_dims == 1:
            n_points = n_mode * n_fwhm
            print(f"{log_prefix}[{spec.name}] grid {n_mode}×{n_fwhm} "
                  f"= {n_points} points…")
            grid_pars = fitter.fit_grid(modes, fwhms,
                                          amplitudes, baselines,
                                          use_correlation_cost=True)
        else:                                       # grid_dims == 2
            n_points = n_mode * n_mode * n_fwhm
            print(f"{log_prefix}[{spec.name}] grid "
                  f"{n_mode}×{n_mode}×{n_fwhm} = {n_points} points…")
            grid_pars = fitter.fit_grid(modes, modes, fwhms,
                                          amplitudes, baselines,
                                          use_correlation_cost=True)
        grid_pars = fitter.refine_baseline_and_amplitude(grid_pars)

        print(f"{log_prefix}[{spec.name}] descent ({n_iter_eff} iters)…")
        pars = fitter.fit(max_n_iterations=n_iter_eff, init_pars=grid_pars)

    pred = model.predict(parameters=pars, paradigm=paradigm)
    r2   = get_rsq(data, pred)
    print(f"{log_prefix}[{spec.name}] mean R² = {float(r2.mean()):.4f}")

    # Keep only the spec's declared save_params + ensure column order
    # is stable. Any extra columns from the fit are dropped (none
    # exist in practice; declaring this explicit guards future model
    # classes that might add hidden state).
    pars = pars[spec.save_params].copy()
    return pars, r2
