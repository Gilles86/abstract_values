"""Declarative registry of aPRF model variants.

Replaces 5 large if-elif blocks in ``fit_aprf.py`` and ``fit_aprf_cv.py``
(plus copies in ``decode_value.py`` etc.) with one spec table per
variant. Adding a new variant becomes ~5 lines of declarative
configuration instead of ~70 lines of duplicated fitting boilerplate.

A :class:`ModelSpec` carries everything the generic fit pipeline
(see :func:`fit_one_model`) needs to know:

  - Which PRF class to instantiate (and its kwargs)
  - Whether the paradigm needs a 'session' column
  - What the output directory is for full + CV outputs
  - Which parameter NIfTIs to save (model-class-specific)
  - Whether to grid-search 1D or 2D in modes
  - Optional warm-start chain (e.g. fwhm-shift warm-starts from
    session-shift; fully-shifted does too) — the new params not in
    the warm-start fit get duplicated from existing ones.

The registry itself lives in :data:`MODEL_SPECS`. Look up by short
``model_type`` tag (``'standard'``, ``'session-shift'``,
``'fwhm-shift'``, ``'fully-shifted'``, ``'gaussian'``,
``'gauss-session-shift'``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from braincoder.models import LogGaussianPRF, EncodingModel
from abstract_values.encoding_models.models import (
    SessionShiftedLogGaussianPRF,
    FwhmShiftedLogGaussianPRF,
    FullyShiftedLogGaussianPRF,
    GaussianValuePRF,
    SessionShiftedGaussianValuePRF,
)


@dataclass
class ModelSpec:
    """All the per-variant config the generic fit pipeline needs.

    Attributes
    ----------
    name : str
        Short CLI-friendly tag (e.g. ``'session-shift'``).
    cls : Type[EncodingModel]
        PRF class to instantiate.
    cls_kwargs : dict
        Constructor kwargs passed to ``cls(**cls_kwargs)``.
    needs_session : bool
        If True, the paradigm DataFrame must carry a ``session`` column
        and the model's ``_session_predict`` reads it.
    save_params : list[str]
        Parameter labels (matching ``cls.parameter_labels``) whose
        per-voxel maps are written out as NIfTIs.
    out_subdir : str
        Subdir under ``derivatives/encoding_models/`` for full fits.
    cv_out_subdir : str
        Subdir for CV outputs (often differs by ``.cv`` suffix).
    grid_dims : int
        1 = single-mode grid (``fit_grid(modes, fwhms, ...)``)
        2 = 2-mode grid for shift models
            (``fit_grid(modes, modes, fwhms, ...)``)
        0 = no grid — must warm-start.
    warm_start_from : str
        Spec name to warm-start from (e.g. ``'session-shift'``). Empty
        string means fit from a grid search directly.
    warm_start_dup : dict[str, str]
        Mapping ``new_param → existing_param`` describing how to
        initialize the new variant's parameters from the warm-start
        fit's parameters. E.g. for ``fwhm-shift`` warm-starting from
        ``session-shift``: ``{'fwhm_1': 'fwhm', 'fwhm_2': 'fwhm'}``.
        Parameters that already share a name across the two specs
        (e.g. ``mode_1``, ``mode_2``) are mapped automatically and
        don't need an entry here.
    """
    name: str
    cls: Type[EncodingModel]
    cls_kwargs: dict = field(default_factory=lambda: {
        'allow_neg_amplitudes': False})
    needs_session: bool = False
    save_params: list = field(default_factory=list)
    out_subdir: str = ''
    cv_out_subdir: str = ''
    grid_dims: int = 1
    warm_start_from: str = ''
    warm_start_dup: dict = field(default_factory=dict)

    # Some PRF classes (notably LogGaussianPRF) need an extra kwarg
    # describing the parameterisation. Captured here so fit_one_model
    # doesn't need a special case.
    parameterisation: str = ''      # e.g. 'mode_fwhm_natural'

    # Null model marker: when True, the fit pipeline skips encoding
    # entirely and just predicts the mean training response (a 0-param
    # baseline used as a reference for cvR² comparisons).
    is_null: bool = False


# ── Registry ────────────────────────────────────────────────────────────────
# Order matters only for the warm-start chain (warm_start_from must
# refer to a spec already in MODEL_SPECS).

MODEL_SPECS: dict[str, ModelSpec] = {
    'null': ModelSpec(
        name='null',
        cls=LogGaussianPRF,                   # ignored — special-cased
        cls_kwargs={},
        needs_session=False,
        save_params=[],                        # no params to save
        out_subdir='aprf-null',
        cv_out_subdir='aprf-null.cv',
        grid_dims=0,
        is_null=True,
    ),

    'standard': ModelSpec(
        name='standard',
        cls=LogGaussianPRF,
        cls_kwargs={'allow_neg_amplitudes': False,
                     'parameterisation': 'mode_fwhm_natural'},
        needs_session=False,
        save_params=['mode', 'fwhm', 'amplitude', 'baseline'],
        out_subdir='aprf',
        cv_out_subdir='aprf.cv',
        grid_dims=1,
    ),

    'session-shift': ModelSpec(
        name='session-shift',
        cls=SessionShiftedLogGaussianPRF,
        needs_session=True,
        save_params=['mode_1', 'mode_2', 'fwhm', 'amplitude', 'baseline'],
        out_subdir='aprf-session-shift',
        cv_out_subdir='aprf-shift.cv',
        grid_dims=2,
    ),

    'fwhm-shift': ModelSpec(
        name='fwhm-shift',
        cls=FwhmShiftedLogGaussianPRF,
        needs_session=True,
        save_params=['mode_1', 'mode_2', 'fwhm_1', 'fwhm_2',
                      'amplitude', 'baseline'],
        out_subdir='aprf-fwhm-shift',
        cv_out_subdir='aprf-fwhm-shift.cv',
        grid_dims=0,                                # no grid; warm-start only
        warm_start_from='session-shift',
        warm_start_dup={'fwhm_1': 'fwhm', 'fwhm_2': 'fwhm'},
    ),

    'fully-shifted': ModelSpec(
        name='fully-shifted',
        cls=FullyShiftedLogGaussianPRF,
        needs_session=True,
        save_params=['mode_1', 'mode_2', 'fwhm_1', 'fwhm_2',
                      'amplitude_1', 'amplitude_2',
                      'baseline_1', 'baseline_2'],
        out_subdir='aprf-fully-shifted',
        cv_out_subdir='aprf-fully-shifted.cv',
        grid_dims=0,
        warm_start_from='session-shift',
        warm_start_dup={'fwhm_1':      'fwhm',
                         'fwhm_2':      'fwhm',
                         'amplitude_1': 'amplitude',
                         'amplitude_2': 'amplitude',
                         'baseline_1':  'baseline',
                         'baseline_2':  'baseline'},
    ),

    'gaussian': ModelSpec(
        name='gaussian',
        cls=GaussianValuePRF,
        needs_session=False,
        save_params=['mode', 'fwhm', 'amplitude', 'baseline'],
        out_subdir='aprf-gauss',
        cv_out_subdir='aprf-gauss.cv',
        grid_dims=1,
    ),

    'gauss-session-shift': ModelSpec(
        name='gauss-session-shift',
        cls=SessionShiftedGaussianValuePRF,
        needs_session=True,
        save_params=['mode_1', 'mode_2', 'fwhm', 'amplitude', 'baseline'],
        out_subdir='aprf-gauss-session-shift',
        cv_out_subdir='aprf-gauss-shift.cv',
        grid_dims=2,
    ),
}


# ── Helpers ────────────────────────────────────────────────────────────────

def get_spec(model_type: str) -> ModelSpec:
    """Look up a ModelSpec by short tag; raise KeyError with a helpful
    list of available tags if not found."""
    if model_type not in MODEL_SPECS:
        avail = ", ".join(sorted(MODEL_SPECS))
        raise KeyError(f"Unknown model_type {model_type!r}. "
                        f"Available: {avail}")
    return MODEL_SPECS[model_type]


def list_model_types() -> list[str]:
    """All registered model types, in insertion order."""
    return list(MODEL_SPECS)


def build_warm_start_init(target_spec: ModelSpec, source_pars):
    """Construct the init_pars DataFrame for ``target_spec``'s fit,
    using ``source_pars`` (the warm-start fit's parameters) and the
    explicit ``warm_start_dup`` rules.

    For each parameter in ``target_spec.save_params``:
      - if it appears in ``warm_start_dup``, copy from
        ``source_pars[warm_start_dup[param]]``;
      - otherwise (param exists in source under the same name), copy
        directly from ``source_pars[param]``.

    Raises KeyError if a target param is missing from both the dup
    map and the source DataFrame.
    """
    import pandas as pd
    cols = {}
    for p in target_spec.save_params:
        src = target_spec.warm_start_dup.get(p, p)
        if src not in source_pars.columns:
            raise KeyError(
                f"Warm-start can't fill {target_spec.name}.{p}: "
                f"source param {src!r} not in warm-start fit "
                f"(have {list(source_pars.columns)}).")
        cols[p] = source_pars[src].values
    return pd.DataFrame(cols, index=source_pars.index)
