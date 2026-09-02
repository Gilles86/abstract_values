"""
Custom braincoder model classes for the abstract-values project.

Backend-agnostic (keras.ops): runs under TensorFlow, JAX, or PyTorch
Keras backends. No in-graph session validation — the ``session`` column
is constructed by our own paradigm-building code (0/1 only), and graph
asserts are not portable across backends (a tf.debugging assert inside a
jax-traced prediction is what originally broke these models under jax).
"""

import math

from keras import ops
from braincoder.models.prf_1d import GaussianPRF
from braincoder.utils.math import lognormal_pdf_mode_fwhm, norm


class SessionShiftedLogGaussianPRF(GaussianPRF):
    """LogGaussianPRF where the mode shifts freely between two sessions.

    All other parameters (fwhm, amplitude, baseline) are shared across sessions.

    Parameters (per voxel)
    ----------------------
    mode_1    : mode (peak) of the log-Gaussian in session 1 (CHF; softplus → positive)
    mode_2    : mode (peak) of the log-Gaussian in session 2 (CHF; softplus → positive)
    fwhm      : full width at half maximum in natural (CHF) space (softplus → positive)
    amplitude : peak response amplitude  (softplus → positive when allow_neg_amplitudes=False)
    baseline  : additive offset           (identity)

    Paradigm
    --------
    DataFrame with columns ``['x', 'session']``:
      * ``x``       — objective CHF value of the gabor stimulus
      * ``session`` — 0 for session 1, 1 for session 2

    Prediction
    ----------
    amplitude * LogNormal_mode_fwhm(x; mode_eff, fwhm)  +  baseline
    where  mode_eff = mode_1 if session == 0 else mode_2

    Notes
    -----
    The two mode parameters can later be reparametrised as mode_1 and
    Φ⁻¹(mode_1) (CDF / invCDF of the value distribution) to test the
    efficient-coding hypothesis.
    """

    parameter_labels = ['mode_1', 'mode_2', 'fwhm', 'amplitude', 'baseline']

    def __init__(self, allow_neg_amplitudes=True, **kwargs):
        # 5-parameter transformations: mode_1, mode_2, fwhm → softplus (positive);
        # amplitude → identity or softplus; baseline → identity.
        if allow_neg_amplitudes:
            self.transformations = ['softplus', 'softplus', 'softplus',
                                    'identity', 'identity']
        else:
            self.transformations = ['softplus', 'softplus', 'softplus',
                                    'softplus', 'identity']

        # Point both amplitude variants at our custom prediction so that
        # _get_basis_predictions(model_stimulus_amplitude=True) picks it up.
        self._basis_predictions_without_amplitude = self._session_predict
        self._basis_predictions_with_amplitude    = self._session_predict

        # model_stimulus_amplitude=True → stimulus type handles 2-column paradigm.
        GaussianPRF.__init__(self, allow_neg_amplitudes=allow_neg_amplitudes,
                             model_stimulus_amplitude=True, **kwargs)

    # ------------------------------------------------------------------
    def _session_predict(self, paradigm, parameters):
        """Core prediction with session-switched mode.

        Parameters
        ----------
        paradigm   : (batch, n_trials, 2)  — columns [x, session]
        parameters : (batch, n_voxels, 5)  — [mode_1, mode_2, fwhm, amplitude, baseline]
        """
        x       = paradigm[..., None, 0]   # (batch, n_trials, 1)
        session = paradigm[..., None, 1]   # (batch, n_trials, 1)

        mode_1    = parameters[:, None, :, 0]  # (batch, 1, n_voxels)
        mode_2    = parameters[:, None, :, 1]
        fwhm      = parameters[:, None, :, 2]
        amplitude = parameters[:, None, :, 3]
        baseline  = parameters[:, None, :, 4]

        mode = ops.where(session < 0.5, mode_1, mode_2)

        return lognormal_pdf_mode_fwhm(x, mode, fwhm) * amplitude + baseline


# ── Symmetric Gaussian variants ─────────────────────────────────────────────

_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))  # ≈ 0.4247


class GaussianValuePRF(GaussianPRF):
    """Symmetric Gaussian pRF on the value dimension (no rightward skew).

    Parameters (per voxel)
    ----------------------
    mode      : centre (peak) of the Gaussian (CHF; softplus → positive)
    fwhm      : full width at half maximum (CHF; softplus → positive)
    amplitude : peak response amplitude  (softplus when allow_neg_amplitudes=False)
    baseline  : additive offset           (identity)

    Prediction
    ----------
    amplitude * exp(-0.5 * ((x - mode) / sigma)²) + baseline
    where  sigma = fwhm / (2 * sqrt(2 * ln 2))
    """

    parameter_labels = ['mode', 'fwhm', 'amplitude', 'baseline']

    def __init__(self, allow_neg_amplitudes=True, **kwargs):
        if allow_neg_amplitudes:
            self.transformations = ['softplus', 'softplus', 'identity', 'identity']
        else:
            self.transformations = ['softplus', 'softplus', 'softplus', 'identity']

        self._basis_predictions_without_amplitude = self._gauss_predict
        self._basis_predictions_with_amplitude = self._gauss_predict

        GaussianPRF.__init__(self, allow_neg_amplitudes=allow_neg_amplitudes,
                             model_stimulus_amplitude=False, **kwargs)

    def _gauss_predict(self, paradigm, parameters):
        x         = paradigm[..., None, 0]
        mode      = parameters[:, None, :, 0]
        fwhm      = parameters[:, None, :, 1]
        amplitude = parameters[:, None, :, 2]
        baseline  = parameters[:, None, :, 3]

        sigma = fwhm * _FWHM_TO_SIGMA
        return norm(x, mode, sigma) * amplitude + baseline


class FwhmOnlyShiftedLogGaussianPRF(GaussianPRF):
    """LogGaussianPRF where FWHM shifts freely between sessions; mode is
    shared across sessions.

    The width-only mirror of :class:`SessionShiftedLogGaussianPRF`
    (mode-only shift). Together with :class:`FwhmShiftedLogGaussianPRF`
    (both shift) and :class:`FullyShiftedLogGaussianPRF` (everything
    shifts), this completes the nested model-comparison ladder for asking
    *which* parameter(s) actually reorganize between conditions:
      SessionShiftedLogGaussianPRF   — mode only
      FwhmOnlyShiftedLogGaussianPRF  — fwhm only   (this class)
      FwhmShiftedLogGaussianPRF      — mode + fwhm
      FullyShiftedLogGaussianPRF     — everything

    Parameters (per voxel)
    ----------------------
    mode      : log-Gaussian mode, shared across sessions (CHF; softplus)
    fwhm_1    : FWHM in session 1 (CHF; softplus)
    fwhm_2    : FWHM in session 2 (CHF; softplus)
    amplitude : peak response amplitude  (shared across sessions)
    baseline  : additive offset           (shared)

    Paradigm
    --------
    DataFrame with columns ``['x', 'session']`` (see
    :class:`SessionShiftedLogGaussianPRF`).

    Prediction
    ----------
    amplitude * LogNormal_mode_fwhm(x; mode, fwhm_eff) + baseline
    where  fwhm_eff = fwhm_1 if session == 0 else fwhm_2
    """

    parameter_labels = ['mode', 'fwhm_1', 'fwhm_2', 'amplitude', 'baseline']

    def __init__(self, allow_neg_amplitudes=True, **kwargs):
        if allow_neg_amplitudes:
            self.transformations = ['softplus', 'softplus', 'softplus',
                                    'identity', 'identity']
        else:
            self.transformations = ['softplus', 'softplus', 'softplus',
                                    'softplus', 'identity']

        self._basis_predictions_without_amplitude = self._session_predict
        self._basis_predictions_with_amplitude    = self._session_predict

        GaussianPRF.__init__(self, allow_neg_amplitudes=allow_neg_amplitudes,
                             model_stimulus_amplitude=True, **kwargs)

    def _session_predict(self, paradigm, parameters):
        x       = paradigm[..., None, 0]
        session = paradigm[..., None, 1]

        mode      = parameters[:, None, :, 0]
        fwhm_1    = parameters[:, None, :, 1]
        fwhm_2    = parameters[:, None, :, 2]
        amplitude = parameters[:, None, :, 3]
        baseline  = parameters[:, None, :, 4]

        fwhm = ops.where(session < 0.5, fwhm_1, fwhm_2)
        return lognormal_pdf_mode_fwhm(x, mode, fwhm) * amplitude + baseline


class FwhmShiftedLogGaussianPRF(GaussianPRF):
    """LogGaussianPRF where the mode AND FWHM shift per session.

    Intermediate model between SessionShiftedLogGaussianPRF (only mode
    shifts; FWHM/amp/baseline shared) and FullyShiftedLogGaussianPRF
    (everything shifts). Tests whether dispersion *also* reorganizes
    between conditions on top of the mode shift — i.e. whether a
    voxel's tuning width is condition-specific.

    Parameters (per voxel)
    ----------------------
    mode_1, mode_2 : per-session log-Gaussian modes (CHF; softplus)
    fwhm_1, fwhm_2 : per-session FWHMs in natural CHF space (softplus)
    amplitude      : peak response amplitude  (shared across sessions)
    baseline       : additive offset           (shared)
    """

    parameter_labels = [
        'mode_1', 'mode_2', 'fwhm_1', 'fwhm_2',
        'amplitude', 'baseline',
    ]

    def __init__(self, allow_neg_amplitudes=True, **kwargs):
        if allow_neg_amplitudes:
            self.transformations = [
                'softplus', 'softplus', 'softplus', 'softplus',
                'identity', 'identity',
            ]
        else:
            self.transformations = [
                'softplus', 'softplus', 'softplus', 'softplus',
                'softplus', 'identity',
            ]
        self._basis_predictions_without_amplitude = self._session_predict
        self._basis_predictions_with_amplitude    = self._session_predict
        GaussianPRF.__init__(self, allow_neg_amplitudes=allow_neg_amplitudes,
                             model_stimulus_amplitude=True, **kwargs)

    def _session_predict(self, paradigm, parameters):
        x       = paradigm[..., None, 0]
        session = paradigm[..., None, 1]

        mode_1 = parameters[:, None, :, 0]
        mode_2 = parameters[:, None, :, 1]
        fwhm_1 = parameters[:, None, :, 2]
        fwhm_2 = parameters[:, None, :, 3]
        amp    = parameters[:, None, :, 4]
        base   = parameters[:, None, :, 5]

        is_ses1 = session < 0.5
        mode = ops.where(is_ses1, mode_1, mode_2)
        fwhm = ops.where(is_ses1, fwhm_1, fwhm_2)
        return lognormal_pdf_mode_fwhm(x, mode, fwhm) * amp + base


class FullyShiftedLogGaussianPRF(GaussianPRF):
    """LogGaussianPRF where ALL parameters shift freely between sessions.

    The maximally-flexible nested model for the no-shift / mode-shift /
    fully-shifted cvR² comparison: if cvR² for this model isn't higher
    than for :class:`SessionShiftedLogGaussianPRF`, then only the mode
    actually remaps between conditions — the clean theoretical story
    for the abstract-values paper.

    Parameters (per voxel)
    ----------------------
    mode_1, mode_2          : per-session log-Gaussian modes (CHF)
    fwhm_1, fwhm_2          : per-session FWHMs in natural CHF space
    amplitude_1, amplitude_2: per-session peak response amplitudes
    baseline_1, baseline_2  : per-session additive offsets
    """

    parameter_labels = [
        'mode_1', 'mode_2', 'fwhm_1', 'fwhm_2',
        'amplitude_1', 'amplitude_2', 'baseline_1', 'baseline_2',
    ]

    def __init__(self, allow_neg_amplitudes=True, **kwargs):
        if allow_neg_amplitudes:
            self.transformations = [
                'softplus', 'softplus', 'softplus', 'softplus',
                'identity', 'identity', 'identity', 'identity',
            ]
        else:
            self.transformations = [
                'softplus', 'softplus', 'softplus', 'softplus',
                'softplus', 'softplus', 'identity', 'identity',
            ]

        self._basis_predictions_without_amplitude = self._session_predict
        self._basis_predictions_with_amplitude    = self._session_predict

        GaussianPRF.__init__(self, allow_neg_amplitudes=allow_neg_amplitudes,
                             model_stimulus_amplitude=True, **kwargs)

    def _session_predict(self, paradigm, parameters):
        x       = paradigm[..., None, 0]
        session = paradigm[..., None, 1]

        mode_1    = parameters[:, None, :, 0]
        mode_2    = parameters[:, None, :, 1]
        fwhm_1    = parameters[:, None, :, 2]
        fwhm_2    = parameters[:, None, :, 3]
        amp_1     = parameters[:, None, :, 4]
        amp_2     = parameters[:, None, :, 5]
        base_1    = parameters[:, None, :, 6]
        base_2    = parameters[:, None, :, 7]

        is_ses1 = session < 0.5
        mode      = ops.where(is_ses1, mode_1, mode_2)
        fwhm      = ops.where(is_ses1, fwhm_1, fwhm_2)
        amplitude = ops.where(is_ses1, amp_1,  amp_2)
        baseline  = ops.where(is_ses1, base_1, base_2)

        return lognormal_pdf_mode_fwhm(x, mode, fwhm) * amplitude + baseline


class LinearValuePRF(GaussianPRF):
    """Linear (non-tuned) encoding model on the value dimension.

    Baseline comparison for the Gaussian/log-Gaussian pRF variants: no
    location/width shape parameters, just a straight line in CHF value.
    Tests whether a voxel's response to value looks more like a tuned
    bump (aPRF) or a monotonic ramp.

    Parameters (per voxel)
    ----------------------
    amplitude : linear slope relating CHF value to response (identity —
                signed, can be negative, unlike the pRF bump amplitude)
    baseline  : response at value = 0 (identity)

    Prediction
    ----------
    amplitude * x + baseline

    Fitting
    -------
    Linear in both parameters, so it is fit by a single closed-form OLS
    regression (``ParameterFitter.refine_baseline_and_amplitude``, with
    ``positive_amplitude=False`` since the slope is signed) — no grid
    search, no gradient descent needed. See ``ModelSpec.is_linear`` in
    ``model_specs.py`` and its dispatch in ``fit_pipeline.py``.
    """

    parameter_labels = ['amplitude', 'baseline']

    def __init__(self, **kwargs):
        self.transformations = ['identity', 'identity']

        self._basis_predictions_without_amplitude = self._linear_predict
        self._basis_predictions_with_amplitude = self._linear_predict

        GaussianPRF.__init__(self, model_stimulus_amplitude=False, **kwargs)

    def _linear_predict(self, paradigm, parameters):
        x         = paradigm[..., None, 0]
        amplitude = parameters[:, None, :, 0]
        baseline  = parameters[:, None, :, 1]

        return amplitude * x + baseline


class SessionShiftedGaussianValuePRF(GaussianPRF):
    """Symmetric Gaussian pRF where the mode shifts freely between sessions.

    All other parameters (fwhm, amplitude, baseline) are shared across sessions.

    Parameters (per voxel)
    ----------------------
    mode_1    : centre of the Gaussian in session 1 (CHF; softplus → positive)
    mode_2    : centre of the Gaussian in session 2 (CHF; softplus → positive)
    fwhm      : full width at half maximum (CHF; softplus → positive)
    amplitude : peak response amplitude  (softplus when allow_neg_amplitudes=False)
    baseline  : additive offset           (identity)

    Paradigm
    --------
    DataFrame with columns ``['x', 'session']``:
      * ``x``       — objective CHF value
      * ``session`` — 0 for session 1, 1 for session 2

    Prediction
    ----------
    amplitude * exp(-0.5 * ((x - mode_eff) / sigma)²) + baseline
    where  mode_eff = mode_1 if session == 0 else mode_2
           sigma    = fwhm / (2 * sqrt(2 * ln 2))
    """

    parameter_labels = ['mode_1', 'mode_2', 'fwhm', 'amplitude', 'baseline']

    def __init__(self, allow_neg_amplitudes=True, **kwargs):
        if allow_neg_amplitudes:
            self.transformations = ['softplus', 'softplus', 'softplus',
                                    'identity', 'identity']
        else:
            self.transformations = ['softplus', 'softplus', 'softplus',
                                    'softplus', 'identity']

        self._basis_predictions_without_amplitude = self._session_predict
        self._basis_predictions_with_amplitude = self._session_predict

        GaussianPRF.__init__(self, allow_neg_amplitudes=allow_neg_amplitudes,
                             model_stimulus_amplitude=True, **kwargs)

    def _session_predict(self, paradigm, parameters):
        x       = paradigm[..., None, 0]
        session = paradigm[..., None, 1]

        mode_1    = parameters[:, None, :, 0]
        mode_2    = parameters[:, None, :, 1]
        fwhm      = parameters[:, None, :, 2]
        amplitude = parameters[:, None, :, 3]
        baseline  = parameters[:, None, :, 4]

        mode = ops.where(session < 0.5, mode_1, mode_2)
        sigma = fwhm * _FWHM_TO_SIGMA

        return norm(x, mode, sigma) * amplitude + baseline
