"""
Custom braincoder model classes for the abstract-values project.

Backend-agnostic (keras.ops): runs under TensorFlow, JAX, or PyTorch
Keras backends. No in-graph session validation — the ``session`` column
is constructed by our own paradigm-building code (0/1 only), and graph
asserts are not portable across backends (a tf.debugging assert inside a
jax-traced prediction is what originally broke these models under jax).
"""

import math

import numpy as np
from keras import ops
from braincoder.models.prf_1d import AxialVonMisesPRF, GaussianPRF
from braincoder.stimuli import (OneDimensionalStimulusWithAmplitude,
                                _Identity, _Periodic)
from braincoder.utils.math import (axial_von_mises_pdf,
                                   lognormal_pdf_mode_fwhm, norm)


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


class MixLambdaLogGaussianPRF(GaussianPRF):
    """Log-Gaussian tuning on the value/quantile mix axis, with lambda a parameter.

    The stimulus axis is x_lambda = (1 - lambda) * v + lambda * q, where v is the
    objective value (CHF) and q = 2 + 40 * F_session(v) is the value's mid-rank
    quantile within its session, rescaled to the same 2-42 range. lambda = 0 is a
    fixed-value code (the standard aPRF), lambda = 1 a quantile code (preferred
    position moves with the session's value distribution; here = orientation rank).

    Parameters (per voxel): mode, fwhm (on the mix axis), amplitude, baseline,
    lam. Fit with ``fixed_pars=['lam']`` for a fixed code or ``shared_pars=['lam']``
    for one lambda shared across the voxels being fitted (neural_priors style).

    Paradigm: DataFrame with columns ``['x', 'q']`` (value, rescaled quantile).
    """

    parameter_labels = ['mode', 'fwhm', 'amplitude', 'baseline', 'lam']

    def __init__(self, allow_neg_amplitudes=True, **kwargs):
        self.transformations = ['softplus', 'softplus',
                                'identity' if allow_neg_amplitudes else 'softplus',
                                'identity', 'identity']
        self._basis_predictions_without_amplitude = self._mix_predict
        self._basis_predictions_with_amplitude = self._mix_predict
        GaussianPRF.__init__(self, allow_neg_amplitudes=allow_neg_amplitudes,
                             model_stimulus_amplitude=True, **kwargs)

    def _mix_predict(self, paradigm, parameters):
        v = paradigm[..., None, 0]
        q = paradigm[..., None, 1]
        mode = parameters[:, None, :, 0]
        fwhm = parameters[:, None, :, 1]
        amplitude = parameters[:, None, :, 2]
        baseline = parameters[:, None, :, 3]
        lam = parameters[:, None, :, 4]
        x = ops.maximum((1. - lam) * v + lam * q, 0.1)     # > 0 for lambda in [-1, 2]
        return lognormal_pdf_mode_fwhm(x, mode, fwhm) * amplitude + baseline


class MixPositionWidthGaussianPRF(GaussianPRF):
    """Gaussian tuning whose POSITION and WIDTH follow the quantile code separately.

    Two mix axes, x_l(v) = (1 - l) v + l q(v), q = 2 + 40 F_mapping(v):
      lam_mu    — WHERE the curve sits. The voxel's preferred position ``mode`` lives
                  on x_{lam_mu}: lam_mu = 0 keeps its preferred value fixed in CHF,
                  lam_mu = 1 keeps its quantile.
      lam_sigma — the axis on which the curve is a Gaussian of fixed width, i.e. HOW
                  WIDE it is: lam_sigma = 0 same width in CHF everywhere, 1 same width
                  in quantile units (narrow where values are dense).
    The response in mapping m is amplitude * exp(-(x_{lam_sigma}(v) - c_m)^2 / 2 sigma^2)
    + baseline, with c_m = x_{lam_sigma}(v*), v* the value whose x_{lam_mu} equals
    ``mode`` under mapping m (piecewise-linear through the 23 presented values plus
    2 and 42 CHF). lam_mu = lam_sigma = lam is a plain Gaussian on the mix axis.

    Parameters: mode, fwhm, amplitude, baseline, lam_mu, lam_sigma.
    Paradigm: DataFrame ``['x', 'q', 'm']``: value, rescaled quantile, mapping
    (0 = cdf / Misaligned, 1 = inverse cdf / Aligned).
    """

    parameter_labels = ['mode', 'fwhm', 'amplitude', 'baseline', 'lam_mu', 'lam_sigma']
    # knots per mapping: presented values (+ 2, 42 CHF) and their rescaled quantiles
    _V = {'cdf': [5.5, 8, 10, 11, 11.5, 12, 12.5, 13, 14, 16, 18.5, 22, 25.5, 28, 30, 31,
                  31.5, 32, 32.5, 33, 34, 36, 38.5],
          'inv': [2.5, 3, 4, 5, 7.5, 11.5, 16, 18.5, 20, 21, 21.5, 22, 22.5, 23, 24, 25.5,
                  28, 32.5, 36.5, 39, 40, 41, 41.5]}

    def __init__(self, allow_neg_amplitudes=True, **kwargs):
        self.transformations = ['softplus', 'softplus',
                                'identity' if allow_neg_amplitudes else 'softplus',
                                'identity', 'identity', 'identity']
        q = 2. + 40. * np.r_[0., (np.arange(23) + .5) / 23, 1.]
        self._knots_v = np.stack([np.r_[2., self._V[k], 42.] for k in ('cdf', 'inv')]
                                 ).astype(np.float32)                         # 2, K
        self._knots_q = np.stack([q, q]).astype(np.float32)
        self._basis_predictions_without_amplitude = self._mixw_predict
        self._basis_predictions_with_amplitude = self._mixw_predict
        GaussianPRF.__init__(self, allow_neg_amplitudes=allow_neg_amplitudes,
                             model_stimulus_amplitude=True, **kwargs)

    @staticmethod
    def _interp(u, xk, yk):
        """Piecewise-linear interpolation of u on monotone knots (xk, yk), linear
        extrapolation clipped to the end segments. u: (..., V); xk, yk: (K, ...)."""
        dx = ops.maximum(xk[1:] - xk[:-1], 1e-3)
        w = ops.clip((u[None] - xk[:-1]) / dx, 0., 1.)
        return yk[0] + ops.sum(w * (yk[1:] - yk[:-1]), axis=0)

    def _mixw_predict(self, paradigm, parameters):
        v = paradigm[..., None, 0]                          # (b, T, 1)
        q = paradigm[..., None, 1]
        m = paradigm[..., None, 2]
        mode = parameters[:, None, :, 0]                    # (b, 1, V)
        sigma = parameters[:, None, :, 1] * _FWHM_TO_SIGMA_MIX
        amplitude = parameters[:, None, :, 2]
        baseline = parameters[:, None, :, 3]
        lm = parameters[:, None, :, 4]
        ls = parameters[:, None, :, 5]
        kv = ops.convert_to_tensor(self._knots_v)[:, :, None, None, None]   # (2, K, 1, 1, 1)
        kq = ops.convert_to_tensor(self._knots_q)[:, :, None, None, None]
        centre = []
        for i in range(2):                                  # mapping: cdf, inv
            xk_mu = (1. - lm) * kv[i] + lm * kq[i]          # (K, b, 1, V)
            xk_sg = (1. - ls) * kv[i] + ls * kq[i]
            centre.append(self._interp(mode, xk_mu, xk_sg))  # (b, 1, V)
        c = ops.where(m < .5, centre[0], centre[1])         # (b, T, V)
        x = (1. - ls) * v + ls * q
        return amplitude * ops.exp(-.5 * ((x - c) / sigma) ** 2) + baseline


_FWHM_TO_SIGMA_MIX = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


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


# ── Orientation-space session shift ─────────────────────────────────────────

class _OrientationSessionStimulus(OneDimensionalStimulusWithAmplitude):
    """A two-column [orientation, session] paradigm.

    braincoder ships no orientation stimulus with a second channel, so this
    borrows the "with amplitude" slot the way SessionShiftedLogGaussianPRF
    borrows it on the value side: column 0 is the orientation (wrapping at pi,
    not 2pi, because gratings are axial) and column 1 is a plain 0/1 session
    index, hence identity rather than the softplus an amplitude would want.
    """

    dimension_labels = ['x (radians, orientation)', 'session']

    def __init__(self, positive_only=True):
        super().__init__()
        self.bijectors = [_Periodic(low=0.0, high=np.pi, name='x'),
                          _Identity(name='session')]


class SessionShiftedAxialVonMisesPRF(AxialVonMisesPRF):
    """AxialVonMisesPRF whose preferred orientation shifts between sessions.

    The orientation-space counterpart of SessionShiftedLogGaussianPRF, and the
    cell the factorial was missing: without it the only flexible orientation
    model was the von Mises *basis set* with free per-session weights, so
    "does tuning change between the cdf and inv_cdf mappings" could not be
    asked of a single-bell model the way it already could in value space.

    Parameters (per voxel)
    ----------------------
    mu_1, mu_2 : preferred orientation in session 1 / session 2 (radians)
    kappa      : concentration, shared across sessions (softplus -> positive)
    amplitude  : peak response amplitude
    baseline   : additive offset

    Paradigm
    --------
    DataFrame with columns ``['x', 'session']`` -- orientation in radians and
    a 0/1 session index, exactly as the value-space shifted model expects.
    """

    parameter_labels = ['mu_1', 'mu_2', 'kappa', 'amplitude', 'baseline']

    def __init__(self, allow_neg_amplitudes=True, **kwargs):
        # mu is an angle and must stay unconstrained (the pdf wraps it);
        # kappa must stay positive.
        if allow_neg_amplitudes:
            self.transformations = ['identity', 'identity', 'softplus',
                                    'identity', 'identity']
        else:
            self.transformations = ['identity', 'identity', 'softplus',
                                    'softplus', 'identity']

        self._basis_predictions_without_amplitude = self._session_predict
        self._basis_predictions_with_amplitude    = self._session_predict

        AxialVonMisesPRF.__init__(self,
                                  allow_neg_amplitudes=allow_neg_amplitudes,
                                  model_stimulus_amplitude=True, **kwargs)

    def _get_stimulus_type(self, model_stimulus_amplitude=False):
        return _OrientationSessionStimulus

    def _get_stimulus(self, **kwargs):
        return _OrientationSessionStimulus()

    def _session_predict(self, paradigm, parameters):
        """Prediction with the preferred orientation switched by session.

        paradigm   : (batch, n_trials, 2)  -- [x, session]
        parameters : (batch, n_voxels, 5)  -- [mu_1, mu_2, kappa, amp, base]
        """
        x       = paradigm[..., None, 0]
        session = paradigm[..., None, 1]

        mu_1      = parameters[:, None, :, 0]
        mu_2      = parameters[:, None, :, 1]
        kappa     = parameters[:, None, :, 2]
        amplitude = parameters[:, None, :, 3]
        baseline  = parameters[:, None, :, 4]

        mu = ops.where(session < 0.5, mu_1, mu_2)

        return axial_von_mises_pdf(x, mu, kappa) * amplitude + baseline


# ── Tuning-shape variants of the axial von Mises (follow-up A4, 2026-10) ────
# Both nest AxialVonMisesPRF exactly (shape = 1 / warp = 0) and keep its
# amplitude convention (amplitude multiplies exp(kappa cos 2d) / (pi I0(kappa))),
# so a fitted vonmises-prf parameter table is a valid warm start as is.

def _cardinal_warp(x, a):
    """Orientation warp F_a(x) = x + (a / 4) sin(4x): the (pi-scaled) CDF of the
    density 1 + a cos(4x), which peaks at the cardinals (0, pi/2) for a > 0 and at
    the obliques for a < 0. Monotone for |a| < 1, F(x + pi) = F(x) + pi.
    Channels evenly spaced in F are denser near the cardinals, and a tuning curve
    of fixed width in F is narrower there by a factor 1 / (1 + a)."""
    return x + 0.25 * a * ops.sin(4. * x)


class GeneralizedAxialVonMisesPRF(AxialVonMisesPRF):
    """Exponential-power ("generalized") axial von Mises tuning.

        f(x) = amplitude * exp(kappa - kappa * d**shape) / (pi I0(kappa)) + baseline,
        d = 1 - cos(2 (x - mu)) in [0, 2]

    shape = 1 is AxialVonMisesPRF exactly. shape < 1: a pointed (cusp-like)
    peak; shape > 1: a flat-topped curve whose tails fall off faster (lighter
    tails) than a von Mises of the same half-width. One shape parameter cannot
    give both a sharper peak and lighter tails at a fixed width; the fit says
    which way the data lean.

    Parameters: mu, kappa, shape, amplitude, baseline.
    """

    parameter_labels = ['mu', 'kappa', 'shape', 'amplitude', 'baseline']

    def __init__(self, allow_neg_amplitudes=False, **kwargs):
        self.transformations = ['identity', 'softplus', 'softplus',
                                'identity' if allow_neg_amplitudes else 'softplus',
                                'identity']
        self._basis_predictions_without_amplitude = self._gvm_predict
        self._basis_predictions_with_amplitude = self._gvm_predict
        AxialVonMisesPRF.__init__(self, allow_neg_amplitudes=allow_neg_amplitudes,
                                  **kwargs)

    def _gvm_predict(self, paradigm, parameters):
        from braincoder.utils.backend import bessel_i0
        x = paradigm[..., None, 0]
        mu = parameters[:, None, :, 0]
        kappa = parameters[:, None, :, 1]
        shape = parameters[:, None, :, 2]
        amplitude = parameters[:, None, :, 3]
        baseline = parameters[:, None, :, 4]
        d = 1. - ops.cos(2. * (x - mu)) + 1e-6          # eps: finite d(d^s)/ds at d = 0
        f = ops.exp(kappa - kappa * d ** shape) / (np.float32(np.pi) * bessel_i0(kappa))
        return f * amplitude + baseline


class WarpedAxialVonMisesPRF(AxialVonMisesPRF):
    """Axial von Mises on a cardinal-warped orientation axis, warp free per voxel.

        f(x) = amplitude * vM_axial(F_a(x); mu, kappa) + baseline,
        F_a(x) = x + (a/4) sin(4x),  a = 0.95 tanh(warp)

    warp = 0 is AxialVonMisesPRF. a > 0 makes tuning narrower (and the
    preferred-orientation axis denser) near the cardinals, a < 0 near the
    obliques. ``mu`` lives on the warped axis (F maps cardinals and obliques to
    themselves).

    Parameters: mu, kappa, warp, amplitude, baseline.
    """

    parameter_labels = ['mu', 'kappa', 'warp', 'amplitude', 'baseline']

    def __init__(self, allow_neg_amplitudes=False, **kwargs):
        self.transformations = ['identity', 'softplus', 'identity',
                                'identity' if allow_neg_amplitudes else 'softplus',
                                'identity']
        self._basis_predictions_without_amplitude = self._warp_predict
        self._basis_predictions_with_amplitude = self._warp_predict
        AxialVonMisesPRF.__init__(self, allow_neg_amplitudes=allow_neg_amplitudes,
                                  **kwargs)

    def _warp_predict(self, paradigm, parameters):
        x = paradigm[..., None, 0]
        mu = parameters[:, None, :, 0]
        kappa = parameters[:, None, :, 1]
        a = 0.95 * ops.tanh(parameters[:, None, :, 2])
        amplitude = parameters[:, None, :, 3]
        baseline = parameters[:, None, :, 4]
        return axial_von_mises_pdf(_cardinal_warp(x, a), mu, kappa) * amplitude + baseline


class CardinalWarpedVonMisesBasis(AxialVonMisesPRF):
    """Fixed-warp version of AxialVonMisesPRF for basis-set (WeightFitter) models.

    Channels with mu evenly spaced and one kappa on the warped axis F_a (see
    ``_cardinal_warp``) are denser and narrower near the cardinals for a > 0.
    ``warp=0`` reproduces the uniform basis of fit_vonmises_cv exactly.
    """

    def __init__(self, warp=0.0, **kwargs):
        self.warp = float(warp)
        AxialVonMisesPRF.__init__(self, **kwargs)

    def _basis_predictions_without_amplitude(self, paradigm, parameters):
        return axial_von_mises_pdf(_cardinal_warp(paradigm[..., None, 0], self.warp),
                                   parameters[:, None, :, 0],
                                   parameters[:, None, :, 1]) * \
            parameters[:, None, :, 2] + parameters[:, None, :, 3]

    _basis_predictions_with_amplitude = _basis_predictions_without_amplitude
