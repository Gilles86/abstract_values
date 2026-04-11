"""
Custom braincoder model classes for the abstract-values project.
"""

import tensorflow as tf
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

        # Session must be exactly 0 or 1 — anything else is a bug.
        tf.debugging.assert_equal(
            tf.reduce_all((session == 0.0) | (session == 1.0)), True,
            message='Session column must contain only 0 and 1')

        mode_1    = parameters[:, None, :, 0]  # (batch, 1, n_voxels)
        mode_2    = parameters[:, None, :, 1]
        fwhm      = parameters[:, None, :, 2]
        amplitude = parameters[:, None, :, 3]
        baseline  = parameters[:, None, :, 4]

        mode = tf.where(session < 0.5, mode_1, mode_2)

        return lognormal_pdf_mode_fwhm(x, mode, fwhm) * amplitude + baseline


# ── Symmetric Gaussian variants ─────────────────────────────────────────────

_FWHM_TO_SIGMA = 1.0 / (2.0 * tf.sqrt(2.0 * tf.math.log(2.0)))  # ≈ 0.4247


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

        tf.debugging.assert_equal(
            tf.reduce_all((session == 0.0) | (session == 1.0)), True,
            message='Session column must contain only 0 and 1')

        mode_1    = parameters[:, None, :, 0]
        mode_2    = parameters[:, None, :, 1]
        fwhm      = parameters[:, None, :, 2]
        amplitude = parameters[:, None, :, 3]
        baseline  = parameters[:, None, :, 4]

        mode = tf.where(session < 0.5, mode_1, mode_2)
        sigma = fwhm * _FWHM_TO_SIGMA

        return norm(x, mode, sigma) * amplitude + baseline
