"""
Custom braincoder model classes for the abstract-values project.
"""

import numpy as np
from braincoder.models.prf_1d import GaussianPRF
from braincoder.utils.math import lognormal_pdf_mode_fwhm
from braincoder.stimuli import OneDimensionalStimulusWithAmplitude


class SessionShiftedLogGaussianPRF(GaussianPRF):
    """LogGaussianPRF where the mode shifts freely between two sessions.

    All other parameters (fwhm, amplitude, baseline) are shared across sessions.

    Parameters (per voxel)
    ----------------------
    mode_1    : mode (peak) of the log-Gaussian in session 1 (CHF; softplus → positive)
    mode_2    : mode (peak) of the log-Gaussian in session 2 (CHF; softplus → positive)
    fwhm      : full width at half maximum in natural (CHF) space (softplus → positive)
    amplitude : peak response amplitude  (identity; allow_neg_amplitudes=True)
    baseline  : additive offset           (identity)

    Paradigm
    --------
    DataFrame with columns ``['x', 'session']``:
      * ``x``       — objective CHF value of the gabor stimulus
      * ``session`` — 0 for session 1, 1 for session 2

    Prediction
    ----------
    amplitude * LogNormal_mode_fwhm(x; mode_eff, fwhm)  +  baseline
    where  mode_eff = (1 − session) * mode_1  +  session * mode_2

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
        """Core prediction with session-interpolated mode.

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

        mode = mode_1 * (1.0 - session) + mode_2 * session

        return lognormal_pdf_mode_fwhm(x, mode, fwhm) * amplitude + baseline
