"""
Custom braincoder model classes for the abstract-values project.
"""

import numpy as np
from braincoder.models.prf_1d import GaussianPRF
from braincoder.utils import lognormalpdf_n
from braincoder.stimuli import OneDimensionalStimulusWithAmplitude


class SessionShiftedLogGaussianPRF(GaussianPRF):
    """LogGaussianPRF where mu shifts freely between two sessions.

    All other parameters (sd, amplitude, baseline) are shared across sessions.

    Parameters (per voxel)
    ----------------------
    mu_1      : mean of the log-Gaussian in session 1  (softplus → positive)
    mu_2      : mean of the log-Gaussian in session 2  (softplus → positive)
    sd        : standard deviation of the log-Gaussian (softplus → positive)
    amplitude : peak response amplitude  (identity; allow_neg_amplitudes=True)
    baseline  : additive offset           (identity)

    Paradigm
    --------
    DataFrame with columns ``['x', 'session']``:
      * ``x``       — objective CHF value of the gabor stimulus
      * ``session`` — 0 for session 1, 1 for session 2

    Prediction
    ----------
    amplitude * LogNormal(x;  mu_eff,  sd)  +  baseline
    where  mu_eff = (1 − session) * mu_1  +  session * mu_2

    Notes
    -----
    The two mu parameters are intentionally named mu_1 / mu_2 rather than
    after a specific transformation.  In a later iteration these can be
    reparametrised as mu_1 and Φ⁻¹(mu_1) (i.e. the CDF / invCDF of the
    value distribution) to test the efficient-coding hypothesis.
    """

    parameter_labels = ['mu_1', 'mu_2', 'sd', 'amplitude', 'baseline']

    def __init__(self, allow_neg_amplitudes=True, **kwargs):
        # 5-parameter transformations: mu_1, mu_2, sd → softplus (positive);
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
        """Core prediction with session-interpolated mu.

        Parameters
        ----------
        paradigm   : (batch, n_trials, 2)  — columns [x, session]
        parameters : (batch, n_voxels, 5)  — [mu_1, mu_2, sd, amplitude, baseline]
        """
        x       = paradigm[..., None, 0]   # (batch, n_trials, 1)
        session = paradigm[..., None, 1]   # (batch, n_trials, 1)

        mu_1      = parameters[:, None, :, 0]  # (batch, 1, n_voxels)
        mu_2      = parameters[:, None, :, 1]
        sd        = parameters[:, None, :, 2]
        amplitude = parameters[:, None, :, 3]
        baseline  = parameters[:, None, :, 4]

        mu = mu_1 * (1.0 - session) + mu_2 * session

        return lognormalpdf_n(x, mu, sd) * amplitude + baseline
