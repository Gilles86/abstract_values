"""Total-noise / perceptual-share parameterisation of the two-stage models.

Why
---
In the sequential and categorical models, ``kappa_r`` (perceptual noise) and
``sigma_rep`` (value noise) trade off: behaviour mostly pins down how noisy the
bids are in total, and much less how that noise is split between the two
stages. Subjects with sigma_rep ~ 2-3 CHF have ``kappa_r`` HDIs ~2.5x as wide
as the rest, a within-subject log(kappa_r)-sigma_rep correlation up to 0.9, and
38% of the NUTS iterations saturate max tree depth
(notes/figures/categorical_posterior_width.pdf). The fix is to sample along
and across that ridge instead of diagonally to it:

    total_noise        T     (CHF)      how noisy the bids are -- well identified
    perceptual_share   s     (0..1)     how much of T is perceptual -- honestly
                                        uncertain when value noise dominates

Definition of "total noise"
---------------------------
Both noises live in an efficient-coding *encoded* space: ``kappa_r`` is a von
Mises concentration on the encoded orientation circle (2*pi = the whole
stimulus range), ``sigma_rep`` a Gaussian SD on the encoded value axis, which is
scaled to the 40 CHF value range (V_MIN..V_MAX). We put the perceptual noise on
the value scale by the same rule -- the fraction of its encoding range it
covers, times the 40 CHF value range:

    perc_chf = 40 * (1/sqrt(kappa_r)) / (2*pi)
             = (40/180 CHF/deg) * orientation_sd_from_kappa(kappa_r)

i.e. the perceptual SD in degrees (under a uniform code) times the AVERAGE
slope of the orientation->value mapping, 40 CHF / 180 deg. That average slope is
the same for the cdf and inverse_cdf mappings by construction (both map the
full circle onto 2..42 CHF), so the definition is mapping-independent and does
not depend on where the stimulus sits on the steep or flat part of a mapping.
Then

    T^2 = sigma_rep^2 + perc_chf^2,     s = perc_chf^2 / T^2

which is a bijection between (kappa_r, sigma_rep) and (T, s): the likelihood is
untouched, only the coordinates and the priors change.

Grid ceiling
------------
An N-point grid cannot resolve kappa_r above (N/2pi)^2 (166 for N = 81). As
s -> 0 the raw kappa_r grows without bound, and a von Mises kernel narrower
than one grid step underflows to zero everywhere. kappa_r is therefore passed
through a smooth saturation, kappa = K tanh(kappa_raw / K) with K = 1.5 x the
ceiling: <2% change below kappa = 60, and nothing above the ceiling was
identifiable anyway. ``kappa_r`` in the trace is the saturated value the
likelihood actually used.

Priors (subject level, hierarchical, non-centred)
-------------------------------------------------
    T:  softplus link, group mean ~ N(1.5, 0.75), group SD ~ HalfNormal(1.0)
    s:  logistic link, group mean ~ N(0.5, 1.0),  group SD ~ HalfNormal(1.0)

Implied subject-level medians (prior predictive, 200k draws) against the old
(kappa_r, sigma_rep) priors: kappa_r 26.5 vs 30, perceptual SD 1.24 vs 1.16 CHF,
sigma_rep 0.96 vs 0.97 CHF, share 0.62 vs 0.71. The old prior also put ~2.5%
of its mass on kappa_r ~ 0 (perceptual SDs of thousands of CHF, from the
softplus of a negative draw); the new one does not.
"""
from __future__ import annotations

import numpy as np

V_RANGE = 40.0                                  # V_MAX - V_MIN, CHF
PERC_SCALE = V_RANGE / (2 * np.pi)              # perc_chf = PERC_SCALE / sqrt(kappa)

TOTAL_NOISE_PRIOR = {'mu_intercept': 1.5, 'sigma_intercept': 0.75,
                     'cauchy_sigma_intercept': 1.0, 'transform': 'softplus'}
SHARE_PRIOR = {'mu_intercept': 0.5, 'sigma_intercept': 1.0,
               'cauchy_sigma_intercept': 1.0, 'transform': 'logistic'}


def to_total_share(kappa_r, sigma_rep):
    """(kappa_r, sigma_rep) -> (total noise in CHF, perceptual share)."""
    perc = PERC_SCALE / np.sqrt(kappa_r)
    total = np.sqrt(perc ** 2 + sigma_rep ** 2)
    return total, perc ** 2 / total ** 2


def from_total_share(total, share):
    """Inverse of :func:`to_total_share` (no saturation)."""
    perc = total * np.sqrt(share)
    return (PERC_SCALE / perc) ** 2, total * np.sqrt(1 - share)


class TotalShareMixin:
    """Sample (total_noise, perceptual_share) instead of (kappa_r, sigma_rep).

    ``free_parameters`` keeps kappa_r / sigma_rep, so get_model_inputs, the PPC
    and the LOO code see exactly the variables they always did -- they are just
    Deterministics now. Everything else (motor noise, prior weight, Fourier
    coefficients) is built as before.
    """

    parameterisation = 'total-share'

    def build_priors(self, hierarchical=True, flat_prior=False):
        import pymc as pm
        import pytensor.tensor as pt
        from bauer.efficient_coding import max_resolvable_kappa

        if not hierarchical:
            raise NotImplementedError("total-share is hierarchical only")
        for key, info in self.free_parameters.items():
            if key not in ('kappa_r', 'sigma_rep'):
                self.build_hierarchical_nodes(key, **info)

        # specify_shape is load-bearing (same trap as bauer's fitted-prior
        # path): without a static subject count the derived (S,) tensors carry
        # a symbolic S and JAX refuses to JIT the hierarchical graph ("Shapes
        # must be 1D sequences of concrete values of integer type").
        n_sub = self._n_subjects(pm.Model.get_context())
        total = pt.specify_shape(
            self.build_hierarchical_nodes('total_noise', **TOTAL_NOISE_PRIOR), (n_sub,))
        share = pt.specify_shape(self._build_share(n_sub), (n_sub,))
        perc = total * pt.sqrt(share)
        kappa_raw = (PERC_SCALE / perc) ** 2
        k_sat = 1.5 * max_resolvable_kappa(self.grid_resolution)
        pm.Deterministic('kappa_r', k_sat * pt.tanh(kappa_raw / k_sat),
                         dims=('subject',))
        pm.Deterministic('sigma_rep', total * pt.sqrt(1.0 - share),
                         dims=('subject',))

    def _build_share(self, n_sub):
        """Per-subject perceptual share: logit-normal, hierarchical."""
        return self.build_hierarchical_nodes('perceptual_share', **SHARE_PRIOR)


# ── two-component population for the share ───────────────────────────────────
#
# The total-share fits showed the share is bimodal across subjects: ~40% sit at
# s ~ 0.99 (value noise ~ 0), the rest at 0.02-0.45. A single logit-normal
# group then needs a group SD ~ 3.3, and the s ~ 1 subjects -- whose likelihood
# is flat in s once sigma_rep is below what the bids can resolve -- wander along
# a long logit tail scaled by that SD: 43% of iterations at max tree depth.
#
# total-share-mix replaces that group distribution by a two-component mixture
# on z = logit(s), with membership marginalised out (no discrete parameters):
#
#   z_s ~ w N(MU_HIGH, SD_HIGH) + (1 - w) N(mu_low, sd_low)
#
#   'perceptual-dominated' component: FIXED at logit 3.5 +- 1.75 (centre
#       s ~ 0.97, sigma_rep ~ 0.2 T). Its exact location is not identified --
#       that is the flat part of the likelihood -- so it is pinned rather than
#       learned. Being fixed it also orders the components: no label switching.
#   'value-noise' component: mu_low ~ N(-1.5, 1)  (s ~ 0.18), sd_low ~
#       HalfNormal(1.5).
#   w ~ Beta(2, 2).
#
# The component widths are deliberately generous. A first version with the
# perceptual component at 4 +- 1 and sd_low ~ HalfNormal(1) left a low-density
# valley at logit ~1-2 between the components; sub-06, whose likelihood is
# broad across that valley, then sat in one mode per chain (2 chains at
# z ~ -2.5, 2 at z ~ +4.8; r_hat 1.67). Overlapping components remove the
# barrier without giving back the long single-Gaussian tail.
#
# z is sampled centred: a mixture has no exact non-centred form, the
# perceptual component's scale is fixed (no funnel), and the value-noise
# component holds the data-informed subjects. The posterior probability that a
# subject belongs to the perceptual-dominated component is stored as
# `share_high_prob`.

MIX_MU_HIGH, MIX_SD_HIGH = 3.5, 1.75
MIX_MU_LOW_PRIOR = (-1.5, 1.0)
MIX_SD_LOW_PRIOR = 1.5
MIX_W_PRIOR = (2.0, 2.0)


class TotalShareMixMixin(TotalShareMixin):
    parameterisation = 'total-share-mix'

    def _build_share(self, n_sub):
        import pymc as pm
        import pytensor.tensor as pt

        w = pm.Beta('share_w_high', *MIX_W_PRIOR)
        mu_low = pm.Normal('share_mu_low', *MIX_MU_LOW_PRIOR)
        sd_low = pm.HalfNormal('share_sd_low', MIX_SD_LOW_PRIOR)
        mus = pt.stack([mu_low, pt.as_tensor(MIX_MU_HIGH)])
        sds = pt.stack([sd_low, pt.as_tensor(MIX_SD_HIGH)])
        ws = pt.stack([1.0 - w, w])
        z = pm.NormalMixture('perceptual_share_untransformed', w=ws, mu=mus,
                             sigma=sds, dims=('subject',))
        lp_low = pm.logp(pm.Normal.dist(mu_low, sd_low), z) + pt.log1p(-w)
        lp_high = pm.logp(pm.Normal.dist(MIX_MU_HIGH, MIX_SD_HIGH), z) + pt.log(w)
        pm.Deterministic('share_high_prob',
                         pt.exp(lp_high - pt.logaddexp(lp_low, lp_high)),
                         dims=('subject',))
        return pm.Deterministic('perceptual_share', pm.math.invlogit(z),
                                dims=('subject',))


def with_total_share(model_cls, mixture=False):
    """A subclass of a bauer two-stage model using the total-share coordinates
    (``mixture=True``: two-component population for the share)."""
    if 'sigma_rep' not in getattr(model_cls, 'base_parameters', []):
        raise ValueError(f"{model_cls.__name__} has no kappa_r/sigma_rep pair")
    mixin = TotalShareMixMixin if mixture else TotalShareMixin
    suffix = "TotalShareMix" if mixture else "TotalShare"
    return type(f"{model_cls.__name__}{suffix}", (mixin, model_cls), {})
