# Toy V1 → IPS network: does rewiring alone give efficient coding of value?

Script: `/Users/gdehol/git/abstract_values/abstract_values/simulations/v1_ips_mapping.py`
(run from the repo root with `~/mambaforge/envs/abstract_values/bin/python -m abstract_values.simulations.v1_ips_mapping`, ~25 s, numpy/scipy only).
Figures:
- Story figure (2 pages): `/Users/gdehol/git/abstract_values/notes/figures/sim_v1_ips_mapping.pdf`
  (+ `_p1.png`, `_p2.png`). Page 1: the mappings, the network wiring in each session,
  how one IPS unit's tuning shifts, and the efficient-coding consequence. Page 2: the
  orientation-wired confound and the preferred-value test.
- Quantitative panels: `/Users/gdehol/git/abstract_values/notes/figures/sim_v1_ips_mapping_details.pdf`
  (+ `.png`). The panel letters below refer to this figure.

## Question

If the only thing that changes between the cdf and inverse-cdf sessions is the
V1→IPS connectivity, and IPS units stay roughly bell-shaped in value, does the
IPS code show efficient coding of the *session's* value distribution
(Fisher information ∝ p(v)², discrimination threshold ∝ 1/p(v))?

Short answer: **yes**, but there is a catch. Any population that codes orientation
uniformly already has FI ∝ p(v)² in value, so efficient coding of value cannot by
itself tell a value code from an orientation code in this design. The test
that can separate them is whether preferred values stay put across the two
mappings, and at realistic voxel-level bandwidths even that test is weak.

## Model

- **Mapping.** f(θ) is a PCHIP fit through the 25-point tables in
  `experiment/settings/sns_multisubject.yml` (0–180°, 2–42 CHF), smoothed in θ with
  a Gaussian of SD 3.75° so that p(v) has no interpolation spikes. The smoothed
  curve is at most 0.14 CHF (cdf) or 0.30 CHF (inverse cdf) away from the table at
  the 23 stimuli. With θ uniform, p(v) = 1 / (180 f′(θ)).
- **V1.** 120 units with r_i(θ) = 1 + 20 exp(κ[cos 2(θ − φ_i) − 1]), κ = 3
  (FWHM 40°), φ_i uniform (`--cardinal 0.5` over-represents 0°/90°).
- **IPS.** h_j(θ) = 1 + g Σ_i W_ij r_i(θ) / N, with independent Poisson noise. g is
  fixed per model and shared by both mappings (peak about 30 spikes/s).
  - **Value-wired (A, the hypothesis):** v_j is fixed and uniform on 2–42 CHF (80 units),
    W_ij = exp(−(f(φ_i) − v_j)² / 2s²) with s = 2 CHF. Only W depends on the mapping.
    A V1 unit projects to the IPS units labelled with the value its orientation
    carries in this session.
  - **Orientation-wired (B, the confound):** W_ij = vonMises(φ_i − θ_j), with κ the same
    as V1. W does not depend on the mapping at all; the unit's preferred value is
    f(θ_j). Since θ is uniform, f(θ_j) is a quantile code, i.e. the
    Ganguli–Simoncelli efficient code.
  - **Fixed value code (control):** Gaussian tuning to value, uniform preferred values,
    one width (SD 4.7 CHF, the median FWHM of the A units), and the same peak rate.
- **Fisher information** (Poisson): FI(v) = Σ_j h′_j(v)² / h_j(v). I report the
  log–log slope against p(v) over 12.5–167.5°. Efficient coding predicts 2.
- **Decoder:** posterior mean from Poisson counts in a 0.25-s window, 2000 trials per
  stimulus. The prior is uniform over θ, which is the session's own value prior.
  I report the log–log slope of RMSE against p(v), leaving out the two edge stimuli.
  Efficient coding predicts −1.
- **Voxels:** 300 per model. Each voxel draws 30 IPS units around a random centre on
  that model's own map (SD 6% of the range), so voxels are local, not random across
  the whole map. The same units are used in both sessions. The voxel response is
  the mean over its units at the 23 stimuli, plus N(0, 0.5) noise. I fit a Gaussian
  in value per mapping (μ bounded to 2–42) and keep voxels with R² > .5 in both
  mappings.

## Results (default parameters)

| | FI slope cdf | FI slope inv | RMSE slope cdf | RMSE slope inv |
|---|---|---|---|---|
| Efficient-coding prediction | 2 | 2 | −1 | −1 |
| **Value-wired (A)** | **1.62** | **1.69** | **−0.75** | **−0.82** |
| Orientation-wired (B) | 2.00 | 2.00 | −0.96 | −0.98 |
| V1 read out in value | 2.00 | 2.00 | – | – |
| Fixed value code | 0.01 | −0.02 | −0.07 | −0.08 |

1. **Tuning stays bell-shaped (panel c).** A Gaussian in value fits the value-wired
   units with median R² = .93 (cdf) and .94 (inverse cdf). Under the two mappings
   each unit keeps the same preferred value; only its gain and width change.
   The worst fit, R² = .60, comes from units near 2 or 42 CHF. Axial wrap-around
   joins 0° and 180°, which carry 2 and 42 CHF, so those units get a side lobe.
2. **Efficient coding from rewiring alone (panels d and e).** For A, FI tracks
   p(v)^1.6–1.7 and the decoding error tracks p(v)^−0.8, under both mappings, with
   the peaks moving to that session's value clusters. The fixed value code is flat.
   Normalising W so every IPS unit gets the same total input (`--normalise`)
   brings A even closer: FI slopes 1.84 / 1.94, RMSE slopes −0.92 / −0.96.
3. **How it gets there (panels f and g).** Without normalisation the gain does most
   of the work. A unit whose value is frequent in the session receives more V1
   inputs: r(log peak, log p) = .92 (cdf) and 1.00 (inverse cdf), with peaks from
   4.5 to 29 spikes/s. Widths also narrow where values are frequent (FWHM about 5–7
   CHF in the clusters, 15 CHF between them). That correlation with 1/p is strong
   under the inverse cdf (r = .92) but weak under the cdf (r = .13). With
   normalisation the gains equalise and only the width effect is left. Preferred
   values in A are uniform by construction. Efficiency comes from how much each unit
   is driven and how sharp it is, not from where the units sit.
4. **Voxel-level prediction (panel h).** Value-wired voxels lie on y = x: the median
   |μ_cdf − μ_inv| is 1.1 CHF. The orientation index (0 = on y = x, 1 = on the raw
   same-orientation curve, computed only where the two references differ by more
   than 4 CHF) has a median of 0.04 (IQR −0.12 to 0.43). **Orientation-wired voxels
   also end up close to y = x**: median |Δμ| = 1.6 CHF, index 0.17 (IQR −0.18 to
   0.89). A Gaussian-in-value fit to a broad orientation bump reports something
   like the local *average* value. Smoothed over about 50°, both mappings are close
   to linear, so the same-orientation prediction collapses toward the diagonal.
5. **When the voxel test can discriminate (panel i).** For orientation-wired voxels
   the index rises only as orientation tuning narrows: 0.11 at 40–57° FWHM, 0.14 at
   28°, 0.39 at 20°, 0.55 at 14°, 0.65 at 10°. Value-wired voxels stay at about 0
   throughout. With the default bandwidth (40° in V1 and in the pooling kernel) the
   two models overlap heavily.

## The orientation / quantile confound

- In this design orientations are uniform and both mappings are monotone, so the
  value quantile F(f(θ)) = θ/180 in **both** sessions. A unit coding a fixed value
  quantile is therefore a unit coding a fixed orientation. Quantile-style efficient
  coding of value (model B) and plain orientation tuning are **the same model** here.
- For any code driven by orientation alone, FI_value(v) = FI_θ(θ) / f′(θ)²
  = FI_θ · (180 p(v))². With uniform orientation coding this is exactly ∝ p(v)²
  (slope 2.00 for V1 and for B). **FI or discrimination thresholds that follow the
  session's p(v) are therefore not evidence for a value code.** They are what a pure
  orientation code predicts. Behavioural or decoded efficient-coding signatures
  cannot tell value from orientation in this paradigm.
- What does tell them apart is **whether a unit or voxel keeps its preferred value
  across mappings**: y = x for a value-labelled code (A), the same-orientation curve
  for B and V1. That signature is attenuated toward y = x when orientation tuning is
  broad (panel i). A real-data scatter that hugs y = x is therefore weak evidence for
  value tuning unless the effective orientation bandwidth is below about 20°. The
  "same orientation" reference curve on the real-data plot should be the
  bandwidth-smoothed one, not the raw mapping.
- A second candidate signature, **amplitude**, is not robust. Unnormalised A predicts
  higher voxel amplitude where the value is frequent in that session:
  r(log amp_cdf/amp_inv, log p_cdf/p_inv) = +.93. B gives −.92, a fit artefact.
  With `--normalise`, A flips to −.91, so the sign depends on an untestable wiring
  assumption.

## Caveats

- These are rate-level toys. Noise is Poisson on IPS only; V1 noise and its
  propagation are not modelled, so FI is that of the IPS code, not a
  data-processing bound.
- The voxel model is a simple local average with Gaussian-in-value fits. The real
  pipeline uses log-Gaussian aPRFs on GLMsingle betas, with an HRF and a different
  noise model.
- s = 2 CHF and κ = 3 are arbitrary. A's FI slope falls as V1 tuning narrows (1.4 at
  κ = 6, 1.1 at κ = 10, 0.7 at κ = 20, measured on cdf during development). The
  reason is that A's value tuning is then set by s rather than by the V1 inputs.
- With cardinal over-representation in V1 (`--cardinal 0.5`), FI picks up the extra
  θ-Fisher information at 0° and 90° (90° is 22 CHF in both mappings). FI slopes
  become A 1.76 / 1.57 and B 2.18 / 1.83.
- Units near 2 and 42 CHF are distorted by axial wrap-around (0° and 180° are the
  same orientation but carry 2 and 42 CHF). Edge stimuli are left out of the slope
  fits.
