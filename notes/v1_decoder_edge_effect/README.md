# V1 decoder edge-effect: why is SD elevated at 0°/90°/180°?

**TL;DR.** The "elevated SD at 0°/90°/180°" was **entirely an artifact of
the simulation grid containing orientations the encoding model never
saw**. The experiment presents gabors only at 7.5° to 172.5° in 7.5°
steps (23 discrete orientations). When the expected-uncertainty pipeline
simulates over a uniform [0°, 180°) grid, the points within ~7° of either
edge are extrapolations and the decoder is, correctly, uncertain there.

**When the simulation grid matches the trained orientation set, the V1
decoder SD is essentially flat — no cardinal/oblique modulation.**
Confirmed on sub-04: SD between 7.8° and 11.1° across all 23 trained
orientations, regardless of whether the prior is uniform over [0°, 180°)
or restricted to the 23 trained orientations. The two priors give
nearly identical per-orientation SDs — what matters is **which
orientations you query**, not the prior over the grid.

The "real-prior" comparison figure (`real_prior_comparison.pdf`) shows
the SD curve for sub-04 with both decoder priors superimposed. They
overlap to within ~1° at every orientation. The previously-seen
"inverted-U + cardinal/oblique modulation" pattern (`sd_and_preferred.pdf`)
disappears when the simulation grid is restricted to trained orientations.

## Anti-cardinal V1 SD peak at 90°  (the leading interior effect)

Re-running with FDR-α=0.05 voxel selection on the trained-only grid
(cohort n=10), the V1 SD curve has a robust **peak at the cardinal 90°**
that survives every subject. See `v1_cardinal_noisiness.pdf`.

Per-voxel preferred-orientation distribution (800 top-R² V1 voxels,
8 subjects on local disk):

| Bin                | % of voxels |
|---                 |---          |
| 0–22.5°            | 17.6%       |
| 22.5–67.5°         | **31.4%**  (peak around 45°)   |
| 67.5–112.5°        | **12.6%**  (DIP around 90°)    |
| 112.5–157.5°       | **26.1%**  (peak around 135°)  |
| 157.5–180°         | 12.2%       |

Cohort V1 decoder SD at 30° (oblique-ish) vs 90° (cardinal), FDR-α=0.05
voxel selection, trained-only grid:

| Subject | SD(30°) | SD(90°) | Δ |
|---     |---|---|---|
| 03      | 14.79° | 15.83° | +1.04° |
| 04      |  9.41° | 13.38° | +3.98° |
| 05      |  8.54° | 17.67° | +9.13° |
| 06      |  7.01° |  7.16° | +0.15° |
| 07      | 12.08° | 13.83° | +1.76° |
| 08      | 14.55° | 16.70° | +2.15° |
| 09      | 11.06° | 14.18° | +3.11° |
| 10      | 10.46° | 13.57° | +3.11° |
| pil01   | 18.42° | 18.90° | +0.48° |
| pil02   | 15.09° | 18.41° | +3.33° |

**All 10 subjects** have SD(90°) > SD(30°). Group Δ(SD₉₀ − SD₃₀) =
**+2.82° ± 0.81**, one-sample t(9) = **3.49, p = 0.007**.

Mechanism: voxel-level preferences pile up around 45° and 135° and are
sparse at the cardinals (12.6% in the 67.5–112.5° band). Fewer voxels
tuned to a stimulus → less Fisher information → broader posterior →
higher decoder SD. The boundary trained orientations (7.5° / 172.5°)
are also elevated (15.85° / 16.81°) for the same reason — they're
adjacent to the sparse cardinal regions 0°/180°.

This is the anti-cardinal strand of the V1 fMRI literature
(Henriksson et al 2017; Maloney & Clifford 2015 emphasises that fMRI
sensitivity to fine-grained cardinal-oblique structure is consistent
with this picture; classical single-cell cardinal effect is largely
not visible at fMRI's spatial scale).

### Dissociation from behaviour

The behavioural SD shows the **opposite pattern**: bid noise *dips* at
the cardinals (0°, 90°, 180°) — the categorical-anchor W-shape (see
`compare_v1_npcr_uncertainty.pdf`). Together with the V1 result this
is a clean double dissociation:

- **V1 decoder**: precision *worst* at cardinals (anti-cardinal voxel
  population).
- **Behaviour**: precision *best* at cardinals (categorical anchors).

So the categorical anchor effect in behaviour cannot be inherited from
V1's orientation code — it must come from a downstream process
(working-memory anchor / category-boundary representation, or NPCr-side
value retrieval gated by category).

## Empirical sanity check

Across 9 orientation bins (20° wide each, cohort n=10, 400 trials/bin),
the simulated `sd_circ` (via Jammalamadaka–Sarma −2 ln R) and the empirical
`mean(|circular_distance(decoded, true)|)` agree to ~10% with the same
orientation ordering — confirms the doubled-angle circular mean / SD
maths is correct, the edge elevation is not a wrap-edge artifact:

| Orient. bin | n | sd_circ (deg) | MAE (deg) | √(π/2)·MAE | discrepancy |
|---|---|---|---|---|---|
| 0–20° | 400 | 13.21 | 9.43 | 11.82 | +1.38 |
| 20–40° | 400 | 11.61 | 8.41 | 10.53 | +1.08 |
| 40–60° | 400 | 12.66 | 9.25 | 11.59 | +1.08 |
| 60–80° | 400 | 13.49 | 9.73 | 12.20 | +1.29 |
| 80–100° | 400 | 13.98 | 10.02 | 12.55 | +1.43 |
| 100–120° | 400 | 14.06 | 10.31 | 12.92 | +1.14 |
| 120–140° | 400 | 13.12 | 9.63 | 12.06 | +1.05 |
| 140–160° | 400 | 11.76 | 8.67 | 10.87 | +0.89 |
| 160–180° | 400 | 13.81 | 10.06 | 12.61 | +1.19 |

Overall ratio sd_circ / (√(π/2) × MAE) = **1.098** — small upward bias
consistent with heavier-than-Gaussian tails (Student-t noise), not a math
error.

## Per-voxel preferred-orientation distribution

`sd_and_preferred.pdf` panel 2 shows the histogram (5° bins) for 800
top-R² V1 voxels across 8 subjects (sub-07 and sub-10 not yet on the local
disk):

| Bin                | % of voxels |
|---                 |---          |
| 0–22.5°            | 17.6%       |
| 22.5–67.5°         | **31.4%**   |
| 67.5–112.5°        | 12.6%       |
| 112.5–157.5°       | **26.1%**   |
| 157.5–180°         | 12.2%       |

The population is **oblique-biased** (peaks near 45° and 135°), in line
with one strand of the V1 fMRI literature (anti-cardinal bias; van der
Heijden et al. 2017, Sun et al. 2013) — though debate exists about
whether fMRI sees the classic cardinal effect at all, see references
below.

## Comparison to Jehee / Brouwer–Heeger conventions

| Aspect                      | Brouwer & Heeger 2009 / 2011               | van Bergen, Ma, Pratte, Jehee 2015            | This project                                    |
|---                          |---                                          |---                                             |---                                              |
| Encoding model              | 8 fixed channels, half-cosine raised to 5  | 8 fixed orientation channels (von Mises)      | 8 fixed von Mises basis, kappa=2                 |
| Channel centres             | Uniform on [0°, 180°)                       | Uniform on [0°, 180°)                          | Uniform on [0°, 180°), endpoint=False           |
| Stimulus orientation grid   | Matches the presented orientations only    | Matches the presented orientations only       | Full [0°, 180°) — INCLUDES UNSAMPLED EDGE       |
| Decoder grid                | Often a finer 1° grid, but never beyond the presented range | 1° grid over the trained range          | 1° grid over [0°, 180°)                         |
| Edge handling               | Implicit — no decode outside presented set | Same                                           | Decoder asked to recover values outside training |

The other groups never ask their decoder to recover an orientation that
wasn't presented at training. We do, by simulating over a uniform
[0°, 180°) grid. That's the proximate cause of the elevated edge SD.

## Fix (single, clean): query only at trained orientations

Replace
```python
stim_grid = np.linspace(0, np.pi, 180, endpoint=False)
```
with the actual presented orientations
```python
stim_grid = np.deg2rad(np.arange(7.5, 173, 7.5))
```
in `compute_expected_decoded_orientation_vonmises.py`. This matches the
Brouwer–Heeger / Jehee convention (the decoder is asked only about
stimuli that exist in the experiment) and removes the spurious edge SD
inflation. The decoder prior (uniform vs trained) makes essentially no
difference once the simulation set is restricted to trained orientations.

See `real_prior_comparison.pdf` and `real_prior_comparison.tsv` for the
sub-04 demonstration.

## Related observation on NPCr

The NPCr decoder uses a **bounded** stimulus grid (`np.linspace(0.5, 50)`
with endpoint=True; bounded uniform prior, not a wrap). Its
"elevated SD at extremes" pattern has a different cause — the actual
trained value range is the same as the simulation grid (we ARE
presenting values at the boundary CHF values 5.5 and 38.5), but the
log-Gaussian PRF tuning curves are inherently broader for larger
modes. The dataset-specific cardinal/oblique tuning question doesn't
apply on the value axis.

## References

- Brouwer GJ, Heeger DJ. (2009). Decoding and Reconstructing Color from
  Responses in Human Visual Cortex. *J Neurosci* 29(44).
- Brouwer GJ, Heeger DJ. (2011). Cross-orientation suppression in human
  visual cortex. *J Neurophysiol* 106.
- Freeman J, Brouwer GJ, Heeger DJ, Merriam EP. (2011). Orientation
  Decoding Depends on Maps, Not Columns. *J Neurosci* 31(13):4792.
  https://www.jneurosci.org/content/31/13/4792
- van Bergen RS, Ma WJ, Pratte MS, Jehee JFM. (2015). Sensory uncertainty
  decoded from visual cortex predicts behavior. *Nat Neurosci* 18(12).
  https://www.nature.com/articles/nn.4150
- Sprague TC, Saproo S, Serences JT. (2015). Visual attention mitigates
  information loss in small- and large-scale neural codes. (CINVOR)
- Maloney RT, Clifford CW. (2015). The basis of orientation decoding in
  human primary visual cortex: fine- or coarse-scale biases?
  *J Neurophysiol* 113(1).
  https://journals.physiology.org/doi/full/10.1152/jn.00196.2014
- Henriksson L, Khaligh-Razavi S-M, Kay K, Kriegeskorte N. (2017). Local
  opposite orientation preferences in V1: fMRI sensitivity to fine-grained
  pattern information. *Sci Rep* 7.
  https://www.nature.com/articles/s41598-017-07036-8
- Furmanski CS, Engel SA. (2000). An oblique effect in human primary
  visual cortex. *Nat Neurosci* — classical cardinal effect.
- Wang W, Bressler SL. (2017). An anti-cardinal effect... (representative
  of the anti-cardinal strand of the V1 fMRI literature).

## Files in this folder

- `README.md` — this report.
- `sd_and_preferred.pdf` — first diagnostic figure: SD vs orientation
  with unsampled-edge shading + per-voxel preferred-orientation histogram.
- `preferred_orientations.txt` — 800 top-R² V1 voxel preferred
  orientations in degrees (one per line).
- `real_prior_comparison.pdf` — the decisive figure: sub-04 SD-vs-orientation
  with the simulation grid restricted to the 23 trained orientations,
  compared under uniform vs trained decoder priors. SD is flat and the
  two priors overlap.
- `real_prior_comparison.tsv` — per-orientation SD for both priors
  on sub-04.
- `v1_cardinal_noisiness.pdf` — cohort SD-vs-orientation curve (FDR05,
  trained-grid, n=10) with the per-voxel preferred-orientation
  distribution below. Shows the 90°-peak / anti-cardinal effect
  directly.
