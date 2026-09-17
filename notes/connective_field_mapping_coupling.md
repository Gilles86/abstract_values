# Does NPC–V1 coupling follow the orientation→value mapping?

**Status (2026-09-17):** the step-0 gates pass; the primary test found a positive effect
(n = 30), replicated with an encoding-model projection of V1. It survives removing the neighbouring trials' stimuli but weakens under heavier
nuisance removal, and the reverse direction shows no effect. Suggestive, not established.

Code: `abstract_values/connective_fields/` (`gates.py`, `test_coupling.py`, `plot_gates.py`,
`plot_coupling.py`, `slurm_jobs/`). Figures: `notes/figures/cf_gates.pdf`,
`notes/figures/cf_coupling*.pdf`. Group statistics: `notes/data/cf_coupling*_group_stats.tsv`.
Per-subject outputs: `derivatives/connective_fields/{gates,gates_v1-all,coupling*}/`.

## Question

Hedger et al. (2026, *Nature*) model each target voxel's BOLD signal as a weighted sum of
signals in a source region (a *connective field*, CF). Reading the CF weights against the
source region's topographic map gives the target voxel a tuning inherited through
connectivity.

Here the source is V1, whose relevant map is **preferred orientation**, and the target is
NPCr, whose voxels are tuned to **value**. The design gives the question some leverage:
the two mappings (`cdf`, `inverse_cdf`) are both monotonic in orientation but warped
differently. The same value sits up to 24° apart in orientation (mean 17.6°), and the
sign of that shift flips several times along the value axis. If an NPC voxel preferring
value *v* couples preferentially to V1 populations coding the orientation that is worth
*v* in the **current** condition, its CF over orientation should differ between sessions
in a specific, predictable way.

## Theory in one figure

`notes/figures/cf_coupling_explainer.pdf` (`plot_coupling_explainer.py`).

- **a** Both mappings rise with orientation, but differently: 16 CHF is worth a 75° gabor
  under CDF and a 52° gabor under inverse CDF. Call the orientation worth a voxel's
  preferred value θ*.
- **b–c** An NPCr voxel that prefers 16 CHF should therefore draw on different V1
  populations in the two sessions: its value tuning read through each mapping predicts a
  coupling profile over V1 orientation that peaks at θ* for that session.
- **d** How far θ* moves depends on the preferred value, and the sign flips four times
  (zero at 2, 22 and 42 CHF). No session-level nuisance has that shape.
- **e** Data (argmax-binned V1 voxels, voxels whose θ* moves ≥ 15°, n = 30): NPCr trial
  fluctuations couple most strongly to the V1 channel at **this session's** θ*. The same
  data aligned to the other session's θ* give a lower, flatter profile. The peak is
  confined to one 22.5° channel.
- **f** V1 channels from the inverted vonmises encoding model (all 8 weights of every
  voxel, 5° grid, pooled to 15° for display): the profile is broad and the two alignments
  barely differ by eye. Each κ = 2 basis function is ~50° wide at half maximum, so this projection blurs orientation
  rather than sharpening it.
- **g** Mapping score (observed − label-shuffled) per subject: binned p = .0004
  (neighbours removed: p = .0006); encoding-model projection p = .0008 (p = .003).

## Design

Three choices depart from the Hedger approach:

1. **Remove the stimulus, not the model prediction.** Per session, every voxel's
   single-trial GLMsingle betas lose an additive orientation (23 levels) + run model.
   Every orientation occurs once per run, so nothing stimulus-locked survives. Removing
   an encoding-model prediction instead would leave its misfit, which is stimulus-locked
   and hence mapping-structured. That artefact already produced a false positive in
   `test_condition_residual.py`. Hedger et al. never fit on residuals: their CFs during
   movie watching mix shared stimulus drive with intrinsic coupling.
2. **Put the CF in orientation space.** V1 voxels (Benson V1, 0.75–3.75° — the stimulated
   annulus) are binned into 8 channels by preferred orientation. Each channel is the mean
   residual of its voxels, minus the mean over channels. An NPC voxel's CF is its residual
   correlation with each channel, centred over channels. Coarse-scale orientation
   preference is not smooth on the cortical surface, so Hedger's Laplace–Beltrami spatial
   basis would impose the wrong prior.
3. **One tuning label per voxel, from a joint fit over both sessions.** Value tuning is a
   log-Gaussian, orientation tuning an axial von Mises. Both are grid-searched with
   correlation cost, the same model families as `aprf` / `vonmises-prf`. Labels depend
   only on the orientation means and the CF only on the residuals around them.

**Score.** For NPC voxel *i* with value tuning *f_i*, the predicted CF under mapping *m_c*
is *f_i(m_c(θ_k))*: its own value curve read out at the values the channels' orientations
are worth in that condition. The score is

> r(CF, prediction under this session's mapping) − r(CF, prediction under the other mapping),

averaged over both sessions and all voxels. A CF that is the same in both sessions adds
equally to both terms and cancels.

**Control.** The same score with tuning labels permuted across target voxels. This keeps
any ROI-wide change in the CF between sessions and removes only the voxel-by-voxel match
between tuning and coupling. The quantity tested is **observed − shuffled**, per subject,
t-tested across subjects. A label-permutation p-value is not used: permuting labels
breaks the spatial autocorrelation that neighbouring voxels share, so its null is far
too narrow.

**Reverse direction.** V1 voxels as targets, 8 equal-count bins of NPC voxels by
preferred value as channels, prediction *g_j(m_c⁻¹(v_k))*. Correlation is symmetric, so
this is a second reading of the same coupling, not an artefact control.

## Step-0 gates (n = 30)

| Gate | Result |
|---|---|
| V1 preferred orientation, split-half circular r | 0.17 within session, 0.20 across sessions, 0.29 odd vs even runs (all p < .001) |
| NPC preferred value, split-half r | 0.17 within, 0.20 across, 0.25 odd vs even (all p < 10⁻¹²) |
| CF reliability, odd vs even runs within a session | per voxel r = 0.16; voxel-specific part (ROI-mean CF removed) r = 0.14, t₂₉ = 11.7 |
| Power, idealised injection into trial-shuffled NPC residuals | 26 % at r = .005, 64 % at .01, 100 % at .02; condition-invariant injection stays at the 5 % false-positive rate |

The real voxel-specific CF spread (SD 0.088) sits well above the shuffled-noise level
(0.063). Channels are built from all V1 voxels in the band: restricting to voxels whose
cvR² beats the null leaves some channels with a single voxel, for no gain in reliability.

## Results (n = 30; session order 15 `cdf`-first / 15 `inverse_cdf`-first)

Target NPC, source V1:

| Nuisance removed per session | Observed | Shuffled | **Observed − shuffled** | Shape r |
|---|---|---|---|---|
| Orientation + run | +0.017 (p = .003) | +0.005 (p = .15) | **+0.0116 ± 0.0029, t₂₉ = 4.03, p = .0004** | 0.19, p = .011 |
| + orientation of trials ±1 | +0.014 (p = .008) | +0.003 | **+0.0114 ± 0.0030, t₂₉ = 3.83, p = .0006** | 0.09, p = .18 |
| + orientation of trials ±2 | +0.008 (p = .22) | +0.002 | +0.0062 ± 0.0040, t₂₉ = 1.57, p = .13 | 0.06, p = .39 |
| ±1 real, ±2 sham (df-matched) | +0.011 (p = .07) | +0.003 | +0.0082 ± 0.0030, t₂₉ = 2.74, p = .010 | 0.09, p = .19 |

*Shape r*: per subject, the correlation across 8 preferred-value bins between the score
and how different the two mappings' predictions are (1 − r between them). Voxels whose
two predictions barely differ cannot carry the effect, so r should be positive.

- The effect holds with the same sign in both session orders (`cdf` first: +0.011,
  p = .04; `inverse_cdf` first: +0.013, p = .003). A plain session effect would flip sign
  between these groups.
- **Neighbouring-trial leak.** Neighbouring single-trial betas overlap in the HRF, so a
  trial's beta carries part of its neighbours' responses. A neighbour's orientation is
  random with respect to this trial's, so the orientation means do not remove it. Within
  a session, value is a function of orientation, so this leak would create exactly the
  mapping-specific NPC–V1 coupling being tested. Removing the ±1 neighbours' orientations
  leaves the effect intact.
- Removing ±2 as well halves it. This model spends ~119 of 184 trial degrees of freedom.
  The df-matched sham (±2 replaced by permuted orientations) loses a similar amount
  (+0.0082), so most of that drop is lost degrees of freedom. A small leak at lag 2
  cannot be excluded.
- The shape test is significant only without neighbour removal.

Target V1, source NPC: nothing at lag 0 (−0.006 ± 0.005, p = .22) or ±1 (p = .13).
With ±2 removed it is negative (−0.008, p = .02), which has no obvious explanation. Note
that the gates only established power for the NPC-target direction; V1 targets have
noisier labels (orientation reliability 0.20) and value channels whose labels cluster at
the grid edges.

## Reading

There is a voxel-specific component of trial-to-trial NPC–V1 coupling that follows the
current condition's mapping. It is small (Δr ≈ 0.01, i.e. between the 64 % and 100 %
power points of the idealised injection), consistent across session orders, and survives
removal of the adjacent trials' stimuli. Three things keep this from being a finding yet:

1. It weakens under the ±2-lag model, and that cannot be cleanly split into lost power
   and leak.
2. The shape test (does the score sit where the predictions differ) is only significant
   in the analysis without neighbour removal.
3. The reverse direction does not replicate it.

## Encoding-model projection (added 2026-09-17)

`--projection iem` replaces argmax bins with an inversion of the vonmises encoding model
(8 basis functions, κ = 2, ridge α = 10, fitted on both sessions):
c_t = (WWᵀ + λI)⁻¹ W r_t, population response = basis(θ) · c_t on a 5° grid, then centred
over orientation. Every voxel contributes through all its weights, in proportion to how
well the model describes it.

| Nuisance | Observed − shuffled | Shape r |
|---|---|---|
| Orientation + run | +0.0094 ± 0.0025, t₂₉ = 3.76, p = .0008 | 0.05, p = .49 |
| + trials ±1 | +0.0083 ± 0.0025, t₂₉ = 3.28, p = .003 | 0.02, p = .76 |

The effect replicates at similar size. It does not reveal more specificity: the basis is
broad, so a sharp coupling peak gets smeared over ~50°. Answering the specificity
question with this approach needs a narrower basis (more functions, higher κ).

## Next steps

- **Settle the neighbour leak directly** instead of by nuisance regression: re-estimate
  single-trial betas with GLMsingle's shared-regressor option, or fit CFs on residual BOLD
  timeseries around each trial, where HRF overlap can be modelled explicitly.
- **Robustness:** 6 and 12 channels; smoothed betas; NPCl and bilateral NPC.
- **Power for the reverse direction:** run the gate-4 injection with V1 as target before
  interpreting its null or its lag-2 negative.
- **Anatomical control target:** an ROI without value tuning (e.g. M1), where the score
  should be zero.
