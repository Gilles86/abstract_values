# A3: the categorical effect at 90° — built in IPS or inherited from V1?

Run 2026-10-28. Code: `/Users/gdehol/git/abstract_values/analyses/followup_2026-10/a3_categorical_90/`
(`compute_rdms.py`, `compute_controls.py` + `slurm_jobs/compute_rdms.sh` on the cluster;
`behavior_90.py`, `analyze_a3.py`, `figure_a3.py` locally). Full numeric log: `analyze_a3_log.txt`.

## What was run

- **Crossnobis RDMs** (23 orientations × session) per participant for V1, V2/V3 (union), IPS0,
  IPS1–2, IPS3–5, IPS (union of the three), NPC R. Same estimator as
  `abstract_values/geometry/crossnobis.py` (reproduces the talk RDMs exactly, r = 1.0),
  recomputed so I could add V2/V3 and IPS unions and **odd-run / even-run halves** for
  split-half reliability. n = 36 MRI participants (35 for IPS ROIs). Unsmoothed GLMsingle betas.
- **Model-implied RDMs** from the fitted encoding models (joint fits): `vonmises-prf`
  (orientation) and `aprf` (log-Gaussian on CHF). Predicted pattern per stimulus, divided by each
  voxel's residual SD, squared distance averaged over voxels; voxels whose cvR² beats `aprf-null.cv`.
- **Controls**: ROI-mean beta per orientation; crossnobis after removing each trial's ROI mean.
- **Behaviour**: all 36 participants, frame-1 bids removed.

Indices (per ROI, d = crossnobis distance):
- *Local ratio* d(90°, 82.5°/97.5°) / d(other 20 adjacent pairs). This is the prompt's index. I
  report it only at group level, as a ratio of group means with a participant bootstrap, because
  per participant the denominator is often ≤ 0. Crossnobis is unbiased, so with weak signal it
  scatters around 0.
- *Global ratio* d(90°, all others) / d(pairs without 90°). I added this after seeing that 90° is
  far from **every** orientation, not only from its neighbours (panel b).
- Per-participant versions, scale-free in units of the ROI's mean pairwise distance:
  LCI = (d90 − dadj)/dall and **GDI** = (d90all − drest)/dall.

## Numbers (95% CIs: participant bootstrap; correlations: Spearman with bootstrap CI)

**Behaviour.** Category errors are at floor. Bids for 82.5°/97.5° land on the wrong side of
22 CHF on 0.3% [0.0, 0.7] of trials under Misaligned (cdf) and 6.0% [3.7, 8.8] under Aligned
(inverse cdf). For comparison, the equivalent crossing rate at other centres with a similar
value gap is 20–26%. Under Aligned the gap to 22 CHF is only 0.5 CHF, so 6% is far below what
the bid noise allows: the boundary is respected behaviourally. Per participant, most have 0
errors out of about 15 trials, so the per-participant measure I used is the **bid-SD dip at 90°**:
log((SD90 + q)/(SD at 75/82.5/97.5/105° + q)), with q = 0.14 CHF, the bid quantisation.
- Data dip: −1.16 (cdf), −0.91 (inverse cdf) log units, i.e. SD at 90° is about 0.3–0.4× that
  of its neighbours. Split-half reliability 0.61 [0.26, 0.81].
- Model dip from the PPC of `Categorical, Fourier K=4` (n = 34): −1.54 / −1.01.

**90° is special in every ROI, already in V1** (crossnobis, pooled over mappings):

| ROI | local ratio | global ratio | GDI |
|---|---|---|---|
| V1 | 6.0 [3.9, 9.1] | 1.41 [1.20, 1.64] | 0.45 [0.21, 0.69] |
| V2/V3 | 6.7 [4.9, 9.6] | 1.44 [1.30, 1.62] | 0.70 [0.43, 1.01] |
| IPS0 | 8.5 [4.7, 25.8] | 2.36 [1.94, 2.86] | 1.67 [1.06, 2.38] |
| IPS1–2 | 7.0 [5.1, 10.6] | 2.38 [2.05, 2.76] | 1.20 [0.91, 1.48] |
| IPS3–5 | 6.6 [4.4, 11.2] | 2.42 [2.04, 2.88] | 1.50 [1.07, 1.95] |
| IPS | 6.7 [4.7, 10.1] | 2.21 [1.93, 2.54] | 1.19 [0.92, 1.47] |
| NPC R | 6.6 [4.8, 10.2] | 2.52 [2.23, 2.86] | 1.51 [1.23, 1.85] |

- The **local** jump at 90° (about 6×) is the same size in V1 and IPS.
- The **global** separation of 90° from all other orientations is larger in IPS:
  - GDI IPS − V1 = +0.72 [0.37, 1.06], Wilcoxon p < .001, n = 35.
  - It holds in every IPS subregion (+0.73 to +1.21) but not in V2/V3 (+0.25 [−0.05, 0.63]).
- **Not a univariate effect.** 90° trials have lower ROI-mean betas (V1 −0.26, IPS −0.26,
  both p < .001), but removing the trial mean leaves every index unchanged (mean-centred vs
  original per-participant indices r ≥ 0.98; IPS global ratio 2.20 vs 2.21).
- **The fitted tuning models do not reproduce the effect.** Model-implied local ratios are
  V1 0.69 [0.49, 0.94] and IPS 1.18 [0.80, 1.66] (von Mises pRF), and IPS 1.32 [0.96, 1.78]
  (aPRF). So the 90° distinctiveness is not carried by smooth single-peak voxel tuning curves.
  Per participant, model-implied and crossnobis GDI correlate modestly in IPS
  (ρ = 0.47 [0.15, 0.71], von Mises).

**Value space: the jump follows orientation, not value.** At 90° the value step to the
neighbours is 3.5 CHF under Misaligned (the steepest part of that mapping) and 0.5 CHF under
Aligned (the flattest). A code whose distances scale with value (d ∝ Δv²) predicts a local ratio
of 5.2 under Misaligned and 0.05 under Aligned, a 114-fold difference between mappings.
Observed:
- IPS: 7.0 [4.5, 12.3] under Misaligned and 6.3 [3.9, 12.1] under Aligned. Ratio between
  mappings 1.07 [0.46, 2.26] (mean-centred 1.21 [0.58, 2.42]).
- Every other ROI is similar (0.82–1.84).
- The fitted aPRF value model itself predicts a mapping ratio of 4.0 [2.7, 6.8] in IPS, so the
  data are below even that.
- In value units (Fisher information per CHF²), the IPS 90° point is unremarkable under
  Misaligned: 1.34 [0.85, 2.37]× the rest. Under Aligned it is 138 [84, 264]× the rest. So the
  22 CHF jump is large in value terms only under Misaligned, and there the measured distance is
  about what the value step alone would give. Under Aligned, IPS keeps a jump that value cannot
  explain.
- GDI does not differ between mappings either (IPS cdf − inverse cdf −0.01 [−0.62, 0.56]).

**Reliability (split-half, Spearman–Brown).**

| | LCI | GDI | Behaviour SD dip |
|---|---|---|---|
| V1 | 0.28 [−0.49, 0.64] | 0.50 [−0.07, 0.77] | |
| IPS | 0.21 [−0.72, 0.65] | 0.63 [0.27, 0.83] | |
| NPC R | | 0.00 [−1.17, 0.52] | |
| Participants | | | 0.61 [0.26, 0.81] |

The local index is too unreliable for individual differences in any ROI, so the
across-participant analyses below use GDI.

**Does IPS go beyond V1 across participants?** (GDI, n = 35)
- IPS on V1: ρ = 0.24 [−0.12, 0.57]; skipped r = 0.24, p = .17; slope 0.20 [−0.19, 0.58].
- The reliability ceiling for that correlation is √(0.50 × 0.63) = 0.56. The observed Pearson
  r is 0.18, which disattenuates to about 0.31.
- So most of the *reliable* between-person variance in IPS's 90° distinctiveness is not
  predicted by V1's. This is weak evidence: the CIs are wide, and V1's reliability CI touches 0.
- LCI gives ρ = 0.44 [0.10, 0.69], but with reliabilities of 0.2–0.3 that number cannot be
  interpreted (disattenuated > 1).

**Brain ↔ behaviour (exploratory, 32 tests, none survives correction).**
- V1 GDI vs the data SD dip: ρ = 0.03 [−0.29, 0.35].
- IPS GDI vs the data SD dip: ρ = −0.04 [−0.39, 0.30].
- The IPS residual on V1 vs the data dip: −0.07 [−0.42, 0.30]; vs the model dip: −0.21 [−0.56, 0.19].
- Isolated nominal hits in the expected direction: IPS3–5 GDI vs the data dip, ρ = −0.34
  [−0.58, −0.04], p = .05, skipped p = .10; NPC R LCI vs the model dip, ρ = −0.41, p = .01.
  Treat these as noise until replicated.

## Figures

- `fig_a3_categorical_90.svg` / `.png` / `.pdf` (slide-sized, 4 panels):
  - a: crossing rates and category errors
  - b: distance of each orientation to all others, V1 vs IPS
  - c: local ratio per ROI × mapping against the value-code predictions
  - d: per-participant IPS vs V1 GDI with reliabilities
- Tables:
  - `index_group.tsv`, `mapping_test.tsv`, `reliability.tsv`, `ips_vs_v1.tsv`,
    `brain_behaviour.tsv`, `model_vs_crossnobis.tsv`, `univariate_control.tsv`
  - `subject_table.tsv`, `index_subject_*.tsv`, `profile_*.tsv`
  - `behavior_crossing.tsv`, `behavior_subject.tsv`

## What it means for the story

Vertical is represented as special all the way down. In V1, 90° is already about six times more
discriminable from its neighbours than other adjacent orientations are from each other. That
local jump is no larger in IPS, so the precision at 90° is plausibly inherited (account 2).

What IPS adds is that 90° moves away from **all** other orientations: about 2.2–2.5× the typical
distance in IPS and NPC, against 1.4× in V1 and V2/V3. That is the signature of a separate
category rather than a steep local gradient, and it is the part that looks built in IPS
(account 1).

Neither the local jump nor the global separation changes with the mapping, even though the
22 CHF value step at 90° is seven times larger under Misaligned than under Aligned. So the IPS
category is anchored to vertical *orientation*, not to the value 22 CHF. This fits the cognitive
model, which puts the boundary on the orientation posterior. It also fits the
behavioural finding that the value-noise dip at 90° appears under both mappings.

The fitted voxel tuning models (von Mises, aPRF) do not reproduce the jump at all, so whatever
carries it is not smooth single-peak tuning. Across people, IPS's 90° distinctiveness is only
loosely predicted by V1's. Neither predicts the behavioural sharpness at 90°, but that test is
underpowered.

**Verdict: a mixed answer, with the two halves at different strengths.**
- The local precision at 90° is inherited from V1: moderate evidence, from equal group ratios.
- IPS adds an orientation-anchored categorical separation on top: moderate at the group level
  (GDI IPS > V1, p < .001, every IPS subregion), weak at the individual level (ρ = 0.24,
  ceiling 0.56).
- The category is not a value category: strong evidence, from the mapping invariance against
  a 114-fold value prediction.

## Flags

- Individual differences are exploratory (n = 35) and limited by reliability. The LCI is
  unusable; GDI is moderate. A V1 reliability of 0.50 [−0.07, 0.77] means that "IPS goes beyond
  V1" across people cannot be established with this n.
- GDI was defined after looking at the row-mean profile. The local ratio is the pre-specified
  index, and it shows no IPS > V1 difference.
- The mean-centring control rules out a global amplitude difference. It does not rule out a
  90°-specific non-sensory signal (attention, certainty, eye movements along vertical) that
  shows up as a distinct pattern in every ROI, including V1. RT does not differ at 90°: the
  across-participant mean of median RT is 2.37 s, against 2.32–2.50 s for the other
  orientations.
- Behavioural category errors are at floor per participant, so the SD dip was used instead.
  The model SD dip comes from 34 participants' PPC of the Categorical K=4 fit. That fit has its
  boundary at 90° by construction, so its dip is partly built in.
- The 7.5°/172.5° pair (15° apart across 0°) also has large distances. It is excluded from the
  adjacent set, so horizontal could not be tested as a second cardinal in the same way.
