# Low-rank, tempered decoding noise vs spherical (production)

Status: complete (29 Oct 2026). n = 36 in every cell (pil01, pil02, 03–36). sub-36 is included
everywhere except the model simulations, which need its mix-λ fits; see the commands at the end.

## What was run

Production decoders, with only the noise model swapped. Folds, fits, voxel selection and
grid are unchanged:

- **Orientation**: `decode_gabor.py --model vonmises`, on V1, V2 and V3.
  - 8-channel axial von Mises basis, κ = 2, ridge α = project default
  - grid = the 23 presented orientations
- **Value**: `decode_value.py --model loggauss`, on V1, NPCr, IPS0, IPS1–2 and IPS3–5.
  - per-voxel log-Gaussian PRF: 20 × 15 correlation grid, then amplitude/baseline, then
    1000 Adam iterations
  - grid = 50 CHF points between the participant's minimum and maximum value
- **Common to both**:
  - leave-one-run-out (16 folds)
  - the top N voxels by training R² (N = 50 and 250)
  - unsmoothed GLMsingle gabor betas
  - 36 participants (pil01, pil02, 03–36)

In every outer fold, three noise models decode the same held-out trials from the same voxels:

| Tag | Noise model |
|---|---|
| `spherical` | Production. `ResidualFitter(spherical=True)`: per-voxel τ and Student-t ν; λ has no effect on this path. |
| `lowrankcv` | `braincoder.noise.LowRankNoise`: Ω = D(FFᵀ+Ψ)D, Student-t, tempered posterior p(y\|s)^(1/T). k ∈ {0,1,2,5,10,20,40,80} and T ∈ 2^[−2..12] ∪ {∞} are chosen jointly by leave-one-run-out CV over the 15 training runs, on the held-out log posterior of the true stimulus. The encoding model is refitted without each inner run. For the von Mises basis, the refitted quantity is the weights (the new `(parameters, weights)` inner-fold option). This is retinonumeral's `--noise_model lowrank_cv --inner_params loro`. |
| `diagcv` | The k = 0 row of the same CV table at its best T: diagonal Ω plus temperature. It separates the effect of the correlations from the effect of the temperature. |

**CV voxel selection** (`--select-voxels`, tree `decoding_lowrankcv_nvcv`): N is chosen
jointly with k and T inside the same inner leave-one-run-out loop.

- Candidates: N ∈ {25, 50, 100, 200, 400, 800, all ROI voxels}.
- Inside inner fold g, voxels are ranked by the training R² of fold g's own inner fit, so the
  ranking uses training trials only and never sees the outer test run.
- The final selection ranks voxels by the outer-fold training R².
- k ≤ N/4.
- Implemented as `braincoder.noise.LowRankNoise.fit_select_voxels`.

**Code**

- braincoder branch `lowrank-inner-weights`, off v0.6.0. It adds the `(parameters, weights)`
  inner-fold option and `fit_select_voxels`, with tests in `tests/test_noise_models.py`.
- The SLURM jobs put a checkout of that branch (cluster `~/git/braincoder-lowrank`) ahead of
  the env's `libs/braincoder`.
- Decoding: `abstract_values/encoding_models/decode_lowrank_cv.py`, run by
  `slurm_jobs/decode_lowrank_cv.sh` (CPU only).
- The numpy port (`encoding_models/lowrank_noise.py`) is used only as a first-fold
  cross-check. Its CV table matches native braincoder to 1e-7.
- The spherical arm's float64 likelihood matches braincoder's float32 `get_stimulus_pdf` to
  ≤ 2e-2 in posterior mass.

**Outputs**

- `derivatives/decoding_lowrankcv/{gabor,value}/sub-*/func/`: native braincoder, fixed N =
  50 and 250.
- `derivatives/decoding_lowrankcv_nvcv/`: CV-selected N.
- `derivatives/decoding_lowrankcv_numpyport/`: an earlier, cancelled port run. It covers a few
  participants and is not used.
- None of these touch `derivatives/decoding/`.

## Results

Tables are in `results/`:

- `lowrank_contrasts.tsv`: every metric, all three paired contrasts.
- `lowrank_group_metrics.tsv`
- `lowrank_k_T.tsv`
- `nvcv_*.tsv`
- `mapping_profiles_*.tsv`

All statistics are paired across participants: the mean difference, a 95% t-CI, and the count
of participants with a positive difference.

### 1. Low-rank vs spherical (fixed N, same voxels)

The low-rank model wins on every metric that is not about bias, in every ROI and nearly every
participant.

At **250 voxels**:

| | r | MAE | Log score vs flat (nats/trial) | 80% coverage | Posterior SD / RMSE |
|---|---|---|---|---|---|
| V1 orientation | 0.43 → 0.61, Δ +0.18 [0.16, 0.20], 36/36 | 28.9° → 22.1°, Δ −6.8° [−7.7, −5.9] | −6.67 → +0.57 | 0.22 → 0.75 | 0.24 → 0.90 |
| V2 orientation | 0.49 → 0.65 | 26.4° → 20.3° | −4.66 → +0.66 | 0.25 → 0.75 | 0.26 → 0.89 |
| V3 orientation | 0.46 → 0.60 | 27.9° → 22.3° | −3.99 → +0.56 | 0.25 → 0.75 | 0.28 → 0.90 |
| V1 value | 0.45 → 0.59, Δ +0.14 [0.11, 0.16] | 8.32 → 6.90 CHF | −6.20 → +0.33 | 0.20 → 0.71 | 0.27 → 0.90 |
| NPCr value | 0.23 → 0.35, Δ +0.12 [0.09, 0.14], 34/36 | 10.63 → 8.81 CHF | −8.47 → +0.09 | 0.13 → 0.69 | 0.27 → 0.97 |

At **50 voxels** (the headline brain–behaviour setting) the gains are smaller but still
significant everywhere:

- V1 orientation: r +0.078 [0.063, 0.092], MAE −2.9°.
- NPCr value: r +0.053 [0.032, 0.073], MAE −0.90 CHF.
- IPS0, IPS1–2 and IPS3–5 value: r +0.04 to +0.08, MAE −0.6 to −1.0 CHF.
- 80% coverage rises from 0.31–0.40 to 0.70–0.77.

IPS0 (median 144 voxels) and IPS3–5 (median 107) have fewer than 250 voxels. They have no
250-voxel arm; IPS1–2 has one only for the 8 participants with ≥ 250 voxels.

**Where the gain comes from.** The `diagcv` arm separates the two ingredients:

- Temperature alone fixes most of the calibration. Coverage and SD/RMSE are about the same
  for `diagcv` and `lowrankcv`.
- Only the correlations improve accuracy. In V1–V3 at 250 voxels, `diagcv` r is 0.44–0.51 and
  `lowrankcv` r is 0.60–0.65.
- The CV score gain of low-rank over k = 0 grows with N: +0.12 nats/trial at 50 voxels and
  +0.30 at 250 (V1–V3).

**Chosen k and T** (median per participant, fixed N):

| Decoder | N | k (IQR) | T (IQR) | Student-t ν |
|---|---|---|---|---|
| V1–V3 orientation | 50 | 4–5 (2–5) | 1.7–2.0 | 40–50 |
| V1–V3 orientation | 250 | 10–15 (10–20) | 2.8–3.1 | 100–130 |
| Value (all ROIs) | 50 | 1 (0–2) | 2.8–3.4 | 25–40 |
| NPCr value | 250 | 2 (2–5) | 6.5 (5.4–6.7) | – |

T = ∞ (a flat posterior) was never chosen.

Value decoding needs a much larger temperature at few factors. The value code's residuals
have little low-rank structure that the decoder can exploit.

**Calibration.** The low-rank model is close to calibrated but still slightly overconfident:

- 50/80/95% coverage is about 0.46 / 0.76 / 0.93 for orientation and 0.69–0.77 at 80% for
  value.
- A plausible cause: with fixed N the voxel ranking inside the inner CV comes from the outer
  fold, which slightly flatters the inner scores. The CV-selected-N runs rank within each inner
  fold and reach 0.79–0.80.

**Central pull gets worse with low-rank, not better.** The slope of decoded on true value
falls in every value ROI:

- NPCr at 50 voxels: 0.205 → 0.121, Δ −0.084 [−0.104, −0.063].
- V1 value: 0.44 → 0.34.

Tempering widens the posterior over a bounded, flat-prior grid, so its mean regresses towards
the centre. The MAE still improves because the trial-to-trial scatter falls by more than the
bias rises. Signed bias at the value extremes is ±15–18 CHF for the IPS/NPCr decoders. The
posterior **mean** of these decoders is therefore not a usable point estimate of value at the
extremes. Use the posterior itself, or the MAP.

### 2. CV-selected voxel count (`decoding_lowrankcv_nvcv`)

N, k and T are chosen jointly inside the inner leave-one-run-out loop, with ranking on
training runs only.

**Many weak voxels win, everywhere.** The median chosen N is close to the whole ROI:

| ROI | N / ROI size |
|---|---|
| V1 orientation | 733 / 817 |
| V2 orientation | 689 / 803 |
| V3 orientation | 594 / 649 |
| NPCr value | 793 / 800 |
| V1 value | 792 / 817 |
| IPS0, IPS1–2, IPS3–5 value | the whole ROI (144 / 211 / 107) |

- **Stability.** The modal N is chosen in 69–81% of folds for V1–V3 and in 90–95% for the
  value ROIs.
- **No interior optimum.** The inner-CV score rises monotonically with N in every ROI, so
  "all voxels" is the answer for IPS too. IPS does not prefer few strong voxels.
- **Chosen k and T** at the selected N:
  - V1–V3 orientation: k = 20, T = 4.8.
  - Value: k = 2–5, T = 6–21.

CV-selected N against fixed N (all low-rank):

| | vs fixed 50 | vs fixed 250 |
|---|---|---|
| V1 orientation r | +0.080 [0.060, 0.100] | +0.014 [0.002, 0.026] |
| V1 orientation MAE | −3.3° [−4.1, −2.5] | −0.6° [−1.1, −0.1] |
| V1 orientation log score | +0.155 [0.120, 0.191] | +0.043 [0.018, 0.068] |
| NPCr value r | +0.059 [0.022, 0.097] | +0.009 [−0.007, 0.024] |
| NPCr value MAE | +0.04 [−0.09, 0.17] | **+0.21 [0.15, 0.28] CHF (worse)** |
| NPCr value log score | +0.212 [0.185, 0.239] | +0.171 [0.150, 0.193] |
| NPCr 80% coverage | 0.70 → 0.80 | 0.69 → 0.80 |
| IPS0, IPS1–2, IPS3–5 value r | +0.026 to +0.053 | – |

Against fixed 50, IPS log score is +0.04 to +0.11 and IPS MAE is unchanged.

CV-selected N gives the best log score and the best coverage everywhere: 80% coverage is
0.79–0.80 in every ROI, i.e. calibrated. It also gives the best accuracy for orientation.

For value, its MAE is slightly worse than fixed 250 in NPCr and V1 (+0.2 to +0.3 CHF). The
larger temperature it picks increases the central pull: NPCr slope 0.135 → 0.086.

### 3. Brain–behaviour

**Decoded value predicts the bid beyond the true value** (partial r, the
`decoded_predicts_bid.py` recipe, n = 36):

| ROI, N | Spherical | Low-rank | Low-rank − spherical |
|---|---|---|---|
| V1, 50 | +0.080 [0.054, 0.106] | +0.108 [0.077, 0.139] | +0.028 [0.012, 0.045] |
| V1, 250 | +0.063 | +0.110 | +0.047 [0.026, 0.068] |
| NPCr, 50 | +0.052 [0.030, 0.074] | +0.058 [0.038, 0.079] | +0.006 [−0.010, 0.023] |
| NPCr, 250 | +0.039 | +0.068 | +0.029 [0.009, 0.049] |

- The earlier V1 +0.085 and NPC +0.047 (n = 30) reproduce as +0.080 and +0.052 (n = 36,
  spherical, 50 voxels).
- Low-rank makes the effect larger, not smaller.
- The next-trial control is small and **negative** for every noise model (−0.02 to −0.04, p
  < .05). It is not null as the summary note says. It was presumably also negative before.

**Decoding fidelity vs cognitive-model noise.** Spearman ρ with the posterior draws of the
sequential fit and a participant bootstrap; the trace has n = 26:

| Pair | Spherical | Low-rank |
|---|---|---|
| NPCr value accuracy vs 1/σ_rep, 50 voxels (−MAE) | 0.12 [−0.23, 0.50] | 0.42 [0.03, 0.72] |
| NPCr value accuracy vs 1/σ_rep, 250 voxels (−MAE) | 0.26 [−0.14, 0.64] | 0.55 [0.19, 0.78] |
| NPCr value accuracy vs 1/σ_rep, 250 voxels (r) | 0.37 [−0.02, 0.67] | 0.46 [0.09, 0.77] |
| V1 orientation accuracy vs κ_r | 0.12 | 0.07 [−0.34, 0.47] |

- The value-precision link becomes clearer with the low-rank model. The CIs are wide and do
  not clearly separate the two noise models.
- The V1 orientation vs κ_r link stays null.
- Both are still the old 26-participant trace. They have to be redone after the 36-participant
  refit.

### 4. Where on the value line: mapping × position

Setting: low-rank + T with CV-selected N. Repo `cdf` = Misaligned, `inverse_cdf` = Aligned.

Figure: `fig_mapping_profiles_nvcv.pdf`.

- Rows are behaviour (bid SD), V1–V3 (orientation decoder), V1 (value decoder), IPS0, IPS1–2,
  IPS3–5 and NPCr (value decoder).
- Columns are the value axis and the orientation axis.
- Model curves (`simulate_decoding_mix`, unsmoothed, n = 35): **λ = 0 is the fixed-value code
  and λ = 1 is the quantile code.** That is the module's own definition. It is not "λ = 0 =
  orientation" as in the request.
- Open circles mark bins whose 80% coverage is off by more than 0.1.

Mapping × position interaction of log posterior SD:

- Each participant's mapping-difference profile is centred, which removes the main effect.
- Statistic: the RMS of the group profile, with a 95% bootstrap CI.
- p comes from 5000 within-participant sign flips.
- Value axis: 10 matched values from 4 to 40 CHF. Orientation axis: the 23 orientations.

| Decoder | Value axis | Orientation axis |
|---|---|---|
| V1 orientation | 0.088 [0.061, 0.133], p < .001 | 0.041, p = .05 |
| V2 orientation | 0.130, p < .001 | 0.043, p = .03 |
| V3 orientation | 0.099, p < .001 | 0.026, p = .84 |
| V1 value | 0.027, p = .01 | 0.041, p < .001 |
| IPS0 value | 0.009, p = .05 | 0.008, p = .03 |
| IPS1–2 value | 0.011, p = .06 | 0.007, p = .77 |
| IPS3–5 value | 0.007, p = .15 | 0.006, p = .43 |
| NPCr value | 0.014 [0.010, 0.022], p = .001 | 0.008, p = .32 |
| Behaviour (bid SD) | 0.29, p < .001 | 0.63, p < .001 |

- **V1–V3 orientation decoders behave like an orientation code.** Their precision profile
  follows each mapping's value density on the value axis. The two mappings' profiles flip
  there and coincide on the orientation axis, which is the predicted pattern. Coverage holds
  per bin (80% coverage 0.70–0.87).
- **The IPS/NPCr value decoders have essentially flat posterior SD.**
  - Their log-SD profiles vary by about 1–2%: 10.5–11.5 CHF on a 2–42 CHF axis.
  - Interactions are tiny in both spaces. NPCr's is significant on the value axis (RMS 0.014)
    but two orders of magnitude below V1's.
  - The local-slope regression is null: β_v (log SD on log |dv/dθ|) for IPS/NPCr is about
    −0.002 [−0.005, 0.001]. For the V1 orientation decoder it is +0.038 [0.014, 0.062].
  - **These SD profiles do not pass the calibration gate.** 80% coverage per bin ranges from
    0.58–0.68 at the value extremes to 0.90–0.95 in the middle. Bias at the extremes is ±18 CHF.
  - The posterior width is near prior-wide and constant, while the errors follow the central
    pull. So the IPS posterior SD must **not** be read as a precision profile.
  - What the value decoders do show is a mapping × position interaction in |error|:
    - value axis: RMS 0.07–0.08, p ≤ .001 in every IPS ROI and NPCr
    - orientation axis: weaker, 0.04–0.05, p = .04–.30
    - In V1, |error| shows it on both axes (0.17 value, 0.11 orientation).
- **The quantile confound.** A quantile (λ = 1) code and an orientation code make the same
  prediction in this design, because quantile = orientation rank. The simulated λ = 0 and
  λ = 1 curves are nearly indistinguishable on the value axis. So no profile here can separate
  "efficient value code" from "orientation code".
- **Behaviour.**
  - Bid SD tracks the mapping on the value axis: β_v = +0.39 [0.31, 0.47].
  - It shows strong mapping differences on the orientation axis: β_o = −0.38 [−0.44, −0.32];
    interaction RMS 0.63.
  - This is the mixed signature of noise in both stages (cf. the cogmodel).
  - The brain value decoders are too imprecise and too prior-dominated to place on the same
    axis.

### Recommendation

**Yes: make low-rank + T the default, with CV-selected N.** Spherical is badly overconfident:

- 80% intervals cover 13–40% of trials.
- The log score is worse than a flat posterior in every ROI.

Low-rank + T is better on every accuracy and calibration metric:

- 80% coverage 0.79–0.80 with CV-selected N.
- It strengthens the decoded-value → bid effect.

Three caveats:

- **Central pull.** For value decoding it **increases** the central pull of the posterior mean
  (slope −0.08 to −0.09). The bias at the value extremes was already large and gets larger.
  Analyses that use the decoded posterior **mean** as a value estimate near the extremes should
  model that pull or use the full posterior.
- **Fixed-N analyses.** Where a fixed N is needed (e.g. matching old analyses), 250 voxels
  beats 50 everywhere the ROI is large enough.
- **Stale production gabor decodes.** The existing production `derivatives/decoding/gabor/`
  files are from 2026-06-03, before ridge α = 10 became the default (2026-09-07). Their
  decoded orientations differ from the current code by a median of 5° per participant. The
  value decodes match the current code (median 0.0007 CHF; 1–2 participants out of 25 differ,
  probably older betas). Regenerate the gabor production decodes before reusing them.

## Figures

- `results/fig_lowrank_decoding.pdf` (fixed N = 50):
  - a: 80% coverage
  - b: log score
  - c: posterior SD / RMSE
  - d: Δ r
  - e: NPCr decoded vs true value, showing the central pull
  - f: decoded value vs bid
- `results/fig_mapping_profiles_nvcv.pdf`: posterior SD per mapping on the value and
  orientation axes, with the model and behaviour overlays.

Error bars and bands are 95% t-CIs across participants.

## Reproduce

```bash
# cluster (CPU only); braincoder branch lowrank-inner-weights at ~/git/braincoder-lowrank
cd ~/git/abstract_values/abstract_values/encoding_models/slurm_jobs
sbatch --partition=lowprio --array=1-36 --export=ALL,QUANTITY=orientation,ROI=BensonV1 decode_lowrank_cv.sh
sbatch --partition=lowprio --array=1-36 --export=ALL,QUANTITY=value,ROI=NPCr,SELECT_VOXELS=1 decode_lowrank_cv.sh
python analyses/followup_2026-10/lowrank_decoding/collect.py [--tree decoding_lowrankcv_nvcv]
python analyses/followup_2026-10/lowrank_decoding/mapping_profiles_collect.py --tree decoding_lowrankcv_nvcv --n-voxels cv
# local, after rsyncing the *.tsv.gz to results/ (and results/nvcv/)
python analyses/followup_2026-10/lowrank_decoding/analyze.py
python analyses/followup_2026-10/lowrank_decoding/compare_nvcv.py
python analyses/followup_2026-10/lowrank_decoding/mapping_profiles.py \
    --profiles results/nvcv/mapping_profiles_nvoxels-cv_noise-lowrankcv.tsv.gz --tag nvcv
# sub-36 simulations, once its mix-lambda fits exist:
sbatch --partition=lowprio --export=ALL,PARTICIPANT_LABEL=36 simulate_decoding_mix.sh
```

Jobs:

| Run | Job IDs |
|---|---|
| Fixed-N arrays | 7871346–7871353 |
| Fixed-N resubmits | 7872052, 7872380, 7872381 |
| CV-N arrays | 7873583–7873590 |
| Simulations | 7873390–7873424 |

The cancelled numpy-port run was 7871007–7871014.

## 5. Precision as a function of value vs simulate-then-decode (29 Oct 2026, evening)

Code is in `/Users/gdehol/git/abstract_values/analyses/followup_2026-10/lowrank_decoding/`:

- `simulate_ideal_decoding.py` and `.sh` simulate.
- `collect_simideal.py` collects.
- `precision_vs_value.py` analyses.
- The decoder gained a `--fast` mode.

Simulations are in the new tree `derivatives/decoding_simideal/`.

### Metric

Posterior SD fails the calibration gate in IPS/NPCr, so precision here is
**error-based**. It is the pooled **within-stimulus SD of the decoded posterior means**,
over the presented stimuli inside 5-CHF value bins and 22.5° orientation bins.

- Using the within-stimulus SD means the spread of bias across stimuli (central pull) is not
  counted as imprecision.
- Bias and |error| are kept alongside (`precision_vs_value_group.tsv`).
- Every source uses the same metric.

### Why the old λ = 0 / λ = 1 simulations coincided

They do not reflect a bug in `simulate_decoding_mix.py`. The cause is the metric plus a
realistic SNR.

**What shows the codes are implemented correctly.** The ideal observer (true tuning and
noise, flat prior) separates the mappings exactly as theory says once noise is low:

- **The fitted V1 orientation code (positive control).**
  - Posterior SD on the value axis flips between mappings.
  - Misaligned is precise at 12/32 CHF and imprecise at 22. Aligned is the reverse.
  - Mapping × position interaction (RMS of log SD): value axis 0.83, orientation axis 0.006.
- **λ = 1 shows the same flipped pattern at moderate noise, and λ = 0 does not.**
  - At noise scale 0.5, IPS1–2 posterior SD under Misaligned is 0.68 CHF at the 14.5-CHF bin
    and 1.11 at 19.5.
  - Under Aligned it is 1.16 and 1.17 at those bins.
  - Under λ = 0 the two mappings give about the same profile (0.66/0.82 vs 0.61/0.80 CHF).
- **At every noise scale the λ = 1 value-axis interaction exceeds λ = 0's.** In-sample SNR,
  IPS1–2: 1.38 vs 1.34 at scale 0.25, and 0.149 vs 0.081 at scale 1.
  - Caveat: below scale 0.5 the posterior SD approaches the 0.8-CHF grid step. log SD then
    turns into a discretisation artifact, which inflates λ = 0 there too.

**Why they coincide at the data's SNR:**

- **Posterior width.**
  - IPS posteriors are 4–6 CHF wide, the width of a mapping's dense region.
  - Their mean is pulled hard towards the centre of the bounded 2–42 CHF range (bias ±10–18
    CHF at the extremes).
  - Mean posterior SD, which the old figure plotted, is then set by the range and the flat
    prior, not by the code: a U-shape identical for λ = 0 and λ = 1.
- **Old simulation settings.** They used only the 30–57 IPS voxels beating the null, with k = 40
  noise.

The local-slope logic (precision ∝ 1/|dv/dθ|) holds only when the posterior is narrow compared
with the scale on which the mapping's slope changes. In IPS it is not.

### Realistic SNR has to be matched

The first simulations used the code fitted to all trials plus the in-sample residual noise
(`--noise insample`; trees `ideal/` and `pipeline_vonmises/`).

- They decode about 3× better than the data: the V1 pipeline simulation gives an orientation
  MAE of 7.9° against the data's 21.5°.
- The in-sample fit of 8 weights (or 4 PRF parameters) per voxel overstates the signal.

The primary simulations (`--noise cvmatched`) therefore:

- shrink each ROI's tuning, with one factor per ROI, to the estimated true signal variance,
  (in-sample R² + cvR²)/2 per voxel, summed over the ROI. cvR² comes from leave-one-run-out
  for the orientation code and from the mix-λ CV fits for λ;
- keep each voxel's noise at its held-out residual level.

Shrink factors:

| Code | Shrink factor (s) |
|---|---|
| V1 orientation code | ~0.8 |
| IPS λ codes | ~0.35–0.4 |

With these settings the real pipeline run on simulated data reproduces the data's overall
precision:

| ROI | Simulated |error| | Data |error| |
|---|---|---|
| IPS | 8.8–9.4 CHF | 8.5–9.1 CHF |
| V1–V3 | 5.4–5.5 CHF | 5.9–6.6 CHF |

### Pipelines

- **(a) Ideal observer.** 10 replicates at matched trial counts, plus a noise sweep. Trees
  `ideal_cvmatched/` and `ideal/`.
- **(b) The real low-rank decoder on simulated data.** The CV-N decoder needs about 2.5 h per
  task and the queue made 360 tasks infeasible, so (b) uses `decode_lowrank_cv.py --fast`:
  - N = all ROI voxels, which is what CV-N picks in 90–95% of folds;
  - k and T by the same nested CV, with the outer encoding parameters held fixed (no inner
    refits);
  - for exact parity it was also run on the real data (`derivatives/decoding_lowrankcv_fast/`,
    source `emp_fast`);
  - trees `pipeline_{vonmises,lam0,lam1}_cvmatched_fast/`.
- The exact CV-N pipeline exists only for the orientation code
  (`pipeline_vonmises_cvmatched/`).
- An in-sample λ pipeline run was cancelled part-way; it sits in `pipeline_lam0/` and is not
  used.

### Results (n = 36, unsmoothed)

Mapping × position interaction of log precision, as RMS of the participant-centred group
mapping-difference profile: a 95% bootstrap CI, then the p from 5000 sign flips.

| Decoder | Data (fast), value axis | Data (fast), orientation axis | Sim λ = 0, value | Sim λ = 1, value | Sim λ = 0, orientation | Sim λ = 1, orientation |
|---|---|---|---|---|---|---|
| V1 orientation | 0.076 [0.061, 0.139], p = .20 | 0.077, p = .05 | (orientation code: 0.215, p < .001) | | (orientation code: 0.042, p = .70) | |
| V2 orientation | 0.106 [0.086, 0.165], p = .013 | 0.068, p = .12 | (orientation code: 0.255, p < .001) | | (orientation code: 0.045, p = .74) | |
| V3 orientation | 0.078, p = .12 | 0.065, p = .08 | (orientation code: 0.184, p < .001) | | (orientation code: 0.059, p = .40) | |
| V1 value | 0.085, p = .03 | 0.069, p = .03 | 0.122, p < .001 | 0.084, p = .007 | 0.076, p = .01 | 0.103, p = .001 |
| IPS0 | 0.066, p = .017 | 0.036, p = .41 | 0.070, p = .04 | 0.054, p = .18 | 0.021, p = .95 | 0.034, p = .42 |
| IPS1–2 | 0.100 [0.071, 0.149], p = .001 | 0.054, p = .010 | 0.069, p = .04 | 0.071, p = .05 | 0.015, p = .99 | 0.038, p = .40 |
| IPS3–5 | 0.095 [0.076, 0.138], p < .001 | 0.028, p = .77 | 0.077, p = .01 | 0.079, p = .004 | 0.035, p = .50 | 0.059, p = .007 |
| NPCr | 0.053, p = .19 | 0.030, p = .73 | 0.057, p = .16 | 0.092, p = .003 | 0.019, p = .99 | 0.047, p = .20 |

The CV-N decoder on the real data gives the same picture: value axis IPS1–2 0.076 (p = .014),
IPS3–5 0.082 (p = .007), NPCr 0.070 (p = .03); orientation axis null in every IPS ROI and NPCr.

**Positive control.** The fitted V1 orientation code, run through the real decoder at matched
SNR, predicts a strong value-axis interaction and none on the orientation axis.

- The data show the same direction: value-axis 0.08–0.11, significant in V2 (and in V1 with
  the CV-N decoder, p = .009).
- The data's effect is weaker than predicted (0.08–0.11 vs 0.18–0.26).
- The data also carry a small orientation-axis term (0.065–0.077) that the code does not
  predict.

**λ = 0 vs λ = 1 through the real pipeline.** The two codes predict nearly the same
value-axis interaction (0.05–0.12). The pipeline's own structure dominates both: the flat prior
on a bounded value grid, the central pull, and the estimation noise.

**Per-participant closeness.** RMS distance between the centred empirical and simulated
log-precision profiles over mapping × bin; Δ = d(λ0) − d(λ1), > 0 = closer to λ1:

| Comparison | Δ | Participants closer to λ1 | Interpretation |
|---|---|---|---|
| Matched pipeline (`pipef` vs `emp_fast`) | −0.018 to +0.017, every CI includes 0 | 11–20 / 36 | No preference in any ROI |
| Ideal observer, orientation axis | −0.012 to −0.037, CIs exclude 0 in IPS0/1–2/3–5 and NPCr | 9–10 / 36 | Leans λ = 0 |
| Ideal observer, value axis | – | – | Undecided |

**The design limit.** The distance between the two simulations' own profiles for a given
participant (0.21–0.32) is about the same as the distance from the data to either simulation
(0.21–0.35). At this SNR one participant's precision profile cannot tell a quantile code from
a fixed-value code; neither can the group.

### Bottom line

- **Diagnosis.** The coincidence of λ = 0 and λ = 1 is real at the data's SNR, not a bug: the
  ideal observer separates the codes once noise drops 2–4×. The positive control (V1
  orientation code) separates strongly, in the predicted direction.
- **IPS data.** They do show a mapping × position interaction in value-axis precision (IPS1–2,
  IPS3–5) and none on the orientation axis. That is the pattern an orientation/quantile code
  predicts.
- **But it cannot be attributed to a code.** The matched pipeline simulations of BOTH codes
  produce interactions of the same size. Through this decoder, no λ preference is detectable,
  per participant or at group level.

### Not done

- **Jehee decodes.** `decode_jehee_cv.py` (jobs 7881393–7881400) was only 44/138 complete,
  so it is not included.
- **Behaviour row.** The bid-SD overlay is in `fig_mapping_profiles_nvcv.pdf` (section 4),
  not repeated in these figures.

### Files

**Figures** (untracked):

| File | Content |
|---|---|
| `results/fig_precision_vs_value_pipef.pdf` | Primary: data (fast decoder) vs the matched pipeline on simulated λ0/λ1/orientation-code data |
| `results/fig_precision_vs_value_ideal.pdf` | Data vs ideal observer |
| `results/fig_precision_vs_value_pipe.pdf` | Exact CV-N pipeline, orientation code only |

**Tables:**

| File | Content |
|---|---|
| `results/precision_vs_value_group.tsv` | Every source × ROI × space × mapping × bin |
| `results/precision_vs_value_interaction.tsv` | Mapping × position interaction tests |
| `results/precision_vs_value_closeness.tsv` | Per-participant closeness, group summary |
| `results/precision_vs_value_noise_sweep.tsv` | Ideal-observer noise sweep (the diagnosis) |

**Reproduce:**

```bash
cd ~/git/abstract_values/analyses/followup_2026-10/lowrank_decoding
sbatch --array=1-36 --export=ALL,MODE=ideal,NOISE=cvmatched simulate_ideal_decoding.sh
sbatch --array=1-36 --export=ALL,MODE=pipeline,ROI=IPS12,CODE=lam1,NOISE=cvmatched,FAST=1 simulate_ideal_decoding.sh
# real data, matched decoder:
sbatch --partition=lowprio --array=1-36 --export=ALL,QUANTITY=value,ROI=IPS12,FAST=1 \
    ../../../abstract_values/encoding_models/slurm_jobs/decode_lowrank_cv.sh
python collect_simideal.py --empirical-fast          # cluster
python precision_vs_value.py                          # local, after rsync to results/{simideal,fast}
```

| Run | Job IDs |
|---|---|
| Ideal | 7882227 (in-sample), 7888369 (cv-matched) |
| Fast real-data decoder | 7889521–7889528 |
| Fast pipeline simulations | 7889529–7889541 |
| Exact CV-N orientation-code pipeline | 7888370–7888372 |
