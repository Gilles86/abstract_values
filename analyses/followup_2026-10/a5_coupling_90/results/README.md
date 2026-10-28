# A5 (exploratory): V1→IPS coupling for 90°-preferring V1 voxels

Run 2026-10-28. n = 35 MRI participants (sub-03 … sub-35, sub-pil01, sub-pil02), both sessions.
Cluster job 7835154 (array 0–34, about 1 min per subject), group step via srun 7836704.

**Question.** Does V1 pass information about 90° to IPS more strongly than information about other
orientations, whether 90° maps to a frequent value (Aligned, `inverse_cdf`, slope 0.07 CHF/deg at
90°) or a rare one (Misaligned, `cdf`, slope 0.47 CHF/deg)?

**Answer.** No. The 90° bin couples slightly *less* to IPS than the other bins do, in both
mappings. The frequency-independent prediction (90° above the other bins in both mappings) fails.

## What was run

Code: `/Users/gdehol/git/abstract_values/analyses/followup_2026-10/a5_coupling_90/`
(`a5_coupling_90.py` + `.sh` per subject, `collect_a5.py` group step on the cluster,
`plot_a5.py` figure made locally).

1. **Residuals.** Same as the existing residual-coupling analysis
   (`connective_fields.gates.residualise`, "Categorical regression isolates trial-specific V1–IPS
   coupling"). Within each session, the GLMsingle single-trial betas were regressed on:
   - 23 orientation dummies
   - run dummies
   - the orientations of the ±1 neighbouring trials

   What remains is the trial-to-trial fluctuation.
2. **V1 bins.** Preferred orientation = `mu` of the joint (both-session) axial von Mises PRF
   (`encoding_models/vonmises-prf`; no voxel had negative amplitude). Because both sessions share
   one fit, bin membership is the same under both mappings. There are 6 bins of 30°, centred on
   0, 30, … 150, so the 90° bin is 75–105°. 8 bins of 22.5° were run as a sensitivity check.
3. **V1 voxel sets.**
   - Primary, fixed before the group results were seen: whole Benson V1, voxels with cvR²(vonmises-prf.cv) > cvR²(aprf-null.cv).
   - Sensitivity: the stimulated 0.75–3.75° annulus that the existing coupling analysis uses (all voxels, or tuned voxels only).
4. **Coupling.** Each bin's mean residual was correlated with every IPS voxel's residual. The
   result was Fisher-z transformed and averaged over the IPS voxels ("raw"). IPS segments are Wang
   IPS0, IPS1–2, IPS3–5 and their union. Primary: all IPS voxels. Sensitivity: aPRF-selected voxels.
   - Secondary "specific" metric: the V1-wide mean residual is removed from each bin first (as the existing channels do), and the score is the mean r² × 100.
5. **Contrast.** Δ90 = coupling(90° bin) − mean coupling(other bins). Δ0 is the same contrast for
   the horizontal bin.
6. **Nulls.** These address Saurabh's point that a bin with more voxels gets a less noisy average,
   and so more coupling by sampling alone.
   - (1) Preferred orientations permuted across V1 voxels, 1000 times. This preserves the marginal distribution and the bin counts. The estimator is Δ observed − Δ permutation mean.
   - (2) Every bin subsampled to the smallest bin's voxel count, 200 draws, with at least 3 voxels per bin. The estimator is Δ on equal-count bins.
7. **Group statistics.** Mean across subjects with a bootstrap 95% CI (10,000 resamples), one-sample
   t, and dz. The mapping difference was computed within subject.

## Bin voxel counts (6 bins; per-subject median)

| V1 set | 0° | 30° | 60° | 90° | 120° | 150° |
|---|---|---|---|---|---|---|
| Whole V1, tuned (primary) | 42 | 53 | 38 | **24** (range 3–56) | 37 | 54 |
| Annulus, all voxels | 12 | 19 | 17 | 11 | 13 | 14 |
| Annulus, tuned | 7 | 9 | 7 | 2 | 5 | 5 |

Per-subject counts: `group/subject_counts.tsv`.

Saurabh's worry assumed that more voxels prefer 90°. Here the opposite holds:
- 90° is the *least* represented preference. It holds 9.2% of tuned V1 voxels, against 16.7% if preferences were uniform.
- The 90°-preferring voxels also have the lowest mean cvR² gain over the null: 0.019, against 0.022–0.026 for the other bins.

The tuned annulus set is too small to bin. It has a median of 2 voxels at 90°, and only 7 of 35
subjects allow subsampling at 6 bins (2 of 35 at 8 bins).

## Results (Δ90; raw coupling, z units; mean [95% CI], n = 35)

**Primary, IPS0–5:**

| Mapping | Δ90, observed − permutation | p | dz | Δ90, equal-count subsampling |
|---|---|---|---|---|
| Misaligned (`cdf`, 90° value rare) | −0.016 [−0.032, −0.002] | .038 | −0.37 | −0.017 [−0.031, −0.004] |
| Aligned (`inverse_cdf`, 90° value frequent) | −0.024 [−0.040, −0.008] | .008 | −0.48 | −0.026 [−0.042, −0.011] |

- **Mapping difference** (Aligned − Misaligned): −0.007 [−0.025, +0.009], p = .39. The coupling does not follow 90°-value frequency.
- **Direction per subject:** 18 of 35 subjects have Δ90 < 0 in both mappings; 6 have Δ90 > 0 in both.
- **By segment:** all eight segment × mapping cells are negative (−0.015 to −0.028):
  - IPS0: −0.018 / −0.024
  - IPS1–2: −0.015 / −0.028
  - IPS3–5: −0.015 / −0.016
- **Horizontal (0°) bin instead:** Δ0 = +0.004 [−0.007, +0.014] (Misaligned) and +0.010 [−0.002, +0.021] (Aligned). So 0° is not special either, and the 90° dip is not a cardinal effect.

**Robustness:**
- **8 bins:** same direction; Misaligned −0.011 [−0.028, +0.005], Aligned −0.018 [−0.035, −0.002].
- **Stimulated annulus, all voxels:** Misaligned −0.018 [−0.035, −0.002], Aligned −0.009 [−0.025, +0.006].
- **IPS value-tuned voxels only:** same direction (figure panel d).
- **"Specific" metric** (V1-wide mean removed, r²): null in both mappings. Misaligned −0.03 [−0.24, +0.21], Aligned +0.04 [−0.25, +0.32] (r² × 100). The 90° dip therefore lives in the *shared* V1 fluctuation, not in an orientation-specific channel.
- **Per-subject permutation p-values** are uninformative: median .69 and .79. The group permutation p for the positive tail is about 1.0, because the effect runs the other way.

Group tables are in `group/`:
- `group_contrasts.tsv`: every cell (V1 set × bins × segment × IPS selection × metric × contrast × estimator)
- `group_mapping_difference.tsv`
- `subject_*.tsv`

## Figure

`a5_coupling_90.pdf` / `.png`, slide-sized:
- **a.** Share of V1 voxels per preferred-orientation bin.
- **b.** Coupling profile over bins, IPS0–5, equal-count bins, both mappings.
- **c.** Δ90 per IPS segment and mapping.
- **d.** Robustness forest at IPS0–5, in dz.

Caption bookkeeping:
- Error bars are bootstrap 95% CIs across 35 subjects.
- In panel c, the faint dots are individual subjects.
- In panel b, each subject's mean over bins has been subtracted.

## Plain-language reading

We asked whether V1 voxels that prefer vertical send more of their trial-to-trial fluctuation to
IPS than other V1 voxels, which would be a "privileged line" for 90°. They do not. With bin size
controlled both ways (permuted labels and equal-count subsampling), 90°-preferring V1 voxels are,
if anything, slightly *less* coupled to IPS. The difference is small (about 0.02 in Fisher z,
dz ≈ −0.4) and the same in both mappings.

It disappears when the V1-wide shared fluctuation is removed. It also coincides with these
voxels being the fewest and the most weakly tuned. The likeliest reading is therefore lower
signal quality, or a retinotopic location effect: under the radial bias, vertical-preferring
voxels sit near the vertical meridian, at the V1/V2 border. Transmission of 90° information does
not look privileged.

For the multi-stage story, this residual-coupling measure gives no support to account 2 ("IPS
inherits 90° precision through a privileged V1→IPS route"). It does not refute that account
either: precision can be inherited through the *content* of the code (the decoded posterior)
without any extra coupling. A3/A4 are the better tests of that.

## Flags and caveats

- **Exploratory**, not preregistered. Many cells were computed. Only the primary cell was fixed beforehand, and none of the cells supports the hypothesis.
- **Preferred orientation is a summary statistic.** A voxel's preferred orientation (the peak of a smooth von Mises fit) summarises a mixture of neurons with different preferences. "90°-preferring voxels" is therefore only loosely meaningful: such a voxel also carries signal about other orientations, and a voxel with a flat profile still gets a label.
- **Label permutation is anti-conservative** under spatial autocorrelation (connective-field pitfalls). Real bins are spatially clustered, random sets are not. Inference therefore rests on the across-subject test of the null-corrected and subsampled Δ, not on permutation p-values.
- **Small V1 sample.** The stimulated annulus holds only about 97 usable voxels per subject on the BOLD grid. The primary analysis therefore used tuned voxels across all of V1, many of them outside the stimulated region, where preferences partly reflect position (radial bias, and the gaze-linked radial map in the right hemisphere, see memory). Hemispheres were not split.
- **Not checked:** whether the 90° dip survives matching bins on tuning cvR² or eccentricity/polar angle. That is the obvious next step if anyone wants to read the negative effect as more than signal quality.
- **Not a mapping-following test.** This analysis does not test mapping-specific coupling. The existing `connective_fields` score already does that, and is positive.
