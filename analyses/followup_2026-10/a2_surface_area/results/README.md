# A2: cortical surface area and noise (exploratory, n = 35 with noise estimates; 36 MRI participants)

Code: /Users/gdehol/git/abstract_values/analyses/followup_2026-10/a2_surface_area/ (extract_area.py and collect_decoding.py ran on the cluster; analyze.py ran locally).
Outputs (BIDS root derivatives/followup_2026-10/a2_surface_area/, local mirror /data/ds-abstractvalue/...): roi_surface_area.tsv, decoding_precision.tsv, subject_table.tsv, area_noise_correlations.tsv, area_area_correlations.tsv, area_vs_total.tsv, decoding_precision_vs_area.tsv, a2_summary.svg/png.

## What was run
- Surfaces: FreeSurfer (fmriprep sourcedata/freesurfer, `sub-*_ses-1`), native space, both hemispheres. Area = sum of `?h.area.mid` (midthickness) inside the ROI and the `cortex.label` (white-surface area `?h.area` and GM volume `?h.volume` also saved). Total cortical area = both hemispheres' midthickness area inside `cortex.label`.
- ROIs: V1, V2, V3, V1-3 (neuropythy Benson14 Varea, subject-specific fsnative atlas); IPS0..IPS5, IPS1-2, IPS3-5, IPS0-5 (Wang15 Mplbl 18-23, fsnative; note IPS4/5 are tiny, ~80 and ~18 mm2); M1 hand area = left hemisphere only: Destrieux G_precentral within 20 mm of MNI (-38,-22,58), the slide-2 definition (fsaverage vertex set from group_cvr2_gain.npz `roi_M1 L`, mapped to native by nearest vertex on the registered sphere); alternative BA4a/BA4p (exvivo thresh labels), left and right. Response hand is right, so left M1 is the primary region.
- Size correction: (a) residual of the area after linear regression on total cortical area, (b) proportion of total. Reported raw, corrected (main), proportion (in tsv).
- Noise: perceptual / value / motor SD in CHF from the motor-noise fit (`talks/2026-10_update_ruff_bedi/data/noise_motor_dissociation.tsv`, n = 35, subjects 1-35; sub-36 has no fit). Sensitivity: K = 4 Fourier fit (`k4_posterior_means.tsv`, sigma_perc / sigma_rep, n = 34). The perceptual/value split is weakly identified across people (see the narrative), so every perceptual/value correlation here inherits that.
- Statistics: Spearman rho with 95% bootstrap CI (2000 resamples); skipped Spearman (MCD outlier removal on ranks) in the tsv, but it removes many points in some cells and is unstable, so read the plain Spearman as primary. All exploratory; about 100+ tests were computed in the tsv, no multiplicity correction.
- Decoding precision: per-subject mean posterior SD per trial (orientation: circular SD in deg; value: SD in CHF), spherical-noise, lambda 0.1 decoders, 100 voxels (matched voxel count); V1 with all voxels also in tsv. Correlated with ROI area.
- Joining: MRI sub-pil01/pil02 = behavioural sub-01/02.

## Numbers (Spearman rho [95% CI], size-corrected; raw in parentheses)
Predicted pairs:
- V1 area vs perceptual noise: +0.18 [-0.15, 0.48] (raw +0.18)
- IPS0-5 area vs value noise: -0.03 [-0.35, 0.30] (raw -0.21 [-0.51, 0.15])
- M1 hand area vs motor noise: -0.06 [-0.43, 0.34] (raw -0.04)
Cross pairs: V1-value -0.13 [-0.49, 0.25]; V1-motor -0.11; IPS-perceptual -0.08 [-0.41, 0.27]; IPS-motor -0.04; M1-perceptual +0.29 [-0.07, 0.61]; M1-value -0.03.
K4 fit: V1-perceptual -0.05 [-0.37, 0.28]; IPS-value +0.03 [-0.33, 0.39].
Per-segment IPS (IPS0, IPS1-2, IPS3-5), V1-3, per hemisphere, BA4: all |rho| < 0.15 with value noise; none of the predicted pairs is supported. Largest hit: M1 (BA4 L, hand area) vs perceptual noise ~ +0.3, and GM volume of M1 hand vs perceptual noise +0.42 [0.10, 0.71] - an unpredicted cross pair among many tests; treat as chance until replicated.
Area matrix: raw V1-IPS +0.35 [0.01, 0.61], V1-M1 +0.40 [0.02, 0.71], IPS-M1 +0.59 [0.29, 0.78]; size-corrected V1-IPS -0.15 [-0.47, 0.19], V1-M1 -0.08 [-0.43, 0.27], IPS-M1 -0.10 [-0.41, 0.23]. Raw areas covary because each tracks total cortical area (rho with total: V1 0.61, IPS 0.73, M1 0.82).
Decoding SD vs area (corrected, 100 voxels): V1 orientation -0.20 [-0.54, 0.18]; IPS1-2 value +0.10; IPS3-5 value -0.20 [-0.51, 0.15]; M1 value +0.26 [-0.08, 0.53]. With voxel count fixed, no sign that larger area gives more precise decoding (all CIs include 0). V1 with all voxels: -0.09.

## What it means for the multi-stage story
Beyond total brain size, the sizes of V1, IPS and the M1 hand area are mutually uncorrelated, which fits the idea of separate stages, but the individual noise levels are not predicted by the matching area either: the three predicted correlations are small with CIs spanning roughly -0.4 to +0.5, and the cross pairs look the same, so there is no specificity pattern in either direction. With n = 35 we can only exclude correlations beyond about |0.4|; this is neither evidence for nor against a size-noise link. The perceptual/value split is itself weakly identified, which lowers power further.

## Caveats
Exploratory; no multiplicity correction. Area uses atlas-defined (template-predicted) ROIs, so retinotopic area boundaries carry error, especially for IPS (Wang) and IPS4-5. M1 hand ROI is defined on fsaverage and mapped via sphere registration, not individually from function. Decoding SD depends on decoder settings (voxel selection, lambda). Skipped correlations unstable.
