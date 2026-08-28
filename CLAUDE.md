# Claude Code Context — Abstract Values

## Project overview

fMRI + behavioral study on abstract value learning. Participants learn orientation→CHF mappings via a BDM auction task. Full experiment description: `experiment/README.md`.

## Participant naming conventions

| Prefix | Example | Meaning |
|--------|---------|---------|
| `sub-##` | `sub-01` | Study participant. |
| `sub-pil##` | `sub-pil01` | **The same person as `sub-##`**, under the label the MRI side uses. |

**`sub-pil01` IS `sub-01`, and `sub-pil02` IS `sub-02`** — the first two participants,
labelled `pil` in the MRI data and numerically in the behavioural data. Verified
byte-for-byte identical events files. They are full study participants and belong in
every analysis.

Practical consequences:

- The behavioural loader (`behavior/data.py`) enumerates `sub-*` and keeps whatever
  parses as an int, so it picks up `sub-01`/`sub-02` and silently skips the `pil`
  directories — which is correct, and avoids double-counting, since the two hold the
  same data.
- The MRI side (BIDS root, fmriprep, GLMsingle, encoding models, expected uncertainty,
  `cvr2_surface_extent`) uses `sub-pil01`/`sub-pil02` only.
- **Joining brain to behaviour therefore needs `pil01 → 1`, `pil02 → 2`.** Any table
  that keys on the numeric id alone silently drops them; that is why
  `brain_behavior_subject_summary.tsv` has 26 rows rather than 28.

## Data access — `Subject` classes

There are **two `Subject` classes**, in different modules, with overlapping method names. Pick by what you need:

| Import | Source data | Methods |
|--------|-------------|---------|
| `from abstract_values.behavior.data import Subject` | `sourcedata/behavior/sub-##/` events TSVs | `get_behavioral_data()` |
| `from abstract_values.utils.data import Subject` | BIDS + fmriprep derivatives | `get_runs`, `get_sessions`, `get_events`, `get_mapping`, `get_roi_mask`, `get_single_trial_estimates`, ... |

Both share `subject_id`, `bids_folder`, `get_sessions()`, `get_runs()`, `get_mapping()`.

### Clean per-trial behavior — one-row-per-trial recipe

`Subject.get_behavioral_data()` returns one row **per event** (gabor, response_bar, feedback, isi, ...), not per trial. The BDM bid lands on the `feedback` row's `response` column. The canonical recipe to collapse to one row per trial (matches `notebooks/behavior_overview.ipynb`):

```python
from abstract_values.behavior.data import get_all_behavioral_data
import pandas as pd

df = get_all_behavioral_data()                                   # all study subjects
df = df[df["event_type"] == "feedback"].copy()                   # one row per trial
df["response"] = pd.to_numeric(df["response"], errors="coerce")  # NaN for non-responses
df["error"]     = df["response"] - df["value"]                   # BDM is truth-telling: value IS the rational bid
df["abs_error"] = df["error"].abs()
df = df.reset_index()                                            # flatten (subject, session, mapping, run, trial_nr)
```

Columns after this: `subject, session, mapping, run, trial_nr, response, value, orientation, rt, error, abs_error, invalid_response`.

**Frame-1 slider confirms are filtered by default.** The BDM slider re-randomises
its marker every trial (`experiment/response_slider.py :: random_init_marker`), so
a trial confirmed on the first frame records a uniform draw from `[0, 42]` CHF, not
a bid. Both `Subject` classes blank `response`/`rt` (behavior) and `bid`
(`utils.data.Subject.get_events`) for RTs below
`abstract_values.utils.data.MIN_VALID_RT` (0.25 s) and set `invalid_response=True`.
The cohort RT distribution is cleanly bimodal — a few trials at ~17 ms (one 60 Hz
frame), nothing else below 534 ms — so any threshold in `[0.02, 0.5]` picks the
same trials. Pass `min_rt=None` to any of these getters to keep the raw bids.

For a single subject: `Subject(3).get_behavioral_data()` + same filter.

For the per-trial brain×behavior table (decoded uncertainty + behavior), see `abstract_values/visualize/build_trial_table.py` — produces `notes/data/trial_table.tsv`.

## Key paths

| Path | Description |
|------|-------------|
| `/data/ds-abstractvalue/` | BIDS dataset root (local) |
| `/data/ds-abstractvalue/sourcedata/mri/` | Raw MRI data (pre-BIDS-conversion) |
| `/data/ds-abstractvalue/sourcedata/behavior/` | Raw behavioral logs from experiment |
| `/data/ds-abstractvalue/sub-*/` | BIDS-converted MRI data |
| `/data/ds-abstractvalue/derivatives/fmriprep/` | fmriprep outputs (T1w + T2w) |
| `/shares/zne.uzh/gdehol/ds-abstractvalue/` | Same dataset on cluster (sciencecluster) |
| `/shares/zne.uzh/containers/fmriprep-25.2.3` | fmriprep Apptainer container |

## BIDS conversion

Script: `fix_and_move_bids.py` (repo root)

```bash
# single subject/session
python fix_and_move_bids.py --subject pil01 --session 1 --dry-run
python fix_and_move_bids.py --subject pil01 --session 1

# all subjects found in sourcedata/mri
python fix_and_move_bids.py --all
```

What it does: copies anat (including FLAIR), func (adds task label + TaskName), fmap (fixes IntendedFor, strips zero-padding from run numbers).

After conversion, sync to cluster:
```bash
rsync -av /data/ds-abstractvalue/sub-<label> sciencecluster:/shares/zne.uzh/gdehol/ds-abstractvalue/
```

## fmriprep (cluster)

Scripts in `abstract_values/prepare/cluster_preproc/`:

| Script | Purpose |
|--------|---------|
| `fmriprep.sh` | T1w + T2w → `derivatives/fmriprep` (all participants) |

```bash
# submit study participants as array
sbatch --array=1-30 fmriprep.sh

# submit a pilot by name
sbatch --export=PARTICIPANT_LABEL=pil02 fmriprep.sh
```

Sync fmriprep results back to local (T1w-space only):
```bash
bash abstract_values/prepare/sync_fmriprep.sh
```

## Encoding models (abstract pRF)

Script: `abstract_values/encoding_models/fit_aprf.py`
Custom model classes: `abstract_values/encoding_models/models.py`

Fits a log-Gaussian pRF to single-trial GLMsingle betas using the **objective CHF value** of each gabor stimulus as the 1-D stimulus dimension. Uses braincoder (`ParameterFitter`): grid search (correlation cost) then Adam gradient descent.

### Model types

| `--model` | Parameters saved | Description |
|-----------|-----------------|-------------|
| `standard` (default) | `mode, fwhm, amplitude, baseline, r2` | Single log-Gaussian per voxel across all sessions. `mode_fwhm_natural` parameterisation. |
| `session-shift` | `mode_1, mode_2, fwhm, amplitude, baseline, r2` | Mode shifts freely per session; fwhm/amplitude/baseline shared. Requires ≥2 sessions. Implemented in `SessionShiftedLogGaussianPRF`. |
| `linear` | `amplitude, baseline, r2` | No tuning bump — signed slope (`amplitude`) + intercept (`baseline`) in CHF value. Fit by one closed-form OLS regression (`refine_baseline_and_amplitude`, `positive_amplitude=False`), not grid search + gradient descent. Baseline comparison for whether a voxel's value response is a tuned bump or a monotonic ramp. Implemented in `LinearValuePRF`. Output dir `aprf-linear` / `aprf-linear.cv`. |

### Output paths

Encoding models are **always fitted jointly across all of a subject's MRI
sessions** — the legacy per-session output path (`.../ses-<N>/...`) was
dropped, since downstream tools (FI, decoding, surface sampling) all assume
the joint fit. Session-shift variants legitimately need ≥2 sessions and
encode per-session information in their parameters (`mode_1`, `mode_2`) but
still live at the joint subject-level path.

```
derivatives/encoding_models/aprf/sub-<subject>/func/
derivatives/encoding_models/aprf-session-shift/sub-<subject>/func/
```

Files follow the pattern: `sub-<subject>_task-abstractvalue_space-T1w_desc-<param>_pe.nii.gz`

### SLURM job

Script: `abstract_values/encoding_models/slurm_jobs/fit_aprf.sh`
Resources: 8 CPUs, 32 GB RAM, 2 h wall time.

```bash
# standard model, single subject
sbatch --export=PARTICIPANT_LABEL=pil01 fit_aprf.sh

# session-shift model
sbatch --export=PARTICIPANT_LABEL=pil01,MODEL=session-shift fit_aprf.sh

# study participants as array
sbatch --array=1-30 fit_aprf.sh

# optional overrides: FMRIPREP_DERIV, SMOOTHED, N_ITERATIONS, MODEL
```

Logs: `/home/gdehol/logs/fit_aprf_<jobid>.txt`

## ROI masks

Volumetric masks (T1w space) live under:
```
derivatives/masks/sub-<subject>/anat/
```

### File naming

| Call | File loaded |
|------|-------------|
| `get_roi_mask('NPC', hemi='LR')` | `sub-<s>_space-T1w_hemi-LR_desc-NPC_mask.nii.gz` |
| `get_roi_mask('NPCr', hemi=None)` | `sub-<s>_space-T1w_desc-NPCr_mask.nii.gz` |
| `get_roi_mask('BensonV1', hemi='L')` | `sub-<s>_space-T1w_hemi-L_desc-BensonV1_mask.nii.gz` |

`hemi=None` omits the hemi entity entirely — required for NPCr/NPCl (which already encode hemisphere in the desc).

### Quick usage

```python
from abstract_values.utils.data import Subject, BIDS_FOLDER
sub  = Subject('pil01', bids_folder=BIDS_FOLDER)
mask = sub.get_roi_mask('NPCr', hemi=None)   # → NIfTI image
```

### How masks are made

Surface labels (fsaverage space) → fsnative (FreeSurfer `SurfaceTransform`) → T1w volume (neuropythy `cortex_to_image`).
Script: `abstract_values/surface/get_surface_roi_mask.py`

Input labels:
```
derivatives/surface_masks/desc-{roi}_{hemi}_space-fsaverage_hemi-{lh|rh}.label.gii
```

### ROIs in use

| ROI desc | Region |
|----------|--------|
| `NPC` / `NPCl` / `NPCr` | Numerosity Parietal Cortex (bilateral / left / right) |
| `BensonV1` … | Visual areas from Benson atlas |

**Default ROI for encoding model analyses: `NPCr` (`hemi=None`).**

## Data ingestion — full recipe

Script: `ingest_new_session.sh` (repo root). Skill: `/ingest`.

### End-to-end flow for a new subject

```
Network drive (SMB)  →  local sourcedata  →  BIDS root  →  cluster  →  SLURM chain
```

### Step-by-step

**1. Rsync source MRI from network drive to local sourcedata**
```bash
NETWORK="/Volumes/g_econ_department$/projects/2026/dehollander_bedi_ruff_abstract_values/data/sourcedata/mri"
rsync -av "$NETWORK/sub-{subject}/ses-{session}" /data/ds-abstractvalue/sourcedata/mri/sub-{subject}/
```

**2. BIDS conversion** (dry-run first, then real)
```bash
python fix_and_move_bids.py --subject {subject} --session {session} --dry-run
python fix_and_move_bids.py --subject {subject} --session {session}
```
Fixes: fmap IntendedFor, task label, FLAIR acq label, run zero-padding.

**3. Verify behavioral data**
```bash
ls /data/ds-abstractvalue/sourcedata/behavior/sub-{subject}/
```
Should already exist (copied by experiment script after each session).

**4. Rsync BIDS + behavior to cluster**
```bash
rsync -av /data/ds-abstractvalue/sub-{subject}/ sciencecluster:/shares/zne.uzh/gdehol/ds-abstractvalue/sub-{subject}/
rsync -av /data/ds-abstractvalue/sourcedata/behavior/sub-{subject}/ sciencecluster:/shares/zne.uzh/gdehol/ds-abstractvalue/sourcedata/behavior/sub-{subject}/
```

**5. Submit SLURM chain on cluster** (via `ingest_new_session.sh` or manually)
```bash
./ingest_new_session.sh --subject {subject} --session {max_session}
```
This submits (all chained with `--dependency=afterok`):
- fmriprep (full subject, all sessions) — 24h, 16 CPU, 64G
- GLMsingle x2 (unsmoothed + smoothed) — 4h, 16 CPU, 64G
- Encoding models (aprf, aprf_cv, session-shift, weighted, vonmises, etc.)
- Decoding (gabor + value, per ROI)
- Fisher information (vonmises + aprf, per ROI)

### Multi-session first-time ingestion

When ingesting a subject with multiple sessions for the first time:
1. Rsync + BIDS-convert ALL sessions first (steps 1-2 for each session)
2. Rsync ALL data to cluster (step 4 once — rsync sends everything)
3. Run `ingest_new_session.sh --subject {subject} --session {max_session}` — use the highest session number so session-shift models are included

**Do NOT run `ingest_new_session.sh` separately per session** — that creates redundant SLURM jobs.

### After fmriprep completes

ROI masks must be created before encoding model jobs can succeed:
```bash
# on cluster, after fmriprep finishes:
python abstract_values/prepare/create_roi_masks.py {subject} 1
```

### Monitoring

```bash
ssh sciencecluster squeue -u gdehol
ssh sciencecluster "tail -20 ~/logs/fmriprep_*.txt"
```

## Cognitive models (`abstract_values/cogmodels/`)

Efficient-coding models from Bedi et al. (2026), implemented in `bauer.efficient_coding`.
The paper itself is at `notes/papers/bedi_et_al2026.pdf`.

| `--model` | What it is |
|-----------|------------|
| `perception` | Efficient coding + Bayesian decoding in orientation space only (`kappa_r`) |
| `valuation` | Veridical perception; efficient coding in value space only (`sigma_rep`) |
| `sequential` | Both stages, perceptual uncertainty marginalised into the value stage |
| `categorical` | `sequential` + the paper's hard cardinal category gate at 90 deg / 22 CHF (Fig. 6, no extra free parameters) |

Flags that matter: `--perceptual-prior long_term|uniform`, `--fit-prior-weight`
(free steepness of the environmental prior), `--prior-fourier-order K` (fit the
prior shape as a K-harmonic circular Fourier series; k=1 horizontal-vs-vertical,
k=2 cardinal-vs-oblique — the paper's prior is a2 ~ 0.31 — k>=3 refinement under
a 0.5/(k-1)^2 roughness prior), `--no-seam-crossing` (1-3 deg is never decoded as
179-180 deg), `--condition` (fit one mapping only).
SLURM: `MODEL=`, `PRIOR=`, `FREE_PRIOR=1`, `FOURIER=K`, `NOSEAM=1`, `CHAIN_METHOD=`.

**Skills to load when working here** (these do not reliably self-trigger from a
debugging conversation, so load them explicitly):

- **bayesian-workflow** — before touching any fit, PPC, or convergence
  question. Covers the order of operations, divergence/r_hat/tree-depth triage,
  identifiability and parameter recovery, and the sampler traps
  (`chain_method="parallel"` silently serialising, `pm.Potential` breaking
  `log_likelihood`, JAX static-shape failures in hierarchical fits).
- **scientific-figures** — before writing any plotting code, including a PPC
  panel. `notes/figures/*.pdf` is the house output format.

## Cluster

Hostname: `sciencecluster`
Scratch: `/scratch/gdehol`
Logs: `/home/gdehol/logs/`
