# Claude Code Context — Abstract Values

## Project overview

fMRI + behavioral study on abstract value learning. Participants learn orientation→CHF mappings via a BDM auction task. Full experiment description: `experiment/README.md`.

## Participant naming conventions

| Prefix | Example | Meaning |
|--------|---------|---------|
| `sub-pil##` | `sub-pil01` | **fMRI pilot** — used to test/validate the MRI protocol. Not study participants. |
| `sub-##` | `sub-01` | **Study participants** — behavioral-only or full fMRI study participants. |

**Do not confuse these.** The pilot MRI data lives under `sourcedata/mri/sub-pil##`. Study behavioral data lives under `sourcedata/behavior/sub-##`.

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

### Output paths

```
derivatives/encoding_models/aprf/sub-<subject>/[ses-<N>/]func/
derivatives/encoding_models/aprf-session-shift/sub-<subject>/[ses-<N>/]func/
```

Files follow the pattern: `sub-<subject>[_ses-<N>]_task-abstractvalue_space-T1w_desc-<param>_pe.nii.gz`

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

# optional overrides: SESSION, FMRIPREP_DERIV, SMOOTHED, N_ITERATIONS, MODEL
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

## Cluster

Hostname: `sciencecluster`
Scratch: `/scratch/gdehol`
Logs: `/home/gdehol/logs/`
