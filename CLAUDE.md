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
| `/data/ds-abstractvalue/derivatives/fmriprep*/` | fmriprep outputs |
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
| `fmriprep.sh` | Main pipeline — T1w only → `derivatives/fmriprep` |
| `fmriprep_t2w.sh` | Pilot (pil02): T1w + T2w → `derivatives/fmriprep-t2w` |
| `fmriprep_acqlong.sh` | Pilot (pil01): T1w + acq-long FLAIR → `derivatives/fmriprep-acqlong` |
| `fmriprep_acqshort.sh` | Pilot (pil01): T1w + acq-short FLAIR → `derivatives/fmriprep-acqshort` |

```bash
# submit study participants as array
sbatch --array=1-30 fmriprep.sh

# submit a pilot by name
sbatch --export=PARTICIPANT_LABEL=pil02 fmriprep_t2w.sh
sbatch --export=PARTICIPANT_LABEL=pil01 fmriprep_acqlong.sh
sbatch --export=PARTICIPANT_LABEL=pil01 fmriprep_acqshort.sh
```

Sync fmriprep results back to local (T1w-space only):
```bash
bash abstract_values/prepare/sync_fmriprep.sh
```

## Cluster

Hostname: `sciencecluster`
Scratch: `/scratch/gdehol`
Logs: `/home/gdehol/logs/`
