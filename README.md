# Abstract Values

fMRI + behavioral study on abstract value learning. Participants learn a nonlinear mapping between visual stimuli (oriented Gabors) and monetary values (CHF), assessed via a Becker-DeGroot-Marschak auction task.

## Repository structure

```
abstract_values/        Python package (analysis code)
  prepare/
    cluster_preproc/    SLURM scripts for fmriprep on sciencecluster
    sync_fmriprep.sh    Sync fmriprep results from cluster to local
experiment/             PsychoPy task scripts + README
notebooks/              Analysis notebooks
fix_and_move_bids.py    BIDS conversion utility (sourcedata → BIDS root)
```

## Participants

There are **two distinct groups** — do not mix them up:

- **fMRI pilots** (`sub-pil01`, `sub-pil02`, …) — scanned to validate the MRI protocol and compare preprocessing options (e.g. with/without FLAIR). Not study participants.
- **Study participants** (`sub-01`, `sub-02`, …) — actual study sample, behavioral sessions and (later) fMRI sessions.

## Data

BIDS dataset: `/data/ds-abstractvalue/` (local) and `/shares/zne.uzh/gdehol/ds-abstractvalue/` (cluster).

```
ds-abstractvalue/
  sourcedata/
    mri/        Raw MRI scans (pre-BIDS, sub-pil## only for now)
    behavior/   Raw behavioral logs from experiment
  sub-pil01/    BIDS-converted MRI for pilot participant
  derivatives/
    fmriprep/           Main fmriprep output
    fmriprep-flair/     Pilot: preprocessed with T1w + FLAIR
    fmriprep-noflair/   Pilot: preprocessed with T1w only
```

## Preprocessing

See `experiment/README.md` for full task design details and `CLAUDE.md` for developer/analysis workflow notes.
