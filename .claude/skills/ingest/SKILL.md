---
name: ingest
description: Ingest new MRI sessions from the network drive, BIDS-convert, sync to cluster, and submit the full fmriprep/GLMsingle/encoding-model SLURM pipeline. Use when the user says "ingest", "process new subject", or "prepare MRI data".
user-invocable: true
argument-hint: "<subject> [subject2 ...] [--dry-run] [--session N]"
---

# MRI Data Ingestion Pipeline

You are running the full data ingestion pipeline for the abstract_values project.
This takes raw MRI data from the network drive through BIDS conversion, cluster sync,
and submits the complete SLURM analysis chain.

## Arguments

Parse the user's arguments:
- One or more subject labels (e.g., `03`, `04`, `pil01`). Accept with or without `sub-` prefix.
- `--dry-run`: only show what would be done, don't write or submit
- `--session N`: only process session N (default: all sessions found on network drive)

## Constants

```
NETWORK_BASE="/Volumes/g_econ_department$/projects/2026/dehollander_bedi_ruff_abstract_values/data/sourcedata/mri"
BIDS_ROOT="/data/ds-abstractvalue"
CLUSTER="sciencecluster"
CLUSTER_BIDS="/shares/zne.uzh/gdehol/ds-abstractvalue"
REPO_DIR="$HOME/git/abstract_values"
```

## Pipeline steps

For each subject, run these steps in order:

### Step 0: Discover sessions
```bash
ls "$NETWORK_BASE/sub-{subject}/"
```
Report which sessions are available. If `--session` was given, only process that one.

### Step 1: Rsync source MRI data (network drive -> local sourcedata)

For each session, rsync from network drive to local sourcedata. Run sessions in parallel.
```bash
rsync -av "$NETWORK_BASE/sub-{subject}/ses-{session}" "$BIDS_ROOT/sourcedata/mri/sub-{subject}/"
```

### Step 2: BIDS conversion

First do a dry-run to verify, then run for real (unless `--dry-run` flag).
```bash
~/mambaforge/envs/abstract_values/bin/python fix_and_move_bids.py --subject {subject} --session {session} --dry-run
~/mambaforge/envs/abstract_values/bin/python fix_and_move_bids.py --subject {subject} --session {session}
```
Run this from the repo root (`/Users/gdehol/git/abstract_values`).

### Step 3: Verify behavioral data exists locally
```bash
ls "$BIDS_ROOT/sourcedata/behavior/sub-{subject}/"
```
Behavioral data should already be there (copied by the experiment script). Warn if missing.

### Step 4: Rsync BIDS data + behavior to cluster

Sync ALL sessions of the subject (not just the new one) to ensure the cluster has everything.
```bash
rsync -av "$BIDS_ROOT/sub-{subject}/" "$CLUSTER:$CLUSTER_BIDS/sub-{subject}/"
rsync -av "$BIDS_ROOT/sourcedata/behavior/sub-{subject}/" "$CLUSTER:$CLUSTER_BIDS/sourcedata/behavior/sub-{subject}/"
```

### Step 5: Submit SLURM chain on cluster

Use the `ingest_new_session.sh` script's cluster section as reference, but submit directly via SSH.
The key is to pass `--session` as the HIGHEST session number so that session-shift models are included.

```bash
ssh sciencecluster bash <<'REMOTE'
set -euo pipefail
cd ~/git/abstract_values && git pull --ff-only

# ... submit fmriprep, GLMsingle, encoding models ...
# (see ingest_new_session.sh steps 4-17 for the full SLURM chain)
REMOTE
```

**IMPORTANT**: Rather than reimplementing the SLURM chain, use `ingest_new_session.sh` on the cluster
if only one session is being added. For multi-session first-time ingestion, submit the chain manually
with the highest session number to include session-shift models.

Actually, the simplest approach: run `ingest_new_session.sh` from the LOCAL machine, which SSHs to
the cluster. But skip steps 1-3 since we already did them. So just run the cluster portion.

For new subjects with multiple sessions to ingest at once:
1. Do steps 1-4 for ALL sessions first
2. Then run `./ingest_new_session.sh --subject {subject} --session {max_session}` but only
   the cluster steps (4+). Since steps 1-3 are idempotent (rsync), it's fine to re-run the
   full script for the last session.

### Step 6: Report summary

Print a summary table of all submitted SLURM jobs with their IDs and dependencies.

## Prerequisite check

Before step 5, verify that ROI masks exist for the subject:
```bash
ssh sciencecluster "ls $CLUSTER_BIDS/derivatives/masks/sub-{subject}/ses-1/anat/ 2>/dev/null | head -5"
```
If masks don't exist, warn the user that encoding model jobs will fail until masks are created
(run `create_roi_masks.py` after fmriprep completes). The fmriprep and GLMsingle jobs will
still run fine.

## Error handling

- If the network drive is not mounted, tell the user to mount it first
- If a subject doesn't exist on the network drive, skip and warn
- If BIDS conversion fails, stop and show the error
- If cluster SSH fails, report the error
