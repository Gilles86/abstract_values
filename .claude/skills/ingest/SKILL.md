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
- `--scope <full|behavior|behavior-cluster>`: optional override
  - `full` (default): MRI + behavior, BIDS-convert, sync to cluster, submit SLURM chain
  - `behavior`: only rsync behavior from network drive → local; stop
  - `behavior-cluster`: rsync behavior network drive → local → cluster; no MRI, no SLURM

  If the user says things like "just curious about the behavior" or "I only want the
  behavioral logs", clarify scope before running — these are very different amounts of
  work. Behavior-only is also the right scope when MRI for the session hasn't been
  acquired yet (network drive has behavior/ses-N but no mri/ses-N).

## Constants

```
NETWORK_MRI="/Volumes/g_econ_department$/projects/2026/dehollander_bedi_ruff_abstract_values/data/sourcedata/mri"
NETWORK_BEHAVIOR="/Volumes/g_econ_department$/projects/2026/dehollander_bedi_ruff_abstract_values/data/sourcedata/behavior"
BIDS_ROOT="/data/ds-abstractvalue"
CLUSTER="sciencecluster"
CLUSTER_BIDS="/shares/zne.uzh/gdehol/ds-abstractvalue"
REPO_DIR="$HOME/git/abstract_values"
```

**Important: behavioral data lives on the network drive too, not just locally.** The
experiment script writes behavior to the network drive (the lab's SMB share), and
nothing on the local Mac pulls it automatically — *this skill is the thing that pulls
it down*. Do not assume `$BIDS_ROOT/sourcedata/behavior/sub-{subject}/ses-{session}/`
already exists; rsync it from `$NETWORK_BEHAVIOR` every time, like the MRI.

**Important: there is a delay between MRI acquisition and BIDS appearance on the
network drive.** The SNS lab's reconstruction/conversion pipeline runs after each
scan and pushes the converted data to `$NETWORK_MRI/sub-{subject}/ses-{session}/`
some time later (hours to a day). So immediately after a scan session, behavior may
already be on the network drive while MRI for that same session is not yet there.
If the user asks to ingest right after a session and MRI is missing for a session
that should have it, surface this — they may want to wait, or proceed with
behavior-only and re-run for MRI later.

**Policy: never start MRI ingestion for incomplete subjects.** Study subjects are
expected to have **2 MRI sessions**; pilots (`sub-pil*`) have **1**. Before
proceeding with `full` scope (MRI + SLURM submission), count
`$NETWORK_MRI/sub-{subject}/ses-*` and refuse if fewer than expected. Examples:

- Study sub-08 has only `ses-1` MRI on the network drive: refuse `full` scope,
  offer `behavior-cluster` instead (behavior-only is always fine for
  incomplete subjects).
- Pilot sub-pil02 has `ses-1` MRI: proceed (only 1 session expected).
- Study sub-07 has `ses-1` and `ses-2` MRI: proceed.

`ingest_new_session.sh` enforces this same check at the script level (exits
with code 2 if incomplete, override with `FORCE_INCOMPLETE=1`). Mirror the
policy in this skill — don't propose `full` scope when the network drive
shows incomplete MRI. The Snakemake pipeline (`abstract_values/snakemake/`)
also blocks at config-load via `require_complete: true`.

## Pipeline steps

For each subject, run these steps in order:

### Step 0: Discover sessions

Check both MRI and behavior network drives — they can diverge (e.g., behavior for ses-2
exists before the MRI session is acquired, or pure behavioral-pilot subjects have no MRI
at all).
```bash
ls "$NETWORK_MRI/sub-{subject}/"       2>/dev/null
ls "$NETWORK_BEHAVIOR/sub-{subject}/"  2>/dev/null
```
Report the union of sessions and which modalities each has. If `--session` was given,
only process that one. If a session has behavior but no MRI, that's fine — just run the
behavior-only path (skip Steps 1–3, 5).

### Step 1: Rsync source MRI data (network drive -> local sourcedata)

For each session that has MRI on the network drive, rsync from network drive to local sourcedata.
Run sessions in parallel.
```bash
rsync -av "$NETWORK_MRI/sub-{subject}/ses-{session}" "$BIDS_ROOT/sourcedata/mri/sub-{subject}/"
```

### Step 2: Rsync behavioral data (network drive -> local sourcedata)

**Always run this** — the behavior on local disk is stale or absent until this rsync
runs. For each session that has behavior on the network drive:
```bash
rsync -av "$NETWORK_BEHAVIOR/sub-{subject}/ses-{session}/" "$BIDS_ROOT/sourcedata/behavior/sub-{subject}/ses-{session}/"
```
Per-session rsync (not the whole subject) lets `--session N` actually restrict to that
session and avoids touching mtimes on other sessions.

### Step 3: BIDS conversion (MRI only)

Only for sessions that have MRI. First do a dry-run to verify, then run for real
(unless `--dry-run` flag).
```bash
~/mambaforge/envs/abstract_values/bin/python fix_and_move_bids.py --subject {subject} --session {session} --dry-run
~/mambaforge/envs/abstract_values/bin/python fix_and_move_bids.py --subject {subject} --session {session}
```
Run this from the repo root (`/Users/gdehol/git/abstract_values`).

### Step 4: Rsync BIDS data + behavior to cluster

Sync ALL sessions of the subject (not just the new one) to ensure the cluster has everything.
The BIDS sync is skipped under `--scope behavior` (no MRI processed); behavior sync still
runs under `--scope behavior-cluster`.
```bash
# only if any MRI was BIDS-converted this run:
rsync -av "$BIDS_ROOT/sub-{subject}/" "$CLUSTER:$CLUSTER_BIDS/sub-{subject}/"
# always (unless --scope behavior):
rsync -av "$BIDS_ROOT/sourcedata/behavior/sub-{subject}/" "$CLUSTER:$CLUSTER_BIDS/sourcedata/behavior/sub-{subject}/"
```

Under `--scope behavior`: stop here without syncing to cluster.

### Step 5: Submit SLURM chain on cluster

Skip this step entirely when `--scope` is `behavior` or `behavior-cluster`, or when no
MRI was processed for any session.

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

### Step 6: Refresh behavior overview notebook (local)

After behavioral data has been synced (Step 2, regardless of scope), always re-execute the
behavior overview notebook **locally** — it auto-discovers all subjects via
`get_all_subject_ids()`, so each ingest refreshes the cohort-wide figures and per-subject
summary table.
```bash
PYTHONUNBUFFERED=1 ~/mambaforge/envs/abstract_values/bin/jupyter nbconvert \
    --to notebook --execute --inplace notebooks/behavior_overview.ipynb
```
Run from the repo root. This is the **last** step — failures here must not block anything
upstream. Surface the error (`tail` the nbconvert traceback) but keep going.

Don't worry about git noise: `.gitattributes` configures `nbstripout` so executed outputs
are stripped on commit. Local working copy keeps outputs; collaborators see clean diffs.

### Step 7: Report summary

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
