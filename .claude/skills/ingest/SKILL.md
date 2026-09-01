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
NETWORK_BIDS="/Volumes/g_econ_department$/projects/2026/dehollander_bedi_ruff_abstract_values/data"
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

### Step 4b: Backup BIDS data to the department SMB share

Only if any MRI was BIDS-converted this run (mirrors the cluster-sync condition in
Step 4). The g_econ share's `sourcedata/` is already the master copy of raw MRI/behavior
(that's where Step 1/2 pull from) — this step backs up the **converted BIDS** tree
(`$BIDS_ROOT/sub-{subject}/`) alongside it, so the department archive holds essential
data in both raw and BIDS form, independent of the cluster copy.

Use openrsync-safe flags (macOS ships protocol-29 rsync; `-a` chokes on SMB ACLs) and
retry on the mount's occasional mid-transfer drop (see the **data-archival** skill):
```bash
for i in 1 2 3; do
  rsync -rt --modify-window=2 --partial "$BIDS_ROOT/sub-{subject}/" "$NETWORK_BIDS/sub-{subject}/" && break
  sleep 5
done
```
Serialize subjects (don't run multiple of these rsyncs concurrently against the SMB
mount). This step is a backup, not a dependency for anything downstream — if the mount
is flaky or off-VPN, warn and continue; don't block cluster submission on it.

### Step 5: Run the cluster pipeline (Snakemake driver)

Skip when `--scope` is `behavior`/`behavior-cluster` or no MRI was processed.

The cluster side is **Snakemake**, not `ingest_new_session.sh` (legacy). The full chain
(fmriprep → ROI masks → GLMsingle → encoding → decoding → Fisher info) lives in
`abstract_values/snakemake/Snakefile`; subjects are processed by being listed in
`config.yaml` and walked by a long-running driver (`run_driver.sh`, itself a SLURM job).
ROI masks are Snakemake rules now — no manual `create_roi_masks.py` step (the
"Prerequisite check" below is obsolete for this path).

**5a.** Nothing to edit — subjects and session counts are auto-discovered from
`{bids_folder}/sub-*/ses-*/func` at config-load (see Snakefile). Once Step 4's rsync has
landed the subject's BIDS data on the cluster, the next driver run (or resume) picks it
up automatically. (`subjects_include`/`subjects_exclude` in `config.yaml` exist only to
override auto-discovery for debugging — not something to touch for a normal ingest.)

**5b.** Decide whether to (re)submit the driver. **Only ever one driver per repo workdir**
— `run_driver.sh` runs `snakemake --unlock` on startup, so a second driver rips the lock
from a live one. Check first:
```bash
ssh sciencecluster 'squeue -u gdehol -h -o "%j %T" | grep -i snake'
```
- **No driver running** → `git pull` on cluster, `sbatch abstract_values/snakemake/run_driver.sh`.
- **Driver already running** (new subject arrived mid-run — common) → do **5a + `git pull`
  only** to *stage* the subject. Do **not** submit a second driver, and do **not** scancel
  to force a unified restart: a 24h-walltime driver will be resubmitted anyway, and
  Snakemake persistence resumes the in-progress subject + starts the staged one. Tell the
  user it begins at the next resubmit (offer to fire it once the current driver ends —
  safe; `run_driver.sh` bakes in `--unlock || true` + `--rerun-incomplete`).

  Cancelling is especially bad if fmriprep has entered FreeSurfer **recon-all**: `scancel`
  strands a `.../freesurfer/sub-XX*/scripts/IsRunning.lh+rh` lock and loses hours. Check
  before any cancel: `ls .../freesurfer/sub-{running}*/scripts/IsRunning* 2>/dev/null`.

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

### Step 6b: Surface bundle (local, after the cluster pipeline finishes)

Not part of the snakemake graph — the pycortex filestore and the flat maps live on
the Mac, not the cluster, so this can only run locally and only once the encoding
models are done. Mention it in the summary rather than running it during ingestion.

```bash
# first time for a subject: also submit the ~35 min flatten job
bash abstract_values/prepare/build_surface_bundle.sh {subject} --flatten
# re-run once autoflatten finishes to pick up the flatten view
bash abstract_values/prepare/build_surface_bundle.sh {subject}
```

Each step is skipped when its output exists, so re-running is cheap. It writes
`derivatives/qa/webgl/sub-{subject}/`; browse every processed subject with:

```bash
python -m abstract_values.visualize.webshow_surface_maps --serve-all
```

### Step 7: Report summary

Print a summary table of all submitted SLURM jobs with their IDs and dependencies.
Remind the user that step 6b still has to be run locally once the cluster jobs finish.

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
