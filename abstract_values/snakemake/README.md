# abstract_values cluster pipeline (Snakemake)

Replaces the `ingest_new_session.sh` SLURM chain (steps 4–17) with a Snakemake
workflow. Pre-cluster steps (network-drive rsync, BIDS conversion, push to
cluster) stay in the bash script — Snakemake takes over from fmriprep onward.

## Layout

| File | Purpose |
|---|---|
| `Snakefile` | Rule definitions |
| `config.yaml` | Subjects, ROIs, hyperparameters, expected-session counts |
| `profile/config.yaml` | Workflow profile (SLURM executor, jobs, polling, default resources) |
| `run_driver.sh` | sbatch wrapper to run the driver itself as a SLURM job |
| `rulegraph.svg` | Rule-level DAG (compact, regenerate with `--rulegraph`) |
| `dag_sub-07.svg` | Full per-wildcard DAG for one subject (regenerate as needed) |

## Completeness gate

`config.yaml` declares `expected_mri_sessions` (default `2`, override per
subject in `subject_expected_overrides`). At config-load the Snakefile raises
`WorkflowError` if any listed subject has fewer MRI sessions than expected.
Override with `--config require_complete=false` for debug runs only.

Same check lives in `ingest_new_session.sh` (`FORCE_INCOMPLETE=1` to override)
and is documented in the `/ingest` skill.

## How to run on the cluster

```bash
ssh sciencecluster 'cd ~/git/abstract_values && \
    git pull && \
    sbatch abstract_values/snakemake/run_driver.sh'
```

The driver runs as a 24h SLURM job on the `lowprio` partition. It submits
per-rule sbatch jobs via `snakemake-executor-plugin-slurm`. Re-submitting the
same script resumes where the previous driver left off (Snakemake persistence
under `.snakemake/`).

Monitor:

```bash
ssh sciencecluster 'squeue -u $USER -h -O "JobID:14,State:10,TimeUsed:10,Comment:80"'
ssh sciencecluster 'tail -f ~/logs/snake_driver_*.log'
```

## Re-rendering the graphs

```bash
# Rule-level (compact, recommended for docs)
snakemake \
    --snakefile abstract_values/snakemake/Snakefile \
    --configfile abstract_values/snakemake/config.yaml \
    --config require_complete=false \
    --rulegraph | dot -Tsvg > abstract_values/snakemake/rulegraph.svg

# Full per-wildcard DAG for a single subject
snakemake \
    --snakefile abstract_values/snakemake/Snakefile \
    --configfile abstract_values/snakemake/config.yaml \
    --config require_complete=false subjects='["07"]' \
    --dag | dot -Tsvg > abstract_values/snakemake/dag_sub-07.svg
```

## Conda env on the cluster

Driver wants its own env (so versions don't drift with `abstract_values`):

```bash
mamba create -n abstract_values_snake -c bioconda -c conda-forge \
    snakemake snakemake-executor-plugin-slurm 'pulp<2.8'
```

## Not yet wired

- `shell:` blocks reference the existing `.sh` job scripts via env vars to
  match how `ingest_new_session.sh` invokes them. End-to-end execution will
  need a one-pass validation run (`--dry-run` on the cluster, then a small
  subject smoke test) before being trusted on real subjects.
- GLMsingle outputs are sentinel-based (`_done.touch`) — fine for the DAG
  but means manual deletion of the actual NIfTIs won't trigger a rerun.
  Worth replacing with a real representative file once the GLMsingle output
  tree is stable.
