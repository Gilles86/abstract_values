"""Registry of the lab's fMRI datasets and file discovery across fmriprep versions.

Each dataset lists one or more fmriprep roots, searched in order, so a partial
copy in /scratch (e.g. confounds + anatomicals copied from the department share)
can complement the preprocessed BOLD that already lives on the cluster share.
Everything outside /scratch/gdehol/dropout_survey is read-only.

fmriprep layouts covered:
  1.4 / 20.x   anat aparcaseg (sub- or ses-level anat/), func-level
               space-T1w aparcaseg, desc-confounds_regressors (1.4) or
               desc-confounds_timeseries (20.x)
  23+ / 25.x   aparcaseg only if --output-spaces asked for it; otherwise
               the FreeSurfer aparc+aseg.mgz (sourcedata/freesurfer) is
               resampled onto the fmriprep T1w grid (fsnative->T1w is checked
               to be ~identity).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SHARES = Path("/shares/zne.uzh")
SCRATCH = Path("/scratch/gdehol/dropout_survey")


@dataclass
class Dataset:
    name: str
    label: str                        # short label for figures
    fmriprep: list                    # fmriprep roots, searched in order
    raw: list                         # raw BIDS roots, for sidecars / headers
    task: str | None = None           # regex on the task entity
    session: str | None = None        # regex on the ses entity
    extra: str | None = None          # regex the bold filename must match
    note: str = ""
    subjects: list | None = None      # optional explicit list

    def fmriprep_roots(self):
        return [Path(p) for p in self.fmriprep if Path(p).exists()]


def _fp(owner, ds):
    return str(SHARES / owner / ds / "derivatives" / "fmriprep")


DATASETS = {d.name: d for d in [
    Dataset("abstractvalue", "Abstract values", [_fp("gdehol", "ds-abstractvalue")],
            [str(SHARES / "gdehol/ds-abstractvalue")], task="abstractvalue", session="1"),
    Dataset("numrisk", "Numerosity risk", [_fp("gdehol", "ds-numrisk")],
            [str(SCRATCH / "ds-numrisk_raw")], task="numrisk",
            note="raw sidecars from the department share (garcia_ruff_riskprecision)"),
    Dataset("risk3t", "Risk 3T", [str(SCRATCH / "ds-risk/derivatives/fmriprep")],
            [str(SCRATCH / "ds-risk")], task="task", session="3t2",
            note="fmriprep copied from the department share (2 runs/subject)"),
    Dataset("risk7t", "Risk 7T", [str(SCRATCH / "ds-risk/derivatives/fmriprep")],
            [str(SCRATCH / "ds-risk")], task="task", session="7t2",
            note="fmriprep copied from the department share (2 runs/subject)"),
    Dataset("neuralpriors", "Neural priors", [_fp("gdehol", "ds-neuralpriors")],
            [str(SHARES / "gdehol/ds-neuralpriors")], task="task", session="1"),
    Dataset("tmsrisk", "TMS risk", [str(SCRATCH / "ds-tmsrisk/derivatives/fmriprep"),
                                    _fp("gdehol", "ds-tmsrisk")],
            [str(SCRATCH / "ds-tmsrisk")], task="task", session="1",
            note="confounds/anat from the department share, BOLD from the cluster"),
    Dataset("stressrisk", "Stress risk", [_fp("mrenke", "ds-stressrisk")],
            [str(SHARES / "mrenke/ds-stressrisk")], task="risk", session="1"),
    Dataset("dnumrisk", "Dot numerosity risk", [_fp("mrenke", "ds-dnumrisk")],
            [str(SHARES / "mrenke/ds-dnumrisk")], task="magjudge", session="1"),
    Dataset("numloss", "Numerosity loss", [_fp("aldavy", "ds-numloss")],
            [str(SHARES / "aldavy/ds-numloss")], task="numloss", session="1"),
    Dataset("foodvalue", "Food value", [_fp("gdehol", "ds-foodvalue")],
            [str(SHARES / "gdehol/ds-foodvalue")], task="food", session="01"),
    Dataset("retsupp", "Retsupp 7T", [_fp("gdehol", "ds-retsupp")],
            [str(SHARES / "gdehol/ds-retsupp")], task="search", session="1",
            note="NORDIC-denoised BOLD (the only fmriprep input)"),
    Dataset("valuecapture", "Value capture 7T", [_fp("gdehol", "ds-valuecapture")],
            [str(SHARES / "gdehol/ds-valuecapture")], task="valuecapture",
            note="NORDIC-denoised BOLD; pilot n=3; main task runs from both sessions"),
]}


def entities(name):
    return dict(re.findall(r"([a-zA-Z]+)-([a-zA-Z0-9]+)", Path(name).name.split(".")[0]))


def _sortkey(p):
    e = entities(p.name)
    return (e.get("ses", ""), int(e["run"]) if e.get("run", "").isdigit() else 0, p.name)


def subjects(ds: Dataset):
    if ds.subjects is not None:
        return list(ds.subjects)
    out = set()
    for root in ds.fmriprep_roots():
        for p in root.glob("sub-*"):
            if p.is_dir() and re.fullmatch(r"sub-[A-Za-z0-9]+", p.name):
                out.add(p.name[4:])
    return sorted(out)


def _find(ds, s, name):
    """First file called `name` anywhere under any root's sub-<s>/."""
    for root in ds.fmriprep_roots():
        hits = sorted((root / f"sub-{s}").glob(f"**/{name}"))
        if hits:
            return hits[0]
    return None


def find_runs(ds: Dataset, s, n_runs=2):
    """Up to n_runs (bold, mask, confounds) triples in T1w space, sorted by ses/run."""
    seen, cands = set(), []
    for root in ds.fmriprep_roots():
        for p in (root / f"sub-{s}").glob("**/func/*space-T1w_desc-preproc_bold.nii.gz"):
            if p.name in seen:
                continue
            e = entities(p.name)
            if ds.task and not re.fullmatch(ds.task, e.get("task", "")):
                continue
            if ds.session and not re.fullmatch(ds.session, e.get("ses", "")):
                continue
            if ds.extra and not re.search(ds.extra, p.name):
                continue
            seen.add(p.name)
            cands.append(p)
    runs = []
    for p in sorted(cands, key=_sortkey):
        prefix = p.name.split("_space-")[0]
        conf = (_find(ds, s, f"{prefix}_desc-confounds_timeseries.tsv")
                or _find(ds, s, f"{prefix}_desc-confounds_regressors.tsv"))
        mask = _find(ds, s, f"{prefix}_space-T1w_desc-brain_mask.nii.gz")
        if conf is None or mask is None:
            continue
        runs.append(dict(bold=p, mask=mask, confounds=conf, prefix=prefix))
        if len(runs) == n_runs:
            break
    return runs


def _anat(ds, s, suffix):
    for root in ds.fmriprep_roots():
        hits = sorted(p for p in (root / f"sub-{s}").glob(f"**/anat/sub-{s}*{suffix}")
                      if "space-" not in p.name and "task-" not in p.name)
        if hits:
            return hits[0]
    return None


def find_anat(ds: Dataset, s, runs):
    """aparcaseg source, preproc T1w, T1w->MNI xfm (None when missing)."""
    out = dict(t1w=_anat(ds, s, "desc-preproc_T1w.nii.gz"),
               xfm=_anat(ds, s, "from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"),
               fs2t1w=_anat(ds, s, "from-fsnative_to-T1w_mode-image_xfm.txt"))
    seg = _anat(ds, s, "desc-aparcaseg_dseg.nii.gz")
    src = "fmriprep anat aparcaseg"
    if seg is None and runs:
        seg = _find(ds, s, f"{runs[0]['prefix']}_space-T1w_desc-aparcaseg_dseg.nii.gz")
        src = "fmriprep func space-T1w aparcaseg"
    if seg is None:
        for root in ds.fmriprep_roots():
            for fs in (root / "sourcedata" / "freesurfer", root.parent / "freesurfer"):
                for d in [fs / f"sub-{s}"] + sorted(fs.glob(f"sub-{s}_ses-*")):
                    if (d / "mri" / "aparc+aseg.mgz").exists():
                        seg, src = d / "mri" / "aparc+aseg.mgz", "FreeSurfer aparc+aseg.mgz"
                        break
                if seg is not None:
                    break
            if seg is not None:
                break
    out["aparcaseg"], out["aparcaseg_source"] = seg, src if seg is not None else None
    return out


def inventory(names=None, n_runs=2):
    """One row per (dataset, subject): what was found."""
    import pandas as pd
    rows = []
    for name in names or DATASETS:
        ds = DATASETS[name]
        for s in subjects(ds):
            runs = find_runs(ds, s, n_runs)
            a = find_anat(ds, s, runs)
            rows.append(dict(dataset=name, subject=s, n_runs=len(runs),
                             runs=";".join(r["bold"].name for r in runs),
                             aparcaseg=a["aparcaseg_source"], t1w=a["t1w"] is not None,
                             xfm=a["xfm"] is not None))
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--out", default=str(SCRATCH / "inventory.tsv"))
    a = p.parse_args()
    df = inventory(a.datasets)
    df.to_csv(a.out, sep="\t", index=False)
    ok = df[(df.n_runs > 0) & df.aparcaseg.notna() & df.t1w]
    print(df.groupby("dataset").agg(n=("subject", "size"), with_runs=("n_runs", lambda x: (x > 0).sum()),
                                    aparc=("aparcaseg", lambda x: x.notna().sum()),
                                    xfm=("xfm", "sum")))
    print("usable:", ok.groupby("dataset").size().to_dict())
