"""Acquisition parameters of the runs the dropout survey used -- cluster-side.

Per (dataset, subject), for the first run collect.py uses:
  * BIDS sidecar with inheritance (top-level task-*_bold.json, subject- and
    session-level jsons, then the run json), plus fmriprep's copied json
  * voxel size / matrix from the raw NIfTI header (or, when the raw BOLD is not
    readable, fmriprep's native-space boldref, which keeps the scanner affine)
  * slice tilt: angle between the slice normal and scanner z, and between the
    slice normal and the MNI z axis (through the affine part of fmriprep's
    T1w->MNI transform; ignores between-session head repositioning)
  * fieldmaps in the raw fmap/ folder and the SDC method fmriprep reports

    python -m abstract_values.dropout_survey.acquisition --max-subjects 6
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from abstract_values.dropout_survey.datasets import (DATASETS, SCRATCH, _find, entities,
                                                     find_anat, find_runs)

FIELDS = ["Manufacturer", "ManufacturersModelName", "MagneticFieldStrength", "RepetitionTime",
          "EchoTime", "FlipAngle", "SliceThickness", "SpacingBetweenSlices",
          "MultibandAccelerationFactor", "ParallelReductionFactorInPlane",
          "ParallelAcquisitionTechnique", "PartialFourier", "PhaseEncodingDirection",
          "SliceEncodingDirection", "TotalReadoutTime", "EffectiveEchoSpacing",
          "PulseSequenceDetails", "SequenceName", "SoftwareVersions", "InPlanePhaseEncodingDirectionDICOM",
          "ImageOrientationPatientDICOM", "BandwidthPerPixelPhaseEncode", "WaterFatShift",
          "EchoTrainLength"]
IGNORE = {"space", "desc", "res", "part"}


def _subset(e_small, e_big):
    return all(e_big.get(k) == v for k, v in e_small.items() if k not in IGNORE)


def raw_sidecars(raw, s, ent):
    """Inherited raw sidecar dict, plus the raw bold path if it is readable."""
    raw = Path(raw)
    levels = [raw, raw / f"sub-{s}"]
    if "ses" in ent:
        levels += [raw / f"sub-{s}" / f"ses-{ent['ses']}"]
    levels += [levels[-1] / "func"]
    meta, used, bold = {}, [], None
    for lev in levels:
        try:
            js = sorted(lev.glob("*_bold.json"), key=lambda p: len(entities(p.name)))
        except PermissionError:
            continue
        for j in js:
            ej = entities(j.name)
            if _subset(ej, ent) and ej.get("part", "mag") == "mag":
                meta.update(json.loads(j.read_text()))
                used.append(str(j))
        try:
            for b in sorted(lev.glob("*_bold.nii*")):
                eb = entities(b.name)
                if _subset(eb, ent) and _subset({k: v for k, v in ent.items() if k not in IGNORE}, eb) \
                        and eb.get("part", "mag") == "mag":
                    bold = b
        except PermissionError:
            pass
    return meta, used, bold


def mni_rotation(xfm):
    """Linear part of the affine in fmriprep's T1w->MNI composite (maps MNI LPS -> T1w LPS)."""
    import h5py
    with h5py.File(str(xfm), "r") as f:
        for k in f["TransformGroup"]:
            g = f["TransformGroup"][k]
            t = g["TransformType"][0]
            t = t.decode() if isinstance(t, bytes) else str(t)
            if "Affine" in t or "MatrixOffset" in t:
                return np.asarray(g["TransformParameters"][:9], float).reshape(3, 3)
    return None


def tilt(affine, slice_axis, M=None):
    n = affine[:3, slice_axis] / np.linalg.norm(affine[:3, slice_axis])     # RAS
    out = dict(tilt_scanner_deg=float(np.degrees(np.arccos(min(1, abs(n[2]))))),
               slice_normal_ras=";".join(f"{x:.3f}" for x in n))
    if M is not None:
        n_lps = n * np.array([-1, -1, 1])
        m = M.T @ n_lps                     # plane normals map with the inverse transpose
        m = m / np.linalg.norm(m)
        out["tilt_mni_deg"] = float(np.degrees(np.arccos(min(1, abs(m[2])))))
        # signed pitch about the left-right axis (+: anterior end of the slab tilted up)
        out["pitch_mni_deg"] = float(np.degrees(np.arctan2(-m[1] * np.sign(m[2]), abs(m[2]))))
    return out


def sdc_method(ds, s, prefix):
    html = _find(ds, s, f"{prefix}_desc-summary_bold.html")
    if html is not None:
        txt = re.sub(r"<[^>]+>", " ", html.read_text(errors="ignore"))
        m = re.search(r"Susceptibility distortion correction:\s*([^\n]+?)\s{2,}", txt)
        if m:
            return m.group(1).strip()
    if _find(ds, s, f"{prefix}_desc-sdc_bold.svg") is not None:
        return "yes (sdc report figure)"
    for root in ds.fmriprep_roots():
        if list((root / f"sub-{s}").glob("**/fmap/*fmapid-*")):
            return "yes (fmriprep fmap derivatives)"
    return "unknown"


def one(ds, s):
    runs = find_runs(ds, s, 2)
    if not runs:
        return None
    run = runs[0]
    ent = {k: v for k, v in entities(run["bold"].name).items() if k not in IGNORE}
    row = dict(dataset=ds.name, subject=s, run=run["prefix"])
    meta, used = {}, []
    bold = None
    for raw in ds.raw:
        meta, used, bold = raw_sidecars(raw, s, ent)
        if meta or bold:
            break
    fp_json = Path(str(run["bold"]).replace(".nii.gz", ".json"))
    fp = json.loads(fp_json.read_text()) if fp_json.exists() else {}
    for k in FIELDS:
        v = meta.get(k, fp.get(k))
        row[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
    if row.get("RepetitionTime") is None:
        row["RepetitionTime"] = fp.get("RepetitionTime")
    row["sidecars"] = ";".join(used)
    hdr_src = bold
    if hdr_src is None:
        hdr_src = (_find(ds, s, f"{run['prefix']}_desc-coreg_boldref.nii.gz")
                   or _find(ds, s, f"{run['prefix']}_desc-hmc_boldref.nii.gz"))
    row["header_source"] = str(hdr_src) if hdr_src else None
    if hdr_src is not None:
        img = nib.load(str(hdr_src))
        z = img.header.get_zooms()
        row.update(vox_x=float(z[0]), vox_y=float(z[1]), vox_z=float(z[2]),
                   matrix="x".join(str(int(x)) for x in img.shape[:3]),
                   n_slices=int(img.shape[2]), descrip=img.header["descrip"].tobytes()
                   .decode(errors="ignore").strip("\x00"))
        ax = {"i": 0, "j": 1, "k": 2}.get(str(meta.get("SliceEncodingDirection", "k"))[0], 2)
        anat = find_anat(ds, s, runs)
        M = mni_rotation(anat["xfm"]) if anat["xfm"] else None
        row.update(tilt(img.affine, ax, M))
    raw_root = Path(ds.raw[0]) / f"sub-{s}"
    try:
        fm = sorted(raw_root.glob("**/fmap/*.nii*"))
        row["fmap_files"] = ";".join(sorted({re.sub(r".*_", "", p.name).split(".")[0] for p in fm}))
    except PermissionError:
        row["fmap_files"] = "unreadable"
    row["sdc"] = sdc_method(ds, s, run["prefix"])
    # fmriprep copies key raw metadata into its own json (e.g. TR, SliceTiming)
    row["fmriprep_json_keys"] = ";".join(sorted(fp))
    return row


def main(names, max_subjects, out):
    rows = []
    for name in names:
        ds = DATASETS[name]
        lst = SCRATCH / f"subjects_{name}.txt"
        subs = [l.strip() for l in lst.read_text().splitlines() if l.strip()] if lst.exists() else []
        n = 0
        for s in subs:
            try:
                r = one(ds, s)
            except Exception as e:      # keep the survey going, record why
                r = dict(dataset=name, subject=s, error=repr(e))
            if r is None:
                continue
            rows.append(r)
            print(r, flush=True)
            n += 1
            if n >= max_subjects:
                break
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="*", default=list(DATASETS))
    p.add_argument("--max-subjects", type=int, default=6)
    p.add_argument("--out", default=str(SCRATCH / "acquisition.tsv"))
    a = p.parse_args()
    main(a.datasets, a.max_subjects, a.out)
