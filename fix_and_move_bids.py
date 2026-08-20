#!/usr/bin/env python3
"""
Fix fmap IntendedFor fields and move MRI data from sourcedata to BIDS root.

Renames fmap files from zero-padded (run-01) to unpadded (run-1) to match
the func naming convention used throughout this dataset.

Fieldmap -> fMRI run mapping (fixed for this protocol: 8 runs, 3 fieldmaps):
  fmap run-1  ->  func run-1, run-2
  fmap run-2  ->  func run-3, run-4, run-5, run-6
  fmap run-3  ->  func run-7, run-8

Usage:
  # process a specific subject / session
  python fix_and_move_bids.py --subject pil001 --session 01

  # process all subjects / sessions found in sourcedata
  python fix_and_move_bids.py --all

  # dry run first (recommended)
  python fix_and_move_bids.py --all --dry-run
"""

import argparse
import json
import re
import shutil
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
BIDS_ROOT = Path("/data/ds-abstractvalue")
SOURCEDATA = BIDS_ROOT / "sourcedata" / "mri"

# ── fmap -> func run mapping ───────────────────────────────────────────────────
# Keys: destination fmap run number (unpadded string)
# Values: list of func run numbers this fieldmap covers
FMAP_TO_FUNC: dict[str, list[str]] = {
    "1": ["1", "2"],
    "2": ["3", "4", "5", "6"],
    "3": ["7", "8"],
}

FMAP_TYPES = ["magnitude1", "magnitude2", "phasediff"]
TASK_LABEL = "abstractvalue"
TASK_NAME  = "Abstract Values"

# Per-(subject, session) FLAIR acquisition label.
# None (or missing key) means: copy with original filename unchanged.
# pil01: ses-1 → acq-long, ses-2 → acq-short
# pil02: ses-1 → acq-short, ses-2 → acq-long  (vice-versa)
FLAIR_ACQ: dict[tuple[str, str], str] = {
    ("pil01", "1"): "long",
    ("pil01", "2"): "short",
    ("pil02", "1"): "short",
    ("pil02", "2"): "long",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def strip_zero_pad(run_str: str) -> str:
    """'01' -> '1', '08' -> '8', '1' -> '1'."""
    return str(int(run_str))


def intended_for(subject: str, session: str, func_runs: list[str]) -> list[str]:
    return [
        f"bids::sub-{subject}/ses-{session}/func/"
        f"sub-{subject}_ses-{session}_task-{TASK_LABEL}_run-{r}_bold.nii.gz"
        for r in func_runs
    ]


def write_json(path: Path, data: dict, dry_run: bool) -> None:
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=4) + "\n")


def copy_file(src: Path, dst: Path, dry_run: bool) -> None:
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


# ── per-modality processing ────────────────────────────────────────────────────

def process_fmap(src_dir: Path, dst_dir: Path,
                 subject: str, session: str, dry_run: bool) -> None:
    """Fix IntendedFor, rename run-0N -> run-N, copy to dst."""
    # Build a map from zero-padded source run -> unpadded dest run
    # Infer from filenames present in source
    src_runs_padded = sorted({
        m.group(1)
        for f in src_dir.iterdir()
        if (m := re.search(r"_run-(\d+)_", f.name))
    })

    if len(src_runs_padded) != len(FMAP_TO_FUNC):
        print(f"  WARNING: found {len(src_runs_padded)} fmap runs, "
              f"expected {len(FMAP_TO_FUNC)}. Check FMAP_TO_FUNC.")

    run_map = {
        padded: strip_zero_pad(padded)
        for padded in src_runs_padded
    }

    for padded, unpadded in run_map.items():
        func_runs = FMAP_TO_FUNC.get(unpadded)
        if func_runs is None:
            print(f"  WARNING: no FMAP_TO_FUNC entry for run-{unpadded}, skipping")
            continue

        for fmap_type in FMAP_TYPES:
            prefix = f"sub-{subject}_ses-{session}"
            src_json = src_dir / f"{prefix}_run-{padded}_{fmap_type}.json"
            src_nii  = src_dir / f"{prefix}_run-{padded}_{fmap_type}.nii.gz"
            dst_json = dst_dir / f"{prefix}_run-{unpadded}_{fmap_type}.json"
            dst_nii  = dst_dir / f"{prefix}_run-{unpadded}_{fmap_type}.nii.gz"

            if not src_json.exists():
                print(f"  WARNING: missing {src_json.name}")
                continue

            # Fix IntendedFor
            data = json.loads(src_json.read_text())
            data["IntendedFor"] = intended_for(subject, session, func_runs)
            print(f"  {dst_json.name}  ->  runs {func_runs}")
            write_json(dst_json, data, dry_run)

            if src_nii.exists():
                print(f"  {dst_nii.name}")
                copy_file(src_nii, dst_nii, dry_run)


def _n_volumes(path: Path) -> int | None:
    """Number of volumes in a BOLD file, read from the NIfTI header only."""
    try:
        import nibabel as nb
        shape = nb.load(str(path)).shape
    except Exception:
        return None
    return shape[3] if len(shape) > 3 else 1


def _detect_aborted_bold(src_dir: Path, threshold: float = 0.9,
                         size_threshold: float = 0.5) -> set[Path]:
    """Return set of BOLD .nii.gz files that are likely aborted/partial runs.

    The primary signal is the **volume count** from the NIfTI header: every
    completed run of this protocol has the same number of volumes, so anything
    below `threshold` × the median volume count was aborted and restarted.

    File size (< `size_threshold` × the median) is only a fallback for when the
    headers cannot be read.  Size is far too blunt on its own: sub-28 ses-1's
    aborted 222-volume Run_1 is ~60 % of a full 367-volume run, so it sails past
    the size cut-off and would silently shift every later run by one.
    """
    bold_files = sorted(
        f for f in src_dir.iterdir()
        if re.search(r"_bold\.nii\.gz$", f.name)
    )
    if len(bold_files) < 2:
        return set()

    n_vols = [_n_volumes(f) for f in bold_files]
    if all(n is not None for n in n_vols):
        median_vols = sorted(n_vols)[len(n_vols) // 2]
        return {f for f, n in zip(bold_files, n_vols) if n < threshold * median_vols}

    print("  WARNING: could not read BOLD headers, falling back to file size "
          "to detect aborted runs")
    sizes = [f.stat().st_size for f in bold_files]
    median_size = sorted(sizes)[len(sizes) // 2]
    return {f for f, sz in zip(bold_files, sizes) if sz < size_threshold * median_size}


def _bold_run_renumber_map(src_dir: Path, aborted_nii: set[Path]) -> dict[str, str]:
    """Build a map from original run labels to sequential 1-based labels.

    Surviving (non-aborted) BOLD runs are sorted by SeriesNumber from their
    JSON sidecar (falling back to the run label itself) and renumbered 1..N.
    Returns e.g. {'05': '1', '2': '2', '3': '3', ...}.
    If no renumbering is needed (runs are already 1..N), returns an empty dict.
    """
    aborted_stems = {f.name.replace(".nii.gz", "") for f in aborted_nii}

    # Collect surviving BOLD run labels with their sort key
    run_info: list[tuple[int, str]] = []   # (series_number, original_run_label)
    for f in sorted(src_dir.iterdir()):
        if not re.search(r"_bold\.nii\.gz$", f.name):
            continue
        if f in aborted_nii:
            continue
        m = re.search(r"_run-(\d+)_bold", f.name)
        if not m:
            continue
        run_label = m.group(1)
        # Try to get SeriesNumber from JSON sidecar
        json_f = f.with_name(f.name.replace(".nii.gz", ".json"))
        series = int(run_label)  # fallback: sort by label
        if json_f.exists():
            try:
                series = json.loads(json_f.read_text()).get("SeriesNumber", series)
            except (json.JSONDecodeError, KeyError):
                pass
        run_info.append((series, run_label))

    run_info.sort()
    expected = [str(i + 1) for i in range(len(run_info))]
    actual   = [label for _, label in run_info]

    if actual == expected:
        return {}

    return {label: new for (_, label), new in zip(run_info, expected)}


def process_func(src_dir: Path, dst_dir: Path,
                 subject: str, session: str, dry_run: bool) -> None:
    """Copy func files, inserting task label and TaskName into BOLD files.

    Aborted/partial BOLD runs (fewer volumes than a completed run) are detected
    and skipped with a warning.  Surviving BOLD runs are renumbered 1..N in
    acquisition order (by SeriesNumber) so they match the behavioral logs.
    """
    aborted_nii = _detect_aborted_bold(src_dir)
    aborted_stems = {f.name.replace(".nii.gz", "") for f in aborted_nii}

    if aborted_nii:
        print(f"\n  WARNING: skipping {len(aborted_nii)} aborted/partial BOLD run(s):")
        for f in sorted(aborted_nii):
            n = _n_volumes(f)
            vols = f"{n} volumes, " if n is not None else ""
            print(f"    {f.name}  ({vols}{f.stat().st_size / 1e6:.1f} MB)")

    renumber = _bold_run_renumber_map(src_dir, aborted_nii)
    if renumber:
        print(f"\n  Renumbering BOLD runs: "
              + ", ".join(f"run-{old}→run-{new}" for old, new in renumber.items()))

    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_file in sorted(src_dir.iterdir()):
        # Skip aborted BOLD runs and their sidecars
        if src_file in aborted_nii:
            continue
        if src_file.name.replace(".json", "") in aborted_stems and "_bold." in src_file.name:
            continue

        # BOLD files: insert task-<label> if not already present, and patch JSON
        if re.search(r"_run-\d+_bold\.(nii\.gz|json)$", src_file.name):
            dst_name = src_file.name
            if f"_task-{TASK_LABEL}_" not in dst_name:
                dst_name = re.sub(
                    r"(_run-\d+_bold)",
                    f"_task-{TASK_LABEL}\\1",
                    dst_name,
                )
            # Apply run renumbering
            if renumber:
                m = re.search(r"_run-(\d+)_bold", dst_name)
                if m and m.group(1) in renumber:
                    dst_name = dst_name.replace(
                        f"_run-{m.group(1)}_bold",
                        f"_run-{renumber[m.group(1)]}_bold",
                    )
            dst_file = dst_dir / dst_name
            if src_file.suffix == ".json":
                data = json.loads(src_file.read_text())
                data["TaskName"] = TASK_NAME
                print(f"  {dst_name}  (TaskName added)")
                if not dry_run:
                    dst_file.write_text(json.dumps(data, indent=4) + "\n")
            else:
                print(f"  {dst_name}")
                copy_file(src_file, dst_file, dry_run)
        else:
            dst_file = dst_dir / src_file.name
            print(f"  {src_file.name}")
            copy_file(src_file, dst_file, dry_run)


def process_anat(src_dir: Path, dst_dir: Path,
                 subject: str, session: str, dry_run: bool) -> None:
    """Copy anat files, renaming FLAIR with acq label when configured."""
    flair_acq = FLAIR_ACQ.get((subject, session))
    for src_file in sorted(src_dir.iterdir()):
        if (flair_acq and re.search(r"_FLAIR\.(nii\.gz|json)$", src_file.name)
                and f"_acq-{flair_acq}_" not in src_file.name):
            dst_name = re.sub(r"(_FLAIR)", f"_acq-{flair_acq}\\1", src_file.name)
        else:
            dst_name = src_file.name
        dst_file = dst_dir / dst_name
        label = f"  {src_file.name}"
        if dst_name != src_file.name:
            label += f"  ->  {dst_name}"
        print(label)
        copy_file(src_file, dst_file, dry_run)


# ── dataset_description.json ───────────────────────────────────────────────────

DATASET_DESCRIPTION = {
    "Name": "Abstract Values",
    "BIDSVersion": "1.9.0",
    "DatasetType": "raw",
    "License": "CC0",
    "Authors": [
        "Gilles de Hollander"
    ],
    "Acknowledgements": "Data collected at the SNS Lab, University of Zurich.",
    "ReferencesAndLinks": [],
    "DatasetDOI": ""
}


def ensure_dataset_description(dry_run: bool) -> None:
    dst = BIDS_ROOT / "dataset_description.json"
    if dst.exists():
        return
    print(f"\nCreating {dst.name}")
    write_json(dst, DATASET_DESCRIPTION, dry_run)


# ── main ───────────────────────────────────────────────────────────────────────

def process_subject_session(subject: str, session: str, dry_run: bool) -> None:
    src_sub = SOURCEDATA / f"sub-{subject}" / f"ses-{session}"
    dst_sub = BIDS_ROOT   / f"sub-{subject}" / f"ses-{session}"

    if not src_sub.exists():
        raise FileNotFoundError(f"Source not found: {src_sub}")

    print(f"\n=== sub-{subject}  ses-{session} ===")

    if (src_sub / "fmap").exists():
        print("\n[fmap] fixing IntendedFor + renaming run-0N -> run-N:")
        process_fmap(src_sub / "fmap", dst_sub / "fmap", subject, session, dry_run)
    else:
        print("\n[fmap] not found, skipping")

    if (src_sub / "anat").exists():
        print("\n[anat]")
        process_anat(src_sub / "anat", dst_sub / "anat", subject, session, dry_run)

    if (src_sub / "func").exists():
        print("\n[func]")
        process_func(src_sub / "func", dst_sub / "func", subject, session, dry_run)


def discover_subject_sessions() -> list[tuple[str, str]]:
    pairs = []
    for sub_dir in sorted(SOURCEDATA.glob("sub-*")):
        subject = sub_dir.name.removeprefix("sub-")
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            session = ses_dir.name.removeprefix("ses-")
            pairs.append((subject, session))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subject", metavar="ID",
                       help="Subject ID without 'sub-' prefix, e.g. pil001")
    group.add_argument("--all", action="store_true",
                       help="Process all subjects/sessions found in sourcedata")
    parser.add_argument("--session", metavar="ID", default=None,
                        help="Session ID without 'ses-' prefix (required with --subject)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing anything")
    args = parser.parse_args()

    if args.subject and not args.session:
        parser.error("--session is required when using --subject")

    dry_run = args.dry_run
    if dry_run:
        print("=== DRY RUN — nothing will be written ===")

    ensure_dataset_description(dry_run)

    if args.all:
        pairs = discover_subject_sessions()
        if not pairs:
            print("No subjects found in sourcedata.")
            return
        for subject, session in pairs:
            process_subject_session(subject, session, dry_run)
    else:
        process_subject_session(args.subject, args.session, dry_run)

    print("\nDone." + (" (dry run)" if dry_run else ""))


if __name__ == "__main__":
    main()
