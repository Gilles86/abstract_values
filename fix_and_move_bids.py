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


def process_func(src_dir: Path, dst_dir: Path,
                 subject: str, session: str, dry_run: bool) -> None:
    """Copy func files, inserting task label and TaskName into BOLD files."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_file in sorted(src_dir.iterdir()):
        # BOLD files: insert task-<label> and patch JSON
        if re.search(r"_run-\d+_bold\.(nii\.gz|json)$", src_file.name):
            dst_name = re.sub(
                r"(_run-\d+_bold)",
                f"_task-{TASK_LABEL}\\1",
                src_file.name,
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


def process_modality(src_dir: Path, dst_dir: Path, dry_run: bool) -> None:
    """Copy all files unchanged (anat)."""
    for src_file in sorted(src_dir.iterdir()):
        dst_file = dst_dir / src_file.name
        print(f"  {src_file.name}")
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

    print("\n[fmap] fixing IntendedFor + renaming run-0N -> run-N:")
    process_fmap(src_sub / "fmap", dst_sub / "fmap", subject, session, dry_run)

    if (src_sub / "anat").exists():
        print("\n[anat]")
        process_modality(src_sub / "anat", dst_sub / "anat", dry_run)

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
