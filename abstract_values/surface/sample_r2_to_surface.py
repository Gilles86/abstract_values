"""Sample encoding-model R² volumes to fsnative + fsaverage for many subjects.

For each subject + model variant, finds the all-sessions R² NIfTI at
    derivatives/encoding_models/<model>/sub-XX/func/sub-XX_task-abstractvalue_space-T1w_desc-r2_pe.nii.gz
and writes the surface-sampled equivalents next to it:
    sub-XX_task-abstractvalue_hemi-{L,R}_space-fsnative_desc-r2_pe.func.gii
    sub-XX_task-abstractvalue_hemi-{L,R}_space-fsaverage_desc-r2_pe.func.gii

The downstream `visualize_mean_r2_fsaverage.py` script loads the fsaverage
files and computes the group mean for pycortex display.

Usage:
    python -m abstract_values.surface.sample_r2_to_surface
    python -m abstract_values.surface.sample_r2_to_surface --subjects 07 08 09 10
    python -m abstract_values.surface.sample_r2_to_surface --models aprf vonmises aprf_session_shift
    python -m abstract_values.surface.sample_r2_to_surface --no-fsaverage   # fsnative only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from abstract_values.surface.sampling import sample_to_surfaces
from abstract_values.utils.data import BIDS_FOLDER

# Model variants we care about by default — the two whose R² maps anchor
# our downstream visualisation. Add more if you want to sample others
# (aprf_session_shift, aprf_weighted, ...).
DEFAULT_MODELS = ["aprf", "vonmises"]


def discover_subjects(bids_folder: Path, model: str = "aprf") -> list[str]:
    """List subject labels with an R² NIfTI on disk for `model`."""
    base = Path(bids_folder) / "derivatives" / "encoding_models" / model
    out = []
    for p in sorted(base.glob("sub-*")):
        if not p.is_dir():
            continue
        if (p / "func" / f"{p.name}_task-abstractvalue_space-T1w_desc-r2_pe.nii.gz").exists():
            out.append(p.name.removeprefix("sub-"))
    return out


def find_r2_volume(subject: str, model: str, bids_folder: Path) -> Path | None:
    p = (Path(bids_folder) / "derivatives" / "encoding_models" / model
         / f"sub-{subject}" / "func"
         / f"sub-{subject}_task-abstractvalue_space-T1w_desc-r2_pe.nii.gz")
    return p if p.exists() else None


def main(subjects: list[str], models: list[str], session: int,
         bids_folder: Path, fmriprep_deriv: str, to_fsaverage: bool) -> None:
    for subject in subjects:
        for model in models:
            vol = find_r2_volume(subject, model, bids_folder)
            if vol is None:
                print(f"sub-{subject} {model}: R² NIfTI not found — skipping")
                continue
            print(f"sub-{subject} {model}: sampling {vol.name}")
            outputs = sample_to_surfaces(
                vol, subject, session, bids_folder,
                fmriprep_deriv=fmriprep_deriv,
                to_fsaverage=to_fsaverage,
            )
            for (hemi, space), path in outputs.items():
                print(f"  → {hemi} {space}: {path.name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+",
                   help="Subject labels (default: discover from disk)")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help=f"Encoding-model variants to sample (default: {' '.join(DEFAULT_MODELS)})")
    p.add_argument("--session", type=int, default=1,
                   help="Session number for the fmriprep surfaces (default 1; "
                        "fmriprep stores anat under ses-1 for multi-session subjects)")
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--fmriprep-deriv", default="fmriprep")
    p.add_argument("--no-fsaverage", action="store_true",
                   help="Skip the fsnative→fsaverage transform (only write fsnative)")
    args = p.parse_args()

    subjects = (args.subjects
                if args.subjects is not None
                else discover_subjects(Path(args.bids_folder)))
    if not subjects:
        raise SystemExit("No subjects found — pass --subjects or check that "
                         "encoding-model R² NIfTIs exist locally.")

    main(subjects, args.models, args.session,
         Path(args.bids_folder), args.fmriprep_deriv,
         to_fsaverage=not args.no_fsaverage)
