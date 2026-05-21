"""Sample encoding-model R² volumes to fsnative + fsaverage for many subjects.

For each subject + model variant, finds the all-sessions R² NIfTI at
    derivatives/encoding_models/<model>/sub-XX/func/sub-XX_task-abstractvalue_space-T1w_desc-<desc>_pe.nii.gz
and writes the surface-sampled equivalents next to it:
    sub-XX_task-abstractvalue_hemi-{L,R}_space-fsnative_desc-<desc>_pe.func.gii
    sub-XX_task-abstractvalue_hemi-{L,R}_space-fsaverage_desc-<desc>_pe.func.gii

`<desc>` defaults to ``r2`` (encoding-model full-fit R²); pass
``--desc cvr2`` to sample the cross-validated R² volumes (produced by
``aggregate_cvr2.py`` for the ``*.cv`` model dirs). The smoothed variant is
selected via ``--smoothed``, which appends ``_smoothed`` to ``<desc>``.

The downstream `visualize_mean_r2_fsaverage.py` script loads the fsaverage
files and computes the group mean for pycortex display.

Usage:
    python -m abstract_values.surface.sample_r2_to_surface
    python -m abstract_values.surface.sample_r2_to_surface --subjects 07 08 09 10
    python -m abstract_values.surface.sample_r2_to_surface --models aprf vonmises aprf_session_shift
    python -m abstract_values.surface.sample_r2_to_surface --models aprf.cv vonmises.cv --desc cvr2
    python -m abstract_values.surface.sample_r2_to_surface --desc cvr2 --smoothed
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


def _desc_label(desc: str, smoothed: bool) -> str:
    """Return the on-disk `desc-...` token, suffixed with _smoothed if needed."""
    return f"{desc}_smoothed" if smoothed else desc


def discover_subjects(bids_folder: Path, model: str = "aprf",
                      desc: str = "r2", smoothed: bool = False) -> list[str]:
    """List subject labels with a `<desc>` NIfTI on disk for `model`."""
    base = Path(bids_folder) / "derivatives" / "encoding_models" / model
    label = _desc_label(desc, smoothed)
    out = []
    for p in sorted(base.glob("sub-*")):
        if not p.is_dir():
            continue
        fn = f"{p.name}_task-abstractvalue_space-T1w_desc-{label}_pe.nii.gz"
        if (p / "func" / fn).exists():
            out.append(p.name.removeprefix("sub-"))
    return out


def find_r2_volume(subject: str, model: str, bids_folder: Path,
                   desc: str = "r2", smoothed: bool = False) -> Path | None:
    label = _desc_label(desc, smoothed)
    p = (Path(bids_folder) / "derivatives" / "encoding_models" / model
         / f"sub-{subject}" / "func"
         / f"sub-{subject}_task-abstractvalue_space-T1w_desc-{label}_pe.nii.gz")
    return p if p.exists() else None


def main(subjects: list[str], models: list[str], session: int,
         bids_folder: Path, fmriprep_deriv: str, to_fsaverage: bool,
         desc: str = "r2", smoothed: bool = False) -> None:
    for subject in subjects:
        for model in models:
            vol = find_r2_volume(subject, model, bids_folder, desc, smoothed)
            if vol is None:
                label = _desc_label(desc, smoothed)
                print(f"sub-{subject} {model}: desc-{label} NIfTI not found — skipping")
                continue
            print(f"sub-{subject} {model}: sampling {vol.name}")
            outputs = sample_to_surfaces(
                vol, subject, session, bids_folder,
                fmriprep_deriv=fmriprep_deriv,
                to_fsaverage=to_fsaverage,
            )
            for (hemi, space), path in outputs.items():
                print(f"  -> {hemi} {space}: {path.name}")


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
    p.add_argument("--desc", default="r2",
                   help="BIDS `desc-` entity of the volume to sample "
                        "(default: r2; use 'cvr2' for cross-validated R² in *.cv dirs)")
    p.add_argument("--smoothed", action="store_true",
                   help="Sample the _smoothed variant (appends _smoothed to --desc)")
    p.add_argument("--no-fsaverage", action="store_true",
                   help="Skip the fsnative->fsaverage transform (only write fsnative)")
    args = p.parse_args()

    subjects = (args.subjects
                if args.subjects is not None
                else discover_subjects(Path(args.bids_folder),
                                       model=args.models[0],
                                       desc=args.desc,
                                       smoothed=args.smoothed))
    if not subjects:
        raise SystemExit("No subjects found — pass --subjects or check that "
                         f"desc-{_desc_label(args.desc, args.smoothed)} "
                         "NIfTIs exist locally.")

    main(subjects, args.models, args.session,
         Path(args.bids_folder), args.fmriprep_deriv,
         to_fsaverage=not args.no_fsaverage,
         desc=args.desc, smoothed=args.smoothed)
