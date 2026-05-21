"""Generic NIfTI-to-surface sampling helpers for fmriprep-processed projects.

Two-step pipeline, wrapping what `sample_aprf_to_surface.py` and friends
were doing by hand:

1. `nilearn.surface.vol_to_surf` with pial+white for grey-matter sampling
   → per-hemisphere `.func.gii` in fsnative space.
2. (optional) FreeSurfer `SurfaceTransform` (via nipype) → fsaverage space.

Output filenames follow BIDS-derivative convention: insert `_hemi-{L,R}`
before `_space-`, swap `_space-T1w` → `_space-fsnative` (and `_space-fsaverage`),
and change extension to `.func.gii`. Outputs land next to the input volume.

See the cogneuro-project skill's surface_sampling reference for the
end-to-end recipe, the FreeSurfer subject naming convention this project
uses (`sub-XX_ses-1`), and why we use `vol_to_surf(pial, inner_mesh=white)`.
"""

from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import surface


HEMI_TO_FS = {"L": "lh", "R": "rh"}


def surface_path(vol_path: Path, hemi: str, space: str) -> Path:
    """Compute the output `.func.gii` path from a T1w-space NIfTI.

    Inserts `_hemi-{L,R}` before `_space-`, replaces `_space-T1w` with
    `_space-{fsnative,fsaverage}`, and changes extension to `.func.gii`.
    Output sits in the same directory as the input.
    """
    if hemi not in HEMI_TO_FS:
        raise ValueError(f"hemi must be L or R, got {hemi!r}")
    stem = vol_path.name
    if stem.endswith(".nii.gz"):
        stem = stem[: -len(".nii.gz")]
    elif stem.endswith(".nii"):
        stem = stem[: -len(".nii")]
    else:
        raise ValueError(f"Expected .nii or .nii.gz, got {vol_path.name}")
    if "_space-T1w" not in stem:
        raise ValueError(f"Expected _space-T1w in {vol_path.name}")
    new_stem = stem.replace("_space-T1w", f"_hemi-{hemi}_space-{space}")
    return vol_path.parent / f"{new_stem}.func.gii"


def find_surfaces(subject: str, session: int, bids_folder: Path,
                  fmriprep_deriv: str = "fmriprep") -> dict[str, Path]:
    """Return {`pial_L`, `pial_R`, `white_L`, `white_R`} for one subject/session.

    Surfaces live under `derivatives/<fmriprep_deriv>/sub-<subject>/ses-<N>/anat/`.
    Raises FileNotFoundError if any of the four are missing.
    """
    anat = (Path(bids_folder) / "derivatives" / fmriprep_deriv
            / f"sub-{subject}" / f"ses-{session}" / "anat")
    out: dict[str, Path] = {}
    for hemi in ("L", "R"):
        for surf in ("pial", "white"):
            p = anat / f"sub-{subject}_ses-{session}_hemi-{hemi}_{surf}.surf.gii"
            if not p.exists():
                raise FileNotFoundError(f"missing fmriprep surface: {p}")
            out[f"{surf}_{hemi}"] = p
    return out


def freesurfer_subject(subject: str, session: int) -> str:
    """The FreeSurfer subject directory name fmriprep uses for this project.

    fmriprep creates per-session FreeSurfer streams named `sub-XX_ses-N`
    rather than just `sub-XX`. SurfaceTransform needs this exact label
    to find the source subject's spherical registration.
    """
    return f"sub-{subject}_ses-{session}"


def freesurfer_subjects_dir(bids_folder: Path,
                            fmriprep_deriv: str = "fmriprep") -> Path:
    return (Path(bids_folder) / "derivatives" / fmriprep_deriv
            / "sourcedata" / "freesurfer")


def volume_to_fsnative(vol_path: Path, pial: Path, white: Path,
                       hemi: str) -> Path:
    """Sample one T1w-space NIfTI to fsnative for one hemisphere; save and return path.

    Uses `nilearn.surface.vol_to_surf(pial, inner_mesh=white)` — averaging
    samples along the pial→white normal samples grey matter and avoids
    leakage from CSF (pial-only sampling) and white matter.
    """
    data = surface.vol_to_surf(str(vol_path), str(pial), inner_mesh=str(white))
    data = data.astype(np.float32)
    out = surface_path(vol_path, hemi, "fsnative")
    nib.save(nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(data)]),
             str(out))
    return out


def fsnative_to_fsaverage(fsnative_path: Path, fs_hemi: str,
                          fs_subject: str, subjects_dir: Path) -> Path:
    """Transform a fsnative `.func.gii` to fsaverage via FreeSurfer SurfaceTransform.

    Output path is the input with `space-fsnative` → `space-fsaverage`.
    """
    from nipype.interfaces.freesurfer import SurfaceTransform
    out_path = Path(str(fsnative_path).replace("space-fsnative", "space-fsaverage"))
    os.environ["SUBJECTS_DIR"] = str(subjects_dir)
    sxfm = SurfaceTransform(subjects_dir=str(subjects_dir))
    sxfm.inputs.source_file = str(fsnative_path)
    sxfm.inputs.out_file = str(out_path)
    sxfm.inputs.source_subject = fs_subject
    sxfm.inputs.target_subject = "fsaverage"
    sxfm.inputs.hemi = fs_hemi
    sxfm.run()
    return out_path


def sample_to_surfaces(vol_path: Path, subject: str, session: int,
                       bids_folder: Path, *,
                       fmriprep_deriv: str = "fmriprep",
                       to_fsaverage: bool = True,
                       ) -> dict[tuple[str, str], Path]:
    """Sample a T1w-space NIfTI to fsnative (and optionally fsaverage), both hemis.

    Returns {(hemi, space): output_path} for hemi ∈ {L, R} and
    space ∈ {fsnative} ∪ ({fsaverage} if to_fsaverage).
    """
    surfaces = find_surfaces(subject, session, Path(bids_folder), fmriprep_deriv)
    subjects_dir = freesurfer_subjects_dir(bids_folder, fmriprep_deriv)
    fs_subject = freesurfer_subject(subject, session)

    out: dict[tuple[str, str], Path] = {}
    for hemi, fs_hemi in HEMI_TO_FS.items():
        fsnative = volume_to_fsnative(
            vol_path,
            surfaces[f"pial_{hemi}"],
            surfaces[f"white_{hemi}"],
            hemi,
        )
        out[(hemi, "fsnative")] = fsnative
        if to_fsaverage:
            out[(hemi, "fsaverage")] = fsnative_to_fsaverage(
                fsnative, fs_hemi, fs_subject, subjects_dir,
            )
    return out


def load_surface_data(path: Path) -> np.ndarray:
    """Load a `.func.gii` and return its first dataarray as float32."""
    img = nib.load(str(path))
    return img.darrays[0].data.astype(np.float32)
