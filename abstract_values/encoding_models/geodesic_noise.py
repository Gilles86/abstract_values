"""Geodesic-distance spatial noise model — shared helpers for decode_gabor.py
and decode_value.py.

Split deliberately into a generic part and a dataset-specific part:

  * ``snap_masker_to_surface`` -- pure geometry: given a fitted nilearn
    ``NiftiMasker`` and a set of surface vertices (already loaded, any
    source), find each voxel's nearest vertex. Knows nothing about BIDS,
    fmriprep, or this project. Pairs naturally with braincoder's existing
    ``braincoder.utils.cortex.geodesic_distance_matrix`` and is a candidate
    for a future ``braincoder.utils.cortex`` addition -- lift it verbatim
    if/when that happens.
  * ``load_subject_white_surface`` -- the abstract_values/numloss-specific
    part: knows how to find *this* dataset's fmriprep white-surface GIfTI
    files on disk. This part should NOT move to braincoder; a different
    project's surface layout would need a different loader.

``geodesic_snap_for_masker`` / ``geodesic_D_for_selection`` are the
call sites decode_gabor.py and decode_value.py actually use; they just
wire the two parts above together.
"""
from pathlib import Path

import numpy as np


def snap_masker_to_surface(masker, vertices):
    """Snap each voxel of a fitted ``NiftiMasker`` to its nearest surface vertex.

    Generic bridge between a nilearn volume-space masker and a surface
    mesh -- independent of dataset layout, so this is the part that's a
    natural fit for a future ``braincoder.utils.cortex`` helper (alongside
    the existing ``geodesic_distance_matrix``).

    Parameters
    ----------
    masker : nilearn.maskers.NiftiMasker
        Already ``.fit()``-ted; ``masker.mask_img_`` is the resampled mask
        that actually determines column order in ``masker.transform(...)``.
    vertices : array (n_v, 3)
        Surface vertex coordinates, in the same space/units as the
        masker's mask affine (e.g. T1w mm for a subject-space white
        surface).

    Returns
    -------
    nearest_vertex : int array (n_voxels,)
        Index into ``vertices`` per voxel, in masker column order (i.e.
        matching ``masker.transform(...)``'s columns).
    snap_distance : float array (n_voxels,)
        Euclidean distance (mm) from each voxel center to its nearest
        vertex -- a QC diagnostic; large values flag voxels far from any
        cortical surface (e.g. mask leakage off the cortical ribbon).
    """
    import nibabel as nib
    from scipy.spatial import cKDTree

    mask_img = masker.mask_img_
    ijk = np.argwhere(np.asarray(mask_img.get_fdata()) > 0)   # C-order == columns
    xyz = nib.affines.apply_affine(mask_img.affine, ijk).astype(np.float32)
    snap_distance, nearest_vertex = cKDTree(vertices).query(xyz)
    return np.asarray(nearest_vertex), np.asarray(snap_distance)


def load_subject_white_surface(hemi, subject, bids_folder, fmriprep_deriv):
    """Load a subject's fmriprep white-matter surface (T1w space, GIfTI).

    Dataset-specific (assumes the abstract_values/numloss BIDS+fmriprep
    layout): ``derivatives/<fmriprep_deriv>/sub-<S>/ses-*/anat/
    sub-<S>_ses-*_hemi-<H>_white.surf.gii``. A different project's layout
    would need its own loader -- this function is intentionally NOT a
    candidate for moving into braincoder.

    Returns
    -------
    vertices : float32 array (n_v, 3)
    faces    : int32 array (n_f, 3)
    """
    import nibabel as nib

    surfs = sorted((Path(bids_folder) / 'derivatives' / fmriprep_deriv
                    / f'sub-{subject}').glob(
                       f'ses-*/anat/sub-{subject}_ses-*_hemi-{hemi}_white.surf.gii'))
    if not surfs:
        raise FileNotFoundError(
            f'No hemi-{hemi} white surface for sub-{subject} under '
            f'{fmriprep_deriv} (needed for geodesic noise model)')
    gii = nib.load(str(surfs[0]))
    vertices = gii.darrays[0].data.astype(np.float32)
    faces = gii.darrays[1].data.astype(np.int32)
    print(f'    loaded hemi-{hemi} white surface: {surfs[0].name}')
    return vertices, faces


def geodesic_snap_for_masker(masker, hemi, subject, bids_folder, fmriprep_deriv):
    """Snap the masker's voxels to the nearest white-matter surface vertex
    (T1w space) and return ``(vertices, faces, nearest_vertex)``.

    ``nearest_vertex`` is in the SAME column order ``masker.transform``
    produces, so ``nearest_vertex[sel]`` gives the source vertices for any
    selected-voxel subset. The (cheap) snap is done once; the (expensive)
    geodesic Dijkstra is then run per fold over just the selected voxels via
    :func:`geodesic_D_for_selection` -- building the full ROI matrix is
    pointless when only ~100 voxels are decoded. Single-hemisphere ROIs only
    (e.g. NPCr -> hemi R); a bilateral mask would need a block-structured D.

    Thin dataset-specific wrapper around :func:`load_subject_white_surface`
    + :func:`snap_masker_to_surface` -- see those for the parts that
    generalize beyond this project.
    """
    vertices, faces = load_subject_white_surface(
        hemi, subject, bids_folder, fmriprep_deriv)
    nearest_vertex, snap_distance = snap_masker_to_surface(masker, vertices)
    print(f'    geodesic snap: {len(nearest_vertex)} voxels → hemi-{hemi} surface; '
          f'snap median {np.median(snap_distance):.2f} mm, '
          f'p95 {np.percentile(snap_distance, 95):.2f} mm')
    return vertices, faces, nearest_vertex


def geodesic_D_for_selection(geo_snap, sel_positions):
    """Geodesic distance matrix (mm) among the selected voxels only.
    ``sel_positions`` index into the masker column order. Cheap because
    Dijkstra runs from just the ~N selected source vertices."""
    from braincoder.utils.cortex import geodesic_distance_matrix
    vertices, faces, nearest = geo_snap
    return geodesic_distance_matrix(
        vertices, faces, source_indices=nearest[np.asarray(sel_positions)],
        progressbar=False).astype(np.float32)
