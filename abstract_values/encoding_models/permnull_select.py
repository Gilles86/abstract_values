#!/usr/bin/env python3
"""Label-permutation null for voxel selection (after retinonumeral).

Which voxels' fit beats what the same fitting procedure gets out of noise?
The encoding model is fitted to the real single-trial betas and to K copies
in which the stimulus rows are shuffled WITHIN each (session, run) — the run's
set of stimuli and the session's mapping stay intact, only the stimulus–beta
link is broken. Real full-data R² against the pooled null R² gives

    p(v) = (1 + #{null >= R²(v)}) / (1 + K·V_pool)

and Benjamini-Hochberg q within each ROI. The permutation machinery is
braincoder's ``braincoder.selection`` (pooled p, BH, exchangeability report,
leave-one-shuffle-out calibration); the fit is exactly
:func:`fit_pipeline.fit_one_model`, so grid search + gradient descent and any
warm-start chain are redone on every shuffle.

``stack=True``: shuffling the stimulus rows is the same as applying the
inverse permutation to the data rows, so the real data and all K shuffles are
fitted in ONE call as (K + 1)·V voxels against the unshuffled paradigm. Valid
because every aPRF prediction for a trial depends only on that trial's row
(value, session) — no temporal structure in single-trial betas.

Only the voxels of the ROIs below are fitted (their union, within the brain
mask); the null is pooled over that union. Permutation k is drawn from
``SeedSequence([crc32(subject), k])``, so it is the same relabelling for every
model and smoothing level.

Output (``derivatives/encoding_models/permnull/<spec.out_subdir>/sub-<S>/func/``):
  ``..._desc-permnull{_smoothed}_voxels.tsv`` — one row per union voxel: flat
      index in the BOLD grid, ROI membership, R², pooled p, BH q per ROI,
      fitted parameters (real data)
  ``..._desc-permnull{_smoothed}_null.npz``   — (K, V) null R² + permutations
  ``..._desc-permnull{_smoothed}_summary.json`` — per-ROI counts,
      exchangeability report and leave-one-shuffle-out calibration
  ``..._desc-{r2,pperm}{_smoothed}_pe.nii.gz`` — maps

Examples:
    python permnull_select.py 05 --model session-shift --n-permutations 20
    python permnull_select.py 05 --model session-shift --smoothed
    python permnull_select.py 05 --debug
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image as nimage
from nilearn.maskers import NiftiMasker

from braincoder.selection import (PermutationSelector, bh_qvalues,
                                  pooled_pvalues)

from abstract_values.encoding_models.fit_aprf_cv import get_paradigm
from abstract_values.encoding_models.fit_pipeline import fit_one_model
from abstract_values.encoding_models.model_specs import (get_spec,
                                                         list_model_types)
from abstract_values.utils.data import BIDS_FOLDER, Subject

# (mask desc, hemi entity)
ROIS = [
    ('BensonV1', 'LR'), ('BensonV2', 'LR'), ('BensonV3', 'LR'),
    ('hV4', None), ('V3ab', None),
    ('IPS0', None), ('IPS12', None), ('IPS345', None),
    ('NPCl', None), ('NPCr', None),
]


def subject_seed(subject):
    """Stable integer seed from a subject label (works for 'pil01' too)."""
    return zlib.crc32(str(subject).encode())


def roi_masks(sub, ref_img):
    """{roi: bool array on ref_img's grid}, ROIs without a mask skipped."""
    out = {}
    for roi, hemi in ROIS:
        try:
            img = sub.get_roi_mask(roi, hemi=hemi)
        except FileNotFoundError:
            print(f"  no {roi} mask — skipped")
            continue
        img = nimage.resample_to_img(img, ref_img, interpolation='nearest',
                                     force_resample=True, copy_header=True)
        out[roi] = np.asarray(img.get_fdata()).squeeze() > 0
    return out


def main(subject, model_type='session-shift', n_permutations=20, q=0.05,
         n_iterations=1000, bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep',
         smoothed=False, debug=False):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder,
                  fmriprep_deriv=fmriprep_deriv)
    sessions = sorted(sub.get_sessions())
    spec = get_spec(model_type)
    if spec.needs_session and len(sessions) < 2:
        raise ValueError(f"--model {spec.name} requires at least 2 sessions.")
    if spec.is_null:
        raise ValueError('the null model has nothing to select on')
    if debug:
        n_iterations, n_permutations = 50, min(n_permutations, 3)
    print(f"sub-{subject}  sessions {sessions}  model={spec.name}  "
          f"K={n_permutations}  smoothed={smoothed}")

    # ── paradigm ────────────────────────────────────────────────────────────
    paradigm = get_paradigm(sub, sessions, spec.needs_session,
                            space=spec.stimulus_space)
    keep = np.ones(len(paradigm), bool)
    if spec.drop_invalid_bids:
        keep = np.asarray(paradigm['bid'].notna() & paradigm['x'].notna())
        paradigm = paradigm[keep]
    cols = ['x', 'session'] if spec.needs_session else ['x']
    groups = np.array([f'{s}|{r}' for s, r in zip(
        paradigm.index.get_level_values('session'),
        paradigm.index.get_level_values('run'))])
    value_min = float(paradigm['x'].min())
    value_max = float(paradigm['x'].max())
    par = paradigm[cols].reset_index(drop=True)

    # ── betas in the ROI union ──────────────────────────────────────────────
    betas_img = sub.get_single_trial_estimates(sessions, desc='gabor',
                                               smoothed=smoothed)
    ref = nimage.index_img(betas_img, 0)
    brain = nimage.resample_to_img(sub.get_brain_mask(sessions[0]), ref,
                                   interpolation='nearest', force_resample=True,
                                   copy_header=True).get_fdata() > 0
    rois = roi_masks(sub, ref)
    union = np.zeros(ref.shape, bool)
    for m in rois.values():
        union |= m
    union &= brain
    mask_img = nib.Nifti1Image(union.astype(np.float32), ref.affine)
    masker = NiftiMasker(mask_img=mask_img).fit()
    data = masker.transform(betas_img).astype(np.float32)[keep]
    flat_idx = np.flatnonzero(union.ravel())   # C-order, same as masker
    data = pd.DataFrame(data)
    finite = np.isfinite(data.values).all(0) & (data.values.std(0) > 0)
    data = data.loc[:, finite]
    flat_idx = flat_idx[finite]
    V = data.shape[1]
    if debug and V > 300:
        cols_dbg = np.sort(np.random.default_rng(0).choice(V, 300, replace=False))
        data, flat_idx, V = data.iloc[:, cols_dbg], flat_idx[cols_dbg], 300
    data.columns = np.arange(V)
    print(f"  {len(par)} trials, {V} voxels in the ROI union; fitting "
          f"{(n_permutations + 1) * V} stacked columns")

    # ── real fit + K shuffles in one stacked fit ────────────────────────────
    fitted = {}

    def fit_fn(d, p):
        pars, r2 = fit_one_model(spec, d, p, value_min=value_min,
                                 value_max=value_max,
                                 n_iterations=n_iterations, debug=debug)
        fitted['pars'] = pars.iloc[:V].reset_index(drop=True)  # real data
        return np.asarray(r2, dtype=float)

    selector = PermutationSelector(fit_fn, n_permutations=n_permutations,
                                   groups=groups, q=q,
                                   seed=(subject_seed(subject),), stack=True)
    res = selector.fit(data, par)
    r2 = res.r2.values
    null = res.null_r2.values                       # (K, V)

    # ── pooled p over the union, BH within each ROI ─────────────────────────
    p = pooled_pvalues(r2, null)
    voxels = pd.DataFrame({'flat_index': flat_idx, 'r2': r2, 'p': p})
    ijk = np.column_stack(np.unravel_index(flat_idx, ref.shape))
    voxels[['i', 'j', 'k']] = ijk
    summary = dict(subject=subject, model=spec.name, smoothed=smoothed,
                   K=int(null.shape[0]), n_voxels=int(V), q=q,
                   seed=[subject_seed(subject)], n_iterations=n_iterations,
                   null_quantiles={str(k): float(v) for k, v in zip(
                       (.5, .95, .99, .999), np.quantile(null, (.5, .95, .99, .999)))},
                   rois={})
    for roi, m in rois.items():
        inroi = m.ravel()[flat_idx]
        voxels[f'in_{roi}'] = inroi
        qv = np.full(V, np.nan)
        if inroi.any():
            qv[inroi] = bh_qvalues(p[inroi])
        voxels[f'q_{roi}'] = qv
        sel = np.nan_to_num(qv, nan=1.0) <= q
        # leave-one-shuffle-out: shuffle k of this ROI against the other K-1
        n_disc = []
        for k in range(null.shape[0]):
            pk = pooled_pvalues(null[k, inroi], np.delete(null, k, axis=0))
            n_disc.append(int((bh_qvalues(pk) <= q).sum()))
        exch = {}
        if inroi.sum() >= 3:
            sub_res = type(res)(r2=res.r2[inroi], null_r2=res.null_r2.loc[:, inroi],
                                p=res.p[inroi], q=res.q[inroi],
                                mask=res.mask[inroi], q_thresh=q,
                                method='pooled', permutations=res.permutations)
            exch = sub_res.exchangeability_report(warn=False)
        summary['rois'][roi] = dict(
            n_voxels=int(inroi.sum()), n_selected=int(sel.sum()),
            frac_selected=float(sel.mean()) if inroi.any() else np.nan,
            min_selected_r2=float(r2[sel].min()) if sel.any() else None,
            calibration_any_discovery_rate=float(np.mean(np.array(n_disc) > 0)),
            calibration_n_discoveries=n_disc,
            exchangeability={k: (v if not isinstance(v, (np.floating, np.integer))
                                 else v.item()) for k, v in exch.items()})
        print(f"  {roi:9s} {int(inroi.sum()):6d} vox  selected {int(sel.sum()):5d}  "
              f"FPR(any) {np.mean(np.array(n_disc) > 0):.2f}  "
              f"split-half r {exch.get('split_half_r', np.nan):.3f}  "
              f"var ratio {exch.get('var_ratio', np.nan):.2f}")
    voxels = pd.concat([voxels, fitted['pars'].add_prefix('par_')], axis=1)

    if debug:
        print(voxels.head())
        return voxels, summary

    # ── save ────────────────────────────────────────────────────────────────
    sm = '_smoothed' if smoothed else ''
    out_dir = (bids_folder / 'derivatives' / 'encoding_models' / 'permnull'
               / spec.out_subdir / f'sub-{subject}' / 'func')
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'sub-{subject}_task-abstractvalue_space-T1w'
    voxels.to_csv(out_dir / f'{stem}_desc-permnull{sm}_voxels.tsv', sep='\t',
                  index=False)
    np.savez_compressed(out_dir / f'{stem}_desc-permnull{sm}_null.npz',
                        null_r2=null.astype(np.float32), flat_index=flat_idx,
                        permutations=res.permutations)
    (out_dir / f'{stem}_desc-permnull{sm}_summary.json').write_text(
        json.dumps(summary, indent=1, default=float))
    for desc, vals in (('r2', r2), ('pperm', p)):
        vol = np.full(ref.shape, np.nan, np.float32)
        vol.ravel()[flat_idx] = vals
        img = nib.Nifti1Image(vol, ref.affine)
        img.header.set_slope_inter(1, 0)
        img.to_filename(str(out_dir / f'{stem}_desc-{desc}{sm}_pe.nii.gz'))
    print(f"  saved to {out_dir}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject')
    ap.add_argument('--model', default='session-shift', choices=list_model_types())
    ap.add_argument('--n-permutations', type=int, default=20)
    ap.add_argument('--q', type=float, default=0.05)
    ap.add_argument('--n-iterations', type=int, default=1000)
    ap.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    ap.add_argument('--fmriprep-deriv', default='fmriprep',
                    choices=['fmriprep', 'fmriprep-t2w'])
    ap.add_argument('--smoothed', action='store_true')
    ap.add_argument('--debug', action='store_true',
                    help='300 voxels, K<=3, 50 iterations, nothing saved')
    a = ap.parse_args()
    main(a.subject, model_type=a.model, n_permutations=a.n_permutations, q=a.q,
         n_iterations=a.n_iterations, bids_folder=a.bids_folder,
         fmriprep_deriv=a.fmriprep_deriv, smoothed=a.smoothed, debug=a.debug)
