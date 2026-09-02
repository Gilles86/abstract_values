#!/usr/bin/env python3
"""Fit an abstract pRF encoding model to single-trial GLMsingle betas.

Thin CLI wrapper around :func:`fit_pipeline.fit_one_model` — the
fitting logic itself lives in the model registry (see
``model_specs.py``) so adding a new variant is ~5 lines of
declarative config, not a 70-line copy-paste here.

The paradigm is the OBJECTIVE CHF value of each gabor trial. All
variants fit jointly across all of a subject's sessions (no per-
session output path).

Available variants (see ``--model`` choices):
  null                — baseline (no parameters, predicts training mean)
  standard            — joint LogGaussianPRF (mode, fwhm, amp, baseline)
  session-shift       — mode shifts per session                (5 params)
  fwhm-only-shift     — fwhm shifts per session, mode shared    (5 params)
  fwhm-shift          — mode + fwhm shift per session          (6 params)
  fully-shifted       — all 4 params shift per session         (8 params)
  gaussian            — joint symmetric Gaussian PRF
  gauss-session-shift — symmetric Gaussian with mode shift     (5 params)
  linear              — signed slope + baseline (no tuning bump); fit by
                         one closed-form OLS regression, not grid+descent

Output: ``derivatives/encoding_models/<spec.out_subdir>/sub-<S>/func/``
with one NIfTI per ``spec.save_params`` entry (plus ``desc-r2``).

Examples:
    python fit_aprf.py pil01
    python fit_aprf.py pil01 --model session-shift
    python fit_aprf.py pil01 --model fwhm-only-shift
    python fit_aprf.py pil01 --model fwhm-shift
    python fit_aprf.py pil01 --model fully-shifted --debug
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from abstract_values.encoding_models.fit_pipeline import fit_one_model
from abstract_values.encoding_models.model_specs import (
    get_spec, list_model_types,
)
from abstract_values.utils.data import Subject, BIDS_FOLDER


def save_f32(img, path):
    """Save a NIfTI image as float32, regardless of mask dtype."""
    nib.Nifti1Image(img.get_fdata().astype(np.float32),
                    img.affine).to_filename(str(path))


def get_value_paradigm(sub, sessions, needs_session: bool,
                       space: str = 'value'):
    """Return paradigm DataFrame with column 'x' and, if ``needs_session``,
    a 'session' column (0-based session index).

    ``space='value'`` puts the objective CHF value in 'x'; ``'orientation'``
    puts the gabor orientation in radians, wrapped to [0, pi) because
    orientation is axial. Same trials, same order — only the stimulus
    dimension the model is asked to explain differs.

    Row order matches gabor betas: session → run (sorted) → events
    sorted by onset, gabor only. For ``needs_session=True`` the
    function additionally validates that all sessions resolve to
    {0, 1} since the shift models' ``_session_predict`` assumes that.
    """
    rows = []
    for ses_idx, session in enumerate(sorted(sessions)):
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            run_ev = events.loc[run].reset_index().sort_values('onset')
            for _, row in run_ev[run_ev['event_type'] == 'gabor'].iterrows():
                stim = (np.deg2rad(float(row['orientation'])) % np.pi
                        if space == 'orientation' else float(row['value']))
                rows.append((stim, float(ses_idx)))
    arr = np.asarray(rows, dtype=np.float32)
    if not needs_session:
        return pd.DataFrame({'x': arr[:, 0]})
    unique = set(np.unique(arr[:, 1]).tolist())
    if not unique <= {0.0, 1.0}:
        raise ValueError(
            "Session column must contain only 0 and 1, got "
            f"{unique}. SessionShifted* models use session=0 for the "
            "first sorted session and session=1 for the second.")
    return pd.DataFrame({'x': arr[:, 0], 'session': arr[:, 1]})


def main(subject, mask=None, n_iterations=1000, model_type='standard',
         bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep',
         smoothed=False, debug=False, allow_incomplete=False):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)

    if not allow_incomplete:
        sub.require_complete_sessions()

    sessions = sorted(sub.get_sessions())
    spec = get_spec(model_type)

    if spec.needs_session and len(sessions) < 2:
        raise ValueError(f"--model {spec.name} requires at least 2 sessions")

    print(f"sub-{subject}  all-sessions ({sessions})  "
          f"[abstract pRF on objective value  model={spec.name}]")

    # ── paradigm ─────────────────────────────────────────────────────────────
    paradigm = get_value_paradigm(sub, sessions, spec.needs_session,
                                  space=spec.stimulus_space)
    value_min = float(paradigm['x'].min())
    value_max = float(paradigm['x'].max())
    print(f"  {len(paradigm)} trials  value range: "
          f"{value_min:.1f}–{value_max:.1f} CHF")

    # ── betas ─────────────────────────────────────────────────────────────────
    betas_img = sub.get_single_trial_estimates(sessions, desc='gabor',
                                                smoothed=smoothed)
    if betas_img.shape[3] != len(paradigm):
        raise ValueError(
            f"Beta count mismatch: {betas_img.shape[3]} vs {len(paradigm)}")

    # ── masker + data ────────────────────────────────────────────────────────
    if mask is None:
        mask = sub.get_brain_mask(sessions[0])
    masker = NiftiMasker(mask_img=mask).fit()
    data = pd.DataFrame(masker.transform(betas_img).astype(np.float32))
    print(f"  {data.shape[1]} voxels in mask")

    _skip_save = False
    if debug and data.shape[1] > 1000:
        rng = np.random.default_rng(0)
        cols = rng.choice(data.shape[1], 1000, replace=False)
        data = data.iloc[:, cols]
        _skip_save = True
        print(f"  [debug] subsampled to {data.shape[1]} voxels (saving skipped)")

    # ── fit (the whole if-elif ladder collapses to this one call) ───────────
    pars, r2 = fit_one_model(spec, data, paradigm,
                              value_min=value_min, value_max=value_max,
                              n_iterations=n_iterations, debug=debug)

    if _skip_save:
        return

    # ── save NIfTIs ──────────────────────────────────────────────────────────
    smooth_label = '_smoothed' if smoothed else ''
    out_dir = (bids_folder / 'derivatives' / 'encoding_models'
               / spec.out_subdir / f'sub-{subject}' / 'func')
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = (f"sub-{subject}_task-abstractvalue"
          f"_space-T1w_desc-{{desc}}{smooth_label}_pe.nii.gz")

    for param in spec.save_params:
        save_f32(masker.inverse_transform(pars[param]),
                  out_dir / fn.format(desc=param))
    save_f32(masker.inverse_transform(r2), out_dir / fn.format(desc='r2'))

    # For the standard (joint LogGaussian) model the legacy pipeline
    # also wrote a sigma-style 'sd' map alongside 'fwhm'. Keep that
    # for backward compat with consumers expecting it.
    if spec.name == 'standard' and 'fwhm' in pars.columns:
        sd = pars['fwhm'] * 0.4247   # 1 / (2 √(2 ln 2))
        save_f32(masker.inverse_transform(sd), out_dir / fn.format(desc='sd'))

    print(f"  saved to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-'")
    parser.add_argument('--model', default='standard',
                         choices=list_model_types(),
                         help='Model variant (see module docstring for '
                              'parameter counts and output dirs).')
    parser.add_argument('--mask', default=None,
                         help='Brain mask NIfTI (default: fmriprep brain mask)')
    parser.add_argument('--n-iterations', type=int, default=1000,
                         help='Max gradient descent iterations (default: 1000)')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep',
                         choices=['fmriprep', 'fmriprep-t2w'])
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--debug', action='store_true',
                         help='Fast smoke test: 50 iters, 1000 voxels, small grid')
    parser.add_argument('--allow-incomplete', action='store_true',
                         help='Skip the all-MRI-sessions-present check.')
    args = parser.parse_args()

    main(args.subject, mask=args.mask,
         n_iterations=args.n_iterations, model_type=args.model,
         bids_folder=args.bids_folder, fmriprep_deriv=args.fmriprep_deriv,
         smoothed=args.smoothed, debug=args.debug,
         allow_incomplete=args.allow_incomplete)
