#!/usr/bin/env python3
"""
Compute Fisher information for gabor orientation from the Von Mises basis set model.

Fits Von Mises weights on the given sessions, selects the top n_voxels within
the supplied mask by R², fits a Student-t noise model (ResidualFitter), then
computes Fisher information via Monte Carlo sampling over a dense orientation grid.

Output
------
  derivatives/encoding_models/vonmises/sub-<subject>/<ses_dir>/func/
    sub-<subject>[_ses-<N>]_task-abstractvalue_mask-<mask_desc>_nvoxels-<n>_desc-fisherinfo_pe.tsv

  TSV: one row per orientation (radians), one column: fisher_information

Usage
-----
  python compute_fisher_information.py pil01 --sessions 1 --roi BensonV1
  python compute_fisher_information.py pil01 --sessions 2 --roi BensonV1
  python compute_fisher_information.py pil01 --roi BensonV1            # all sessions
  python compute_fisher_information.py pil01 --roi NPC --hemi None
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import VonMisesPRF
from braincoder.optimize import WeightFitter, ResidualFitter
from braincoder.utils import get_rsq

from abstract_values.utils.data import Subject, BIDS_FOLDER


def get_gabor_paradigm(sub, sessions):
    """Return DataFrame with column 'x' (orientation in radians, float32)."""
    rows = []
    for session in sessions:
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            run_ev = events.loc[run].reset_index().sort_values('onset')
            for _, row in run_ev[run_ev['event_type'] == 'gabor'].iterrows():
                rows.append(np.deg2rad(float(row['orientation'])))
    return pd.DataFrame({'x': np.array(rows, dtype=np.float32)})


def make_basis_parameters(n_basis, kappa):
    mus = np.linspace(0, np.pi, n_basis, endpoint=False).astype(np.float32)
    return pd.DataFrame({
        'mu':        mus,
        'kappa':     np.full(n_basis, kappa, dtype=np.float32),
        'amplitude': np.ones(n_basis,  dtype=np.float32),
        'baseline':  np.zeros(n_basis, dtype=np.float32),
    })


def main(subject, sessions=None, roi='BensonV1', hemi='LR', n_voxels=250,
         n_basis=8, kappa=2.0, n_orientations=200, n_noise_iterations=1000,
         n_mc_samples=1000,
         bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep',
         smoothed=False):

    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)

    if sessions is None:
        sessions = sub.get_sessions()
    sessions = sorted(sessions)

    ses_dir    = f'ses-{sessions[0]}' if len(sessions) == 1 else ''
    ses_entity = f'_ses-{sessions[0]}' if len(sessions) == 1 else ''
    print(f'sub-{subject}  {ses_dir or "all-sessions"}  [Fisher information]')

    # ── paradigm + betas ──────────────────────────────────────────────────────
    paradigm = get_gabor_paradigm(sub, sessions)
    print(f'  {len(paradigm)} gabor trials')

    betas_img = sub.get_single_trial_estimates(sessions, desc='gabor',
                                               smoothed=smoothed)
    assert betas_img.shape[3] == len(paradigm), (
        f'Beta count mismatch: {betas_img.shape[3]} vs {len(paradigm)}')

    # ── mask ──────────────────────────────────────────────────────────────────
    hemi_arg = None if hemi == 'None' else hemi
    mask_img = sub.get_roi_mask(roi=roi, hemi=hemi_arg)
    mask_desc = f'{roi}{"_hemi-" + hemi if hemi_arg else ""}'
    masker = NiftiMasker(mask_img=mask_img,
                         target_affine=betas_img.affine,
                         target_shape=betas_img.shape[:3]).fit()
    data = pd.DataFrame(masker.transform(betas_img).astype(np.float32))
    print(f'  {data.shape[1]} voxels in mask ({mask_desc})')

    # ── basis parameters ──────────────────────────────────────────────────────
    basis_pars = make_basis_parameters(n_basis, kappa)
    print(f'  {n_basis} Von Mises basis functions  kappa={kappa}')

    # ── fit weights on all data ───────────────────────────────────────────────
    model = VonMisesPRF()
    weights = WeightFitter(model, basis_pars, data, paradigm).fit()

    # ── voxel selection by R² ─────────────────────────────────────────────────
    basis_pred = model.basis_predictions(paradigm, basis_pars)
    pred = pd.DataFrame(basis_pred @ weights.values,
                        index=data.index, columns=data.columns)
    r2 = get_rsq(data, pred)

    if n_voxels == 0:
        sel = r2[r2 > 0].index
    else:
        sel = r2.sort_values(ascending=False).index[:n_voxels]

    print(f'  {len(sel)} voxels selected  '
          f'(R² ≥ {float(r2.loc[sel].min()):.3f})')

    weights_sel = weights[sel]
    data_sel    = data[sel]

    # ── Student-t noise model ─────────────────────────────────────────────────
    print(f'  fitting noise model ({n_noise_iterations} iterations)...')
    residfit = ResidualFitter(model, data_sel, paradigm,
                              parameters=basis_pars, weights=weights_sel)
    omega, dof = residfit.fit(
        init_sigma2=1e-2, init_dof=10.0,
        learning_rate=0.05,
        max_n_iterations=n_noise_iterations)
    dof_str = f'{float(dof):.1f}' if dof is not None else 'None (Gaussian)'
    print(f'  noise model: dof={dof_str}')

    # ── Fisher information ────────────────────────────────────────────────────
    stimuli = np.linspace(0, np.pi, n_orientations, dtype=np.float32)

    if dof is None:
        # Gaussian noise → analytical Fisher information.
        # braincoder's analytical mode expects omega in basis-function space,
        # because _gradient gives d(basis_pred)/d(theta), not d(voxel_pred)/d(theta).
        # FI(θ) = grad_basis^T  [W Σ⁻¹ Wᵀ]  grad_basis
        #       = grad_basis^T  omega_eff⁻¹  grad_basis
        # where omega_eff = inv(W @ Sigma^{-1} @ W^T)
        W = weights_sel.values  # (n_basis, n_voxels)
        precision_eff = W @ np.linalg.solve(omega.values, W.T)  # (n_basis, n_basis)
        omega_eff = np.linalg.inv(precision_eff)
        print(f'  computing Fisher information over {n_orientations} orientations '
              f'(analytical, Gaussian)...')
        fisher_info = model.get_fisher_information(
            stimuli=stimuli,
            omega=omega_eff,
            dof=None,
            weights=None,
            parameters=basis_pars,
            analytical=True,
        )
    else:
        # Student-t noise → Monte Carlo.
        print(f'  computing Fisher information over {n_orientations} orientations '
              f'(n={n_mc_samples} MC samples, Student-t)...')
        fisher_info = model.get_fisher_information(
            stimuli=stimuli,
            omega=omega,
            dof=dof,
            weights=weights_sel,
            parameters=basis_pars,
            analytical=False,
            n=n_mc_samples,
        )
    print(f'  mean FI={float(fisher_info.mean()):.4f}  '
          f'peak at {float(np.rad2deg(stimuli[fisher_info.values.argmax()])):.1f}°')

    # ── save ──────────────────────────────────────────────────────────────────
    smooth_label = '_smoothed' if smoothed else ''
    out_dir = bids_folder / 'derivatives' / 'encoding_models' / 'vonmises' / f'sub-{subject}'
    if ses_dir:
        out_dir = out_dir / ses_dir
    out_dir = out_dir / 'func'
    out_dir.mkdir(parents=True, exist_ok=True)

    out_fn = (out_dir /
              f'sub-{subject}{ses_entity}_task-abstractvalue'
              f'_mask-{mask_desc}_nvoxels-{n_voxels}{smooth_label}_desc-fisherinfo_pe.tsv')

    fisher_info.to_csv(out_fn, sep='\t', header=True)
    print(f'  saved to {out_fn}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-'")
    parser.add_argument('--sessions', type=int, nargs='+', default=None)
    parser.add_argument('--roi', default='BensonV1',
                        help='ROI label (default: BensonV1)')
    parser.add_argument('--hemi', default='LR',
                        help="Hemisphere: LR, L, R, or None (default: LR)")
    parser.add_argument('--n-voxels', type=int, default=250,
                        help='Top-N voxels by R² within mask (0 = all R²>0, default: 250)')
    parser.add_argument('--n-basis', type=int, default=8)
    parser.add_argument('--kappa', type=float, default=2.0)
    parser.add_argument('--n-orientations', type=int, default=200,
                        help='Number of orientations in Fisher information grid (default: 200)')
    parser.add_argument('--n-noise-iterations', type=int, default=1000)
    parser.add_argument('--n-mc-samples', type=int, default=1000,
                        help='Monte Carlo samples for Fisher information (default: 1000)')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep',
                        choices=['fmriprep', 'fmriprep-t2w', 'fmriprep-flair'])
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()

    main(args.subject, sessions=args.sessions,
         roi=args.roi, hemi=args.hemi,
         n_voxels=args.n_voxels, n_basis=args.n_basis, kappa=args.kappa,
         n_orientations=args.n_orientations,
         n_noise_iterations=args.n_noise_iterations,
         n_mc_samples=args.n_mc_samples,
         bids_folder=args.bids_folder, fmriprep_deriv=args.fmriprep_deriv,
         smoothed=args.smoothed)
