#!/usr/bin/env python3
"""
Compute Fisher information for abstract value from the aPRF (LogGaussianPRF) model.

Loads already-fitted aPRF parameters, selects the top n_voxels within the
supplied mask by R², fits a noise model (ResidualFitter), then computes
Fisher information over a dense value grid.

For Gaussian noise (default method) the Fisher information is computed
analytically:
    FI(x) = grad(x)ᵀ Ω⁻¹ grad(x)
where grad(x)[v] = df_v/dx and Ω is the voxel noise covariance.

Output
------
  derivatives/encoding_models/aprf/sub-<subject>/<ses_dir>/func/
    sub-<subject>[_ses-<N>]_task-abstractvalue_mask-<mask_desc>_nvoxels-<n>_desc-fisherinfo_pe.tsv

  TSV: one row per value (CHF), one column: fisher_information

Usage
-----
  python compute_fisher_information_aprf.py pil01 --sessions 1 --roi BensonV1
  python compute_fisher_information_aprf.py pil01 --roi BensonV1          # all sessions
  python compute_fisher_information_aprf.py pil01 --roi NPC --hemi None
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.models import LogGaussianPRF
from braincoder.optimize import ResidualFitter

from abstract_values.utils.data import Subject, BIDS_FOLDER


def get_value_paradigm(sub, sessions):
    """Return DataFrame with column 'x' = objective CHF value (float32)."""
    rows = []
    for session in sessions:
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            run_ev = events.loc[run].reset_index().sort_values('onset')
            for _, row in run_ev[run_ev['event_type'] == 'gabor'].iterrows():
                rows.append(float(row['value']))
    return pd.DataFrame({'x': np.array(rows, dtype=np.float32)})


def main(subject, sessions=None, roi='BensonV1', hemi='LR', n_voxels=250,
         n_values=200, n_noise_iterations=1000, n_mc_samples=1000,
         bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep',
         smoothed=False):

    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)

    if sessions is None:
        sessions = sub.get_sessions()
    sessions = sorted(sessions)

    ses_dir    = f'ses-{sessions[0]}' if len(sessions) == 1 else ''
    ses_entity = f'_ses-{sessions[0]}' if len(sessions) == 1 else ''
    print(f'sub-{subject}  {ses_dir or "all-sessions"}  [aPRF Fisher information]')

    # ── paradigm + betas ──────────────────────────────────────────────────────
    paradigm = get_value_paradigm(sub, sessions)
    value_min = float(paradigm['x'].min())
    value_max = float(paradigm['x'].max())
    print(f'  {len(paradigm)} gabor trials  value range: {value_min:.1f}–{value_max:.1f} CHF')

    betas_img = sub.get_single_trial_estimates(sessions, desc='gabor',
                                               smoothed=smoothed)
    assert betas_img.shape[3] == len(paradigm), (
        f'Beta count mismatch: {betas_img.shape[3]} vs {len(paradigm)}')

    # ── mask + extract voxels ─────────────────────────────────────────────────
    hemi_arg = None if hemi == 'None' else hemi
    mask_img = sub.get_roi_mask(roi=roi, hemi=hemi_arg)
    mask_desc = f'{roi}{"_hemi-" + hemi if hemi_arg else ""}'
    masker = NiftiMasker(mask_img=mask_img,
                         target_affine=betas_img.affine,
                         target_shape=betas_img.shape[:3]).fit()
    data = pd.DataFrame(masker.transform(betas_img).astype(np.float32))
    print(f'  {data.shape[1]} voxels in mask ({mask_desc})')

    # ── load already-fitted aPRF parameters ───────────────────────────────────
    pars_imgs = sub.get_prf_parameters(sessions, smoothed=smoothed)
    pars_df = pd.DataFrame({
        'mu':        masker.transform(pars_imgs['mu']).squeeze().astype(np.float32),
        'sd':        masker.transform(pars_imgs['sd']).squeeze().astype(np.float32),
        'amplitude': masker.transform(pars_imgs['amplitude']).squeeze().astype(np.float32),
        'baseline':  masker.transform(pars_imgs['baseline']).squeeze().astype(np.float32),
    })
    r2 = pd.Series(masker.transform(pars_imgs['r2']).squeeze().astype(np.float32))

    # ── voxel selection by R² ─────────────────────────────────────────────────
    if n_voxels == 0:
        sel = r2[r2 > 0].index
    else:
        sel = r2.sort_values(ascending=False).index[:n_voxels]

    print(f'  {len(sel)} voxels selected  '
          f'(R² ≥ {float(r2.loc[sel].min()):.3f})')

    data_sel = data[sel]

    # ── noise model ───────────────────────────────────────────────────────────
    # apply_mask sets model.parameters to the selected voxels, then
    # init_pseudoWWT is called with those parameters so the WWT shape
    # (n_sel_voxels × n_sel_voxels) matches omega.
    model = LogGaussianPRF(allow_neg_amplitudes=True, parameterisation='mu_sd_natural')
    model.parameters = pars_df          # all ROI voxels
    model.apply_mask(sel)               # keeps only selected voxels
    model.init_pseudoWWT(paradigm['x'].values, model.parameters)
    print(f'  fitting noise model ({n_noise_iterations} iterations)...')
    residfit = ResidualFitter(model, data_sel, paradigm)
    omega, dof = residfit.fit(
        init_sigma2=1e-2, init_dof=10.0,
        learning_rate=0.05,
        max_n_iterations=n_noise_iterations)
    dof_str = f'{float(dof):.1f}' if dof is not None else 'None (Gaussian)'
    print(f'  noise model: dof={dof_str}')

    # ── Fisher information ────────────────────────────────────────────────────
    # For aPRF each voxel has its own parameters → gradient is in voxel space
    # (n_stimuli, n_voxels). omega is also voxel-space (n_voxels, n_voxels).
    # analytical=True works directly; no basis-space projection needed.
    stimuli = np.linspace(value_min, value_max, n_values, dtype=np.float32)

    if dof is None:
        print(f'  computing Fisher information over {n_values} values '
              f'(analytical, Gaussian)...')
        fisher_info = model.get_fisher_information(
            stimuli=stimuli,
            omega=omega,
            dof=None,
            weights=None,
            parameters=model.parameters,
            analytical=True,
        )
    else:
        print(f'  computing Fisher information over {n_values} values '
              f'(n={n_mc_samples} MC samples, Student-t)...')
        fisher_info = model.get_fisher_information(
            stimuli=stimuli,
            omega=omega,
            dof=dof,
            weights=None,
            parameters=model.parameters,
            analytical=False,
            n=n_mc_samples,
        )

    print(f'  mean FI={float(fisher_info.mean()):.4f}  '
          f'peak at {float(stimuli[fisher_info.values.argmax()]):.2f} CHF')

    # ── save ──────────────────────────────────────────────────────────────────
    smooth_label = '_smoothed' if smoothed else ''
    out_dir = bids_folder / 'derivatives' / 'encoding_models' / 'aprf' / f'sub-{subject}'
    if ses_dir:
        out_dir = out_dir / ses_dir
    out_dir = out_dir / 'func'
    out_dir.mkdir(parents=True, exist_ok=True)

    out_fn = (out_dir /
              f'sub-{subject}{ses_entity}_task-abstractvalue'
              f'_mask-{mask_desc}_nvoxels-{n_voxels}{smooth_label}_desc-fisherinfo_pe.tsv')

    pd.DataFrame({'fisher_information': fisher_info.values}, index=stimuli).to_csv(
        out_fn, sep='\t', header=True)
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
    parser.add_argument('--n-values', type=int, default=200,
                        help='Number of value points in Fisher information grid (default: 200)')
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
         n_voxels=args.n_voxels, n_values=args.n_values,
         n_noise_iterations=args.n_noise_iterations,
         n_mc_samples=args.n_mc_samples,
         bids_folder=args.bids_folder, fmriprep_deriv=args.fmriprep_deriv,
         smoothed=args.smoothed)
