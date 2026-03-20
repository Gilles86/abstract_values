#!/usr/bin/env python3
"""
Pycortex flatmap visualization of aPRF and gabor-model parameters.

Loads the surface-sampled aPRF parameters (mu, sd, r2) and the gabor-model
R² from the fsnative GIfTI files produced by sample_aprf_to_surface.py and
displays them as an interactive pycortex webgl viewer.

The pycortex subject must already exist in the pycortex filestore
(typically as 'abstractvalue.sub-<subject>').

Usage
-----
  python visualize_subject_model.py pil01 --session 1
  python visualize_subject_model.py pil01 --session 1 --smoothed --cvr2-thr 0.05
  python visualize_subject_model.py pil01 --session 1 --params mu sd r2 gabor-r2
"""

import argparse
from pathlib import Path

import cortex
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from scipy.stats import norm

from abstract_values.utils.data import BIDS_FOLDER

# Default value range for mu visualization (CHF)
MU_VMIN = 2.0
MU_VMAX = 42.0


def load_surface_param(aprf_dir, subject, ses_label, hemi, desc, smoothed):
    """Load one aPRF surface parameter for one hemisphere.

    Returns a 1-D float32 array (n_vertices,).
    """
    smooth_tag = '_smoothed' if smoothed else ''
    fn = (aprf_dir / f'sub-{subject}_{ses_label}_task-abstractvalue'
                     f'_hemi-{hemi}_space-fsnative_desc-{desc}{smooth_tag}_pe.func.gii')
    if not fn.exists():
        raise FileNotFoundError(f'Surface file not found: {fn}')
    img = nib.load(str(fn))
    return img.darrays[0].data.astype(np.float32)


def r2_alpha(r2, thr, sigma):
    """Smooth alpha via Gaussian CDF centred on thr with width sigma."""
    return norm.cdf(r2, loc=thr, scale=sigma).astype(np.float32)


def get_masked_vertex(values, alpha, subject, vmin, vmax, cmap='RdBu_r'):
    """Create a pycortex Vertex blended onto curvature using a float alpha array."""
    v = cortex.Vertex(np.nan_to_num(values).astype(np.float32),
                      subject, vmin=vmin, vmax=vmax, cmap=cmap)
    return v.blend_curvature(alpha)


def load_bilateral(aprf_dir, subject, ses_label, desc, smoothed):
    """Load and concatenate L + R hemisphere data (pycortex convention)."""
    lh = load_surface_param(aprf_dir, subject, ses_label, 'L', desc, smoothed)
    rh = load_surface_param(aprf_dir, subject, ses_label, 'R', desc, smoothed)
    return np.concatenate([lh, rh])


def save_colorbar_pdf(colorbars, out_path):
    """Save a PDF with one horizontal colorbar per entry.

    Parameters
    ----------
    colorbars : list of (label, cmap, vmin, vmax)
    out_path : Path
    """
    n = len(colorbars)
    fig, axes = plt.subplots(n, 1, figsize=(5, 1.2 * n))
    if n == 1:
        axes = [axes]
    for ax, (label, cmap, vmin, vmax) in zip(axes, colorbars):
        cb = ColorbarBase(ax, cmap=cmap,
                          norm=Normalize(vmin=vmin, vmax=vmax),
                          orientation='horizontal')
        cb.set_label(label)
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)
    print(f'Colorbars saved to {out_path}')


def main(subject, session, sessions=None, bids_folder=BIDS_FOLDER,
         fmriprep_deriv='fmriprep-flair', smoothed=False,
         r2_thr=0.0, gabor_r2_thr=None, r2_sigma=0.01, gabor_r2_sigma=None,
         mu_vmin=MU_VMIN, mu_vmax=MU_VMAX, params=None, cx_subject=None,
         make_colorbars=False):
    bids_folder = Path(bids_folder)

    if sessions is None:
        sessions = [session]
    ses_label = f'ses-{sessions[0]}' if len(sessions) == 1 else 'ses-all'

    if cx_subject is None:
        cx_subject = f'abstractvalue.sub-{subject}'

    aprf_dir = (bids_folder / 'derivatives' / 'encoding_models' / 'aprf'
                / f'sub-{subject}' / ses_label / 'func')

    if params is None:
        params = ['mu', 'sd', 'r2', 'gabor-r2']

    print(f'sub-{subject}  {ses_label}  pycortex subject: {cx_subject}')

    if gabor_r2_thr is None:
        gabor_r2_thr = r2_thr
    if gabor_r2_sigma is None:
        gabor_r2_sigma = r2_sigma

    # Load R² for alpha masking (aPRF r2)
    r2_all = load_bilateral(aprf_dir, subject, ses_label, 'r2', smoothed)
    alpha_mask = r2_alpha(r2_all, r2_thr, r2_sigma)

    ds = {}
    cbars = []  # (label, cmap, vmin, vmax)

    if 'mu' in params:
        mu_all = load_bilateral(aprf_dir, subject, ses_label, 'mu', smoothed)
        in_range = ((mu_all >= mu_vmin) & (mu_all <= mu_vmax)).astype(np.float32)
        valid = alpha_mask * in_range
        ds[f'{subject}.mu'] = get_masked_vertex(
            mu_all, valid, cx_subject,
            vmin=mu_vmin, vmax=mu_vmax, cmap='nipy_spectral')
        cbars.append(('mu (CHF)', 'nipy_spectral', mu_vmin, mu_vmax))

    if 'sd' in params:
        sd_all = load_bilateral(aprf_dir, subject, ses_label, 'sd', smoothed)
        sd_vmax = (mu_vmax - mu_vmin) / 2
        ds[f'{subject}.sd'] = get_masked_vertex(
            sd_all, alpha_mask, cx_subject,
            vmin=0.0, vmax=sd_vmax, cmap='hot')
        cbars.append(('sd (CHF)', 'hot', 0.0, sd_vmax))

    if 'r2' in params:
        r2_vmax = float(np.nanpercentile(r2_all[r2_all > 0], 99.9)) if (r2_all > 0).any() else 0.3
        ds[f'{subject}.r2'] = get_masked_vertex(
            r2_all, alpha_mask, cx_subject,
            vmin=r2_thr, vmax=r2_vmax, cmap='hot')
        cbars.append(('aPRF R²', 'hot', r2_thr, r2_vmax))

    if 'gabor-r2' in params:
        gabor_r2 = load_bilateral(aprf_dir, subject, ses_label, 'gabor-r2', smoothed)
        gabor_alpha = r2_alpha(gabor_r2, gabor_r2_thr, gabor_r2_sigma)
        gabor_r2_vmax = float(np.nanpercentile(gabor_r2[gabor_r2 > 0], 99.9)) if (gabor_r2 > 0).any() else 0.3
        ds[f'{subject}.gabor_r2'] = get_masked_vertex(
            gabor_r2, gabor_alpha, cx_subject,
            vmin=gabor_r2_thr, vmax=gabor_r2_vmax, cmap='hot')
        cbars.append(('Gabor R²', 'hot', gabor_r2_thr, gabor_r2_vmax))

    if 'fwhm' in params:
        fwhm_all = load_bilateral(aprf_dir, subject, ses_label, 'fwhm', smoothed)
        fwhm_vmax = mu_vmax - mu_vmin
        ds[f'{subject}.fwhm'] = get_masked_vertex(
            fwhm_all, alpha_mask, cx_subject,
            vmin=0.0, vmax=fwhm_vmax, cmap='viridis')
        cbars.append(('fwhm (CHF)', 'viridis', 0.0, fwhm_vmax))

    if make_colorbars:
        smooth_tag = '_smoothed' if smoothed else ''
        pdf_path = (aprf_dir / f'sub-{subject}_{ses_label}_task-abstractvalue'
                               f'{smooth_tag}_colorbars.pdf')
        save_colorbar_pdf(cbars, pdf_path)

    print(f'Launching pycortex viewer with {len(ds)} dataset(s)...')
    cortex.webgl.show(ds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-', e.g. pil01")
    parser.add_argument('--session', type=int, required=True,
                        help='Session number (used to find surface files)')
    parser.add_argument('--sessions', type=int, nargs='+', default=None,
                        help='Sessions used for aPRF fitting (ses_label). Defaults to [--session].')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep-flair',
                        choices=['fmriprep', 'fmriprep-flair', 'fmriprep-noflair'])
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--r2-thr', type=float, default=0.0,
                        help='aPRF R² threshold for alpha masking (default: 0.0)')
    parser.add_argument('--gabor-r2-thr', type=float, default=None,
                        help='Gabor model R² threshold (default: same as --r2-thr)')
    parser.add_argument('--r2-sigma', type=float, default=0.01,
                        help='Gaussian CDF width for aPRF R² alpha transition (default: 0.01)')
    parser.add_argument('--gabor-r2-sigma', type=float, default=None,
                        help='Gaussian CDF width for gabor R² alpha transition (default: same as --r2-sigma)')
    parser.add_argument('--mu-vmin', type=float, default=MU_VMIN,
                        help=f'Lower bound for mu colorscale in CHF (default: {MU_VMIN})')
    parser.add_argument('--mu-vmax', type=float, default=MU_VMAX,
                        help=f'Upper bound for mu colorscale in CHF (default: {MU_VMAX})')
    parser.add_argument('--params', nargs='+',
                        default=['mu', 'sd', 'r2', 'fwhm', 'gabor-r2'],
                        choices=['mu', 'sd', 'r2', 'fwhm', 'gabor-r2'],
                        help='Parameters to visualize (default: mu sd r2 fwhm gabor-r2)')
    parser.add_argument('--cx-subject',
                        help="Pycortex subject name (default: abstractvalue.sub-<subject>)")
    parser.add_argument('--make-colorbars', action='store_true',
                        help='Save a PDF with colorbars alongside the surface files')
    args = parser.parse_args()

    sessions = args.sessions if args.sessions is not None else [args.session]
    main(args.subject, args.session, sessions=sessions,
         bids_folder=args.bids_folder, fmriprep_deriv=args.fmriprep_deriv,
         smoothed=args.smoothed, r2_thr=args.r2_thr, gabor_r2_thr=args.gabor_r2_thr,
         r2_sigma=args.r2_sigma, gabor_r2_sigma=args.gabor_r2_sigma,
         mu_vmin=args.mu_vmin, mu_vmax=args.mu_vmax,
         params=args.params, cx_subject=args.cx_subject,
         make_colorbars=args.make_colorbars)
