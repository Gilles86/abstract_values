#!/usr/bin/env python3
"""
Benchmark braincoder ParameterFitter + ResidualFitter across keras backends.

Runs grid search → gradient descent → noise model on real fMRI betas using
TensorFlow, JAX, and PyTorch backends.  Each backend runs in a fresh subprocess
(keras backend cannot be changed after import).

Usage
-----
    python benchmark_backends.py pil01 --sessions 1 \
        --mask /data/.../BensonV1_mask.nii.gz --mask-desc BensonV1

    # specific backends only
    python benchmark_backends.py pil01 --sessions 1 \
        --mask /data/.../BensonV1_mask.nii.gz --mask-desc BensonV1 \
        --backends tensorflow jax
"""

import argparse
import subprocess
import sys
import time
import json
from pathlib import Path

BACKENDS = ['tensorflow', 'jax', 'torch']


# ── inner worker (called per-backend in subprocess) ───────────────────────────

def _run_worker(subject, sessions, mask, mask_desc,
                n_grid_mus, n_grid_sds, n_iterations, bids_folder,
                fmriprep_deriv, smoothed):
    """Fit ParameterFitter + ResidualFitter and print timing as JSON."""
    import os, time
    import numpy as np
    import pandas as pd
    from nilearn.maskers import NiftiMasker
    from braincoder.models import LogGaussianPRF
    from braincoder.optimize import ParameterFitter
    from braincoder.utils import get_rsq
    from abstract_values.utils.data import Subject, BIDS_FOLDER

    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)
    sessions = sessions if sessions else sub.get_sessions()

    # paradigm
    rows = []
    for session in sessions:
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            run_ev = events.loc[run].reset_index().sort_values('onset')
            for _, row in run_ev[run_ev['event_type'] == 'gabor'].iterrows():
                rows.append({'x': float(row['value'])})
    paradigm = pd.DataFrame(rows, dtype='float32')
    value_min, value_max = float(paradigm['x'].min()), float(paradigm['x'].max())

    betas_img = sub.get_single_trial_estimates(sessions, desc='gabor', smoothed=smoothed)
    if mask is None:
        mask = sub.get_brain_mask(sessions[0])
    masker = NiftiMasker(mask_img=mask).fit()
    data = pd.DataFrame(masker.transform(betas_img).astype('float32'))

    model = LogGaussianPRF(allow_neg_amplitudes=True, parameterisation='mu_sd_natural')
    fitter = ParameterFitter(model, data, paradigm)

    mus = np.linspace(value_min, value_max, n_grid_mus, dtype='float32')
    sds = np.linspace(1.0, (value_max - value_min) / 2, n_grid_sds, dtype='float32')
    amplitudes = np.array([1.0], dtype='float32')
    baselines  = np.array([0.0], dtype='float32')

    t0 = time.perf_counter()
    grid_pars = fitter.fit_grid(mus, sds, amplitudes, baselines,
                                use_correlation_cost=True)
    grid_pars = fitter.refine_baseline_and_amplitude(grid_pars)
    t_grid = time.perf_counter() - t0

    t0 = time.perf_counter()
    pars = fitter.fit(max_n_iterations=n_iterations, init_pars=grid_pars,
                      progressbar=False)
    t_gd = time.perf_counter() - t0

    pred = model.predict(parameters=pars, paradigm=paradigm)
    r2 = get_rsq(data, pred)

    import keras
    result = dict(
        backend=os.environ.get('KERAS_BACKEND', 'tensorflow'),
        keras_version=keras.__version__,
        n_voxels=data.shape[1],
        n_trials=len(paradigm),
        mean_r2=float(r2.mean()),
        t_grid_s=round(t_grid, 2),
        t_gd_s=round(t_gd, 2),
        t_total_s=round(t_grid + t_gd, 2),
    )
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject')
    parser.add_argument('--sessions', type=int, nargs='+', default=None)
    parser.add_argument('--mask', default=None,
                        help='Brain mask NIfTI (default: fmriprep brain mask)')
    parser.add_argument('--mask-desc', default='brainmask')
    parser.add_argument('--n-grid-mus', type=int, default=20)
    parser.add_argument('--n-grid-sds', type=int, default=15)
    parser.add_argument('--n-iterations', type=int, default=500)
    parser.add_argument('--bids-folder', default='/data/ds-abstractvalue')
    parser.add_argument('--fmriprep-deriv', default='fmriprep-flair')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--backends', nargs='+', default=BACKENDS,
                        choices=BACKENDS)
    parser.add_argument('--out-json', default=None,
                        help='Save results JSON to this path (optional)')
    parser.add_argument('--worker', action='store_true',
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    # ── if called as worker (backend already set in env), run directly ────────
    if args.worker:
        _run_worker(
            subject=args.subject,
            sessions=args.sessions,
            mask=args.mask,
            mask_desc=args.mask_desc,
            n_grid_mus=args.n_grid_mus,
            n_grid_sds=args.n_grid_sds,
            n_iterations=args.n_iterations,
            bids_folder=args.bids_folder,
            fmriprep_deriv=args.fmriprep_deriv,
            smoothed=args.smoothed,
        )
        sys.exit(0)

    # ── orchestrator: spawn one subprocess per backend ────────────────────────
    import os
    worker_args = [
        sys.executable, __file__,
        args.subject, '--worker',
        '--mask-desc', args.mask_desc,
        '--n-grid-mus', str(args.n_grid_mus),
        '--n-grid-sds', str(args.n_grid_sds),
        '--n-iterations', str(args.n_iterations),
        '--bids-folder', args.bids_folder,
        '--fmriprep-deriv', args.fmriprep_deriv,
    ]
    if args.mask:
        worker_args += ['--mask', args.mask]
    if args.sessions:
        worker_args += ['--sessions'] + [str(s) for s in args.sessions]
    if args.smoothed:
        worker_args.append('--smoothed')

    results = []
    for backend in args.backends:
        print(f'\n[{backend}] starting...', flush=True)
        env = {**os.environ, 'KERAS_BACKEND': backend}
        t0 = time.perf_counter()
        proc = subprocess.run(worker_args, env=env, capture_output=True, text=True)
        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            print(f'  FAILED (exit {proc.returncode})')
            print(proc.stderr[-1000:])
            continue

        # last line of stdout is the JSON result
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith('{'):
                result = json.loads(line)
                results.append(result)
                print(f'  grid={result["t_grid_s"]:.1f}s  '
                      f'gd={result["t_gd_s"]:.1f}s  '
                      f'total={result["t_total_s"]:.1f}s  '
                      f'mean_R²={result["mean_r2"]:.4f}')
                break

    if not results:
        sys.exit(1)

    # ── summary table ─────────────────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame(results).set_index('backend')[
        ['t_grid_s', 't_gd_s', 't_total_s', 'mean_r2']]
    fastest = df['t_total_s'].min()
    df['speedup_vs_tf'] = (df.loc['tensorflow', 't_total_s'] / df['t_total_s']).round(2) \
        if 'tensorflow' in df.index else None

    print(f'\n{"─"*60}')
    print(f'  Backend benchmark  ({args.n_voxels} voxels, {args.n_iterations} GD iters)')
    print(f'{"─"*60}')
    print(df.to_string())
    print(f'{"─"*60}')

    if args.out_json:
        import datetime
        out = {
            'timestamp': datetime.datetime.now().isoformat(),
            'n_voxels': args.n_voxels,
            'n_iterations': args.n_iterations,
            'n_grid_mus': args.n_grid_mus,
            'n_grid_sds': args.n_grid_sds,
            'subject': args.subject,
            'results': results,
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f'\n  Results saved to {args.out_json}')
