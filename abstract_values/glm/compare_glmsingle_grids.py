"""Compare GLMsingle single-trial betas fitted with the old (linspace) and exact upsampling grids.

The old grid drifted against the design by 0 s at run start to ~0.46 s at run end, so any
difference should grow with a trial's time within its run. Reports, over voxels with R² above
a threshold (from the exact fit):
  - per-voxel correlation of betas across trials (old vs exact), median and 5th percentile
  - regression slope of old on exact betas (amplitude ratio)
  - the same per tertile of within-run trial time (early / middle / late)
  - R² of both fits

    python compare_glmsingle_grids.py 03 --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue
"""
import argparse
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('subject')
ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-abstractvalue')
ap.add_argument('--r2-min', type=float, default=10.)
a = ap.parse_args()

d = Path(a.bids_folder) / 'derivatives'
load = lambda g: np.load(d / f'glmsingle.{g}grid' / f'sub-{a.subject}' / 'func' /  # noqa: E731
                         'TYPED_FITHRF_GLMDENOISE_RR.npy', allow_pickle=True).item()
ex, old = load('exact'), load('linspace')
info = np.load(d / 'glmsingle.exactgrid' / f'sub-{a.subject}' / 'func' / 'DESIGNINFO.npy',
               allow_pickle=True).item()

# within-run onset time (s) of every trial, in GLMsingle's trial order (run by run, by onset)
tr_up = float(info['tr'])
n_trials = np.asarray(info['designSINGLE'][0]).shape[1]   # columns = all trials, all runs
times = np.full(n_trials, np.nan)
for dm in info['designSINGLE']:
    dm = np.asarray(dm)
    for j in np.flatnonzero(dm.any(0)):                    # this run's trials
        times[j] = np.flatnonzero(dm[:, j])[0] * tr_up
assert np.isfinite(times).all()

be = ex['betasmd'].reshape(-1, ex['betasmd'].shape[-1])
bo = old['betasmd'].reshape(-1, old['betasmd'].shape[-1])
r2 = ex['R2'].ravel()
keep = np.isfinite(r2) & (r2 > a.r2_min)
be, bo = be[keep].astype(np.float64), bo[keep].astype(np.float64)
print(f'sub-{a.subject}: {keep.sum()} voxels with R² > {a.r2_min} (exact fit), {be.shape[1]} trials, '
      f'trial times 0–{times.max():.0f} s')
print(f'R² median (those voxels): exact {np.nanmedian(r2[keep]):.2f}, '
      f'linspace {np.nanmedian(old["R2"].ravel()[keep]):.2f}')


def summary(e, o, label):
    ec, oc = e - e.mean(1, keepdims=True), o - o.mean(1, keepdims=True)
    r = (ec * oc).sum(1) / np.sqrt((ec ** 2).sum(1) * (oc ** 2).sum(1))
    slope = (ec * oc).sum(1) / (ec ** 2).sum(1)
    print(f'{label:>22s}: r median {np.nanmedian(r):.4f} (5th pct {np.nanpercentile(r, 5):.4f}); '
          f'slope old~exact median {np.nanmedian(slope):.3f}; '
          f'mean beta exact {e.mean():.3f}, linspace {o.mean():.3f}')


summary(be, bo, 'all trials')
edges = np.quantile(times, [0, 1 / 3, 2 / 3, 1])
for lo, hi, lab in zip(edges[:-1], edges[1:], ['early in run', 'middle', 'late in run']):
    m = (times >= lo) & (times <= hi)
    summary(be[:, m], bo[:, m], f'{lab} ({lo:.0f}-{hi:.0f} s)')
