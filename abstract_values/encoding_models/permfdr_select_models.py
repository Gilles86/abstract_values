"""Model-neutral voxel selection with a gated permutation FDR (after retinonumeral).

Reads the ``permnull_select.py`` outputs of several aPRF models for one
participant (real training R² + K label shuffles; the shuffles are the same
relabellings for every model because the seed depends only on the subject —
asserted below) and selects, per ROI, voxels that carry signal under ANY of
the models:

  p_m(v) = (1 + #{null R²_m >= R²_m(v)}) / (1 + pool), pooled over the ROI-union
           voxels and shuffles (real copy never pooled; a null copy never in
           its own pool)
  T(v)   = max_m -log10 p_m(v)          (each model against its own null)
  SAM step-up FDR within the ROI (pi0 = 1), then the shuffle-level gate:
  keep the ROI's selection only if p_gate <= .05.

Also written, for the selection-bias check: for every null copy the top 5% of
union voxels by T; per voxel the number of copies in which it is in that top
5% (``n_top5_null``), and per model the share of null-top-5% voxels whose max
came from that model vs over all null voxels.

Selection is on full-data fits, so the held-out run helped choose the voxels
(double dipping; per-fold selection was not run here).

Usage (cluster, aggregation over all subjects):
    python -m abstract_values.encoding_models.permfdr_select_models \\
        --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue \\
        --out notes/data/permfdr_models_voxels.tsv.gz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from abstract_values.encoding_models.model_specs import get_spec
from abstract_values.encoding_models.permfdr import (
    ALPHA, fold_statistics, gate_p, sam_select)

MODELS = ['gaussian', 'quantile', 'quantile-wrapped', 'gauss-session-shift',
          'linear']
TOP = 0.05


def load(bids_folder, model, subject, smoothed):
    sm = '_smoothed' if smoothed else ''
    d = (Path(bids_folder) / 'derivatives' / 'encoding_models' / 'permnull'
         / get_spec(model).out_subdir / f'sub-{subject}' / 'func')
    stem = f'sub-{subject}_task-abstractvalue_space-T1w_desc-permnull{sm}'
    vox_fn, npz_fn = d / f'{stem}_voxels.tsv', d / f'{stem}_null.npz'
    if not (vox_fn.exists() and npz_fn.exists()):
        return None
    vox = pd.read_csv(vox_fn, sep='\t')
    z = np.load(npz_fn)
    assert np.array_equal(vox.flat_index.to_numpy(), z['flat_index'])
    return vox, z['null_r2'].astype(float), z['permutations']


def select_one(bids_folder, subject, smoothed, models=MODELS):
    got = [load(bids_folder, m, subject, smoothed) for m in models]
    if any(g is None for g in got):
        missing = [m for m, g in zip(models, got) if g is None]
        print(f'  sub-{subject} smoothed={smoothed}: missing {missing}')
        return None, []
    vox0 = got[0][0]
    for m, (v, _, perms) in zip(models[1:], got[1:]):
        assert np.array_equal(v.flat_index.to_numpy(), vox0.flat_index.to_numpy()), m
        assert np.array_equal(perms, got[0][2]), f'{m}: different shuffles'
    real = np.stack([g[0].r2.to_numpy(float) for g in got])          # (M, V)
    null = np.stack([g[1] for g in got])                              # (M, K, V)
    Tr, ar, Tn, an = fold_statistics(real, null)
    K, V = Tn.shape

    out = vox0[['flat_index']].copy()
    out.insert(0, 'smoothed', smoothed)
    out.insert(0, 'subject', subject)
    out['T'] = Tr
    out['best_model'] = np.asarray(models)[ar]
    # null top-5% per copy (bias check)
    thr = np.quantile(Tn, 1 - TOP, axis=1, keepdims=True)
    top = Tn >= thr
    out['n_top5_null'] = top.sum(0)
    rows = []
    for roi in [c.removeprefix('in_') for c in vox0 if c.startswith('in_')]:
        idx = vox0[f'in_{roi}'].astype(bool).to_numpy()
        if idx.sum() == 0:
            continue
        sel = sam_select(Tr[idx], Tn[:, idx], ALPHA)
        D_null = np.array([sam_select(Tn[k, idx], np.delete(Tn[:, idx], k, 0),
                                      ALPHA)['n_sel'] for k in range(K)])
        pg = gate_p(sel['n_sel'], D_null)
        pg_null = np.array([gate_p(D_null[k], np.delete(D_null, k))
                            for k in range(K)])
        keep = np.zeros(V, bool)
        keep[np.flatnonzero(idx)[sel['keep']]] = pg <= ALPHA
        out[f'sel_{roi}'] = keep
        r = dict(subject=subject, smoothed=smoothed, roi=roi, K=K,
                 n_voxels=int(idx.sum()), n_sel_ungated=sel['n_sel'],
                 t_star=sel['t_star'], p_gate=pg, n_selected=int(keep.sum()),
                 null_any_ungated=float(np.mean(D_null > 0)),
                 null_any_gated=float(np.mean((D_null > 0) & (pg_null <= ALPHA))))
        for i, m in enumerate(models):
            r[f'share_all_{m}'] = float(np.mean(an[:, idx] == i))
            r[f'share_top5_{m}'] = float(np.mean(an[:, idx][top[:, idx]] == i))
            r[f'share_sel_{m}'] = (float(np.mean(ar[keep] == i))
                                   if keep.any() else np.nan)
        rows.append(r)
    return out, rows


def subjects(bids_folder):
    d = (Path(bids_folder) / 'derivatives' / 'encoding_models' / 'permnull'
         / get_spec(MODELS[0]).out_subdir)
    return sorted(p.name.removeprefix('sub-') for p in d.glob('sub-*'))


def main(out, bids_folder, models=MODELS, only=None):
    vox, summ = [], []
    for s in (only or subjects(bids_folder)):
        for smoothed in (False, True):
            v, rows = select_one(bids_folder, s, smoothed, models)
            if v is None:
                continue
            vox.append(v)
            summ += rows
            sel = {r['roi']: r['n_selected'] for r in rows}
            print(f'  sub-{s} smoothed={smoothed}: ' + ', '.join(
                f'{k}={n}' for k, n in sel.items()), flush=True)
    vox = pd.concat(vox, ignore_index=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    vox.to_csv(out, sep='\t', index=False, float_format='%.5g')
    out_s = str(out).replace('_voxels.tsv.gz', '_summary.tsv')
    pd.DataFrame(summ).to_csv(out_s, sep='\t', index=False, float_format='%.5g')
    print(f'Wrote {out} ({len(vox)} rows) and {out_s}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='notes/data/permfdr_models_voxels.tsv.gz')
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-abstractvalue')
    ap.add_argument('--models', nargs='+', default=MODELS)
    ap.add_argument('--subjects', nargs='+', default=None)
    a = ap.parse_args()
    main(a.out, a.bids_folder, a.models, a.subjects)
