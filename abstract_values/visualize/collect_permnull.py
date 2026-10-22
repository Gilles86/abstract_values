"""Collect the permutation-null selection and every model's cvR² into one table.

Cluster-side reduce step for ``encoding_models/permnull_select.py``: for each
(subject, smoothing) it reads the permnull voxel table (ROI-union voxels: R²,
pooled p, BH q per ROI, fitted session-shift parameters) and samples, at the
same voxels, the leave-one-run-out cvR² of a ladder of aPRF variants (mean of
the per-fold maps, as in ``aggregate_cvr2``). The session-shift modes are
remapped from session order to mapping (``Subject.get_mapping``).

One row per (subject, smoothed, voxel); ROI membership and per-ROI q are
columns. A second table (``..._summary.tsv``) has the per-ROI calibration and
exchangeability numbers from the permnull summary JSONs. Feeds ``plot_permnull_selection.py`` and ``plot_quantile_models.py``.

Usage (cluster):
    python -m abstract_values.visualize.collect_permnull \\
        --out notes/data/permnull_voxels.tsv.gz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from abstract_values.encoding_models.aggregate_cvr2 import find_per_run_cvr2
from abstract_values.encoding_models.model_specs import get_spec
from abstract_values.utils.data import BIDS_FOLDER, Subject

# column name -> cv output dir
CV_MODELS = {
    'null': 'aprf-null.cv',
    'standard': 'aprf.cv',
    'gaussian': 'aprf-gauss.cv',
    'linear': 'aprf-linear.cv',
    'quantile': 'aprf-quantile.cv',
    'quantile_wrapped': 'aprf-quantile-wrapped.cv',
    'vonmises_prf': 'vonmises-prf.cv',
    'session_shift': 'aprf-shift.cv',
    'gauss_session_shift': 'aprf-gauss-shift.cv',
}
# fixed-value / quantile mixture axis, one shared lambda per spec (model_specs.MIX_LAMBDAS)
from abstract_values.encoding_models.model_specs import MIX_LAMBDAS, mix_tag  # noqa: E402
for _lam in MIX_LAMBDAS:
    CV_MODELS[f'mix_{mix_tag(_lam)}'] = f'aprf-mix-{mix_tag(_lam)}.cv'


def mean_cvr2(subject, model_dir, smoothed, bids_folder):
    files = find_per_run_cvr2(subject, model_dir, smoothed, bids_folder)
    if not files:
        return None, None
    imgs = [nib.load(str(f)) for f in files]
    data = np.mean([np.asarray(i.dataobj, dtype=np.float32) for i in imgs], 0)
    return data, imgs[0]


def collect_one(subject, smoothed, bids_folder, model='session-shift'):
    spec = get_spec(model)
    sm = '_smoothed' if smoothed else ''
    fn = (Path(bids_folder) / 'derivatives' / 'encoding_models' / 'permnull'
          / spec.out_subdir / f'sub-{subject}' / 'func'
          / f'sub-{subject}_task-abstractvalue_space-T1w_desc-permnull{sm}_voxels.tsv')
    if not fn.exists():
        return None
    vox = pd.read_csv(fn, sep='\t')
    ref = nib.load(str(fn.parent / f'sub-{subject}_task-abstractvalue_space-T1w'
                                    f'_desc-r2{sm}_pe.nii.gz'))
    for col, d in CV_MODELS.items():
        data, img = mean_cvr2(subject, d, smoothed, bids_folder)
        if data is None:
            vox[f'cvr2_{col}'] = np.nan
            continue
        if data.shape[:3] != ref.shape[:3] or not np.allclose(img.affine, ref.affine):
            raise ValueError(f'sub-{subject} {d}: grid differs from permnull grid')
        vox[f'cvr2_{col}'] = data.ravel()[vox['flat_index'].to_numpy()]
    sub = Subject(subject, bids_folder=Path(bids_folder))
    sessions = sorted(sub.get_sessions())
    maps = [sub.get_mapping(s) for s in sessions]
    if 'par_mode_1' in vox and set(maps) == {'cdf', 'inverse_cdf'}:
        first_cdf = maps[0] == 'cdf'
        vox['mode_cdf'] = vox['par_mode_1' if first_cdf else 'par_mode_2']
        vox['mode_invcdf'] = vox['par_mode_2' if first_cdf else 'par_mode_1']
    vox.insert(0, 'smoothed', smoothed)
    vox.insert(0, 'subject', subject)
    return vox


def collect_summary(subject, smoothed, bids_folder, model='session-shift'):
    """Per-ROI rows from the permnull summary JSON (counts, calibration,
    exchangeability)."""
    import json
    sm = '_smoothed' if smoothed else ''
    fn = (Path(bids_folder) / 'derivatives' / 'encoding_models' / 'permnull'
          / get_spec(model).out_subdir / f'sub-{subject}' / 'func'
          / f'sub-{subject}_task-abstractvalue_space-T1w_desc-permnull{sm}_summary.json')
    if not fn.exists():
        return []
    js = json.loads(fn.read_text())
    rows = []
    for roi, r in js['rois'].items():
        ex = r.get('exchangeability', {})
        rows.append(dict(subject=subject, smoothed=smoothed, roi=roi, K=js['K'],
                         n_voxels=r['n_voxels'], n_selected=r['n_selected'],
                         fpr_any=r['calibration_any_discovery_rate'],
                         mean_null_discoveries=np.mean(r['calibration_n_discoveries']),
                         split_half_r=ex.get('split_half_r'),
                         var_ratio=ex.get('var_ratio'),
                         excess_exceed_frac=ex.get('excess_exceed_frac'),
                         null_q95=js['null_quantiles'].get('0.95')))
    return rows


def swap_selection(vox, sel_fn):
    """Replace the per-ROI q columns of ``vox`` with those of another collected
    voxels table (a permnull run on a different model, same ROI-union voxels),
    so selection can come from one model while modes/cvR² come from another."""
    sel = pd.read_csv(sel_fn, sep='\t', dtype={'subject': str})
    qcols = [c for c in sel if c.startswith('q_')]
    vox = vox.drop(columns=[c for c in qcols if c in vox])
    return vox.merge(sel[['subject', 'smoothed', 'flat_index'] + qcols],
                     on=['subject', 'smoothed', 'flat_index'], how='left')


def subjects(bids_folder, model='session-shift'):
    d = (Path(bids_folder) / 'derivatives' / 'encoding_models' / 'permnull'
         / get_spec(model).out_subdir)
    return sorted(p.name.removeprefix('sub-') for p in d.glob('sub-*'))


def main(out, bids_folder=BIDS_FOLDER, model='session-shift'):
    rows, summ = [], []
    for s in subjects(bids_folder, model):
        for smoothed in (False, True):
            summ += collect_summary(s, smoothed, bids_folder, model)
            v = collect_one(s, smoothed, bids_folder, model)
            if v is None:
                print(f'  sub-{s} smoothed={smoothed}: no permnull output')
                continue
            print(f'  sub-{s} smoothed={smoothed}: {len(v)} voxels, '
                  + ', '.join(f'{c}={v[c].notna().mean():.2f}'
                              for c in v if c.startswith('cvr2_')))
            rows.append(v)
    df = pd.concat(rows, ignore_index=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep='\t', index=False, float_format='%.5g')
    print(f'Wrote {out}: {len(df)} rows, {df.subject.nunique()} subjects')
    out_s = str(out).replace('_voxels.tsv.gz', '_summary.tsv')
    pd.DataFrame(summ).to_csv(out_s, sep='\t', index=False)
    print(f'Wrote {out_s}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='notes/data/permnull_voxels.tsv.gz')
    ap.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    ap.add_argument('--model', default='session-shift')
    a = ap.parse_args()
    main(a.out, a.bids_folder, a.model)
