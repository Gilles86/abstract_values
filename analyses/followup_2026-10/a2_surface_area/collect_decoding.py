"""Mean per-trial posterior SD (orientation: circular SD in deg, period 180; value: SD in CHF) per subject x ROI.
Spherical noise, lambda 0.1 decoders (project default); n_voxels = 0 (all voxels in ROI) or 100."""
import glob, os, numpy as np, pandas as pd
D = '/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives/decoding'
OUT = '/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives/followup_2026-10/a2_surface_area'
specs = [('gabor', r, n) for r, n in [('BensonV1', 0), ('BensonV1', 100), ('BensonV2', 100), ('BensonV3', 100),
         ('IPS0', 100), ('IPS1', 100), ('IPS2', 100), ('IPS3', 100)]] + \
        [('value', r, n) for r, n in [('BensonV1', 0), ('BensonV1', 100), ('IPS0', 100), ('IPS12', 100),
         ('IPS345', 100), ('M1l', 100)]]
rows = []
for kind, roi, nv in specs:
    for f in sorted(glob.glob(f'{D}/{kind}/sub-*/func/sub-*_mask-{roi}_nvoxels-{nv}_noise-spherical_lambda-0.1_pars.tsv')):
        sub = os.path.basename(f).split('_')[0]
        d = pd.read_csv(f, sep='\t')
        grid = np.array([float(c) for c in d.columns[4:]])
        P = d.iloc[:, 4:].values.astype(float); P = P / P.sum(1, keepdims=True)
        if kind == 'gabor':
            R = np.abs((P * np.exp(2j * grid)).sum(1)).clip(1e-9, 1)
            sd = np.degrees(np.sqrt(-2 * np.log(R)) / 2)
        else:
            m = (P * grid).sum(1); sd = np.sqrt((P * (grid - m[:, None]) ** 2).sum(1))
        meta = pd.read_csv(f.replace('_pars', '_meta'), sep='\t')
        rows.append(dict(sub=sub, kind=kind, roi=roi, nvox_req=nv, n_trials=len(sd), mean_sd=sd.mean(), median_sd=np.median(sd),
                         n_vox=meta.n_voxels_selected.mean()))
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/decoding_precision.tsv', sep='\t', index=False)
print(df.groupby(['kind', 'roi', 'nvox_req']).agg(n=('sub', 'size'), sd=('mean_sd', 'mean'), nvox=('n_vox', 'mean')))
