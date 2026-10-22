"""Preferred orientation per session (vonmises-prf-session-shift mu_1/mu_2, mapped to
cdf / inverse cdf) at the permutation-null voxel table rows, for figure_rf_shift.py.

Cluster: python abstract_values/visualize/collect_vm_shift.py notes/data/vm_shift_voxels.tsv.gz
"""
import sys, numpy as np, pandas as pd, nibabel as nib
from pathlib import Path
from abstract_values.utils.data import Subject
B=Path('/shares/zne.uzh/gdehol/ds-abstractvalue'); D=B/'derivatives/encoding_models'
rows=[]
for sub_dir in sorted((D/'permnull/aprf-session-shift').glob('sub-*')):
    s=sub_dir.name[4:]
    for sm in (False, True):
        tag='_smoothed' if sm else ''
        vt=sub_dir/'func'/f'sub-{s}_task-abstractvalue_space-T1w_desc-permnull{tag}_voxels.tsv'
        f=lambda p: D/'vonmises-prf-session-shift'/f'sub-{s}'/'func'/f'sub-{s}_task-abstractvalue_space-T1w_desc-{p}{tag}_pe.nii.gz'
        if not (vt.exists() and f('mu_1').exists()): continue
        vox=pd.read_csv(vt,sep='\t',usecols=['flat_index'])
        d={p: np.asarray(nib.load(str(f(p))).dataobj,dtype=np.float32).ravel()[vox.flat_index] for p in ('mu_1','mu_2','kappa','r2')}
        sub=Subject(s,bids_folder=B); ses=sorted(sub.get_sessions()); maps=[sub.get_mapping(x) for x in ses]
        first_cdf = maps[0]=='cdf'
        rows.append(pd.DataFrame(dict(subject=s,smoothed=sm,flat_index=vox.flat_index,
            mu_cdf=d['mu_1' if first_cdf else 'mu_2'], mu_invcdf=d['mu_2' if first_cdf else 'mu_1'], kappa=d['kappa'], r2_vm=d['r2'])))
pd.concat(rows).to_csv(sys.argv[1],sep='\t',index=False,float_format='%.5g'); print('ok', sum(len(r) for r in rows))
