"""A2: per-subject cortical surface area / GM volume of V1-V3, IPS0-5, M1 (FreeSurfer, native space).

Run on the cluster. ROI labels: neuropythy Benson14 (Varea 1-3) and Wang15 (Mplbl 18-23) atlases in
fsnative space (derivatives/neuropythy_atlas); M1 hand = fsaverage vertex set (Destrieux G_precentral within
20 mm of MNI (-38,-22,58), from talks/.../make_pathway_brain.py) mapped to native via nearest vertex on
the fsaverage-registered sphere; BA4a/BA4p = FreeSurfer exvivo thresh labels (native).
Area = sum of lh.area.mid (midthickness) in ROI; also white-surface area (lh.area), GM volume (lh.volume),
mean thickness.
"""
import sys, glob, os, numpy as np, pandas as pd, nibabel as nib
from scipy.spatial import cKDTree
D = '/shares/zne.uzh/gdehol/ds-abstractvalue/derivatives'
FS = f'{D}/fmriprep/sourcedata/freesurfer'
OUT = f'{D}/followup_2026-10/a2_surface_area'
os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
m1_fsavg = np.loadtxt(f'{HERE}/m1_hand_fsaverage_lh_vertices.txt', dtype=int)
fsavg_sphere = nib.freesurfer.read_geometry(f'{FS}/fsaverage/surf/lh.sphere.reg')[0]
m1_flag = np.zeros(len(fsavg_sphere), bool); m1_flag[m1_fsavg] = True
tree = cKDTree(fsavg_sphere)

def gifti(path):
    return nib.load(path).darrays[0].data

rows = []
for sdir in sorted(glob.glob(f'{FS}/sub-*_ses-1')):
    sub = os.path.basename(sdir).replace('_ses-1', '')
    for hemi, H in (('lh', 'L'), ('rh', 'R')):
        S = f'{sdir}/surf/{hemi}.'
        area = nib.freesurfer.read_morph_data(S + 'area.mid')
        warea = nib.freesurfer.read_morph_data(S + 'area')
        vol = nib.freesurfer.read_morph_data(S + 'volume')
        thk = nib.freesurfer.read_morph_data(S + 'thickness')
        cortex = np.zeros(len(area), bool)
        cortex[nib.freesurfer.read_label(f'{sdir}/label/{hemi}.cortex.label')] = True
        atl = f'{D}/neuropythy_atlas/{sub}/{sub}_desc-%s_space-fsnative_hemi-{H}.func.gii'
        varea = gifti(atl % 'benson14Varea').round().astype(int)
        wang = gifti(atl % 'wang15Mplbl').round().astype(int)
        assert len(varea) == len(area), (sub, len(varea), len(area))
        rois = {'V1': varea == 1, 'V2': varea == 2, 'V3': varea == 3, 'V1-3': np.isin(varea, [1, 2, 3])}
        for k in range(6): rois[f'IPS{k}'] = wang == 18 + k
        rois['IPS12'] = np.isin(wang, [19, 20]); rois['IPS345'] = np.isin(wang, [21, 22, 23])
        rois['IPS0-5'] = np.isin(wang, range(18, 24))
        for nm in ('BA4a', 'BA4p'):
            m = np.zeros(len(area), bool)
            m[nib.freesurfer.read_label(f'{sdir}/label/{hemi}.{nm}_exvivo.thresh.label')] = True
            rois[nm] = m
        rois['BA4'] = rois['BA4a'] | rois['BA4p']
        if hemi == 'lh':
            sph = nib.freesurfer.read_geometry(S + 'sphere.reg')[0]
            _, idx = tree.query(sph)
            rois['M1hand'] = m1_flag[idx]
        rois['cortex_total'] = cortex
        for nm, m in rois.items():
            m = m & cortex if nm != 'cortex_total' else m
            rows.append(dict(sub=sub, hemi=H, roi=nm, n_vertices=int(m.sum()), area_mid=area[m].sum(),
                             area_white=warea[m].sum(), gm_volume=vol[m].sum(), thickness=thk[m].mean() if m.any() else np.nan))
    print(sub, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/roi_surface_area.tsv', sep='\t', index=False)
print(df.groupby('roi').area_mid.describe())
