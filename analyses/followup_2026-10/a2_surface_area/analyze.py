"""A2 analysis (local): areas vs noise, decoding precision vs area, area correlation matrix. Exploratory, n~35."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.covariance import MinCovDet
D = '/data/ds-abstractvalue/derivatives/followup_2026-10/a2_surface_area'
T = '/Users/gdehol/git/abstract_values/talks/2026-10_update_ruff_bedi/data'
rng = np.random.default_rng(0)

def bid(s): return int(s.replace('sub-', '').replace('pil', ''))
A = pd.read_csv(f'{D}/roi_surface_area.tsv', sep='\t'); A['subject'] = A['sub'].map(bid)
tot = A[A.roi == 'cortex_total'].groupby('subject').area_mid.sum().rename('total_area')
def get(roi, hemi=None, col='area_mid'):
    d = A[A.roi == roi] if hemi is None else A[(A.roi == roi) & (A.hemi == hemi)]
    return d.groupby('subject')[col].sum()
meas = {'V1': get('V1'), 'V1-3': get('V1-3'), 'IPS0-5': get('IPS0-5'), 'IPS0': get('IPS0'), 'IPS12': get('IPS12'),
        'IPS345': get('IPS345'), 'M1 hand (L)': get('M1hand', 'L'), 'BA4 (L)': get('BA4', 'L'), 'BA4 (R)': get('BA4', 'R'),
        'V1 (L)': get('V1', 'L'), 'V1 (R)': get('V1', 'R'), 'IPS0-5 (L)': get('IPS0-5', 'L'), 'IPS0-5 (R)': get('IPS0-5', 'R')}
vol = {'V1': get('V1', col='gm_volume'), 'IPS0-5': get('IPS0-5', col='gm_volume'), 'M1 hand (L)': get('M1hand', 'L', 'gm_volume')}
S = pd.DataFrame(meas).join(tot)
for k, v in vol.items(): S['vol ' + k] = v
def resid(y, x):
    b = np.polyfit(x, y, 1); return y - np.polyval(b, x)
for k in list(meas) + [c for c in S if c.startswith('vol ')]:
    S[k + ' | corr'] = resid(S[k], S.total_area)   # residual of regression on total cortical area
    S[k + ' | prop'] = S[k] / S.total_area
N = pd.read_csv(f'{T}/noise_motor_dissociation.tsv', sep='\t').set_index('subject')
K = pd.read_csv(f'{T}/k4_posterior_means.tsv', sep='\t').set_index('subject')
S = S.join(N.rename(columns={'perceptual_sd_chf': 'perc', 'value_sd_chf': 'value', 'motor_sd_chf': 'motor'}))
S = S.join(K[['sigma_perc_mean', 'sigma_rep_mean']].rename(columns={'sigma_perc_mean': 'perc_k4', 'sigma_rep_mean': 'value_k4'}))
S.to_csv(f'{D}/subject_table.tsv', sep='\t')

def skipped(x, y):
    r = np.c_[stats.rankdata(x), stats.rankdata(y)]
    try:
        m = MinCovDet(random_state=0).fit(r); d2 = m.mahalanobis(r)
        keep = d2 < stats.chi2.ppf(0.975, 2)
    except Exception: keep = np.ones(len(x), bool)
    return stats.spearmanr(x[keep], y[keep])[0], int((~keep).sum())
def corr(x, y, B=2000):
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = np.asarray(x)[ok], np.asarray(y)[ok]; n = len(x)
    rho, p = stats.spearmanr(x, y)
    bs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0: continue
        bs.append(stats.spearmanr(x[i], y[i])[0])
    sk, nout = skipped(x, y)
    return dict(n=n, rho=rho, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), p=p, skipped_rho=sk, n_outl=nout)

# ---- area x noise
rows = []
areas_main = ['V1', 'V1-3', 'IPS0-5', 'IPS0', 'IPS12', 'IPS345', 'M1 hand (L)', 'BA4 (L)', 'V1 (L)', 'V1 (R)', 'IPS0-5 (L)', 'IPS0-5 (R)']
noises = [('perc', 'motor-fit'), ('value', 'motor-fit'), ('motor', 'motor-fit'), ('perc_k4', 'K4-fit'), ('value_k4', 'K4-fit')]
for a in areas_main + ['vol V1', 'vol IPS0-5', 'vol M1 hand (L)']:
    for ver, suf in [('raw', ''), ('corrected', ' | corr'), ('proportion', ' | prop')]:
        for nz, fit in noises:
            rows.append(dict(area=a, version=ver, noise=nz, fit=fit, **corr(S[a + suf].values, S[nz].values)))
R = pd.DataFrame(rows); R.to_csv(f'{D}/area_noise_correlations.tsv', sep='\t', index=False)

# ---- area x area
base = ['V1', 'IPS0-5', 'M1 hand (L)']
rows = []; M = {}
for ver, suf in [('raw', ''), ('corrected', ' | corr')]:
    for i, a in enumerate(base):
        for b in base[i + 1:]:
            c = corr(S[a + suf].values, S[b + suf].values); rows.append(dict(a=a, b=b, version=ver, **c)); M[(ver, a, b)] = c['rho']
AA = pd.DataFrame(rows); AA.to_csv(f'{D}/area_area_correlations.tsv', sep='\t', index=False)
# total area dependence
tr = pd.DataFrame([dict(area=a, **corr(S[a].values, S.total_area.values)) for a in base]); tr.to_csv(f'{D}/area_vs_total.tsv', sep='\t', index=False)

# ---- decoding precision vs area
P = pd.read_csv(f'{D}/decoding_precision.tsv', sep='\t'); P['subject'] = P['sub'].map(bid)
dm = [('gabor', 'BensonV1', 100, 'V1'), ('gabor', 'BensonV1', 0, 'V1'), ('value', 'BensonV1', 100, 'V1'),
      ('gabor', 'IPS0', 100, 'IPS0'), ('value', 'IPS0', 100, 'IPS0'), ('value', 'IPS12', 100, 'IPS12'),
      ('value', 'IPS345', 100, 'IPS345'), ('value', 'M1l', 100, 'M1 hand (L)')]
rows = []
for kind, roi, nv, a in dm:
    p = P[(P.kind == kind) & (P.roi == roi) & (P.nvox_req == nv)].set_index('subject').mean_sd
    for ver, suf in [('raw', ''), ('corrected', ' | corr')]:
        d = pd.concat([p.rename('sd'), S[a + suf].rename('x')], axis=1).dropna()
        rows.append(dict(decoder=kind, roi=roi, nvox=nv, area=a, version=ver, **corr(d.x.values, d.sd.values)))
DP = pd.DataFrame(rows); DP.to_csv(f'{D}/decoding_precision_vs_area.tsv', sep='\t', index=False)

# ---- figure (16:9)
import matplotlib as mpl
mpl.rcParams.update({'font.family': 'Helvetica' if False else 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig = plt.figure(figsize=(13.33, 6.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.5, 1.1], wspace=0.55)
# A matrix
ax = fig.add_subplot(gs[0]); names = ['V1', 'IPS', 'M1']
mat = np.full((3, 3), np.nan)
for i, a in enumerate(base):
    for j, b in enumerate(base):
        if i == j: continue
        key = (('corrected', a, b) if i < j else ('corrected', b, a)); mat[i, j] = M[key]
        if i > j: pass
for i, a in enumerate(base):
    for j, b in enumerate(base):
        if i < j: mat[j, i] = M[('raw', a, b)]
ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
for i in range(3):
    for j in range(3):
        if i == j: ax.text(j, i, '1', ha='center', va='center', color='0.6')
        else: ax.text(j, i, f'{mat[i, j]:+.2f}', ha='center', va='center')
ax.set_xticks(range(3), names); ax.set_yticks(range(3), names)
ax.set_title('A  Area correlations\n(lower: raw, upper: size-corrected)', loc='left', fontsize=11)
# B forest
ax = fig.add_subplot(gs[1])
pairs = [('V1', 'perc', 'V1 – perceptual'), ('IPS0-5', 'value', 'IPS – value'), ('M1 hand (L)', 'motor', 'M1 – motor'),
         ('V1', 'value', 'V1 – value'), ('V1', 'motor', 'V1 – motor'), ('IPS0-5', 'perc', 'IPS – perceptual'),
         ('IPS0-5', 'motor', 'IPS – motor'), ('M1 hand (L)', 'perc', 'M1 – perceptual'), ('M1 hand (L)', 'value', 'M1 – value')]
for k, (a, nz, lab) in enumerate(pairs):
    for off, ver, col in [(-0.17, 'raw', '0.55'), (0.17, 'corrected', '#1B6FA8')]:
        r = R[(R.area == a) & (R.noise == nz) & (R.version == ver) & (R.fit == 'motor-fit')].iloc[0]
        ax.errorbar(r.rho, -k + off, xerr=[[r.rho - r.lo], [r.hi - r.rho]], fmt='o', color=col, ms=5, lw=1.4,
                    label=ver.capitalize() if k == 0 else None)
ax.axvline(0, color='k', lw=0.6); ax.axhline(-2.5, color='0.7', lw=0.6, ls=':')
ax.set_yticks(-np.arange(len(pairs)), [p[2] for p in pairs]); ax.set_xlabel('Spearman ρ (95% bootstrap CI)')
ax.set_title('B  Area vs noise (top 3: predicted; rest: specificity)', loc='left', fontsize=11); ax.legend(frameon=False, loc='lower right')
# C decoding
ax = fig.add_subplot(gs[2])
sel = [('gabor', 'BensonV1', 100, 'V1 orient.'), ('value', 'IPS12', 100, 'IPS1–2 value'), ('value', 'IPS345', 100, 'IPS3–5 value'), ('value', 'M1l', 100, 'M1 value')]
for k, (kind, roi, nv, lab) in enumerate(sel):
    r = DP[(DP.decoder == kind) & (DP.roi == roi) & (DP.nvox == nv) & (DP.version == 'corrected')].iloc[0]
    ax.errorbar(r.rho, -k, xerr=[[r.rho - r.lo], [r.hi - r.rho]], fmt='o', color='#B5651D', ms=5, lw=1.4)
ax.axvline(0, color='k', lw=0.6); ax.set_yticks(-np.arange(len(sel)), [s[3] for s in sel]); ax.set_xlim(-1, 1)
ax.set_xlabel('Spearman ρ (95% CI)'); ax.set_title('C  Decoding SD vs ROI area\n(size-corrected, 100 voxels)', loc='left', fontsize=11)
fig.suptitle('Exploratory (n ≈ 35): cortical surface area does not clearly predict stage-specific noise', x=0.01, ha='left', fontsize=13)
fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.14)
for e in ('svg', 'png'): fig.savefig(f'{D}/a2_summary.{e}', dpi=200)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
print(S[['total_area', 'V1', 'IPS0-5', 'M1 hand (L)']].describe().round(0))
print(R[(R.fit == 'motor-fit') & R.area.isin(['V1', 'IPS0-5', 'M1 hand (L)']) & R.version.isin(['raw', 'corrected'])].round(2).to_string())
print(R[(R.fit == 'K4-fit') & R.area.isin(['V1', 'IPS0-5']) & (R.version == 'corrected')].round(2).to_string())
print(R[(R.version == 'corrected') & (R.fit == 'motor-fit') & ~R.area.isin(['V1', 'IPS0-5', 'M1 hand (L)'])].round(2).to_string())
print(AA.round(2).to_string()); print(tr.round(2).to_string()); print(DP.round(2).to_string())
