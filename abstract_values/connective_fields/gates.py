#!/usr/bin/env python3
"""Step-0 gates for the orientation-space connective-field analysis.

The planned analysis asks whether trial-to-trial coupling between an NPC voxel
and V1 orientation channels follows that voxel's preferred VALUE, read out
through the orientation->value mapping of the current condition. Before
building it, four things have to hold. This script measures all four for one
subject; ``plot_gates.py`` summarises the cohort.

Gate 1  V1 preferred orientation is reliable -- within a session (odd vs even
        runs), across sessions (the two mappings), and pooled odd vs even.
        Across-session is the load-bearing one: the V1 channels for one
        session's CF are labelled with the OTHER session's tuning.
Gate 2  NPC preferred value is reliable, same splits. The joint value label
        is what the predicted CF is built from.
Gate 3  The residual NPC x V1-channel CF is itself reliable between odd and
        even runs of one session. If a CF cannot reproduce within a
        condition, a between-condition difference in it is uninterpretable.
Gate 4  Power and specificity by injection. NPC residuals are shuffled across
        trials (destroying any real coupling), then a known CF is injected:
          'specific'  f_i(m_c(theta)), the current condition's mapping
          'invariant' f_i(mean mapping), identical in both conditions
        The primary score must detect 'specific' and stay at zero for
        'invariant' -- the within-voxel cancellation the design relies on.
        The unshuffled real-data score is deliberately NOT computed here.

Tuning is a grid search with correlation cost (log-Gaussian over value for
NPC, axial von Mises over orientation for V1): the same model families as
``aprf`` / ``vonmises-prf``, without the gradient-descent refinement, so every
split is cheap and fitted identically.

Residuals remove, per block of trials, an additive orientation x run model
(one mean per orientation, one per run). Every orientation occurs once per run,
so this removes all stimulus-locked variance and nothing can be left of it for
a CF to pick up.

Outputs, under derivatives/connective_fields/gates/sub-<S>/:
  sub-<S>_desc-tuningreliability.tsv   gates 1-2 summary
  sub-<S>_desc-tuningpairs.tsv.gz      gates 1-2 per-voxel estimate pairs
  sub-<S>_desc-cfreliability.tsv       gate 3 summary
  sub-<S>_desc-cfprofile.tsv           gate 3 ROI-mean CF profiles
  sub-<S>_desc-simulation.tsv          gate 4

Usage:
  python -m abstract_values.connective_fields.gates 03
  python -m abstract_values.connective_fields.gates pil01 --n-perm 10
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from nilearn import image
from nilearn.maskers import NiftiMasker

from abstract_values.utils.data import Subject, BIDS_FOLDER

REPO = Path(__file__).resolve().parents[2]
SETTINGS = REPO / 'experiment' / 'settings' / 'sns_fmri.yml'


# ── stimulus / mappings ──────────────────────────────────────────────────────

def load_mappings():
    """orientation (deg, 0..180) and value under each mapping, from the task settings."""
    with open(SETTINGS) as f:
        m = yaml.safe_load(f)['mappings']
    ori = np.asarray(m['orientations'], dtype=float)
    return ori, {c: np.asarray(m[c], dtype=float) for c in ('cdf', 'inverse_cdf')}


def get_paradigm(sub, sessions):
    """One row per gabor trial, in GLMsingle beta order (session -> run -> onset)."""
    rows = []
    for session in sessions:
        cond = sub.get_mapping(session)
        runs = sub.get_runs(session)
        events = sub.get_events(session, runs)
        for run in runs:
            ev = events.loc[run].reset_index().sort_values('onset')
            for _, r in ev[ev['event_type'] == 'gabor'].iterrows():
                rows.append({'session': session, 'run': run, 'condition': cond,
                             'orientation': float(r['orientation']),
                             'value': float(r['value'])})
    return pd.DataFrame(rows)


# ── small numerics ───────────────────────────────────────────────────────────

def zscore(x, axis=0):
    x = x - x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return np.divide(x, sd, out=np.zeros_like(x), where=sd > 0)


def colcorr(a, b):
    """Pearson r between matching columns of a and b (rows = observations)."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))


def circcorr_axial(a_deg, b_deg):
    """Jammalamadaka circular correlation for axial (180-deg periodic) angles."""
    a = np.deg2rad(2 * np.asarray(a_deg))
    b = np.deg2rad(2 * np.asarray(b_deg))
    ma = np.angle(np.exp(1j * a).mean())
    mb = np.angle(np.exp(1j * b).mean())
    sa, sb = np.sin(a - ma), np.sin(b - mb)
    return float((sa * sb).sum() / np.sqrt((sa ** 2).sum() * (sb ** 2).sum()))


def lognormal_mode_fwhm(x, mode, fwhm):
    """Shape of braincoder's lognormal_pdf_mode_fwhm (peak 1 at x = mode)."""
    sigma2 = (np.arcsinh(fwhm / (2 * mode)) / np.sqrt(2 * np.log(2))) ** 2
    return (mode / x) * np.exp(.5 * sigma2 - .5 * (np.log(x / mode) - sigma2) ** 2 / sigma2)


VALUE_MODES = np.linspace(2, 42, 41)
VALUE_FWHMS = np.linspace(2, 40, 15)
ORI_MUS = np.arange(0, 180, 5.)
ORI_KAPPAS = np.geomspace(.25, 16, 10)


def grid_fit(y, x, kind):
    """Best (location, width) per voxel by correlation over trials.

    y: trials x voxels (already demeaned within run); x: stimulus per trial.
    Returns location (CHF or deg), width, and the winning correlation.
    """
    if kind == 'value':
        loc, wid = np.meshgrid(VALUE_MODES, VALUE_FWHMS, indexing='ij')
        g = lognormal_mode_fwhm(x[None, :], loc.ravel()[:, None], wid.ravel()[:, None])
    else:
        loc, wid = np.meshgrid(ORI_MUS, ORI_KAPPAS, indexing='ij')
        d = np.deg2rad(x[None, :] - loc.ravel()[:, None])
        g = np.exp(wid.ravel()[:, None] * np.cos(2 * d))
    r = zscore(g, axis=1) @ zscore(y, axis=0) / len(x)      # grid x voxels
    best = np.nanargmax(r, axis=0)
    return loc.ravel()[best], wid.ravel()[best], r[best, np.arange(r.shape[1])]


def demean_runs(y, par):
    out = y.copy()
    for _, idx in par.groupby(['session', 'run']).indices.items():
        out[idx] -= out[idx].mean(0)
    return out


def residualise(y, par):
    """Remove an additive orientation + run model fitted on these trials only."""
    x = pd.get_dummies(par['orientation'].astype(str) + '_o').join(
        pd.get_dummies(par['session'].astype(str) + '_' + par['run'].astype(str) + '_r'))
    x = x.to_numpy(float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


# ── data ─────────────────────────────────────────────────────────────────────

def load_roi(sub, betas, roi, model_cv, bids_folder, smoothed):
    """ROI betas (trials x voxels) and a 'model beats null cvR2' selection."""
    mask = sub.get_roi_mask(roi, hemi=None)
    masker = NiftiMasker(mask_img=mask).fit()
    y = masker.transform(betas).astype(np.float64)
    ok = np.isfinite(y).all(0) & (y.std(0) > 0)

    tag = '_smoothed' if smoothed else ''
    def cvr2(model):
        fn = (Path(bids_folder) / 'derivatives' / 'encoding_models' / model
              / f'sub-{sub.subject_id}' / 'func'
              / f'sub-{sub.subject_id}_task-abstractvalue_space-T1w_desc-cvr2{tag}_pe.nii.gz')
        if not fn.exists():
            print(f'  [warn] missing {fn}; selection falls back to all voxels')
            return None
        return masker.transform(image.load_img(str(fn))).ravel()
    m, n = cvr2(model_cv), cvr2('aprf-null.cv')
    sel = ok.copy() if m is None or n is None else ok & np.isfinite(m - n) & (m - n > 0)
    print(f'  {roi}: {ok.sum()} usable voxels, {sel.sum()} beat the null ({model_cv})')
    return y[:, ok], sel[ok]


# ── gates 1-2 ────────────────────────────────────────────────────────────────

def tuning_splits(par):
    """Named (boolean A, boolean B) trial splits."""
    odd = (par['run'] % 2 == 1).to_numpy()
    splits = {}
    for s in sorted(par['session'].unique()):
        ses = (par['session'] == s).to_numpy()
        splits[f'within-ses{s}'] = (ses & odd, ses & ~odd)
    s1, s2 = sorted(par['session'].unique())
    splits['across-ses'] = ((par['session'] == s1).to_numpy(), (par['session'] == s2).to_numpy())
    splits['pooled-oddeven'] = (odd, ~odd)
    return splits


def gate_tuning(subject, par, y_npc, sel_npc, y_v1, sel_v1):
    rows, pairs = [], []
    for roi, y, sel, kind, col in (('NPC', y_npc, sel_npc, 'value', 'value'),
                                   ('V1', y_v1, sel_v1, 'orientation', 'orientation')):
        yd = demean_runs(y, par)
        for split, (a, b) in tuning_splits(par).items():
            la, _, ra = grid_fit(yd[a], par.loc[a, col].to_numpy(), kind)
            lb, _, rb = grid_fit(yd[b], par.loc[b, col].to_numpy(), kind)
            for selection, keep in (('all', np.ones_like(sel)), ('selected', sel)):
                if keep.sum() < 10:
                    continue
                rel = (circcorr_axial(la[keep], lb[keep]) if kind == 'orientation'
                       else float(np.corrcoef(la[keep], lb[keep])[0, 1]))
                rows.append({'subject': subject, 'roi': roi, 'split': split,
                             'selection': selection, 'n_vox': int(keep.sum()),
                             'reliability': rel,
                             'metric': 'circular r' if kind == 'orientation' else 'pearson r',
                             'median_fit_r': float(np.median((ra[keep] + rb[keep]) / 2))})
            if split in ('across-ses', 'pooled-oddeven'):
                pairs.append(pd.DataFrame({'subject': subject, 'roi': roi, 'split': split,
                                           'est_a': la[sel], 'est_b': lb[sel]}))
    return pd.DataFrame(rows), pd.concat(pairs, ignore_index=True)


# ── gates 3-4 ────────────────────────────────────────────────────────────────

def channel_series(res_v1, v1_pref, n_channels):
    """V1 orientation channels as deviations from the V1 mean residual.

    Binning V1 voxels by preferred orientation (bins centred on 0, 180/K, ...),
    averaging residuals within a bin, then subtracting the across-channel mean,
    so the shared trial-to-trial V1 fluctuation cannot masquerade as tuning.
    Returns trials x channels (z-scored), bin centres, and voxels per bin.
    """
    width = 180. / n_channels
    centres = np.arange(n_channels) * width
    idx = np.floor(((v1_pref + width / 2) % 180) / width).astype(int)
    counts = np.bincount(idx, minlength=n_channels)
    ch = np.stack([res_v1[:, idx == k].mean(1) if counts[k] else
                   np.full(len(res_v1), np.nan) for k in range(n_channels)], 1)
    return zscore(ch - ch.mean(1, keepdims=True)), centres, counts


def connective_field(res_npc, d):
    """channels x voxels correlation, centred over channels (the CF shape)."""
    cf = zscore(d).T @ zscore(res_npc) / len(d)
    return cf - cf.mean(0, keepdims=True)


def profile_corr(a, b):
    """Per-voxel correlation over channels between two channels x voxels arrays."""
    return colcorr(a, b)


def predicted_cf(mode, fwhm, centres, ori, values):
    """f_i(m(theta_k)): each voxel's value curve read out at the channels' values."""
    v = np.interp(centres, ori, values)
    p = lognormal_mode_fwhm(v[:, None], mode[None, :], fwhm[None, :])
    return p - p.mean(0, keepdims=True)


def gate_cf(subject, par, y_npc, sel_npc, y_v1, sel_v1, n_channels, n_perm, amplitudes,
            rng, v1_voxels='selected'):
    ori, maps = load_mappings()
    sessions = sorted(par['session'].unique())
    other = dict(zip(sessions, sessions[::-1]))
    if v1_voxels == 'all':
        sel_v1 = np.ones_like(sel_v1)
    yv = demean_runs(y_v1, par)

    # V1 labels per session come from the OTHER session's tuning.
    v1_label = {}
    for s in sessions:
        o = (par['session'] == other[s]).to_numpy()
        v1_label[s], _, _ = grid_fit(yv[o][:, sel_v1], par.loc[o, 'orientation'].to_numpy(),
                                     'orientation')

    # Joint (both sessions) value tuning for the NPC label.
    mode, fwhm, _ = grid_fit(demean_runs(y_npc, par)[:, sel_npc],
                             par['value'].to_numpy(), 'value')

    rel_rows, prof_rows, sim_rows = [], [], []
    npc_sel = y_npc[:, sel_npc]
    v1_sel = y_v1[:, sel_v1]
    for s in sessions:
        ses = (par['session'] == s).to_numpy()
        cond = par.loc[ses, 'condition'].iloc[0]
        p_ses = par[ses].reset_index(drop=True)
        odd = (p_ses['run'] % 2 == 1).to_numpy()

        # Gate 3: odd vs even runs, residualised separately.
        cfs, counts = {}, None
        for half, keep in (('odd', odd), ('even', ~odd)):
            ph = p_ses[keep].reset_index(drop=True)
            rn = residualise(npc_sel[ses][keep], ph)
            rv = residualise(v1_sel[ses][keep], ph)
            d, centres, counts = channel_series(rv, v1_label[s], n_channels)
            cfs[half] = connective_field(rn, d)
            glob = zscore(rv.mean(1)) @ zscore(rn) / len(rn)
            prof = cfs[half].mean(1)
            for k, c in enumerate(centres):
                prof_rows.append({'subject': subject, 'session': s, 'condition': cond,
                                  'half': half, 'channel_deg': c, 'cf': prof[k]})
            cfs[half + '_glob'] = float(np.nanmean(glob))
        full = connective_field(residualise(npc_sel[ses], p_ses),
                                channel_series(residualise(v1_sel[ses], p_ses),
                                               v1_label[s], n_channels)[0])
        a, b = cfs['odd'], cfs['even']
        roi_a, roi_b = a.mean(1), b.mean(1)
        rel_rows.append({
            'subject': subject, 'session': s, 'condition': cond,
            'n_npc': int(sel_npc.sum()), 'n_v1': int(sel_v1.sum()),
            'min_v1_per_channel': int(counts.min()),
            'voxel_rel': float(np.nanmean(profile_corr(a, b))),
            'voxel_specific_rel': float(np.nanmean(profile_corr(a - roi_a[:, None],
                                                                b - roi_b[:, None]))),
            'roi_profile_rel': float(np.corrcoef(roi_a, roi_b)[0, 1]),
            'cf_modulation_sd': float(np.nanmean(np.r_[a.std(0), b.std(0)])),
            'global_coupling': float(np.mean([cfs['odd_glob'], cfs['even_glob']])),
            'full_cf_sd': float(np.nanmean(full.std(0))),
            'full_cf_specific_sd': float(np.nanmean(
                (full - full.mean(1, keepdims=True)).std(0))),
            'v1_voxels': v1_voxels,
        })

    # Gate 4: shuffled NPC residuals plus an injected CF, full sessions.
    res = {}
    for s in sessions:
        ses = (par['session'] == s).to_numpy()
        p_ses = par[ses].reset_index(drop=True)
        d, centres, _ = channel_series(residualise(v1_sel[ses], p_ses), v1_label[s], n_channels)
        cond = p_ses['condition'].iloc[0]
        wrong = 'cdf' if cond == 'inverse_cdf' else 'inverse_cdf'
        res[s] = dict(
            rn=zscore(residualise(npc_sel[ses], p_ses)), d=d,
            right=predicted_cf(mode, fwhm, centres, ori, maps[cond]),
            wrong=predicted_cf(mode, fwhm, centres, ori, maps[wrong]),
            invariant=predicted_cf(mode, fwhm, centres, ori,
                                   (maps['cdf'] + maps['inverse_cdf']) / 2))
    for perm in range(n_perm):
        order = {s: rng.permutation(len(r['rn'])) for s, r in res.items()}
        for injection in ('specific', 'invariant'):
            for amp in amplitudes:
                scores, mods, spec = [], [], []
                for s, r in res.items():
                    inj = r['right'] if injection == 'specific' else r['invariant']
                    y = np.sqrt(1 - amp ** 2) * r['rn'][order[s]] + amp * zscore(r['d'] @ inj)
                    cf = connective_field(y, r['d'])
                    scores.append(profile_corr(cf, r['right']) - profile_corr(cf, r['wrong']))
                    mods.append(cf.std(0))
                    spec.append((cf - cf.mean(1, keepdims=True)).std(0))
                sim_rows.append({'subject': subject, 'perm': perm, 'injection': injection,
                                 'amplitude': amp,
                                 'score': float(np.nanmean(np.concatenate(scores))),
                                 'cf_modulation_sd': float(np.nanmean(np.concatenate(mods))),
                                 'cf_specific_sd': float(np.nanmean(np.concatenate(spec))),
                                 'v1_voxels': v1_voxels})

    return pd.DataFrame(rel_rows), pd.DataFrame(prof_rows), pd.DataFrame(sim_rows)


# ── main ─────────────────────────────────────────────────────────────────────

def main(subject, bids_folder=BIDS_FOLDER, target_roi='NPCr',
         source_roi='BensonV1ecc075-375', n_channels=8, n_perm=50,
         amplitudes=(0., .05, .1, .2), smoothed=False, seed=0, v1_voxels='selected'):
    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sorted(sub.get_sessions())
    if len(sessions) != 2:
        raise SystemExit(f'sub-{subject}: needs exactly 2 sessions, has {sessions}')

    par = get_paradigm(sub, sessions)
    print(f'sub-{subject}: {len(par)} gabor trials, mappings '
          f'{dict(par.groupby("session")["condition"].first())}')
    betas = sub.get_single_trial_estimates(sessions, desc='gabor', smoothed=smoothed)
    if betas.shape[3] != len(par):
        raise SystemExit(f'{betas.shape[3]} betas vs {len(par)} trials')

    y_npc, sel_npc = load_roi(sub, betas, target_roi, 'aprf.cv', bids_folder, smoothed)
    y_v1, sel_v1 = load_roi(sub, betas, source_roi, 'vonmises.cv', bids_folder, smoothed)

    out = (bids_folder / 'derivatives' / 'connective_fields'
           / ('gates' if v1_voxels == 'selected' else f'gates_v1-{v1_voxels}') / f'sub-{subject}')
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f'sub-{subject}'

    tun, pairs = gate_tuning(subject, par, y_npc, sel_npc, y_v1, sel_v1)
    tun.to_csv(f'{stem}_desc-tuningreliability.tsv', sep='\t', index=False)
    pairs.to_csv(f'{stem}_desc-tuningpairs.tsv.gz', sep='\t', index=False)
    print(tun[tun.selection == 'selected'].to_string(index=False))

    rel, prof, sim = gate_cf(subject, par, y_npc, sel_npc, y_v1, sel_v1, n_channels,
                             n_perm, list(amplitudes), np.random.default_rng(seed),
                             v1_voxels=v1_voxels)
    rel.to_csv(f'{stem}_desc-cfreliability.tsv', sep='\t', index=False)
    prof.to_csv(f'{stem}_desc-cfprofile.tsv', sep='\t', index=False)
    sim.to_csv(f'{stem}_desc-simulation.tsv', sep='\t', index=False)
    print(rel.to_string(index=False))
    print(sim.groupby(['injection', 'amplitude'])[['score', 'cf_modulation_sd']].mean())
    print(f'saved to {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('subject')
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    p.add_argument('--target-roi', default='NPCr')
    p.add_argument('--source-roi', default='BensonV1ecc075-375')
    p.add_argument('--n-channels', type=int, default=8)
    p.add_argument('--n-perm', type=int, default=50)
    p.add_argument('--amplitudes', type=float, nargs='+', default=[0., .05, .1, .2],
                   help='Injected coupling strengths (correlation scale)')
    p.add_argument('--smoothed', action='store_true')
    p.add_argument('--v1-voxels', choices=['selected', 'all'], default='selected',
                   help='V1 voxels that feed the orientation channels')
    args = p.parse_args()
    main(args.subject, bids_folder=args.bids_folder, target_roi=args.target_roi,
         source_roi=args.source_roi, n_channels=args.n_channels, n_perm=args.n_perm, amplitudes=args.amplitudes,
         smoothed=args.smoothed, v1_voxels=args.v1_voxels)
