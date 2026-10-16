#!/usr/bin/env python3
"""The NPCr <- V1 mapping-coupling analysis, laid out on the brain, in one PDF.

Page 1, the idea. An example participant's right hemisphere, flattened, with
the stimulated part of V1 (0.75-3.75 deg) magnified:
  a  V1 coloured by preferred orientation, NPCr by preferred value.
  b  The same V1 map re-coloured by what each orientation is WORTH in a CDF
     session, so V1 and NPCr share one colour scale (CHF). An NPCr voxel that
     prefers 16 CHF should couple to the V1 populations that now carry its
     colour (arrows).
  c  The same for an inverse-CDF session: V1's colours move, NPCr's do not, so
     the predicted partners of the same NPCr voxel move.
  d-g  The logic in one dimension (task mappings only, no brain data).
Page 2, the data (every participant with the analysis):
  h-i  Predicted vs observed CDF - inverse-CDF coupling maps.
  j    The observed map read out along the two theta* curves.
  k    Coupling aligned to theta* of this session vs of the other session.
  l    Every participant's mapping score, coloured by session order.
  m    The mapping score under every V1 readout and control.

Brain maps are fsaverage surfaces of the participant's own fits (smoothed),
shown where the fit beats that participant's null. Everything else reads the
connective-field outputs under derivatives/connective_fields/.

Needs pycortex: run in the pycortex2 env.

  python -m abstract_values.connective_fields.plot_overview
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import ConnectionPatch, Rectangle
from scipy import stats
from scipy.ndimage import gaussian_filter

from abstract_values.connective_fields.gates import load_mappings, lognormal_mode_fwhm
from abstract_values.connective_fields.plot_coupling_explainer import (
    CUR_COL, MAP_COL, OTH_COL, ROOT, wrapped_profile)
from abstract_values.connective_fields.plot_coupling_maps import (
    CONDS, load, map_agreement, observed_maps, predicted_maps, ridge_readout, show)
from abstract_values.connective_fields.plot_gates import read  # sets rcParams
from abstract_values.utils.data import BIDS_FOLDER
from abstract_values.visualize import qc_flatmaps as qf
from abstract_values.visualize.group_surface_maps import load_fsaverage
from abstract_values.visualize.qc_tuning_maps import ORI_CMAP, VALUE_CMAP

REPO = Path(__file__).resolve().parents[2]
DERIV = BIDS_FOLDER / 'derivatives'
OUT = REPO / 'notes' / 'figures' / 'cf_mapping_coupling_overview.pdf'
NV = 163842
EXAMPLE = '12'
EX_VALUE = 16.
CROP = dict(x=(2, 150), y=(-54, 84))          # flat coords, RH: V1 pole to IPS
ZOOM = dict(x=(9, 41), y=(-41, 9))            # the stimulated V1 annulus
MAIN = 'iem-k24-kappa16'
MAP_NAME = {'cdf': 'CDF', 'inverse_cdf': 'Inverse CDF'}
STAR = dict(marker='*', ms=10, mfc='white', mec='k', mew=.8, zorder=8)


def letter(ax, s, x=-.12, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='bottom', ha='right')


def pfmt(p):
    if p < 1e-3:
        m, e = f'{p:.0e}'.split('e')
        return f'p = {m}·10$^{{{int(e)}}}$'
    return f'p = {p:.2g}' if p < .01 else f'p = {p:.2f}'


# ── brain ────────────────────────────────────────────────────────────────────

def surface_maps(s):
    """The participant's preferred orientation (V1 annulus) and value (NPCr)."""
    g = lambda m, d: load_fsaverage(DERIV, m, s, d, True)  # noqa: E731
    null = g('aprf-null.cv', 'cvr2')
    theta = np.rad2deg(.5 * np.arctan2(g('vonmises', 'orient_sin'),
                                       g('vonmises', 'orient_cos'))) % 180
    atlas = {}
    for d in ('Varea', 'Eccen'):
        atlas[d] = np.concatenate([nib.load(str(
            DERIV / 'neuropythy_atlas' / f'sub-{s}' /
            f'sub-{s}_desc-benson14{d}_space-fsaverage_hemi-{h}.func.gii')).darrays[0].data
            for h in 'LR'])
    ann = (atlas['Varea'] == 1) & (atlas['Eccen'] >= .75) & (atlas['Eccen'] <= 3.75)
    ann[:NV] = False                                       # right hemisphere only
    lab = nib.load(str(DERIV / 'surface_masks' /
                       'desc-NPC_R_space-fsaverage_hemi-rh.label.gii')).darrays[0].data
    npc = np.zeros(2 * NV, bool)
    npc[NV:] = lab > 0
    return dict(theta=theta, ori_ok=g('vonmises.cv', 'cvr2') > null, ann=ann,
                value=g('aprf', 'mode'), val_ok=g('aprf.cv', 'cvr2') > null,
                val_r2=g('aprf.cv', 'cvr2') - null, npc=npc)


def flat_geometry():
    import cortex
    pts, polys = cortex.db.get_surf('fsaverage', 'flat', merge=True, nudge=True)
    return pts[:, :2], polys


def in_box(pts, box):
    return ((pts[:, 0] > box['x'][0]) & (pts[:, 0] < box['x'][1]) &
            (pts[:, 1] > box['y'][0]) & (pts[:, 1] < box['y'][1]))


def outline(ax, pts, polys, mask, **kw):
    """Contour of a vertex mask on the flat mesh."""
    keep = mask[polys].any(1)
    ax.tricontour(pts[:, 0], pts[:, 1], polys[keep], mask.astype(float), levels=[.5], **kw)


def brain_vertex(m, v1_values, v1_lims, v1_cmap):
    """V1 annulus in its own colours, NPCr in value colours, curvature elsewhere."""
    import cortex
    show_v1 = m['ann'] & m['ori_ok']
    show_npc = m['npc'] & m['val_ok']
    v1 = qf.blended(np.where(show_v1, v1_values, 0), show_v1 * 1., *v1_lims, v1_cmap)
    npc = qf.blended(np.where(show_npc, m['value'], 0), show_npc * 1., 2, 42, VALUE_CMAP)
    rgb = [np.where(show_npc, getattr(npc, c).data, getattr(v1, c).data)
           for c in ('red', 'green', 'blue')]
    return cortex.VertexRGB(*rgb, 'fsaverage')


def brain_panel(fig, rect, m, pts, polys, vtx):
    """Main crop plus a magnifier on the V1 annulus; returns (main, inset)."""
    ax = fig.add_axes(rect)
    rois = ['V1', 'V2', 'V3', 'V3a', 'IPS0', 'IPS1', 'IPS2', 'IPS3']
    qf.draw_flat(ax, vtx, rois=rois, labels=False)
    ax.set_xlim(*CROP['x'])
    ax.set_ylim(*CROP['y'])
    outline(ax, pts, polys, m['npc'], colors='k', linewidths=.9)
    ax.text(128, 74, 'NPCr', fontsize=7, fontweight='bold', ha='center')
    ax.add_patch(Rectangle((ZOOM['x'][0], ZOOM['y'][0]), np.ptp(ZOOM['x']), np.ptp(ZOOM['y']),
                           fill=False, ec='k', lw=.7, zorder=6))
    # magnifier, top-left, same aspect as the zoom box
    zw = .33
    zh = zw * np.ptp(ZOOM['y']) / np.ptp(ZOOM['x']) * np.ptp(CROP['x']) / np.ptp(CROP['y'])
    ins = ax.inset_axes([.0, 1 - zh, zw, zh])
    qf.draw_flat(ins, vtx, rois=['V1', 'V2'], labels=False)
    ins.set_xlim(*ZOOM['x'])
    ins.set_ylim(*ZOOM['y'])
    ins.set_axis_on()
    ins.set_xticks([])
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(.9)
    ins.text(1.04, .985, 'V1, 0.75–3.75°\n(magnified)', transform=ins.transAxes, fontsize=6,
             fontweight='bold', va='top', ha='left', linespacing=1.1, zorder=20,
             bbox=dict(facecolor='w', edgecolor='none', alpha=.8, pad=.8))
    for (xa, ya), (xb, yb) in (((ZOOM['x'][0], ZOOM['y'][1]), (0, 0)),
                               ((ZOOM['x'][1], ZOOM['y'][1]), (1, 0))):
        fig.add_artist(ConnectionPatch((xa, ya), (xb, yb), ax.transData, ins.transAxes,
                                       color='k', lw=.5, zorder=5))
    return ax, ins


def example_voxel(m, pts):
    """A well-fitted NPCr vertex near EX_VALUE CHF, central in the ROI."""
    ok = m['npc'] & m['val_ok'] & (np.abs(m['value'] - EX_VALUE) < 1.5)
    c = pts[m['npc']].mean(0)
    score = m['val_r2'] - 2e-4 * np.linalg.norm(pts - c, axis=1)
    return np.flatnonzero(ok)[np.argmax(score[ok])]


def partners(m, pts, worth, n=4, tol=2.):
    """Up to n spread-out V1 vertices whose orientation is worth ~EX_VALUE."""
    cand = np.flatnonzero(m['ann'] & m['ori_ok'] & in_box(pts, ZOOM) &
                          (np.abs(worth - EX_VALUE) < tol))
    if not len(cand):
        return cand
    chosen = [cand[np.argmin(np.abs(worth[cand] - EX_VALUE))]]
    for _ in range(n - 1):
        d = np.min(np.linalg.norm(pts[cand][:, None] - pts[chosen][None], axis=2), 1)
        if d.max() < 4:
            break
        chosen.append(cand[np.argmax(d)])
    return np.array(chosen)


def arrows(fig, ax, ins, pts, src, dst):
    for k, j in enumerate(dst):
        fig.add_artist(ConnectionPatch(
            pts[src], pts[j], ax.transData, ins.transData, arrowstyle='-|>',
            connectionstyle=f'arc3,rad={.18 + .06 * k}', mutation_scale=7, lw=1.0,
            color='k', shrinkA=5, shrinkB=3, zorder=9))
        ins.plot(*pts[j], 'o', ms=4.5, mfc='none', mec='k', mew=1, zorder=8)


def ori_key(fig, rect):
    k = fig.add_axes(rect, projection='polar')
    th = np.linspace(0, np.pi, 181)
    for a0, a1 in zip(th[:-1], th[1:]):
        k.bar((a0 + a1) / 2, 1, width=a1 - a0, bottom=.45,
              color=ORI_CMAP(np.rad2deg(a0) / 180), edgecolor='none')
    for deg in (0, 45, 90, 135):
        a = np.deg2rad(deg)
        k.plot([a, a], [.5, 1.4], color='k', lw=1.1, solid_capstyle='round')
        k.text(a, 1.95, f'{deg}°', ha='center', va='center', fontsize=5.5)
    k.set_thetamin(0)
    k.set_thetamax(180)
    k.set_ylim(0, 1.45)
    k.set_axis_off()
    return k


def page_idea(pdf, ori, maps):
    pts, polys = flat_geometry()
    m = surface_maps(EXAMPLE)
    src = example_voxel(m, pts)
    W, H = 7.25, 5.5
    fig = plt.figure(figsize=(W, H))
    fig.text(.01, .995, 'Does NPCr couple to the V1 populations that currently carry '
             'its preferred value?', fontsize=9.5, fontweight='bold', va='top')
    fig.text(.01, .968, f'Example participant (sub-{EXAMPLE}), right hemisphere, flattened; '
             'boxed and magnified: the part of V1 the gabor covered. Coloured where the '
             'tuning model beats its null.', fontsize=6.8, color='.35',
             va='top')
    pw = .318
    ph = pw * W / H * np.ptp(CROP['y']) / np.ptp(CROP['x'])
    y0 = .925 - ph
    worth = {c: np.interp(m['theta'], ori, maps[c]) for c in CONDS}
    specs = [('What each region is tuned to', 'k', brain_vertex(m, m['theta'], (0, 180),
                                                                ORI_CMAP), None)]
    specs += [(f'{MAP_NAME[c]} session: V1 read out in CHF', MAP_COL[c],
               brain_vertex(m, worth[c], (2, 42), VALUE_CMAP), c) for c in CONDS]
    for i, (title, col, vtx, c) in enumerate(specs):
        ax, ins = brain_panel(fig, [.008 + i * .33, y0, pw, ph], m, pts, polys, vtx)
        ax.plot(*pts[src], **STAR)
        if c is not None:
            arrows(fig, ax, ins, pts, src, partners(m, pts, worth[c]))
        ax.set_title(title, fontsize=7.5, color=col, pad=3)
        letter(ax, 'abc'[i], x=.0, y=1.0)

    # keys
    ky = y0 - .085
    ori_key(fig, [.075, ky - .015, .14, .07])
    fig.text(.145, ky - .03, 'V1: preferred orientation', ha='center', fontsize=6.5)
    cb = qf.colorbar(fig, [.37, ky + .02, .28, .011], VALUE_CMAP, 2, 42, '',
                     ticks=[2, 12, 22, 32, 42])
    cb.ax.tick_params(labelsize=6)
    fig.text(.51, ky - .03, 'NPCr: preferred value  ·  V1 (b, c): value of the preferred '
             'orientation (CHF)', ha='center', fontsize=6.5)
    kx, kyy = .695, ky + .035
    fig.add_artist(plt.Line2D([kx], [kyy], transform=fig.transFigure, **STAR))
    fig.text(kx + .015, kyy, f'NPCr voxel preferring {EX_VALUE:g} CHF', fontsize=6.5,
             va='center')
    fig.add_artist(ConnectionPatch((kx - .008, kyy - .03), (kx + .01, kyy - .03),
                                   fig.transFigure, fig.transFigure, arrowstyle='-|>',
                                   mutation_scale=7, lw=1, color='k'))
    fig.add_artist(plt.Line2D([kx + .012], [kyy - .03], transform=fig.transFigure,
                                marker='o', ms=4.5, mfc='none', mec='k', mew=1))
    fig.text(kx + .03, kyy - .03, 'V1 vertices of the same colour:\nits predicted partners',
             fontsize=6.5, va='center')

    # d-g: the logic in one dimension
    lo, hh = .075, .23
    ax_d = fig.add_axes([.065, lo, .19, hh])
    ax_e = fig.add_axes([.315, lo, .12, hh])
    ax_f = fig.add_axes([.5, lo, .21, hh])
    ax_g = fig.add_axes([.785, lo, .2, hh])
    th_ex = {c: np.interp(EX_VALUE, maps[c], ori) for c in CONDS}
    oo = np.linspace(0, 180, 361)
    for c in CONDS:
        ax_d.plot(oo, np.interp(oo, ori, maps[c]), color=MAP_COL[c], lw=1.4)
        ax_d.plot([th_ex[c]] * 2, [2, EX_VALUE], color=MAP_COL[c], lw=.8, ls=':')
    ax_d.plot([0, max(th_ex.values())], [EX_VALUE] * 2, color='.4', lw=.8, ls=':')
    ax_d.text(100, 9, 'CDF', color=MAP_COL['cdf'], fontsize=6.5)
    ax_d.text(60, 31, 'Inverse CDF', color=MAP_COL['inverse_cdf'], fontsize=6.5, ha='right')
    ax_d.set(xticks=[0, 90, 180], yticks=[2, 22, 42], xlabel='Gabor orientation (°)',
             ylabel='Value (CHF)', xlim=(0, 180), ylim=(2, 42))
    ax_d.set_title('Two mappings', fontsize=7.5)

    vv = np.linspace(2, 42, 300)
    resp = lognormal_mode_fwhm(vv, EX_VALUE, 8.)
    ax_e.fill_between(vv, resp, color=VALUE_CMAP((EX_VALUE - 2) / 40), alpha=.45, lw=0)
    ax_e.plot(vv, resp, color='k', lw=1.2)
    ax_e.plot(EX_VALUE, resp.max() * 1.12, **STAR)
    ax_e.set(xticks=[2, 16, 42], yticks=[], xlabel='Value (CHF)', ylabel='Response',
             ylim=(0, resp.max() * 1.22))
    ax_e.set_title('NPCr voxel', fontsize=7.5)

    for c in CONDS:
        ax_f.plot(oo, lognormal_mode_fwhm(np.interp(oo, ori, maps[c]), EX_VALUE, 8.),
                  color=MAP_COL[c], lw=1.4)
        ax_f.text(th_ex[c] + (6 if c == 'cdf' else -6), 1.05, f'θ* = {th_ex[c]:.0f}°',
                  color=MAP_COL[c], ha='left' if c == 'cdf' else 'right', fontsize=6.5)
    ax_f.set(ylim=(0, 1.3), xticks=[0, 45, 90, 135, 180], yticks=[],
             xlabel='V1 preferred orientation (°)', ylabel='Predicted coupling')
    ax_f.set_title('Its predicted V1 partners', fontsize=7.5)

    sh = np.interp(vv, maps['cdf'], ori) - np.interp(vv, maps['inverse_cdf'], ori)
    ax_g.fill_between(vv, 0, sh, color='.88', lw=0)
    ax_g.plot(vv, sh, color='.2', lw=1.2)
    ax_g.plot(EX_VALUE, th_ex['cdf'] - th_ex['inverse_cdf'], **STAR)
    ax_g.axhline(0, color='.6', lw=.6)
    ax_g.set(xticks=[2, 12, 22, 32, 42], yticks=[-20, 0, 20], xlabel='Preferred value (CHF)',
             ylabel='θ*, CDF − inverse CDF (°)')
    ax_g.set_title('The shift depends on the voxel', fontsize=7.5)
    for ax, s in zip((ax_d, ax_e, ax_f, ax_g), 'defg'):
        sns.despine(ax=ax, offset=3, trim=True)
        letter(ax, s, x=-.2, y=1.06)
    for ax in (ax_e, ax_f):
        ax.spines['left'].set_visible(False)
    pdf.savefig(fig, dpi=300)
    plt.close(fig)
    print(f'page 1: example vertex {src}, value {m["value"][src]:.1f}')


# ── results ──────────────────────────────────────────────────────────────────

def mapping_score(variant, direction='npc_from_v1'):
    s = read(ROOT / variant, 'scores')
    s = s[s.direction == direction]
    obs = s[s.kind == 'observed'].set_index('subject').score
    return obs - s[s.kind == 'label-shuffle'].groupby('subject').score.mean().reindex(obs.index)


def session_order():
    """First session's mapping per CF subject label (pil01 -> 1, pil02 -> 2)."""
    ps = pd.read_csv(REPO / 'notes' / 'data' / 'qc_behavior_per_session.tsv', sep='\t')
    first = ps[ps.session == 1].set_index('subject').mapping
    lab = lambda s: int(s[3:]) if s.startswith('pil') else int(s)  # noqa: E731
    return lambda s: first.get(lab(s), np.nan)


FOREST = [  # (section, label, variant, direction)
    ('Readout of V1', 'Encoding model, κ = 16 (main)', 'coupling_iem-k24-kappa16', 'npc_from_v1'),
    ('Readout of V1', 'Encoding model, κ = 8', 'coupling_iem-k16-kappa8', 'npc_from_v1'),
    ('Readout of V1', 'Encoding model, κ = 2', 'coupling_iem', 'npc_from_v1'),
    ('Readout of V1', '8 orientation bins', 'coupling', 'npc_from_v1'),
    ('Readout of V1', '16 orientation bins', 'coupling_bins-16', 'npc_from_v1'),
    ('Neighbouring trials', '±1 removed (8 bins)', 'coupling_lags-1', 'npc_from_v1'),
    ('Neighbouring trials', '±1 removed (κ = 2)', 'coupling_lags-1_iem', 'npc_from_v1'),
    ('Neighbouring trials', '±2 removed (8 bins)', 'coupling_lags-2', 'npc_from_v1'),
    ('Neighbouring trials', '±1 real, ±2 sham', 'coupling_lags-2-sham', 'npc_from_v1'),
    ('Shared gain (κ = 16)', 'Gain regressed out', 'coupling_iem-k24-kappa16_gain', 'npc_from_v1'),
    ('Shared gain (κ = 16)', 'Low-drive trials', 'coupling_iem-k24-kappa16_drive-low', 'npc_from_v1'),
    ('Shared gain (κ = 16)', 'High-drive trials', 'coupling_iem-k24-kappa16_drive-high', 'npc_from_v1'),
    ('NPCr voxels (κ = 16)', 'Value-tuned', 'coupling_iem-k24-kappa16_npc-value', 'npc_from_v1'),
    ('NPCr voxels (κ = 16)', 'Orientation-tuned', 'coupling_iem-k24-kappa16_npc-orientation', 'npc_from_v1'),
    ('Reverse direction', 'V1 as target (8 bins)', 'coupling', 'v1_from_npc'),
]


def forest(ax):
    y, ticks, labels, sec_prev = 0, [], [], None
    for sec, lab, variant, direction in FOREST:
        if not (ROOT / variant).exists():
            continue
        if sec != sec_prev:
            y -= .55 if sec_prev else 0
            ax.text(-.033, y, sec, fontsize=6.5, fontweight='bold', color='.25', va='center',
                    ha='right')
            y -= 1
            sec_prev = sec
        d = mapping_score(variant, direction)
        t = stats.ttest_1samp(d, 0)
        ci = stats.t.ppf(.975, len(d) - 1) * d.sem()
        main = lab.endswith('(main)')
        col = CUR_COL if (t.pvalue < .05 and d.mean() > 0) else '.55'
        ax.plot([d.mean() - ci, d.mean() + ci], [y, y], color=col, lw=1.3 if main else .9,
                solid_capstyle='butt')
        ax.plot(d.mean(), y, 'D' if main else 'o', ms=4.8 if main else 3.4, mfc=col,
                mec='k' if main else col, mew=.6)
        ax.text(.047, y, pfmt(t.pvalue), fontsize=6, va='center', ha='left',
                color='k' if t.pvalue < .05 else '.5')
        ticks.append(y)
        labels.append(lab)
        y -= 1
    ax.axvline(0, color='.6', lw=.6, ls='--', zorder=0)
    ax.set_yticks(ticks, labels, fontsize=6.3)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(-.032, .045)
    ax.set_xticks([-.02, 0, .02, .04])
    ax.set_ylim(y + .4, .6)
    ax.set_xlabel('Mapping score, mean ± 95% CI (Δr)')


def page_results(pdf, ori, maps):
    fig = plt.figure(figsize=(7.25, 9.4))
    d_main = mapping_score(f'coupling_{MAIN}')
    n = len(d_main)
    t_main = stats.ttest_1samp(d_main, 0)
    fig.text(.01, .997, f'NPCr–V1 coupling follows the current mapping (n = {n})',
             fontsize=9.5, fontweight='bold', va='top')
    fig.text(.01, .977, 'Trial-to-trial fluctuations only: every stimulus-locked response is '
             'removed first (one mean per orientation and run, per session). The two '
             'sessions\' coupling maps are subtracted, so coupling that does not change with '
             'the mapping cancels.', fontsize=6.8, color='.35', va='top', wrap=True)

    # h, i: predicted and observed difference maps; j: ridge readout
    mp, pkv = load('peaks')
    obs, subs, vbins = observed_maps(mp)
    pred = predicted_maps(pkv, subs, vbins, maps, ori)
    r = map_agreement(obs, pred)
    tr = stats.ttest_1samp(r, 0)
    pdiff = np.nanmean(pred['cdf'] - pred['inverse_cdf'], 0)
    odiff = (obs['cdf'] - obs['inverse_cdf']).mean(0)
    vv = np.linspace(2, 42, 400)
    top, mh = .67, .24
    for k, (title, mat, smooth, lab, cb_lab, cmap) in enumerate((
            ('Predicted', pdiff, False, 'h', 'Δ predicted (z)', 'PuOr_r'),
            ('Observed', odiff, True, 'i', 'Δ coupling (r)', 'vlag'))):
        ax = fig.add_axes([.075 + k * .31, top, .2, mh])
        vis = gaussian_filter(np.nan_to_num(mat), 1, mode=('wrap', 'nearest')) if smooth else mat
        im = show(ax, mat, vbins, np.nanquantile(np.abs(vis), .99), cmap, smooth=smooth)
        for c in CONDS:
            ax.plot(vv, np.interp(vv, maps[c], ori), color=MAP_COL[c], lw=1)
        ax.set(xticks=[2, 12, 22, 32, 42], yticks=[0, 45, 90, 135, 180], xlim=(2, 42),
               ylim=(-2.5, 177.5), xlabel='NPCr preferred value (CHF)',
               ylabel='V1 orientation (°)' if k == 0 else '')
        ax.set_title(f'{title}: CDF − inverse-CDF sessions', fontsize=7.5)
        cax = fig.add_axes([.075 + k * .31 + .205, top + .04, .007, .16])
        cb = fig.colorbar(im, cax=cax)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=5.5, length=2)
        cb.set_label(cb_lab, fontsize=6, labelpad=1)
        letter(ax, lab, x=-.2, y=1.06)
        if k == 0:
            ax.text(40, 8, 'θ* CDF', color=MAP_COL['cdf'], ha='right', fontsize=6.3)
            ax.text(4, 160, 'θ* inverse CDF', color=MAP_COL['inverse_cdf'], fontsize=6.3)
        else:
            ax.text(.03, .03, f'Match to h: r = {r.mean():.3f}\n{pfmt(tr.pvalue)}',
                    transform=ax.transAxes, fontsize=6.3, va='bottom',
                    bbox=dict(facecolor='w', edgecolor='none', alpha=.85, pad=1))

    ax_j = fig.add_axes([.74, top, .25, mh])
    rr = ridge_readout(obs['cdf'] - obs['inverse_cdf'], vbins, maps, ori)
    vfine = np.linspace(2, 42, 2001)
    sh = np.interp(vfine, maps['cdf'], ori) - np.interp(vfine, maps['inverse_cdf'], ori)
    for x in vfine[1:-1][np.diff(np.sign(sh))[:-1] != 0]:
        ax_j.axvline(x, color='.9', lw=3, zorder=0)
    for c in CONDS:
        sns.lineplot(data=rr[rr.ridge == c], x='value', y='delta', errorbar=('se', 1),
                     color=MAP_COL[c], ax=ax_j, marker='o', ms=3, lw=1.2,
                     err_kws={'lw': 0, 'alpha': .2})
    ax_j.axhline(0, color='.6', lw=.6, ls='--')
    sep = rr.pivot_table(index='subject', columns='ridge', values='delta')
    td = stats.ttest_1samp(sep['cdf'] - sep['inverse_cdf'], 0)
    ax_j.set(xticks=[2, 12, 22, 32, 42], xlabel='NPCr preferred value (CHF)',
             ylabel='Δ coupling along the curve (r)')
    lo_, hi_ = ax_j.get_ylim()
    ax_j.set_ylim(lo_, hi_ + .3 * (hi_ - lo_))
    ax_j.text(.03, .97, 'Along θ* CDF (predicted > 0)', transform=ax_j.transAxes, va='top',
              fontsize=6.3, color=MAP_COL['cdf'])
    ax_j.text(.03, .89, 'Along θ* inverse CDF (predicted < 0)', transform=ax_j.transAxes,
              va='top', fontsize=6.3, color=MAP_COL['inverse_cdf'])
    ax_j.text(.97, .03, f'Difference: {pfmt(td.pvalue)}', transform=ax_j.transAxes,
              ha='right', fontsize=6.3)
    ax_j.set_title('Map i read along the curves', fontsize=7.5)
    letter(ax_j, 'j', x=-.2, y=1.06)

    # k: aligned profile (8 bins: the readout with one channel per 22.5 deg)
    ax_k = fig.add_axes([.075, .38, .3, .2])
    a = wrapped_profile(ROOT / 'profiles', 0)
    for ref, col, z in (('other', OTH_COL, 2), ('current', CUR_COL, 3)):
        sns.lineplot(data=a[a.reference == ref], x='offset', y='cf', errorbar=('se', 1),
                     color=col, ax=ax_k, lw=1.5, zorder=z, marker='o', ms=3,
                     err_kws={'lw': 0, 'alpha': .22})
    ax_k.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax_k.axvline(0, color='.85', lw=.6, zorder=0)
    ax_k.set(xticks=[-90, -45, 0, 45, 90], xlabel='V1 orientation relative to θ* (°)',
             ylabel='Coupling (r)')
    c0 = a[a.offset == 0].pivot_table(index='subject', columns='reference', values='cf')
    tp = stats.ttest_rel(c0['current'], c0['other'])
    lo_, hi_ = ax_k.get_ylim()
    ax_k.set_ylim(lo_, hi_ + .5 * (hi_ - lo_))
    ax_k.text(.03, .97, 'Aligned to θ* of this session', transform=ax_k.transAxes,
              color=CUR_COL, fontsize=6.3, va='top')
    ax_k.text(.03, .88, 'Aligned to θ* of the other session', transform=ax_k.transAxes,
              color=OTH_COL, fontsize=6.3, va='top')
    ax_k.text(.97, .97, f'At 0°: {pfmt(tp.pvalue)}', transform=ax_k.transAxes, ha='right',
              va='top', fontsize=6.3)
    ax_k.set_title('Voxels whose θ* moves ≥ 15° (8 V1 orientation bins)', fontsize=7.5)
    letter(ax_k, 'k', x=-.15, y=1.06)

    # l: every participant
    ax_l = fig.add_axes([.53, .07, .2, .51])
    first = session_order()
    dd = d_main.sort_values()
    order_col = {c: MAP_COL[c] for c in CONDS}
    cols = [order_col.get(first(s), '.5') for s in dd.index]
    ax_l.barh(np.arange(len(dd)), dd.values, color=cols, height=.72, lw=0)
    ax_l.axvline(0, color='.4', lw=.6)
    ax_l.set_yticks(np.arange(len(dd)), list(dd.index), fontsize=5.5)
    ax_l.tick_params(axis='y', length=0)
    ax_l.set_xlabel('Mapping score (Δr)')
    ax_l.set_ylim(-.7, len(dd) - .3)
    npos = int((dd > 0).sum())
    sign_p = stats.binomtest(npos, len(dd), .5).pvalue
    ax_l.text(1.04, .62, f'Mean {d_main.mean():+.4f} Δr\nt({n - 1}) = '
              f'{t_main.statistic:.1f}, {pfmt(t_main.pvalue)}\n\n{npos} of {n} '
              f'participants > 0\nSign test: {pfmt(sign_p)}', transform=ax_l.transAxes,
              va='top', fontsize=6.3, linespacing=1.4)
    grp = pd.DataFrame({'d': d_main, 'first': [first(s) for s in d_main.index]})
    yy = .3
    for c in CONDS:
        g = grp[grp['first'] == c].d
        tt = stats.ttest_1samp(g, 0)
        ax_l.add_patch(Rectangle((1.04, yy - .006), .07, .012, transform=ax_l.transAxes,
                                 color=MAP_COL[c], clip_on=False, lw=0))
        ax_l.text(1.14, yy, f'First session {MAP_NAME[c].lower() if c == "cdf" else "inverse CDF"}'
                  .replace('cdf', 'CDF'), transform=ax_l.transAxes, fontsize=6.3,
                  color=MAP_COL[c], va='center')
        ax_l.text(1.14, yy - .035, f'{g.mean():+.4f}, {pfmt(tt.pvalue)} (n = {len(g)})',
                  transform=ax_l.transAxes, fontsize=6, color='.3', va='center')
        yy -= .1
    ax_l.set_title('Every participant\n(encoding model, κ = 16)', fontsize=7.5)
    letter(ax_l, 'l', x=-.18, y=1.02)

    # m: forest
    ax_m = fig.add_axes([.24, .07, .14, .23])
    forest(ax_m)
    ax_m.set_title('Every readout and control', fontsize=7.5, x=-.2)
    letter(ax_m, 'm', x=-1.5, y=1.04)

    sns.despine(ax=ax_j, offset=3, trim=True)
    sns.despine(ax=ax_k, offset=3, trim=True)
    sns.despine(ax=ax_l, offset=3, left=True)
    sns.despine(ax=ax_m, offset=3, left=True)
    pdf.savefig(fig, dpi=300)
    plt.close(fig)
    print(f'page 2: main {d_main.mean():.4f}, t={t_main.statistic:.2f}, p={t_main.pvalue:.2g}, '
          f'{npos}/{n} > 0, sign p={sign_p:.2g}; map r={r.mean():.3f} p={tr.pvalue:.2g}; '
          f'ridge p={td.pvalue:.2g}; aligned 0 p={tp.pvalue:.2g}')


def main(out=OUT):
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    ori, maps = load_mappings()
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        page_idea(pdf, ori, maps)
        page_results(pdf, ori, maps)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
