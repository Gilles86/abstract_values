"""Toy V1 -> IPS network: can rewiring alone produce efficient coding of value?

V1: von Mises orientation-tuned units (axial, 180-deg period), uniform
preferred orientations (``--cardinal`` adds a cardinal over-representation).
IPS: units whose input weights from V1 depend on the session's
orientation->value mapping f. Only the weights W change between the two
mappings ('cdf' and 'inverse_cdf'); everything else is fixed.

Three IPS variants:

  value-wired   (model A, the hypothesis) IPS unit j has a FIXED preferred
                value v_j (uniform on 2-42 CHF), and
                    W_ij = exp(-(f(theta_i) - v_j)^2 / 2 s^2):
                a V1 unit projects to the IPS units whose value matches the
                value its orientation carries in this session.
  orient-wired  (model B, the confound) IPS unit j pools V1 around a fixed
                orientation theta_j, W_ij = vonMises(theta_i - theta_j). W does
                not depend on f; the unit's preferred VALUE is f(theta_j).
                Because orientations are uniform, f(theta_j) is the
                value-quantile code (Ganguli & Simoncelli efficient code).
  fixed value   (control) Gaussian tuning to value itself, uniform preferred
                values, one width, no dependence on f.

IPS rate h_j = b + g * sum_i W_ij r_i / N_V1 with one gain g per model shared by
both mappings; independent Poisson noise on IPS.

Outputs (repo-relative):
  notes/figures/sim_v1_ips_mapping.pdf (+ _p1/_p2.png): two-page story figure
  notes/figures/sim_v1_ips_mapping_details.pdf / .png: all quantitative panels
  stdout: all numbers quoted in notes/sim_v1_ips_mapping.md

Run from the repo root:
  ~/mambaforge/envs/abstract_values/bin/python -m abstract_values.simulations.v1_ips_mapping
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from functools import lru_cache

from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[2]

# --- Design ---------------------------------------------------------------
# 25-point tables from experiment/settings/sns_multisubject.yml
# (0..180 deg in 7.5-deg steps; 7.5..172.5 are the 23 presented stimuli).
ORI_TABLE = np.arange(0, 180.1, 7.5)
MAPPINGS = {
    'cdf': np.array([2.0, 5.5, 8.0, 10.0, 11.0, 11.5, 12.0, 12.5, 13.0, 14.0,
                     16.0, 18.5, 22.0, 25.5, 28.0, 30.0, 31.0, 31.5, 32.0,
                     32.5, 33.0, 34.0, 36.0, 38.5, 42.0]),
    'inverse_cdf': np.array([2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 11.5, 16.0, 18.5,
                             20.0, 21.0, 21.5, 22.0, 22.5, 23.0, 24.0, 25.5,
                             28.0, 32.5, 36.5, 39.0, 40.0, 41.0, 41.5, 42.0]),
}
STIM_ORI = ORI_TABLE[1:-1]
THETA = np.arange(7.5, 172.5 + 1e-9, 0.25)       # fine grid, stimulus range
INNER = (THETA > 12.5) & (THETA < 167.5)          # drop gradient edge effects
MODELS = ('value', 'orient')


SMOOTH_DEG = 3.75   # half a stimulus step


@lru_cache(maxsize=None)
def mapping_fn(name):
    """Orientation -> value mapping and its slope.

    PCHIP through the 25-point table, lightly smoothed in orientation
    (Gaussian, SD = SMOOTH_DEG) so that p(v) = 1 / (180 f') has no
    interpolation spikes. Max deviation from the table at the 23 stimuli is
    printed by report() (< 0.4 CHF).
    """
    x = np.arange(-60, 240.001, 0.05)
    f0 = PchipInterpolator(ORI_TABLE, MAPPINGS[name], extrapolate=False)
    y = f0(np.clip(x, 0, 180))
    slope_lo = (MAPPINGS[name][1] - MAPPINGS[name][0]) / 7.5
    slope_hi = (MAPPINGS[name][-1] - MAPPINGS[name][-2]) / 7.5
    y = np.where(x < 0, MAPPINGS[name][0] + slope_lo * x, y)
    y = np.where(x > 180, MAPPINGS[name][-1] + slope_hi * (x - 180), y)
    y = gaussian_filter1d(y, SMOOTH_DEG / 0.05)
    f = PchipInterpolator(x, y)
    return f, f.derivative()


def value_density(name, theta=THETA):
    """p(v) at v = f(theta) for theta ~ Uniform(0, 180): p(v) = 1 / (180 f')."""
    _, df = mapping_fn(name)
    return 1.0 / (180.0 * df(theta))


def value_to_theta(name, v):
    f, _ = mapping_fn(name)
    return np.interp(np.clip(v, f(THETA[0]), f(THETA[-1])), f(THETA), THETA)


# --- Network --------------------------------------------------------------
def vm_axial(d_deg, kappa):
    """Axial von Mises bump (peak 1) for an orientation difference in deg."""
    return np.exp(kappa * (np.cos(np.deg2rad(2 * d_deg)) - 1))


def v1_preferred(n, cardinal=0.0):
    """Preferred orientations at equally spaced quantiles of
    p(phi) ~ 1 + cardinal * cos(4 phi) (cardinal > 0: more units at 0/90)."""
    phi = np.linspace(0, 180, 4001)
    cdf = np.cumsum(1 + cardinal * np.cos(np.deg2rad(4 * phi)))
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    return np.interp((np.arange(n) + 0.5) / n, cdf, phi)


def v1_fwhm(kappa):
    """Full width at half maximum (deg) of the axial von Mises bump."""
    return np.rad2deg(np.arccos(1 - np.log(2) / kappa))


def v1_rates(theta, pref, kappa, amp=20.0, base=1.0):
    return base + amp * vm_axial(theta[:, None] - pref[None, :], kappa)


class Network:
    def __init__(self, n_v1=120, n_ips=80, kappa_v1=3.0, s_value=2.0,
                 kappa_w=3.0, ips_peak=30.0, ips_base=1.0, cardinal=0.0,
                 normalise=False):
        self.normalise = normalise
        self.v1_pref = v1_preferred(n_v1, cardinal)
        self.kappa_v1, self.kappa_w, self.s_value = kappa_v1, kappa_w, s_value
        self.ips_base = ips_base
        self.v_pref = np.linspace(2, 42, n_ips)                 # model A labels
        self.t_pref = (np.arange(n_ips) + 0.5) * 180 / n_ips    # model B labels
        self.gain = {}
        for model in MODELS:   # strongest unit across both mappings -> ips_peak
            peak = max(self._drive(model, m, THETA).max() for m in MAPPINGS)
            self.gain[model] = (ips_peak - ips_base) / peak

    def weights(self, model, mapping):
        """V1 -> IPS weights (n_v1, n_ips). The only mapping-dependent part."""
        if model == 'value':
            fv = mapping_fn(mapping)[0](self.v1_pref)
            w = np.exp(-(fv[:, None] - self.v_pref[None, :]) ** 2
                       / (2 * self.s_value ** 2))
            # normalise: every IPS unit receives the same total weight, so the
            # mapping can change only its width, not its gain
            return w / w.sum(0, keepdims=True) if self.normalise else w
        return vm_axial(self.v1_pref[:, None] - self.t_pref[None, :], self.kappa_w)

    def _drive(self, model, mapping, theta):
        r = v1_rates(theta, self.v1_pref, self.kappa_v1)
        return r @ self.weights(model, mapping) / len(self.v1_pref)

    def ips_rates(self, model, mapping, theta=THETA):
        return self.ips_base + self.gain[model] * self._drive(model, mapping, theta)


def control_rates(v, v_pref, width_sd, peak=30.0, base=1.0):
    return base + (peak - base) * np.exp(-(v[:, None] - v_pref[None, :]) ** 2
                                         / (2 * width_sd ** 2))


# --- Analyses -------------------------------------------------------------
def fisher_value(rates, v):
    """Poisson Fisher information about value: sum_j h'_j(v)^2 / h_j(v)."""
    dh = np.gradient(rates, v, axis=0)
    return (dh ** 2 / rates).sum(1)


def loglog_slope(x, y):
    return np.polyfit(np.log(x), np.log(y), 1)[0]


def gauss(v, base, amp, mu, sd):
    return base + amp * np.exp(-(v - mu) ** 2 / (2 * sd ** 2))


def fit_gauss(v, y):
    """Gaussian-in-value 'pRF' fit (preferred value bounded to 2-42 CHF)."""
    i = np.argmax(y)
    p0 = [y.min(), np.ptp(y) + 1e-6, np.clip(v[i], 2, 42), 5.0]
    try:
        p, _ = curve_fit(gauss, v, y, p0=p0, maxfev=20000,
                         bounds=([-np.inf, 0, 2, 0.5], [np.inf, np.inf, 42, 60]))
    except RuntimeError:
        return np.full(4, np.nan), np.nan
    r2 = 1 - ((y - gauss(v, *p)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return p, r2


def unit_summary(net, model, mapping):
    """Per unit: preferred value (argmax), value FWHM, peak rate above base,
    and R2 of a Gaussian-in-value fit (how bell-shaped the tuning is)."""
    v = mapping_fn(mapping)[0](THETA)
    h = net.ips_rates(model, mapping)
    rows = []
    for j in range(h.shape[1]):
        y = h[:, j] - net.ips_base
        i0 = np.argmax(y)
        hi_ = y >= y.max() / 2
        # contiguous half-max region around the peak (ignores the side lobe
        # that axial wrap-around gives units near 2 / 42 CHF)
        lo_i = i0 - np.argmin(hi_[i0::-1]) + 1 if not hi_[:i0 + 1].all() else 0
        up_i = i0 + np.argmin(hi_[i0:]) - 1 if not hi_[i0:].all() else len(y) - 1
        above = v[lo_i:up_i + 1]
        _, r2 = fit_gauss(v, h[:, j])
        rows.append(dict(model=model, mapping=mapping, unit=j,
                         pref=v[np.argmax(y)], fwhm=above.max() - above.min(),
                         peak=y.max(), r2=r2))
    return pd.DataFrame(rows)


def decode_rmse(rates_stim, rates_grid, v_grid, v_stim, n_trials, rng):
    """Posterior-mean decoder from Poisson IPS counts. The grid is uniform in
    orientation, i.e. the prior is the session's own value distribution."""
    log_r = np.log(rates_grid)
    out = []
    for s in range(len(v_stim)):
        k = rng.poisson(rates_stim[s], size=(n_trials, rates_stim.shape[1]))
        ll = k @ log_r.T - rates_grid.sum(1)[None, :]
        post = np.exp(ll - ll.max(1, keepdims=True))
        est = (post @ v_grid) / post.sum(1)
        out.append(np.sqrt(np.mean((est - v_stim[s]) ** 2)))
    return np.array(out)


def voxel_weights(net, model, centre, spread):
    labels = net.v_pref if model == 'value' else net.t_pref
    rng_ = 40 if model == 'value' else 180
    d = labels - centre
    if model == 'orient':
        d = (d + 90) % 180 - 90
    w = np.exp(-d ** 2 / (2 * (spread * rng_) ** 2))
    return w / w.sum()


def simulate_voxels(net, model, n_vox, k_units, spread, rng, noise_sd=0.5):
    """Voxel = K units drawn around a random centre on the model's own map
    (value for A, orientation for B; SD = spread x range). The same units
    form the voxel in both sessions; a Gaussian in value is fit per mapping
    to the 23 stimulus responses (+ Gaussian measurement noise)."""
    lo, hi = (2, 42) if model == 'value' else (0, 180)
    resp = {m: net.ips_rates(model, m, STIM_ORI) for m in MAPPINGS}
    out = []
    for vox in range(n_vox):
        c = rng.uniform(lo, hi)
        idx = rng.choice(len(net.v_pref), size=k_units,
                         p=voxel_weights(net, model, c, spread))
        row = dict(model=model, voxel=vox, centre=c)
        for m in MAPPINGS:
            y = resp[m][:, idx].mean(1) + rng.normal(0, noise_sd, len(STIM_ORI))
            p, r2 = fit_gauss(mapping_fn(m)[0](STIM_ORI), y)
            row.update({f'mu_{m}': p[2], f'amp_{m}': p[1], f'r2_{m}': r2})
        out.append(row)
    return pd.DataFrame(out)


def voxel_locus(net, model, spread, n=181):
    """Noise-free voxel preferred values along the map (infinite K)."""
    lo, hi = (2, 42) if model == 'value' else (0, 180)
    resp = {m: net.ips_rates(model, m, STIM_ORI) for m in MAPPINGS}
    rows = []
    for c in np.linspace(lo, hi, n):
        w = voxel_weights(net, model, c, spread)
        rows.append({m: fit_gauss(mapping_fn(m)[0](STIM_ORI), resp[m] @ w)[0][2]
                     for m in MAPPINGS})
    return pd.DataFrame(rows)


def orientation_index(vox):
    """0 = preferred value identical across mappings (value code),
    1 = on the raw same-orientation curve. Only voxels where the two
    references differ by > 4 CHF."""
    f_c, _ = mapping_fn('cdf')
    f_i, _ = mapping_fn('inverse_cdf')
    so = f_i(value_to_theta('cdf', vox.mu_cdf))
    sep = np.abs(so - vox.mu_cdf) > 4
    return ((vox.mu_inverse_cdf - vox.mu_cdf) / (so - vox.mu_cdf))[sep]


def run(args):
    rng = np.random.default_rng(args.seed)
    net = Network(n_v1=args.n_v1, n_ips=args.n_ips, kappa_v1=args.kappa_v1,
                  kappa_w=args.kappa_v1, s_value=args.s_value,
                  cardinal=args.cardinal, normalise=args.normalise)
    units = pd.concat([unit_summary(net, mo, m) for mo in MODELS for m in MAPPINGS])
    # control: same number of units, uniform preferred values, one width
    # (median FWHM of the value-wired units), same peak rate
    ctrl_sd = units.query("model == 'value'")['fwhm'].median() / 2.355
    res = dict(net=net, units=units, ctrl_sd=ctrl_sd, fi={}, rmse={},
               slopes=[], args=args)

    for m in MAPPINGS:
        f, _ = mapping_fn(m)
        v, v_stim = f(THETA), f(STIM_ORI)
        p, p_stim = value_density(m), value_density(m, STIM_ORI)
        grid = {mo: net.ips_rates(mo, m) for mo in MODELS}
        stim = {mo: net.ips_rates(mo, m, STIM_ORI) for mo in MODELS}
        grid['control'] = control_rates(v, net.v_pref, ctrl_sd)
        stim['control'] = control_rates(v_stim, net.v_pref, ctrl_sd)
        grid['v1'] = v1_rates(THETA, net.v1_pref, net.kappa_v1)
        for mo, h in grid.items():
            fi = fisher_value(h, v)
            res['fi'][(mo, m)] = fi
            row = dict(model=mo, mapping=m,
                       fi_slope=loglog_slope(p[INNER], fi[INNER]))
            if mo != 'v1':
                rm = decode_rmse(stim[mo] * args.dur, grid[mo] * args.dur, v,
                                 v_stim, args.n_trials, rng)
                res['rmse'][(mo, m)] = rm
                row['rmse_slope'] = loglog_slope(p_stim[1:-1], rm[1:-1])
            res['slopes'].append(row)

    res['voxels'] = pd.concat([simulate_voxels(net, mo, args.n_vox, args.k_units,
                                               args.spread, rng) for mo in MODELS])
    res['locus'] = {mo: voxel_locus(net, mo, args.spread) for mo in MODELS}

    # how the voxel-level prediction depends on orientation bandwidth
    sweep = []
    for k in args.kappa_sweep:
        n2 = Network(n_v1=args.n_v1, n_ips=args.n_ips, kappa_v1=k, kappa_w=k,
                     s_value=args.s_value, cardinal=args.cardinal,
                     normalise=args.normalise)
        fwhm = v1_fwhm(k)
        for mo in MODELS:
            vx = simulate_voxels(n2, mo, args.n_vox, args.k_units, args.spread,
                                 np.random.default_rng(args.seed))
            vx = vx[(vx.r2_cdf > .5) & (vx.r2_inverse_cdf > .5)]
            oi = orientation_index(vx)
            sweep.append(dict(kappa=k, v1_fwhm=fwhm, model=mo,
                              oi_median=np.median(oi),
                              oi_lo=np.percentile(oi, 25),
                              oi_hi=np.percentile(oi, 75)))
    res['sweep'] = pd.DataFrame(sweep)
    return res


def report(res):
    pd.set_option('display.width', 120)
    sl = pd.DataFrame(res['slopes'])
    print('\nlog-log slope of Fisher information on p(v) (efficient coding: 2)')
    print(sl.pivot(index='model', columns='mapping', values='fi_slope').round(2))
    print('\nlog-log slope of decoding RMSE on p(v) (efficient coding: -1)')
    print(sl.dropna().pivot(index='model', columns='mapping',
                            values='rmse_slope').round(2))
    u = res['units']
    print(f"\nControl width SD = {res['ctrl_sd']:.2f} CHF")
    print('\nUnits: median R2 of Gaussian-in-value fit; corr with p(v) at pref')
    for (mo, m), g in u.groupby(['model', 'mapping']):
        p = value_density(m, value_to_theta(m, g.pref))
        ok = (g.pref > 3) & (g.pref < 41)
        print(f'  {mo:6s} {m:12s} R2={g.r2.median():.3f} (min {g.r2.min():.2f})  '
              f'r(log peak, log p)={np.corrcoef(np.log(g.peak[ok]), np.log(p[ok]))[0, 1]:+.2f}  '
              f'r(log FWHM, log 1/p)={np.corrcoef(np.log(g.fwhm[ok]), -np.log(p[ok]))[0, 1]:+.2f}  '
              f'peak range {g.peak.min():.1f}-{g.peak.max():.1f}  '
              f'FWHM range {g.fwhm.min():.1f}-{g.fwhm.max():.1f}')
    vox = res['voxels']
    print('\nVoxels (both fits R2 > .5): preferred value cdf vs inverse_cdf')
    for mo, g in vox.groupby('model'):
        g = g[(g.r2_cdf > .5) & (g.r2_inverse_cdf > .5)]
        oi = orientation_index(g)
        lr = np.log(g.amp_cdf / g.amp_inverse_cdf)
        lp = np.log(value_density('cdf', value_to_theta('cdf', g.mu_cdf))
                    / value_density('inverse_cdf',
                                    value_to_theta('inverse_cdf', g.mu_inverse_cdf)))
        print(f'  {mo:6s} n={len(g)}  median |mu_cdf - mu_inv| = '
              f'{np.median(np.abs(g.mu_cdf - g.mu_inverse_cdf)):.1f} CHF; '
              f'orientation index median {np.median(oi):.2f} '
              f'(IQR {np.percentile(oi, 25):.2f}-{np.percentile(oi, 75):.2f}, n={len(oi)}); '
              f'r(log amp ratio, log p ratio) = {np.corrcoef(lr, lp)[0, 1]:+.2f}, '
              f'slope {np.polyfit(lp, lr, 1)[0]:.2f}')
    print('\nOrientation index vs V1 bandwidth')
    print(res['sweep'].round(2).to_string(index=False))


# --- Figure ---------------------------------------------------------------
COLORS = {'cdf': '#3B5BA5', 'inverse_cdf': '#C44E52'}
NAMES = {'cdf': 'Cdf', 'inverse_cdf': 'Inverse cdf'}
MODEL_COL = {'value': '0.15', 'orient': '#8172B2', 'control': '0.6'}
MODEL_NAME = {'value': 'Value-wired', 'orient': 'Orientation-wired',
              'control': 'Fixed value code'}


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 8,
        'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'mathtext.fontset': 'stixsans',
        'axes.linewidth': 0.8, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.labelpad': 3,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'lines.linewidth': 1.2, 'lines.markersize': 4,
        'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02})


def scale_to(ref, target):
    """Scale `ref` by a constant to best match `target` in log space."""
    return ref * np.exp(np.mean(np.log(target) - np.log(ref)))


def plot(res, out):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    import seaborn as sns
    style()
    net = res['net']
    fig, axes = plt.subplots(3, 3, figsize=(7.25, 6.9))
    ax = axes.ravel()
    vt = [2, 12, 22, 32, 42]

    def letter(a, s):
        a.text(-0.24, 1.06, s, transform=a.transAxes, fontsize=8,
               fontweight='bold', va='bottom', ha='right')

    # a: mappings
    a = ax[0]
    for m in MAPPINGS:
        f, _ = mapping_fn(m)
        a.plot(THETA, f(THETA), color=COLORS[m], lw=1)
        a.plot(STIM_ORI, f(STIM_ORI), 'o', ms=2.4, color=COLORS[m])
    a.text(60, 7, NAMES['cdf'], color=COLORS['cdf'], fontsize=7, ha='left')
    a.text(52, 26, NAMES['inverse_cdf'], color=COLORS['inverse_cdf'],
           fontsize=7, ha='right')
    a.set(xticks=[0, 45, 90, 135, 180], yticks=vt, xlim=(0, 180), ylim=(2, 42),
          xlabel='Orientation (deg)', ylabel='Value (CHF)')

    # b: value density
    a = ax[1]
    for m in MAPPINGS:
        f, _ = mapping_fn(m)
        a.plot(f(THETA), value_density(m), color=COLORS[m], lw=1.2)
        a.plot(f(STIM_ORI), np.full(23, -0.006), '|', ms=4, mew=0.8,
               color=COLORS[m], clip_on=False)
    a.set(xticks=vt, xlim=(2, 42), ylim=(0, None), xlabel='Value (CHF)',
          ylabel='p(value)')

    # c: value-wired tuning curves
    a = ax[2]
    show = [np.argmin(np.abs(net.v_pref - x)) for x in (6, 14, 22, 30, 38)]
    for m in MAPPINGS:
        f, _ = mapping_fn(m)
        h = net.ips_rates('value', m)
        for j in show:
            a.plot(f(THETA), h[:, j], color=COLORS[m], lw=1)
    for j in show:
        a.plot(net.v_pref[j], -1.2, 'v', ms=3, color='0.3', clip_on=False)
    a.set(xticks=vt, xlim=(2, 42), ylim=(0, 40), yticks=[0, 10, 20, 30, 40],
          xlabel='Value (CHF)', ylabel='Rate (spikes/s)')
    a.text(0.03, 0.97, 'Value-wired IPS units', transform=a.transAxes,
           fontsize=6.5, va='top', color='0.2')
    key(a, [('Preferred value, fixed', '0.3', 'marker', {'marker': 'v', 'ms': 3})],
        x=0.03, y=0.86)

    # d: Fisher information
    a = ax[3]
    for m in MAPPINGS:
        v = mapping_fn(m)[0](THETA)[INNER]
        p2 = value_density(m)[INNER] ** 2
        fi = res['fi'][('value', m)][INNER]
        a.plot(v, scale_to(p2, fi), color=COLORS[m], lw=2.6, alpha=.25,
               solid_capstyle='butt')
        a.plot(v, fi, color=COLORS[m], lw=1)
        a.plot(v, res['fi'][('control', m)][INNER], color=MODEL_COL['control'],
               lw=1, ls='--')
    a.set_yscale('log')
    a.set(xticks=vt, xlim=(2, 42), xlabel='Value (CHF)',
          ylabel='Fisher information (1/CHF²)')
    sl = pd.DataFrame(res['slopes']).set_index(['model', 'mapping'])['fi_slope']
    a.set_ylim(0.5, 5e3)
    key(a, [(f"Value-wired, $\\propto$ p$^{{{sl[('value', 'cdf')]:.1f}}}$ / "
             f"p$^{{{sl[('value', 'inverse_cdf')]:.1f}}}$", '0.2', 'line', {}),
            (r'$\propto$ p(value)$^2$ = orientation-wired', '0.2', 'bar',
             {'alpha': .25}),
            ('Fixed value code', MODEL_COL['control'], 'line', {'ls': '--'})],
        x=0.03, y=0.97, dy=0.085)

    # e: decoding error
    a = ax[4]
    for m in MAPPINGS:
        f, _ = mapping_fn(m)
        vs = f(STIM_ORI)
        rm = res['rmse'][('value', m)]
        inv_p = 1 / value_density(m, THETA[INNER])
        a.plot(f(THETA[INNER]), scale_to(inv_p, np.interp(f(THETA[INNER]), vs, rm)),
               color=COLORS[m], lw=2.6, alpha=.25, solid_capstyle='butt')
        a.plot(vs[1:-1], rm[1:-1], 'o', ms=2.8, color=COLORS[m])
        a.plot(vs[1:-1], res['rmse'][('control', m)][1:-1], 'o', ms=2.6,
               mfc='none', mew=0.7, color=COLORS[m], alpha=.6)
    a.set_yscale('log')
    a.set_ylim(0.15, 6)
    a.set_yticks([0.2, 0.5, 1, 2])
    a.yaxis.set_major_formatter(mt.FuncFormatter(lambda x, _: f'{x:g}'))
    a.yaxis.set_minor_formatter(mt.NullFormatter())
    a.set(xticks=vt, xlim=(2, 42), xlabel='Value (CHF)',
          ylabel='Decoding error (CHF)')
    key(a, [('Value-wired', '0.2', 'marker', {}),
            ('Fixed value code', '0.2', 'marker', {'mfc': 'none', 'mew': .7}),
            (r'$\propto$ 1/p(value)', '0.2', 'bar', {'alpha': .25})],
        x=0.03, y=0.97, dy=0.085)

    # f: gain vs preferred value
    u = res['units']
    a = ax[5]
    for m in MAPPINGS:
        g = u.query("model == 'value' and mapping == @m")
        f, _ = mapping_fn(m)
        p = value_density(m)
        a.plot(f(THETA), scale_to(p, np.interp(f(THETA), g.pref, g.peak)),
               color=COLORS[m], lw=2.6, alpha=.25, solid_capstyle='butt')
        a.plot(g.pref, g.peak, 'o', ms=2.2, color=COLORS[m])
    a.set(xticks=vt, xlim=(2, 42), ylim=(0, 50), yticks=[0, 10, 20, 30, 40, 50],
          xlabel='Preferred value (CHF)', ylabel='Peak rate (spikes/s)')
    key(a, [('Value-wired units', '0.2', 'marker', {'ms': 2.2}),
            (r'$\propto$ p(value)', '0.2', 'bar', {'alpha': .25})],
        x=0.03, y=0.97, dy=0.085)

    # g: FWHM vs preferred value
    a = ax[6]
    for m in MAPPINGS:
        g = u.query("model == 'value' and mapping == @m and 3 < pref < 41")
        a.plot(g.pref, g.fwhm, 'o', ms=2.2, color=COLORS[m])
        go = u.query("model == 'orient' and mapping == @m and 3 < pref < 41")
        a.plot(go.pref, go.fwhm, 'x', ms=2.6, mew=0.7, color=COLORS[m], alpha=.7)
    a.axhline(res['ctrl_sd'] * 2.355, color=MODEL_COL['control'], lw=0.8, ls='--')
    a.set(xticks=vt, xlim=(2, 42), ylim=(0, 40), yticks=[0, 10, 20, 30, 40],
          xlabel='Preferred value (CHF)', ylabel='Tuning width, FWHM (CHF)')
    key(a, [('Value-wired', '0.2', 'marker', {'ms': 2.2}),
            ('Orientation-wired', '0.2', 'marker', {'marker': 'x', 'mew': .7,
                                                    'ms': 2.6, 'mfc': '0.2'}),
            ('Fixed value code', MODEL_COL['control'], 'line', {'ls': '--', 'lw': .8})],
        x=0.03, y=0.97, dy=0.085)

    # h: voxel preferred values
    a = ax[7]
    vox = res['voxels']
    vox = vox[(vox.r2_cdf > .5) & (vox.r2_inverse_cdf > .5)]
    f_c, _ = mapping_fn('cdf')
    f_i, _ = mapping_fn('inverse_cdf')
    a.plot([2, 42], [2, 42], color='0.55', lw=0.8, zorder=0)
    a.plot(f_c(THETA), f_i(THETA), color='0.55', lw=0.8, ls=':', zorder=0)
    for mo, mk in (('value', 'o'), ('orient', 'o')):
        g = vox[vox.model == mo]
        a.plot(g.mu_cdf, g.mu_inverse_cdf, mk, ms=2, alpha=.5,
               color=MODEL_COL[mo], mew=0)
        lc = res['locus'][mo]
        a.plot(lc['cdf'], lc['inverse_cdf'], color=MODEL_COL[mo], lw=1)
    a.set(xticks=vt, yticks=vt, xlim=(2, 42), ylim=(2, 42), aspect='equal',
          xlabel='Preferred value, cdf (CHF)',
          ylabel='Preferred value, inverse cdf (CHF)')
    key(a, [('Value-wired voxels', MODEL_COL['value'], 'marker', {'ms': 2.6}),
            ('Orientation-wired voxels', MODEL_COL['orient'], 'marker', {'ms': 2.6}),
            ('Same orientation', '0.55', 'line', {'ls': ':', 'lw': .8}),
            ('Value tuning, y = x', '0.55', 'line', {'lw': .8})],
        x=0.36, y=0.05, dy=0.075, seg=0.06, up=True)

    # i: orientation index vs V1 bandwidth
    a = ax[8]
    sw = res['sweep']
    for mo in MODELS:
        g = sw[sw.model == mo]
        a.fill_between(g.v1_fwhm, g.oi_lo, g.oi_hi, color=MODEL_COL[mo],
                       alpha=.2, lw=0)
        a.plot(g.v1_fwhm, g.oi_median, 'o-', ms=2.6, color=MODEL_COL[mo])
    a.axhline(1, color='0.55', lw=0.8, ls=':', zorder=0)
    a.axhline(0, color='0.55', lw=0.8, zorder=0)
    a.axvline(v1_fwhm(res['args'].kappa_v1), color='0.8', lw=0.6, ls='--', zorder=0)
    a.set_xscale('log')
    a.set_xticks([10, 20, 40])
    a.xaxis.set_major_formatter(mt.FuncFormatter(lambda x, _: f'{x:g}'))
    a.xaxis.set_minor_formatter(mt.NullFormatter())
    a.set(xlim=(8, 70), ylim=(-0.3, 1.1), yticks=[0, 0.5, 1],
          xlabel='V1 bandwidth, FWHM (deg)', ylabel='Orientation index')
    a.text(9, -0.2, 'Value-wired', color=MODEL_COL['value'], fontsize=6.5,
           va='center')
    a.text(21, 0.85, 'Orientation-wired', color=MODEL_COL['orient'],
           fontsize=6.5, va='center')
    a.text(68, 1.04, 'Same orientation', ha='right', va='bottom', fontsize=6.5,
           color='0.45')
    a.text(v1_fwhm(res['args'].kappa_v1) * 0.96, -0.27, 'Panels c-h', ha='right',
           va='bottom', fontsize=6, color='0.5')

    for a, s in zip(ax, 'abcdefghi'):
        letter(a, s)
    sns.despine(fig=fig, offset=4, trim=False)
    fig.tight_layout(w_pad=1.6, h_pad=1.4)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'), dpi=200)
    print(f'\nwrote {out} and {out.with_suffix(".png")}')


def wiring_panel(ax, net, mapping, model='value', hi_theta=37.5, s=2.0):
    """Schematic of the V1 -> IPS wiring. V1 units (oriented bars, top row)
    at the 23 stimulus orientations; line opacity = weight; one V1 unit
    highlighted. Value-wired: IPS units (dots, bottom row) sit at their fixed
    preferred value and the lines follow the session's mapping. Orientation-
    wired: IPS units sit under the V1 units they pool, never rewired, and
    their value is read off through each session's mapping (labels below)."""
    f, _ = mapping_fn(mapping)
    col = COLORS[mapping] if model == 'value' else MODEL_COL['orient']
    if model == 'value':
        ips_v = np.arange(2, 42.1, 2.0)
        ips_x = (ips_v - 2) / 40
    else:
        ips_t = STIM_ORI
        ips_x = ips_t / 180
        ips_v = mapping_fn('cdf')[0](ips_t)
    for th in STIM_ORI:
        if model == 'value':
            w = np.exp(-(f(th) - ips_v) ** 2 / (2 * s ** 2))
        else:   # local pooling; drawn without the 0/180 wrap-around
            w = vm_axial(th - ips_t, 12.0) * (np.abs(th - ips_t) < 45)
        hi = np.isclose(th, hi_theta)
        for xj, wj in zip(ips_x, w):
            if wj < 0.05:
                continue
            ax.plot([th / 180, xj], [1, 0], color=col if hi else '0.35',
                    lw=1.4 if hi else 0.5, alpha=min(1, wj) * (1 if hi else 0.45),
                    solid_capstyle='round', zorder=3 if hi else 1)
    for th in STIM_ORI:   # gabor-like bar
        ax.plot(th / 180, 1.0, marker=(2, 0, th), ms=6, mew=1.1,
                color=col if np.isclose(th, hi_theta) else '0.15', zorder=4)
    for th in (0, 45, 90, 135, 180):
        ax.text(th / 180, 1.13, f'{th}°', ha='center', va='bottom', fontsize=6,
                color='0.35')
    ax.text(1.0, 1.27, 'Preferred orientation', ha='right', va='bottom',
            fontsize=6, color='0.35')
    ax.text(-0.04, 1.0, 'V1', ha='right', va='center', fontsize=7, fontweight='bold')
    ax.text(-0.04, 0.0, 'IPS', ha='right', va='center', fontsize=7,
            fontweight='bold')
    if model == 'value':
        ax.scatter(ips_x, np.zeros_like(ips_x), s=14, c=ips_v, cmap='YlGnBu',
                   vmin=-8, vmax=46, edgecolor='0.15', lw=0.4, zorder=4)
        for v in (2, 12, 22, 32, 42):
            ax.text((v - 2) / 40, -0.13, f'{v}', ha='center', va='top',
                    fontsize=6, color='0.35')
        ax.text(1.0, -0.29, 'Preferred value (CHF)', ha='right', va='top',
                fontsize=6, color='0.35')
        ax.set(xlim=(-0.03, 1.03), ylim=(-0.32, 1.3))
    else:
        ax.scatter(ips_x, np.zeros_like(ips_x), s=14, c=MODEL_COL['orient'],
                   edgecolor='0.15', lw=0.4, zorder=4)
        show = np.arange(1, len(ips_t), 4)
        for row, m in enumerate(MAPPINGS):
            fv = mapping_fn(m)[0](ips_t)
            for k in show:
                ax.text(ips_x[k], -0.13 - 0.13 * row, f'{fv[k]:.0f}', ha='center',
                        va='top', fontsize=6, color=COLORS[m])
            ax.text(-0.04, -0.13 - 0.13 * row, f'{NAMES[m]} (CHF)',
                    ha='right', va='top', fontsize=6, color=COLORS[m])
        ax.text(0.5, -0.45, 'Each IPS unit\'s preferred value is whatever its '
                'orientation is worth in the session', ha='center', va='top',
                fontsize=6.5, color='0.3')
        ax.set(xlim=(-0.03, 1.03), ylim=(-0.55, 1.3))
    ax.axis('off')


def plot_narrative(res, out):
    """Two-page story figure: what the network is, how rewiring shifts it,
    what that buys (efficient coding), and why orientation coding is the
    confound."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    style()
    net = res['net']
    vt = [2, 12, 22, 32, 42]
    hi_theta = 37.5
    demo_v = (16.0, 26.0)

    def letter(a, s, x=-0.16):
        a.text(x, 1.04, s, transform=a.transAxes, fontsize=8, fontweight='bold',
               va='bottom', ha='right')

    def headline(fig, spec, n, text, dy=0.025):
        y = spec.get_position(fig).y1 + dy
        fig.text(0.01, y, f'{n}  {text}', fontsize=8.5, fontweight='bold',
                 color='0.1', va='bottom')

    def despine(a):
        for sp in ('left', 'bottom'):
            a.spines[sp].set_position(('outward', 3))

    figs = []
    # ---------------- page 1: the network and how it shifts -----------------
    fig = plt.figure(figsize=(7.25, 9.4))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1.15, 1, 1], hspace=0.75,
                          wspace=0.32, left=0.08, right=0.98, top=0.95,
                          bottom=0.05)

    headline(fig, gs[0, :], '1', 'The same orientations are worth different '
             'amounts in the two sessions')
    a = fig.add_subplot(gs[0, 0])
    for m in MAPPINGS:
        f, _ = mapping_fn(m)
        a.plot(THETA, f(THETA), color=COLORS[m], lw=1.2)
        a.plot(STIM_ORI, f(STIM_ORI), 'o', ms=2.4, color=COLORS[m])
    a.text(150, 27, 'Cdf session', color=COLORS['cdf'], ha='center')
    a.text(122, 18, 'Inverse-cdf\nsession', color=COLORS['inverse_cdf'],
           ha='left', va='top')
    for m in MAPPINGS:
        v = float(mapping_fn(m)[0](hi_theta))
        a.plot([hi_theta, hi_theta, 0], [2, v, v], color=COLORS[m], lw=0.6,
               ls=(0, (2, 1.5)), zorder=0)
    a.text(hi_theta + 3, 3.2, f'{hi_theta:g}°', fontsize=6.5, color='0.3')
    a.set(xticks=[0, 45, 90, 135, 180], yticks=vt, xlim=(0, 180), ylim=(2, 42),
          xlabel='Orientation (deg)', ylabel='Value (CHF)')
    despine(a)
    letter(a, 'a')

    a = fig.add_subplot(gs[0, 1])
    for m in MAPPINGS:
        f, _ = mapping_fn(m)
        a.fill_between(f(THETA), value_density(m), color=COLORS[m], alpha=.15,
                       lw=0)
        a.plot(f(THETA), value_density(m), color=COLORS[m], lw=1.2)
    a.annotate('Values cluster at\n12 and 32 CHF', xy=(13.2, 0.07),
               xytext=(15.5, 0.13), color=COLORS['cdf'], fontsize=6.5,
               arrowprops=dict(arrowstyle='-|>', color=COLORS['cdf'], lw=0.8,
                               mutation_scale=7, shrinkB=3))
    a.annotate('Values cluster at\n2, 22 and 42 CHF', xy=(23.2, 0.07),
               xytext=(26, 0.16),
               color=COLORS['inverse_cdf'], fontsize=6.5,
               arrowprops=dict(arrowstyle='-|>', color=COLORS['inverse_cdf'],
                               lw=0.8, mutation_scale=7, shrinkB=3))
    a.set(xticks=vt, xlim=(2, 42), ylim=(0, 0.2), yticks=[0, 0.1, 0.2],
          xlabel='Value (CHF)', ylabel='How often a value is shown\n(density)')
    despine(a)
    letter(a, 'b')

    headline(fig, gs[1, :], '2', 'Model: V1 codes orientation; each V1 unit '
             'projects to the IPS units that code its current value', dy=0.05)
    for j, m in enumerate(MAPPINGS):
        a = fig.add_subplot(gs[1, j])
        wiring_panel(a, net, m, hi_theta=hi_theta)
        a.set_title(f'{NAMES[m]} session', color=COLORS[m], fontsize=7.5,
                    pad=14, loc='center')
        letter(a, 'cd'[j], x=0.0)
    a = fig.axes[-1]
    a.annotate('Many V1 units converge\non the frequent values', xy=(0.5, 0.06),
               xytext=(0.73, 0.55), fontsize=6.5, color='0.2', ha='left',
               arrowprops=dict(arrowstyle='-|>', color='0.2', lw=0.9,
                               mutation_scale=7, shrinkB=4,
                               connectionstyle='angle3,angleA=180,angleB=70'))
    fig.text(0.5, gs[1, :].get_position(fig).y0 - 0.035, f'Only the wiring differs between sessions: the '
             f'{hi_theta:g}° unit (coloured) feeds the ~'
             f'{float(mapping_fn("cdf")[0](hi_theta)):.0f}-CHF IPS units in '
             f'one session and the ~'
             f'{float(mapping_fn("inverse_cdf")[0](hi_theta)):.0f}-CHF units '
             'in the other', ha='center', fontsize=6.5, color='0.35')

    headline(fig, gs[2, :], '3', 'Each IPS unit keeps its preferred value, so its '
             'preferred orientation moves')
    a_o = fig.add_subplot(gs[2, 0])
    a_v = fig.add_subplot(gs[2, 1])
    for m in MAPPINGS:
        f, _ = mapping_fn(m)
        h = net.ips_rates('value', m)
        for k, v0 in enumerate(demo_v):
            j = int(np.argmin(np.abs(net.v_pref - v0)))
            ls = '-' if k == 0 else (0, (3, 1.5))
            a_o.plot(THETA, h[:, j], color=COLORS[m], lw=1.1, ls=ls)
            a_v.plot(f(THETA), h[:, j], color=COLORS[m], lw=1.1, ls=ls)
            i_pk = np.argmax(h[:, j])
            a_o.plot(THETA[i_pk], h[i_pk, j] + 1.8, 'v', ms=3, color=COLORS[m])
            a_v.plot(f(THETA[i_pk]), h[i_pk, j] + 1.8, 'v', ms=3, color=COLORS[m])
    for k, v0 in enumerate(demo_v):
        a_v.axvline(v0, color='0.6', lw=0.6, ls='--', zorder=0)
        a_v.text(v0 + (-0.8 if k == 0 else 0.8), 27, f'{v0:g}-CHF unit',
                 ha='right' if k == 0 else 'left', va='bottom', fontsize=6.5,
                 color='0.3')
        j = int(np.argmin(np.abs(net.v_pref - v0)))
        pk = {m: THETA[np.argmax(net.ips_rates('value', m)[:, j])] for m in MAPPINGS}
        a_o.annotate('', xy=(pk['inverse_cdf'], 30), xytext=(pk['cdf'], 30),
                     arrowprops=dict(arrowstyle='-|>', color='0.3', lw=0.9,
                                     mutation_scale=7, shrinkA=0, shrinkB=0))
        a_o.text(np.mean(list(pk.values())), 24, f'{v0:g}-CHF unit',
                 ha='center', va='bottom', fontsize=6.5, color='0.3')
    a_o.text(75, 36, 'Preferred orientation moves', ha='center', va='bottom',
             fontsize=6.5, color='0.3')
    a_o.set(xticks=[0, 45, 90, 135, 180], xlim=(0, 180), ylim=(0, 42),
            yticks=[0, 20, 40], xlabel='Orientation (deg)',
            ylabel='IPS rate (spikes/s)')
    a_v.set(xticks=vt, xlim=(2, 42), ylim=(0, 42), yticks=[0, 20, 40],
            xlabel='Value (CHF)', ylabel='IPS rate (spikes/s)')
    a_v.text(22, 36, 'Preferred value stays (roughly)', ha='center', va='bottom',
             fontsize=6.5, color='0.3')
    key(a_o, [('Cdf session', COLORS['cdf'], 'line', {}),
              ('Inverse-cdf session', COLORS['inverse_cdf'], 'line', {})],
        x=0.72, y=0.97, dy=0.1)
    for a, s in ((a_o, 'e'), (a_v, 'f')):
        despine(a)
        letter(a, s)

    headline(fig, gs[3, :], '4', 'Consequence: the IPS code is most precise where '
             'values are most frequent (efficient coding)')
    a = fig.add_subplot(gs[3, 0])
    for m in MAPPINGS:
        f, _ = mapping_fn(m)
        w_tot = net.weights('value', m).sum(0)
        p = value_density(m)
        a.plot(f(THETA), scale_to(p, np.interp(f(THETA), net.v_pref, w_tot)),
               color=COLORS[m], lw=2.8, alpha=.25, solid_capstyle='butt')
        a.plot(net.v_pref, w_tot, 'o', ms=2, color=COLORS[m])
    key(a, [('Total V1 input per IPS unit', '0.25', 'marker', {'ms': 2.4}),
            (r'$\propto$ How often the value is shown', '0.25', 'bar', {'alpha': .25})],
        x=0.03, y=0.97, dy=0.1)
    a.set(xticks=vt, xlim=(2, 42), ylim=(0, 65), xlabel='Preferred value (CHF)',
          ylabel='V1 input (summed weight)')
    despine(a)
    letter(a, 'g')

    a = fig.add_subplot(gs[3, 1])
    sl = pd.DataFrame(res['slopes']).set_index(['model', 'mapping'])['fi_slope']
    for m in MAPPINGS:
        v = mapping_fn(m)[0](THETA)[INNER]
        fi = res['fi'][('value', m)][INNER]
        a.plot(v, scale_to(value_density(m)[INNER] ** 2, fi), color=COLORS[m],
               lw=2.8, alpha=.25, solid_capstyle='butt')
        a.plot(v, fi, color=COLORS[m], lw=1.1)
        a.plot(v, res['fi'][('control', m)][INNER], color=MODEL_COL['control'],
               lw=1, ls=(0, (3, 1.5)))
    a.set_yscale('log')
    a.set(xticks=vt, xlim=(2, 42), ylim=(0.5, 5e3), xlabel='Value (CHF)',
          ylabel='Precision\n(Fisher information)')
    key(a, [('Rewired IPS', '0.25', 'line', {}),
            (r'$\propto$ p(value)², the efficient code', '0.25', 'bar', {'alpha': .25}),
            ('Fixed value code, no rewiring', MODEL_COL['control'], 'line',
             {'ls': (0, (3, 1.5))})],
        x=0.03, y=0.97, dy=0.1)
    despine(a)
    letter(a, 'h')
    figs.append(fig)

    # ---------------- page 2: the confound and the test ----------------------
    fig = plt.figure(figsize=(7.25, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.7, wspace=0.45,
                          left=0.08, right=0.98, top=0.88, bottom=0.08)
    headline(fig, gs[0, :], '5', 'The catch: a plain orientation code, never '
             'rewired, is just as efficient', dy=0.06)
    a = fig.add_subplot(gs[0, 0:2])
    wiring_panel(a, net, 'cdf', model='orient', hi_theta=hi_theta)
    a.set_title('Orientation-wired IPS: same wiring in both sessions',
                color=MODEL_COL['orient'], fontsize=7.5, pad=14)
    letter(a, 'i', x=0.0)
    a = fig.add_subplot(gs[0, 2])
    for m in MAPPINGS:
        v = mapping_fn(m)[0](THETA)[INNER]
        a.plot(v, res['fi'][('orient', m)][INNER] /
               res['fi'][('orient', m)][INNER].mean(), color=COLORS[m], lw=1.1)
        a.plot(v, res['fi'][('value', m)][INNER] /
               res['fi'][('value', m)][INNER].mean(), color=COLORS[m], lw=1.1,
               ls=(0, (3, 1.5)), alpha=.7)
    a.set_yscale('log')
    a.set(xticks=vt, xlim=(2, 42), xlabel='Value (CHF)',
          ylabel='Precision (relative)')
    key(a, [('Orientation-wired', '0.25', 'line', {}),
            ('Value-wired', '0.25', 'line', {'ls': (0, (3, 1.5))})],
        x=0.03, y=0.97, dy=0.1)
    a.set_ylim(0.03, 40)
    despine(a)
    letter(a, 'j')

    headline(fig, gs[1, :], '6', 'What does differ: does the preferred value stay '
             'put across sessions?', dy=0.06)
    f_c, _ = mapping_fn('cdf')
    f_i, _ = mapping_fn('inverse_cdf')
    u = res['units'].pivot_table(index=['model', 'unit'], columns='mapping',
                                 values='pref').reset_index()
    a = fig.add_subplot(gs[1, 0])
    a.plot([2, 42], [2, 42], color='0.6', lw=0.8, zorder=0)
    a.plot(f_c(THETA), f_i(THETA), color='0.6', lw=0.8, ls=':', zorder=0)
    for mo in MODELS:
        g = u[u.model == mo]
        a.plot(g['cdf'], g['inverse_cdf'], 'o', ms=2.4, mew=0,
               color=MODEL_COL[mo])
    a.text(33, 37.5, 'Same value', rotation=45, ha='center', fontsize=6.5,
           color='0.4')
    a.text(33, 16, 'Same\norientation', ha='left', fontsize=6.5, color='0.4')
    a.set(xticks=vt, yticks=vt, xlim=(2, 42), ylim=(2, 42), aspect='equal',
          xlabel='Preferred value, cdf (CHF)',
          ylabel='Preferred value,\ninverse cdf (CHF)')
    a.set_title('Single units', fontsize=7, loc='left')
    key(a, [('Value-wired', MODEL_COL['value'], 'marker', {}),
            ('Orientation-wired', MODEL_COL['orient'], 'marker', {})],
        x=0.03, y=0.97, dy=0.1)
    despine(a)
    letter(a, 'k')

    a = fig.add_subplot(gs[1, 1])
    vox = res['voxels']
    vox = vox[(vox.r2_cdf > .5) & (vox.r2_inverse_cdf > .5)]
    a.plot([2, 42], [2, 42], color='0.6', lw=0.8, zorder=0)
    a.plot(f_c(THETA), f_i(THETA), color='0.6', lw=0.8, ls=':', zorder=0)
    for mo in MODELS:
        g = vox[vox.model == mo]
        a.plot(g.mu_cdf, g.mu_inverse_cdf, 'o', ms=1.8, alpha=.45, mew=0,
               color=MODEL_COL[mo])
    a.set(xticks=vt, yticks=vt, xlim=(2, 42), ylim=(2, 42), aspect='equal',
          xlabel='Preferred value, cdf (CHF)', ylabel='')
    a.set_title('Voxels (30 units, 40° V1 tuning)', fontsize=7, loc='left')
    a.text(0.97, 0.03, 'Both hug the diagonal', transform=a.transAxes,
           ha='right', fontsize=6.5, color='0.3')
    despine(a)
    letter(a, 'l')

    a = fig.add_subplot(gs[1, 2])
    sw = res['sweep']
    for mo in MODELS:
        g = sw[sw.model == mo]
        a.fill_between(g.v1_fwhm, g.oi_lo, g.oi_hi, color=MODEL_COL[mo],
                       alpha=.2, lw=0)
        a.plot(g.v1_fwhm, g.oi_median, 'o-', ms=2.4, color=MODEL_COL[mo])
    a.axhline(1, color='0.6', lw=0.8, ls=':', zorder=0)
    a.axhline(0, color='0.6', lw=0.8, zorder=0)
    a.set_xscale('log')
    a.set_xticks([10, 20, 40])
    import matplotlib.ticker as mt
    a.xaxis.set_major_formatter(mt.FuncFormatter(lambda x, _: f'{x:g}'))
    a.xaxis.set_minor_formatter(mt.NullFormatter())
    a.set(xlim=(8, 70), ylim=(-0.3, 1.15), yticks=[0, 0.5, 1],
          xlabel='V1 tuning width (FWHM, deg)',
          ylabel='Voxel shift')
    a.text(70, 1.03, 'Same orientation', ha='right', va='bottom', fontsize=6.5,
           color='0.45')
    a.text(8.5, 0.05, 'Same value', ha='left', va='bottom', fontsize=6.5,
           color='0.45')
    a.set_title('Only sharp tuning separates them', fontsize=7, loc='left')
    a.text(9, 0.87, 'Orientation-wired', color=MODEL_COL['orient'], fontsize=6.5)
    a.text(9, -0.2, 'Value-wired', color=MODEL_COL['value'], fontsize=6.5)
    despine(a)
    letter(a, 'm')
    figs.append(fig)

    out = Path(out)
    with PdfPages(out) as pdf:
        for i, f in enumerate(figs):
            pdf.savefig(f)
            f.savefig(out.with_name(f'{out.stem}_p{i + 1}.png'), dpi=200)
    print(f'\nwrote {out} (+ _p1/_p2.png)')


def key(ax, entries, x=.04, y=.96, dy=.085, seg=.07, fs=6.5, up=False):
    """Inline glyph legend (house style); up=True stacks upward from y."""
    import matplotlib.pyplot as plt
    tf = ax.transAxes
    for i, (lab, col, kind, o) in enumerate(entries):
        yy = y + i * dy if up else y - i * dy
        if kind == 'marker':
            ax.plot(x + seg / 2, yy, o.get('marker', 'o'), transform=tf,
                    ms=o.get('ms', 2.8), color=col, mfc=o.get('mfc', col),
                    mew=o.get('mew', 0), clip_on=False)
        else:
            ax.plot([x, x + seg], [yy, yy], transform=tf, color=col,
                    ls=o.get('ls', '-'),
                    lw=o.get('lw', 2.6 if kind == 'bar' else 1.0),
                    alpha=o.get('alpha', 1), solid_capstyle='butt', clip_on=False)
        ax.text(x + seg + .025, yy, lab, transform=tf, color='0.2', fontsize=fs,
                va='center')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-v1', type=int, default=120)
    ap.add_argument('--n-ips', type=int, default=80)
    ap.add_argument('--kappa-v1', type=float, default=3.0,
                    help='V1 von Mises concentration (3 -> FWHM ~40 deg); also '
                         'used for the orientation-wired pooling kernel')
    ap.add_argument('--s-value', type=float, default=2.0,
                    help='Width (SD, CHF) of the value-wired V1->IPS kernel')
    ap.add_argument('--cardinal', type=float, default=0.0,
                    help='V1 cardinal over-representation (0 uniform, 0.5 moderate)')
    ap.add_argument('--normalise', action='store_true',
                    help='Value-wired: equal total input weight per IPS unit')
    ap.add_argument('--dur', type=float, default=0.25,
                    help='Counting window (s) for the decoder')
    ap.add_argument('--n-trials', type=int, default=2000)
    ap.add_argument('--n-vox', type=int, default=300)
    ap.add_argument('--k-units', type=int, default=30)
    ap.add_argument('--spread', type=float, default=0.06,
                    help='Voxel scatter on the IPS map, SD as fraction of range')
    ap.add_argument('--kappa-sweep', type=float, nargs='+',
                    default=[1.5, 3, 6, 12, 25, 50])
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=str(REPO / 'notes/figures/sim_v1_ips_mapping.pdf'))
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()
    res = run(args)
    report(res)
    if not args.no_plot:
        plot_narrative(res, args.out)
        plot(res, str(Path(args.out).with_name(Path(args.out).stem + '_details.pdf')))


if __name__ == '__main__':
    main()
