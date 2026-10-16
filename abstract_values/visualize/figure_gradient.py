"""Paper figure: from perception to representation along the dorsal hierarchy.

Across participants, two links between brain and the two-stage efficient-coding
model:
  perception      orientation-code strength (best von Mises cvR² over the null,
                  ROI mean) vs the model's perceptual precision kappa_r
  representation  value-code strength (best aPRF cvR² over the null) vs the
                  model's value precision, -log sigma_rep
Both are partial correlations controlling for head motion (mean FD) and the
participant's whole-cortex signal of the OTHER code, so neither link can be
carried by general data quality.

a, b  Each ROI painted by its perception / representation link.
c     The two links along the dorsal hierarchy, V1 -> IPS3-5: perception fades,
      representation grows. Bands: 95% participant-bootstrap intervals.
d     The gradient as one number: per bootstrap sample, the slope of
      (representation - perception) across hierarchy rank.

    python -m abstract_values.visualize.figure_gradient [--params TSV]   # pycortex2 env
    -> notes/figures/fig_gradient.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cortex
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abstract_values.utils.data import BIDS_FOLDER
from abstract_values.visualize import qc_flatmaps as qf
from abstract_values.visualize.group_surface_maps import discover_subjects
from abstract_values.visualize.qc_tuning_maps import collect

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "notes" / "figures" / "fig_gradient.pdf"
PARAMS = Path("/data/ds-abstractvalue/derivatives/cogmodels/"
              "efficient_coding_sequential_grid161_hn_total-share_erf_subject_params.tsv")
PIL = {"pil01": "01", "pil02": "02"}
CX_ROI = {"V1": ["V1"], "V2": ["V2"], "V3": ["V3"], "hV4": ["hV4"], "LO": ["LO"],
          "TO": ["TO1", "TO2"], "V3a/b": ["V3a"], "IPS0": ["IPS0"],
          "IPS1–2": ["IPS1", "IPS2"], "IPS3–5": ["IPS3", "IPS4", "IPS5"],
          "M1 L": ["M1_lh"]}
DORSAL = ["V1", "V2", "V3", "V3a/b", "IPS0", "IPS1–2", "IPS3–5"]
# IPS3-5 come straight from the Wang-15 labels (the pycortex overlay has no
# IPS4/5); the numerosity map (NPC) is left out: it is a functional localiser
# that overlaps IPS0-2 rather than a further stage of the hierarchy.
WANG_DIR = Path("/data/ds-abstractvalue/derivatives/surface_masks")
WANG_ONLY = ["IPS3", "IPS4", "IPS5"]


def roi_verts():
    """fsaverage vertex indices (lh then rh) for every name in CX_ROI."""
    import nibabel as nib
    names = [n for n in sum(CX_ROI.values(), []) if n not in WANG_ONLY and n != "M1_lh"]
    verts = cortex.get_roi_verts("fsaverage", names + ["M1"])
    n_lh = cortex.db.get_surf("fsaverage", "flat", "lh")[0].shape[0]
    verts["M1_lh"] = verts["M1"][verts["M1"] < n_lh]       # control: left motor cortex
    for n in WANG_ONLY:
        idx = []
        for h, hh, off in (("L", "lh", 0), ("R", "rh", n_lh)):
            d = nib.load(WANG_DIR / f"desc-{n}_{h}_space-fsaverage_hemi-{hh}.label.gii")
            idx.append(np.flatnonzero(d.darrays[0].data > 0) + off)
        verts[n] = np.concatenate(idx)
    return verts


def exclusive_rois(verts):
    """Disjoint vertex sets: each vertex goes to the EARLIEST region in the
    dorsal order, so atlases that overlap at their borders (pycortex IPS1-3
    vs Wang IPS3) never double-count a vertex."""
    order = DORSAL + [r for r in CX_ROI if r not in DORSAL]
    taken, out = set(), {}
    for r in order:
        vs = set(np.concatenate([verts[n] for n in CX_ROI[r]]).tolist())
        out[r] = np.array(sorted(vs - taken))
        taken |= vs
    return out
PER_C, REP_C = "#1B8A80", "#D9730D"   # teal = perception, amber = representation
N_BOOT = 4000

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def pcorr(x, y, Z):
    Z = np.column_stack([np.ones(len(x)), Z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1]


def data(params):
    deriv = Path(BIDS_FOLDER) / "derivatives"
    d = collect(deriv, discover_subjects(deriv), True)
    subs = [PIL.get(s, s) for s in d["subjects"]]
    par = pd.read_csv(params, sep="\t")
    par.index = par.subject.map(lambda s: f"{int(s):02d}")
    mo = pd.read_csv(REPO / "notes/data/motion_summary.tsv", sep="\t", dtype={"subject": str})
    mo["subject"] = mo.subject.str.replace("sub-", "").replace(PIL)
    fd = mo.groupby("subject").mean_fd.mean()
    keep = [i for i, s in enumerate(subs) if s in par.index and s in fd.index]
    sel = [subs[i] for i in keep]
    ori, val = d["ori_gain"][keep], d["val_gain"][keep]
    verts = exclusive_rois(roi_verts())
    roi = {r: (np.nanmean(ori[:, vx], 1), np.nanmean(val[:, vx], 1))
           for r, vx in verts.items()}
    cov = dict(fd=fd.loc[sel].to_numpy(), g_ori=np.nanmean(ori, 1), g_val=np.nanmean(val, 1),
               lk=np.log(par.loc[sel, "kappa_r"].to_numpy()),
               lp=-np.log(par.loc[sel, "sigma_rep"].to_numpy()))
    return roi, cov, verts, len(sel)


def links(roi, cov, idx=None):
    i = slice(None) if idx is None else idx
    out = {}
    for r, (o, v) in roi.items():
        per = pcorr(cov["lk"][i], o[i], np.column_stack([cov["fd"][i], cov["g_val"][i]]))
        rep = pcorr(cov["lp"][i], v[i], np.column_stack([cov["fd"][i], cov["g_ori"][i]]))
        out[r] = (per, rep)
    return out


def paint(values, verts):
    n = sum(cortex.db.get_surf("fsaverage", "flat", h)[0].shape[0] for h in ("lh", "rh"))
    m = np.full(n, np.nan, np.float32)
    for r, vx in verts.items():
        m[vx] = values[r]
    return m


def brain(ax, values, verts, cmap, lim):
    m = paint(values, verts)
    alpha = np.where(np.isfinite(m), 1.0, 0.0).astype(np.float32)
    qf.draw_flat(ax, qf.blended(np.nan_to_num(m), alpha, -lim, lim, cmap), labels=False)


def main(params=PARAMS, out=OUT):
    roi, cov, verts, n = data(params)
    est = links(roi, cov)
    rng = np.random.default_rng(0)
    boots = [links(roi, cov, rng.integers(0, n, n)) for _ in range(N_BOOT)]
    B = {r: np.array([[b[r][0], b[r][1]] for b in boots]) for r in CX_ROI}
    rank = np.arange(len(DORSAL))
    slopes = np.array([np.polyfit(rank, [b[r][1] - b[r][0] for r in DORSAL], 1)[0]
                       for b in boots])
    slope_est = np.polyfit(rank, [est[r][1] - est[r][0] for r in DORSAL], 1)[0]

    fig = plt.figure(figsize=(7.25, 4.4))
    ax_a = fig.add_axes([0.01, 0.47, 0.47, 0.48])
    ax_b = fig.add_axes([0.52, 0.47, 0.47, 0.48])
    ax_c = fig.add_axes([0.09, 0.11, 0.60, 0.27])
    ax_d = fig.add_axes([0.83, 0.11, 0.07, 0.27])

    lim = 0.5
    cm_per = sns.blend_palette(["#EEF3F2", "#8CC9C0", PER_C, "#0B3D3A"], as_cmap=True)
    cm_rep = sns.blend_palette(["#F7F1EA", "#F2B87A", REP_C, "#6A3205"], as_cmap=True)
    for ax, k, cm, lab, letter, x0 in ((ax_a, 0, cm_per, "Perception", "a", 0.01),
                                       (ax_b, 1, cm_rep, "Representation", "b", 0.52)):
        m = paint({r: e[k] for r, e in est.items()}, verts)
        alpha = np.where(np.isfinite(m), 1.0, 0.0).astype(np.float32)
        qf.draw_flat(ax, qf.blended(np.clip(np.nan_to_num(m), 0, lim), alpha, 0, lim, cm),
                     labels=False)
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        ax.set_xlim(x_lo + 0.20 * (x_hi - x_lo), x_lo + 0.80 * (x_hi - x_lo))
        ax.set_ylim(y_lo + 0.30 * (y_hi - y_lo), y_lo + 0.86 * (y_hi - y_lo))
        col = PER_C if k == 0 else REP_C
        fig.text(x0, 0.955, letter, fontsize=8, fontweight="bold")
        fig.text(x0 + 0.025, 0.955, lab, fontsize=8, color=col, fontweight="bold")
        fig.text(x0 + 0.025 + (0.085 if k == 0 else 0.125), 0.955,
                 "orientation code ~ perceptual precision" if k == 0
                 else "value code ~ value precision", fontsize=7, color="0.3")
        cb = qf.colorbar(fig, [x0 + 0.30, 0.50, 0.14, 0.010], cm, 0, lim, "",
                         ticks=[0, 0.25, 0.5])
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_visible(False)
        fig.text(x0 + 0.37, 0.528, "Partial r", fontsize=6, ha="center", color="0.3")

    # c -- the gradient
    x = rank
    for k, (col, lab) in enumerate(((PER_C, "Perception"), (REP_C, "Representation"))):
        y = np.array([est[r][k] for r in DORSAL])
        lo = np.array([np.quantile(B[r][:, k], 0.025) for r in DORSAL])
        hi = np.array([np.quantile(B[r][:, k], 0.975) for r in DORSAL])
        dx = -0.08 if k == 0 else 0.08
        se = np.array([B[r][:, k].std() for r in DORSAL])
        ax_c.vlines(x + dx, y - se, y + se, color=col, lw=1.1, alpha=0.6)
        ax_c.plot(x + dx, y, "-", color=col, lw=1.8, solid_capstyle="round")
        ax_c.plot(x + dx, y, "o", color=col, ms=4.5, mec="white", mew=0.7)
        ax_c.text(x[-1] + 0.25, y[-1] + (0.06 if k else -0.06), lab, color=col,
                  fontsize=7, va="center", fontweight="bold")
    ax_c.axhline(0, color="0.75", lw=0.6, ls="--", zorder=0)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(DORSAL)
    ax_c.set_xlim(-0.35, len(DORSAL) + 0.9)
    ax_c.set_ylim(-0.5, 0.75)
    ax_c.set_yticks([-0.4, 0, 0.4])
    ax_c.set_ylabel(f"Partial r with model (n = {n})")
    ax_c.set_xlabel("Dorsal hierarchy (V1 → IPS3–5)", color="0.35", labelpad=3)
    sns.despine(ax=ax_c, offset=4, trim=True)
    fig.text(0.01, 0.40, "c", fontsize=8, fontweight="bold")

    # d -- the gradient as one number
    p_one = np.mean(slopes <= 0)
    ci = np.quantile(slopes, [0.025, 0.975])
    ax_d.vlines(0, ci[0], ci[1], color="0.25", lw=1.2)
    ax_d.plot(0, slope_est, "o", ms=6, color="0.15", mec="white", mew=0.7)
    ax_d.axhline(0, color="0.75", lw=0.6, ls="--", zorder=0)
    ax_d.set_xlim(-0.8, 0.8)
    ax_d.set_xticks([])
    ax_d.set_ylim(-0.15, 0.3)
    ax_d.set_yticks([-0.1, 0, 0.1, 0.2])
    ax_d.set_ylabel("Representation − perception,\nchange per step (Δr)")
    ax_d.text(0.22, slope_est, f"P(≤ 0) = {p_one:.2f}", fontsize=6, va="center",
              color="0.3")
    sns.despine(ax=ax_d, offset=4, bottom=True, trim=True)
    fig.text(0.74, 0.40, "d", fontsize=8, fontweight="bold")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".svg"))
    tab = pd.DataFrame({r: dict(perception=est[r][0], representation=est[r][1],
                                per_lo=np.quantile(B[r][:, 0], .025), per_hi=np.quantile(B[r][:, 0], .975),
                                rep_lo=np.quantile(B[r][:, 1], .025), rep_hi=np.quantile(B[r][:, 1], .975))
                        for r in CX_ROI}).T
    tab.to_csv(REPO / "notes/data/gradient_links.tsv", sep="\t", float_format="%.3f")
    print(tab.round(2).to_string())
    print(f"gradient slope {slope_est:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}], P(<=0) = {p_one:.4f}, n = {n}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", default=str(PARAMS))
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()
    main(a.params, a.out)
