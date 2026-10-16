"""One-page summary for collaborators: fixed codes, flexible routing, noise along the hierarchy.

Row 1  Geometry (crossnobis RDMs over the 23 orientations): V1 is a circle,
       parietal cortex (IPS0-5) an ordered line -- and the parietal geometry
       does not re-scale when the mapping changes.
Row 2  Routing: trial-to-trial coupling between parietal voxels and V1
       orientation channels follows the orientation -> value mapping in force
       (stimulus means and the neighbouring trials' orientations, +-1,
       removed), increasingly so along IPS.
Row 3  Individual differences: across participants, perceptual noise of the
       two-stage model predicts how well early visual cortex carries value,
       value noise predicts it in parietal cortex; and people whose parietal
       routing follows the mapping more closely have less value noise.

    python -m abstract_values.visualize.figure_summary      # pycortex2 env
    -> notes/figures/fig_summary.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.manifold import MDS

from abstract_values.connective_fields import decode_mapping as dm
from abstract_values.connective_fields import individual_differences as idf
from abstract_values.connective_fields.figure_connectopy import panel_idea
from abstract_values.geometry.analyze_crossnobis import load as load_rdms
from abstract_values.geometry.figure_circularity import subject_rdms
from abstract_values.visualize import figure_gradient_versions as fv

REPO = Path(__file__).resolve().parents[2]
V1_C, PAR_C = "#1B8A80", "#D9730D"
PER_C, REP_C = fv.PER_C, fv.REP_C
LAG = "lags-1_"                                   # neighbours' orientations (+-1) removed

plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 7, "axes.labelsize": 7.5,
    "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def letter(fig, x, y, s):
    fig.text(x, y, s, fontsize=8.5, fontweight="bold", va="top")


def headline(fig, y, s):
    fig.text(0.01, y, s, fontsize=8.2, fontweight="bold", color="0.15", va="top")


# ── row 1: geometry ──────────────────────────────────────────────────────────
def panel_ends(ax, rd, regions=(("V1", None), ("Parietal (IPS0–5)", None))):
    """regions: (key in rd, colour or None for the default)."""
    ends = []
    for name, col in regions:
        col = col or (V1_C if name == "V1" else PAR_C)
        per = np.array([(R[0, 1:] + R[-1, :-1][::-1]) / 2 for R in rd[name][0].values()])
        x = np.arange(1, per.shape[1] + 1) * 7.5
        m, se = per.mean(0), per.std(0) / np.sqrt(len(per))
        ax.fill_between(x, m - se, m + se, color=col, alpha=0.15, lw=0)
        ax.plot(x, m, "-", color=col, lw=1.5)
        ends.append([m[-1], name.split(" (")[0], col, x[-1]])
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):                 # keep end labels apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.24)
    for y, lab, col, xe in ends:
        ax.text(xe + 4, y, lab, color=col, fontsize=6.6, fontweight="bold", va="center")
    ax.axvline(90, color="0.8", lw=0.6, ls=(0, (3, 2)))
    ax.text(88, 0.05, "Opposite on\nthe circle", fontsize=6, color="0.5", va="bottom",
            ha="right")
    ax.set_xticks([15, 45, 90, 135, 165])
    ax.set_xlim(0, 200)
    ax.set_xlabel("Distance from an end orientation (7.5° or 172.5°)")
    ax.set_ylabel("Neural distance\n(crossnobis, relative)")
    sns.despine(ax=ax, offset=3, trim=True)


def panel_mds(ax, R, oris, col, title):
    tw = plt.get_cmap("twilight")
    Y = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
            normalized_stress="auto").fit_transform(np.sqrt(np.clip(R, 0, None)))
    Y -= Y.mean(0)
    ax.plot(Y[:, 0], Y[:, 1], "-", color="0.65", lw=0.8, zorder=1)
    ax.plot([Y[-1, 0], Y[0, 0]], [Y[-1, 1], Y[0, 1]], color="0.25", lw=1.0,
            ls=(0, (2, 1.5)), zorder=1)
    ax.scatter(Y[:, 0], Y[:, 1], c=[tw(o / 180) for o in oris], s=16, edgecolor="white",
               lw=0.4, zorder=2)
    for i in (0, len(oris) - 1):
        ax.text(Y[i, 0], Y[i, 1] + 0.07 * np.ptp(Y[:, 1]), f"{oris[i]:g}°", fontsize=5.8,
                ha="center", color="0.3")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", color=col, fontweight="bold", fontsize=7)


def panel_stability(ax):
    """Mapping-change slope (does the geometry re-scale with the session's values?),
    observed vs what a code that re-scales would give (injected into the real data)."""
    import glob
    pw = pd.concat(pd.read_csv(f, sep="\t", dtype={"subject": str}) for f in
                   glob.glob("/data/ds-abstractvalue/derivatives/geometry/power/sub-*/*.tsv"))
    pw = pw.drop_duplicates(["subject", "roi", "code", "frac"])
    rois = [("BensonV1", "V1"), ("IPS0", "IPS0"), ("IPS12", "IPS1–2"), ("IPS345", "IPS3–5")]
    for i, (key, lab) in enumerate(rois):
        g = pw[(pw.roi == key) & (pw.code == "value")]
        obs = g[g.frac == 0].slope
        inj = g[g.frac == 0.25].slope
        ax.vlines(i - 0.1, obs.mean() - obs.sem(), obs.mean() + obs.sem(), color="0.2", lw=1.1)
        ax.plot(i - 0.1, obs.mean(), "o", color="0.2", ms=4.5, mec="white", mew=0.7)
        ax.vlines(i + 0.1, inj.mean() - inj.sem(), inj.mean() + inj.sem(), color="0.6", lw=1.1)
        ax.plot(i + 0.1, inj.mean(), "o", mfc="white", mec="0.55", ms=4.5, mew=1.0)
    ax.axhline(0, color="0.75", lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels([r[1] for r in rois], rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Geometry follows\nthe mapping (slope)")
    for k, (lab, kw) in enumerate((("Observed", dict(color="0.2")),
                                   ("If the code re-scaled (25% injected)",
                                    dict(mfc="white", mec="0.55", mew=1.0)))):
        ax.plot(0.05, 1.16 - 0.13 * k, "o", transform=ax.transAxes, ms=4, clip_on=False, **kw)
        ax.text(0.1, 1.16 - 0.13 * k, lab, transform=ax.transAxes, fontsize=5.6, va="center",
                color="0.3")
    sns.despine(ax=ax, offset=3, trim=True)


# ── row 2: routing ───────────────────────────────────────────────────────────
def panel_aligned(ax, d, maps):
    oris = np.sort(d.orientation.unique())
    for ref, col, lab in (("this", PAR_C, "Mapping in force"), ("other", "0.55", "Other mapping")):
        rel, P = dm.aligned(d, maps, oris, ref)
        m, se = P.mean(0).to_numpy(), P.sem(0).to_numpy()
        ax.fill_between(rel, m - se, m + se, color=col, alpha=0.2, lw=0)
        ax.plot(rel, m, color=col, lw=1.6)
        i = int(np.argmax(m))
        xy = (rel[i] + 12, m[i]) if ref == "this" else (-88, 0.6 * m[i])
        ax.text(*xy, lab, color=col, fontsize=6.8, fontweight="bold", va="center")
    ax.axvline(0, color="0.75", lw=0.6, ls=(0, (3, 2)))
    ax.axhline(0, color="0.85", lw=0.6)
    ax.set_xlim(-90, 90)
    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_xlabel("V1 channel relative to the orientation\nworth the voxel's preferred value (°)")
    ax.set_ylabel("Parietal–V1 coupling\n(trial-to-trial)")
    sns.despine(ax=ax, offset=3, trim=True)


def panel_maps(ax):
    regions = [("IPS0", "IPS0"), ("IPS1–2", "IPS12"), ("IPS3–5", "IPS345")]
    rows = []
    for lab, tag in regions:
        x = idf.scores(f"coupling_{LAG}iem-k24-kappa16_target-{tag}").dropna()
        rows.append(dict(roi=lab, m=x.mean(), se=x.sem(), p=stats.ttest_1samp(x, 0)[1],
                         vals=x.values))
    r = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    for i, row in r.iterrows():
        jit = rng.uniform(-0.15, 0.15, len(row.vals))
        ax.scatter(i + jit, row.vals, s=6, color="0.75", lw=0, zorder=1)
    ax.vlines(r.index, r.m - r.se, r.m + r.se, color=PAR_C, lw=1.3, zorder=3)
    ax.plot(r.index, r.m, "-o", color=PAR_C, ms=5, mec="white", mew=0.8, lw=1.4, zorder=4)
    for i, row in r.iterrows():
        ptxt = "p < .001" if row.p < .001 else f"p = {row.p:.3f}".replace("0.", ".")
        ax.text(i, row.m + row.se + 0.004, ptxt, fontsize=6, color="0.25", ha="center",
                va="bottom", bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.8), zorder=5)
    ax.axhline(0, color="0.6", lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(r.index)
    ax.set_xticklabels(r.roi)
    ax.set_xlim(-0.4, 2.7)
    ax.set_ylabel("Coupling follows the mapping\n(observed − shuffled)")
    sns.despine(ax=ax, offset=3, trim=True)
    return r


# ── row 3: individual differences ────────────────────────────────────────────
def panel_gradient_est(ax):
    tab = fv.subject_table(fv.PARAMS)
    m = pd.read_csv(REPO / "notes/data/decoding_multiverse_measures.tsv", sep="\t",
                    dtype={"subject": str})
    m = m[(m.noise == "spherical") & (m.prior == "flat")]
    acc = m.pivot(index="subject", columns="roi", values="accuracy")
    regs = ["perc_prec", "val_prec"]
    subs = [s for s in tab.index if s in acc.index]
    est = fv.profile(acc, tab, regs)
    B = fv.boot(lambda i: fv.profile(acc, tab, regs, i), len(subs))
    g = fv.gradient_stat(est)[0]
    Bg = np.array([fv.gradient_stat(b)[0] for b in B])
    x = np.arange(len(fv.DORSAL))
    for k, (col, lab) in enumerate(((PER_C, "Perceptual\nprecision"),
                                    (REP_C, "Value\nprecision"))):
        y = np.array([est[r][k] for r in fv.DORSAL])
        se = np.array([np.std([b[r][k] for b in B]) for r in fv.DORSAL])
        ax.fill_between(x, y - se, y + se, color=col, alpha=0.16, lw=0)
        ax.plot(x, y, "-o", color=col, lw=1.6, ms=4, mec="white", mew=0.7)
        ax.text(x[-1] + 0.2, y[-1], lab, color=col, fontsize=6.8, fontweight="bold",
                va="center")
    ax.axhline(0, color="0.7", lw=0.6, ls=(0, (3, 2)))
    ax.set_xlim(-0.3, len(x) + 1.3)
    ax.set_ylabel("Precision predicts\nvalue decoding (std. β)")
    p = np.mean(Bg <= 0)
    ax.set_title(f"Interaction (value − perceptual, parietal − early):\n"
                 f"{g:+.2f} ± {Bg.std():.2f}, P = {p:.3f}", loc="left", fontsize=6,
                 color="0.3")
    sns.despine(ax=ax, offset=3, trim=True)
    ax.set_xticks(x)
    ax.set_xticklabels(fv.DORSAL, rotation=35, ha="right", rotation_mode="anchor")
    return (g, Bg.std(), p), est


def panel_indiv(ax):
    S = idf.scores(f"coupling_{LAG}iem-k24-kappa16_target-IPS0IPS12IPS345")
    b = idf.behaviour(list(S.index))
    j = pd.concat([b.value_precision, S.rename("cf")], axis=1).dropna()
    rho, p = stats.spearmanr(j.iloc[:, 0], j.cf)
    r, _ = stats.pearsonr(j.iloc[:, 0], j.cf)
    sns.regplot(x=j.iloc[:, 0], y=j.cf, ax=ax, color=PAR_C, ci=68,
                scatter_kws=dict(s=13, edgecolor="white", linewidths=0.5),
                line_kws=dict(lw=1.3))
    ax.axhline(0, color="0.8", lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.set_xlabel("Value precision\n(model, −log σ_rep)")
    ax.set_ylabel("Routing follows\nthe mapping")
    ax.set_title(f"ρ = {rho:+.2f}, p = {p:.3f}\nr = {r:+.2f}, n = {len(j)}", loc="left",
                 fontsize=6, color="0.3")
    sns.despine(ax=ax, offset=3, trim=True)
    return rho, p, len(j)


ROI_COL = {"V1": "#1B8A80", "V2": "#55A79D", "V3": "#93C7BF", "V3a/b": "#C9DEDA",
           "IPS0": "#F5C08A", "IPS1–2": "#E08A2E", "IPS3–5": "#9C4A05"}


def _flat(ax, vtx, crop=True):
    import cortex
    cortex.quickflat.make_figure(vtx, fig=ax, with_curvature=False, with_colorbar=False,
                                 with_rois=True, with_labels=False, with_sulci=False,
                                 roi_list=["V1", "V2", "V3", "V3a", "IPS0", "IPS1", "IPS2"],
                                 linewidth=1.5, linecolor=(1, 1, 1, 0.9), height=1600)
    ax.set_axis_off()
    if crop:
        ax.set_xlim(-185, 128)
        ax.set_ylim(-88, 68)
        ax.set_aspect("equal", adjustable="box")


def _curv_rgb():
    import cortex
    n = sum(cortex.db.get_surf("fsaverage", "flat", h)[0].shape[0] for h in ("lh", "rh"))
    z = np.zeros(n, np.float32)
    c = cortex.Vertex(z, "fsaverage").blend_curvature(z, brightness=0.82, contrast=0.12)
    return np.stack([c.red.data, c.green.data, c.blue.data], 1) / 255.0


def _rgb_vertex(rgb):
    import cortex
    u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return cortex.VertexRGB(u8[:, 0], u8[:, 1], u8[:, 2], "fsaverage")


def roi_brain(ax, verts):
    """The regions, coloured from early visual (teal) to parietal (amber)."""
    import cortex
    from matplotlib.colors import to_rgb
    rgb = _curv_rgb()
    for r, c in ROI_COL.items():
        rgb[verts[r]] = to_rgb(c)
    _flat(ax, _rgb_vertex(rgb))
    pts = cortex.db.get_surf("fsaverage", "flat", merge=True, nudge=True)[0][:, :2]
    n_lh = cortex.db.get_surf("fsaverage", "flat", "lh")[0].shape[0]
    cents = {r: np.median(pts[verts[r][verts[r] < n_lh]], 0) for r in ROI_COL}
    order = sorted(cents, key=lambda t: cents[t][1])
    ys = np.array([cents[t][1] for t in order], float)
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 15)
    ys -= max(0, ys[-1] - 62)
    x_txt = -118
    for t, y in zip(order, ys):
        cx, cy = cents[t]
        ax.plot([x_txt + 3, cx], [y, cy], color="0.35", lw=0.5)
        ax.text(x_txt, y, t, ha="right", va="center", fontsize=6.2,
                color=ROI_COL[t] if t not in ("V3a/b",) else "0.4", fontweight="bold")


def brain3d(ax, verts, azim=-115, elev=18):
    """Inflated left hemisphere seen from behind and the side, regions coloured:
    tells a reader where the flattened patch sits on a real brain."""
    import cortex
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    pts, polys = cortex.db.get_surf("fsaverage", "inflated", "lh")
    curv = cortex.db.get_surfinfo("fsaverage", "curvature").data[:len(pts)]
    col = np.where(curv[:, None] > 0, 0.64, 0.80) * np.ones(3)
    for r, c in ROI_COL.items():
        col[verts[r][verts[r] < len(pts)]] = to_rgb(c)
    ax.add_collection3d(Poly3DCollection(pts[polys], facecolors=col[polys].mean(1),
                                         edgecolor="none", linewidth=0))
    for i, lim in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        lim(pts[:, i].min(), pts[:, i].max())
    ax.set_box_aspect(np.ptp(pts, 0))
    ax.view_init(elev=elev, azim=azim)
    ax.axis("off")


def value_brain(ax, verts, vals, col, lim):
    """ROI-level values painted in one stage colour (sequential, 0 .. lim)."""
    from matplotlib.colors import to_rgb
    cm = fv.stage_cmap(col)
    rgb = _curv_rgb()
    for r, v in vals.items():
        if np.isfinite(v):
            rgb[verts[r]] = cm(np.clip(v, 0, lim) / lim)[:3]
    _flat(ax, _rgb_vertex(rgb))
    return cm


def small_cbar(fig, rect, cm, lim, label, fmt="{:g}"):
    cax = fig.add_axes(rect)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, lim), cmap=cm), cax=cax,
                      orientation="horizontal")
    cb.set_ticks([0, lim])
    cb.set_ticklabels(["≤ 0", fmt.format(lim)])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    cb.set_label(label, fontsize=5.8, labelpad=1)


def body(fig, y, s, x=0.01, w=None):
    fig.text(x, y, s, fontsize=6.5, color="0.3", va="top", linespacing=1.35)


MAP_COL = {"cdf": "#5B4A9E", "inverse_cdf": "#B23A6E"}
PPC = REPO / "notes/data/cogmodel_sweep_cache/ppc_77f1ddc958.tsv"
PPC_MODEL = "Cat · LT"                       # two-stage model with the cardinal category gate
CAT_PARAMS = Path("/data/ds-abstractvalue/derivatives/cogmodels/"
                  "efficient_coding_categorical_grid161_trunc_hn_total-share_erf_subject_params.tsv")


def panel_ppc(ax, stat, ylab):
    d = pd.read_csv(PPC, sep="\t")
    d = d[d.model == PPC_MODEL]
    for m in ("cdf", "inverse_cdf"):
        sim = d[(d.kind == "simulated") & (d.mapping == m)]
        band = sim.groupby("orientation")[stat].quantile([0.025, 0.975]).unstack()
        ax.fill_between(band.index, band[0.025], band[0.975], color=MAP_COL[m], alpha=0.22,
                        lw=0)
        ob = d[(d.kind == "observed") & (d.mapping == m)].set_index("orientation")[stat]
        ax.plot(ob.index, ob.values, "o", ms=2.6, color=MAP_COL[m], mec="white", mew=0.3)
    if stat == "bias":
        ax.axhline(0, color="0.7", lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks([0, 90, 180])
    ax.set_xlabel("Orientation (°)")
    ax.set_ylabel(ylab)
    sns.despine(ax=ax, offset=3, trim=True)


_PPC_F = REPO / "notes/data/ppc_subjects_categorical.tsv"
SUBJ_PPC = pd.read_csv(_PPC_F, sep="\t") if _PPC_F.exists() else None
EXAMPLES = [(27, "Precise bidder"), (4, "Mostly perceptual noise"), (26, "Mostly value noise")]
EX_MARK = {27: "s", 4: "^", 26: "D"}


def panel_example(axb, axs, df, sub, title, par, show_y):
    """One participant's bid error and bid SD per orientation, both mappings."""
    g = df[df.subject == sub].dropna(subset=["response"])
    for m in ("cdf", "inverse_cdf"):
        h = g[g.mapping == m]
        st = h.assign(err=h.response - h.value).groupby("orientation").agg(
            bias=("err", "mean"), sd=("response", "std"))
        ppc = SUBJ_PPC
        for ax, col in ((axb, "bias"), (axs, "sd")):
            if ppc is not None:
                q = ppc[(ppc.subject == sub) & (ppc.mapping == m)].sort_values("orientation")
                ax.fill_between(q.orientation, q[f"{col}_0.1"], q[f"{col}_0.9"],
                                color=MAP_COL[m], alpha=0.12, lw=0)
                ax.fill_between(q.orientation, q[f"{col}_0.25"], q[f"{col}_0.75"],
                                color=MAP_COL[m], alpha=0.25, lw=0)
            ax.plot(st.index, st[col], "o-", color=MAP_COL[m], ms=2.4, lw=0.7)
    r = par.set_index("subject").loc[sub]
    perc = r.total_noise * np.sqrt(r.perceptual_share)
    axb.set_title(f"{title} (sub-{sub:02d})", loc="left", fontsize=6.4, fontweight="bold",
                  color="0.2")
    axb.text(0.99, 1.0, f"perceptual {perc:.2f}, value σ_rep {r.sigma_rep:.2f} CHF",
             transform=axb.transAxes, fontsize=5.5, va="bottom", ha="right", color="0.4")
    axb.plot(0.97, 0.12, EX_MARK[sub], transform=axb.transAxes, ms=4.5, mfc="white",
             mec="0.2", mew=0.9)
    axb.axhline(0, color="0.75", lw=0.6, ls=(0, (3, 2)))
    axb.set_ylim(-4, 4)
    axs.set_ylim(0, 4.5)
    for ax in (axb, axs):
        ax.set_xticks([0, 90, 180])
        sns.despine(ax=ax, offset=2, trim=True)
    axb.set_xticklabels([])
    axs.set_xlabel("Orientation (°)")
    if show_y:
        axb.set_ylabel("Bid − value\n(CHF)")
        axs.set_ylabel("Bid SD\n(CHF)")


def panel_noise(ax):
    par = pd.read_csv(CAT_PARAMS, sep="\t")
    perc = par.total_noise * np.sqrt(par.perceptual_share)
    ax.scatter(perc, par.sigma_rep, s=12, color="0.25", edgecolor="white", linewidths=0.4)
    for sub, _ in EXAMPLES:
        i = par.index[par.subject == sub][0]
        ax.plot(perc[i], max(par.sigma_rep[i], 0.012), EX_MARK[sub], ms=6, mfc="white",
                mec="0.1", mew=1.0, zorder=5)
    lim = [0.01, 5]
    ax.plot(lim, lim, color="0.75", lw=0.7, ls=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.15, 4)
    ax.set_ylim(0.01, 5)
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel("Perceptual noise (CHF)")
    ax.set_ylabel("Value noise σ_rep (CHF)")
    ax.text(0.22, 3.6, "More value noise", fontsize=5.8, color="0.5", va="top")
    ax.text(3.6, 0.015, "More perceptual\nnoise", fontsize=5.8, color="0.5", ha="right")
    sns.despine(ax=ax, offset=3)


ORI_CMAP = sns.blend_palette(["#CFE8E4", "#4FA89F", "#1B8A80", "#0B4A44"], n_colors=256,
                             as_cmap=True)


def panel_retuning(ax):
    d = pd.read_csv(REPO / "notes/data/retuning_summary.tsv", sep="\t", dtype={"subject": str})
    fams = [("aPRF", "#B35A00", "aPRF (value)"), ("wPRF", "#E8A060", "wPRF (value)"),
            ("von Mises", V1_C, "von Mises (orientation)")]
    x = np.arange(len(fv.DORSAL))
    lines = []
    for k, (f, col, lab) in enumerate(fams):
        g = d[d.family == f].groupby("roi").delta
        m, se = (g.mean().reindex(fv.DORSAL) * 100).to_numpy(), \
            (g.sem().reindex(fv.DORSAL) * 100).to_numpy()
        dx = (k - 1) * 0.2
        ax.vlines(x + dx, m - se, m + se, color=col, lw=1.1)
        ax.plot(x + dx, m, "o", color=col, ms=4, mec="white", mew=0.6)
        ax.text(1.02, 0.88 - 0.16 * k, lab, transform=ax.transAxes, color=col, fontsize=6.2,
                va="center", fontweight="bold")
        n = d[d.family == f].subject.nunique()
        lines.append(f"{lab}: {np.nanmin(m):+.2f} to {np.nanmax(m):+.2f}% (n = {n})")
    ax.axhline(0, color="0.6", lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks(x)
    ax.set_xticklabels(fv.DORSAL)
    ax.set_ylabel("cvR², shift − fixed\n(percentage points)")
    sns.despine(ax=ax, offset=3, trim=True)
    return ("Mean ± SE over participants.\n"
            "aPRF: retuning hurts in every region\n(−0.5 to −0.8 points).\n"
            "wPRF and von Mises: small gains\n(≤ 0.75 points), mostly within 1 SE;\n"
            "largest in IPS1–2 (wPRF +0.74 ± 0.34).\nn = 29–32 with both model variants.")


def gabor_icon(ax, x, y, theta, r, col):
    """A small grating icon: oriented stripes in a circle (theta in degrees, 0 = vertical
    stripes rotated clockwise as in the task display)."""
    from matplotlib.patches import Circle
    circ = Circle((x, y), r, fc="white", ec=col, lw=0.8, zorder=3)
    ax.add_patch(circ)
    t = np.deg2rad(theta)
    d = np.array([np.sin(t), np.cos(t)])          # stripe direction
    n = np.array([np.cos(t), -np.sin(t)])         # across stripes
    for k in (-0.5, 0, 0.5):
        c = np.array([x, y]) + k * r * n
        half = np.sqrt(max(r ** 2 - (k * r) ** 2, 0)) * 0.95
        a, b = c - half * d, c + half * d
        ln, = ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=1.1, zorder=4,
                      solid_capstyle="butt")
        ln.set_clip_path(circ)


def explain_schematic(fig, top, height, maps):
    """Two sessions side by side: one IPS voxel tuned to 16 CHF and the V1 orientation
    channels it should couple to; line weight = predicted coupling f_v(m(theta))."""
    tw = plt.get_cmap("twilight")
    oris = np.arange(7.5, 180, 15.0)
    v0, width = 16.0, 4.0
    for k, (m, lab) in enumerate((("cdf", "Session with the CDF mapping"),
                                  ("inverse_cdf", "Session with the inverse-CDF mapping"))):
        ax = fig.add_axes([0.02 + 0.33 * k, top - height, 0.31, height])
        ax.set_xlim(-14, 190)
        ax.set_ylim(-26, 122)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        val = np.interp(oris, maps[m].index, maps[m].values)
        w = np.exp(-0.5 * ((val - v0) / width) ** 2)
        vx, vy = 90, 98
        for o, vv, wi in zip(oris, val, w):
            ax.plot([vx, o], [vy - 8, 7], color=MAP_COL[m], lw=0.3 + 3.2 * wi,
                    alpha=0.2 + 0.8 * wi, zorder=1, solid_capstyle="round")
            gabor_icon(ax, o, 0.0, o, 6.0, tw(o / 180))
            ax.text(o, -16, f"{vv:.0f}", fontsize=5.0, ha="center", color="0.4")
        ax.text(-9, -16, "CHF", fontsize=5.0, ha="right", color="0.4")
        ax.add_patch(plt.Circle((vx, vy), 8, fc=ROI_COL["IPS1–2"], ec="white", lw=0.8,
                                zorder=3))
        ax.text(vx + 12, vy, "IPS voxel tuned\nto 16 CHF", fontsize=6.0, va="center",
                color="0.2")
        ax.text(-9, 0.0, "V1", fontsize=6.3, ha="right", va="center", color=V1_C,
                fontweight="bold")
        th = float(np.interp(v0, maps[m].values, maps[m].index))
        ax.annotate(f"Strongest coupling:\n{th:.0f}° (worth 16 CHF)", xy=(th, 9),
                    xytext=(128, 52), fontsize=5.6, color=MAP_COL[m],
                    arrowprops=dict(arrowstyle="-", color=MAP_COL[m], lw=0.6))
        ax.set_title(lab, fontsize=6.6, color=MAP_COL[m], fontweight="bold", loc="left",
                     pad=0)
    # predicted profiles
    ax = fig.add_axes([0.72, top - height + 0.25 * height, 0.26, 0.62 * height])
    th = np.linspace(0, 180, 361)
    for m in ("cdf", "inverse_cdf"):
        val = np.interp(th, maps[m].index, maps[m].values)
        ax.plot(th, np.exp(-0.5 * ((val - v0) / width) ** 2), color=MAP_COL[m], lw=1.5)
    ax.set_xticks([0, 90, 180])
    ax.set_yticks([])
    ax.set_xlabel("V1 channel (°)", fontsize=6)
    ax.set_title("Predicted coupling profile\nof the same voxel", fontsize=6.3, loc="left",
                 pad=2)
    sns.despine(ax=ax, left=True, offset=2, trim=True)


def tuning_maps(fig, top_in, H, verts):
    """Group preferred orientation (von Mises basis, circular mean) and preferred value
    (aPRF mode, median), smoothed fits, opacity = share of participants with signal."""
    from abstract_values.utils.data import BIDS_FOLDER
    from abstract_values.visualize import qc_tuning_maps as qt
    from abstract_values.visualize import qc_flatmaps as qf
    from abstract_values.visualize.group_surface_maps import discover_subjects
    deriv = Path(BIDS_FOLDER) / "derivatives"
    d = qt.collect(deriv, discover_subjects(deriv), True)
    n = len(d["subjects"])
    theta, prev_o, _ = qt.preferred_orientation(d)
    med, prev_v = qt.preferred_value(d)
    import cortex

    def light(vals, alpha, vmin, vmax, cmap):
        v = cortex.Vertex(np.nan_to_num(vals).astype(np.float32), "fsaverage", vmin=vmin,
                          vmax=vmax, cmap=cmap)
        return v.blend_curvature(np.clip(np.nan_to_num(alpha), 0, 1).astype(np.float32),
                                 brightness=0.82, contrast=0.12)
    vo = light(theta, qt.ramp(prev_o, 0.25, 0.55), 0, 180, qt.ORI_CMAP)
    vv = light(med, qt.ramp(prev_v, 0.25, 0.55), qt.VALUE_MIN, qt.VALUE_MAX, qt.VALUE_CMAP)
    fig.text(0.01, 1 - (top_in + 0.05) / H, "Where voxels prefer which orientation and "
             "which value: group maps of the fitted tuning (smoothed fits, n = "
             f"{n}). Opacity: share of participants whose\nmodel beats the null at that vertex "
             "(fades in from 25% to 55%).", fontsize=6.3, color="0.3", va="top")
    for k, (vtx, title) in enumerate(((vo, "Preferred orientation (von Mises basis)"),
                                      (vv, "Preferred value (aPRF)"))):
        ax = fig.add_axes([0.01 + 0.5 * k, 1 - (top_in + 0.45 + 1.5) / H, 0.4, 1.5 / H])
        _flat(ax, vtx)
        ax.set_title(title, fontsize=6.8, loc="left", fontweight="bold", color="0.2", pad=1)
    qf.orientation_key(fig, [0.41, 1 - (top_in + 1.35) / H, 0.08, 0.7 / H], qt.ORI_CMAP, "")
    cb = qf.colorbar(fig, [0.6, 1 - (top_in + 2.05) / H, 0.2, 0.006], qt.VALUE_CMAP,
                     qt.VALUE_MIN, qt.VALUE_MAX, "Preferred value (CHF)",
                     ticks=[2, 12, 22, 32, 42])
    cb.ax.tick_params(labelsize=5.5)
    cb.set_label("Preferred value (CHF)", fontsize=6)
    fig.text(0.01, 1 - (top_in + 2.5) / H,
             "Orientation preference varies smoothly over early visual cortex (partly a radial "
             "bias that is right-hemisphere-heavy and tracks gaze drift along the bar). "
             "Preferred values\nare spread over the whole range in both early visual and "
             "parietal cortex, without one smooth value map (top left: left M1).",
             fontsize=5.7, color="0.42", va="top", linespacing=1.3)


def caveats(fig, top, height):
    ax = fig.add_axes([0.01, top - height, 0.98, height])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc="#F4F1EC", ec="none", transform=ax.transAxes))
    ax.text(0.012, 0.92, "Caveats and open questions", fontsize=7, fontweight="bold",
            va="top", color="0.15")
    txt = ("• Several choices were made after looking at the data: the ±1 lag control, the "
           "IPS3–5 emphasis, decoding accuracy as the measure in n, and the mapping-error\n"
           "   measure. Effects with pre-specified choices: coupling IPS0–5 at lag 0 / ±1 "
           "(p < .001); routing–value-noise link ρ = .35 (p = .04) is modest at n = 35.\n"
           "• The 90° (vertical) gabor is an outlier in both V1 and parietal geometry — also "
           "the one orientation worth 22 CHF in both mappings and the categorical boundary.\n"
           "• The flexible value model (wPRF) gains slightly from retuning in IPS1–2 "
           "(+0.74 ± 0.34 points, uncorrected); 'no retuning' is strongest for the aPRF.\n"
           "• Left M1 carries value signal (right-hand bids), so it is a motor control region "
           "rather than a null region. Rows 1 and 4 use different model fits (categorical\n"
           "   n = 34 vs sequential n = 35); the categorical n = 35 fit, with a free prior "
           "steepness, is running.")
    ax.text(0.012, 0.74, txt, fontsize=5.9, va="top", color="0.3", linespacing=1.4)


def explain_connectivity(fig, top, height):
    """Four numbered steps of the coupling analysis, as a strip of boxes."""
    steps = [("1  Remove the stimulus",
              "Per voxel and session, remove the\nmean response to each of the 23\n"
              "orientations, each run's mean and\nthe neighbouring trials' orientations.\n"
              "What is left: trial-to-trial noise."),
             ("2  V1 orientation channels",
              "Project V1 residuals onto 24\norientation channels (inverted von\n"
              "Mises encoding model, fitted on\nboth sessions together)."),
             ("3  Coupling profile per voxel",
              "Correlate each parietal voxel's\nresidual with every V1 channel: its\n"
              "coupling profile over orientation\n(a connective field)."),
             ("4  Compare with the mappings",
              "Prediction for a voxel tuned to v:\npeak at the orientation worth v under\n"
              "the mapping in force. Score = fit to\nthis prediction − fit to the other\n"
              "mapping's, minus a label-shuffle null.")]
    w = 0.235
    for i, (head, txt) in enumerate(steps):
        x0 = 0.01 + i * (w + 0.012)
        ax = fig.add_axes([x0, top - height, w, height])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc="#F4F1EC", ec="none",
                                   transform=ax.transAxes))
        ax.text(0.05, 0.93, head, fontsize=6.6, fontweight="bold", color="0.15", va="top")
        ax.text(0.05, 0.74, txt, fontsize=5.9, color="0.3", va="top", linespacing=1.3)
        if i < len(steps) - 1:
            fig.text(x0 + w + 0.002, top - height / 2, "›", fontsize=10, color="0.5",
                     va="center")


def main():
    from abstract_values.visualize.figure_gradient import exclusive_rois, roi_verts
    verts = exclusive_rois(roi_verts())
    maps = dm.mapping_table()
    # inserted bands: (insert after this inch on the original grid, height)
    BANDS = [(2.8, 3.0),        # preferred orientation / value maps
             (5.95, 2.75),      # example participants
             (9.55, 2.85),      # retuning (encoding-model comparison)
             (10.5, 3.0)]       # how the connectivity analysis works
    H = 16.1 + sum(b[1] for b in BANDS) + 1.35     # + caveats box at the bottom
    fig = plt.figure(figsize=(7.25, H))

    def Y(inch):
        return 1 - (inch + sum(h for t, h in BANDS if inch > t)) / H

    def band_top(i):
        """Absolute top (inches) of inserted band i."""
        t, _ = BANDS[i]
        return t + sum(h for tt, h in BANDS[:i]) + 0.05

    def ax_(x, top, w, h):
        return fig.add_axes([x, Y(top + h), w, h / H])

    def L(x, top, s):
        letter(fig, x, Y(top), s)

    def cap(x, top, text, w=None):
        fig.text(x, Y(top), text, fontsize=5.7, color="0.42", va="top", linespacing=1.3)

    fig.text(0.01, Y(0.05), "Abstract value from orientation: fixed codes, flexible routing",
             fontsize=9.5, fontweight="bold", va="top")
    body(fig, Y(0.32),
         "35 participants learned to value oriented gratings (2–42 CHF, BDM auction) under two "
         "orientation-to-value mappings, one per fMRI session (single-trial GLMsingle betas,\n"
         "23 orientations × 8 runs × 2 sessions). Both mappings rise with orientation but bend "
         "differently, so the same value sits at a different orientation in each session.")

    # ── row 0: regions, mappings, where orientation / value are coded ──────
    L(0.01, 0.8, "a")
    roi_brain(ax_(0.075, 0.85, 0.215, 1.3), verts)
    ax3 = fig.add_axes([0.0, Y(1.7), 0.075, 0.75 / H])
    ax3.imshow(plt.imread(REPO / "notes/figures/brain3d_rois_back_inflated.png"))
    ax3.axis("off")
    fig.text(0.0375, Y(1.72), "Seen from\nbehind", fontsize=5.2,
             color="0.45", ha="center", va="top")
    cap(0.03, 2.25, "Cortex cut along the calcarine sulcus and flattened\n(both hemispheres; "
        "inset: the same regions on the\ninflated left hemisphere). V1–V3a/b from Benson,\n"
        "IPS0, IPS1–2, IPS3–5 from Wang.")
    L(0.29, 0.8, "b")
    panel_idea(ax_(0.34, 0.9, 0.16, 1.15), maps)
    cap(0.30, 2.45, "The two mappings. A voxel preferring\n16 CHF should read V1 at 52° "
        "in one\nsession and at 75° in the other.")
    _, _, _, gmaps = fv.strength_measures()
    for k, (key, cm_, title, col) in enumerate((("ori", ORI_CMAP, "Orientation coded", V1_C),
                                                ("val", fv.SIG_CMAP, "Value coded", "#B2301F"))):
        x0 = 0.53 + 0.235 * k
        L(x0, 0.8, "cd"[k])
        vtx, lo, hi = fv.signal_vertex(gmaps[key], cm_)
        axk = ax_(x0 + 0.01, 0.85, 0.22, 1.2)
        _flat(axk, vtx)
        axk.set_title(title, fontsize=6.8, loc="left", pad=1, color=col, fontweight="bold")
        if key == "val":
            axk.annotate("Left M1\n(right-hand bid)", xy=(-172, 52), xytext=(-150, -78),
                         fontsize=5.5, color="0.2", ha="left",
                         arrowprops=dict(arrowstyle="-", color="0.3", lw=0.6))
        small_cbar(fig, [x0 + 0.06, Y(2.13), 0.12, 0.006], cm_, 1, "")
        fig.axes[-1].set_xticks([0, 1])
        fig.axes[-1].set_xticklabels([f"{lo * 100:.1f}%", f"{hi * 100:.1f}%"])
    cap(0.54, 2.35, "Group-mean cross-validated R² over the null model (n = 35): left, von "
        "Mises\ntuning over orientation; right, log-Gaussian tuning over value (CHF). Shown "
        "for the\ntop quarter of cortex. Value signal in motor cortex is left-lateralised (left M1\n0.47%, right M1 −0.23%): participants moved the bid slider with the right hand.")

    # ── tuning maps band ─────────────────────────────────────────────────
    tm = band_top(0)
    tuning_maps(fig, tm, H, verts)

    # ── row 1: behaviour + model ──────────────────────────────────────────
    headline(fig, Y(3.0), "1  A two-stage model splits bid noise into perceptual noise "
             "and value noise")
    body(fig, Y(3.22), "Efficient coding in orientation space (perceptual noise, cardinal "
         "prior) followed by efficient coding in value space (value noise σ_rep), with the "
         "categorical gate at\n90° / 22 CHF; it beats the same model without the gate by "
         "ΔELPD ≈ 270 (LOO). Fitted hierarchically to all bids (NumPyro, 4 chains × 1500 draws, "
         "r̂ ≤ 1.01).")
    L(0.01, 3.7, "e")
    panel_ppc(ax_(0.09, 3.8, 0.24, 1.3), "bias", "Bid − value (CHF)")
    cap(0.05, 5.55, "Mean bid error per orientation (dots: data; CDF violet,\ninverse CDF "
        "(rose) with the categorical two-stage model's 95%\nposterior-predictive band. It misses "
        "the largest\nswings under the inverse-CDF mapping (n = 34 fit).")
    L(0.36, 3.7, "f")
    panel_ppc(ax_(0.44, 3.8, 0.22, 1.3), "sd", "Bid SD (CHF)")
    cap(0.40, 5.55, "Spread of the bids per orientation: largest\nwhere the mapping is "
        "steepest, as predicted\nwhen perceptual noise passes through it.")
    L(0.70, 3.7, "g")
    panel_noise(ax_(0.78, 3.8, 0.20, 1.3))
    cap(0.72, 5.55, "Each participant's noise split by the model\n(categorical; posterior means, n = 34). "
        "Points far below\nthe diagonal: σ_rep at the model's floor.")

    # ── example participants (inserted band, 6.0 .. 6.0 + DELTA) ─────────
    from abstract_values.behavior.data import get_all_behavioral_data
    beh = get_all_behavioral_data()
    beh = beh[beh.event_type == "feedback"].copy()
    beh["response"] = pd.to_numeric(beh.response, errors="coerce")
    beh = beh.reset_index()
    cpar = pd.read_csv(CAT_PARAMS, sep="\t")
    fig.text(0.01, 1 - (band_top(1) + 0.2) / H,
             "Three example participants: bid error (top) and bid spread (bottom) per "
             "orientation, both mappings. Dots: data (8 bids each); bands: 50% and 80% of datasets\n"
             "simulated from that participant's posterior (categorical model). Perceptual noise "
             "passes through the mapping, so error and spread swing with its slope;\nvalue "
             "noise adds spread everywhere. Symbols mark the three participants in g.",
             fontsize=6.3, color="0.3", va="top")
    for k, (sub, title) in enumerate(EXAMPLES):
        t0 = band_top(1)
        axb = fig.add_axes([0.09 + 0.31 * k, 1 - (t0 + 0.75 + 0.75) / H, 0.25, 0.75 / H])
        axs = fig.add_axes([0.09 + 0.31 * k, 1 - (t0 + 1.6 + 0.75) / H, 0.25, 0.75 / H])
        panel_example(axb, axs, beh, sub, title, cpar, k == 0)

    # ── row 2: geometry ───────────────────────────────────────────────────
    headline(fig, Y(6.35), "2  Fixed codes: V1 represents orientation as a circle, parietal "
             "cortex as an ordered line")
    body(fig, Y(6.57), "Cross-validated Mahalanobis (crossnobis) distances between the voxel "
         "patterns of the 23 orientations, noise covariance estimated on training runs only.\n"
         "Neither geometry re-scales when the mapping changes (change between mappings ≈ 0, "
         "although an injected re-scaling component\nof 10–25% of the region's structure would "
         "have been detected).")
    rdm = load_rdms()
    rd = {"V1": subject_rdms(rdm, ["V1"]), "Parietal (IPS0–5)":
          subject_rdms(rdm, ["IPS0", "IPS1–2", "IPS3–5"])}
    for r in ("IPS0", "IPS1–2", "IPS3–5"):
        rd[r] = subject_rdms(rdm, [r])
    oris = rd["V1"][1]
    L(0.01, 7.2, "h")
    panel_ends(ax_(0.08, 7.3, 0.26, 1.35), rd,
               (("V1", V1_C), ("IPS0", ROI_COL["IPS0"]), ("IPS1–2", ROI_COL["IPS1–2"]),
                ("IPS3–5", ROI_COL["IPS3–5"])))
    cap(0.03, 9.05, "Distinctness from the two ends of the range\n(7.5°, 172.5°; averaged) by "
        "how far along the\nrange an orientation lies. V1 falls back at the\nfar end, as on a "
        "circle; none of the IPS maps\ndoes. Mean ± SE, n = 34.")
    L(0.37, 7.2, "i")
    fig.text(0.39, Y(7.22), "Multidimensional scaling (MDS) of the distance matrices",
             fontsize=6.3, color="0.2", va="top", fontweight="bold")
    for k, (name, col) in enumerate((("V1", V1_C), ("Parietal (IPS0–5)", PAR_C))):
        R = np.mean(list(rd[name][0].values()), 0)
        panel_mds(ax_(0.39 + 0.17 * k, 7.45, 0.155, 1.3), R, oris, col,
                  name.replace(" (IPS0–5)", ""))
    cap(0.39, 9.05, "2-D MDS: each dot is one orientation, placed so that\ndistances between "
        "dots match the neural distances.\nDashed: the step from 172.5° back to 7.5°. V1 closes "
        "\ninto a loop, parietal stays an open arc. 90° is an\noutlier in both regions.")
    L(0.74, 7.2, "j")
    panel_stability(ax_(0.81, 7.3, 0.17, 1.25))
    cap(0.75, 9.05, "Does the geometry change between\nmappings the way the values do? "
        "No:\nslope ≈ 0 everywhere, while a value\ncode that re-scaled (injected into the\n"
        "real data) would be detected.")

    # ── retuning band ──────────────────────────────────────────────────────
    t1 = band_top(2)
    fig.text(0.01, 1 - (t1 + 0.15) / H, "Tuning does not follow the mapping either: "
             "encoding models with one tuning for both sessions vs a twin whose preferred\n"
             "value / orientation may change between sessions (leave-one-run-out cvR², vertices "
             "where the fixed model beats the null; positive = retuning helps).\n"
             "aPRF: one log-Gaussian tuning curve over value per voxel (preferred value, width, "
             "amplitude, baseline). wPRF: a weighted sum of 8 fixed log-Gaussian basis\n"
             "functions over value, so any tuning shape. von Mises: a weighted sum of 8 von "
             "Mises basis functions over orientation (180° circle). All fitted to single-trial\n"
             "betas of both sessions; the shift twins refit the preferred value / the weights "
             "per session.", fontsize=6.3, color="0.3", va="top")
    L(0.01, t1 + 0.45 - sum(h for t, h in BANDS[:1]), "")
    axr = fig.add_axes([0.09, 1 - (t1 + 1.15 + 1.2) / H, 0.5, 1.2 / H])
    rt = panel_retuning(axr)
    letter(fig, 0.01, 1 - (t1 + 1.1) / H, "k")
    fig.text(0.79, 1 - (t1 + 1.15) / H, rt, fontsize=5.8, color="0.42", va="top",
             linespacing=1.35)

    # ── row 3: routing ────────────────────────────────────────────────────
    headline(fig, Y(9.75), "3  Flexible routing: parietal cortex reads V1 through the "
             "mapping currently in force")
    body(fig, Y(9.97), "Trial-to-trial coupling between parietal voxels and V1 orientation "
         "channels (inverted von Mises model), after removing every voxel's mean response to "
         "each\nstimulus and the neighbouring trials' orientations (±1). Score: how well the "
         "coupling matches the in-force mapping minus the other mapping, minus\nthe same with "
         "tuning labels shuffled across voxels.")
    t2 = band_top(3)
    explain_schematic(fig, 1 - (t2 + 0.05) / H, 1.65 / H, maps)
    explain_connectivity(fig, 1 - (t2 + 1.85) / H, 1.0 / H)
    d = pd.concat(pd.read_csv(f, sep="\t", dtype={"subject": str}) for f in
                  (dm.CF / "peaks_lags-1_target-IPS0IPS12IPS345").glob("sub-*/*_desc-maps.tsv"))
    L(0.01, 10.6, "l")
    panel_aligned(ax_(0.11, 10.7, 0.27, 1.3), d, maps)
    cap(0.05, 12.6, "IPS0–5 voxels' coupling profiles re-centred on the V1\norientation "
        "worth their preferred value, under the mapping\nin force or the other one. Mean ± "
        "SE, n = 35.")
    L(0.42, 10.6, "m")
    r_maps = panel_maps(ax_(0.5, 10.7, 0.24, 1.3))
    cap(0.44, 12.45, "Mapping score per IPS map (dots: participants;\nmean ± SE, t-test "
        "against 0, n = 35). Grows\nfrom IPS0 to IPS3–5; brain: the same scores.")
    vals = {r: np.nan for r in ROI_COL}
    vals.update({row.roi: row.m for row in r_maps.itertuples()})
    axb = ax_(0.765, 10.8, 0.23, 1.05)
    cm = value_brain(axb, verts, vals, REP_C, 0.02)
    small_cbar(fig, [0.83, Y(11.95), 0.1, 0.005], cm, 0.02, "Mapping score")

    # ── row 4: individual differences ─────────────────────────────────────
    headline(fig, Y(13.05), "4  Individual differences: perceptual noise in early visual "
             "cortex, value noise in parietal cortex and in how it reads V1")
    body(fig, Y(13.27), "Across participants: each region's value decoding (leave-one-run-out "
         "Bayesian decoding, r between decoded and true value) regressed on the model's\n"
         "perceptual and value precision together, with head motion as covariate.")
    L(0.01, 13.75, "n")
    grad, est = panel_gradient_est(ax_(0.11, 13.95, 0.27, 1.2))
    cap(0.05, 15.85, "Std. β per region, ±1 SE (participant bootstrap).")
    for k, (col, lab) in enumerate(((PER_C, "Perceptual precision"),
                                    (REP_C, "Value precision"))):
        ax = ax_(0.415 + 0.15 * k, 14.05, 0.145, 0.9)
        cmk = value_brain(ax, verts, {r: est[r][k] for r in fv.DORSAL}, col, 0.3)
        ax.set_title(lab, fontsize=6, color=col, fontweight="bold", pad=1)
        small_cbar(fig, [0.445 + 0.15 * k, Y(15.05), 0.085, 0.005], cmk, 0.3, "Std. β")
    cap(0.42, 15.85, "The same β painted on the regions.")
    L(0.73, 13.75, "o")
    ind = panel_indiv(ax_(0.82, 13.95, 0.16, 1.2))
    cap(0.74, 15.85, "Routing score (IPS0–5, ±1 lag) vs the\nmodel's value precision; "
        "line ± 68% CI.")

    caveats(fig, 1 - (H - 1.3) / H, 1.2 / H)
    out = REPO / "notes/figures/fig_summary.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(r_maps[["roi", "m", "se", "p"]].round(4).to_string(index=False))
    print("gradient", np.round(grad, 3), "indiv", ind)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
