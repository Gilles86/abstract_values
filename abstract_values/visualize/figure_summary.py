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
def panel_ends(ax, rd):
    for name, col in (("V1", V1_C), ("Parietal (IPS0–5)", PAR_C)):
        per = np.array([(R[0, 1:] + R[-1, :-1][::-1]) / 2 for R in rd[name][0].values()])
        x = np.arange(1, per.shape[1] + 1) * 7.5
        m, se = per.mean(0), per.std(0) / np.sqrt(len(per))
        ax.fill_between(x, m - se, m + se, color=col, alpha=0.18, lw=0)
        ax.plot(x, m, "-", color=col, lw=1.6)
        ax.text(x[-1] + 4, m[-1], name.split(" (")[0], color=col, fontsize=7,
                fontweight="bold", va="center")
    ax.axvline(90, color="0.8", lw=0.6, ls=(0, (3, 2)))
    ax.text(88, 0.05, "Opposite on\nthe circle", fontsize=6, color="0.5", va="bottom",
            ha="right")
    ax.set_xticks([0, 45, 90, 135, 165])
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
def panel_gradient(ax):
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
    ax.set_title(f"Interaction (value − perceptual, parietal − early): "
                 f"{g:+.2f} ± {Bg.std():.2f}, P = {p:.3f}", loc="left", fontsize=6.3,
                 color="0.3")
    sns.despine(ax=ax, offset=3, trim=True)
    ax.set_xticks(x)
    ax.set_xticklabels(fv.DORSAL, rotation=35, ha="right", rotation_mode="anchor")
    return g, Bg.std(), p


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
    ax.set_xlabel("Value precision (model), −log σ_rep")
    ax.set_ylabel("Parietal routing follows\nthe mapping (IPS0–5)")
    ax.set_title(f"ρ = {rho:+.2f}, p = {p:.3f} (r = {r:+.2f}), n = {len(j)}", loc="left",
                 fontsize=6.3, color="0.3")
    sns.despine(ax=ax, offset=3, trim=True)
    return rho, p, len(j)


def main():
    fig = plt.figure(figsize=(7.25, 8.2))
    # row 1
    headline(fig, 0.995, "Fixed codes: V1 represents orientation as a circle, "
             "parietal cortex as an ordered line")
    rdm = load_rdms()
    rd = {"V1": subject_rdms(rdm, ["V1"]), "Parietal (IPS0–5)":
          subject_rdms(rdm, ["IPS0", "IPS1–2", "IPS3–5"])}
    oris = rd["V1"][1]
    letter(fig, 0.01, 0.955, "a")
    panel_ends(fig.add_axes([0.09, 0.745, 0.36, 0.18]), rd)
    letter(fig, 0.52, 0.955, "b")
    for k, (name, col) in enumerate((("V1", V1_C), ("Parietal (IPS0–5)", PAR_C))):
        R = np.mean(list(rd[name][0].values()), 0)
        panel_mds(fig.add_axes([0.55 + 0.23 * k, 0.73, 0.2, 0.2]), R, oris, col,
                  name.replace(" (IPS0–5)", ""))
    fig.text(0.55, 0.715, "Dots: the 23 orientations (circular colour scale)\n"
             "Dashed: the step from 172.5° back to 7.5°", fontsize=5.8, color="0.45", va="top")

    # row 2
    headline(fig, 0.665, "Flexible routing: parietal cortex reads V1 through the mapping "
             "currently in force")
    maps = dm.mapping_table()
    d = pd.concat(pd.read_csv(f, sep="\t", dtype={"subject": str}) for f in
                  (dm.CF / "peaks_lags-1_target-IPS0IPS12IPS345").glob("sub-*/*_desc-maps.tsv"))
    letter(fig, 0.01, 0.625, "c")
    panel_aligned(fig.add_axes([0.11, 0.405, 0.34, 0.19]), d, maps)
    letter(fig, 0.52, 0.625, "d")
    r_maps = panel_maps(fig.add_axes([0.62, 0.405, 0.33, 0.19]))

    # row 3
    headline(fig, 0.315, "Individual differences: value noise lives in parietal cortex "
             "and in how it reads V1")
    letter(fig, 0.01, 0.275, "e")
    grad = panel_gradient(fig.add_axes([0.11, 0.065, 0.34, 0.18]))
    letter(fig, 0.52, 0.275, "f")
    ind = panel_indiv(fig.add_axes([0.62, 0.065, 0.33, 0.18]))

    out = REPO / "notes/figures/fig_summary.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(r_maps[["roi", "m", "se", "p"]].round(4).to_string(index=False))
    print("gradient", np.round(grad, 3), "indiv", ind)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
