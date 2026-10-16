"""Main figure: parietal cortex reads V1 through the mapping currently in force.

  a  The idea. Both mappings rise with orientation but bend differently, so
     the same value is worth a different orientation in each session. A
     parietal voxel preferring 16 CHF should couple to the V1 population at
     the orientation worth 16 CHF under the mapping in force.
  b  Coupling maps (IPS0-5, all 35 participants): trial-to-trial coupling
     of parietal voxels, binned by preferred value, to V1 orientation
     channels, separately for the two mappings. Stimulus means removed.
  c  Every voxel's coupling profile re-centred on the orientation worth its
     preferred value under the mapping in force, or under the other one.
  d  Mapping score (observed - label-shuffled) per IPS map, with the
     neighbouring trials' stimuli removed (+-1) or not (lag 0).
  e  Which mapping was in force, read out from each session's coupling map.
  f  Individual differences: the mapping score (IPS0-5, +-2 lags) against
     value precision from the model (+-1 lag) and against how far the mean bids sit
     from the actual mapping.

    python -m abstract_values.connective_fields.figure_connectopy
    -> notes/figures/fig_connectopy.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from abstract_values.connective_fields import decode_mapping as dm
from abstract_values.connective_fields import individual_differences as idf

REPO = Path(__file__).resolve().parents[2]
CF = dm.CF
ACC = "#D9730D"
MAPS = [("IPS0", "IPS0"), ("IPS1–2", "IPS12"), ("IPS3–5", "IPS345")]
UNION = ("IPS0–5", "IPS0IPS12IPS345")
MAP_COL = {"cdf": "#3B5BA5", "inverse_cdf": "#B8860B"}

plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 7, "axes.labelsize": 7.5,
    "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def letter(fig, x, y, s):
    fig.text(x, y, s, fontsize=8.5, fontweight="bold", va="top")


def panel_idea(ax, maps):
    vv = np.linspace(2, 42, 300)
    for m, lab in (("cdf", "CDF"), ("inverse_cdf", "Inverse CDF")):
        th = dm.theta_of_value(maps[m], vv)
        ax.plot(th, vv, color=MAP_COL[m], lw=1.8)
    v0 = 16.0
    th0 = {m: float(dm.theta_of_value(maps[m], v0)) for m in maps}
    ax.axhline(v0, color="0.6", lw=0.7, ls=(0, (3, 2)), xmax=0.98)
    for m in maps:
        ax.plot([th0[m]] * 2, [0, v0], color=MAP_COL[m], lw=0.9, ls=(0, (1.5, 1.5)))
        ax.plot(th0[m], v0, "o", color=MAP_COL[m], ms=4.5, mec="white", mew=0.8, zorder=4)
    ax.annotate("", xy=(th0["cdf"], 1.0), xytext=(th0["inverse_cdf"], 1.0),
                arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.8))
    ax.text(np.mean(list(th0.values())), 2.5, f"{abs(th0['cdf'] - th0['inverse_cdf']):.0f}°",
            ha="center", fontsize=6.5, color="0.3")
    ax.text(92, v0 + 1.0, "Parietal voxel\npreferring 16 CHF", fontsize=6.3, color="0.3",
            va="bottom")
    xs = {"cdf": (110, 12), "inverse_cdf": (95, 33)}
    for m, lab in (("cdf", "CDF mapping"), ("inverse_cdf", "Inverse-CDF\nmapping")):
        ax.text(*xs[m], lab, color=MAP_COL[m], fontsize=7, fontweight="bold")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 42)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_yticks([2, 12, 22, 32, 42])
    ax.set_xlabel("Gabor orientation (°)")
    ax.set_ylabel("Value (CHF)")
    sns.despine(ax=ax, offset=3, trim=True)


def coupling_maps(fig, rects, d, maps):
    values = np.sort(d.value_bin.unique())
    oris = np.sort(d.orientation.unique())
    cmap = sns.color_palette("rocket", as_cmap=True)
    vv = np.linspace(2, 42, 200)
    for (c, title), rect in zip((("cdf", "CDF sessions"), ("inverse_cdf", "Inverse-CDF sessions")),
                                rects):
        ax = fig.add_axes(rect)
        G = d[d.condition == c].groupby(["value_bin", "orientation"]).cf.mean().unstack() \
            .reindex(index=values, columns=oris)
        X = dm.smooth_rows(G)
        X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
        ax.pcolormesh(oris, values, X, cmap=cmap, vmin=-2, vmax=2.2, shading="gouraud",
                      rasterized=True)
        other = [m for m in dm.CONDS if m != c][0]
        ax.plot(dm.theta_of_value(maps[other], vv), vv, color="#BBBBBB", lw=1.0,
                ls=(0, (2, 1.5)))
        ax.plot(dm.theta_of_value(maps[c], vv), vv, color="#7FDBFF", lw=1.6)
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 90, 180])
        ax.set_xlabel("V1 channel (°)")
        if rect is rects[0]:
            ax.set_ylabel("Preferred value of\nparietal voxels (CHF)")
        else:
            ax.set_yticklabels([])
        ax.set_title(title, loc="left", color=MAP_COL[c], fontweight="bold")
        sns.despine(ax=ax, offset=3, trim=True)
    return cmap


def panel_aligned(ax, d, maps):
    oris = np.sort(d.orientation.unique())
    for ref, col, lab in (("this", ACC, "Mapping in force"), ("other", "0.55", "Other mapping")):
        rel, P = dm.aligned(d, maps, oris, ref)
        m, se = P.mean(0).to_numpy(), P.sem(0).to_numpy()
        ax.fill_between(rel, m - se, m + se, color=col, alpha=0.2, lw=0)
        ax.plot(rel, m, color=col, lw=1.6)
        i = int(np.argmax(m))
        if ref == "this":
            ax.text(rel[i] + 12, m[i], lab, color=col, fontsize=6.8, fontweight="bold",
                    va="center")
        else:
            ax.text(-88, 0.62 * m[i], lab, color=col, fontsize=6.8, fontweight="bold",
                    va="center")
    ax.axvline(0, color="0.75", lw=0.6, ls=(0, (3, 2)))
    ax.axhline(0, color="0.8", lw=0.6)
    ax.set_xlim(-90, 90)
    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_xlabel("V1 channel relative to the orientation\nworth the voxel's preferred value (°)")
    ax.set_yticks([-0.005, 0, 0.005, 0.01])
    ax.set_ylabel("Coupling (voxel-specific)")
    sns.despine(ax=ax, offset=3, trim=True)


def panel_maps(ax):
    rows = []
    for lab, tag in MAPS + [UNION]:
        for lag, pre in (("Lag 0", ""), ("±1 lag", "lags-1_")):
            x = idf.scores(f"coupling_{pre}iem-k24-kappa16_target-{tag}").dropna()
            t, p = stats.ttest_1samp(x, 0)
            rows.append(dict(roi=lab, lag=lag, m=x.mean(), se=x.sem(), p=p, n=len(x)))
    r = pd.DataFrame(rows)
    pos = {lab: i for i, (lab, _) in enumerate(MAPS)}
    pos[UNION[0]] = len(MAPS) + 0.8
    for lag, dx, fill in (("Lag 0", -0.12, ACC), ("±1 lag", 0.12, "white")):
        g = r[r.lag == lag]
        x = np.array([pos[q] for q in g.roi]) + dx
        ax.vlines(x, g.m - g.se, g.m + g.se, color=ACC, lw=1.1)
        main = g.roi != UNION[0]
        ax.plot(x[main.values], g.m[main], "-", color=ACC, lw=1.0, alpha=0.6)
        ax.plot(x, g.m, "o", color=ACC, mfc=fill, mec=ACC, ms=4.5, mew=1.1, zorder=4)
    ax.axhline(0, color="0.7", lw=0.6, ls=(0, (3, 2)))
    ax.axvline(len(MAPS) + 0.2, color="0.85", lw=0.8, ls=(0, (1, 2)))
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels(list(pos.keys()))
    ax.set_xlim(-0.5, len(MAPS) + 1.3)
    ax.set_ylim(-0.006, 0.03)
    ax.set_ylabel("Mapping score\n(observed − shuffled)")
    # key, drawn as glyphs
    for i, (lab, fill) in enumerate((("Lag 0", ACC), ("Neighbouring trials removed", "white"))):
        y = 0.97 - 0.09 * i
        ax.plot(0.04, y, "o", transform=ax.transAxes, color=ACC, mfc=fill, mec=ACC, ms=4,
                mew=1.1, clip_on=False)
        ax.text(0.08, y, lab, transform=ax.transAxes, fontsize=6.3, va="center", color="0.3")
    sns.despine(ax=ax, offset=3, trim=True)
    return r


def panel_decoder(ax):
    rows = []
    for lab, tag in MAPS + [UNION]:
        f = REPO / f"notes/data/cf_decode_mapping_peaks_target-{tag}.tsv"
        res = pd.read_csv(f, sep="\t", dtype={"subject": str})
        k, n = int((res.score > 0).sum()), len(res)
        rows.append(dict(roi=lab, acc=k / n, k=k, n=n,
                         p=stats.binomtest(k, n, 0.5, alternative="greater").pvalue,
                         se=np.sqrt(0.25 / n)))
    r = pd.DataFrame(rows)
    x = np.array([0, 1.15, 2.3, 4.0])
    ax.bar(x, r.acc * 100, width=0.75, color=[ACC] * 3 + ["#A8550A"], lw=0)
    ax.axhline(50, color="0.5", lw=0.7, ls=(0, (3, 2)))
    ax.text(x[-1] + 0.5, 50, "Chance", fontsize=6, color="0.45", va="center")
    for xi, row in zip(x, r.itertuples()):
        ax.text(xi, row.acc * 100 + 1.5, f"{row.k}/{row.n}", ha="center", fontsize=5.8,
                color="0.3")
    ax.set_xticks(x)
    ax.set_xticklabels(r.roi, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylim(40, 72)
    ax.set_yticks([40, 50, 60, 70])
    ax.set_xlim(-0.55, 5.2)
    ax.set_ylabel("Sessions whose mapping\nis read out correctly (%)")
    sns.despine(ax=ax, offset=3, trim=True)
    ax.set_xticks(x)
    ax.set_xticklabels(r.roi, rotation=35, ha="right", rotation_mode="anchor")
    return r


def panel_scatter(ax, x, y, xlab, show_y):
    j = pd.concat([x, y], axis=1).dropna()
    rho, p = stats.spearmanr(j.iloc[:, 0], j.iloc[:, 1])
    r, _ = stats.pearsonr(j.iloc[:, 0], j.iloc[:, 1])
    sns.regplot(x=j.iloc[:, 0], y=j.iloc[:, 1], ax=ax, color=ACC, ci=68,
                scatter_kws=dict(s=13, edgecolor="white", linewidths=0.5),
                line_kws=dict(lw=1.3))
    ax.set_xlabel(xlab)
    ax.set_ylabel("Mapping score\n(IPS0–5, ±1 lag)" if show_y else "")
    ax.set_title(f"ρ = {rho:+.2f}, p = {p:.3f}   (r = {r:+.2f}, n = {len(j)})",
                 loc="left", fontsize=6.5, color="0.25")
    ax.axhline(0, color="0.8", lw=0.6, ls=(0, (3, 2)), zorder=0)
    sns.despine(ax=ax, offset=3, trim=True)


def main():
    maps = dm.mapping_table()
    d = pd.concat(pd.read_csv(f, sep="\t", dtype={"subject": str}) for f in
                  (CF / f"peaks_lags-1_target-{UNION[1]}").glob("sub-*/sub-*_desc-maps.tsv"))
    fig = plt.figure(figsize=(7.25, 7.8))

    # row 1
    letter(fig, 0.01, 0.985, "a")
    panel_idea(fig.add_axes([0.075, 0.71, 0.25, 0.25]), maps)
    letter(fig, 0.38, 0.985, "b")
    cmap = coupling_maps(fig, [[0.44, 0.71, 0.2, 0.25], [0.67, 0.71, 0.2, 0.25]], d, maps)
    cax = fig.add_axes([0.905, 0.78, 0.012, 0.15])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-2, 2.2), cmap=cmap), cax=cax)
    cb.set_ticks([-2, 0, 2])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=1.5)
    cb.set_label("Coupling\n(z within row)", fontsize=6)
    kx = fig.add_axes([0.89, 0.66, 0.11, 0.07])
    kx.axis("off")
    kx.plot([0.0, 0.18], [0.75, 0.75], color="#7FDBFF", lw=1.6)
    kx.text(0.24, 0.75, "In force", fontsize=6.2, va="center")
    kx.plot([0.0, 0.18], [0.25, 0.25], color="#BBBBBB", lw=1.0, ls=(0, (2, 1.5)))
    kx.text(0.24, 0.25, "Other\nmapping", fontsize=6.2, va="center")
    kx.set_xlim(0, 1)
    kx.set_ylim(0, 1)

    # row 2
    letter(fig, 0.01, 0.62, "c")
    panel_aligned(fig.add_axes([0.095, 0.38, 0.27, 0.21]), d, maps)
    letter(fig, 0.41, 0.62, "d")
    r_maps = panel_maps(fig.add_axes([0.48, 0.38, 0.24, 0.21]))
    letter(fig, 0.75, 0.62, "e")
    r_dec = panel_decoder(fig.add_axes([0.82, 0.38, 0.16, 0.21]))

    # row 3
    S = idf.scores(idf.SPECS["IPS0–5, ±1 lag (primary)"])
    b = idf.behaviour(list(S.index))
    letter(fig, 0.01, 0.29, "f")
    panel_scatter(fig.add_axes([0.095, 0.085, 0.26, 0.16]), b.value_precision, S,
                  "Value precision, −log σ_rep (model)", True)
    panel_scatter(fig.add_axes([0.43, 0.085, 0.26, 0.16]), b.mapping_error, S,
                  "Mapping error (CHF)\nmean bid vs actual mapping", False)
    ax = fig.add_axes([0.74, 0.085, 0.24, 0.16])
    ax.axis("off")
    j = pd.concat([S.rename("cf"), b], axis=1).dropna(subset=["cf"])
    rp = stats.spearmanr(j.cf, j.perceptual_precision, nan_policy="omit")
    ax.text(0, 1, "Participants whose parietal cortex follows\nthe mapping in force more "
            "closely have\nless value noise. The link with how far\nthe mean bids sit from "
            f"the mapping is\nweaker; perceptual precision is unrelated\n"
            f"(ρ = {rp[0]:+.2f}, p = {rp[1]:.2f}).",
            fontsize=6.5, color="0.3", va="top", linespacing=1.4)

    out = REPO / "notes/figures/fig_connectopy.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(r_maps.round(4).to_string(index=False))
    print(r_dec.round(3).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
