"""Figure: V1 represents orientation as a circle, parietal cortex as an ordered line.

All from crossnobis RDMs over the 23 orientations (7.5..172.5 deg), both
sessions averaged, each participant's RDM divided by its mean before
averaging. Parietal = Wang IPS0, IPS1-2 and IPS3-5 (no NPC).

  a  Group RDMs. A circular code puts 7.5 and 172.5 deg next to each other
     (the corners of the matrix are dark); an ordered code puts them furthest
     apart (bright corners).
  b  The same as distance from the two END orientations, against how far
     along the 7.5..172.5 range the other orientation lies. A circle rises
     then falls back (the far end is near again); a line keeps rising.
  c  2-D MDS of the group RDMs, points coloured by orientation, the seam
     (172.5 -> 7.5) dashed.
  d  Per region, the weight of circular and of linear (ordered) orientation
     distance in a joint regression on each participant's RDM.

    python -m abstract_values.geometry.figure_circularity
    -> notes/figures/fig_circularity.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import MDS

from abstract_values.geometry.analyze_crossnobis import DORSAL, load

REPO = Path(__file__).resolve().parents[2]
V1_C, PAR_C = "#1B8A80", "#D9730D"
PARIETAL = ["IPS0", "IPS1–2", "IPS3–5"]

mpl.rcParams.update({
    "font.family": "Helvetica", "font.size": 7, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def subject_rdms(d, rois):
    """subject -> 23x23 RDM (mean over sessions and the given ROIs, normalised)."""
    oris = np.sort(d.ori_i.unique().tolist() + [d.ori_j.max()])
    oris = np.unique(oris)
    k = {o: i for i, o in enumerate(oris)}
    out = {}
    for s, g in d[d.roi.isin(rois)].groupby("subject"):
        m = g.groupby(["ori_i", "ori_j"]).d.mean()
        R = np.zeros((len(oris), len(oris)))
        for (a, b), v in m.items():
            R[k[a], k[b]] = R[k[b], k[a]] = v
        out[s] = R / R[np.triu_indices(len(oris), 1)].mean()
    return out, oris


def main():
    d = load()
    regions = {"V1": ["V1"], "Parietal (IPS0–5)": PARIETAL}
    cols = {"V1": V1_C, "Parietal (IPS0–5)": PAR_C}
    rd = {name: subject_rdms(d, rois) for name, rois in regions.items()}
    oris = rd["V1"][1]
    n = len(rd["V1"][0])

    fig = plt.figure(figsize=(7.25, 5.0))
    # a: RDMs
    for k, name in enumerate(regions):
        ax = fig.add_axes([0.06 + 0.2 * k, 0.56, 0.17, 0.34])
        R = np.mean(list(rd[name][0].values()), 0)
        np.fill_diagonal(R, np.nan)
        im = ax.imshow(R, cmap="mako_r", vmin=0, vmax=np.nanpercentile(R, 98),
                       origin="upper")
        ax.set_xticks([0, 11, 22])
        ax.set_xticklabels(["7.5°", "90°", "172.5°"])
        ax.set_yticks([0, 11, 22])
        ax.set_yticklabels(["7.5°", "90°", "172.5°"] if k == 0 else [])
        ax.set_title(name, color=cols[name], fontweight="bold", fontsize=7.5, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.annotate("Seam pair", xy=(22, 0), xytext=(15.5, -3.2), fontsize=6,
                    color="0.3", arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6),
                    annotation_clip=False)
    cax = fig.add_axes([0.06, 0.53, 0.37, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["0", "Mean"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=1.5)
    cax.set_xlabel("Crossnobis distance (relative to participant's mean)", fontsize=6.5,
                   labelpad=1)
    fig.text(0.01, 0.94, "a", fontsize=8, fontweight="bold")

    # b: distance from the end orientations
    ax = fig.add_axes([0.53, 0.58, 0.42, 0.33])
    sep = np.arange(1, len(oris))                       # steps along the range
    for name in regions:
        per = []
        for R in rd[name][0].values():
            a = R[0, 1:]                                # from 7.5 deg
            b = R[-1, :-1][::-1]                        # from 172.5 deg, mirrored
            per.append((a + b) / 2)
        per = np.array(per)
        m, se = per.mean(0), per.std(0) / np.sqrt(len(per))
        x = sep * 7.5
        ax.fill_between(x, m - se, m + se, color=cols[name], alpha=0.18, lw=0)
        ax.plot(x, m, "-o", color=cols[name], ms=3, lw=1.6, mec="white", mew=0.5)
        ax.text(x[-1] + 4, m[-1], name, color=cols[name], fontsize=7,
                fontweight="bold", va="center")
    ax.axvline(90, color="0.75", lw=0.6, ls=(0, (3, 2)))
    ax.text(91, ax.get_ylim()[1], "Opposite on the circle", fontsize=6, color="0.45",
            va="top")
    ax.set_xticks([0, 45, 90, 135, 165])
    ax.set_xlabel("Distance along 7.5–172.5° from an end orientation (°)")
    ax.set_ylabel("Crossnobis distance\n(relative)")
    ax.set_xlim(0, 215)
    sns.despine(ax=ax, offset=4, trim=True)
    fig.text(0.47, 0.94, "b", fontsize=8, fontweight="bold")

    # c: MDS
    tw = plt.get_cmap("twilight")
    for k, name in enumerate(regions):
        ax = fig.add_axes([0.04 + 0.22 * k, 0.07, 0.19, 0.34])
        R = np.mean(list(rd[name][0].values()), 0)
        R = np.clip(R, 0, None)
        Y = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
                normalized_stress="auto").fit_transform(np.sqrt(R))
        Y -= Y.mean(0)
        ax.plot(Y[:, 0], Y[:, 1], "-", color="0.6", lw=0.8, zorder=1)
        ax.plot([Y[-1, 0], Y[0, 0]], [Y[-1, 1], Y[0, 1]], color="0.3", lw=1.0,
                ls=(0, (2, 1.5)), zorder=1)
        ax.scatter(Y[:, 0], Y[:, 1], c=[tw(o / 180) for o in oris], s=22,
                   edgecolor="white", lw=0.5, zorder=2)
        for i in (0, 11, 22):
            ax.text(Y[i, 0], Y[i, 1] + 0.06 * np.ptp(Y[:, 1]), f"{oris[i]:g}°",
                    fontsize=6, ha="center", color="0.25")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(name, color=cols[name], fontweight="bold", fontsize=7.5, loc="left")
    fig.text(0.01, 0.44, "c", fontsize=8, fontweight="bold")
    fig.text(0.04, 0.035, "Dashed: the seam between 172.5° and 7.5°", fontsize=6,
             color="0.4")

    # d: circular vs linear weight along the hierarchy
    ax = fig.add_axes([0.56, 0.11, 0.39, 0.29])
    dd = d.copy()
    dd["lin_dist"] = np.abs(dd.ori_i - dd.ori_j) / 165
    rows = []
    for (s, r), g in dd.groupby(["subject", "roi"]):
        X = np.column_stack([g.ori_dist, g.lin_dist, g.val_dist,
                             g.session == g.session.min(),
                             g.session != g.session.min()]).astype(float)
        y = g.d.to_numpy() / g.d.mean()
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        rows.append(dict(subject=s, roi=r, circ=b[0], lin=b[1]))
    w = pd.DataFrame(rows)
    x = np.arange(len(DORSAL))
    for key, col, lab, dx in (("circ", "0.15", "Circular", -0.08),
                              ("lin", "0.6", "Ordered (linear)", 0.08)):
        m = w.groupby("roi")[key].mean().reindex(DORSAL)
        se = w.groupby("roi")[key].sem().reindex(DORSAL)
        ax.vlines(x + dx, m - se, m + se, color=col, lw=1.1)
        ax.plot(x + dx, m, "-o", color=col, ms=4, lw=1.5, mec="white", mew=0.6)
        ax.text(x[-1] + 0.3, m.iloc[-1], lab, color=col, fontsize=7, fontweight="bold",
                va="center")
    ax.axhline(0, color="0.75", lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(x)
    ax.set_xticklabels(DORSAL)
    ax.set_xlim(-0.4, len(DORSAL) + 1.2)
    ax.set_ylabel("Weight of orientation\ndistance (relative)")
    sns.despine(ax=ax, offset=4, trim=True)
    fig.text(0.49, 0.44, "d", fontsize=8, fontweight="bold")

    fig.text(0.99, 0.005, f"n = {n}; bands and bars ±1 SE", fontsize=6, color="0.5",
             ha="right")
    out = REPO / "notes/figures/fig_circularity.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(w.groupby("roi")[["circ", "lin"]].agg(["mean", "sem"]).reindex(DORSAL).round(3))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
