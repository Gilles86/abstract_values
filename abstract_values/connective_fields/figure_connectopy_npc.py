"""Companion to fig_connectopy: the NPC1-3 numerosity maps next to the Wang IPS maps.

  a  Coupling maps (+-1 lag) for the IPS0-5 union and the NPC1-3 union, as in
     fig_connectopy panel b.
  b  Mapping score (observed - label-shuffled) per map, lag 0 and +-1 lag
     (fig_connectopy panel d, extended).
  c  Which mapping was in force, read out from each session's lag-0 coupling
     map (fig_connectopy panel e, extended).
  d  Sensitivity: the same score with each voxel's full value-basis tuning
     curve (aprf-weighted design) as the template instead of its single peak.
  e  Single-peak score (+-1 lag) split by which model wins the voxel: value
     vs orientation aPRF, single-peak vs value-basis.

    python -m abstract_values.connective_fields.scores_by_roi      # TSV first
    python -m abstract_values.connective_fields.figure_connectopy_npc
    -> notes/figures/fig_connectopy_npc.pdf
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abstract_values.connective_fields import decode_mapping as dm
from abstract_values.connective_fields import figure_connectopy as fc  # sets rcParams
from abstract_values.connective_fields.scores_by_roi import REPO, ROIS

ACC = fc.ACC
NPC_COL = "#2C7A7B"
UNIONS = ("IPS0–5", "NPC1–3")


def colour(roi):
    return ACC if roi.startswith("IPS") else NPC_COL


def xpos():
    pos, x = {}, 0.0
    for lab, _ in ROIS:
        pos[lab] = x
        x += 1.0 + (0.6 if lab in UNIONS else 0.0) + (0.5 if lab == "IPS0–5" else 0.0)
    return pos


def panel_scores(ax, r, template="bell"):
    pos = xpos()
    r = r[(r.template == template) & (r.voxels == "all")]
    for lag, dx in (("Lag 0", -0.14), ("±1 lag", 0.14)):
        g = r[r.lag == lag].dropna(subset=["mean"])
        for fam in ("IPS", "NPC"):
            h = g[g.family == fam]
            x = np.array([pos[q] for q in h.roi]) + dx
            col = ACC if fam == "IPS" else NPC_COL
            fill = col if lag == "Lag 0" else "white"
            ax.vlines(x, h["mean"] - h["sem"], h["mean"] + h["sem"], color=col, lw=1.1)
            main = ~h.roi.isin(UNIONS).to_numpy()
            ax.plot(x[main], h["mean"].to_numpy()[main], "-", color=col, lw=1.0, alpha=0.6)
            ax.plot(x, h["mean"], "o", color=col, mfc=fill, mec=col, ms=4.5, mew=1.1,
                    zorder=4)
    ax.axhline(0, color="0.7", lw=0.6, ls=(0, (3, 2)))
    ax.set_ylabel("Mapping score\n(observed − shuffled)")
    for i, (lab, fill) in enumerate((("Lag 0", "0.35"), ("Neighbouring trials removed",
                                                         "white"))):
        x, y = 0.02 + 0.2 * i, 1.04
        ax.plot(x, y, "o", transform=ax.transAxes, color="0.35", mfc=fill, mec="0.35",
                ms=4, mew=1.1, clip_on=False)
        ax.text(x + 0.025, y, lab, transform=ax.transAxes, fontsize=6.3, va="center",
                color="0.3")
    sns.despine(ax=ax, offset=3, trim=True)
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels(list(pos.keys()), rotation=35, ha="right", rotation_mode="anchor")


def panel_decoder(ax, r):
    pos = xpos()
    g = r[(r.lag == "Lag 0") & (r.template == "bell") & (r.voxels == "all")] \
        .dropna(subset=["dec_acc"])
    x = np.array([pos[q] for q in g.roi])
    ax.bar(x, g.dec_acc * 100, width=0.75, lw=0,
           color=[colour(q) for q in g.roi],
           alpha=1.0)
    ax.axhline(50, color="0.5", lw=0.7, ls=(0, (3, 2)))
    ax.text(max(pos.values()) + 0.5, 50, "Chance", fontsize=6, color="0.45", va="bottom")
    ax.set_xlim(-0.6, max(pos.values()) + 2.2)
    for xi, row in zip(x, g.itertuples()):
        ax.text(xi, row.dec_acc * 100 + 1.2, f"{row.dec_k:.0f}/{row.dec_n:.0f}", ha="center",
                fontsize=5.6, color="0.3")
    ax.set_ylim(40, 72)
    ax.set_yticks([40, 50, 60, 70])
    ax.set_ylabel("Sessions whose mapping\nis read out correctly (%)")
    sns.despine(ax=ax, offset=3, trim=True)
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels(list(pos.keys()), rotation=35, ha="right", rotation_mode="anchor")


SPLITS = (("value", "Value model wins", "o", None),
          ("orientation", "Orientation model wins", "o", "white"),
          ("bell", "Single peak wins", "s", None),
          ("basis", "Value basis wins", "s", "white"))


def panel_splits(ax, r):
    pos = xpos()
    r = r[(r.template == "bell") & (r.lag == "±1 lag")]
    for k, (cls, lab, mk, fill) in enumerate(SPLITS):
        g = r[r.voxels == cls].dropna(subset=["mean"])
        dx = -0.27 + 0.18 * k
        for row in g.itertuples():
            col = colour(row.roi)
            x = pos[row.roi] + dx
            ax.vlines(x, row.mean - row.sem, row.mean + row.sem, color=col, lw=0.9)
            ax.plot(x, row.mean, mk, color=col, mfc=fill or col, mec=col, ms=3.6, mew=1.0,
                    zorder=4)
    ax.axhline(0, color="0.7", lw=0.6, ls=(0, (3, 2)))
    ax.set_ylabel("Mapping score, ±1 lag\n(observed − shuffled)")
    for i, (cls, lab, mk, fill) in enumerate(SPLITS):
        x, y = 0.02 + 0.5 * (i // 2), 1.10 - 0.07 * (i % 2)
        ax.plot(x, y, mk, transform=ax.transAxes, color="0.35", mfc=fill or "0.35",
                mec="0.35", ms=3.6, mew=1.0, clip_on=False)
        ax.text(x + 0.035, y, lab, transform=ax.transAxes, fontsize=6.3, va="center",
                color="0.3")
    sns.despine(ax=ax, offset=3, trim=True)
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels(list(pos.keys()), rotation=35, ha="right", rotation_mode="anchor")


def load_maps(tag):
    files = list((dm.CF / f"peaks_lags-1_target-{tag}").glob("sub-*/sub-*_desc-maps.tsv"))
    return pd.concat(pd.read_csv(f, sep="\t", dtype={"subject": str}) for f in files), len(files)


def main():
    r = pd.read_csv(REPO / "notes/data/cf_scores_by_roi.tsv", sep="\t")
    maps = dm.mapping_table()
    fig = plt.figure(figsize=(7.25, 8.4))

    fc.letter(fig, 0.01, 0.985, "a")
    for k, (lab, tag) in enumerate((("IPS0–5", "IPS0IPS12IPS345"),
                                    ("NPC1–3", "NPC1NPC2NPC3"))):
        d, n = load_maps(tag)
        x0 = 0.09 + 0.45 * k
        cmap = fc.coupling_maps(fig, [[x0, 0.75, 0.17, 0.19], [x0 + 0.195, 0.75, 0.17, 0.19]],
                                d, maps)
        fig.text(x0, 0.975, f"{lab} (n = {n})", fontsize=7.5, fontweight="bold",
                 color=colour(lab), va="top")
    cax = fig.add_axes([0.935, 0.79, 0.012, 0.12])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-2, 2.2), cmap=cmap), cax=cax)
    cb.set_ticks([-2, 0, 2])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=1.5)
    cb.set_label("Coupling (z within row)", fontsize=6)

    fc.letter(fig, 0.01, 0.672, "b")
    ax_b = fig.add_axes([0.09, 0.43, 0.48, 0.19])
    panel_scores(ax_b, r)
    ax_b.set_title("Single-peak template (primary)", loc="left", fontsize=7, pad=22)
    fc.letter(fig, 0.63, 0.672, "c")
    panel_decoder(fig.add_axes([0.71, 0.43, 0.27, 0.19]), r)
    fc.letter(fig, 0.01, 0.352, "d")
    ax_d = fig.add_axes([0.09, 0.08, 0.40, 0.21], sharey=ax_b)
    panel_scores(ax_d, r, "basis")
    ax_d.set_title("Value-basis template", loc="left", fontsize=7, pad=22)
    fc.letter(fig, 0.53, 0.352, "e")
    ax_e = fig.add_axes([0.60, 0.08, 0.38, 0.21])
    panel_splits(ax_e, r)
    ax_e.set_title("Single-peak template, voxels by winning model", loc="left", fontsize=7, pad=22)

    out = REPO / "notes/figures/fig_connectopy_npc.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
