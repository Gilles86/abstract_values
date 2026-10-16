"""Does each region's geometry follow orientation or value? (crossnobis RDMs)

The design dissociates the two within every participant:
  * orientation distance is circular and identical in both sessions
    (7.5 and 172.5 deg are neighbours);
  * value distance |v_i - v_j| is not circular (7.5 and 172.5 deg sit at
    opposite ends of the CHF scale) and it CHANGES between sessions, because
    the two mappings bend the orientation -> CHF map differently.

Per participant and region, the 2 x 253 pairwise crossnobis distances (both
sessions) are regressed on the two model distances (each scaled to unit
max) plus a session intercept. Stated before looking:
  * prediction: the value weight grows and the orientation weight shrinks
    from V1 to IPS;
  * group test per region: one-sample t over participants;
  * gradient test: (value - orientation weight), parietal (IPS0, IPS1-2,
    IPS3-5) minus early (V1-V3), paired over participants.

Two checks added after the first look (2026-10-16), both reported:
  * three-predictor model: circular orientation, LINEAR orientation
    (|theta_i - theta_j|, non-circular) and value. Within a session value
    distance correlates 0.94 with linear orientation distance (both mappings
    rise monotonically with orientation), so only the between-session change
    separates "value" from "an ordered orientation axis";
  * mapping-change test: (d_cdf - d_inverse_cdf) regressed on
    (|dv|_cdf - |dv|_inverse_cdf) per participant and region, with linear and
    circular orientation distance as covariates (a session-wise gain change
    times the fixed geometry would otherwise masquerade as a mapping change;
    see power_mapping_change.change_slope). Only re-scaling to the session's
    CHF values survives.

    python -m abstract_values.geometry.analyze_crossnobis
    -> notes/figures/fig_crossnobis_geometry.pdf, notes/data/crossnobis_weights.tsv
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from abstract_values.geometry.power_mapping_change import change_slope

REPO = Path(__file__).resolve().parents[2]
DIR = Path("/data/ds-abstractvalue/derivatives/geometry/crossnobis")
ROI_LAB = {"BensonV1": "V1", "BensonV2": "V2", "BensonV3": "V3", "V3ab": "V3a/b",
           "IPS0": "IPS0", "IPS12": "IPS1–2", "IPS345": "IPS3–5", "NPCr": "NPC R"}
DORSAL = ["V1", "V2", "V3", "V3a/b", "IPS0", "IPS1–2", "IPS3–5"]
EARLY, LATE = DORSAL[:3], DORSAL[4:]
ORI_C, VAL_C = "#1B8A80", "#D9730D"


def load():
    d = pd.concat(pd.read_csv(f, sep="\t", dtype={"subject": str})
                  for f in DIR.glob("sub-*/sub-*_desc-crossnobis_rdm.tsv"))
    d["roi"] = d.roi.map(ROI_LAB)
    dt = np.abs(d.ori_i - d.ori_j)
    d["ori_dist"] = np.minimum(dt, 180 - dt) / 90
    d["val_dist"] = np.abs(d.val_i - d.val_j) / 40
    return d


def weights(d):
    rows = []
    for (s, r), g in d.groupby(["subject", "roi"]):
        X = np.column_stack([g.ori_dist, g.val_dist,
                             (g.session == g.session.min()).astype(float),
                             (g.session != g.session.min()).astype(float)])
        b = np.linalg.lstsq(X, g.d.to_numpy(), rcond=None)[0]
        rows.append(dict(subject=s, roi=r, w_ori=b[0], w_val=b[1],
                         mean_d=g.d.mean(), n_voxels=g.n_voxels.iloc[0]))
    return pd.DataFrame(rows)


def main():
    d = load()
    w = weights(d)
    w.to_csv(REPO / "notes/data/crossnobis_weights.tsv", sep="\t", index=False)
    n = w.subject.nunique()
    summ = w.groupby("roi")[["w_ori", "w_val", "mean_d"]].agg(["mean", "sem"]).reindex(
        [r for r in ROI_LAB.values() if r in w.roi.unique()])
    print(summ.round(4).to_string())
    for r in DORSAL:
        g = w[w.roi == r]
        if len(g):
            print(f"{r:7s} ori t={stats.ttest_1samp(g.w_ori, 0)[0]:+.2f}  "
                  f"val t={stats.ttest_1samp(g.w_val, 0)[0]:+.2f}")
    piv = w.assign(diff=w.w_val - w.w_ori).pivot(index="subject", columns="roi",
                                                  values="diff")
    grad = piv[LATE].mean(1) - piv[EARLY].mean(1)
    t, p = stats.ttest_1samp(grad.dropna(), 0)
    print(f"Gradient (value - orientation weight, parietal - early): "
          f"{grad.mean():+.4f} ± {grad.sem():.4f}, t = {t:.2f}, p = {p:.4f}, n = {grad.notna().sum()}")

    d["lin_dist"] = np.abs(d.ori_i - d.ori_j) / 165
    rows = []
    for (s_, r), g in d.groupby(["subject", "roi"]):
        X = np.column_stack([g.ori_dist, g.lin_dist, g.val_dist,
                             g.session == g.session.min(),
                             g.session != g.session.min()]).astype(float)
        b = np.linalg.lstsq(X, g.d.to_numpy(), rcond=None)[0]
        rows.append(dict(subject=s_, roi=r, circ=b[0], lin=b[1], val=b[2]))
    w3 = pd.DataFrame(rows)
    k = ["subject", "roi", "ori_i", "ori_j"]
    j = (d[d.mapping == "cdf"].set_index(k)[["d", "val_dist"]]
         .join(d[d.mapping == "inverse_cdf"].set_index(k)[["d", "val_dist"]],
               lsuffix="_c", rsuffix="_i").dropna().reset_index())
    mc = (j.groupby(["subject", "roi"])
          .apply(lambda g: change_slope((g.d_c - g.d_i).to_numpy(),
                                        (g.val_dist_c - g.val_dist_i).to_numpy(),
                                        g.ori_i.to_numpy(), g.ori_j.to_numpy()),
                 include_groups=False).rename("slope").reset_index())
    for r in DORSAL:
        g, h = w3[w3.roi == r], mc[mc.roi == r].slope
        print(f"{r:7s} " + "  ".join(f"{c} t={stats.ttest_1samp(g[c], 0)[0]:+.2f}"
                                     for c in ("circ", "lin", "val"))
              + f"   mapping-change slope {h.mean():+.4f} ± {h.sem():.4f}")
    w3.merge(mc, on=["subject", "roi"]).to_csv(
        REPO / "notes/data/crossnobis_three_models.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.6), constrained_layout=True)
    x = np.arange(len(DORSAL))
    for ax, col, key, lab in ((axes[0], ORI_C, "w_ori", "Orientation geometry"),
                              (axes[1], VAL_C, "w_val", "Value geometry")):
        m = summ[(key, "mean")].reindex(DORSAL).to_numpy()
        se = summ[(key, "sem")].reindex(DORSAL).to_numpy()
        ax.fill_between(x, m - se, m + se, color=col, alpha=0.18, lw=0)
        ax.plot(x, m, "-o", color=col, lw=1.8, ms=4.5, mec="white", mew=0.8)
        ax.axhline(0, color="0.7", lw=0.6, ls=(0, (3, 2)))
        ax.set_xticks(x)
        ax.set_xticklabels(DORSAL)
        ax.set_title(lab, loc="left", color=col, fontweight="bold", fontsize=8)
        ax.set_ylabel("Weight (crossnobis per voxel)")
        sns.despine(ax=ax, offset=4, trim=True)
    fig.suptitle(f"n = {n}; gradient {grad.mean():+.3f} ± {grad.sem():.3f}, p = {p:.3g}",
                 fontsize=7, x=0.01, ha="left")
    out = REPO / "notes/figures/fig_crossnobis_geometry.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
