"""Decode the session's orientation -> value mapping from parietal-V1 coupling maps.

peaks.py gives, per participant and session, a coupling map: for parietal
voxels binned by preferred value (2-CHF bins), the voxel-specific connective
field over V1 orientation (inverted von Mises channels, 5-deg grid). If a
voxel preferring value v couples to the V1 population coding the orientation
that is worth v under the CURRENT mapping, the ridge of that map traces the
mapping itself: value on one axis, orientation on the other.

Decoder, fixed before looking: for each mapping m, a template
T_m(v, theta) = exp(kappa cos 2(theta - theta_m(v))), kappa = 4, where
theta_m(v) is the orientation worth v under m; centred over orientation within
each value bin like the data. A session's score is
    r(map, T_this) - r(map, T_other)
(cells weighted by voxel count); > 0 means the map is closer to the mapping
actually in force. Anything shared by both sessions or both templates cancels.

    python -m abstract_values.connective_fields.decode_mapping [--folder peaks_target-...]
    -> notes/figures/fig_cf_decode_mapping.pdf, notes/data/cf_decode_mapping.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
CF = Path("/data/ds-abstractvalue/derivatives/connective_fields")
KAPPA = 4.0
CONDS = ("cdf", "inverse_cdf")
COL = {"cdf": "#3B5BA5", "inverse_cdf": "#B8860B"}
LAB = {"cdf": "CDF mapping", "inverse_cdf": "Inverse-CDF mapping"}


def mapping_table():
    from abstract_values.behavior.data import get_all_behavioral_data
    d = get_all_behavioral_data()
    d = d[d.event_type == "feedback"].reset_index()
    return {m: g.groupby("orientation").value.first().sort_index()
            for m, g in d.groupby("mapping")}


def theta_of_value(mp, v):
    """Orientation worth v under mapping mp (monotonic interpolation)."""
    return np.interp(v, mp.values, mp.index.values)


def template(mp, values, oris):
    th = theta_of_value(mp, values)[:, None]
    T = np.exp(KAPPA * np.cos(np.deg2rad(2 * (oris[None] - th))))
    return T - T.mean(1, keepdims=True)


def wcorr(a, b, w):
    a = a - np.average(a, weights=w)
    b = b - np.average(b, weights=w)
    return np.sum(w * a * b) / np.sqrt(np.sum(w * a * a) * np.sum(w * b * b))


def main(folder):
    maps = mapping_table()
    d = pd.concat(pd.read_csv(f, sep="\t", dtype={"subject": str})
                  for f in (CF / folder).glob("sub-*/sub-*_desc-maps.tsv"))
    values = np.sort(d.value_bin.unique())
    oris = np.sort(d.orientation.unique())
    rows = []
    for (s, c), g in d.groupby(["subject", "condition"]):
        M = g.pivot(index="value_bin", columns="orientation", values="cf").reindex(
            index=values, columns=oris)
        n = g.groupby("value_bin").n.first().reindex(values).fillna(0).to_numpy()
        ok = np.isfinite(M.to_numpy()).all(1) & (n > 0)
        W = np.repeat(n[ok, None], len(oris), 1).ravel()
        r = {m: wcorr(M.to_numpy()[ok].ravel(), template(maps[m], values[ok], oris).ravel(), W)
             for m in CONDS}
        other = [m for m in CONDS if m != c][0]
        rows.append(dict(subject=s, condition=c, r_this=r[c], r_other=r[other],
                         score=r[c] - r[other]))
    res = pd.DataFrame(rows)
    res.to_csv(REPO / "notes/data/cf_decode_mapping.tsv", sep="\t", index=False)
    subj = res.groupby("subject").score.mean()
    t, p = stats.ttest_1samp(subj, 0)
    acc = (res.score > 0).mean()
    pb = stats.binomtest(int((res.score > 0).sum()), len(res), 0.5, alternative="greater").pvalue
    both = (res.groupby("subject").score.apply(lambda x: (x > 0).all())).mean()
    print(f"{folder}: n = {subj.size} participants, {len(res)} sessions")
    print(f"  score {subj.mean():+.4f} ± {subj.sem():.4f}, t = {t:.2f}, p = {p:.4g}")
    print(f"  sessions decoded correctly: {acc:.0%} (binomial p = {pb:.3g}); "
          f"both sessions right: {both:.0%} of participants (chance 25%)")

    plt.rcParams.update({"font.family": "Helvetica", "font.size": 7, "axes.labelsize": 7.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(7.25, 2.9))
    grp = {c: d[d.condition == c].groupby(["value_bin", "orientation"]).cf.mean().unstack()
           .reindex(index=values, columns=oris) for c in CONDS}
    lim = np.nanpercentile(np.abs(np.r_[grp["cdf"].values.ravel(),
                                        grp["inverse_cdf"].values.ravel()]), 98)
    panels = [("cdf", grp["cdf"]), ("inverse_cdf", grp["inverse_cdf"]),
              ("diff", grp["cdf"] - grp["inverse_cdf"])]
    for k, (name, G) in enumerate(panels):
        ax = fig.add_axes([0.06 + 0.24 * k, 0.17, 0.2, 0.68])
        L = lim if name != "diff" else np.nanpercentile(np.abs(G.values), 98)
        ax.pcolormesh(oris, values, G.values, cmap="RdBu_r", vmin=-L, vmax=L,
                      shading="nearest", rasterized=True)
        for m in CONDS:
            vv = np.linspace(2, 42, 200)
            ls = "-" if (name == m or name == "diff") else (0, (2, 1.5))
            ax.plot(theta_of_value(maps[m], vv), vv, color=COL[m], lw=1.4, ls=ls)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xlabel("V1 orientation (°)")
        ax.set_ylabel("Preferred value of parietal voxels (CHF)" if k == 0 else "")
        title = {"cdf": "Sessions with CDF mapping", "inverse_cdf":
                 "Sessions with inverse-CDF mapping", "diff": "Difference (CDF − inverse)"}[name]
        ax.set_title(title, loc="left", fontsize=7, color=COL.get(name, "0.2"))
        sns.despine(ax=ax, offset=3, trim=True)
    roi = "NPC R" if "target" not in folder else "IPS0–5"
    fig.text(0.06, 0.95, f"Coupling of parietal ({roi}) voxels to V1 orientation channels, "
             f"mean over {subj.size} participants; lines: the two mappings "
             "(solid = in force)", fontsize=6.5, color="0.35")
    ax = fig.add_axes([0.86, 0.17, 0.12, 0.68])
    sns.swarmplot(y=subj.values, ax=ax, color="0.4", size=3)
    ax.hlines(subj.mean(), -0.3, 0.3, color="k", lw=1.6)
    ax.vlines(0, subj.mean() - subj.sem(), subj.mean() + subj.sem(), color="k", lw=1.1)
    ax.axhline(0, color="0.7", lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks([])
    ax.set_ylabel("Mapping decoded from coupling\nr(this) − r(other)")
    ax.set_title(f"{acc:.0%} of sessions correct\nt = {t:.2f}, p = {p:.3g}",
                 fontsize=6.5, loc="left")
    sns.despine(ax=ax, offset=3, trim=True, bottom=True)
    out = REPO / "notes/figures" / f"fig_cf_decode_mapping{'' if folder.endswith('IPS345') and 'lags' not in folder else '_' + folder}.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default="peaks_target-IPS0IPS12IPS345")
    a = ap.parse_args()
    main(a.folder)
