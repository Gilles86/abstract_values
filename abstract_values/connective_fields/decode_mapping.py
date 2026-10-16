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
COL = {"cdf": "#5B4A9E", "inverse_cdf": "#B23A6E"}
LAB = {"cdf": "CDF mapping", "inverse_cdf": "inverse-CDF mapping"}


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
    res.to_csv(REPO / f"notes/data/cf_decode_mapping_{folder}.tsv", sep="\t", index=False)
    subj = res.groupby("subject").score.mean()
    t, p = stats.ttest_1samp(subj, 0)
    acc = (res.score > 0).mean()
    pb = stats.binomtest(int((res.score > 0).sum()), len(res), 0.5, alternative="greater").pvalue
    both = (res.groupby("subject").score.apply(lambda x: (x > 0).all())).mean()
    print(f"{folder}: n = {subj.size} participants, {len(res)} sessions")
    print(f"  score {subj.mean():+.4f} ± {subj.sem():.4f}, t = {t:.2f}, p = {p:.4g}")
    print(f"  sessions decoded correctly: {acc:.0%} (binomial p = {pb:.3g}); "
          f"both sessions right: {both:.0%} of participants (chance 25%)")

    figure(d, maps, values, oris, subj, t, p, acc, folder)
    return res


def aligned(d, maps, oris, ref):
    """Per participant: coupling profile re-centred on the orientation worth each
    value bin's value under the mapping in force (ref='this') or the other one."""
    step = oris[1] - oris[0]
    rel = (np.arange(len(oris)) - len(oris) // 2) * step
    out = {}
    for s, gs in d.groupby("subject"):
        acc, wsum = np.zeros(len(oris)), 0.0
        for (c, v), g in gs.groupby(["condition", "value_bin"]):
            m = c if ref == "this" else [x for x in CONDS if x != c][0]
            prof = g.set_index("orientation").cf.reindex(oris).to_numpy()
            n = g.n.iloc[0]
            if not np.isfinite(prof).all() or n == 0:
                continue
            k = int(np.round(theta_of_value(maps[m], v) / step)) % len(oris)
            acc += n * np.roll(prof, len(oris) // 2 - k)
            wsum += n
        out[s] = acc / wsum
    return rel, pd.DataFrame(out).T


def smooth_rows(G, sv=1.0, so=2.0):
    """Display only: Gaussian smoothing over value bins and (circularly) orientation."""
    from scipy.ndimage import gaussian_filter
    X = np.nan_to_num(G.to_numpy())
    return gaussian_filter(X, (sv, so), mode=("nearest", "wrap"))


def figure(d, maps, values, oris, subj, t, p, acc, folder):
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 7, "axes.labelsize": 7.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    roi = {"": "NPC R"}.get("target" in folder and "x" or "", None) or (
        folder.split("target-")[1].split("_")[0].replace("IPS0IPS12IPS345", "IPS0–5")
        if "target" in folder else "NPC R")
    fig = plt.figure(figsize=(7.25, 4.6))
    grp = {c: d[d.condition == c].groupby(["value_bin", "orientation"]).cf.mean().unstack()
           .reindex(index=values, columns=oris) for c in CONDS}
    S = {c: smooth_rows(grp[c]) for c in CONDS}
    S = {c: (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True) for c, X in S.items()}
    cmap = sns.color_palette("rocket", as_cmap=True)
    vv = np.linspace(2, 42, 200)
    for k, c in enumerate(CONDS):
        ax = fig.add_axes([0.07 + 0.31 * k, 0.56, 0.25, 0.36])
        ax.pcolormesh(oris, values, S[c], cmap=cmap, vmin=-2, vmax=2.2, shading="gouraud",
                      rasterized=True)
        other = [m for m in CONDS if m != c][0]
        ax.plot(theta_of_value(maps[other], vv), vv, color="#BBBBBB", lw=1.0, ls=(0, (2, 1.5)))
        ax.plot(theta_of_value(maps[c], vv), vv, color="#7FDBFF", lw=1.6)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xlim(0, 180)
        ax.set_xlabel("V1 orientation channel (°)")
        ax.set_ylabel("Preferred value of\nparietal voxels (CHF)" if k == 0 else "")
        ax.set_title(f"Sessions with the {LAB[c]}", loc="left", fontsize=7.5,
                     color="0.15")
        sns.despine(ax=ax, offset=3, trim=True)
    fig.text(0.07, 0.965, "a", fontsize=8, fontweight="bold")
    fig.text(0.38, 0.965, "b", fontsize=8, fontweight="bold")
    # key for the overlays
    kx = fig.add_axes([0.70, 0.62, 0.28, 0.25])
    kx.axis("off")
    kx.plot([0, 0.12], [0.85, 0.85], color="#7FDBFF", lw=1.6)
    kx.text(0.16, 0.85, "Mapping in force", va="center", fontsize=6.5)
    kx.plot([0, 0.12], [0.68, 0.68], color="#BBBBBB", lw=1.0, ls=(0, (2, 1.5)))
    kx.text(0.16, 0.68, "The other session's mapping", va="center", fontsize=6.5)
    kx.text(0, 0.45, f"Coupling of {roi} voxels' trial-to-trial\nfluctuations to V1 orientation "
            f"channels,\nstimulus means removed; mean of {subj.size}\nparticipants, each row "
            "z-scored (display only)", fontsize=6, color="0.35", va="top")
    kx.set_xlim(0, 1)
    kx.set_ylim(0, 1)
    cax = fig.add_axes([0.70, 0.57, 0.15, 0.015])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-2, 2.2), cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.set_ticks([-2, 0, 2])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=1.5)
    cb.set_label("Coupling (z within row)", fontsize=6)
    # c: aligned profiles
    ax = fig.add_axes([0.115, 0.09, 0.40, 0.33])
    for ref, col, lab in (("this", "#D9730D", "Centred on the mapping\nin force"),
                          ("other", "0.55", "Centred on the\nother mapping")):
        rel, P = aligned(d, maps, oris, ref)
        m, se = P.mean(0).to_numpy(), P.sem(0).to_numpy()
        ax.fill_between(rel, m - se, m + se, color=col, alpha=0.2, lw=0)
        ax.plot(rel, m, color=col, lw=1.6)
        i = int(np.argmax(m))
        if ref == "this":
            ax.text(rel[i] + 14, m[i], lab, color=col, fontsize=6.5, va="center")
        else:
            ax.text(-86, 0.6 * m[i], lab, color=col, fontsize=6.5, va="center")
    ax.axvline(0, color="0.75", lw=0.6, ls=(0, (3, 2)))
    ax.axhline(0, color="0.75", lw=0.6)
    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_xlim(-90, 90)
    ax.set_xlabel("V1 orientation relative to the orientation worth the voxel's preferred value (°)")
    ax.set_ylabel("Coupling\n(voxel-specific CF)")
    sns.despine(ax=ax, offset=3, trim=True)
    fig.text(0.01, 0.45, "c", fontsize=8, fontweight="bold")
    # d: decoder
    ax = fig.add_axes([0.66, 0.09, 0.12, 0.33])
    sns.swarmplot(y=subj.values, ax=ax, color="0.45", size=2.8)
    ax.hlines(subj.mean(), -0.3, 0.3, color="#D9730D", lw=2)
    ax.vlines(0, subj.mean() - subj.sem(), subj.mean() + subj.sem(), color="#D9730D", lw=1.2)
    ax.axhline(0, color="0.7", lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks([])
    ax.set_ylabel("Mapping read out from coupling\nr(in force) − r(other)")
    sns.despine(ax=ax, offset=3, trim=True, bottom=True)
    fig.text(0.80, 0.30, f"{acc:.0%} of sessions\nclassified correctly\n\n"
             f"t{subj.size - 1} = {t:.2f}\np = {p:.3g}", fontsize=6.5, color="0.25", va="top")
    fig.text(0.58, 0.45, "d", fontsize=8, fontweight="bold")
    out = REPO / "notes/figures" / f"fig_cf_decode_mapping_{folder}.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default="peaks_target-IPS0IPS12IPS345")
    a = ap.parse_args()
    main(a.folder)
