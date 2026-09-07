"""Decoded vs true, and absolute error across the stimulus range.

Two rows -- orientation decoded from V1, value decoded from NPCr -- each with
the pooled trial scatter and the error profile across the stimulus axis, split
by mapping condition.

Splitting the error profile by condition is the point of the right-hand column.
The two sessions warp the orientation->value mapping differently (both
monotone, Pearson 0.93 -- not inversions). A representation genuinely in value
space should show the same error profile in both; one that tracks orientation
should show profiles that differ where the two mappings disagree, which is the
mid-range and the low end rather than a mirror image.

Reads the per-trial dump from ``decoding_quality_scatter --trials-tsv``.

    python -m abstract_values.visualize.plot_decoding_scatter_and_error
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "lines.linewidth": 1.2, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})

COND_COLOUR = {"cdf": "#2A6F97", "inverse_cdf": "#C4442B"}
COND_LABEL = {"cdf": "CDF", "inverse_cdf": "Inverse CDF"}


def load(trials_tsv, key_tsv):
    df = pd.read_csv(trials_tsv, sep="\t")
    key = pd.read_csv(key_tsv, sep="\t")
    # The key is numeric; the decoding dump keeps the pilots' own labels, and
    # sub-pil01/02 ARE subjects 1 and 2 (see CLAUDE.md), so map them across.
    df["subject_num"] = (df["subject"].astype(str)
                         .str.replace("pil", "", regex=False).astype(int))
    df = df.merge(key.rename(columns={"subject": "subject_num"}),
                  on=["subject_num", "session"], how="left")
    ori = df["quantity"] == "gabor"
    df.loc[ori, ["true", "decoded"]] = np.rad2deg(
        df.loc[ori, ["true", "decoded"]].to_numpy())
    err = df["decoded"] - df["true"]
    # Orientation is pi-periodic: wrap the error onto [-90, 90).
    err[ori] = (err[ori] + 90) % 180 - 90
    df["error"] = err
    df["abs_error"] = err.abs()
    return df


def scatter(ax, d, lim, ticks, label, wrap=None):
    ax.scatter(d["true"], d["decoded"], s=2, alpha=.06, color="0.25",
               linewidths=0, rasterized=True)
    ax.plot(lim, lim, color="#E9C46A", lw=1.2, zorder=3)
    if wrap:                       # the two wrapped diagonals, faint
        for off in (-wrap, wrap):
            ax.plot(lim, [lim[0] + off, lim[1] + off], color="#E9C46A",
                    lw=0.8, ls=(0, (3, 3)), alpha=.7, zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Decoded {label}")


def error_profile(ax, d, label, chance, n_bins=None):
    """Mean |error| across the stimulus range, one line per condition,
    averaged over subjects (mean +/- SEM across subjects, not trials)."""
    d = d.copy()
    if n_bins:
        d["bin"] = pd.cut(d["true"], n_bins).apply(lambda i: i.mid).astype(float)
    else:
        d["bin"] = d["true"].round(1)
    per_sub = (d.groupby(["mapping", "bin", "subject"], observed=True)["abs_error"]
                 .mean().reset_index())
    for cond, g in per_sub.groupby("mapping"):
        m = g.groupby("bin")["abs_error"].agg(["mean", "sem"]).reset_index()
        ax.plot(m["bin"], m["mean"], color=COND_COLOUR[cond], lw=1.4)
        ax.fill_between(m["bin"], m["mean"] - m["sem"], m["mean"] + m["sem"],
                        color=COND_COLOUR[cond], alpha=.20, lw=0)
        ax.annotate(COND_LABEL[cond], (m["bin"].iloc[-1], m["mean"].iloc[-1]),
                    xytext=(3, 0), textcoords="offset points", fontsize=6.5,
                    color=COND_COLOUR[cond], va="center")
    ax.axhline(chance, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.annotate("Chance", (ax.get_xlim()[0], chance), xytext=(2, 2),
                textcoords="offset points", fontsize=6.5, color="0.45")
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel("Absolute error")


def main(trials_tsv, key_tsv, out):
    df = load(trials_tsv, key_tsv)
    ori = df[(df["quantity"] == "gabor") & (df["roi"] == "BensonV1")]
    val = df[df["quantity"] == "value-weighted"]

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.4), constrained_layout=True)

    scatter(axes[0, 0], ori, (0, 180), [0, 45, 90, 135, 180],
            "orientation (deg)", wrap=180)
    axes[0, 0].set_title("Orientation · V1", fontsize=8, color="0.2")
    error_profile(axes[0, 1], ori, "orientation (deg)", 45.0)
    axes[0, 1].set_ylabel("Absolute error (deg)")
    axes[0, 1].set_xticks([0, 45, 90, 135, 180])
    axes[0, 1].set_ylim(0, 50)

    lim = (0, 45)
    scatter(axes[1, 0], val, lim, [0, 15, 30, 45], "value (CHF)")
    axes[1, 0].set_title("Value · NPCr", fontsize=8, color="0.2")
    rng = np.random.default_rng(0)
    v = val["true"].to_numpy()
    cmae = float(np.abs(rng.choice(v, 20000) - rng.choice(v, 20000)).mean())
    error_profile(axes[1, 1], val, "value (CHF)", cmae, n_bins=8)
    axes[1, 1].set_ylabel("Absolute error (CHF)")
    axes[1, 1].set_ylim(0, cmae * 1.25)

    sns.despine(fig=fig, offset=4, trim=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")
    for name, d in [("orientation V1", ori), ("value NPCr", val)]:
        s = (d.groupby(["mapping", "subject"])["abs_error"].mean()
              .groupby("mapping").agg(["mean", "sem"]))
        print(f"\n{name}: mean |error| by condition\n{s.round(3).to_string()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", default="notes/data/decoding_trials_alpha10.tsv")
    p.add_argument("--key", default="notes/data/session_mapping_key.tsv")
    p.add_argument("--out", default="notes/figures/decoding_scatter_error.pdf")
    a = p.parse_args()
    main(Path(a.trials), Path(a.key), Path(a.out))
