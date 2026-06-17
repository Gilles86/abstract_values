#!/usr/bin/env python3
"""
NPCr expected decoding uncertainty as a function of ORIENTATION (not value).

The EU is computed in value (CHF) space per condition; here we remap each
condition's value grid back to the gabor orientation that produced it
(value = mapping(orientation)), so both conditions share the physical
orientation x-axis (0-180 deg) — matching how `behavior_overview.ipynb`
plots quantities against orientation, and letting the two mappings be
compared at the same orientation.

Reads the per-(subject, condition) sidecar written by
`expected_uncertainty_per_condition.py` (columns include value, sd_E,
condition, subject). The value->orientation map per condition comes from
the behavioural (orientation, value) gabor pairs.

Usage:
  python -m abstract_values.visualize.expected_uncertainty_vs_orientation \
      --tsv notes/data/eu_fwhmshift_n12.tsv --out notes/figures/eu_fwhmshift_vs_orientation.pdf
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

from abstract_values.behavior.data import get_all_behavioral_data

mpl.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.dpi": 300})
PALETTE = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}
LABEL = {"cdf": "CDF", "inverse_cdf": "Inverse CDF"}


def behavioral_sd_by_orientation():
    """Per (subject, condition, orientation) SD of the BDM bid — the
    behavioural analogue of the neural decoded uncertainty (same CHF units)."""
    b = get_all_behavioral_data()
    b = b[b["event_type"] == "feedback"].reset_index()
    b["response"] = pd.to_numeric(b["response"], errors="coerce")
    b = b.dropna(subset=["response"])
    g = (b.groupby(["subject", "mapping", "orientation"])["response"].std()
         .reset_index().rename(columns={"mapping": "condition",
                                        "response": "beh_sd"}))
    return g


def value_to_orientation_map():
    """{condition: DataFrame(value, orientation)} from the gabor pairs."""
    b = get_all_behavioral_data()
    b = b[b["event_type"] == "feedback"].reset_index()
    b = b[["mapping", "orientation", "value"]].dropna().drop_duplicates()
    out = {}
    for cond, g in b.groupby("mapping"):
        m = (g.groupby("orientation", as_index=False)["value"].mean())
        m["vkey"] = m["value"].round(1)
        out[cond] = m
    return out


def run(tsv, out):
    eu = pd.read_csv(tsv, sep="\t")
    eu = eu[["subject", "condition", "value", "sd_E"]].copy()
    eu["vkey"] = eu["value"].round(1)
    vmap = value_to_orientation_map()

    rows = []
    for cond, m in vmap.items():
        sub = eu[eu["condition"] == cond].merge(
            m[["vkey", "orientation"]], on="vkey", how="inner")
        rows.append(sub)
    df = pd.concat(rows, ignore_index=True)
    n_sub = df["subject"].nunique()
    print(f"{n_sub} subjects · {len(df)} (subject,orientation,condition) points")

    beh = behavioral_sd_by_orientation()

    with sns.plotting_context("talk"), sns.axes_style("ticks"):
        fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
        for cond in ("cdf", "inverse_cdf"):
            d = df[df["condition"] == cond]
            sns.lineplot(data=d, x="orientation", y="sd_E", color=PALETTE[cond],
                         errorbar=("se", 1), marker="o", ms=4, ax=ax,
                         label=f"{LABEL[cond]} — NPCr decoded")
            bd = beh[beh["condition"] == cond]
            sns.lineplot(data=bd, x="orientation", y="beh_sd", color=PALETTE[cond],
                         errorbar=("se", 1), marker="s", ms=4, ls="--", ax=ax,
                         label=f"{LABEL[cond]} — behavioral bid")
        ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xlabel("Orientation (deg)")
        ax.set_ylabel("Uncertainty about value √Var (CHF)")
        ax.set_title(f"Value uncertainty vs orientation: neural vs behavioral "
                     f"(n={n_sub})", fontsize=10)
        ax.legend(frameon=False, fontsize=7.5, ncol=2)
        sns.despine(ax=ax, offset=4, trim=True)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tsv", default="notes/data/eu_fwhmshift_n12.tsv")
    p.add_argument("--out", default="notes/figures/eu_fwhmshift_vs_orientation.pdf")
    args = p.parse_args()
    run(args.tsv, args.out)


if __name__ == "__main__":
    main()
