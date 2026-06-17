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
from matplotlib.backends.backend_pdf import PdfPages

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
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    def _page(data, ycol, ylab, title):
        with sns.plotting_context("talk"), sns.axes_style("ticks"):
            fig, ax = plt.subplots(figsize=(5.4, 3.9), constrained_layout=True)
            for cond in ("cdf", "inverse_cdf"):
                sns.lineplot(data=data[data["condition"] == cond],
                             x="orientation", y=ycol, color=PALETTE[cond],
                             errorbar=("se", 1), marker="o", ms=4,
                             label=LABEL[cond], ax=ax)
            ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_xlabel("Orientation (deg)"); ax.set_ylabel(ylab)
            ax.set_title(title, fontsize=11)
            ax.legend(frameon=False, fontsize=9)
            sns.despine(ax=ax, offset=4, trim=True)
            return fig

    with PdfPages(out) as pdf:
        f = _page(df, "sd_E", "NPCr decoded value uncertainty √Var (CHF)",
                  f"NPCr decoded VALUE uncertainty vs orientation (n={n_sub})")
        pdf.savefig(f, bbox_inches="tight"); plt.close(f)
        f = _page(beh, "beh_sd", "Behavioral bid SD (CHF)",
                  f"Behavioral bid variability vs orientation "
                  f"(n={beh['subject'].nunique()})")
        pdf.savefig(f, bbox_inches="tight"); plt.close(f)
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
