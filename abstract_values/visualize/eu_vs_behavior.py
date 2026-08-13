"""Neural expected-uncertainty (σ_E, NPCr) vs behavioral response noise.

Two questions this answers:

1. As a function of stimulus value, does the *shape* of neural σ_E track
   the shape of behavioral response SD? Page 1 plots both (+ stimulus
   density) per mapping condition (cdf / inverse_cdf) so the curves can be
   compared directly.

   Caveat carried over from ``npcr_uncertainty_vs_value.py`` (same EU
   pipeline): raw σ_E is dominated by *regression-to-mode bias variance* —
   near a stimulus-density peak, the Bayesian decoder's posterior mean gets
   pulled toward the mode, which can *inflate* σ_E right where the
   population code is actually most discriminable. The project's existing
   fix is to test the **1/σ_E** (∝ √Fisher information ∝ density under
   efficient coding) against density, not raw σ_E. Page 1 shows both raw
   σ_E and 1/σ_E so the flip is visible directly.

2. Across subjects, does behavioral precision (−SD of bid error) predict
   neural precision (−mean σ_E in NPCr)? Page 2 is the brain-behavior
   scatter, per condition and pooled.

Data sources
------------
Behavioral: ``abstract_values.behavior.data.get_all_behavioral_data()``,
feedback rows, ``error = response - value`` (BDM is truth-telling, so
``value`` IS the rational bid — see CLAUDE.md).

Neural: the sidecar TSV written by ``expected_uncertainty_per_condition.py``
(``--out``'s ``.tsv`` companion), columns ``subject, session, condition,
value, sd_E, variant``. Pass the current one with ``--eu-tsv``.

Usage:
    python -m abstract_values.visualize.eu_vs_behavior \\
        --eu-tsv notes/figures/expected_uncertainty_per_condition.tsv
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
from scipy import stats

from abstract_values.behavior.data import get_all_behavioral_data
from abstract_values.visualize.npcr_uncertainty_vs_value import _aggregate

DEFAULT_OUT = "notes/figures/eu_vs_behavior.pdf"
COL_BEH = "#3B5BA5"   # blue — behavior
COL_NEU = "#C44E52"   # red — neural
COL_DENS = "#9C9C9C"  # gray — stimulus density
CONDITIONS = ["cdf", "inverse_cdf"]

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_behavior() -> pd.DataFrame:
    df = get_all_behavioral_data()
    df = df[df["event_type"] == "feedback"].copy()
    df["response"] = pd.to_numeric(df["response"], errors="coerce")
    df = df.reset_index().dropna(subset=["response"])
    df["error"] = df["response"] - df["value"]
    # Zero-padded string subject id, matching the neural TSV's convention.
    df["subject"] = df["subject"].apply(lambda s: f"{int(s):02d}")
    return df


def page_value_profiles(beh: pd.DataFrame, eu: pd.DataFrame, variant: str, pdf: PdfPages):
    grid = np.linspace(beh["value"].min(), beh["value"].max(), 80)
    eu = eu[eu["variant"] == variant]

    fig, axes = plt.subplots(3, 2, figsize=(9.5, 8.2), constrained_layout=True,
                             sharex=True, sharey="row")
    for col, mapping in enumerate(CONDITIONS):
        b = (beh[beh.mapping == mapping]
             .groupby(["subject", "value"])["response"].std()
             .reset_index(name="y"))
        n_ = eu[eu.condition == mapping][["subject", "value", "sd_E"]].rename(columns={"sd_E": "y"})
        n_inv = n_.assign(y=1.0 / n_["y"])
        dens = (beh[beh.mapping == mapping].groupby(["subject", "value"])
                .size().reset_index(name="y"))

        b_med, b_q25, b_q75, _, b_n = _aggregate(b, "value", "y", grid)
        n_med, n_q25, n_q75, _, n_n = _aggregate(n_, "value", "y", grid)
        ninv_med, ninv_q25, ninv_q75, _, _ = _aggregate(n_inv, "value", "y", grid)
        d_med, d_q25, d_q75, _, _ = _aggregate(dens, "value", "y", grid)

        ax = axes[0, col]
        ax.plot(grid, b_med, color=COL_BEH, lw=1.4)
        ax.fill_between(grid, b_q25, b_q75, color=COL_BEH, alpha=0.2, lw=0)
        title = "CDF mapping" if mapping == "cdf" else "Inverse-CDF mapping"
        ax.set_title(title)
        if col == 0:
            ax.set_ylabel("Behavioral SD (CHF)")
        ax.text(0.03, 0.93, f"n={b_n} subj", transform=ax.transAxes,
                fontsize=7.5, va="top", color=COL_BEH)

        ax = axes[1, col]
        ax.plot(grid, n_med, color=COL_NEU, lw=1.4, label="Raw σ_E")
        ax.fill_between(grid, n_q25, n_q75, color=COL_NEU, alpha=0.2, lw=0)
        if col == 0:
            ax.set_ylabel("Neural σ_E (CHF)")
        # Correlate the two group-median profiles directly (n = grid points
        # with both curves defined) — the plain-language answer to "do
        # these two curves move together or oppositely?"
        valid = np.isfinite(b_med) & np.isfinite(n_med)
        r, p = stats.spearmanr(b_med[valid], n_med[valid])
        ax.text(0.03, 0.93, f"vs behavioral SD: r={r:+.2f}, p={p:.3f}",
                transform=ax.transAxes, fontsize=7.5, va="top", color="0.15")

        ax = axes[2, col]
        ax.plot(grid, d_med, color=COL_DENS, lw=1.4)
        ax.fill_between(grid, d_q25, d_q75, color=COL_DENS, alpha=0.25, lw=0)
        if col == 0:
            ax.set_ylabel("Stimulus density\n(trials / subject / value)")
        ax.set_xlabel("Value (CHF)")
        valid = np.isfinite(ninv_med) & np.isfinite(d_med)
        r_eff, p_eff = stats.spearmanr(ninv_med[valid], d_med[valid])
        ax.text(0.03, 0.93,
                f"1/σ_E vs density: r={r_eff:+.2f}, p={p_eff:.3f}\n"
                f"(efficient coding predicts positive)",
                transform=ax.transAxes, fontsize=7.5, va="top", color="0.15")

    fig.suptitle(
        f"Behavioral SD vs neural σ_E vs stimulus value  ({variant})\n"
        "Row 3's r tests raw σ_E's own known bias (regression-to-mode "
        "near density peaks can inflate σ_E there — see script docstring); "
        "row 2's r is the direct behavioral-vs-neural comparison.",
        fontsize=9, color="0.15")
    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_brain_behavior(beh: pd.DataFrame, eu: pd.DataFrame, variant: str, pdf: PdfPages):
    eu = eu[eu["variant"] == variant]

    beh_prec = ((-beh.groupby(["subject", "mapping"])["error"].std())
                .reset_index(name="beh_precision"))
    neu_prec = ((-eu.groupby(["subject", "condition"])["sd_E"].mean())
                .reset_index(name="neu_precision")
                .rename(columns={"condition": "mapping"}))
    merged = beh_prec.merge(neu_prec, on=["subject", "mapping"])

    pooled = (merged.groupby("subject")[["beh_precision", "neu_precision"]]
              .mean().reset_index())
    pooled["mapping"] = "pooled (both mappings)"

    panels = CONDITIONS + ["pooled (both mappings)"]
    plot_df = pd.concat([merged, pooled], ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True,
                             sharey=True)
    for ax, mapping in zip(axes, panels):
        d = plot_df[plot_df.mapping == mapping].dropna()
        ax.scatter(d["beh_precision"], d["neu_precision"], s=22,
                   color=COL_BEH, alpha=0.75, edgecolor="white", linewidth=0.4)
        if len(d) >= 3:
            r, p = stats.pearsonr(d["beh_precision"], d["neu_precision"])
            b1, b0 = np.polyfit(d["beh_precision"], d["neu_precision"], 1)
            xx = np.linspace(d["beh_precision"].min(), d["beh_precision"].max(), 50)
            ax.plot(xx, b0 + b1 * xx, color="0.2", lw=1.1, zorder=0)
            ax.text(0.04, 0.96, f"n={len(d)}\nr={r:+.2f}, p={p:.3f}",
                    transform=ax.transAxes, va="top", fontsize=8.5)
        title = mapping if mapping == "pooled (both mappings)" else (
            "CDF mapping" if mapping == "cdf" else "Inverse-CDF mapping")
        ax.set_title(title)
        ax.set_xlabel("Behavioral precision\n(−SD of bid error, CHF)")
    axes[0].set_ylabel("Neural precision\n(−mean σ_E, NPCr, CHF)")
    fig.suptitle(
        "Brain-behavior correlation: are behaviorally-precise subjects "
        f"also neurally precise in NPCr?  ({variant})",
        fontsize=9.5, color="0.15")
    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return plot_df


def main(eu_tsv: Path, out: Path, variants: tuple[str, ...]):
    beh = load_behavior()
    eu = pd.read_csv(eu_tsv, sep="\t")
    eu["subject"] = eu["subject"].astype(str)

    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        for variant in variants:
            page_value_profiles(beh, eu, variant, pdf)
        corr_df = None
        for variant in variants:
            corr_df = page_brain_behavior(beh, eu, variant, pdf)
    if corr_df is not None:
        tsv = out.with_suffix(".tsv")
        corr_df.to_csv(tsv, sep="\t", index=False)
        print(f"Sidecar: {tsv}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eu-tsv", required=True,
                   help="Sidecar TSV from expected_uncertainty_per_condition.py")
    p.add_argument("--variants", nargs="+", default=["unsmoothed"],
                   choices=["unsmoothed", "smoothed"])
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()
    main(Path(args.eu_tsv), Path(args.out), tuple(args.variants))
