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
   neural precision in NPCr? Tried with three neural measures, since raw
   posterior SD (σ_E) "tends to not work great" (regression-to-mode bias,
   see above): −mean σ_E, −mean |decoding error| (same simulated-decoding
   runs, bias+noise instead of dispersion alone), and the empirical
   decoded-vs-true Pearson r from REAL single-trial decoding
   (``decode_value.py`` output — no simulation involved). One
   brain-behavior scatter page per measure.

3. Neural σ_E and behavioral SD live on very different scales (σ_E is
   several-fold wider — compare page 1's row 1 vs row 2 y-axes), which
   confounds a raw-scale between-subject correlation. The condition-
   difference page instead correlates the *within-subject* cdf −
   inverse_cdf contrast for behavior against the same contrast for each
   neural measure — removes each subject's idiosyncratic scale/offset,
   asks a sharper question: does whichever mapping is behaviorally harder
   for a subject also look neurally harder to their decoder?

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
from abstract_values.utils.data import BIDS_FOLDER, Subject
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


def neural_precision_sdE(eu: pd.DataFrame, variant: str) -> pd.DataFrame:
    """−mean σ_E (NPCr), from the simulated expected-uncertainty pipeline.
    Known caveat (see module docstring): raw σ_E is inflated by
    regression-to-mode bias variance near stimulus-density peaks, so this
    is not a clean noise measure — kept as a reference point, not the
    primary one.
    """
    eu = eu[eu["variant"] == variant]
    return ((-eu.groupby(["subject", "condition"])["sd_E"].mean())
            .reset_index(name="neu_precision")
            .rename(columns={"condition": "mapping"}))


def neural_precision_mae(eu: pd.DataFrame, variant: str) -> pd.DataFrame:
    """−mean |decoding error| (NPCr) from the same simulated-decoding runs
    as σ_E, but using the raw expected absolute error instead of its SD —
    less sensitive to the regression-to-mode variance-inflation than σ_E,
    since it's bias+noise combined rather than dispersion alone."""
    eu = eu[eu["variant"] == variant]
    return ((-eu.groupby(["subject", "condition"])["mean_abs_error"].mean())
            .reset_index(name="neu_precision")
            .rename(columns={"condition": "mapping"}))


def neural_precision_decoded_r(subjects: list[str], nvoxels: str = "100",
                               noise: str = "spherical", smoothed: bool = False,
                               roi: str = "NPCr") -> pd.DataFrame:
    """Empirical decoding accuracy: per (subject, mapping) Pearson r between
    true and decoded (posterior-mean) value on REAL single trials, in NPCr —
    the actual decode_value.py output, not a simulation. Session ↔ mapping
    via Subject.get_mapping (deterministic from subject parity + session).
    """
    smooth = "_smoothed" if smoothed else ""
    deriv = Path(BIDS_FOLDER) / "derivatives" / "decoding" / "value"
    rows = []
    for s in subjects:
        stem = f"sub-{s}_mask-{roi}_nvoxels-{nvoxels}_noise-{noise}{smooth}"
        d = deriv / f"sub-{s}" / "func"
        hits = sorted(d.glob(f"{stem}_pars.tsv")) or sorted(d.glob(f"{stem}_lambda-*_pars.tsv"))
        if not hits:
            continue
        df = pd.read_csv(hits[0], sep="\t")
        meta = ["session", "run", "trial_nr", "true_value_chf"]
        grid_cols = [c for c in df.columns if c not in meta]
        grid = np.array([float(c) for c in grid_cols])
        post = df[grid_cols].to_numpy(dtype=float)
        w = post / np.clip(post.sum(axis=1, keepdims=True), 1e-12, None)
        df = df.assign(decoded=(w * grid).sum(1))
        sub = Subject(s, bids_folder=BIDS_FOLDER)
        for session, g in df.groupby("session"):
            g = g.dropna(subset=["true_value_chf", "decoded"])
            if len(g) < 5:
                continue
            mapping = sub.get_mapping(session=int(session))
            r, _ = stats.pearsonr(g["true_value_chf"], g["decoded"])
            rows.append(dict(subject=s, mapping=mapping,
                             neu_precision=r, n_trials=len(g)))
    return pd.DataFrame(rows)


def page_brain_behavior(beh: pd.DataFrame, neu: pd.DataFrame, *, neu_label: str,
                        title: str, pdf: PdfPages):
    beh_prec = ((-beh.groupby(["subject", "mapping"])["error"].std())
                .reset_index(name="beh_precision"))
    merged = beh_prec.merge(neu[["subject", "mapping", "neu_precision"]],
                            on=["subject", "mapping"])

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
        panel_title = mapping if mapping == "pooled (both mappings)" else (
            "CDF mapping" if mapping == "cdf" else "Inverse-CDF mapping")
        ax.set_title(panel_title)
        ax.set_xlabel("Behavioral precision\n(−SD of bid error, CHF)")
    axes[0].set_ylabel(neu_label)
    fig.suptitle(title, fontsize=9.5, color="0.15")
    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return plot_df


def _condition_delta(df: pd.DataFrame, value_col: str) -> pd.Series:
    """cdf − inverse_cdf per subject, indexed by subject. Requires both
    mappings present for a subject (inner join via pivot, NaN otherwise)."""
    wide = df.pivot(index="subject", columns="mapping", values=value_col)
    if "cdf" not in wide or "inverse_cdf" not in wide:
        return pd.Series(dtype=float)
    return (wide["cdf"] - wide["inverse_cdf"]).dropna()


def page_condition_difference(beh: pd.DataFrame, neu_measures: dict[str, pd.DataFrame],
                              pdf: PdfPages):
    """Within-subject condition contrast (cdf − inverse_cdf), behavior vs
    each neural measure. Raw-scale behavior-vs-neural comparisons are
    confounded by the two modalities living on very different scales
    (neural σ_E/likelihood width is several-fold larger than behavioral
    response SD in CHF — compare page 1's row 1 vs row 2 y-axes). Taking
    the *within-subject* condition difference for each measure separately
    removes each subject's idiosyncratic overall scale/offset and asks a
    sharper question: does whichever mapping is behaviorally harder for a
    subject also look neurally harder to that subject's decoder?
    """
    beh_prec = ((-beh.groupby(["subject", "mapping"])["error"].std())
                .reset_index(name="beh_precision"))
    beh_delta = _condition_delta(beh_prec, "beh_precision")

    fig, axes = plt.subplots(1, len(neu_measures), figsize=(3.6 * len(neu_measures), 3.6),
                             constrained_layout=True, sharex=True)
    if len(neu_measures) == 1:
        axes = [axes]
    for ax, (label, neu_df) in zip(axes, neu_measures.items()):
        neu_delta = _condition_delta(neu_df, "neu_precision")
        common = beh_delta.index.intersection(neu_delta.index)
        x, y = beh_delta.loc[common], neu_delta.loc[common]
        ax.axhline(0, color="0.75", lw=0.7, ls="--", zorder=0)
        ax.axvline(0, color="0.75", lw=0.7, ls="--", zorder=0)
        ax.scatter(x, y, s=22, color=COL_NEU, alpha=0.75,
                  edgecolor="white", linewidth=0.4)
        if len(common) >= 3:
            r, p = stats.pearsonr(x, y)
            b1, b0 = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, b0 + b1 * xx, color="0.2", lw=1.1, zorder=0)
            ax.text(0.04, 0.96, f"n={len(common)}\nr={r:+.2f}, p={p:.3f}",
                    transform=ax.transAxes, va="top", fontsize=8.5)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Δ behavioral precision\n(cdf − inverse_cdf, CHF)")
    axes[0].set_ylabel("Δ neural precision\n(cdf − inverse_cdf)")
    fig.suptitle(
        "Condition contrast, within subject: does the harder mapping "
        "match between behavior and neural decoding?",
        fontsize=9.5, color="0.15")
    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main(eu_tsv: Path, out: Path, variants: tuple[str, ...],
         nvoxels: str, noise: str):
    beh = load_behavior()
    eu = pd.read_csv(eu_tsv, sep="\t")
    eu["subject"] = eu["subject"].astype(str)
    subjects = sorted(set(beh["subject"]) & set(eu["subject"]))
    decoded_r = neural_precision_decoded_r(subjects, nvoxels=nvoxels, noise=noise)

    out.parent.mkdir(parents=True, exist_ok=True)
    corr_frames = []
    with PdfPages(out) as pdf:
        for variant in variants:
            page_value_profiles(beh, eu, variant, pdf)

        variant = variants[0]
        neu_measures = {
            "−mean σ_E (NPCr)\n[simulated, SD of posterior]":
                neural_precision_sdE(eu, variant),
            "−mean |decoding error| (NPCr)\n[simulated]":
                neural_precision_mae(eu, variant),
            "Decoded-true r (NPCr)\n[real single trials]":
                decoded_r,
        }
        titles = {
            "−mean σ_E (NPCr)\n[simulated, SD of posterior]":
                "Brain-behavior: −mean σ_E (simulated posterior SD)",
            "−mean |decoding error| (NPCr)\n[simulated]":
                "Brain-behavior: −mean |decoding error| (simulated)",
            "Decoded-true r (NPCr)\n[real single trials]":
                "Brain-behavior: empirical decoded-vs-true r (real trials)",
        }
        for label, neu_df in neu_measures.items():
            cdf = page_brain_behavior(beh, neu_df, neu_label=label,
                                      title=f"{titles[label]}  ({variant})", pdf=pdf)
            cdf["measure"] = label.split("\n")[0]
            corr_frames.append(cdf)

        page_condition_difference(beh, neu_measures, pdf)

    if corr_frames:
        tsv = out.with_suffix(".tsv")
        pd.concat(corr_frames, ignore_index=True).to_csv(tsv, sep="\t", index=False)
        print(f"Sidecar: {tsv}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eu-tsv", required=True,
                   help="Sidecar TSV from expected_uncertainty_per_condition.py")
    p.add_argument("--variants", nargs="+", default=["unsmoothed"],
                   choices=["unsmoothed", "smoothed"])
    p.add_argument("--nvoxels", default="100",
                   help="nvoxels tag for the real decode_value.py pars TSVs")
    p.add_argument("--noise", default="spherical", choices=["full", "spherical", "geodesic"])
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()
    main(Path(args.eu_tsv), Path(args.out), tuple(args.variants),
         nvoxels=args.nvoxels, noise=args.noise)
