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

import nibabel as nib
from nilearn import image as nli
from pingouin import circ_r, partial_corr

from abstract_values.behavior.data import get_all_behavioral_data
from abstract_values.utils.data import BIDS_FOLDER, Subject
from abstract_values.visualize.npcr_uncertainty_vs_value import _aggregate
from abstract_values.visualize.cvr2_model_comparison import _cvr2_path, _load_roi_mask

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


def neural_precision_decoded_r_orientation(subjects: list[str], nvoxels: str = "100",
                                           noise: str = "spherical", smoothed: bool = False,
                                           roi: str = "BensonV1") -> pd.DataFrame:
    """Empirical orientation-decoding fidelity in V1, per (subject, mapping),
    from REAL single-trial decode_gabor.py output — circular analogue of
    :func:`neural_precision_decoded_r`. Uses the doubled-angle error
    resultant (not circular-circular correlation — see
    decoding_quality_scatter.py's ``_circular_fidelity`` docstring for why:
    a decode that straddles the 0/180° wrap can spuriously drag a plain
    circular correlation negative).
    """
    smooth = "_smoothed" if smoothed else ""
    deriv = Path(BIDS_FOLDER) / "derivatives" / "decoding" / "gabor"
    rows = []
    for s in subjects:
        stem = f"sub-{s}_mask-{roi}_nvoxels-{nvoxels}_noise-{noise}{smooth}"
        d = deriv / f"sub-{s}" / "func"
        hits = sorted(d.glob(f"{stem}_pars.tsv")) or sorted(d.glob(f"{stem}_lambda-*_pars.tsv"))
        if not hits:
            continue
        df = pd.read_csv(hits[0], sep="\t")
        meta = ["session", "run", "trial_nr", "true_orientation_rad"]
        grid_cols = [c for c in df.columns if c not in meta]
        grid = np.array([float(c) for c in grid_cols])
        post = df[grid_cols].to_numpy(dtype=float)
        w = post / np.clip(post.sum(axis=1, keepdims=True), 1e-12, None)
        ang = 2.0 * grid
        dec = 0.5 * np.arctan2((w * np.sin(ang)).sum(1), (w * np.cos(ang)).sum(1)) % np.pi
        df = df.assign(decoded=dec)
        sub = Subject(s, bids_folder=BIDS_FOLDER)
        for session, g in df.groupby("session"):
            g = g.dropna(subset=["true_orientation_rad", "decoded"])
            if len(g) < 5:
                continue
            mapping = sub.get_mapping(session=int(session))
            err2 = np.angle(np.exp(2j * (g["decoded"].to_numpy() - g["true_orientation_rad"].to_numpy())))
            fidelity = float(circ_r(err2))
            rows.append(dict(subject=s, mapping=mapping,
                             neu_precision=fidelity, n_trials=len(g)))
    return pd.DataFrame(rows)


def neural_n_signal_voxels(subjects: list[str], model_cv: str, roi: str, hemi: str | None,
                           baseline_cv: str = "aprf-null.cv",
                           smoothed: bool = False) -> pd.DataFrame:
    """Per-subject count (and fraction) of ROI voxels where ``model_cv``'s
    cvR² beats ``baseline_cv``'s cvR² at that voxel (the project's standard
    "signal voxel" test — cvR² > cvR²_null, not > 0; see CLAUDE.md /
    project_cvr2_null_baseline memory). Subject-level trait, not per-mapping
    (the CV fit pools both mapping sessions).
    """
    rows = []
    for s in subjects:
        p_model = _cvr2_path(model_cv, s, smoothed)
        p_base = _cvr2_path(baseline_cv, s, smoothed)
        if not (p_model.exists() and p_base.exists()):
            continue
        try:
            mask_img = _load_roi_mask(s, roi, hemi)
        except Exception:
            continue
        mask_arr = np.squeeze(mask_img.get_fdata()) > 0.5
        model_vals = nli.resample_to_img(nib.load(str(p_model)), mask_img,
                                         interpolation="nearest").get_fdata()[mask_arr]
        base_vals = nli.resample_to_img(nib.load(str(p_base)), mask_img,
                                        interpolation="nearest").get_fdata()[mask_arr]
        finite = np.isfinite(model_vals) & np.isfinite(base_vals)
        n_signal = int((model_vals[finite] > base_vals[finite]).sum())
        n_total = int(finite.sum())
        rows.append(dict(subject=s, n_signal=n_signal, n_total=n_total,
                         neu_precision=n_signal / n_total if n_total else np.nan))
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


def _broadcast_to_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicate a subject-level (subject, neu_precision) trait across both
    mapping rows, so it plugs into the same per-condition/pooled scatter
    machinery as the genuinely per-mapping measures."""
    return pd.concat([df.assign(mapping=m) for m in CONDITIONS], ignore_index=True)


def load_gaze_dispersion(gaze_tsv: Path, min_frac_valid: float = 0.5) -> pd.DataFrame:
    """Per-trial gaze dispersion (sqrt(var_x + var_y), px, EyeLink screen
    pixels) during the response_bar/estimation phase -> per-subject mean
    log-dispersion (log because raw dispersion is heavily right-skewed —
    a few long fixation excursions dominate a linear mean). Trials with
    <50% valid (non-blink) samples are dropped. Subject-level trait,
    pooled across both mapping sessions (same rationale as the voxel-count
    measures — one number per subject, not per condition)."""
    df = pd.read_csv(gaze_tsv, sep="\t", dtype={"subject": str})
    df["subject"] = df["subject"].apply(
        lambda s: f"{int(s):02d}" if s.isdigit() else s)
    df = df[(df["frac_valid"] >= min_frac_valid) & df["gaze_dispersion"].notna()
             & (df["gaze_dispersion"] > 0)]
    df["log_gaze_dispersion"] = np.log(df["gaze_dispersion"])
    return (df.groupby("subject")["log_gaze_dispersion"].mean()
            .reset_index(name="gaze_precision")
            .assign(gaze_precision=lambda d: -d["gaze_precision"]))  # sign: higher = steadier gaze


def page_gaze_confound(beh: pd.DataFrame, neu_measures: dict[str, pd.DataFrame],
                       gaze: pd.DataFrame, pdf: PdfPages):
    """Does eye movement during the estimation phase explain the
    brain-behavior correlations? For each subject-level neural measure,
    shows: neural measure vs gaze steadiness, behavioral precision vs gaze
    steadiness, and the behavioral-vs-neural correlation with gaze
    steadiness partialled out (pingouin.partial_corr) next to the raw one.
    """
    beh_prec = ((-beh.groupby(["subject", "mapping"])["error"].std())
                .reset_index(name="beh_precision"))
    beh_pooled = beh_prec.groupby("subject")["beh_precision"].mean().reset_index()

    n_measures = len(neu_measures)
    fig, axes = plt.subplots(2, n_measures, figsize=(3.6 * n_measures, 7.0),
                             constrained_layout=True)
    if n_measures == 1:
        axes = axes.reshape(2, 1)

    for col, (label, neu_df) in enumerate(neu_measures.items()):
        neu_pooled = (neu_df.groupby("subject")["neu_precision"].mean()
                      .reset_index())
        d = (beh_pooled.merge(neu_pooled, on="subject")
             .merge(gaze, on="subject").dropna())

        ax = axes[0, col]
        ax.scatter(d["gaze_precision"], d["neu_precision"], s=20,
                  color=COL_DENS, alpha=0.8, edgecolor="white", linewidth=0.4)
        if len(d) >= 3:
            r, p = stats.pearsonr(d["gaze_precision"], d["neu_precision"])
            ax.text(0.04, 0.96, f"r={r:+.2f}, p={p:.3f}", transform=ax.transAxes,
                    va="top", fontsize=8)
        ax.set_title(label.split("\n")[0], fontsize=8.5)
        ax.set_ylabel("Neural precision")
        if col == 0:
            ax.set_xlabel("")

        ax = axes[1, col]
        ax.scatter(d["beh_precision"], d["neu_precision"], s=20,
                  color=COL_NEU, alpha=0.4, edgecolor="white", linewidth=0.4,
                  label="raw")
        if len(d) >= 4:
            raw_r, raw_p = stats.pearsonr(d["beh_precision"], d["neu_precision"])
            pc = partial_corr(data=d, x="beh_precision", y="neu_precision",
                             covar="gaze_precision", method="pearson")
            part_r, part_p = float(pc["r"].iloc[0]), float(pc["p-val"].iloc[0])
            ax.text(0.04, 0.96,
                    f"raw:      r={raw_r:+.2f}, p={raw_p:.3f}\n"
                    f"|gaze:  r={part_r:+.2f}, p={part_p:.3f}",
                    transform=ax.transAxes, va="top", fontsize=8)
        ax.set_xlabel("Behavioral precision\n(−SD bid error, CHF)")
        if col == 0:
            ax.set_ylabel("Neural precision")

    fig.suptitle(
        "Eye-movement confound check — steadier gaze during estimation "
        "(top row), and brain-behavior r with gaze steadiness partialled "
        "out (bottom row, \"|gaze\")",
        fontsize=9.5, color="0.15")
    sns.despine(fig=fig, offset=5, trim=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main(eu_tsv: Path, out: Path, variants: tuple[str, ...],
         nvoxels: str, noise: str, gaze_tsv: Path | None = None):
    beh = load_behavior()
    eu = pd.read_csv(eu_tsv, sep="\t")
    eu["subject"] = eu["subject"].astype(str)
    subjects = sorted(set(beh["subject"]) & set(eu["subject"]))
    decoded_r = neural_precision_decoded_r(subjects, nvoxels=nvoxels, noise=noise)
    decoded_r_ori = neural_precision_decoded_r_orientation(subjects, nvoxels=nvoxels, noise=noise)
    npcr_voxels = _broadcast_to_mappings(
        neural_n_signal_voxels(subjects, "aprf.cv", "NPCr", None))
    v1_voxels = _broadcast_to_mappings(
        neural_n_signal_voxels(subjects, "vonmises.cv", "BensonV1", "LR"))

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
            "Signal-voxel frac. (NPCr, aprf.cv>null)\n[cvR² count]":
                npcr_voxels,
            "Signal-voxel frac. (V1, vonmises.cv>null)\n[cvR² count]":
                v1_voxels,
            "Decoded-true fidelity, orientation (V1)\n[real single trials]":
                decoded_r_ori,
        }
        titles = {
            "−mean σ_E (NPCr)\n[simulated, SD of posterior]":
                "Brain-behavior: −mean σ_E (simulated posterior SD)",
            "−mean |decoding error| (NPCr)\n[simulated]":
                "Brain-behavior: −mean |decoding error| (simulated)",
            "Decoded-true r (NPCr)\n[real single trials]":
                "Brain-behavior: empirical decoded-vs-true r (real trials)",
            "Signal-voxel frac. (NPCr, aprf.cv>null)\n[cvR² count]":
                "Brain-behavior: fraction of NPCr voxels beating the null (cvR²)",
            "Signal-voxel frac. (V1, vonmises.cv>null)\n[cvR² count]":
                "Specificity check — V1: fraction of V1 voxels beating the null (cvR²)",
            "Decoded-true fidelity, orientation (V1)\n[real single trials]":
                "Specificity check — V1: empirical orientation-decoding fidelity",
        }
        for label, neu_df in neu_measures.items():
            cdf = page_brain_behavior(beh, neu_df, neu_label=label,
                                      title=f"{titles[label]}  ({variant})", pdf=pdf)
            cdf["measure"] = label.split("\n")[0]
            corr_frames.append(cdf)

        # Condition-difference is only meaningful for genuinely per-mapping
        # measures — voxel-count traits are identical across mappings by
        # construction (Δ≡0), so they're excluded here.
        page_condition_difference(beh, {
            k: v for k, v in neu_measures.items()
            if k in ("−mean σ_E (NPCr)\n[simulated, SD of posterior]",
                    "−mean |decoding error| (NPCr)\n[simulated]",
                    "Decoded-true r (NPCr)\n[real single trials]",
                    "Decoded-true fidelity, orientation (V1)\n[real single trials]")
        }, pdf)

        if gaze_tsv is not None:
            gaze = load_gaze_dispersion(gaze_tsv)
            page_gaze_confound(beh, {
                "Signal-voxel frac. (NPCr, aprf.cv>null)\n[cvR² count]": npcr_voxels,
                "Decoded-true r (NPCr)\n[real single trials]": decoded_r,
            }, gaze, pdf)

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
    p.add_argument("--gaze-tsv", default=None,
                   help="Per-trial gaze-dispersion TSV (subject, session, mapping, "
                        "run, trial_nr, gaze_dispersion, n_samples, frac_valid) from "
                        "extract_gaze_dispersion.py. Adds the eye-movement confound page.")
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()
    main(Path(args.eu_tsv), Path(args.out), tuple(args.variants),
         nvoxels=args.nvoxels, noise=args.noise,
         gaze_tsv=Path(args.gaze_tsv) if args.gaze_tsv else None)
