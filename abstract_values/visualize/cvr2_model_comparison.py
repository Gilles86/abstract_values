"""Cross-validated R² comparison across the nested-model ladder.

For NPCr (value coding):
  null  →  standard  →  session-shift  →  fwhm-shift  →  fully-shifted

For V1 (orientation coding):
  vonmises (joint, 'fixed tuning')  vs  vonmises-shift (per-session weights,
                                                          'flexible tuning')

Reads ``cvr2`` maps from each model's ``.cv`` output directory (one
per subject × smoothing variant), masks to an ROI, and aggregates
per-voxel cvR² across subjects.

Per page (one (ROI, smoothing) cell):
  - Bar plot of median cvR² across voxels, per model. Per-subject
    points overlaid (thin gray + connecting lines).
  - Side annotation: each model's median cvR², the Δ vs the simpler
    nested model, and whether that Δ is significantly > 0 across
    subjects (paired Wilcoxon).

Usage:
    python -m abstract_values.visualize.cvr2_model_comparison
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from nilearn import image as nli
from scipy import stats

from abstract_values.utils.data import BIDS_FOLDER, Subject

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelpad": 4,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2, "legend.frameon": False, "legend.handlelength": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})
sns.set_context("paper")

DERIV = Path(BIDS_FOLDER) / "derivatives" / "encoding_models"
DEFAULT_OUT = (Path(BIDS_FOLDER) / "derivatives" / "qa"
               / "cvr2_model_comparison.pdf")

# Two ROI-specific model ladders (nested in flexibility, left → right).
NPCR_LADDER = [
    ("null",          "aprf-null.cv",          "null"),
    ("standard",      "aprf.cv",                "standard"),
    ("session-shift", "aprf-shift.cv",          "mode shifts"),
    ("fwhm-shift",    "aprf-fwhm-shift.cv",     "mode + fwhm"),
    ("fully-shifted", "aprf-fully-shifted.cv",  "all 4 shift"),
]
V1_LADDER = [
    # The null model is paradigm-independent (just per-voxel training
    # mean over the same gabor-trial betas), so the aprf-null.cv map
    # is the correct null baseline for V1 too.
    ("null",           "aprf-null.cv",       "null"),
    ("vonmises",       "vonmises.cv",        "fixed tuning"),
    ("vonmises-shift", "vonmises-shift.cv",  "flexible tuning"),
]
# Orientation vs value dissociation. Applied to both ROIs:
#   NPCr — does IPS encode the gabor's orientation (low-level visual) or
#       its CHF value (with PRFs that shift between conditions)?
#   V1   — symmetric check. Expectation: orientation wins. If value rungs
#       beat orientation in V1, something's confounded.
# Per-voxel argmax across the five rungs answers the question voxel-by-
# voxel; per-subject winning-fraction stacked bar gives the headline.
DISSOC_LADDER = [
    ("null",            "aprf-null.cv",       "null"),
    ("vonmises",        "vonmises.cv",        "orient (fixed)"),
    ("vonmises-shift",  "vonmises-shift.cv",  "orient (flexible)"),
    ("aprf",            "aprf.cv",            "value (fixed)"),
    ("aprf-shift",      "aprf-shift.cv",      "value (shifted)"),
]
# Back-compat alias used by older callers.
NPCR_DISSOC_LADDER = DISSOC_LADDER
PALETTE = ["#9C9C9C", "#3B5BA5", "#5D8C3F", "#E76F51", "#C44E52", "#8172B2"]


def _cvr2_path(model_subdir, subject, smoothed):
    smooth = "_smoothed" if smoothed else ""
    return (DERIV / model_subdir / f"sub-{subject}" / "func"
            / f"sub-{subject}_task-abstractvalue_space-T1w"
              f"_desc-cvr2{smooth}_pe.nii.gz")


def _load_roi_mask(subject, roi, hemi):
    sub = Subject(subject, bids_folder=Path(BIDS_FOLDER))
    return sub.get_roi_mask(roi, hemi=hemi)


def _collect(ladder, subjects, roi, hemi, smoothed, *,
              filter_null_loses=True):
    """For each (subject, model), load cvR² masked to ROI. Returns long
    DataFrame: subject × model × median_cvr2 + n_voxels, plus
    per-voxel arrays indexed by (subject, model).

    With ``filter_null_loses=True`` AND the ladder including ``"null"``,
    voxels where the null model is the per-voxel argmax across the
    ladder are excluded — keeps only 'real-signal voxels' for which
    at least one encoding model beats predicting the training mean.
    The filter is applied per-subject (each subject's voxel selection
    depends on that subject's own argmax across loaded models). If a
    subject is missing the null map, that subject's voxels are kept
    unfiltered with a warning.
    """
    # ── First pass: load all per-voxel maps, aligned to a per-subject ROI mask
    raw = {}                                       # (subject, model) → array
    n_voxels_per_subj = {}                         # subject → n_voxels in ROI
    for model_name, subdir, _ in ladder:
        for s in subjects:
            p = _cvr2_path(subdir, s, smoothed)
            if not p.exists():
                continue
            try:
                mask_img = _load_roi_mask(s, roi, hemi)
            except Exception:
                continue
            mask_arr = np.squeeze(mask_img.get_fdata()) > 0.5
            cv_img = nli.resample_to_img(nib.load(str(p)), mask_img,
                                           interpolation="nearest")
            vals = cv_img.get_fdata()[mask_arr].astype(np.float32)
            raw[(s, model_name)] = vals
            n_voxels_per_subj.setdefault(s, vals.size)

    # ── Per-subject argmax across the ladder.
    # Two derived products:
    #   keep_masks: bool array (drop voxels where null wins) — used as the
    #               cvR² filter on the left panel.
    #   wins_frac:  DataFrame (subject, model, frac) — used by the right
    #               panel to show what fraction of ROI voxels each model
    #               is the per-voxel cvR² winner for.
    ladder_models = [m for m, _, _ in ladder]
    keep_masks = {}                                # subject → bool array
    wins_rows  = []                                # subject × model rows
    for s in subjects:
        arrs = {m: raw.get((s, m)) for m in ladder_models}
        if any(a is None for a in arrs.values()):
            # Missing at least one ladder model → can't run the per-voxel
            # argmax. Skip both the null filter and the winning-fraction
            # computation for this subject.
            continue
        stack = np.column_stack([arrs[m] for m in ladder_models])
        stack = np.where(np.isfinite(stack), stack, -np.inf)
        argmax = np.argmax(stack, axis=1)
        n_total = argmax.size
        for i, m in enumerate(ladder_models):
            wins_rows.append({"subject": s, "model": m,
                                "wins_frac": float((argmax == i).mean()),
                                "wins_count": int((argmax == i).sum()),
                                "n_voxels_total": n_total})
        if filter_null_loses and "null" in ladder_models:
            null_idx = ladder_models.index("null")
            keep_masks[s] = argmax != null_idx

    wins_df = pd.DataFrame(wins_rows) if wins_rows else pd.DataFrame(
        columns=["subject", "model", "wins_frac", "wins_count",
                  "n_voxels_total"])

    # ── Second pass: build the long DataFrame using the filter where avail
    rows = []
    per_voxel = {}
    for (s, model_name), vals in raw.items():
        mask = keep_masks.get(s)
        v = vals[mask] if mask is not None else vals
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        rows.append({"subject": s, "model": model_name,
                      "median_cvr2": float(np.median(v)),
                      "mean_cvr2":   float(np.mean(v)),
                      "n_voxels":    int(v.size),
                      "n_voxels_total": int(vals.size),
                      "filtered":   mask is not None})
        per_voxel[(s, model_name)] = v
    return pd.DataFrame(rows), per_voxel, wins_df


def _paired_test(df, model_a, model_b):
    """Paired Wilcoxon across subjects: returns (n_paired, median_delta, p)."""
    sub_a = df[df.model == model_a].set_index("subject")["median_cvr2"]
    sub_b = df[df.model == model_b].set_index("subject")["median_cvr2"]
    paired = pd.concat({"a": sub_a, "b": sub_b}, axis=1).dropna()
    if len(paired) < 3:
        return len(paired), float("nan"), float("nan")
    delta = paired["b"] - paired["a"]
    try:
        _, p = stats.wilcoxon(delta, alternative="greater")
    except ValueError:                            # all-zero deltas etc.
        p = float("nan")
    return len(paired), float(delta.median()), float(p)


def page(ladder, subjects, roi_label, roi, hemi, smoothed, pdf, *,
          title_prefix="", filter_null_loses=True):
    df, _, wins_df = _collect(ladder, subjects, roi, hemi, smoothed,
                                filter_null_loses=filter_null_loses)
    if df.empty:
        return
    model_order = [m for m, _, _ in ladder]

    fig, (ax, ax_frac) = plt.subplots(
        1, 2, figsize=(11.0, 4.6), constrained_layout=True,
        gridspec_kw={"width_ratios": [3.2, 1]})

    # Per-subject points + connecting lines so paired structure is visible
    for s in df.subject.unique():
        sub_df = df[df.subject == s].set_index("model").reindex(model_order)
        xs = list(range(len(model_order)))
        ax.plot(xs, sub_df["median_cvr2"].values, "-o",
                 color="0.7", lw=0.6, ms=3, alpha=0.6, zorder=1)

    # Group median + IQR diamond per model
    for i, (m, _, _) in enumerate(ladder):
        vals = df[df.model == m]["median_cvr2"].dropna().values
        if vals.size == 0: continue
        med = float(np.median(vals))
        q25, q75 = np.percentile(vals, [25, 75])
        c = PALETTE[i % len(PALETTE)]
        ax.errorbar([i], [med], yerr=[[med - q25], [q75 - med]],
                     fmt="D", ms=11, mec="black", mew=1.4, lw=0,
                     color=c, ecolor="black", elinewidth=0.8,
                     capsize=0, zorder=5)
        # Median value label
        ax.text(i, q75 + 0.003,
                 f"{med:.3f}", ha="center", va="bottom",
                 fontsize=7.5, color="0.2")
    ax.axhline(0, color="0.6", lw=0.5, ls=":", zorder=0)

    # Pairwise nested model deltas — only between adjacent ladder rungs
    annot_lines = []
    for (m_a, _, _), (m_b, _, _) in zip(ladder[:-1], ladder[1:]):
        n, d, p = _paired_test(df, m_a, m_b)
        if not np.isfinite(p):
            annot_lines.append(f"{m_b} − {m_a}: n={n}, p=n/a")
        else:
            sig = " *" if p < 0.05 else ""
            annot_lines.append(
                f"{m_b} − {m_a}: Δ={d:+.4f}  p={p:.1e}{sig}  (n={n})")
    ax.text(1.02, 0.98, "\n".join(annot_lines),
             transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
             color="0.25", family="monospace")

    ax.set_xticks(list(range(len(model_order))))
    # Skip the parenthetical when it just repeats the model name (null,
    # standard) — redundant text, and the leading cause of adjacent two-line
    # labels colliding at this axis width.
    xtick_labels = [m if lbl == m else f"{m}\n({lbl})" for m, _, lbl in ladder]
    ax.set_xticklabels(xtick_labels, fontsize=7.5, rotation=18, ha="right")
    ax.set_ylabel("Per-subject median voxel cv-R²  (signal voxels only)")
    smooth_lbl = "smoothed" if smoothed else "unsmoothed"
    # Voxel-filter status: report the cohort median fraction of ROI
    # voxels kept after the null-loses filter (per subject the kept
    # fraction can vary).
    if filter_null_loses and "filtered" in df.columns and df["filtered"].any():
        keep_frac = (df.groupby("subject")
                       .apply(lambda g: g["n_voxels"].iloc[0]
                                          / max(g["n_voxels_total"].iloc[0], 1)))
        filt_note = (f"signal voxels only ({100*keep_frac.median():.0f}% "
                      f"of ROI kept, null-loses filter)")
    else:
        filt_note = "all ROI voxels (no null filter)"
    # Page-level header spans the full figure width (both panels) so it
    # can't visually run into ax_frac's own title the way a long
    # ax.set_title() on the narrower left panel would.
    fig.suptitle(f"{title_prefix}{roi_label}  ·  cvR² across nested models  "
                  f"·  {smooth_lbl}  ·  n_subjects={df.subject.nunique()}  ·  "
                  f"{filt_note}",
                  fontsize=9.5, color="0.15", x=0.01, ha="left")

    # ── Right panel: per-subject stacked bar of % ROI voxels where each
    # ladder model is the per-voxel cvR² argmax. Bars sum to 100% per
    # subject; null gets the leftmost (gray) slice — its size IS the
    # complement of the signal-voxel prevalence ("null wins" = no model).
    if not wins_df.empty:
        wide = (wins_df.pivot(index="subject", columns="model",
                                values="wins_frac")
                       .reindex(columns=model_order)
                       .fillna(0.0)) * 100.0
        # Sort subjects by signal-voxel prevalence (1 − null fraction).
        if "null" in wide.columns:
            wide = wide.assign(_signal=100 - wide["null"]).sort_values(
                "_signal", ascending=False).drop(columns="_signal")
        subs = list(wide.index)
        bottoms = np.zeros(len(subs))
        for i, m in enumerate(model_order):
            if m not in wide.columns:
                continue
            vals = wide[m].values
            color = "#9C9C9C" if m == "null" else PALETTE[i % len(PALETTE)]
            ax_frac.bar(range(len(subs)), vals, bottom=bottoms,
                          color=color, edgecolor="white", linewidth=0.4,
                          label=m)
            bottoms += vals
        ax_frac.set_xticks(range(len(subs)))
        ax_frac.set_xticklabels([f"sub-{s}" for s in subs],
                                  rotation=90, fontsize=7)
        ax_frac.set_ylabel("% of ROI voxels (per-voxel cvR² argmax)")
        ax_frac.set_ylim(0, 100)
        ax_frac.set_title("Per-subject winning-model proportions",
                            fontsize=9, color="0.2")
        ax_frac.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                         fontsize=7, frameon=False, handlelength=1.2,
                         title="model", title_fontsize=7.5)
    else:
        ax_frac.axis("off")
        ax_frac.text(0.5, 0.5,
                       "Not all ladder models present\n— can't compute argmax",
                       transform=ax_frac.transAxes,
                       ha="center", va="center", color="0.5", fontsize=8)
    sns.despine(fig=fig, offset=4, trim=False)
    # sns.despine() resets tick-label rotation as a side effect — always
    # reapply rotation AFTER despine, never before (see CLAUDE.md note on
    # this cvr2_model_comparison rendering gotcha).
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=7.5)
    if not wins_df.empty:
        plt.setp(ax_frac.get_xticklabels(), rotation=90, fontsize=7)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def page_signal_winners(ladder, subjects, roi_label, roi, hemi, smoothed, pdf, *,
                          title_prefix=""):
    """Dedicated page: per-subject and cohort-mean winning proportions
    *among signal voxels* (voxels where null does NOT win per-voxel
    argmax). Slices sum to 100% per subject. For ladders mixing model
    families (e.g. NPCR_DISSOC_LADDER: orientation vs value PRFs), this
    answers "of the voxels with real tuning, what % are best fit by
    each family?".
    """
    _, _, wins_df = _collect(ladder, subjects, roi, hemi, smoothed,
                               filter_null_loses=True)
    if wins_df.empty:
        return
    nonnull = [m for m, _, _ in ladder if m != "null"]
    if not nonnull:
        return
    sig = wins_df[wins_df["model"].isin(nonnull)].copy()
    totals = sig.groupby("subject")["wins_frac"].sum()
    sig["signal_pct"] = sig.apply(
        lambda r: 100 * r["wins_frac"] / totals[r["subject"]]
                  if totals.get(r["subject"], 0) > 0 else 0,
        axis=1,
    )
    wide = (sig.pivot(index="subject", columns="model", values="signal_pct")
                .reindex(columns=nonnull).fillna(0.0))
    # Sort by the dominant model's share — descending — so the most
    # consistently-X subjects are leftmost.
    dominant = wide.idxmax(axis=1).value_counts().index[0]
    wide = wide.sort_values(dominant, ascending=False)

    fig, (ax_sub, ax_mean) = plt.subplots(
        1, 2, figsize=(11, 5), constrained_layout=True,
        gridspec_kw={"width_ratios": [4, 1]})

    subs = list(wide.index)
    # Use palette positions matching the ladder rung index (skip 0=null).
    color_of = {m: PALETTE[i % len(PALETTE)]
                  for i, (m, _, _) in enumerate(ladder)}
    label_of = {m: lbl for m, _, lbl in ladder}

    bottoms = np.zeros(len(subs))
    for m in nonnull:
        vals = wide[m].values
        ax_sub.bar(range(len(subs)), vals, bottom=bottoms,
                     color=color_of[m], edgecolor="white", linewidth=0.4,
                     label=f"{m}  ({label_of[m]})")
        bottoms += vals
    ax_sub.set_xticks(range(len(subs)))
    ax_sub.set_xticklabels([f"sub-{s}" for s in subs],
                              rotation=90, fontsize=7)
    ax_sub.set_ylim(0, 100)
    ax_sub.set_ylabel("% of signal voxels (null-loses)")
    smooth_lbl = "smoothed" if smoothed else "unsmoothed"
    # Full-figure-width header (not ax_sub.set_title) so a long roi_label
    # (e.g. "NPCr (orientation vs value)") can't spill into ax_mean's
    # "Cohort" title sharing the same row.
    fig.suptitle(
        f"{title_prefix}{roi_label}  ·  per-subject winning-model proportions "
        f"·  signal voxels only  ·  {smooth_lbl}  ·  "
        f"n_subjects={wide.shape[0]}",
        fontsize=9.5, color="0.15", x=0.01, ha="left")
    ax_sub.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
                    ncol=len(nonnull), fontsize=8, frameon=False,
                    handlelength=1.2)

    # Cohort-mean bar
    mean_vals = wide.mean(axis=0).values
    bottom = 0.0
    for i, m in enumerate(nonnull):
        v = float(mean_vals[i])
        ax_mean.bar([0], [v], bottom=bottom, color=color_of[m],
                      edgecolor="white", linewidth=0.4)
        if v >= 4:                                     # only label slices wide enough
            ax_mean.text(0, bottom + v / 2, f"{v:.0f}%",
                            ha="center", va="center", fontsize=9,
                            color="white", weight="bold")
        bottom += v
    ax_mean.set_xticks([0]); ax_mean.set_xticklabels(["cohort mean"])
    ax_mean.set_ylim(0, 100)
    ax_mean.set_yticks([])
    ax_mean.set_title("Cohort", fontsize=9, color="0.2")

    sns.despine(fig=fig, offset=4, trim=False)
    plt.setp(ax_sub.get_xticklabels(), rotation=90, fontsize=7)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _fit_dirichlet_multinomial(wins_df, ladder):
    """Hierarchical Dirichlet-Multinomial across subjects.

    Model
    -----
        y_s | p_s, N_s   ~ Multinomial(N_s, p_s)            (per subject)
        p_s | α           ~ Dirichlet(α)                     (subject draws)
        α_k               ~ Gamma(2, 1)                      (vague cohort prior)
        p_cohort_k = α_k / Σ_k α_k                           (deterministic)

    Returns
    -------
    (idata, counts) where counts is the per-(subject, non-null model)
    integer wins matrix actually fed to the model. Returns (None, None)
    if there's not enough data to fit (≥2 non-null models, ≥3 subjects).
    """
    import numpy as np
    import pymc as pm

    nonnull = [m for m, _, _ in ladder if m != "null"]
    if len(nonnull) < 2:
        return None, None
    counts = (wins_df[wins_df["model"].isin(nonnull)]
                .pivot(index="subject", columns="model", values="wins_count")
                .reindex(columns=nonnull).fillna(0).astype(int))
    counts = counts[counts.sum(axis=1) > 0]                  # drop empty rows
    if counts.shape[0] < 3:
        return None, None
    y = counts.values
    n_per_sub = y.sum(axis=1)
    K = len(nonnull)

    coords = {"subject": list(counts.index), "model": nonnull}
    with pm.Model(coords=coords):
        # Mildly informative prior on cohort concentration. Gamma(2,1) has
        # mode 1 and mean 2 — enough to keep α away from 0 (which would
        # cause label-switching when one component is absent) but lets
        # the data dominate.
        alpha = pm.Gamma("alpha", alpha=2.0, beta=1.0, dims="model")
        p_sub = pm.Dirichlet("p_sub", a=alpha, dims=("subject", "model"))
        pm.Multinomial("wins_obs", n=n_per_sub, p=p_sub,
                         observed=y, dims=("subject", "model"))
        pm.Deterministic("p_cohort", alpha / pm.math.sum(alpha),
                           dims="model")
        idata = pm.sample(1500, tune=1500, chains=4, target_accept=0.95,
                            progressbar=False, random_seed=42)
    return idata, counts


def page_bayes_proportions(ladder, subjects, roi_label, roi, hemi, smoothed,
                             pdf, *, title_prefix=""):
    """Bayesian hierarchical winning-model proportions.

    Inputs: per-(subject, non-null model) voxel-win counts among signal
    voxels (where per-voxel argmax across the ladder ≠ null).
    Outputs: per-subject observed proportions stacked bar + cohort-level
    posterior with 50% and 95% HDIs.
    """
    import arviz as az
    import numpy as np

    _, _, wins_df = _collect(ladder, subjects, roi, hemi, smoothed,
                               filter_null_loses=True)
    if wins_df.empty:
        return
    idata, counts = _fit_dirichlet_multinomial(wins_df, ladder)
    if idata is None:
        return
    nonnull = list(counts.columns)
    # Flatten (chain, draw) → sample. Resulting shape: (K, n_samples)
    cohort = (idata.posterior["p_cohort"]
              .stack(sample=("chain", "draw")).values)

    # Per-subject observed proportions, sorted by dominant component.
    obs = counts.div(counts.sum(axis=1), axis=0).multiply(100)
    dominant = obs.idxmax(axis=1).value_counts().index[0]
    obs = obs.sort_values(dominant, ascending=False)

    color_of = {m: PALETTE[i % len(PALETTE)]
                  for i, (m, _, _) in enumerate(ladder)}
    label_of = {m: lbl for m, _, lbl in ladder}

    fig, (ax_sub, ax_post) = plt.subplots(
        1, 2, figsize=(11.0, 5.0), constrained_layout=True,
        gridspec_kw={"width_ratios": [3, 1.6]})

    # Per-subject stacked bar (observed)
    bottoms = np.zeros(obs.shape[0])
    for m in nonnull:
        vals = obs[m].values
        ax_sub.bar(range(obs.shape[0]), vals, bottom=bottoms,
                     color=color_of[m], edgecolor="white", linewidth=0.4,
                     label=f"{m}  ({label_of[m]})")
        bottoms += vals
    ax_sub.set_xticks(range(obs.shape[0]))
    ax_sub.set_xticklabels([f"sub-{s}" for s in obs.index],
                              rotation=90, fontsize=7)
    ax_sub.set_ylim(0, 100)
    ax_sub.set_ylabel("% of signal voxels  (observed, per subject)")
    smooth_lbl = "smoothed" if smoothed else "unsmoothed"
    fig.suptitle(
        f"{title_prefix}{roi_label}  ·  Bayesian hierarchical "
        f"Dirichlet-Multinomial  ·  signal voxels  ·  {smooth_lbl}",
        fontsize=9.5, color="0.15", x=0.01, ha="left")
    ax_sub.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
                    ncol=len(nonnull), fontsize=7.5, frameon=False,
                    handlelength=1.2)

    # Cohort-level posterior: median + 50% + 95% HDI per model
    K = len(nonnull)
    medians_c = np.median(cohort, axis=1) * 100
    hdi95 = np.stack([az.hdi(cohort[k], hdi_prob=0.95) for k in range(K)]) * 100
    hdi50 = np.stack([az.hdi(cohort[k], hdi_prob=0.50) for k in range(K)]) * 100
    for i, m in enumerate(nonnull):
        ax_post.plot([i, i], [hdi95[i, 0], hdi95[i, 1]],
                       color=color_of[m], lw=2.0, alpha=0.55, solid_capstyle="round")
        ax_post.plot([i, i], [hdi50[i, 0], hdi50[i, 1]],
                       color=color_of[m], lw=5.5, solid_capstyle="round")
        ax_post.plot([i], [medians_c[i]], "o", color="black",
                       ms=7, mec="white", mew=1.0, zorder=5)
        ax_post.text(i + 0.18, medians_c[i],
                       f"{medians_c[i]:.0f}%\n[{hdi95[i, 0]:.0f}–{hdi95[i, 1]:.0f}]",
                       fontsize=7.5, va="center", ha="left", color="0.2")
    ax_post.set_xticks(range(K))
    ax_post.set_xticklabels(nonnull, rotation=35, ha="right", fontsize=7.5)
    ax_post.set_xlim(-0.6, K - 0.4 + 1.0)                   # room for HDI labels
    ax_post.set_ylim(0, 100)
    ax_post.set_ylabel("Cohort proportion (%)\nmedian · 50% · 95% HDI")
    ax_post.set_title("Cohort-level posterior", fontsize=9, color="0.2")

    sns.despine(fig=fig, offset=4, trim=False)
    plt.setp(ax_sub.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax_post.get_xticklabels(), rotation=35, ha="right", fontsize=7.5)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _rfx_bms(L, n_iter=1000, tol=1e-6, n_samples=20000, seed=42):
    """SPM-style random-effects Bayesian Model Selection.

    Stephan et al. (2009) NeuroImage 46:1004; Rigoux et al. (2014)
    NeuroImage 84:971. Variational Bayes inference over per-subject
    model assignments and cohort-level Dirichlet on model frequencies.

    Parameters
    ----------
    L : (S, K) array
        Per-subject log marginal likelihood per model (S subjects, K models).

    Returns
    -------
    dict with:
      alpha (K,) — Dirichlet posterior on cohort model frequencies
      g     (S, K) — per-subject posterior model assignments
      expected_freq (K,) — <r_k> = α_k / Σα
      ep    (K,) — exceedance probability  P(freq_k > all others)
      pep   (K,) — protected exceedance probability (Rigoux 2014)
      bor   float — Bayes omnibus risk (P that all models are equally frequent)
    """
    import numpy as np
    from scipy.special import digamma, gammaln, logsumexp

    L = np.asarray(L, float)
    S, K = L.shape
    a0 = np.ones(K)                                # uniform Dirichlet prior
    alpha = a0.copy()

    # Variational Bayes iteration: alternate updates of subject-level
    # responsibilities g and cohort Dirichlet α.
    for _ in range(n_iter):
        alpha_old = alpha.copy()
        log_g = L + (digamma(alpha) - digamma(alpha.sum()))[None, :]
        log_g = log_g - logsumexp(log_g, axis=1, keepdims=True)
        g = np.exp(log_g)
        alpha = a0 + g.sum(axis=0)
        if np.abs(alpha - alpha_old).max() < tol:
            break

    # Free energies for BOR (Rigoux 2014 Eq. 13–15).
    # F1 under H1 (different models for different subjects)
    F1 = (gammaln(alpha.sum()) - gammaln(alpha).sum()
          - gammaln(a0.sum())  + gammaln(a0).sum()
          + (g * (L - log_g)).sum())
    # F0 under H0 (uniform per-subject prior; no RFX structure)
    F0 = (logsumexp(L, axis=1) - np.log(K)).sum()
    bor = float(np.clip(1.0 / (1.0 + np.exp(F1 - F0)), 0.0, 1.0))

    # Exceedance probabilities via Monte Carlo from the cohort Dirichlet
    rng = np.random.default_rng(seed)
    samples = rng.dirichlet(alpha, size=n_samples)
    winners = np.argmax(samples, axis=1)
    ep  = np.array([float(np.mean(winners == k)) for k in range(K)])
    # Protected EP corrects for chance-under-null
    pep = (1.0 - bor) * ep + bor / K

    return dict(alpha=alpha, g=g,
                expected_freq=alpha / alpha.sum(),
                ep=ep, pep=pep, bor=bor)


def page_rfx_bms(ladder, subjects, roi_label, roi, hemi, smoothed, pdf,
                   n_trials=368, *, title_prefix=""):
    """SPM-style RFX BMS page for the dissociation ladder.

    Per-subject log-evidence per model = exact Gaussian-residual held-out
    log-likelihood-ratio over null, averaged across signal voxels:
        L_sm = (N_test / 2) × mean_voxel(−log(1 − cvR²))
    This is the *exact* held-out log-likelihood-ratio for OLS with iid
    Gaussian residuals (not the leading-order N·cvR²/2 Taylor expansion);
    voxels with cvR² < 0 contribute negatively (model is worse than
    predicting the test-set mean for that voxel). We take the mean (not
    sum) across voxels so spatial correlation doesn't inflate the
    effective sample size.
    """
    import numpy as np

    # Signal voxels only: null is excluded as a per-voxel argmax winner,
    # so the comparison among non-null models isn't drowned by ROI-wide
    # null background. (BMS over all-ROI mean cvR² with null included
    # in the ladder mostly tells us "did the ROI contain enough signal
    # to beat predicting the mean", not "which family wins").
    df, per_voxel, _ = _collect(ladder, subjects, roi, hemi, smoothed,
                                  filter_null_loses=True)
    if df.empty:
        return
    models = [m for m, _, _ in ladder if m != "null"]

    # Exact held-out log-likelihood-ratio over null per voxel:
    #   ΔLL_v = -N_test/2 · log(1 − cvR²_v)
    # Aggregate by *mean* across signal voxels (not sum — avoid inflating
    # by spatial correlation). cvR² very close to 1 is clipped to avoid
    # log(0).
    rows = []
    for s in sorted({s for s, _ in per_voxel.keys()}):
        if not all((s, m) in per_voxel for m in models):
            continue
        row = {"subject": s}
        for m in models:
            cvr2 = per_voxel[(s, m)]
            cvr2 = np.clip(cvr2, -np.inf, 0.999)        # avoid log(0)
            row[m] = float(np.nanmean(-np.log1p(-cvr2)))
        rows.append(row)
    if len(rows) < 3:
        return
    pivot = pd.DataFrame(rows).set_index("subject")[models]
    L = (n_trials / 2.0) * pivot.values            # (S, K) log-evidence

    result = _rfx_bms(L)
    expected = result["expected_freq"] * 100
    ep, pep  = result["ep"] * 100, result["pep"] * 100
    bor      = result["bor"]
    color_of = {m: PALETTE[i % len(PALETTE)]
                for i, (m, _, _) in enumerate(ladder)}

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    K = len(models)
    x = np.arange(K)
    w = 0.27
    cols = [color_of[m] for m in models]
    ax.bar(x - w, expected, w, color=cols, alpha=0.95,
              edgecolor="white", label="<r>: expected freq")
    ax.bar(x,     ep,       w, color=cols, alpha=0.55,
              edgecolor="white", label="EP: exceedance prob")
    ax.bar(x + w, pep,      w, color=cols, alpha=0.30,
              edgecolor="white", label="PEP: protected EP")
    for i, m in enumerate(models):
        for xi, v in zip([x[i] - w, x[i], x[i] + w],
                            [expected[i], ep[i], pep[i]]):
            ax.text(xi, v + 1.5, f"{v:.0f}", ha="center",
                       fontsize=7, color="0.2")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, max(100, max(expected.max(), ep.max(), pep.max()) + 10))
    ax.set_ylabel("%")
    smooth_lbl = "smoothed" if smoothed else "unsmoothed"
    ax.set_title(f"{title_prefix}{roi_label}  ·  SPM-style RFX BMS  ·  "
                   f"{smooth_lbl}  ·  n={pivot.shape[0]}  ·  "
                   f"BOR={bor:.3f}",
                   fontsize=10, color="0.15")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    sns.despine(fig=fig, offset=4, trim=False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def discover_subjects():
    """Subjects appearing in any aprf*.cv directory or vonmises*.cv."""
    seen = set()
    for sub_glob in ("aprf*.cv/sub-*", "vonmises*.cv/sub-*"):
        for p in DERIV.glob(sub_glob):
            seen.add(p.name.removeprefix("sub-"))
    return sorted(seen, key=lambda s: (0 if s[0].isdigit() else 1, s))


def run(subjects, out):
    if subjects is None:
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No subjects with CV outputs found.")
    print(f"Subjects: {subjects}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        for smoothed in (False, True):
            print(f"\n=== {'smoothed' if smoothed else 'unsmoothed'} ===")
            print("  NPCr (value coding)…")
            page(NPCR_LADDER, subjects, "NPCr (value)",
                 "NPCr", None, smoothed, pdf, title_prefix="")
            page_signal_winners(NPCR_LADDER, subjects, "NPCr (value)",
                                "NPCr", None, smoothed, pdf)
            print("  V1 (orientation coding)…")
            page(V1_LADDER, subjects, "V1 (orientation)",
                 "BensonV1", "LR", smoothed, pdf, title_prefix="")
            page_signal_winners(V1_LADDER, subjects, "V1 (orientation)",
                                "BensonV1", "LR", smoothed, pdf)
            print("  NPCr (orientation vs value DISSOCIATION)…")
            page(DISSOC_LADDER, subjects,
                 "NPCr (orientation vs value)",
                 "NPCr", None, smoothed, pdf, title_prefix="")
            page_signal_winners(DISSOC_LADDER, subjects,
                                "NPCr (orientation vs value)",
                                "NPCr", None, smoothed, pdf)
            print("  NPCr Bayesian hierarchical DM…")
            page_bayes_proportions(DISSOC_LADDER, subjects,
                                   "NPCr (orientation vs value)",
                                   "NPCr", None, smoothed, pdf)
            print("  NPCr SPM-style RFX BMS…")
            page_rfx_bms(DISSOC_LADDER, subjects,
                         "NPCr (orientation vs value)",
                         "NPCr", None, smoothed, pdf)
            print("  V1 (orientation vs value DISSOCIATION)…")
            page(DISSOC_LADDER, subjects,
                 "V1 (orientation vs value)",
                 "BensonV1", "LR", smoothed, pdf, title_prefix="")
            page_signal_winners(DISSOC_LADDER, subjects,
                                "V1 (orientation vs value)",
                                "BensonV1", "LR", smoothed, pdf)
            print("  V1 Bayesian hierarchical DM…")
            page_bayes_proportions(DISSOC_LADDER, subjects,
                                   "V1 (orientation vs value)",
                                   "BensonV1", "LR", smoothed, pdf)
            print("  V1 SPM-style RFX BMS…")
            page_rfx_bms(DISSOC_LADDER, subjects,
                         "V1 (orientation vs value)",
                         "BensonV1", "LR", smoothed, pdf)
    print(f"\nWrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, Path(args.out))


if __name__ == "__main__":
    main()
