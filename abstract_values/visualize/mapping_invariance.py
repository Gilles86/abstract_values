"""V1 vs NPCr: does the decoder follow the *physical stimulus* or the
*learned mapping*?

The disambiguation figure of the project. Two panels share an orientation
x-axis (0°–180°). Within each panel, both task conditions (CDF and
Inverse-CDF) are drawn separately, plus the theoretical mappings as
gray dashed reference lines.

  - V1 (vonmises, decoded orientation): theoretical reference = identity
    (orientation in = orientation out). Both condition lines should sit
    on identity — V1's representation of a given gabor orientation is
    the same whether the subject is currently in CDF or InvCDF.

  - NPCr (aprf-session-shift, decoded value, projected onto orientation
    via the per-condition CHF↔orientation lookup): theoretical reference
    = the two mapping curves V_CDF(θ) and V_InvCDF(θ). The two condition
    lines should track their respective mapping curves — NPCr decodes
    the *learned value*, which inverts when the mapping inverts.

By default uses spherical-noise expected-decoded TSVs and nv=fdr05 voxel
selection (the cleanest combination, see compare_noise_models.pdf).

Usage:
    python -m abstract_values.visualize.mapping_invariance
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
DECODING_ROOT = Path(BIDS_FOLDER) / "derivatives" / "decoding"
DEFAULT_OUT = (Path(BIDS_FOLDER) / "derivatives" / "qa"
               / "mapping_invariance.pdf")

COND_COLOUR = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}
TRAINED_MIN, TRAINED_MAX = 7.5, 172.5

# Selection / noise defaults — these wins from compare_noise_models.pdf
# (spherical halves MAE; fdr05 keeps a principled threshold). Override
# via CLI for sensitivity checks.
DEFAULT_SEL = "nvoxels-fdr05"
DEFAULT_NOISE = "spherical"


def _v1_tsv(subject, session, sel_tag, smoothed, noise,
             session_shift=False):
    """Path to the per-session V1 EU TSV. When ``session_shift=True``,
    look in ``vonmises-session-shift/`` (the per-session-weights output
    that lets V1 voxels remap between conditions); otherwise the
    legacy joint-fit ``vonmises/`` output."""
    smooth = "_smoothed" if smoothed else ""
    noise_tag = f"_noise-{noise}" if noise else ""
    subdir = "vonmises-session-shift" if session_shift else "vonmises"
    return (DERIV / subdir / f"sub-{subject}" / f"ses-{session}" / "func"
            / f"sub-{subject}_ses-{session}_task-abstractvalue"
              f"_mask-BensonV1_hemi-LR_{sel_tag}_nsims-1000"
              f"{noise_tag}{smooth}_desc-expected_decoded_orientation_pe.tsv")


def _npcr_tsv(subject, session, sel_tag, smoothed, noise):
    smooth = "_smoothed" if smoothed else ""
    noise_tag = f"_noise-{noise}" if noise else ""
    return (DERIV / "aprf-session-shift" / f"sub-{subject}"
            / f"ses-{session}" / "func"
            / f"sub-{subject}_ses-{session}_task-abstractvalue"
              f"_mask-NPCr_{sel_tag}_nsims-1000"
              f"{noise_tag}{smooth}_desc-expected_decoded_pe.tsv")


def discover_subjects():
    seen = set()
    for p in (DERIV / "vonmises").glob("sub-*"):
        seen.add(p.name.removeprefix("sub-"))
    for p in (DERIV / "aprf-session-shift").glob("sub-*"):
        seen.add(p.name.removeprefix("sub-"))
    return sorted(seen, key=lambda s: (0 if s[0].isdigit() else 1, s))


def _orientation_lookup(subjects):
    """Per-condition (orientation_deg → value_chf) lookup from gabor events.
    This is the *theoretical mapping* that NPCr's decoded value should
    track inside each condition."""
    pairs = {"cdf": set(), "inverse_cdf": set()}
    for s in subjects:
        try:
            sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
            for ses in sub.get_sessions():
                cond = sub.get_mapping(ses)
                ev = sub.get_events(ses, sub.get_runs(ses))
                for _, row in ev[ev.event_type == "gabor"].iterrows():
                    pairs[cond].add((float(row["orientation"]),
                                      float(row["value"])))
        except Exception:
            pass
    out = {}
    for c, ps in pairs.items():
        out[c] = (pd.DataFrame(sorted(ps),
                                columns=["orientation_deg", "value_chf"])
                   .drop_duplicates("orientation_deg")
                   .sort_values("orientation_deg")
                   .reset_index(drop=True))
    return out


def load_v1(subjects, sel_tag, smoothed, noise, session_shift=False):
    rows = []
    for s in subjects:
        try:
            sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
        except Exception:
            continue
        for ses in sub.get_sessions():
            p = _v1_tsv(s, ses, sel_tag, smoothed, noise,
                         session_shift=session_shift)
            if not p.exists():
                continue
            df = pd.read_csv(p, sep="\t")
            # V1 'value' column is orientation in radians
            df["orientation_deg"] = np.rad2deg(df["value"])
            df["decoded_deg"]      = np.rad2deg(df["mean_E"])
            # SD of the decoded posterior in deg (var_E is in radians²)
            df["decoded_sd_deg"]   = np.rad2deg(np.sqrt(df["var_E"]))
            df["subject"] = s
            df["session"] = ses
            df["condition"] = sub.get_mapping(ses)
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_npcr(subjects, sel_tag, smoothed, noise, lookup):
    """Loaded TSV has 'value' in CHF + 'mean_E' as decoded CHF. We invert
    via the per-condition lookup to put orientation on the x-axis."""
    rows = []
    for s in subjects:
        try:
            sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
        except Exception:
            continue
        for ses in sub.get_sessions():
            p = _npcr_tsv(s, ses, sel_tag, smoothed, noise)
            if not p.exists():
                continue
            df = pd.read_csv(p, sep="\t")
            df["subject"] = s
            df["session"] = ses
            df["condition"] = sub.get_mapping(ses)
            df["decoded_chf"]    = df["mean_E"]
            df["decoded_sd_chf"] = np.sqrt(df["var_E"])
            df["true_chf"]       = df["value"]
            # Map true CHF back to orientation via this condition's lookup
            lut = lookup.get(df["condition"].iloc[0])
            if lut is None or lut.empty:
                continue
            df["orientation_deg"] = np.interp(
                df["true_chf"].values,
                lut["value_chf"].values,
                lut["orientation_deg"].values,
                left=np.nan, right=np.nan)
            rows.append(df)
    return (pd.concat(rows, ignore_index=True).dropna(subset=["orientation_deg"])
            if rows else pd.DataFrame())


def _load_behavior_bias():
    """Per-(condition, orientation, subject) signed BDM bid bias from raw
    behavior events. Returns DataFrame with columns
    ``[condition, orientation_deg, subject, bias_chf]`` — one row per cell.

    Lazy import so this script still works if the behavior data isn't
    locally available (figure just skips the behavior overlay)."""
    try:
        from abstract_values.behavior.data import get_all_behavioral_data
    except Exception:
        return pd.DataFrame()
    df = get_all_behavioral_data().reset_index()
    df = df[(df["event_type"] == "feedback") & df["response"].notna()].copy()
    df["response_num"] = pd.to_numeric(df["response"], errors="coerce")
    df = df.dropna(subset=["response_num", "value", "orientation"])
    df["bias_chf"] = df["response_num"] - df["value"]
    out = (df.groupby(["mapping", "orientation", "subject"])["bias_chf"]
             .mean().reset_index()
             .rename(columns={"mapping": "condition",
                                "orientation": "orientation_deg"}))
    return out


def _load_pars_empirical(subjects, folder, mask, true_col,
                          sel_tag, smoothed, noise_label,
                          to_orientation_fn=None):
    """Per-trial posterior **mean** and **SD** from pars.tsv files,
    aggregated by (subject, session, condition, true_stim bin).
    Returns one row per cell with columns ``decoded_mean`` and
    ``posterior_sd``.

    The empirical decoded mean is the per-trial sum p_i · v_i over the
    posterior, then averaged across trials with the same true stimulus.
    This is the quantity that actually tests the 'decoder follows the
    mapping' claim — far more interesting than the model-simulated
    ``mean_E`` from the EU pipeline, because it's measured on real
    held-out BOLD trials rather than predicted from the encoding fit.
    """
    base = DECODING_ROOT / folder
    if not base.exists():
        return pd.DataFrame()
    nv_tag = sel_tag.replace("nvoxels-", "")
    smooth_seg = "_smoothed" if smoothed else ""
    out_rows = []
    for sub_dir in sorted(base.glob("sub-*")):
        subj = sub_dir.name.removeprefix("sub-")
        if subj not in subjects:
            continue
        try:
            sub_obj = Subject(subj, bids_folder=Path(BIDS_FOLDER))
        except Exception:
            continue
        glob_pat = (f"sub-{subj}_mask-{mask}_nvoxels-{nv_tag}"
                     f"_noise-{noise_label}{smooth_seg}*_pars.tsv")
        for p in (sub_dir / "func").glob(glob_pat):
            df = pd.read_csv(p, sep="\t")
            if df.empty or true_col not in df.columns:
                continue
            bins = []
            for c in df.columns:
                try: bins.append(float(c))
                except ValueError: pass
            bins = np.asarray(bins, dtype=np.float64)
            probs = df[[str(b) for b in bins]].to_numpy(dtype=np.float64)
            probs = probs / probs.sum(axis=1, keepdims=True)
            mean = probs @ bins
            second = probs @ (bins ** 2)
            sd = np.sqrt(np.maximum(second - mean ** 2, 0.0))
            df["decoded_value"] = mean
            df["posterior_sd"]  = sd
            df["true_stim"]     = df[true_col]
            cond_lookup = {ses: sub_obj.get_mapping(ses)
                           for ses in sub_obj.get_sessions()}
            df["condition"] = df["session"].map(cond_lookup)
            df = df.dropna(subset=["condition"])
            df["subject"] = subj
            agg = (df.groupby(
                ["subject", "session", "condition", "true_stim"])
                    [["decoded_value", "posterior_sd"]]
                    .mean().reset_index())
            if to_orientation_fn is not None:
                agg["orientation_deg"] = agg.apply(
                    lambda r: to_orientation_fn(r["true_stim"],
                                                  r["condition"]), axis=1)
            else:
                agg["orientation_deg"] = np.rad2deg(agg["true_stim"])
            out_rows.append(agg)
    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True).dropna(
        subset=["orientation_deg"])


def _aggregate(df, x_col, y_col, ori_grid):
    """For each subject, interpolate y_col onto ori_grid; return cohort
    mean + SEM."""
    per_sub = []
    for _, g in df.groupby("subject"):
        g = g.sort_values(x_col)
        if g[x_col].nunique() < 3:
            continue
        per_sub.append(np.interp(ori_grid, g[x_col].values, g[y_col].values,
                                  left=np.nan, right=np.nan))
    if not per_sub:
        return None, None, 0
    per_sub = np.asarray(per_sub)
    mean = np.nanmean(per_sub, axis=0)
    n_eff = np.maximum(np.sum(~np.isnan(per_sub), axis=0), 1)
    sem = np.nanstd(per_sub, axis=0, ddof=1) / np.sqrt(n_eff)
    return mean, sem, per_sub.shape[0]


def _draw_line_with_band(ax, x, mean, sem, color, label, lw=2.0):
    ax.plot(x, mean, color=color, lw=lw, label=label)
    ax.fill_between(x, mean - sem, mean + sem,
                     color=color, alpha=0.22, linewidth=0)


def page(subjects, sel_tag, smoothed, noise, lookup, pdf):
    # Simulated curves (from EU TSV — sqrt(var_E)) used for panels B/D
    # only. The "decoded mean" panels A/C below use **empirical** per-trial
    # posterior means from real held-out BOLD (pars.tsv), not the
    # simulator's mean_E. The simulated mean would just retrace the
    # encoding model and isn't an interesting test of the mapping
    # invariance / remapping claim.
    # Prefer V1 session-shift EU TSVs when they exist (the new fair-
    # comparison weights — V1 voxels allowed to remap between conditions).
    # Fall back to the joint vonmises fit if the session-shift batch
    # hasn't landed yet for this (sel, smooth, noise) cell.
    df_v1_sim = load_v1(subjects, sel_tag, smoothed, noise,
                          session_shift=True)
    v1_sim_source = "session-shift"
    if df_v1_sim.empty:
        df_v1_sim = load_v1(subjects, sel_tag, smoothed, noise,
                              session_shift=False)
        v1_sim_source = "joint"
    df_npcr_sim = load_npcr(subjects, sel_tag, smoothed, noise, lookup)

    # Empirical (real-trial) data. Per ROI, prefer spherical real-trial
    # decodes when available; fall back to noise-full when the spherical
    # batch hasn't completed for that selection × ROI cell.
    def _try_load(folder, mask, true_col, to_ori=None):
        for label in ("spherical", "full"):
            df = _load_pars_empirical(
                subjects, folder, mask, true_col,
                sel_tag, smoothed, label, to_orientation_fn=to_ori)
            if not df.empty:
                return df, label
        return pd.DataFrame(), "none"

    df_v1_real,   v1_noise   = _try_load(
        "gabor", "BensonV1", "true_orientation_rad")
    def _npcr_chf_to_ori(chf, condition):
        lut = lookup.get(condition)
        if lut is None or lut.empty:
            return np.nan
        return float(np.interp(chf, lut["value_chf"].values,
                                 lut["orientation_deg"].values,
                                 left=np.nan, right=np.nan))
    df_npcr_real, npcr_noise = _try_load(
        "value", "NPCr", "true_value_chf", to_ori=_npcr_chf_to_ori)
    real_noise_label = (f"V1:{v1_noise}, NPCr:{npcr_noise}"
                         if v1_noise != npcr_noise
                         else v1_noise)
    # Convert V1 decoded mean to degrees (TSV has radians).
    # Bias = decoded - true (same units as the decoded variable).
    if not df_v1_real.empty:
        df_v1_real["decoded_deg"]      = np.rad2deg(df_v1_real["decoded_value"])
        df_v1_real["posterior_sd_deg"] = np.rad2deg(df_v1_real["posterior_sd"])
        df_v1_real["bias_deg"]         = (df_v1_real["decoded_deg"]
                                            - df_v1_real["orientation_deg"])
        # Mapping-specific bias = raw bias minus the across-condition mean
        # per orientation. Removes the shared regression-toward-prior-median
        # bias (which is identical between conditions because they share a
        # prior median) and keeps only the condition-specific divergence.
        df_v1_real["bias_demeaned_deg"] = (
            df_v1_real["bias_deg"]
            - df_v1_real.groupby("orientation_deg")["bias_deg"]
                          .transform("mean"))
    if not df_npcr_real.empty:
        df_npcr_real["decoded_chf"] = df_npcr_real["decoded_value"]
        df_npcr_real["bias_chf"]    = (df_npcr_real["decoded_value"]
                                         - df_npcr_real["true_stim"])
        df_npcr_real["bias_demeaned_chf"] = (
            df_npcr_real["bias_chf"]
            - df_npcr_real.groupby("orientation_deg")["bias_chf"]
                            .transform("mean"))

    # Behavioral bid bias per (condition, orientation): mean(bid − true_CHF).
    df_behavior_bias = _load_behavior_bias()
    if not df_behavior_bias.empty:
        df_behavior_bias["bias_demeaned_chf"] = (
            df_behavior_bias["bias_chf"]
            - df_behavior_bias.groupby("orientation_deg")["bias_chf"]
                                 .transform("mean"))

    # Use simulated data only for the SD panels; empirical for the mean.
    df_v1   = df_v1_sim if df_v1_sim is not None else pd.DataFrame()
    df_npcr = df_npcr_sim if df_npcr_sim is not None else pd.DataFrame()
    if df_v1.empty and df_npcr.empty and df_v1_real.empty and df_npcr_real.empty:
        return

    ori_grid = np.linspace(TRAINED_MIN, TRAINED_MAX, 60)
    # 2×3 paper double-column (7.25" wide). Rows = ROI (V1 top, NPCr
    # bottom); columns = quantity (decoded mean | expected SD | bias).
    # The bias panel includes a thin gray within-orientation CDF−InvCDF
    # difference so the shared regression-to-mean cancels and any
    # remaining structure is mapping-specific.
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 6.0),
                              constrained_layout=True)
    # No suptitle — the figure should stand on the panels alone.
    # Caption (not rendered) carries n, error metric, model details.
    smooth_lbl = "smoothed" if smoothed else "unsmoothed"

    # Direct-label helpers: write the condition name at the right endpoint
    # of each curve in the curve's own color. Pulls the legend onto the
    # data and removes the eight separate frame-less legend boxes.
    def _label_endpoint(ax, x, y, color, text, dx=2.0, dy=0.0, fontsize=8):
        ax.text(x[-1] + dx, y[-1] + dy, text, color=color, fontsize=fontsize,
                ha="left", va="center", fontweight="bold")

    def _common_x(ax):
        ax.set_xlim(TRAINED_MIN, TRAINED_MAX + 25)  # +25 leaves room for endpoint labels
        ax.set_xticks([15, 45, 90, 135, 165])
        ax.set_xlabel("Orientation (deg)")

    def _panel_letter(ax, letter):
        ax.text(-0.18, 1.05, letter,
                transform=ax.transAxes, fontsize=12, fontweight="bold",
                va="bottom", ha="right")

    # ═══ A — V1 decoded orientation: condition-invariant (lies on identity) ═
    ax = axes[0, 0]
    ax.plot([TRAINED_MIN, TRAINED_MAX], [TRAINED_MIN, TRAINED_MAX],
            color="0.55", ls="--", lw=0.9, zorder=0)
    # Inline label on the identity line — no arrow, rotated 45° to lie on
    # the line itself. Placed at ~135° to leave room for the right-endpoint
    # condition labels.
    ax.text(135, 142, "y = x", color="0.4", fontsize=7.5,
             rotation=45, rotation_mode="anchor",
             ha="center", va="bottom")
    # Empirical decoded orientation from real held-out trials (pars.tsv).
    cond_lines_v1 = []
    v1_iter = (df_v1_real.groupby("condition")
                if (not df_v1_real.empty and "condition" in df_v1_real.columns)
                else iter([]))
    for cond, sub in v1_iter:
        mean, sem, n = _aggregate(sub, "orientation_deg", "decoded_deg",
                                    ori_grid)
        if mean is None: continue
        _draw_line_with_band(ax, ori_grid, mean, sem,
                              COND_COLOUR[cond], "_nolegend_")
        cond_lines_v1.append((cond, ori_grid, mean, n))
    # Direct labels at right endpoint
    for cond, x, y, n in cond_lines_v1:
        # Stagger labels vertically so cdf/invcdf don't overprint
        dy = 8 if cond == "cdf" else -8
        _label_endpoint(ax, x, y, COND_COLOUR[cond],
                          "CDF" if cond == "cdf" else "InvCDF", dy=dy)
    _common_x(ax)
    ax.set_ylim(TRAINED_MIN, TRAINED_MAX)
    ax.set_yticks([15, 45, 90, 135, 165])
    ax.set_xlim(TRAINED_MIN, TRAINED_MAX + 25)
    ax.set_ylabel("V1 decoded orientation (deg)")
    _panel_letter(ax, "A")

    # ═══ B — V1 expected SD: condition-invariant precision ═════════════
    ax = axes[0, 1]
    cond_lines_v1_sd = []
    for cond, sub in df_v1.groupby("condition"):
        mean, sem, n = _aggregate(sub, "orientation_deg", "decoded_sd_deg",
                                    ori_grid)
        if mean is None: continue
        _draw_line_with_band(ax, ori_grid, mean, sem,
                              COND_COLOUR[cond], "_nolegend_", lw=1.6)
        cond_lines_v1_sd.append((cond, ori_grid, mean, n))
    sd_v1_max = max((float(np.nanmax(m)) for _, _, m, _ in cond_lines_v1_sd),
                    default=1.0)
    for cond, x, y, n in cond_lines_v1_sd:
        dy = sd_v1_max * 0.06 if cond == "cdf" else -sd_v1_max * 0.06
        _label_endpoint(ax, x, y, COND_COLOUR[cond],
                          "CDF" if cond == "cdf" else "InvCDF", dy=dy)
    _common_x(ax)
    ax.set_ylim(0, sd_v1_max * 1.20)
    ax.set_ylabel("V1 expected SD (deg)")
    _panel_letter(ax, "B")

    # ═══ C — NPCr decoded value: each condition tracks its own mapping ═
    ax = axes[1, 0]
    # Theoretical mappings as gray dashed reference (NOT condition-colored
    # — the empirical lines should carry the condition color; the
    # references just show what each mapping looks like).
    for cond in ("cdf", "inverse_cdf"):
        lut = lookup.get(cond)
        if lut is None or lut.empty: continue
        ax.plot(lut["orientation_deg"], lut["value_chf"],
                color="0.55", ls="--", lw=0.9, zorder=0)
    # Empirical decoded value from real held-out trials (pars.tsv).
    cond_lines_npcr = []
    npcr_iter = (df_npcr_real.groupby("condition")
                  if (not df_npcr_real.empty
                      and "condition" in df_npcr_real.columns)
                  else iter([]))
    for cond, sub in npcr_iter:
        mean, sem, n = _aggregate(sub, "orientation_deg", "decoded_chf",
                                    ori_grid)
        if mean is None: continue
        _draw_line_with_band(ax, ori_grid, mean, sem,
                              COND_COLOUR[cond], "_nolegend_")
        cond_lines_npcr.append((cond, ori_grid, mean, n))
    # Direct labels at right endpoint
    for cond, x, y, n in cond_lines_npcr:
        _label_endpoint(ax, x, y, COND_COLOUR[cond],
                          "CDF" if cond == "cdf" else "InvCDF")
    # Inline label on one of the gray dashed mapping curves — no arrow.
    # The grey dashed style + the matching color of the data line is
    # enough; the label just names what the gray reference IS.
    if lookup.get("cdf") is not None and not lookup["cdf"].empty:
        lut = lookup["cdf"]
        # Pick a mid-point of the CDF mapping near the LEFT edge so it
        # doesn't collide with the right-endpoint condition labels.
        idx = max(1, len(lut) // 6)
        ax.text(lut["orientation_deg"].iloc[idx],
                 lut["value_chf"].iloc[idx] - 2.0,
                 "Theoretical mapping", color="0.4", fontsize=7,
                 ha="left", va="top", style="italic")
    _common_x(ax)
    ax.set_ylabel("NPCr decoded value (CHF)")
    _panel_letter(ax, "C")

    # ═══ D — NPCr expected SD: precision is mapping-adapted ════════════
    ax = axes[1, 1]
    cond_lines_npcr_sd = []
    for cond, sub in df_npcr.groupby("condition"):
        mean, sem, n = _aggregate(sub, "orientation_deg", "decoded_sd_chf",
                                    ori_grid)
        if mean is None: continue
        _draw_line_with_band(ax, ori_grid, mean, sem,
                              COND_COLOUR[cond], "_nolegend_", lw=1.6)
        cond_lines_npcr_sd.append((cond, ori_grid, mean, n))
    sd_npcr_max = max((float(np.nanmax(m)) for _, _, m, _ in cond_lines_npcr_sd),
                      default=1.0)
    for cond, x, y, n in cond_lines_npcr_sd:
        dy = sd_npcr_max * 0.05 if cond == "cdf" else -sd_npcr_max * 0.05
        _label_endpoint(ax, x, y, COND_COLOUR[cond],
                          "CDF" if cond == "cdf" else "InvCDF", dy=dy)
    _common_x(ax)
    ax.set_ylim(0, sd_npcr_max * 1.20)
    ax.set_ylabel("NPCr expected SD (CHF)")
    _panel_letter(ax, "D")

    # ═══ E — V1 mapping-specific bias (cross-condition mean subtracted) ═
    # By construction the two condition curves are mirror images about
    # zero (subtracting the within-orientation mean of both conditions
    # leaves equal-and-opposite residuals). The amplitude is what matters:
    # zero amplitude means V1 has no condition-specific bias.
    ax = axes[0, 2]
    ax.axhline(0, color="0.6", lw=0.7, zorder=0)
    bias_iter = (df_v1_real.groupby("condition")
                  if (not df_v1_real.empty
                      and "condition" in df_v1_real.columns)
                  else iter([]))
    v1_bias_max = 0.0
    for cond, sub in bias_iter:
        mean, sem, n = _aggregate(sub, "orientation_deg",
                                    "bias_demeaned_deg", ori_grid)
        if mean is None: continue
        _draw_line_with_band(ax, ori_grid, mean, sem,
                              COND_COLOUR[cond], "_nolegend_", lw=1.5)
        amp = float(np.nanmax(np.abs(mean)))
        v1_bias_max = max(v1_bias_max, amp + float(np.nanmax(sem)))
        dy = amp * 0.18 if cond == "cdf" else -amp * 0.18
        _label_endpoint(ax, ori_grid, mean, COND_COLOUR[cond],
                          "CDF" if cond == "cdf" else "InvCDF", dy=dy)
    _common_x(ax)
    if v1_bias_max > 0:
        ax.set_ylim(-v1_bias_max * 1.20, v1_bias_max * 1.20)
    ax.set_ylabel("V1 mapping-specific bias (deg)")
    _panel_letter(ax, "E")

    # ═══ F — NPCr mapping-specific bias + behavioral overlay ═══════════
    # Same demeaning as panel E. Behavioral BDM bid bias gets the same
    # treatment (subtract within-orientation mean across conditions) so
    # the comparison is apples-to-apples — both 'how much does this
    # readout deviate from the prior-driven baseline in a way that
    # depends on which mapping is active'.
    ax = axes[1, 2]
    ax.axhline(0, color="0.6", lw=0.7, zorder=0)
    bias_iter = (df_npcr_real.groupby("condition")
                  if (not df_npcr_real.empty
                      and "condition" in df_npcr_real.columns)
                  else iter([]))
    npcr_bias_max = 0.0
    for cond, sub in bias_iter:
        mean, sem, n = _aggregate(sub, "orientation_deg",
                                    "bias_demeaned_chf", ori_grid)
        if mean is None: continue
        _draw_line_with_band(ax, ori_grid, mean, sem,
                              COND_COLOUR[cond], "_nolegend_", lw=1.5)
        amp = float(np.nanmax(np.abs(mean)))
        npcr_bias_max = max(npcr_bias_max, amp + float(np.nanmax(sem)))
        dy = amp * 0.18 if cond == "cdf" else -amp * 0.18
        _label_endpoint(ax, ori_grid, mean, COND_COLOUR[cond],
                          "CDF" if cond == "cdf" else "InvCDF", dy=dy)
    if not df_behavior_bias.empty:
        for cond, sub in df_behavior_bias.groupby("condition"):
            mean, sem, n = _aggregate(sub, "orientation_deg",
                                        "bias_demeaned_chf", ori_grid)
            if mean is None: continue
            ax.plot(ori_grid, mean, color=COND_COLOUR[cond], lw=1.2,
                    ls="--", alpha=0.75, label="_nolegend_")
        # Single label naming the dashed lines.
        ax.text(0.02, 0.97, "Dashed: behavioural BDM bid bias",
                transform=ax.transAxes, fontsize=7, color="0.35",
                ha="left", va="top", style="italic")
    _common_x(ax)
    if npcr_bias_max > 0:
        ax.set_ylim(-npcr_bias_max * 1.20, npcr_bias_max * 1.20)
    ax.set_ylabel("NPCr mapping-specific bias (CHF)")
    _panel_letter(ax, "F")

    # Trim spines — keep custom x-limit (with label-room) by not asking
    # for trim, which would chop the extra 25° we added.
    sns.despine(fig=fig, offset=4)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def run(subjects, out, sel_tag, noise, both_smoothings):
    if subjects is None:
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No subjects discovered.")
    print(f"Subjects: {subjects}")
    lookup = _orientation_lookup(subjects)
    for c, lut in lookup.items():
        print(f"  lookup[{c}]: {len(lut)} (orientation, CHF) pairs")

    smoothings = (False, True) if both_smoothings else (False,)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        for smoothed in smoothings:
            print(f"\n=== {sel_tag}  noise={noise}  smoothed={smoothed} ===")
            page(subjects, sel_tag, smoothed, noise, lookup, pdf)
    print(f"\nWrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--selection", default=DEFAULT_SEL,
                    choices=["nvoxels-100", "nvoxels-fdr05",
                              "nvoxels-50", "nvoxels-250"])
    p.add_argument("--noise", default=DEFAULT_NOISE,
                    choices=["spherical", ""],
                    help="'spherical' or '' for residual (no tag).")
    p.add_argument("--both-smoothings", action="store_true",
                    help="Render unsmoothed + smoothed pages (default: "
                         "unsmoothed only — smoothed hurts decoding on "
                         "this cohort, see decoding_accuracy_correlation.pdf).")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args.subjects, Path(args.out), args.selection, args.noise,
        args.both_smoothings)


if __name__ == "__main__":
    main()
