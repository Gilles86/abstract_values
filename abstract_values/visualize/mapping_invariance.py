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


def _v1_tsv(subject, session, sel_tag, smoothed, noise):
    smooth = "_smoothed" if smoothed else ""
    noise_tag = f"_noise-{noise}" if noise else ""
    return (DERIV / "vonmises" / f"sub-{subject}" / f"ses-{session}" / "func"
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


def load_v1(subjects, sel_tag, smoothed, noise):
    rows = []
    for s in subjects:
        try:
            sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
        except Exception:
            continue
        for ses in sub.get_sessions():
            p = _v1_tsv(s, ses, sel_tag, smoothed, noise)
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


def _load_pars_posterior_sd(subjects, folder, mask, true_col,
                              sel_tag, smoothed, noise_label,
                              to_orientation_fn=None):
    """Compute per-trial posterior SD from pars.tsv files and aggregate
    by (subject, session, condition, true-stimulus bin). 'Decoded
    uncertainty' as defined in the literature: SD of the posterior
    P(stim|y) per trial, NOT the SD of point estimates across trials
    (which is what the EU pipeline's var_E captures).

    `to_orientation_fn(true_val, condition)` optional: when given,
    converts the true_value into orientation_deg so V1 and NPCr can
    share an x-axis.
    """
    from re import match
    base = DECODING_ROOT / folder
    if not base.exists():
        return pd.DataFrame()
    # nv may need a different naming token in pars.tsv; the decoding
    # filename convention uses nvoxels-<token> too.
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
        # Decoding outputs are not per-session — they pool the full
        # held-out set. Each row carries its own session column.
        # `noise-<label>` is part of the filename; allow either explicit
        # spherical or full.
        glob_pat = (f"sub-{subj}_mask-{mask}_nvoxels-{nv_tag}"
                     f"_noise-{noise_label}{smooth_seg}*_pars.tsv")
        for p in (sub_dir / "func").glob(glob_pat):
            df = pd.read_csv(p, sep="\t")
            if df.empty or true_col not in df.columns:
                continue
            # bin centres = numeric column names
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
            df["posterior_sd"] = sd
            df["true_stim"]    = df[true_col]
            # Session → condition lookup
            cond_lookup = {ses: sub_obj.get_mapping(ses)
                           for ses in sub_obj.get_sessions()}
            df["condition"] = df["session"].map(cond_lookup)
            df = df.dropna(subset=["condition"])
            df["subject"] = subj
            # Aggregate per (subject, session, condition, true_stim bin)
            agg = (df.groupby(
                ["subject", "session", "condition", "true_stim"])
                    ["posterior_sd"].mean().reset_index())
            if to_orientation_fn is not None:
                agg["orientation_deg"] = agg.apply(
                    lambda r: to_orientation_fn(r["true_stim"],
                                                  r["condition"]), axis=1)
            else:
                # V1: true_stim is orientation in radians
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
    df_v1   = load_v1(subjects, sel_tag, smoothed, noise)
    df_npcr = load_npcr(subjects, sel_tag, smoothed, noise, lookup)
    # Decoded uncertainty (per-trial posterior SD) from real pars.tsv.
    # Currently only the legacy `noise-full` real-trial decoder exists
    # cohort-wide; the spherical real-trial batch was submitted but may
    # still be in flight. Pull whichever variant is most populated.
    real_noise_label = "spherical"
    df_v1_real   = _load_pars_posterior_sd(
        subjects, "gabor", "BensonV1", "true_orientation_rad",
        sel_tag, smoothed, real_noise_label)
    if df_v1_real.empty:
        real_noise_label = "full"
        df_v1_real = _load_pars_posterior_sd(
            subjects, "gabor", "BensonV1", "true_orientation_rad",
            sel_tag, smoothed, real_noise_label)
    # NPCr: needs CHF→orientation lookup
    def _npcr_chf_to_ori(chf, condition):
        lut = lookup.get(condition)
        if lut is None or lut.empty:
            return np.nan
        return float(np.interp(chf, lut["value_chf"].values,
                                 lut["orientation_deg"].values,
                                 left=np.nan, right=np.nan))
    df_npcr_real = _load_pars_posterior_sd(
        subjects, "value", "NPCr", "true_value_chf",
        sel_tag, smoothed, real_noise_label,
        to_orientation_fn=_npcr_chf_to_ori)
    # SD is in posterior_sd; for V1 convert radians→degrees.
    if not df_v1_real.empty:
        df_v1_real["posterior_sd_deg"] = np.rad2deg(df_v1_real["posterior_sd"])

    if df_v1.empty and df_npcr.empty:
        return

    # Empirical bias = decoded mean minus true (NPCr in CHF, V1 in degrees).
    # Subject to central-tendency (regression-to-mean) bias — but plotting
    # both conditions overlaid makes the condition-specific deviation
    # readable: the central-tendency component is shared, the divergence
    # between curves is the mapping-specific piece.
    if not df_v1.empty:
        df_v1["bias_deg"] = df_v1["decoded_deg"] - df_v1["orientation_deg"]
    if not df_npcr.empty:
        df_npcr["bias_chf"] = df_npcr["decoded_chf"] - df_npcr["true_chf"]

    ori_grid = np.linspace(TRAINED_MIN, TRAINED_MAX, 60)
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 7.5),
                              constrained_layout=True)
    smooth_lbl = "smoothed" if smoothed else "unsmoothed"
    fig.suptitle(
        "V1 vs NPCr: physical-stimulus code vs adapted-mapping code\n"
        f"({sel_tag} · simulated noise={noise} · real-trial noise="
        f"{real_noise_label} · {smooth_lbl})",
        fontsize=11, y=1.02, color="0.15")

    def _styled(ax, xlabel, ylabel, title, ylim=None):
        ax.set_xlim(TRAINED_MIN, TRAINED_MAX)
        ax.set_xticks([15, 45, 90, 135, 165])
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8.5, color="0.2")

    # ═══ ROW 0: V1 ══════════════════════════════════════════════════════
    # ─── (0,0) Decoded mean — should sit on identity in BOTH conditions ─
    ax = axes[0, 0]
    ax.plot([TRAINED_MIN, TRAINED_MAX], [TRAINED_MIN, TRAINED_MAX],
            color="0.6", ls="--", lw=1.0, label="Identity", zorder=0)
    for cond, sub in df_v1.groupby("condition"):
        mean, sem, n = _aggregate(sub, "orientation_deg", "decoded_deg",
                                    ori_grid)
        if mean is None: continue
        _draw_line_with_band(ax, ori_grid, mean, sem,
                              COND_COLOUR[cond], f"{cond}  (n={n})")
    ax.set_ylim(TRAINED_MIN, TRAINED_MAX)
    ax.set_yticks([15, 45, 90, 135, 165])
    ax.set_aspect("equal")
    _styled(ax, "True orientation (deg)", "V1 decoded θ (deg)",
            "V1: decoded ORIENTATION (condition-invariant)")
    ax.legend(loc="lower right", fontsize=7)

    # ─── (0,1) Expected uncertainty (V1) ────────────────────────────────
    ax = axes[0, 1]
    sd_max = 0
    for cond, sub in df_v1.groupby("condition"):
        mean, sem, n = _aggregate(sub, "orientation_deg", "decoded_sd_deg",
                                    ori_grid)
        if mean is None: continue
        _draw_line_with_band(ax, ori_grid, mean, sem,
                              COND_COLOUR[cond], f"{cond}  (n={n})", lw=1.8)
        sd_max = max(sd_max, float(np.nanmax(mean + sem)))
    _styled(ax, "True orientation (deg)", "V1 expected SD (deg)",
            "V1 expected uncertainty (sim, sqrt(var_E))",
            ylim=(0, sd_max * 1.15) if sd_max > 0 else None)
    ax.legend(loc="upper right", fontsize=7)

    # ─── (0,2) Decoded uncertainty (V1, real-trial posterior SD) ───────
    ax = axes[0, 2]
    if df_v1_real.empty:
        ax.text(0.5, 0.5, f"No V1 pars.tsv (noise={real_noise_label})\nfor "
                "this selection — real-trial spherical batch may still be in flight.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="0.5"); ax.set_xticks([]); ax.set_yticks([])
    else:
        v1_max = 0
        for cond, sub in df_v1_real.groupby("condition"):
            mean, sem, n = _aggregate(sub, "orientation_deg",
                                        "posterior_sd_deg", ori_grid)
            if mean is None: continue
            _draw_line_with_band(ax, ori_grid, mean, sem,
                                  COND_COLOUR[cond], f"{cond}  (n={n})", lw=1.8)
            v1_max = max(v1_max, float(np.nanmax(mean + sem)))
        _styled(ax, "True orientation (deg)",
                "V1 posterior SD per trial (deg)",
                f"V1 decoded uncertainty (real, noise={real_noise_label})",
                ylim=(0, v1_max * 1.15) if v1_max > 0 else None)
        ax.legend(loc="upper right", fontsize=7)

    # ─── (0,3) Empirical bias (V1: decoded - true, per condition) ──────
    ax = axes[0, 3]
    ax.axhline(0, color="0.6", lw=0.8, zorder=0)
    for cond, sub in df_v1.groupby("condition"):
        mean, sem, n = _aggregate(sub, "orientation_deg", "bias_deg",
                                    ori_grid)
        if mean is None: continue
        _draw_line_with_band(ax, ori_grid, mean, sem,
                              COND_COLOUR[cond], f"{cond}  (n={n})", lw=1.8)
    _styled(ax, "True orientation (deg)",
            "V1 bias (decoded − true)  (deg)",
            "V1 bias: should be ~0, condition-invariant")
    ax.legend(loc="upper right", fontsize=7)

    # ═══ ROW 1: NPCr ════════════════════════════════════════════════════
    # ─── (1,0) Decoded mean — conditions track their mapping ───────────
    ax = axes[1, 0]
    for cond in ("cdf", "inverse_cdf"):
        lut = lookup.get(cond)
        if lut is None or lut.empty: continue
        ax.plot(lut["orientation_deg"], lut["value_chf"],
                color=COND_COLOUR[cond], ls="--", lw=1.0, alpha=0.7,
                label=f"{cond} mapping", zorder=0)
    for cond, sub in df_npcr.groupby("condition"):
        mean, sem, n = _aggregate(sub, "orientation_deg", "decoded_chf",
                                    ori_grid)
        if mean is None: continue
        _draw_line_with_band(ax, ori_grid, mean, sem,
                              COND_COLOUR[cond], f"{cond} decoded  (n={n})")
    _styled(ax, "True orientation (deg)", "NPCr decoded V (CHF)",
            "NPCr: decoded VALUE follows the active mapping")
    ax.legend(loc="upper right", fontsize=7)

    # ─── (1,1) Expected uncertainty (NPCr) ─────────────────────────────
    ax = axes[1, 1]
    sd_max = 0
    for cond, sub in df_npcr.groupby("condition"):
        mean, sem, n = _aggregate(sub, "orientation_deg", "decoded_sd_chf",
                                    ori_grid)
        if mean is None: continue
        _draw_line_with_band(ax, ori_grid, mean, sem,
                              COND_COLOUR[cond], f"{cond}  (n={n})", lw=1.8)
        sd_max = max(sd_max, float(np.nanmax(mean + sem)))
    _styled(ax, "True orientation (deg)", "NPCr expected SD (CHF)",
            "NPCr expected uncertainty — mapping-specific (adapted)",
            ylim=(0, sd_max * 1.15) if sd_max > 0 else None)
    ax.legend(loc="upper right", fontsize=7)

    # ─── (1,2) Decoded uncertainty (NPCr, real-trial posterior SD) ─────
    ax = axes[1, 2]
    if df_npcr_real.empty:
        ax.text(0.5, 0.5, f"No NPCr pars.tsv (noise={real_noise_label})\nfor "
                "this selection — real-trial spherical batch may still be in flight.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="0.5"); ax.set_xticks([]); ax.set_yticks([])
    else:
        n_max = 0
        for cond, sub in df_npcr_real.groupby("condition"):
            mean, sem, n = _aggregate(sub, "orientation_deg",
                                        "posterior_sd", ori_grid)
            if mean is None: continue
            _draw_line_with_band(ax, ori_grid, mean, sem,
                                  COND_COLOUR[cond], f"{cond}  (n={n})", lw=1.8)
            n_max = max(n_max, float(np.nanmax(mean + sem)))
        _styled(ax, "True orientation (deg)",
                "NPCr posterior SD per trial (CHF)",
                f"NPCr decoded uncertainty (real, noise={real_noise_label})",
                ylim=(0, n_max * 1.15) if n_max > 0 else None)
        ax.legend(loc="upper right", fontsize=7)

    # ─── (1,3) Empirical bias (NPCr: decoded - true, per condition) ────
    # The central-tendency component is shared across conditions
    # (regression to the prior median); the *divergence* between curves
    # is the mapping-specific bias — the part of the story you cannot
    # explain with regression-to-mean alone.
    ax = axes[1, 3]
    ax.axhline(0, color="0.6", lw=0.8, zorder=0)
    for cond, sub in df_npcr.groupby("condition"):
        mean, sem, n = _aggregate(sub, "orientation_deg", "bias_chf",
                                    ori_grid)
        if mean is None: continue
        _draw_line_with_band(ax, ori_grid, mean, sem,
                              COND_COLOUR[cond], f"{cond}  (n={n})", lw=1.8)
    _styled(ax, "True orientation (deg)",
            "NPCr bias (decoded − true)  (CHF)",
            "NPCr bias: diverges between conditions (mapping-specific)")
    ax.legend(loc="upper right", fontsize=7)

    sns.despine(fig=fig, offset=5, trim=True)
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
