"""Identical (single, non-shifted) decoder vs. mode-shift-only decoder.

"Identical" = standard aPRF fit (one mode/fwhm/amplitude/baseline per
voxel, jointly fit across both conditions) -- computed by
compute_identical_decoding_aprf.py.

"Mode-shift" = session-shift aPRF fit (mode differs per condition,
fwhm/amplitude/baseline shared) decoded with each trial's OWN matched
session -- the "matched_mean"/"matched_sd" columns already computed by
compute_cross_condition_decoding_aprf.py.

Both use identical voxel selection, noise-model fitting, and decode-grid
machinery, so the only thing that differs is whether the tuning curve's
preferred value is allowed to shift between conditions. Tests directly
whether that shift improves real-trial decoding, complementing the
earlier cross-condition / shuffled-null result (which showed decoding
with the WRONG session's mode is much worse than the right one).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

from abstract_values.utils.data import BIDS_FOLDER, Subject

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150, "savefig.dpi": 300,
})
sns.set_context("paper")
COND_COLOUR = {"cdf": "#E76F51", "inverse_cdf": "#2A9D8F"}
COND_LABEL = {"cdf": "CDF", "inverse_cdf": "InvCDF"}
QA = Path("/data/ds-abstractvalue/derivatives/qa")
DERIV = Path(BIDS_FOLDER) / "derivatives" / "encoding_models" / "aprf-session-shift"

ALL_SUBJECTS = [f"{i:02d}" for i in range(3, 25)] + ["pil01", "pil02"]

# ── load identical-decoder + mode-shift-matched-decoder TSVs, merge per trial
rows = []
for s in ALL_SUBJECTS:
    p_id = (DERIV / f"sub-{s}"
            / f"sub-{s}_task-abstractvalue_mask-NPCr_nvoxels-100_noise-spherical_desc-identicaldecoded_pe.tsv")
    p_cc = (DERIV / f"sub-{s}"
            / f"sub-{s}_task-abstractvalue_mask-NPCr_nvoxels-100_noise-spherical_desc-crossdecoded_pe.tsv")
    if not (p_id.exists() and p_cc.exists()):
        print(f"  skip sub-{s}: missing TSV")
        continue
    d_id = pd.read_csv(p_id, sep="\t")
    d_cc = pd.read_csv(p_cc, sep="\t")[["run", "trial_nr", "test_session",
                                        "matched_mean", "matched_sd"]]
    d = d_id.merge(d_cc, on=["run", "trial_nr", "test_session"], how="inner")
    d["subject"] = s
    rows.append(d)
df = pd.concat(rows, ignore_index=True)
df["abs_err_identical"] = (df["identical_mean"] - df["true_value"]).abs()
df["abs_err_modeshift"] = (df["matched_mean"] - df["true_value"]).abs()
print(f"{df['subject'].nunique()} subjects, {len(df)} trials")

per_sub = df.groupby("subject")[["abs_err_identical", "abs_err_modeshift",
                                  "identical_sd", "matched_sd"]].mean()
per_sub_cond = df.groupby(["subject", "test_condition"])[
    ["abs_err_identical", "abs_err_modeshift"]].mean().reset_index()

t, p = stats.ttest_rel(per_sub["abs_err_modeshift"], per_sub["abs_err_identical"])
w, pw = stats.wilcoxon(per_sub["abs_err_modeshift"], per_sub["abs_err_identical"])
n_better = int((per_sub["abs_err_modeshift"] < per_sub["abs_err_identical"]).sum())
n_sub = len(per_sub)
print(f"MAE: identical mean={per_sub['abs_err_identical'].mean():.2f}  "
      f"mode-shift mean={per_sub['abs_err_modeshift'].mean():.2f}")
print(f"{n_better}/{n_sub} subjects: mode-shift better than identical")
print(f"paired t={t:.2f} p={p:.2e}   wilcoxon p={pw:.2e}")

t_sd, p_sd = stats.ttest_rel(per_sub["matched_sd"], per_sub["identical_sd"])
n_better_sd = int((per_sub["matched_sd"] < per_sub["identical_sd"]).sum())
print(f"SD: identical mean={per_sub['identical_sd'].mean():.2f}  "
      f"mode-shift mean={per_sub['matched_sd'].mean():.2f}  "
      f"{n_better_sd}/{n_sub} sharper  t={t_sd:.2f} p={p_sd:.2e}")

out = QA / "identical_vs_modeshift_decoding.pdf"
with PdfPages(out) as pdf:

    # ── Page 1: MAE + SD paired comparison ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    ax = axes[0]
    for cond, sub in per_sub_cond.groupby("test_condition"):
        ax.scatter(sub["abs_err_identical"], sub["abs_err_modeshift"],
                  color=COND_COLOUR[cond], s=30, edgecolor="0.3", linewidth=0.5,
                  label=COND_LABEL[cond], zorder=3)
    lim = (per_sub_cond[["abs_err_identical", "abs_err_modeshift"]].min().min() * 0.9,
           per_sub_cond[["abs_err_identical", "abs_err_modeshift"]].max().max() * 1.05)
    ax.plot(lim, lim, "--", color="0.5", lw=0.9, zorder=0)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("MAE, identical decoder (CHF)")
    ax.set_ylabel("MAE, mode-shift decoder (CHF)")
    ax.set_title(f"MAE  ({n_better}/{n_sub} subjects better with mode-shift)\n"
                f"paired t={t:.2f}, p={p:.1e}", fontsize=9.5, color="0.2")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    ax.scatter(per_sub["identical_sd"], per_sub["matched_sd"],
              color="#264653", s=30, edgecolor="0.3", linewidth=0.5, zorder=3)
    lim2 = (per_sub[["identical_sd", "matched_sd"]].min().min() * 0.9,
            per_sub[["identical_sd", "matched_sd"]].max().max() * 1.05)
    ax.plot(lim2, lim2, "--", color="0.5", lw=0.9, zorder=0)
    ax.set_xlim(lim2); ax.set_ylim(lim2)
    ax.set_xlabel("Posterior SD, identical decoder (CHF)")
    ax.set_ylabel("Posterior SD, mode-shift decoder (CHF)")
    ax.set_title(f"Posterior SD  ({n_better_sd}/{n_sub} sharper with mode-shift)\n"
                f"paired t={t_sd:.2f}, p={p_sd:.1e}", fontsize=9.5, color="0.2")
    ax.set_aspect("equal", adjustable="box")

    sns.despine(fig=fig, offset=4)
    fig.suptitle("Identical (single, non-shifted) decoder vs. mode-shift-only decoder\n"
                f"n={n_sub} subjects (full cohort)", fontsize=10, color="0.3", y=1.08)
    fig.text(0.5, -0.02, "Decoding from: NPCr, top-100 voxels by R² (own model's r2), "
             "spherical noise, unsmoothed, real single-trial GLMsingle betas. "
             "Identical: standard aPRF (one mode/fwhm/amp/baseline, jointly fit). "
             "Mode-shift: session-shift aPRF (mode differs per condition, rest shared), matched session.",
             ha="center", va="top", fontsize=7, color="0.4", style="italic")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── Page 2: per-condition MAE bars ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    summary = per_sub_cond.groupby("test_condition")[
        ["abs_err_identical", "abs_err_modeshift"]].agg(["mean", "sem"])
    x = np.arange(2)
    w = 0.35
    for i, cond in enumerate(("cdf", "inverse_cdf")):
        vals = [summary.loc[cond, ("abs_err_identical", "mean")],
                summary.loc[cond, ("abs_err_modeshift", "mean")]]
        errs = [summary.loc[cond, ("abs_err_identical", "sem")],
                summary.loc[cond, ("abs_err_modeshift", "sem")]]
        ax.bar(x + (i - 0.5) * w, vals, width=w, yerr=errs, capsize=3,
              color=COND_COLOUR[cond], alpha=[0.5, 1.0][0] if False else None,
              label=COND_LABEL[cond])
        # distinguish identical (lighter) vs mode-shift (solid) via hatch
        ax.patches[-2].set_alpha(0.45)
        ax.patches[-1].set_alpha(1.0)
    ax.set_xticks(x); ax.set_xticklabels(["Identical", "Mode-shift"])
    ax.set_ylabel("Mean abs(decoded − true value)  (CHF)")
    ax.set_title("MAE by condition and decoder\n(light bar: identical, solid bar: mode-shift)",
                fontsize=10, color="0.2")
    ax.legend(loc="upper right", fontsize=8)
    sns.despine(fig=fig, offset=4)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"\nWrote {out}")
