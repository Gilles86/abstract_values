"""Bias and absolute error vs. orientation, identical vs. mode-shift decoder,
both conditions overlaid.

Bias = mean(decoded - true value)   -- signed, shows systematic over/under-
       estimation as a function of orientation.
Error = mean abs(decoded - true value)  -- magnitude, already shown per-
        decoder elsewhere but not side-by-side with bias.

Decoder distinguished by linestyle (solid = mode-shift, dashed = identical),
condition by color (COND_COLOUR), so all 4 combinations sit in one panel
per row.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

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

# ── load + merge identical + mode-shift TSVs ────────────────────────────
rows = []
for s in ALL_SUBJECTS:
    p_id = (DERIV / f"sub-{s}"
            / f"sub-{s}_task-abstractvalue_mask-NPCr_nvoxels-100_noise-spherical_desc-identicaldecoded_pe.tsv")
    p_cc = (DERIV / f"sub-{s}"
            / f"sub-{s}_task-abstractvalue_mask-NPCr_nvoxels-100_noise-spherical_desc-crossdecoded_pe.tsv")
    if not (p_id.exists() and p_cc.exists()):
        continue
    d_id = pd.read_csv(p_id, sep="\t")
    d_cc = pd.read_csv(p_cc, sep="\t")[["run", "trial_nr", "test_session", "matched_mean"]]
    d = d_id.merge(d_cc, on=["run", "trial_nr", "test_session"], how="inner")
    d["subject"] = s
    rows.append(d)
df = pd.concat(rows, ignore_index=True)
df["bias_identical"] = df["identical_mean"] - df["true_value"]
df["bias_modeshift"] = df["matched_mean"] - df["true_value"]
df["err_identical"] = df["bias_identical"].abs()
df["err_modeshift"] = df["bias_modeshift"].abs()
print(f"{df['subject'].nunique()} subjects, {len(df)} trials")

# ── per-condition (orientation_deg <-> value) lookup ────────────────────
pairs = {"cdf": set(), "inverse_cdf": set()}
for s in ALL_SUBJECTS:
    try:
        sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
        for ses in sub.get_sessions():
            cond = sub.get_mapping(ses)
            ev = sub.get_events(ses, sub.get_runs(ses))
            for _, row in ev[ev.event_type == "gabor"].iterrows():
                pairs[cond].add((float(row["orientation"]), float(row["value"])))
    except Exception:
        pass
lookup = {c: pd.DataFrame(sorted(p), columns=["orientation_deg", "value"])
             .drop_duplicates("value").sort_values("value").reset_index(drop=True)
          for c, p in pairs.items()}

out_rows = []
for cond, sub in df.groupby("test_condition"):
    lut = lookup[cond]
    sub = sub.copy()
    sub["orientation_deg"] = np.interp(sub["true_value"].values,
                                        lut["value"].values, lut["orientation_deg"].values,
                                        left=np.nan, right=np.nan)
    out_rows.append(sub)
df = pd.concat(out_rows, ignore_index=True).dropna(subset=["orientation_deg"])

n_bins = 14
bins = np.linspace(0, 180, n_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2
df["bin"] = pd.cut(df["orientation_deg"], bins, labels=False, include_lowest=True)

group = (df.groupby(["test_condition", "bin"])
         [["bias_identical", "bias_modeshift", "err_identical", "err_modeshift"]]
         .agg(["mean", "sem"]).reset_index())

DECODER_STYLE = {"identical": dict(ls="--", lw=1.6, alpha_band=0.12),
                 "modeshift": dict(ls="-", lw=2.0, alpha_band=0.22)}
DECODER_LABEL = {"identical": "Identical", "modeshift": "Mode-shift"}

fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.5), constrained_layout=True, sharex=True)

# ── top: bias ────────────────────────────────────────────────────────────
ax = axes[0]
ax.axhline(0, color="0.6", lw=0.8, zorder=0)
for cond, g in group.groupby("test_condition"):
    g = g.sort_values("bin")
    x = bin_centers[g["bin"].astype(int).values]
    for dec in ("identical", "modeshift"):
        m = g[(f"bias_{dec}", "mean")]
        s = g[(f"bias_{dec}", "sem")]
        st = DECODER_STYLE[dec]
        ax.plot(x, m, color=COND_COLOUR[cond], ls=st["ls"], lw=st["lw"],
                marker="o", ms=3,
                label=f"{COND_LABEL[cond]} — {DECODER_LABEL[dec]}")
        ax.fill_between(x, m - s, m + s, color=COND_COLOUR[cond],
                        alpha=st["alpha_band"], linewidth=0)
ax.set_ylabel("Bias: mean(decoded − true value)  (CHF)")
ax.set_title("Decoder bias vs. orientation\n"
             "(solid: mode-shift, dashed: identical)", fontsize=10, color="0.2")
ax.legend(loc="upper left", fontsize=7.5, ncol=2)

# ── bottom: absolute error ─────────────────────────────────────────────
ax = axes[1]
for cond, g in group.groupby("test_condition"):
    g = g.sort_values("bin")
    x = bin_centers[g["bin"].astype(int).values]
    for dec in ("identical", "modeshift"):
        m = g[(f"err_{dec}", "mean")]
        s = g[(f"err_{dec}", "sem")]
        st = DECODER_STYLE[dec]
        ax.plot(x, m, color=COND_COLOUR[cond], ls=st["ls"], lw=st["lw"],
                marker="o", ms=3,
                label=f"{COND_LABEL[cond]} — {DECODER_LABEL[dec]}")
        ax.fill_between(x, m - s, m + s, color=COND_COLOUR[cond],
                        alpha=st["alpha_band"], linewidth=0)
ax.set_xlabel("Orientation (deg)")
ax.set_ylabel("Mean abs(decoded − true value)  (CHF)")
ax.set_title("Decoder absolute error vs. orientation\n"
             "(solid: mode-shift, dashed: identical)", fontsize=10, color="0.2")
ax.set_xlim(0, 180)
ax.legend(loc="upper center", fontsize=7.5, ncol=2)

sns.despine(fig=fig, offset=4, trim=True)
fig.text(0.5, -0.012, "Decoding from: NPCr, top-100 voxels by R² (own model's r2), "
         "spherical noise, unsmoothed, real single-trial GLMsingle betas. "
         "Identical: standard aPRF (one mode/fwhm/amp/baseline, jointly fit). "
         "Mode-shift: session-shift aPRF (mode differs per condition, rest shared), matched session.",
         ha="center", va="top", fontsize=7, color="0.4", style="italic")

out = QA / "decoder_bias_error_vs_orientation.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Wrote {out}")
