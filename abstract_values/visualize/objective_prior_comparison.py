from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
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

ALL_SUBJECTS = [f"{i:02d}" for i in range(3, 25)] + ["pil01", "pil02"]  # n=24, full cohort
MP_SUBJECTS = ALL_SUBJECTS
EU_SUBJECTS = ALL_SUBJECTS

out = QA / "objective_prior_comparison.pdf"

with PdfPages(out) as pdf:

    # ═══ Page 1+2: real single-trial matched decoder, flat vs objective prior ═
    rows = []
    for s in MP_SUBJECTS:
        p = (DERIV / f"sub-{s}"
             / f"sub-{s}_task-abstractvalue_mask-NPCr_nvoxels-100_noise-spherical_desc-matcheddecodedprior_pe.tsv")
        if not p.exists():
            continue
        df = pd.read_csv(p, sep="\t")
        df["subject"] = s
        rows.append(df)
    mp = pd.concat(rows, ignore_index=True)
    mp["abs_err_flat"] = (mp["flat_mean"] - mp["true_value"]).abs()
    mp["abs_err_objective"] = (mp["objective_mean"] - mp["true_value"]).abs()

    print(f"Matched-decoder-prior: {mp['subject'].nunique()} subjects, {len(mp)} trials")
    per_sub_mp = mp.groupby("subject")[["abs_err_flat", "abs_err_objective", "flat_sd", "objective_sd"]].mean()
    print(per_sub_mp.round(2))

    # Page 1: MAE + SD, flat vs objective, paired per subject
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.scatter(per_sub_mp["abs_err_flat"], per_sub_mp["abs_err_objective"],
               color="#E9C46A", s=32, edgecolor="0.3", linewidth=0.5, zorder=3)
    lim = (per_sub_mp[["abs_err_flat", "abs_err_objective"]].min().min() * 0.9,
           per_sub_mp[["abs_err_flat", "abs_err_objective"]].max().max() * 1.05)
    ax.plot(lim, lim, "--", color="0.5", lw=0.9, zorder=0)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("MAE, flat prior (CHF)")
    ax.set_ylabel("MAE, objective prior (CHF)")
    t, p_ = stats.ttest_rel(per_sub_mp["abs_err_objective"], per_sub_mp["abs_err_flat"])
    n_better = int((per_sub_mp["abs_err_objective"] < per_sub_mp["abs_err_flat"]).sum())
    ax.set_title(f"MAE  ({n_better}/{len(per_sub_mp)} better with objective prior)\n"
                f"paired t={t:.2f}, p={p_:.3f}", fontsize=9.5, color="0.2")
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    ax.scatter(per_sub_mp["flat_sd"], per_sub_mp["objective_sd"],
               color="#264653", s=32, edgecolor="0.3", linewidth=0.5, zorder=3)
    lim2 = (per_sub_mp[["flat_sd", "objective_sd"]].min().min() * 0.9,
            per_sub_mp[["flat_sd", "objective_sd"]].max().max() * 1.05)
    ax.plot(lim2, lim2, "--", color="0.5", lw=0.9, zorder=0)
    ax.set_xlim(lim2); ax.set_ylim(lim2)
    ax.set_xlabel("Posterior SD, flat prior (CHF)")
    ax.set_ylabel("Posterior SD, objective prior (CHF)")
    t2, p2_ = stats.ttest_rel(per_sub_mp["objective_sd"], per_sub_mp["flat_sd"])
    n_better2 = int((per_sub_mp["objective_sd"] < per_sub_mp["flat_sd"]).sum())
    ax.set_title(f"Posterior SD  ({n_better2}/{len(per_sub_mp)} sharper with objective prior)\n"
                f"paired t={t2:.2f}, p={p2_:.3f}", fontsize=9.5, color="0.2")
    ax.set_aspect("equal", adjustable="box")

    sns.despine(fig=fig, offset=4)
    fig.suptitle("Real single-trial matched decoder, flat vs. objective prior\n"
                f"n={len(per_sub_mp)} subjects (full cohort)",
                fontsize=10, color="0.3", y=1.08)
    fig.text(0.5, -0.02, "Decoding from: NPCr, top-100 voxels by R² (session-shift aPRF fit), "
             "spherical noise, unsmoothed, real single-trial GLMsingle betas",
             ha="center", va="top", fontsize=7.5, color="0.4", style="italic")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # Page 2: error vs VALUE, flat vs objective, density overlay per condition
    # -- direct test of whether the objective prior recovers density-tracking
    # that error_vs_density.py found absent under the flat prior.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True, sharey=True)
    val_grid = np.linspace(mp["true_value"].min(), mp["true_value"].max(), 200)
    n_bins = 12
    vbins = np.linspace(mp["true_value"].min(), mp["true_value"].max(), n_bins + 1)
    vbin_centers = (vbins[:-1] + vbins[1:]) / 2
    for ax, cond in zip(axes, ("cdf", "inverse_cdf")):
        sub = mp[mp.test_condition == cond].copy()
        dens = gaussian_kde(sub["true_value"].values)(val_grid)
        ax2 = ax.twinx()
        ax2.fill_between(val_grid, dens, color=COND_COLOUR[cond], alpha=0.12, linewidth=0)
        ax2.set_yticks([]); ax2.set_ylabel("")
        ax2.spines["right"].set_visible(False); ax2.spines["top"].set_visible(False)

        sub["vbin"] = pd.cut(sub["true_value"], vbins, labels=False, include_lowest=True)
        flat_g = sub.groupby("vbin")["abs_err_flat"].agg(["mean", "sem"]).reindex(range(n_bins))
        obj_g = sub.groupby("vbin")["abs_err_objective"].agg(["mean", "sem"]).reindex(range(n_bins))
        ax.errorbar(vbin_centers, flat_g["mean"], yerr=flat_g["sem"], color="0.55", lw=1.6,
                    ls="--", marker="o", ms=3, label="Flat prior")
        ax.errorbar(vbin_centers, obj_g["mean"], yerr=obj_g["sem"], color=COND_COLOUR[cond],
                    lw=2.0, marker="o", ms=3.5, label="Objective prior")

        valid = flat_g["mean"].notna()
        dens_at_bins = np.interp(vbin_centers, val_grid, dens)
        r_flat, p_flat = stats.spearmanr(dens_at_bins[valid.values], flat_g["mean"][valid].values)
        r_obj, p_obj = stats.spearmanr(dens_at_bins[valid.values], obj_g["mean"][valid].values)
        ax.set_xlabel("True value (CHF)")
        ax.set_title(f"{COND_LABEL[cond]}  (density shaded)\n"
                    f"Spearman(density,err): flat r={r_flat:.2f} p={p_flat:.3f}  |  "
                    f"objective r={r_obj:.2f} p={p_obj:.3f}",
                    fontsize=8.5, color=COND_COLOUR[cond])
        ax.legend(loc="upper center", fontsize=7.5)
    axes[0].set_ylabel("Mean abs(decoded − true)  (CHF)")
    sns.despine(fig=fig, offset=4, trim=True)
    fig.suptitle("Does the objective prior recover density-tracking that the flat prior lacked?\n"
                f"n={mp['subject'].nunique()} subjects (full cohort)", fontsize=10, color="0.3", y=1.12)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ═══ Page 3: simulated EU (predicted decoding variance), flat vs objective ═
    none_df = pd.read_csv(QA / "expected_uncertainty_per_condition_spherical.tsv", sep="\t")
    none_df = none_df[(none_df["variant"] == "unsmoothed")
                      & (none_df["subject"].astype(str).isin(EU_SUBJECTS))].copy()
    none_df["prior"] = "flat"

    obj_rows = []
    for s in EU_SUBJECTS:
        sub_obj = Subject(s, bids_folder=Path(BIDS_FOLDER))
        for ses in sub_obj.get_sessions():
            p = (DERIV / f"sub-{s}" / f"ses-{ses}" / "func"
                 / f"sub-{s}_ses-{ses}_task-abstractvalue_mask-NPCr_nvoxels-100_nsims-1000"
                   f"_noise-spherical_prior-objective_desc-expected_decoded_pe.tsv")
            if not p.exists():
                continue
            d = pd.read_csv(p, sep="\t")
            d["subject"] = s; d["session"] = ses
            d["condition"] = sub_obj.get_mapping(ses)
            d["sd_E"] = np.sqrt(d["var_E"])
            obj_rows.append(d)
    obj_df = pd.concat(obj_rows, ignore_index=True)
    obj_df["prior"] = "objective"
    print(f"\nEU objective prior: {obj_df['subject'].nunique()} subjects, {len(obj_df)} value points")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True, sharey=True)
    for ax, cond in zip(axes, ("cdf", "inverse_cdf")):
        n_none = none_df[none_df.condition == cond]
        n_obj = obj_df[obj_df.condition == cond]

        bins2 = np.linspace(min(n_none.value.min(), n_obj.value.min()),
                            max(n_none.value.max(), n_obj.value.max()), n_bins + 1)
        bc2 = (bins2[:-1] + bins2[1:]) / 2
        n_none = n_none.copy(); n_none["vbin"] = pd.cut(n_none.value, bins2, labels=False, include_lowest=True)
        n_obj = n_obj.copy(); n_obj["vbin"] = pd.cut(n_obj.value, bins2, labels=False, include_lowest=True)
        g_none = n_none.groupby("vbin")["sd_E"].agg(["mean", "sem"]).reindex(range(n_bins))
        g_obj = n_obj.groupby("vbin")["sd_E"].agg(["mean", "sem"]).reindex(range(n_bins))

        ax.errorbar(bc2, g_none["mean"], yerr=g_none["sem"], color="0.55", lw=1.6,
                    ls="--", marker="o", ms=3, label="Flat prior")
        ax.errorbar(bc2, g_obj["mean"], yerr=g_obj["sem"], color=COND_COLOUR[cond],
                    lw=2.0, marker="o", ms=3.5, label="Objective prior")
        ax.set_xlabel("True value (CHF)")
        ax.set_title(f"{COND_LABEL[cond]}", fontsize=10, color=COND_COLOUR[cond])
        ax.legend(loc="upper center", fontsize=7.5)
    axes[0].set_ylabel(r"$\sqrt{\mathrm{Var}[\hat{V}]}$ across simulations (CHF)")
    sns.despine(fig=fig, offset=4, trim=True)
    fig.suptitle("Predicted decoding variance (simulated EU), flat vs. objective prior\n"
                f"n={obj_df['subject'].nunique()} subjects (full cohort)", fontsize=10, color="0.3", y=1.12)
    fig.text(0.5, -0.02, "Decoding from: NPCr, top-100 voxels by R² (session-shift aPRF fit), "
             "spherical noise, unsmoothed, simulated (get_expected_uncertainty / --prior objective)",
             ha="center", va="top", fontsize=7.5, color="0.4", style="italic")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"Wrote {out}")
