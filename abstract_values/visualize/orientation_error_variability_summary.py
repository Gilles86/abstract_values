from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from abstract_values.utils.data import BIDS_FOLDER, Subject
from abstract_values.behavior.data import get_all_behavioral_data

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

# ── per-condition (orientation_deg <-> value) lookup, pooled across subjects
subjects_all = [s.name.removeprefix("sub-") for s in
                sorted((Path(BIDS_FOLDER) / "derivatives" / "encoding_models"
                        / "aprf-session-shift").glob("sub-*"))]
pairs = {"cdf": set(), "inverse_cdf": set()}
for s in subjects_all:
    try:
        sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
        for ses in sub.get_sessions():
            cond = sub.get_mapping(ses)
            ev = sub.get_events(ses, sub.get_runs(ses))
            for _, row in ev[ev.event_type == "gabor"].iterrows():
                pairs[cond].add((float(row["orientation"]), float(row["value"])))
    except Exception:
        pass
lookup = {}
for cond, ps in pairs.items():
    lookup[cond] = (pd.DataFrame(sorted(ps), columns=["orientation_deg", "value"])
                     .drop_duplicates("value").sort_values("value").reset_index(drop=True))


def to_orientation(df, value_col, cond_col):
    out = []
    for cond, sub in df.groupby(cond_col):
        lut = lookup[cond]
        sub = sub.copy()
        sub["orientation_deg"] = np.interp(sub[value_col].values,
                                            lut["value"].values, lut["orientation_deg"].values,
                                            left=np.nan, right=np.nan)
        out.append(sub)
    return pd.concat(out, ignore_index=True)


n_bins = 14
bins = np.linspace(0, 180, n_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2


def make_page(df, metric_col, title, ylabel, pdf, subject_col="subject", cond_col="condition",
             source=""):
    df = df.dropna(subset=["orientation_deg", metric_col]).copy()
    df[subject_col] = df[subject_col].astype(str)
    df["bin"] = pd.cut(df["orientation_deg"], bins, labels=False, include_lowest=True)

    per_sub = (df.groupby([subject_col, cond_col, "bin"])[metric_col]
               .mean().reset_index())
    group = (per_sub.groupby([cond_col, "bin"])[metric_col]
             .agg(["mean", "sem"]).reset_index())

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.0), constrained_layout=True, sharex=True)

    ax = axes[0]
    for cond, g in group.groupby(cond_col):
        g = g.sort_values("bin")
        x = bin_centers[g["bin"].astype(int).values]
        ax.plot(x, g["mean"], color=COND_COLOUR[cond], lw=2.0, marker="o", ms=3.5,
                label=COND_LABEL[cond])
        ax.fill_between(x, g["mean"] - g["sem"], g["mean"] + g["sem"],
                        color=COND_COLOUR[cond], alpha=0.22, linewidth=0)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\ngroup mean ± SEM  (n={df[subject_col].nunique()} subjects)",
                fontsize=10, color="0.2")
    ax.legend(loc="best", fontsize=8, ncol=2)

    ax2 = axes[1]
    for cond, g in per_sub.groupby(cond_col):
        for s, gs in g.groupby(subject_col):
            gs = gs.sort_values("bin")
            x = bin_centers[gs["bin"].astype(int).values]
            ax2.plot(x, gs[metric_col], color=COND_COLOUR[cond], lw=0.8,
                    alpha=0.45, zorder=1)
    for cond, g in group.groupby(cond_col):
        g = g.sort_values("bin")
        x = bin_centers[g["bin"].astype(int).values]
        ax2.plot(x, g["mean"], color=COND_COLOUR[cond], lw=2.2, zorder=3)
    ax2.set_xlabel("Orientation (deg)")
    ax2.set_ylabel(ylabel)
    ax2.set_title("Per-subject variability  (thin: individual subjects, thick: group mean)",
                 fontsize=10, color="0.2")
    ax2.set_xlim(0, 180)

    sns.despine(fig=fig, offset=4, trim=True)
    if source:
        fig.text(0.5, -0.012, f"Decoding from: {source}", ha="center", va="top",
                 fontsize=7.5, color="0.4", style="italic")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


out = QA / "orientation_error_variability_summary.pdf"
with PdfPages(out) as pdf:

    # ── Page 1+2: matched ("correct") decoder, real single-trial decoding ──
    real_df = pd.read_csv(QA / "cross_condition_decoding_summary.tsv", sep="\t")
    real_df["abs_err_matched"] = (real_df["matched_mean"] - real_df["true_value"]).abs()
    real_df = to_orientation(real_df, "true_value", "test_condition")

    real_source = ("NPCr, top-100 voxels by R² (session-shift aPRF fit), spherical noise, "
                  "unsmoothed — real single-trial GLMsingle betas, session-matched tuning curves "
                  "(mode_<session>, shared fwhm/amplitude/baseline)")

    make_page(real_df, "abs_err_matched",
             "Matched (\"correct\") decoder — absolute error, abs(decoded − true value)",
             "Mean abs(decoded − true value)  (CHF)", pdf,
             cond_col="test_condition", source=real_source)

    make_page(real_df, "matched_sd",
             "Matched (\"correct\") decoder — decoded uncertainty (posterior SD, real trials)",
             "Posterior SD of decoded value (CHF)", pdf,
             cond_col="test_condition", source=real_source)

    # ── Page 3: predicted decoding variance, expected_uncertainty machinery ─
    eu_df = pd.read_csv(QA / "expected_uncertainty_per_condition_spherical.tsv", sep="\t")
    eu_df = eu_df[eu_df["variant"] == "unsmoothed"].copy()
    eu_df = to_orientation(eu_df, "value", "condition")

    make_page(eu_df, "sd_E",
             "Predicted decoding variance (simulated expected-uncertainty, spherical noise)",
             r"$\sqrt{\mathrm{Var}[\hat{V}]}$ across simulations (CHF)", pdf,
             cond_col="condition",
             source=("NPCr, top-100 voxels by R² (session-shift aPRF fit), spherical noise, "
                     "unsmoothed — simulated: encoding model + fitted noise model -> 1000 simulated "
                     "responses per stimulus -> decoded posterior (get_expected_uncertainty)"))

    # ── Page 4: Fisher information (full-noise + spherical variants combined
    # -- disjoint subject sets: 03-10+pilots full-noise, 11-24 spherical) ───
    fi_full = pd.read_csv(QA / "fisher_information_per_condition.tsv", sep="\t")
    fi_sph  = pd.read_csv(QA / "fisher_information_per_condition_spherical.tsv", sep="\t")
    fi_df = pd.concat([fi_full, fi_sph], ignore_index=True)
    fi_df = fi_df[fi_df["variant"] == "unsmoothed"].copy()
    fi_df = to_orientation(fi_df, "value", "condition")

    make_page(fi_df, "fi",
             "Fisher information  (full-noise: subs 03-10+pilots, spherical: subs 11-24; "
             "disjoint subject sets, combined for n=24 coverage)",
             "Population Fisher information (a.u.)", pdf,
             cond_col="condition",
             source=("NPCr, top-N voxels by R² (session-shift aPRF fit), unsmoothed — analytic "
                     "Fisher information from fitted noise model (noise variant differs by subject: "
                     "full-covariance for 03-10+pilots, spherical for 11-24)"))

    # ── Page 5: behavioral BDM bid error (not fMRI-derived -- the actual
    # bid subjects placed vs the objective CHF value; canonical recipe from
    # project CLAUDE.md). Uses ALL study subjects with behavior (n=28,
    # includes behavior-only subjects beyond the n=24 with fMRI decoding). ──
    beh = get_all_behavioral_data()
    beh = beh[beh["event_type"] == "feedback"].reset_index().copy()
    beh["response"] = pd.to_numeric(beh["response"], errors="coerce")
    beh = beh.dropna(subset=["response", "value", "orientation"])
    beh["abs_error"] = (beh["response"] - beh["value"]).abs()
    beh = beh.rename(columns={"mapping": "condition", "orientation": "orientation_deg"})
    beh["subject"] = beh["subject"].astype(str)

    make_page(beh, "abs_error",
             "Behavioral BDM bid — absolute error, abs(bid − objective value)  "
             "[NOT fMRI-derived]",
             "Mean abs(bid − value)  (CHF)", pdf,
             cond_col="condition",
             source="Raw BDM auction bids vs. objective CHF value, all study subjects "
                    "with behavior (may include subjects without fMRI decoding data)")

    # ── Page 6: direct neural-vs-behavioral overlay, twin axes, per condition.
    # Tests the "mirror image" impression from page 1 vs page 5 quantitatively
    # (Pearson r across the same 14 orientation bins). ─────────────────────
    beh["bin"] = pd.cut(beh["orientation_deg"], bins, labels=False, include_lowest=True)
    beh_grp = beh.groupby(["condition", "bin"])["abs_error"].agg(["mean", "sem"]).reset_index()
    real_df["bin"] = pd.cut(real_df["orientation_deg"], bins, labels=False, include_lowest=True)
    neural_grp = (real_df.groupby(["test_condition", "bin"])["abs_err_matched"]
                  .agg(["mean", "sem"]).reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, cond in zip(axes, ("cdf", "inverse_cdf")):
        n = neural_grp[neural_grp.test_condition == cond].set_index("bin").reindex(range(n_bins))
        b = beh_grp[beh_grp.condition == cond].set_index("bin").reindex(range(n_bins))
        valid = n["mean"].notna() & b["mean"].notna()
        r, p = stats.pearsonr(n["mean"][valid], b["mean"][valid])

        ax.errorbar(bin_centers, n["mean"], yerr=n["sem"], color="#264653", lw=1.8,
                    marker="o", ms=3, label="Neural (matched decoder)")
        ax.set_ylabel("Neural: mean abs(decoded − true)  (CHF)", color="#264653")
        ax.tick_params(axis="y", colors="#264653")

        ax2 = ax.twinx()
        ax2.errorbar(bin_centers, b["mean"], yerr=b["sem"], color="#E9C46A", lw=1.8,
                     ls="--", marker="o", ms=3, label="Behavioral (BDM bid)")
        ax2.set_ylabel("Behavioral: mean abs(bid − value)  (CHF)", color="#B8860B")
        ax2.tick_params(axis="y", colors="#B8860B")
        ax2.spines["right"].set_visible(True)

        ax.set_xlabel("Orientation (deg)")
        ax.set_xlim(0, 180)
        ax.set_title(f"{COND_LABEL[cond]}   Pearson r={r:.2f}, p={p:.3f}  (n={valid.sum()} bins)",
                    fontsize=10, color=COND_COLOUR[cond])
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=7.5)

    sns.despine(fig=fig, offset=4, right=False)
    fig.text(0.5, -0.03,
             "Decoding from: NPCr matched decoder (top-100 voxels, spherical noise, unsmoothed, real trials)  "
             "vs.  raw BDM bids (all study subjects) — tests whether the neural error-vs-orientation curve "
             "(page 1) mirrors the behavioral one (page 5)",
             ha="center", va="top", fontsize=7.5, color="0.4", style="italic")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"Wrote {out}")
