"""Does session-shift beat standard? Depends entirely on how you pick voxels.

Formalizes the ad hoc comparison from 2026-08 into a proper figure. Two
voxel-selection criteria for "signal voxels" in NPCr, both cross-validated
(leave-one-run-out):

  Biased  (session-shift-gated):  cvR²(session-shift) > cvR²(null)
  Neutral (standard-gated):       cvR²(standard)      > cvR²(null)

The biased criterion selects on the very model being championed in the
head-to-head comparison — a winner's-curse setup that inflates
session-shift's apparent advantage. The neutral criterion selects on the
simplest, common model instead (nested inside every shift variant), which
if anything tilts the deck toward *standard* looking good. Comparing cvR²
for standard / session-shift / fully-shifted under both selections shows
how much the conclusion depends on that choice.

Usage:
    python -m abstract_values.visualize.model_comparison_selection_bias
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
from scipy import stats

from abstract_values.utils.data import Subject, BIDS_FOLDER
from abstract_values.visualize.shifted_preferred_value import _resampled_mask
from abstract_values.visualize.compare_voxel_selection import discover_subjects, DERIV_ROOT

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
    "lines.linewidth": 1.2, "lines.markersize": 4, "patch.linewidth": 0.5,
    "legend.frameon": False, "legend.handlelength": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
sns.set_context("paper")

MODELS = ["standard", "session-shift", "fully-shifted"]
MODEL_LABEL = {"standard": "Standard\n(no shift)", "session-shift": "Session-shift\n(mode only)",
              "fully-shifted": "Fully-shifted\n(everything)"}
MODEL_COLOUR = {"standard": "#9CBEDD", "session-shift": "#3B6FA0", "fully-shifted": "#0B3D6B"}

CV_DIRS = {
    "standard": DERIV_ROOT / "aprf.cv",
    "session-shift": DERIV_ROOT / "aprf-shift.cv",
    "fully-shifted": DERIV_ROOT / "aprf-fully-shifted.cv",
}
NULL_CV = DERIV_ROOT / "aprf-null.cv"

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "notes" / "figures" / "model_comparison_selection_bias.pdf"
DEFAULT_TSV = Path(__file__).resolve().parents[2] / "notes" / "data" / "model_comparison_selection_bias.tsv"


def _cv_path(base: Path, s: str) -> Path:
    return base / f"sub-{s}/func/sub-{s}_task-abstractvalue_space-T1w_desc-cvr2_pe.nii.gz"


def collect_source_data(subjects) -> pd.DataFrame:
    rows = []
    for s in subjects:
        sub = Subject(s, bids_folder=Path(BIDS_FOLDER))
        paths = {name: _cv_path(base, s) for name, base in CV_DIRS.items()}
        p_null = _cv_path(NULL_CV, s)
        if not all(p.exists() for p in list(paths.values()) + [p_null]):
            continue

        roi_img = sub.get_roi_mask("NPCr", hemi=None)
        ref_img = nib.load(str(paths["standard"]))
        mask_arr = _resampled_mask(roi_img, ref_img)
        cvr2 = {name: nib.load(str(p)).get_fdata().astype(np.float32)[mask_arr]
               for name, p in paths.items()}
        cvr2_null = nib.load(str(p_null)).get_fdata().astype(np.float32)[mask_arr]
        finite = np.isfinite(cvr2_null)
        for name in MODELS:
            finite &= np.isfinite(cvr2[name])

        for selection, gate_model in [("biased", "session-shift"), ("neutral", "standard")]:
            sel = finite & (cvr2[gate_model] > cvr2_null)
            n = int(sel.sum())
            if n < 10:
                continue
            row = dict(subject=s, selection=selection, n=n)
            for name in MODELS:
                row[name] = float(cvr2[name][sel].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, out: Path):
    long = df.melt(id_vars=["subject", "selection", "n"], value_vars=MODELS,
                   var_name="model", value_name="cvr2")
    long["model"] = pd.Categorical(long["model"], categories=MODELS, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.4), constrained_layout=True, sharey=True)
    panel_titles = {"biased": "Biased selection\ncvR²(session-shift) > cvR²(null)",
                    "neutral": "Neutral selection\ncvR²(standard) > cvR²(null)"}

    for ax, selection in zip(axes, ["biased", "neutral"]):
        d = long[long.selection == selection]
        xpos = {m: i for i, m in enumerate(MODELS)}

        # per-subject spaghetti lines
        for s, ds in d.groupby("subject"):
            ds = ds.sort_values("model")
            ax.plot([xpos[m] for m in ds.model], ds.cvr2, "-", color="0.75",
                   lw=0.6, alpha=0.7, zorder=1)

        # group mean +/- SEM, bold marker
        stats_by_model = d.groupby("model", observed=True).cvr2.agg(["mean", "sem"])
        for m in MODELS:
            x = xpos[m]
            mean, sem = stats_by_model.loc[m, "mean"], stats_by_model.loc[m, "sem"]
            ax.errorbar([x], [mean], yerr=[sem], fmt="D", color=MODEL_COLOUR[m],
                       markersize=9, markeredgecolor="black", markeredgewidth=1.4,
                       ecolor="0.2", elinewidth=1.3, capsize=0, zorder=3)

        ax.axhline(0, color="0.4", lw=0.6, ls="--", zorder=0)
        ax.set_xticks([xpos[m] for m in MODELS])
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS])
        ax.set_xlim(-0.5, 2.5)
        n_sub = d.subject.nunique()
        ax.set_title(f"{panel_titles[selection]}  (n={n_sub})", fontsize=9, color="0.15")

        # direct annotation: which model wins, pointing at the WINNING marker
        piv = d.pivot(index="subject", columns="model", values="cvr2")
        delta = piv["session-shift"] - piv["standard"]
        w = stats.wilcoxon(delta)
        favor = (delta > 0).sum()
        shift_wins = favor > len(delta) / 2
        winner_model = "session-shift" if shift_wins else "standard"
        winner_x = xpos[winner_model]
        winner_y = stats_by_model.loc[winner_model, "mean"]
        winner = "Session-shift wins" if shift_wins else "Standard wins"
        ax.annotate(winner, xy=(winner_x, winner_y),
                   xytext=(winner_x + 0.55, winner_y + 0.028),
                   fontsize=8.5, ha="left", va="center", color="0.1",
                   arrowprops=dict(arrowstyle="-|>", color="0.2", lw=1.2,
                                   mutation_scale=9, shrinkA=3, shrinkB=8,
                                   relpos=(0.0, 0.5)))
        ax.text(0.03, 0.04, f"{favor}/{len(delta)} favor shift\np={w.pvalue:.1e}",
               transform=ax.transAxes, fontsize=7.5, color="0.25", va="bottom")

    axes[0].set_ylabel("Cross-validated R² (NPCr, mean per subject)")
    sns.despine(fig=fig, offset=5, trim=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def run(subjects, out, tsv):
    df = collect_source_data(subjects)
    if df.empty:
        raise SystemExit("No subjects with complete standard/session-shift/fully-shifted/null cv fits.")
    tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tsv, sep="\t", index=False)
    make_figure(df, out)
    print(f"Wrote {out}\nSource data: {tsv}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--tsv", default=str(DEFAULT_TSV))
    args = p.parse_args()
    subjects = args.subjects or discover_subjects(smoothed=False)
    run(subjects, Path(args.out), Path(args.tsv))


if __name__ == "__main__":
    main()
