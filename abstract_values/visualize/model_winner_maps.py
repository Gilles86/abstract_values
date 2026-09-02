"""Which encoding model wins where? Per-vertex winner-take-all on fsaverage.

Compares a set of cross-validated models against each other and against the
null (``aprf-null.cv``, "predict the training mean"):

    vonMises            gabor orientation tuning — the bottom-up control
    aPRF                one log-Gaussian in value space, shared across sessions
    aPRF session-shift  mode free per session
    aPRF fully shifted  every parameter free per session

The last one matters because the two sessions use *inverted* orientation→value
mappings, so a model forced to hold one tuning across both is handicapped in a
way the orientation model is not. Comparing on cvR² rather than R² is what
makes this fair: the freer models have more parameters and would win on
full-fit R² by construction.

A vertex counts as "signal" for a subject when at least one model beats that
subject's own null. Among signal vertices, the winner is the argmax of cvR².

Two outputs:

``--html``      a webgl bundle: one "fraction of subjects won" map per model,
                plus a modal-winner map coloured by which model wins most often.
``--summary``   a PDF: what share of each subject's signal vertices each model
                wins, as a per-subject swarm with the group mean — the direct
                answer to "which usually wins".

Run from the ``pycortex2`` env.

Usage
-----
    python -m abstract_values.visualize.model_winner_maps --summary \\
        notes/figures/model_winner_summary.pdf
    python -m abstract_values.visualize.model_winner_maps --html --serve
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cortex
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from abstract_values.utils.data import BIDS_FOLDER
from abstract_values.visualize.group_surface_maps import (
    CX_FSAVERAGE, discover_subjects, load_fsaverage)
from abstract_values.visualize.webshow_surface_maps import (
    DEFAULT_WEBGL_ROOT, blended, inject_legend, save_colorbar_pdf,
    serve_directory, write_root_index)

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})

NULL_MODEL = "aprf-null.cv"

# (cv dir, short label, colour) — colours double as the modal-map palette
CANDIDATES = [
    ("vonmises.cv", "vonMises", "#3B5BA5"),
    ("aprf.cv", "aPRF", "#E76F51"),
    ("aprf-shift.cv", "aPRF shift", "#2A9D8F"),
    ("aprf-fully-shifted.cv", "aPRF free", "#B07AA1"),
]


def load_stack(deriv, subject, models, smoothed=False):
    """(n_models, n_vertices) cvR², the null, or None if anything is missing."""
    null = load_fsaverage(deriv, NULL_MODEL, subject, "cvr2", smoothed)
    if null is None:
        return None, None, []
    rows, kept = [], []
    for cvdir, label, _ in models:
        m = load_fsaverage(deriv, cvdir, subject, "cvr2", smoothed)
        if m is None:
            continue
        rows.append(np.nan_to_num(m, nan=-np.inf))
        kept.append(label)
    if not rows:
        return None, None, []
    return np.vstack(rows), np.nan_to_num(null, nan=-np.inf), kept


def winner_per_subject(deriv, subjects, models=CANDIDATES, smoothed=False):
    """Per subject: the argmax model index, and the signal mask.

    Returns ``(wins, signal, labels, used)`` where ``wins``/``signal`` are
    (n_subjects, n_vertices).
    """
    wins, signal, used, labels = [], [], [], None
    for s in subjects:
        stack, null, kept = load_stack(deriv, s, models, smoothed)
        if stack is None or len(kept) < 2:
            continue
        if labels is None:
            labels = kept
        elif kept != labels:
            print(f"  skip sub-{s}: has {kept}, expected {labels}")
            continue
        wins.append(np.argmax(stack, axis=0))
        # "Signal" means the best real model beats this subject's own null —
        # not that it beats zero (see project_cvr2_null_baseline).
        signal.append(stack.max(axis=0) > null)
        used.append(s)
    if not used:
        return None, None, [], []
    return np.vstack(wins), np.vstack(signal), labels, used


def win_share(wins, signal, n_models):
    """Per subject, the share of that subject's signal vertices each model wins."""
    out = np.zeros((wins.shape[0], n_models))
    for i in range(wins.shape[0]):
        sig = signal[i]
        if sig.sum() == 0:
            out[i] = np.nan
            continue
        for m in range(n_models):
            out[i, m] = np.mean(wins[i][sig] == m)
    return out


def summary_figure(deriv, subjects, out_pdf, models=CANDIDATES,
                   smoothing=(False, True)):
    fig, axes = plt.subplots(1, len(smoothing),
                             figsize=(5.2 * len(smoothing), 4.2), squeeze=False)
    for ax, sm in zip(axes[0], smoothing):
        wins, signal, labels, used = winner_per_subject(deriv, subjects, models, sm)
        if wins is None:
            ax.set_axis_off()
            continue
        share = win_share(wins, signal, len(labels))
        colours = [c for _, l, c in models if l in labels]
        x = np.arange(len(labels))
        rng = np.random.default_rng(0)
        for m in range(len(labels)):
            jitter = rng.uniform(-0.16, 0.16, share.shape[0])
            ax.scatter(x[m] + jitter, 100 * share[:, m], s=16, alpha=.55,
                       color=colours[m], linewidths=0, zorder=3)
            mean = 100 * np.nanmean(share[:, m])
            ax.hlines(mean, x[m] - 0.3, x[m] + 0.3, color=colours[m],
                      lw=2.4, zorder=4)
            # Offset to the right of the mean bar, not above it: centred
            # above sits on the bar itself and is unreadable.
            ax.annotate(f"{mean:.0f}%", (x[m] + 0.32, mean),
                        textcoords="offset points", xytext=(3, -3),
                        ha="left", fontsize=9, color=colours[m],
                        fontweight="bold")
        ax.set_xlim(-0.6, len(labels) - 0.15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Share of signal vertices won (%)")
        ax.set_ylim(0, 100)
        ax.axhline(100 / len(labels), color="#999", lw=0.8, ls=(0, (4, 3)),
                   zorder=1)
        pct_sig = 100 * signal.mean()
        ax.set_title(f"{'Smoothed' if sm else 'Unsmoothed'} · n={len(used)} · "
                     f"{pct_sig:.1f}% of vertices have signal", fontsize=9)
        print(f"  {'smoothed' if sm else 'unsmoothed'}: n={len(used)}, "
              f"signal at {pct_sig:.1f}% of vertices")
        for lab, m in zip(labels, range(len(labels))):
            print(f"      {lab:12s} wins {100 * np.nanmean(share[:, m]):5.1f}% "
                  f"of signal vertices")

    fig.suptitle("Which encoding model wins, per vertex, cross-validated",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def build_winner_datasets(deriv, subjects, models=CANDIDATES,
                          smoothing=(False, True), min_prevalence=0.25):
    ds, cbars = {}, []
    for sm in smoothing:
        tag = " smoothed" if sm else ""
        wins, signal, labels, used = winner_per_subject(deriv, subjects, models, sm)
        if wins is None:
            continue
        n = len(used)
        colours = [c for _, l, c in models if l in labels]
        prevalence = signal.mean(axis=0)
        gate = (prevalence >= min_prevalence).astype(np.float32)
        print(f"  {'smoothed' if sm else 'unsmoothed'}: n={n}, "
              f"{100 * gate.mean():.1f}% of vertices pass prevalence "
              f">= {min_prevalence:.0%}")

        # per-model: among subjects with signal here, how often does it win?
        frac = []
        for m in range(len(labels)):
            num = ((wins == m) & signal).sum(axis=0)
            den = np.maximum(signal.sum(axis=0), 1)
            f = (num / den).astype(np.float32)
            frac.append(f)
            name = f"Wins {labels[m]}{tag} (frac of subjects)"
            ds[name] = blended(f, gate * np.clip(f, 0, 1), CX_FSAVERAGE,
                               0.0, 1.0, "hot")
            cbars.append((f"{labels[m]} win fraction", "hot", 0.0, 1.0))

        # modal winner: which model wins most often, coloured categorically.
        # Encoded as the model index so a discrete colormap reads as a label,
        # with opacity on how dominant that winner is over the runner-up.
        stack = np.vstack(frac)
        modal = np.argmax(stack, axis=0).astype(np.float32)
        top = np.sort(stack, axis=0)
        margin = (top[-1] - top[-2]) if stack.shape[0] > 1 else top[-1]
        cmap = mpl.colors.ListedColormap(colours)
        name = f"Modal winner{tag} ({' / '.join(labels)})"
        # Opacity on the margin over the runner-up, starting above the median
        # margin (~0.25). Scaling from zero shows every vertex including the
        # near-ties, which is most of cortex and reads as a solid wash.
        confident = np.clip((margin - 0.15) / 0.35, 0, 1)
        ds[name] = blended(modal, gate * confident, CX_FSAVERAGE,
                           -0.5, len(labels) - 0.5, cmap)
        cbars.append((f"Modal winner: {' / '.join(labels)}", cmap,
                      -0.5, len(labels) - 0.5))

        # The value models split their wins three ways, so none of them is
        # modal even where the family collectively beats orientation — vonMises
        # is modal at 84% of vertices while the value family wins 42% of them.
        # Collapsing the family is the contrast that actually answers
        # "value or orientation here?".
        ori = [i for i, l in enumerate(labels) if "vonMises" in l]
        val = [i for i, l in enumerate(labels) if i not in ori]
        if ori and val:
            diff = stack[val].sum(axis=0) - stack[ori].sum(axis=0)
            name = f"Value family minus vonMises{tag}"
            ds[name] = blended(diff, gate * np.clip(np.abs(diff) / 0.4, 0, 1),
                               CX_FSAVERAGE, -1.0, 1.0, "RdBu_r")
            cbars.append(("Value family - vonMises (win-fraction)",
                          "RdBu_r", -1.0, 1.0))
            print(f"    value family beats vonMises at "
                  f"{100 * np.mean(diff[gate.astype(bool)] > 0):.1f}% "
                  f"of gated vertices")
    return ds, cbars


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--models", nargs="+", default=None,
                   help="cv dirs to compare (default: vonmises.cv aprf.cv "
                        "aprf-shift.cv aprf-fully-shifted.cv)")
    p.add_argument("--summary", default=None, help="write the summary PDF here")
    p.add_argument("--html", nargs="?", const="", default=None,
                   help="build the winner webgl bundle (default "
                        "<out-root>/model-winner)")
    p.add_argument("--out-root", default=str(DEFAULT_WEBGL_ROOT))
    p.add_argument("--serve", type=int, nargs="?", const=8000, default=None)
    p.add_argument("--min-prevalence", type=float, default=0.25,
                   help="Fraction of subjects that must have signal at a "
                        "vertex for it to be drawn (default 0.25)")
    p.add_argument("--smoothing", default="both",
                   choices=["both", "unsmoothed", "smoothed"])
    args = p.parse_args()

    deriv = Path(args.bids_folder) / "derivatives"
    subjects = args.subjects or discover_subjects(deriv)
    models = ([(m, m.replace(".cv", ""), c) for m, c in
               zip(args.models, [c for _, _, c in CANDIDATES] * 4)]
              if args.models else CANDIDATES)
    smoothing = {"both": (False, True), "unsmoothed": (False,),
                 "smoothed": (True,)}[args.smoothing]
    print(f"{len(subjects)} subjects, models: "
          f"{', '.join(l for _, l, _ in models)}\n")

    if args.summary:
        summary_figure(deriv, subjects, args.summary, models, smoothing)

    if args.html is not None:
        dest = Path(args.html) if args.html else \
            Path(args.out_root) / "model-winner"
        ds, cbars = build_winner_datasets(deriv, subjects, models, smoothing,
                                          args.min_prevalence)
        if not ds:
            raise SystemExit("No winner datasets built.")
        dest.mkdir(parents=True, exist_ok=True)
        for stale in (dest / "data").glob("*"):
            stale.unlink()
        print(f"\nBuilding winner bundle ({len(ds)} maps) in {dest} ...")
        cortex.webgl.make_static(str(dest), ds, types=("inflated",),
                                 title=f"Model winner maps (n={len(subjects)})",
                                 recache=False, curvature_brightness=0.62,
                                 curvature_contrast=0.28,
                                 curvature_smoothness=2.0)
        save_colorbar_pdf(cbars, dest / "colorbars.pdf")
        inject_legend(dest / "index.html", list(ds.keys()), cbars)
        write_root_index(dest.parent)
        print(f"Wrote winner bundle → {dest / 'index.html'}")
        if args.serve is not None:
            serve_directory(dest.parent, args.serve)

    if args.summary is None and args.html is None:
        raise SystemExit("Nothing to do — pass --summary and/or --html.")


if __name__ == "__main__":
    main()
