#!/usr/bin/env python3
"""Per-trial gaze trajectories during the response_bar (estimation) phase,
averaged per orientation: one large grand-average panel plus one small
panel per subject.

Input: notes/data/gaze_trajectories_all.tsv, produced by
abstract_values.eyetracking.extract_gaze_trajectories (run on
sciencecluster — needs the Linux edf2asc binary, not available locally)
and rsynced back. Columns: subject, session, mapping, run, trial_nr,
orientation, sample_idx (0..N_RESAMPLE-1, normalized time within the
response_bar window), x_deg, y_deg (visual angle, screen-centre-relative,
converted from EyeLink pixels using each run's own expsettings.yml
geometry — see that script's docstring).

Per-trial recentring
---------------------
x_deg/y_deg as extracted are relative to *screen centre*, which is a
geometric assumption, not where gaze actually was. EyeLink calibration
drifts across runs/sessions, and gaze at response_bar onset needn't sit
exactly on the pixel screen centre anyway — so plotting raw
screen-centre-relative trajectories conflates real within-trial gaze
movement with between-run calibration offset (this is why the first cut
of this figure looked bad: panels didn't line up at a common start, and
one subject's calibration offset blew out the shared axis for everyone).

Fix: recentre every trial on its OWN first resampled sample (sample_idx
0, i.e. gaze position right as the response_bar phase begins — the
participant was on/near the preceding fixation cross a moment before).
After recentring, "(0, 0)" means "gaze position when the response bar
appeared" for every trial, subject, and session alike, and the plotted
paths show only genuine within-trial drift.

Axes are NOT shared across subject panels — deliberately reverted from
an earlier version. A shared axis let one high-amplitude subject flatten
every other panel to a sliver; per-panel axes (still symmetric around
zero and equal-aspect) show each subject's own trajectory shape clearly.
Cross-subject amplitude comparison, if needed, is what the grand-average
panel plus notes/data/gaze_trajectories_all.tsv are for.

Orientation is cyclic (0 deg == 180 deg, gratings are axis-symmetric) so
colour uses the 'hsv' colormap wrapped at 180 deg, matching the
convention in visualize_mean_orientation_fsaverage.py (ORI_CMAP = 'hsv').

Usage (local, plain matplotlib/seaborn — no cluster needed):
    python -m abstract_values.visualize.plot_gaze_trajectories \\
        --tsv notes/data/gaze_trajectories_all.tsv \\
        --out notes/figures/gaze_trajectories.pdf

Group-only variant for a talk slide (grand-average panel + gabor colour
legend, talk-scale type, panel proportioned to the data, callouts
auto-placed clear of the paths; writes both .pdf and .png):
    python -m abstract_values.visualize.plot_gaze_trajectories --group-only \\
        --tsv notes/data/gaze_trajectories_gabor.tsv \\
        --onset-label "Gabor onset" --offset-label "Gabor offset" \\
        --out .../figures/gaze_trajectories_gabor_group.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

ORI_CMAP = "hsv"        # cyclic — orientation wraps at 180 deg
ORI_PERIOD = 180.0
MIN_TRIALS = 3           # minimum trials per (subject, orientation) to plot
MIN_FRAC_VALID = 0.5     # per-trial minimum fraction of non-blink samples, same
                         # threshold as eu_vs_behavior.py's load_gaze_dispersion()
ORIGIN_SAMPLES = 3       # average this many leading samples for the recentring
                         # origin instead of a single (noisier) point
TRIAL_KEYS = ["subject", "session", "mapping", "run", "trial_nr"]
REPO_ROOT = Path(__file__).resolve().parents[2]
GRATING_SETTINGS = REPO_ROOT / "experiment" / "settings" / "sns_fmri.yml"

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 150,
    "savefig.dpi": 300,
})
sns.set_context("paper")


def orientation_color(orientation: float) -> tuple:
    cmap = plt.get_cmap(ORI_CMAP)
    return cmap((orientation % ORI_PERIOD) / ORI_PERIOD)


def subject_sort_key(s: str):
    return (0, int(s)) if s.isdigit() else (1, s)


def recenter_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract each trial's own early-window position from the whole trial.

    Origin = mean of the first ORIGIN_SAMPLES resampled points (gaze just
    as the response_bar phase begins), not a single sample — a lone point
    is noisier and gives every downstream mean an extra jittery offset.
    """
    origin = (df[df["sample_idx"] < ORIGIN_SAMPLES]
                .groupby(TRIAL_KEYS)[["x_deg", "y_deg"]].mean()
                .rename(columns={"x_deg": "x0", "y_deg": "y0"}).reset_index())
    df = df.merge(origin, on=TRIAL_KEYS, how="left")
    df["x_deg"] = df["x_deg"] - df["x0"]
    df["y_deg"] = df["y_deg"] - df["y0"]
    return df.drop(columns=["x0", "y0"])


def symmetric_limits(x: np.ndarray, y: np.ndarray, pad_frac: float = 0.15) -> tuple:
    m = max(np.abs(x).max(), np.abs(y).max()) * (1 + pad_frac)
    m = max(m, 0.1)  # guard against degenerate all-zero panels
    return (-m, m)


def smooth_path(x: pd.Series, y: pd.Series, window: int = 3) -> tuple:
    """Light centred rolling mean, display-only — the underlying per-sample
    mean is noisy (SEM comparable to point-to-point steps even pooling
    ~400 trials), so unsmoothed paths read as jagged zig-zags that
    overstate how much the true average trajectory actually wiggles."""
    x_s = x.rolling(window, center=True, min_periods=1).mean().to_numpy()
    y_s = y.rolling(window, center=True, min_periods=1).mean().to_numpy()
    return x_s, y_s


def draw_trajectories(ax, d: pd.DataFrame, lw: float, dot_ms: float):
    for orientation, g in d.groupby("orientation"):
        g = g.sort_values("sample_idx")
        x, y = smooth_path(g["x_deg"], g["y_deg"])
        color = orientation_color(orientation)
        ax.plot(x, y, color=color, lw=lw, alpha=0.85, zorder=2)
        ax.plot(x[-1], y[-1], "o", color=color, ms=dot_ms, zorder=3, markeredgewidth=0)
    ax.plot(0, 0, "+", color="0.15", ms=6, mew=1.2, zorder=4)
    ax.set_aspect("equal", adjustable="box")


def plot_orientation_wheel(ax):
    """Compact polar swatch: colour = orientation (0-180, mirrored at 360)."""
    n = 360
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    orientation = np.degrees(theta) % ORI_PERIOD
    colors = plt.get_cmap(ORI_CMAP)(orientation / ORI_PERIOD)
    ax.axis("off")
    ax_polar = ax.figure.add_axes(ax.get_position(), projection="polar")
    ax_polar.bar(theta, np.ones(n), width=2 * np.pi / n, bottom=0.6,
                 color=colors, linewidth=0)
    ax_polar.set_ylim(0, 1.6)
    ax_polar.set_xticks([])
    ax_polar.set_yticks([])
    ax_polar.spines["polar"].set_visible(False)
    ax_polar.set_title("Orientation", fontsize=9, pad=2)


def load_grating_params() -> dict:
    with open(GRATING_SETTINGS) as f:
        s = yaml.safe_load(f)
    return s["grating"]


def render_gabor(orientation_deg: float, size_deg: float, hole_deg: float,
                  sf: float, contrast: float = 1.0, res: int = 200, phase: float = 0.25) -> np.ndarray:
    """Static annular grating patch, same geometry as experiment/stimuli.py's
    AnnulusGrating (outer circle minus inner hole, raised-cosine-ish edges).

    Orientation convention verified against PsychoPy's own source
    (psychopy/visual/basevisual.py, BaseVisualStim.ori setter): vertex
    rotation matrix [[cosθ, -sinθ], [sinθ, cosθ]] applied to a local point
    (vx, vy) — reproduced here as the grating's spatial-gradient direction
    (cosθ, -sinθ) rather than derived independently, so a 45° patch is
    guaranteed to tilt the same way PsychoPy actually renders it (0 =
    vertical bars, positive = clockwise).
    """
    half = size_deg / 2
    lin = np.linspace(-half, half, res)
    xx, yy = np.meshgrid(lin, lin)
    theta = np.radians(orientation_deg)
    grad = xx * np.cos(theta) - yy * np.sin(theta)
    grating = np.cos(2 * np.pi * sf * grad + 2 * np.pi * phase)
    r = np.sqrt(xx ** 2 + yy ** 2)
    fringe = 0.4  # deg, soft edge
    outer_mask = np.clip((size_deg / 2 - r) / fringe, 0, 1)
    inner_mask = np.clip((r - hole_deg / 2) / fringe, 0, 1)
    mask = outer_mask * inner_mask
    return 0.5 + 0.5 * contrast * grating * mask


def plot_gabor_examples(gs_row, orientations: list, grating_params: dict, cmap_note: bool = True,
                         title_fontsize: float = 7.5, frame_lw: float = 3):
    """One small annular-grating patch per orientation, framed in that
    orientation's colour, so the colour <-> physical-grating mapping used
    everywhere else in the figure is unambiguous. Nothing but the grating
    itself is drawn (no fixation cross, no response bar) — this is a
    colour legend, not a trial reconstruction."""
    n = len(orientations)
    sub_gs = gs_row.subgridspec(1, n)
    fig = gs_row.get_gridspec().figure
    for i, ori in enumerate(orientations):
        ax = fig.add_subplot(sub_gs[0, i])
        img = render_gabor(ori, size_deg=grating_params["size"], hole_deg=grating_params["hole_size"],
                            sf=grating_params["spatial_freq"], contrast=1.0)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=(-1, 1, -1, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "bottom", "left", "right"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(orientation_color(ori))
            ax.spines[side].set_linewidth(frame_lw)
        ax.set_title(f"{ori:g}°", fontsize=title_fontsize, pad=2)
        # Attach the caveat as an xlabel (not a floating fig.text) so
        # constrained_layout reserves real space for it instead of letting
        # it drift into whatever panel happens to sit below.
        if cmap_note and i == 0:
            ax.set_xlabel("Example gratings — contrast boosted for visibility "
                           f"(actual stimulus contrast = {grating_params['contrast']:g})",
                           fontsize=6, labelpad=3, ha="left", x=0)


def annotate_onset_offset(ax, offset_xy: tuple, onset_label: str, offset_label: str,
                           fontsize: float = 8):
    ax.annotate(
        f"{onset_label}\n(gaze recentred here)",
        xy=(0, 0), xytext=(0.35, 0.92), textcoords="axes fraction",
        fontsize=fontsize, ha="left", va="top",
        arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=70",
                         color="0.2", lw=1.1, mutation_scale=11, shrinkA=3, shrinkB=8,
                         relpos=(0.0, 0.0)),
    )
    ax.annotate(
        offset_label,
        xy=offset_xy, xytext=(0.05, 0.08), textcoords="axes fraction",
        fontsize=fontsize, ha="left", va="bottom",
        arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=-70",
                         color="0.2", lw=1.1, mutation_scale=11, shrinkA=3, shrinkB=8,
                         relpos=(0.0, 1.0)),
    )


TALK_RCPARAMS = {
    # Talk-sized version of the paper rcParams above: same look, fonts and
    # line weights scaled up for a figure that occupies half a 16:9 slide
    # (rendered ~5 in wide, seen from the back of a lecture hall).
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.1,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
}


def padded_limits(x: np.ndarray, y: np.ndarray, pad_frac: float = 0.10,
                   min_span: float = 0.6) -> tuple:
    """Per-axis limits hugging the data, with a little padding.

    symmetric_limits() (used by the paper figure) centres both axes on the
    recentring origin and gives them a common span, which leaves most of
    the panel empty — fine in a 6-column grid, wasteful when the panel has
    to earn half a slide. Aspect stays equal (a degree of gaze is a degree
    in both directions); only the box shape follows the data.
    """
    def one(v):
        lo, hi = v.min(), v.max()
        span = max(hi - lo, min_span)
        pad = span * pad_frac
        return (lo - pad, hi + pad)
    return one(x), one(y)


def nice_ticks(lim: tuple, max_ticks: int = 6) -> list:
    """Round ticks inside `lim`, on a step that includes 0 (the origin)."""
    span = lim[1] - lim[0]
    for step in (0.25, 0.5, 1.0, 2.0, 5.0):
        if span / step <= max_ticks:
            break
    k0 = int(np.ceil(lim[0] / step))
    k1 = int(np.floor(lim[1] / step))
    return [round(k * step, 10) for k in range(k0, k1 + 1)]


# Candidate text anchors in axes fractions, with their text alignment.
# The arrow-tail anchor (relpos) is not fixed per corner but derived from
# where the target sits relative to the label — a fixed relpos on a wide
# label puts the tail halfway across the panel and the connector then
# dives back through the data.
LABEL_ANCHORS = [
    ((0.01, 0.99), "left",   "top"),
    ((0.50, 0.99), "center", "top"),
    ((0.99, 0.99), "right",  "top"),
    ((0.01, 0.02), "left",   "bottom"),
    ((0.50, 0.02), "center", "bottom"),
    ((0.99, 0.02), "right",  "bottom"),
    ((0.01, 0.50), "left",   "center"),
    ((0.99, 0.50), "right",  "center"),
]


def _clearance(px: np.ndarray, py: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> float:
    """Smallest distance (axes-fraction units) from any probe point to the data."""
    d = (px[:, None] - fx[None, :]) ** 2 + (py[:, None] - fy[None, :]) ** 2
    return float(np.sqrt(d.min()))


def place_callouts(gx, gy, xlim, ylim, ends) -> tuple:
    """Choose corners for the two callouts: free label spot AND free arrow route.

    Scoring only the label position isn't enough — the winning corner is
    often across the panel from its target, so the straight connector then
    runs the length of the fan and reads as one more trajectory. Each
    candidate is therefore scored on the clearance of its whole route
    (label point plus samples along the connector, stopping short of the
    target the arrow has to reach), and the two labels are assigned
    jointly so they can't claim the same corner.
    """
    def to_frac(x, y):
        return ((x - xlim[0]) / (xlim[1] - xlim[0]),
                (y - ylim[0]) / (ylim[1] - ylim[0]))

    fx, fy = to_frac(gx, gy)
    ox, oy = to_frac(0.0, 0.0)
    ex, ey = to_frac(ends[:, 0], ends[:, 1])
    t = np.linspace(0.0, 1.0, 24)

    def score(anchor, tx, ty):
        ax_, ay_ = anchor[0]
        px, py = ax_ + t * (tx - ax_), ay_ + t * (ty - ay_)
        # Ignore the final approach: every route ends on data by
        # construction (that is what it is pointing at), and near the
        # origin all orientations converge, so scoring it would rank every
        # candidate equally bad.
        far = np.hypot(px - tx, py - ty) > 0.12
        if not far.any():
            return 0.0
        return _clearance(px[far], py[far], fx, fy)

    onset = [score(a, ox, oy) for a in LABEL_ANCHORS]
    # The offset label points at whichever endpoint dot is nearest it, so
    # its target depends on the anchor — resolve per candidate.
    offset, offset_target = [], []
    for a in LABEL_ANCHORS:
        j = np.argmin((ex - a[0][0]) ** 2 + (ey - a[0][1]) ** 2)
        offset.append(score(a, ex[j], ey[j]))
        offset_target.append(ends[j])
    def joint(ij):
        i, j = ij
        sep = np.hypot(LABEL_ANCHORS[i][0][0] - LABEL_ANCHORS[j][0][0],
                       LABEL_ANCHORS[i][0][1] - LABEL_ANCHORS[j][0][1])
        # Clearance first (rounded, so near-ties really are ties), label
        # separation only as the tie-break — clearance differences are a
        # couple of axes-fraction hundredths, so any additive separation
        # bonus large enough to notice would outvote them entirely.
        return (round(min(onset[i], offset[j]), 3), sep)
    best = max(((i, j) for i in range(len(LABEL_ANCHORS)) for j in range(len(LABEL_ANCHORS)) if i != j),
               key=joint)
    i, j = best
    return LABEL_ANCHORS[i], LABEL_ANCHORS[j], tuple(offset_target[j])


def callout(ax, text: str, target: tuple, anchor: tuple, xlim: tuple, ylim: tuple,
             fontsize: float):
    (xy_text, ha, va) = anchor
    ty = (target[1] - ylim[0]) / (ylim[1] - ylim[0])
    # Tail on the label's outer edge (the panel margin), not the edge
    # facing the target: a label sitting in a corner is wider than the
    # gap it sits in, so anchoring on the inner edge starts the connector
    # well inside the data. Leaving from the margin lets it run down the
    # empty edge and turn in at the end, the way a leader line should.
    relpos = (0.0 if ha == "left" else 1.0,
              {"top": 0.0, "bottom": 1.0}.get(va, 0.5) if va != "center"
              else (0.0 if ty < xy_text[1] else 1.0))
    ax.annotate(text, xy=target, xytext=xy_text, textcoords="axes fraction",
                fontsize=fontsize, ha=ha, va=va,
                arrowprops=dict(arrowstyle="-|>", color="0.2", lw=1.2, mutation_scale=9,
                                connectionstyle="arc3,rad=0",  # straight: a curved
                                # connector sweeps through whichever paths happen to
                                # lie under it, and the two epochs fill the panel
                                # differently
                                shrinkA=4, shrinkB=9, relpos=relpos))


def plot_group_figure(agg: dict, args) -> "plt.Figure":
    """Grand-average panel only, sized for the right half of a 16:9 slide.

    Same aggregation, recentring and colour convention as the full
    multi-subject figure — just the group panel plus the gabor colour
    legend, with talk-scale type. The per-subject grid is dropped, so the
    N/trial-count bookkeeping moves into the panel title, and the two
    onset/offset callouts are auto-placed in whichever corners this
    epoch's mean paths leave empty.
    """
    mpl.rcParams.update(TALK_RCPARAMS)
    grand, subjects, n_trials, keep = agg["grand"], agg["subjects"], agg["n_trials"], agg["keep"]
    gx, gy = grand["x_deg"].to_numpy(), grand["y_deg"].to_numpy()
    xlim, ylim = padded_limits(gx, gy)

    # Panel proportions follow the data's own aspect (equal aspect means the
    # box can't be square unless the data is), and the figure follows the
    # panel, so no slide space is spent on empty axes.
    if args.figsize is None:
        panel_h = 4.2
        panel_w = float(np.clip(panel_h * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]), 2.6, 5.2))
        figsize = (panel_w + 1.1, panel_h + 1.5)
    else:
        figsize = args.figsize

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.19])

    ax = fig.add_subplot(gs[0, :])
    draw_trajectories(ax, grand, lw=2.4, dot_ms=7)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(nice_ticks(xlim))
    ax.set_yticks(nice_ticks(ylim))
    ax.set_xlabel("Gaze x (deg)")
    ax.set_ylabel("Gaze y (deg)")
    n_used = int(n_trials.loc[n_trials.index.isin(keep)].sum())
    ax.set_title(f"N = {len(subjects)} subjects, {n_used:,} trials", pad=6)

    # The offset callout points at an endpoint dot, not the mean of all
    # endpoints: during the gabor epoch the mean paths radiate in every
    # direction, so their centroid sits in empty space mid-fan and the
    # arrow would point at nothing.
    ends = (grand.sort_values("sample_idx").groupby("orientation")
                 .tail(1)[["x_deg", "y_deg"]].to_numpy())
    if not args.no_callouts:
        a_onset, a_offset, end_xy = place_callouts(gx, gy, xlim, ylim, ends)
        # Short labels only: the paper figure's "(gaze recentred here)"
        # gloss makes a box wide enough to shove the arrow tail into the data.
        callout(ax, args.onset_label, (0, 0), a_onset, xlim, ylim, 11)
        callout(ax, args.offset_label, end_xy, a_offset, xlim, ylim, 11)
    sns.despine(ax=ax, offset=4, trim=True)

    plot_gabor_examples(gs[1, :], [0, 30, 60, 90, 120, 150], load_grating_params(),
                        cmap_note=False, title_fontsize=11, frame_lw=3.5)
    return fig


def aggregate_df(df: pd.DataFrame, qc_label: str = "") -> dict:
    """QC-filter -> recentre -> two-stage aggregate an already-loaded
    gaze_trajectories dataframe (one or more full TSVs, or a subset of
    one — e.g. filtered to a single `mapping` value). Shared by
    plot_gaze_trajectories.py and compare_gaze_epochs.py so QC/recentring/
    averaging can't drift apart between the two scripts.

    Returns dict with: df (trial-level, QC'd, recentred), per_subj
    (subject x orientation x sample_idx mean), grand (orientation x
    sample_idx mean, subjects weighted equally), subjects (sorted list),
    n_trials (subject x orientation trial counts, post-QC, pre-MIN_TRIALS).
    """
    n_before = df.drop_duplicates(TRIAL_KEYS).shape[0]
    df = df[df["frac_valid"] >= MIN_FRAC_VALID]
    n_after = df.drop_duplicates(TRIAL_KEYS).shape[0]
    print(f"QC ({qc_label}): dropped {n_before - n_after}/{n_before} trials with "
          f"frac_valid < {MIN_FRAC_VALID} (mostly blinks/track loss)")
    df = recenter_trials(df)

    n_trials = (df.drop_duplicates(TRIAL_KEYS + ["orientation"])
                  .groupby(["subject", "orientation"]).size())
    keep = n_trials[n_trials >= MIN_TRIALS].index

    per_subj = (df.groupby(["subject", "orientation", "sample_idx"])[["x_deg", "y_deg"]]
                  .mean().reset_index().set_index(["subject", "orientation"]))
    per_subj = per_subj.loc[per_subj.index.isin(keep)].reset_index()

    # Two-stage average (trial -> subject -> group), each subject weighted
    # equally, rather than pooling raw trials (which would let subjects
    # with more usable trials dominate the "grand" mean).
    grand = (per_subj.groupby(["orientation", "sample_idx"])[["x_deg", "y_deg"]]
                     .mean().reset_index())

    subjects = sorted(per_subj["subject"].unique(), key=subject_sort_key)
    return dict(df=df, per_subj=per_subj, grand=grand, subjects=subjects, n_trials=n_trials, keep=keep)


def load_and_aggregate(tsv_path: str, drop_pilots: bool = False) -> dict:
    df = pd.read_csv(tsv_path, sep="\t", dtype={"subject": str})
    if drop_pilots:
        # sub-pil## are MRI-protocol pilots, not study participants (see
        # CLAUDE.md); they are in the eyetracking extraction because they
        # ran the same task, but a figure captioned "N subjects" should
        # not silently count them.
        pilots = sorted(s for s in df["subject"].unique() if not s.isdigit())
        df = df[df["subject"].str.isdigit()]
        print(f"Dropped pilot subjects: {', '.join(pilots) if pilots else '(none)'}")
    return aggregate_df(df, qc_label=tsv_path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tsv", default="notes/data/gaze_trajectories_all.tsv")
    p.add_argument("--out", default="notes/figures/gaze_trajectories.pdf")
    p.add_argument("--ncols", type=int, default=6)
    p.add_argument("--epoch-title", default="value estimation",
                   help="Human-readable epoch name for the figure suptitle, "
                        "e.g. 'value estimation' or 'gabor presentation'.")
    p.add_argument("--onset-label", default="Response-bar onset")
    p.add_argument("--offset-label", default="Feedback onset\n(trial end)")
    p.add_argument("--drop-pilots", action="store_true",
                   help="Exclude sub-pil## (MRI-protocol pilots, not study "
                        "participants) from the aggregate.")
    p.add_argument("--no-callouts", action="store_true",
                   help="Omit the onset/offset arrow annotations; --group-only only.")
    p.add_argument("--group-only", action="store_true",
                   help="Grand-average panel only, talk-scaled type, sized for "
                        "half a 16:9 slide (drops the per-subject grid).")
    p.add_argument("--figsize", default=None,
                   help="W,H in inches; --group-only only. Default: derived from "
                        "the grand average's own aspect ratio.")
    args = p.parse_args()
    if args.figsize is not None:
        args.figsize = tuple(float(v) for v in args.figsize.split(","))
    args.onset_label = args.onset_label.replace("\\n", "\n")
    args.offset_label = args.offset_label.replace("\\n", "\n")

    agg = load_and_aggregate(args.tsv, drop_pilots=args.drop_pilots)

    if args.group_only:
        fig = plot_group_figure(agg, args)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Both formats on purpose: PDF for anything print-like, PNG because
        # the MARP decks composite raster images (a PDF <img> does not render).
        for path in (out, out.with_suffix(".png" if out.suffix == ".pdf" else ".pdf")):
            fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
            print(f"Wrote {path}")
        return

    per_subj, grand, subjects, n_trials, keep = (
        agg["per_subj"], agg["grand"], agg["subjects"], agg["n_trials"], agg["keep"])
    ncols = args.ncols
    nrows_subj = int(np.ceil(len(subjects) / ncols))
    top_rows = 4  # grand-average panel + wheel + gabor-example strip

    panel_size = 1.7
    fig_w = ncols * panel_size
    fig_h = (top_rows + nrows_subj) * panel_size
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(top_rows + nrows_subj, ncols)

    plot_gabor_examples(gs[top_rows - 1, :], [0, 30, 60, 90, 120, 150], load_grating_params())

    grand_cols = max(3, ncols // 2)
    ax_grand = fig.add_subplot(gs[0:top_rows - 1, 0:grand_cols])
    draw_trajectories(ax_grand, grand, lw=1.8, dot_ms=5)
    lim = symmetric_limits(grand["x_deg"].to_numpy(), grand["y_deg"].to_numpy())
    ax_grand.set_xlim(lim)
    ax_grand.set_ylim(lim)
    ax_grand.set_xlabel("Gaze x (deg)")
    ax_grand.set_ylabel("Gaze y (deg)")
    ax_grand.set_title(f"Grand average (N={len(subjects)} subjects, "
                        f"mean of {n_trials.loc[n_trials.index.isin(keep)].sum()} trials)",
                        fontsize=9)
    end_xy = (grand.sort_values("sample_idx").groupby("orientation").tail(1)[["x_deg", "y_deg"]].mean())
    annotate_onset_offset(ax_grand, (end_xy["x_deg"], end_xy["y_deg"]), args.onset_label, args.offset_label)
    sns.despine(ax=ax_grand, offset=3, trim=True)

    ax_wheel = fig.add_subplot(gs[0:top_rows - 1, grand_cols:grand_cols + 2])
    plot_orientation_wheel(ax_wheel)

    col_bottom_row = {}
    for i in range(len(subjects)):
        r, c = divmod(i, ncols)
        col_bottom_row[c] = max(col_bottom_row.get(c, -1), r)

    for i, subject in enumerate(subjects):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[top_rows + row, col])
        d = per_subj[per_subj["subject"] == subject]
        draw_trajectories(ax, d, lw=0.9, dot_ms=2.5)
        lim = symmetric_limits(d["x_deg"].to_numpy(), d["y_deg"].to_numpy())
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        label = f"sub-{subject}" if subject.isdigit() else subject
        ax.set_title(label, fontsize=8)
        if col == 0:
            ax.set_ylabel("Gaze y (deg)")
        if row == col_bottom_row[col]:
            ax.set_xlabel("Gaze x (deg)")
        n_ticks = 3
        ax.set_xticks(np.linspace(*lim, n_ticks).round(1))
        ax.set_yticks(np.linspace(*lim, n_ticks).round(1))
        sns.despine(ax=ax, offset=2, trim=True)

    fig.suptitle(f"Gaze trajectories during {args.epoch_title}, by orientation "
                 f"(per-trial recentred on {args.onset_label.splitlines()[0].lower()}; "
                 f"per-panel axes, mean of ≥{MIN_TRIALS} trials/orientation, "
                 f"frac_valid≥{MIN_FRAC_VALID})",
                 fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
