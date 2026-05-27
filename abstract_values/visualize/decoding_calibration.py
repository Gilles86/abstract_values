"""Decoder calibration: empirical error vs posterior SD vs expected uncertainty.

For each decoded quantity, as a function of the *true* stimulus, compares three
spread measures that should roughly agree if the decoder is well-calibrated and
the simulation captures reality:

  - **Empirical error** — RMS of (decoded − true) across real trials
    (circular distance for orientation). The decoder's actual error.
  - **Posterior SD** — mean per-trial width of the decoded posterior
    (circular SD for orientation). The decoder's *claimed* uncertainty.
  - **Expected uncertainty** — sd_E = √var_E from the simulation-based
    expected-decoded product (compute_expected_decoded_*). The *predicted* SD.

  value       : decoding/value (NPCr)        vs aprf-session-shift expected_decoded
  orientation : decoding/gabor (BensonV1)    vs vonmises expected_decoded_orientation

Real decoded estimates come from the per-trial posterior in the decoding
``*_pars.tsv``; expected uncertainty is pooled across the subject's sessions.

    python -m abstract_values.visualize.decoding_calibration
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
from matplotlib.backends.backend_pdf import PdfPages

from abstract_values.utils.data import BIDS_FOLDER

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150, "savefig.dpi": 300,
})

BIDS = Path(BIDS_FOLDER)
DECODE = BIDS / "derivatives" / "decoding"
ENC = BIDS / "derivatives" / "encoding_models"

CURVE = {"empirical": "#E76F51", "posterior": "#3B5BA5", "expected": "#2A9D8F"}
LABEL = {"empirical": "Empirical error (RMS)",
         "posterior": "Posterior SD (claimed)",
         "expected": "Expected uncertainty (sim)"}

# quantity -> config
Q = {
    "value": dict(
        decode_dir=DECODE / "value", roi="NPCr", true_col="true_value_chf",
        circular=False, unit="CHF",
        exp_dir=ENC / "aprf-session-shift",
        exp_glob="*_mask-NPCr_nvoxels-100_nsims-1000_desc-expected_decoded_pe.tsv",
        exp_x="value", exp_sd=None),                      # sd from sqrt(var_E)
    "orientation": dict(
        decode_dir=DECODE / "gabor", roi="BensonV1", true_col="true_orientation_rad",
        circular=True, unit="deg",
        exp_dir=ENC / "vonmises",
        exp_glob="*_mask-BensonV1_hemi-LR_nvoxels-100_nsims-1000_desc-expected_decoded_orientation_pe.tsv",
        exp_x="value_deg", exp_sd="sd_E"),                # sd_E already in rad
}


# Noise model whose decoding pars to read (set from --noise in main()).
NOISE = "full"


def _decode_pars(quantity, subject, nvoxels, smoothed):
    cfg = Q[quantity]
    smooth = "_smoothed" if smoothed else ""
    d = cfg["decode_dir"] / f"sub-{subject}" / "func"
    stem = f"sub-{subject}_mask-{cfg['roi']}_nvoxels-{nvoxels}_noise-{NOISE}{smooth}"
    hits = sorted(d.glob(f"{stem}_pars.tsv")) or sorted(d.glob(f"{stem}_lambda-*_pars.tsv"))
    return hits[0] if hits else d / f"{stem}_pars.tsv"


def _per_trial(quantity, df):
    """Return (true, decoded, post_sd) per trial. Units: rad for orientation."""
    cfg = Q[quantity]
    meta = ["session", "run", "trial_nr", cfg["true_col"]]
    grid = np.array([float(c) for c in df.columns if c not in meta])
    post = df[[c for c in df.columns if c not in meta]].to_numpy(float)
    w = post / np.clip(post.sum(1, keepdims=True), 1e-12, None)
    true = df[cfg["true_col"]].to_numpy(float)
    if cfg["circular"]:
        a = 2.0 * grid
        c, s = (w * np.cos(a)).sum(1), (w * np.sin(a)).sum(1)
        dec = 0.5 * np.arctan2(s, c) % np.pi
        R = np.clip(np.sqrt(c**2 + s**2), 1e-12, 1.0)
        post_sd = np.sqrt(-2.0 * np.log(R)) / 2.0           # circular SD (rad)
    else:
        dec = (w * grid).sum(1)
        post_sd = np.sqrt((w * (grid[None, :] - dec[:, None])**2).sum(1))
    return true, dec, post_sd


def _err(quantity, true, dec):
    if Q[quantity]["circular"]:
        d = (dec - true) % np.pi
        return np.where(d > np.pi / 2, d - np.pi, d)         # signed circ dist (rad)
    return dec - true


def collect_real(quantity, subject, nvoxels, smoothed):
    """Per true-stimulus level: empirical RMS error and mean posterior SD."""
    p = _decode_pars(quantity, subject, nvoxels, smoothed)
    if not p.exists():
        return None
    true, dec, psd = _per_trial(quantity, pd.read_csv(p, sep="\t"))
    err = _err(quantity, true, dec)
    deg = Q[quantity]["unit"] == "deg"
    df = pd.DataFrame({"true": true, "err": err, "psd": psd})
    g = df.groupby(np.round(df["true"], 4)).agg(
        rms=("err", lambda e: np.sqrt(np.mean(e**2))),
        psd=("psd", "mean")).reset_index(names="true")
    if deg:
        g["true"] = np.rad2deg(g["true"]); g["rms"] = np.rad2deg(g["rms"])
        g["psd"] = np.rad2deg(g["psd"])
    return g


def collect_expected(quantity, subject, smoothed):
    cfg = Q[quantity]
    smooth = "_smoothed" if smoothed else ""
    rows = []
    for ses in (1, 2):
        d = cfg["exp_dir"] / f"sub-{subject}" / f"ses-{ses}" / "func"
        glob = cfg["exp_glob"].replace("nsims-1000", f"nsims-1000{smooth}") \
            if smoothed else cfg["exp_glob"]
        hits = list(d.glob(glob)) if d.exists() else []
        for f in hits:
            e = pd.read_csv(f, sep="\t")
            sd = e[cfg["exp_sd"]] if cfg["exp_sd"] else np.sqrt(e["var_E"])
            x = e[cfg["exp_x"]]
            if cfg["unit"] == "deg" and cfg["exp_sd"]:
                sd = np.rad2deg(sd)
            rows.append(pd.DataFrame({"true": x.to_numpy(float),
                                      "expected": np.asarray(sd, float)}))
    if not rows:
        return None
    return pd.concat(rows).groupby("true", as_index=False)["expected"].mean()


def _panel(ax, quantity, reals, exps, subjects):
    """Group-mean ± SEM of the three curves on a common x (true stimulus)."""
    cfg = Q[quantity]
    # empirical & posterior live on the discrete true grid; pool across subjects
    real = pd.concat([r.assign(subject=s) for s, r in zip(subjects, reals)
                      if r is not None], ignore_index=True)
    for key in ("rms", "psd"):
        g = real.groupby("true")[key].agg(["mean", "sem"]).reset_index()
        which = "empirical" if key == "rms" else "posterior"
        ax.plot(g["true"], g["mean"], color=CURVE[which], lw=1.8, label=LABEL[which])
        ax.fill_between(g["true"], g["mean"] - g["sem"], g["mean"] + g["sem"],
                        color=CURVE[which], alpha=0.2, lw=0)
    exp = pd.concat([e for e in exps if e is not None], ignore_index=True)
    if not exp.empty:
        # expected is on a fine grid; bin lightly for a smooth mean curve
        exp["bin"] = np.round(exp["true"], 1)
        g = exp.groupby("bin")["expected"].agg(["mean", "sem"]).reset_index()
        ax.plot(g["bin"], g["mean"], color=CURVE["expected"], lw=1.8,
                label=LABEL["expected"])
        ax.fill_between(g["bin"], g["mean"] - g["sem"], g["mean"] + g["sem"],
                        color=CURVE["expected"], alpha=0.2, lw=0)
    ax.set_xlabel(f"True {quantity} ({cfg['unit']})")
    ax.set_ylabel(f"Spread ({cfg['unit']})")
    n = sum(r is not None for r in reals)
    ax.set_title(f"{quantity.capitalize()} · {cfg['roi']}  (n={n})",
                 fontsize=10, color="0.2")
    ax.set_ylim(bottom=0)
    if cfg["unit"] == "deg":
        ax.set_xticks([0, 45, 90, 135, 180])
    ax.legend(loc="upper center", fontsize=7.5, ncol=1)


def run(subjects, nvoxels, smoothed, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for ax, q in zip(axes, ("value", "orientation")):
            reals = [collect_real(q, s, nvoxels, smoothed) for s in subjects]
            exps = [collect_expected(q, s, smoothed) for s in subjects]
            _panel(ax, q, reals, exps, subjects)
        fig.suptitle("Decoder calibration: empirical error vs posterior SD vs "
                     f"expected uncertainty  ·  {'smoothed' if smoothed else 'unsmoothed'}"
                     f"  ·  nvoxels={nvoxels}", fontsize=11, y=1.05)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {out}")


def discover(nvoxels, smoothed):
    smooth = "_smoothed" if smoothed else ""
    subs = set()
    for q in Q:
        for pat in (f"sub-*/func/*_nvoxels-{nvoxels}_noise-{NOISE}{smooth}_pars.tsv",
                    f"sub-*/func/*_nvoxels-{nvoxels}_noise-{NOISE}{smooth}_lambda-*_pars.tsv"):
            for p in (Q[q]["decode_dir"]).glob(pat):
                subs.add(p.name.split("_")[0].removeprefix("sub-"))
    # only subjects that also have at least one expected product
    keep = [s for s in subs
            if any(collect_expected(q, s, smoothed) is not None for q in Q)]
    return sorted(keep, key=lambda s: (0 if s[0].isdigit() else 1, s))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--nvoxels", default="100")
    p.add_argument("--noise", default="full", choices=["full", "spherical", "geodesic"],
                   help="noise model whose decoding pars to read")
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--out", default=str(BIDS / "derivatives" / "qa"
                                        / "decoding_calibration.pdf"))
    args = p.parse_args()
    global NOISE
    NOISE = args.noise
    subjects = args.subjects or discover(args.nvoxels, args.smoothed)
    if not subjects:
        raise SystemExit("No subjects with both decoding + expected-decoded data")
    print(f"Subjects ({len(subjects)}): {subjects}")
    run(subjects, args.nvoxels, args.smoothed, Path(args.out))


if __name__ == "__main__":
    main()
