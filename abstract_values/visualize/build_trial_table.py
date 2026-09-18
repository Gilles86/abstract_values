"""Per-trial brain × behavior table.

Merges trial-level decoded brain measures (V1-orientation, V1-value,
NPCr-orientation, NPCr-value) with per-trial behavioral outcomes (BDM
response, value, error, rt). For each (quantity, ROI) decoder we extract
the posterior mean, posterior SD, and decoder error per trial; then
merge wide on (subject, session, run, trial_nr).

In addition to the raw columns, the script writes within-stimulus
residualized variants of every numeric column: per (subject, session,
true_value) we subtract the mean. This removes the stimulus-conditional
component from both regressors and outcomes so downstream models can ask
the trial-to-trial coupling question directly.

The BDM auction here is truth-telling: the objective CHF value of the
gabor IS the rational bid. So `error = response - value` is the right
behavioral noise measure.

Output: notes/data/trial_table.tsv (one row per trial × subject).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from abstract_values.behavior.data import get_all_behavioral_data
from abstract_values.utils.data import BIDS_FOLDER

BIDS = Path(BIDS_FOLDER)
DECODE = BIDS / "derivatives" / "decoding"

# (quantity_dir, roi, short_tag, circular)
DECODERS = [
    ("gabor", "BensonV1", "v1_ori",   True),
    ("gabor", "NPCr",     "npcr_ori", True),
    ("value", "BensonV1", "v1_val",   False),
    ("value", "NPCr",     "npcr_val", False),
]


def _pars_path(quantity, subject, roi, nvoxels, smoothed, noise, lam):
    smooth = "_smoothed" if smoothed else ""
    d = DECODE / quantity / f"sub-{subject}" / "func"
    lam_tag = f"_lambda-{lam}" if lam else ""
    stem = f"sub-{subject}_mask-{roi}_nvoxels-{nvoxels}_noise-{noise}{smooth}{lam_tag}"
    hit = d / f"{stem}_pars.tsv"
    return hit if hit.exists() else None


def _per_trial_brain(pars_path, circular):
    """Return DataFrame indexed by (session, run, trial_nr) with columns
    post_mean, post_sd, decode_error, abs_decode_error.

    Orientation decoders carry posteriors on radians; SD / |err| are
    converted to degrees and `decode_error` is wrapped to a signed
    [-90, 90)° interval on a circular axis."""
    df = pd.read_csv(pars_path, sep="\t")
    true_col = "true_orientation_rad" if circular else "true_value_chf"
    meta = ["session", "run", "trial_nr", true_col]
    grid = np.array([float(c) for c in df.columns if c not in meta])
    post = df[[c for c in df.columns if c not in meta]].to_numpy(float)
    w = post / np.clip(post.sum(1, keepdims=True), 1e-12, None)
    true = df[true_col].to_numpy(float)
    if circular:
        a = 2.0 * grid
        c, s = (w * np.cos(a)).sum(1), (w * np.sin(a)).sum(1)
        dec = 0.5 * np.arctan2(s, c) % np.pi
        R = np.clip(np.sqrt(c**2 + s**2), 1e-12, 1.0)
        post_sd = np.sqrt(-2.0 * np.log(R)) / 2.0
        err = (dec - true + np.pi / 2) % np.pi - np.pi / 2
        post_mean = np.rad2deg(dec)
        post_sd = np.rad2deg(post_sd)
        decode_error = np.rad2deg(err)
    else:
        dec = (w * grid).sum(1)
        post_sd = np.sqrt((w * (grid[None, :] - dec[:, None]) ** 2).sum(1))
        post_mean = dec
        decode_error = dec - true
    return pd.DataFrame({
        "session": df["session"].astype(int).values,
        "run": df["run"].astype(int).values,
        "trial_nr": df["trial_nr"].astype(int).values,
        "post_mean": post_mean,
        "post_sd": post_sd,
        "decode_error": decode_error,
        "abs_decode_error": np.abs(decode_error),
    }).set_index(["session", "run", "trial_nr"])


def _load_behavior():
    """One row per trial (feedback event) with response, value, orientation,
    rt, error, abs_error. Mirrors notebooks/behavior_overview.ipynb."""
    df = get_all_behavioral_data()
    df = df[df["event_type"] == "feedback"].copy()
    df["response"] = pd.to_numeric(df["response"], errors="coerce")
    df["error"] = df["response"] - df["value"]
    df["abs_error"] = df["error"].abs()
    df = df.reset_index()
    keep = ["subject", "session", "mapping", "run", "trial_nr",
            "response", "value", "orientation", "rt",
            "error", "abs_error"]
    return df[keep]


def _residualize_within_stim(df, group_cols, num_cols):
    """For each numeric col, add `<col>_resid = col − mean(col | group_cols)`."""
    grp = df.groupby(group_cols, sort=False)
    for c in num_cols:
        df[f"{c}_resid"] = df[c] - grp[c].transform("mean")
    return df


def build(subjects, nvoxels, noise, lam, smoothed):
    beh = _load_behavior()
    if subjects is not None:
        beh = beh[beh["subject"].isin([int(s) for s in subjects])]
    if beh.empty:
        raise SystemExit("No behavioral data for requested subjects.")

    rows = []
    for s, beh_s in beh.groupby("subject", sort=True):
        label = f"{int(s):02d}"
        merged = beh_s.set_index(["session", "run", "trial_nr"]).copy()
        any_brain = False
        for quantity, roi, tag, circular in DECODERS:
            p = _pars_path(quantity, label, roi, nvoxels, smoothed, noise, lam)
            if p is None:
                continue
            brain = _per_trial_brain(p, circular).add_prefix(f"{tag}_")
            merged = merged.join(brain, how="left")
            any_brain = True
        if not any_brain:
            print(f"  sub-{label}: no decoded pars for {nvoxels}/{noise}/lam{lam}, skipping")
            continue
        rows.append(merged.reset_index())
        has_brain = merged.filter(like="_post_sd").notna().any(axis=1).sum()
        print(f"  sub-{label}: {len(merged)} trials  ({has_brain} with any brain data)")
    if not rows:
        raise SystemExit("No subjects yielded data.")
    return pd.concat(rows, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Subject labels (without 'sub-'). Default: all.")
    p.add_argument("--nvoxels", default="100")
    p.add_argument("--noise", default="spherical",
                   choices=["spherical", "full"])
    p.add_argument("--lam", default="0.1",
                   help="Lambda tag in pars filename ('' to drop)")
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--out", default=str(Path("notes") / "data" / "trial_table.tsv"))
    args = p.parse_args()

    print(f"Variant: nvoxels={args.nvoxels} noise={args.noise} "
          f"lam={args.lam} smoothed={args.smoothed}")

    table = build(args.subjects, args.nvoxels, args.noise,
                  args.lam, args.smoothed)

    # Residualize: per (subject, session, value) means subtracted from all
    # numeric brain & behavior columns. value = trained CHF level — that
    # removes stimulus-conditional brain effects (efficient-coding shape,
    # trained-grid edge artefacts) and stimulus-conditional behavioral
    # effects (boundary regression to mean).
    skip = {"subject", "session", "run", "trial_nr", "mapping",
            "value", "orientation"}
    num_cols = [c for c in table.columns
                if c not in skip and pd.api.types.is_numeric_dtype(table[c])]
    table = _residualize_within_stim(
        table, group_cols=["subject", "session", "value"], num_cols=num_cols)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path}")
    print(f"  rows: {len(table)}   cols: {len(table.columns)}")
    print(f"  subjects: {sorted(table['subject'].unique().tolist())}")
    print("\nNon-null coverage of brain regressors:")
    for tag in [d[2] for d in DECODERS]:
        col = f"{tag}_post_sd"
        if col in table.columns:
            n = table[col].notna().sum()
            print(f"  {col:24s}  n={n}  ({n/len(table):.0%})")
    print("\nBehavioral coverage:")
    for c in ("response", "error", "rt"):
        n = table[c].notna().sum()
        print(f"  {c:24s}  n={n}  ({n/len(table):.0%})")


if __name__ == "__main__":
    main()
