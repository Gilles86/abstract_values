"""Which encoding model wins in which retinotopic area?

Takes the per-vertex winner-take-all of ``model_winner_maps`` and aggregates it
over the anatomical atlases from ``surface/infer_neuropythy_atlas.py``
(Benson-14 visual areas, Wang-15 max-probability labels, which unlike Benson
extend into parietal cortex). The question it answers: is the value model's
advantage anatomically specific, and if so where.

Atlas labels are per subject in fsaverage space; a vertex is assigned the
*modal* label across subjects, so a region is where the cohort agrees.

Win share is pooled over (subject, vertex) pairs that carry signal -- the same
"beats that subject's own null" criterion the maps use -- rather than averaged
over per-vertex fractions, so a vertex where only 3 subjects have signal does
not count as much as one where all 29 do.

Usage
-----
    python -m abstract_values.visualize.winner_by_visual_area
    python -m abstract_values.visualize.winner_by_visual_area --atlas wang15 \
        --tsv notes/data/winner_by_area_wang15.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from abstract_values.utils.data import BIDS_FOLDER
from abstract_values.visualize.group_surface_maps import discover_subjects
from abstract_values.visualize.model_winner_maps import (
    CANDIDATES, winner_per_subject)

# From surface/infer_neuropythy_atlas.py's docstring.
BENSON14 = {1: "V1", 2: "V2", 3: "V3", 4: "hV4", 5: "VO1", 6: "VO2",
            7: "LO1", 8: "LO2", 9: "TO1", 10: "TO2", 11: "V3b", 12: "V3a"}
WANG15 = {1: "V1v", 2: "V1d", 3: "V2v", 4: "V2d", 5: "V3v", 6: "V3d",
          7: "hV4", 8: "VO1", 9: "VO2", 10: "PHC1", 11: "PHC2",
          12: "TO2", 13: "TO1", 14: "LO2", 15: "LO1", 16: "V3B", 17: "V3A",
          18: "IPS0", 19: "IPS1", 20: "IPS2", 21: "IPS3", 22: "IPS4",
          23: "IPS5", 24: "SPL1", 25: "FEF"}
ATLASES = {"benson14": ("benson14Varea", BENSON14),
           "wang15": ("wang15Mplbl", WANG15)}


def load_atlas(deriv, subjects, desc):
    """Modal atlas label per fsaverage vertex, L+R concatenated."""
    stack = []
    for s in subjects:
        hemis = []
        for hemi in ("L", "R"):
            fn = (deriv / "neuropythy_atlas" / f"sub-{s}" /
                  f"sub-{s}_desc-{desc}_space-fsaverage_hemi-{hemi}.func.gii")
            if not fn.exists():
                hemis = None
                break
            hemis.append(nib.load(str(fn)).darrays[0].data)
        if hemis is not None:
            stack.append(np.concatenate(hemis))
    if not stack:
        raise SystemExit(f"No {desc} atlas files found.")
    lab = np.rint(np.vstack(stack)).astype(np.int16)
    lab[lab < 0] = 0
    n_lab = int(lab.max()) + 1
    counts = np.zeros((n_lab, lab.shape[1]), dtype=np.int32)
    for row in lab:
        counts[row, np.arange(lab.shape[1])] += 1
    print(f"  atlas {desc}: {len(stack)} subjects, {n_lab - 1} labels")
    return counts.argmax(axis=0)


def area_table(wins, signal, labels_model, atlas, names, min_prevalence):
    gate = signal.mean(axis=0) >= min_prevalence
    rows = []
    for code, name in sorted(names.items()):
        mask = (atlas == code) & gate
        n_vox = int(mask.sum())
        if n_vox == 0:
            continue
        sig = signal[:, mask]
        n_sig = int(sig.sum())
        if n_sig == 0:
            continue
        row = {"area": name, "n_vertices": n_vox,
               "prevalence": float(signal[:, mask].mean())}
        for m, label in enumerate(labels_model):
            row[label] = float(((wins[:, mask] == m) & sig).sum() / n_sig)
        rows.append(row)
    df = pd.DataFrame(rows)
    ori = [l for l in labels_model if "vonMises" in l]
    val = [l for l in labels_model if l not in ori]
    df["orientation"] = df[ori].sum(axis=1)
    df["value"] = df[val].sum(axis=1)
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-folder", default=str(BIDS_FOLDER))
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--atlas", default="both",
                   choices=["benson14", "wang15", "both"])
    p.add_argument("--smoothed", action="store_true")
    p.add_argument("--min-prevalence", type=float, default=0.4)
    p.add_argument("--tsv", default=None, help="write the table here")
    args = p.parse_args()

    deriv = Path(args.bids_folder) / "derivatives"
    subjects = args.subjects or discover_subjects(deriv)
    wins, signal, labels, used = winner_per_subject(
        deriv, subjects, CANDIDATES, args.smoothed)
    if wins is None:
        raise SystemExit("No winner data.")
    print(f"n={len(used)} subjects, models: {', '.join(labels)}\n")

    out = []
    for key in (["benson14", "wang15"] if args.atlas == "both" else [args.atlas]):
        desc, names = ATLASES[key]
        atlas = load_atlas(deriv, used, desc)
        df = area_table(wins, signal, labels, atlas, names, args.min_prevalence)
        df.insert(0, "atlas", key)
        pct = df.copy()
        for c in labels + ["orientation", "value", "prevalence"]:
            pct[c] = (100 * pct[c]).round(1)
        print(f"\n=== {key}: win share (%) of signal (subject, vertex) pairs, "
              f"prevalence >= {args.min_prevalence:.0%} ===")
        print(pct.drop(columns="atlas").to_string(index=False))
        out.append(df)

    if args.tsv:
        pd.concat(out).to_csv(args.tsv, sep="\t", index=False)
        print(f"\nWrote {args.tsv}")


if __name__ == "__main__":
    main()
