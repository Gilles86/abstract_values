"""How much of the mapping change does a signal actually follow?

``mapping_invariance.pdf`` plots a "mapping-specific bias": the bias at each
orientation minus the mean bias across the two conditions. With exactly two
conditions that makes the two curves mirror images of each other *by
construction*, so their symmetry is arithmetic, not evidence. What the curves
do carry is amplitude, and amplitude has an exact reference point.

If a decoder (or a bidder) returns the SAME value for a given orientation in
both conditions -- i.e. ignores the mapping change completely -- then its
demeaned bias is exactly

    +/- (v_inv(theta) - v_cdf(theta)) / 2

which peaks at 3.0 CHF. If instead it follows each condition's mapping
perfectly, the demeaned bias is zero. So regressing the observed demeaned bias
on that prediction gives a slope with a scale everyone can read:

    slope 1  ignores the mapping entirely
    slope 0  fully mapping-specific

    python -m abstract_values.visualize.mapping_specificity
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def half_delta(lut_tsv):
    lut = pd.read_csv(lut_tsv, sep="\t")
    p = lut.pivot(index="orientation", columns="mapping", values="value")
    return ((p["inverse_cdf"] - p["cdf"]) / 2).rename("half_dv").reset_index(), lut


def specificity(df, lut, half):
    """df needs columns mapping, orientation, subject, bias."""
    g = (df.groupby(["mapping", "orientation", "subject"])["bias"].mean()
           .reset_index().merge(half, on="orientation"))
    g["dm"] = g["bias"] - g.groupby(["orientation", "subject"])["bias"].transform("mean")
    g["pred"] = np.where(g.mapping == "cdf", g.half_dv, -g.half_dv)
    return g.groupby("subject").apply(
        lambda x: np.polyfit(x.pred, x.dm, 1)[0], include_groups=False)


def main(trials, decode, key_tsv, lut_tsv):
    half, lut = half_delta(lut_tsv)

    t = pd.read_csv(trials, sep="\t").dropna(subset=["response"])
    t["bias"] = t["response"] - t["value"]
    slope_b = specificity(t, lut, half)

    d = pd.read_csv(decode, sep="\t")
    d = d[d.quantity == "value-weighted"].rename(columns={"true": "value",
                                                          "decoded": "dec"})
    key = pd.read_csv(key_tsv, sep="\t")
    d["sn"] = d.subject.astype(str).str.replace("pil", "", regex=False).astype(int)
    d = d.merge(key.rename(columns={"subject": "sn"}), on=["sn", "session"])
    d = d.merge(lut, on=["mapping", "value"])
    d["bias"] = d.dec - d.value
    slope_n = specificity(d, lut, half)

    for name, s in [("Behaviour (BDM bids)", slope_b), ("NPCr value decode", slope_n)]:
        t1, p1 = stats.ttest_1samp(s, 1)
        t0, p0 = stats.ttest_1samp(s, 0)
        print(f"{name:22s} slope = {s.mean():.3f} (SEM {s.sem():.3f}, n={len(s)}) "
              f"-> {100 * (1 - s.mean()):.0f}% mapping-specific")
        print(f"{'':22s}   vs 1 (ignores mapping): t={t1:6.2f} p={p1:.1e} | "
              f"vs 0 (fully specific): t={t0:5.2f} p={p0:.1e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", default="notes/data/trial_table.tsv")
    p.add_argument("--decode", default="notes/data/decoding_trials_alpha10.tsv")
    p.add_argument("--key", default="notes/data/session_mapping_key.tsv")
    p.add_argument("--lut", default="notes/data/value_orientation_lut.tsv")
    a = p.parse_args()
    main(Path(a.trials), Path(a.decode), Path(a.key), Path(a.lut))
