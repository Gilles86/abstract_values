"""Mapping score (parietal <- V1 coupling that follows the mapping in force) per
target ROI: Wang IPS0 / IPS1-2 / IPS3-5 / IPS0-5 next to the NPC1 / NPC2 / NPC3 /
NPC1-3 numerosity maps (bilateral masks, resampled to the BOLD grid).

Primary spec throughout: V1 channels from the inverted von Mises encoding model
(24 basis functions, kappa 16); lag 0 and with the neighbouring (+-1) trials'
orientations removed. Score per participant = observed - mean(label-shuffled)
(individual_differences.scores). Group: mean +- SEM, one-sample t.

Sensitivity (2026-10-21, Gilles): template 'basis' predicts each voxel's
coupling profile from its full joint value-basis tuning curve (aprf-weighted
design) instead of the single-peak label; and the single-peak score split by
voxel winner: value (aprf.cv) vs orientation (vonmises-prf.cv) model, bell
(aprf.cv) vs basis (aprf-weighted.cv) model, at +-1 lag.

Decoder (decode_mapping.py, lag-0 peaks maps): share of sessions whose mapping
is read out correctly, binomial p against 50 %, and the participant-mean
decoder score with a one-sample t.

    python -m abstract_values.connective_fields.scores_by_roi
    -> notes/data/cf_scores_by_roi.tsv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from abstract_values.connective_fields import individual_differences as idf

REPO = Path(__file__).resolve().parents[2]
ROIS = [("IPS0", "IPS0"), ("IPS1–2", "IPS12"), ("IPS3–5", "IPS345"),
        ("IPS0–5", "IPS0IPS12IPS345"),
        ("NPC1", "NPC1"), ("NPC2", "NPC2"), ("NPC3", "NPC3"), ("NPC1–3", "NPC1NPC2NPC3")]
LAGS = (("Lag 0", ""), ("±1 lag", "lags-1_"))


def decoder(tag):
    f = REPO / f"notes/data/cf_decode_mapping_peaks_target-{tag}.tsv"
    if not f.exists():
        return {}
    res = pd.read_csv(f, sep="\t", dtype={"subject": str})
    k, n = int((res.score > 0).sum()), len(res)
    subj = res.groupby("subject").score.mean()
    return dict(dec_acc=k / n, dec_k=k, dec_n=n,
                dec_p_binom=stats.binomtest(k, n, 0.5, alternative="greater").pvalue,
                dec_score=subj.mean(), dec_score_sem=subj.sem(),
                dec_p_t=stats.ttest_1samp(subj, 0).pvalue)


# (template, voxel class, lags): the single-peak template on all selected voxels
# is the primary; the value-basis template and the voxel-winner splits are
# sensitivity analyses.
VARIANTS = ([("bell", "all", lag) for lag in LAGS] + [("basis", "all", lag) for lag in LAGS]
            + [("bell", c, LAGS[1]) for c in ("value", "orientation", "bell", "basis")])


def folder(tag, template, cls, lag_prefix):
    return (f"coupling_{lag_prefix}iem-k24-kappa16"
            + ("" if cls == "all" else f"_npc-{cls}") + f"_target-{tag}"
            + ("" if template == "bell" else f"_tpl-{template}"))


def median_voxels(fo):
    """Median number of target voxels per participant (n_target in the scores files)."""
    n = [pd.read_csv(f, sep="\t").n_target.iloc[0]
         for f in (idf.CF / fo).glob("sub-*/sub-*_desc-scores.tsv")]
    return float(np.median(n)) if n else np.nan


def table():
    rows = []
    for lab, tag in ROIS:
        dec = decoder(tag)
        for template, cls, (lag, pre) in VARIANTS:
            fo = folder(tag, template, cls, pre)
            x = (idf.scores(fo).dropna() if (idf.CF / fo).exists()
                 else pd.Series(dtype=float))
            t, p = stats.ttest_1samp(x, 0) if len(x) > 1 else (np.nan, np.nan)
            rows.append(dict(roi=lab, roi_tag=tag, family=lab[:3], template=template,
                             voxels=cls, lag=lag, n=len(x), median_voxels=median_voxels(fo),
                             mean=x.mean(), sem=x.sem(),
                             t=t, p=p, **(dec if (template, cls) == ("bell", "all") else {})))
    return pd.DataFrame(rows)


def main():
    r = table()
    out = REPO / "notes/data/cf_scores_by_roi.tsv"
    r.to_csv(out, sep="\t", index=False)
    with pd.option_context("display.width", 200):
        print(r[["roi", "template", "voxels", "lag", "n", "median_voxels", "mean", "sem", "t", "p", "dec_acc", "dec_k", "dec_n",
                 "dec_p_binom", "dec_p_t"]].round(4).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
