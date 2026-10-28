"""A3 behaviour: category errors around 90 deg / 22 CHF and per-subject behavioural dips.

Outputs (derivatives/followup_2026-10/a3_categorical_90/):
  behavior_crossing.tsv       group crossing rate per (mapping, centre orientation)
                              with participant-bootstrap 95% CI and value gap
  behavior_subject.tsv        per subject x mapping: n category errors at 82.5/97.5,
                              empirical SD dip at 90, model SD dip at 90

Definitions
  crossing rate at centre c: fraction of bids for c-7.5 at or above v(c), plus bids
      for c+7.5 at or below v(c) (pooled). At c = 90 this is the category-error rate
      (bid on the wrong side of 22 CHF for 82.5 / 97.5 deg). Ties (bid == v(c)) count
      as half an error. Comparing 90 to other centres with a similar value gap
      |v(c +- 7.5) - v(c)| asks whether the boundary is respected more than noise allows.
  SD dip (data): log((SD_90 + q) / (mean SD at 75, 82.5, 97.5, 105 + q)), q = 0.144 CHF
      = SD of the 0.5-CHF bid quantisation, so a participant who bids exactly 22 on
      every 90-deg trial does not give -inf. Negative = less variable at 90.
      Also on odd and even runs separately (split-half reliability).
  SD dip (model): same quantity from the posterior predictive of the best cognitive
      model ('Categorical, Fourier K=4', talks/2026-10_update_ruff_bedi/data/ppc_subject.tsv,
      n = 34, 100 draws): dip computed per draw, then averaged over draws.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from abstract_values.behavior.data import get_all_behavioral_data

REPO = Path(__file__).resolve().parents[3]
OUT = Path("/data/ds-abstractvalue/derivatives/followup_2026-10/a3_categorical_90")
Q = 0.5 / np.sqrt(12)
NB = [75.0, 82.5, 97.5, 105.0]
PPC = REPO / "talks/2026-10_update_ruff_bedi/data/ppc_subject.tsv"
PPC_MODEL = "Categorical, Fourier K=4"


def trials():
    df = get_all_behavioral_data()
    df = df[df["event_type"] == "feedback"].copy()
    df["response"] = pd.to_numeric(df["response"], errors="coerce")
    df = df.reset_index().dropna(subset=["response"])
    df["subject"] = df.subject.astype(int)
    return df


def crossing(df):
    """Per subject x mapping x centre: number of crossings and trials."""
    vmap = df.groupby(["mapping", "orientation"]).value.first()
    oris = np.sort(df.orientation.unique())
    rows = []
    for c in oris[1:-1]:
        lo, hi = c - 7.5, c + 7.5
        for m in ["cdf", "inverse_cdf"]:
            vc = vmap[(m, c)]
            gap = np.mean([vc - vmap[(m, lo)], vmap[(m, hi)] - vc])
            s = df[(df.mapping == m) & df.orientation.isin([lo, hi])]
            err = np.where(s.orientation == lo, (s.response > vc) + 0.5 * (s.response == vc),
                           (s.response < vc) + 0.5 * (s.response == vc))
            g = pd.DataFrame(dict(subject=s.subject.values, err=err)).groupby("subject").err.agg(["sum", "count"])
            g = g.reset_index().assign(mapping=m, centre=c, gap=gap)
            rows.append(g)
    return pd.concat(rows)


def boot_rate(g, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(s), (n, len(s)))
    r = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), *np.percentile(r, [2.5, 97.5])


def sd_dip_data(df):
    sd = df.groupby(["subject", "mapping", "orientation"]).response.std().unstack()
    nb = sd[NB].mean(1)
    return np.log((sd[90.0] + Q) / (nb + Q)).rename("sd_dip_data")


def sd_dip_model():
    p = pd.read_csv(PPC, sep="\t")
    p = p[(p.model == PPC_MODEL) & (p.draw >= 0)]
    sd = p.pivot_table(index=["subject", "mapping", "draw"], columns="orientation", values="sd")
    dip = np.log((sd[90.0] + Q) / (sd[NB].mean(1) + Q))
    return dip.groupby(["subject", "mapping"]).mean().rename("sd_dip_model")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = trials()
    cr = crossing(df)
    grp = []
    for (m, c), g in cr.groupby(["mapping", "centre"]):
        r, lo, hi = boot_rate(g)
        grp.append(dict(mapping=m, centre=c, gap=g.gap.iloc[0], rate=r, ci_lo=lo, ci_hi=hi,
                        n_trials=int(g["count"].sum())))
    grp = pd.DataFrame(grp)
    grp.to_csv(OUT / "behavior_crossing.tsv", sep="\t", index=False)
    cat = cr[cr.centre == 90].set_index(["subject", "mapping"])[["sum", "count"]]
    cat.columns = ["cat_err", "cat_n"]
    subj = cat.join(sd_dip_data(df), how="outer").join(sd_dip_model(), how="left")
    run = df.run.astype(int)
    subj = subj.join(sd_dip_data(df[run % 2 == 1]).rename("sd_dip_data_odd"))
    subj = subj.join(sd_dip_data(df[run % 2 == 0]).rename("sd_dip_data_even"))
    subj.reset_index().to_csv(OUT / "behavior_subject.tsv", sep="\t", index=False)
    print(grp[grp.centre.between(60, 120)].round(3).to_string())
    print(subj.groupby("mapping").describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
