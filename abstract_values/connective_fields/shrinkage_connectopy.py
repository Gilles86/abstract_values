"""Does the internal mapping's shrinkage track how parietal cortex reads V1?

The mapping-shrinkage efficient-coding model gives each participant a lambda:
their internal orientation -> value mapping is (1 - lambda) G + lambda * line.
A participant with a large lambda follows the mapping in force less faithfully,
so the prediction is that their parietal <- V1 coupling also follows it less:
a NEGATIVE correlation between lambda and the routing score (observed -
label-shuffled match to the in-force mapping, individual_differences.SPECS).

Primary: Spearman rho on the posterior MEANS, participant bootstrap 95% CI and
a permutation p. The measurement error in lambda is already in the
across-participant scatter of those means, so the bootstrap carries it.
Sensitivity (lower bound): rho per posterior draw x participant bootstrap
(brain_behavior.draws_correlation). Each draw is the mean plus posterior noise,
so that version adds the measurement error a second time -- it attenuates rho
toward 0 and widens the interval (over-conservative).
Value precision already predicts the routing score (rho ~ +0.38, figure_summary
panel o), and lambda may share variance with it, so a partial correlation
(ranks; value precision, perceptual precision and head motion removed, all
posterior means) is reported alongside.

    python -m abstract_values.connective_fields.shrinkage_connectopy \\
        --trace <cogmodels_n35>/efficient_coding_categorical_freeprior_..._shrink_trace.nc
    -> notes/data/shrinkage_connectopy.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy import stats

from abstract_values.cogmodels.brain_behavior import draws_correlation
from abstract_values.connective_fields.individual_differences import (SPECS, mapping_fidelity_all,
                                                                      scores)

REPO = Path(__file__).resolve().parents[2]


def ranks_resid(y, Z):
    y = stats.rankdata(y)
    Z = np.column_stack([np.ones(len(y))] + [stats.rankdata(z) for z in Z.T])
    return y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True)
    p.add_argument("--param", default="mapping_shrinkage")
    p.add_argument("--out", default=str(REPO / "notes/data/shrinkage_connectopy.tsv"))
    a = p.parse_args()

    post = az.from_netcdf(a.trace).posterior
    subs = [f"{int(s):02d}" for s in post.subject.values]
    draws = post[a.param].values.reshape(-1, len(subs))
    mean = lambda v: pd.Series(post[v].values.reshape(-1, len(subs)).mean(0), index=subs)
    perc = mean("total_noise") * np.sqrt(mean("perceptual_share"))
    cov = pd.DataFrame({"val_prec": -np.log(mean("sigma_rep")), "perc_prec": -np.log(perc)})
    mo = pd.read_csv(REPO / "notes/data/motion_summary.tsv", sep="\t", dtype={"subject": str})
    mo["subject"] = mo.subject.str.replace("sub-", "").replace({"pil01": "01", "pil02": "02"})
    cov["fd"] = mo.groupby("subject").mean_fd.mean()

    targets = {k: scores(v) for k, v in SPECS.items()}
    targets.update({f"Behaviour: {k}": v for k, v in mapping_fidelity_all().items()})
    rows = []
    for name, y in targets.items():
        keep = [i for i, s in enumerate(subs) if s in y.index and not np.isnan(y.get(s, np.nan))
                and s in cov.dropna().index]
        ss = [subs[i] for i in keep]
        yy = y.loc[ss].to_numpy(float)
        xm = draws[:, keep].mean(0)
        rho_m = stats.spearmanr(xm, yy).statistic
        rng = np.random.default_rng(0)
        bs = [stats.spearmanr(xm[b], yy[b]).statistic
              for b in (rng.integers(0, len(yy), len(yy)) for _ in range(5000))]
        perm = np.array([stats.spearmanr(xm, rng.permutation(yy)).statistic
                         for _ in range(5000)])
        rho, lo, hi = draws_correlation(draws[:, keep], yy, n_draws=2000)
        r_part, p_part = stats.pearsonr(ranks_resid(xm, cov.loc[ss].to_numpy()),
                                        ranks_resid(yy, cov.loc[ss].to_numpy()))
        rows.append(dict(target=name, n=len(ss), rho=rho_m,
                         lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5),
                         p_perm=(np.sum(np.abs(perm) >= abs(rho_m)) + 1) / (len(perm) + 1),
                         rho_draws=rho, lo_draws=lo, hi_draws=hi,
                         partial_rho=r_part, partial_p=p_part))
    res = pd.DataFrame(rows)
    res.insert(0, "param", a.param)
    res.to_csv(a.out, sep="\t", index=False)
    print(res.round(3).to_string(index=False))
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
