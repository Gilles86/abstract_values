"""Does parietal <- V1 coupling follow each participant's OWN (subjective) mapping?

decode_mapping.py scores a session's coupling map (parietal voxels binned by
preferred value x V1 orientation channel) against templates built from the TRUE
orientation -> value mapping. Here the template is built from the participant's
SUBJECTIVE mapping in that session instead:

  empirical  mean bid per orientation (isotonic, so it is invertible)
  model      the fitted internal mapping of the efficient-coding model:
             (1 - lambda) G + lambda * line + mapping_offset (posterior means)

Two scores per participant x session, both > 0 if coupling tracks the
subjective mapping:

  own - others   r(map, T_own) - mean r(map, T_other participant) over the other
                 participants' subjective mappings FOR THE SAME true mapping.
                 Everything the participants share cancels; only the
                 idiosyncratic part of the subjective mapping is left.
  own - true     r(map, T_own) - r(map, T_true mapping)

Group test: one-sample t over participants (sessions averaged), plus a
permutation test that re-assigns subjective mappings across participants within
each true mapping.

    python -m abstract_values.connective_fields.subjective_mapping_connectopy \\
        [--source empirical | --source model --trace <..._offsets_..._trace.nc>]
    -> notes/data/cf_subjective_mapping_<source>.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from abstract_values.connective_fields.decode_mapping import (CF, CONDS, mapping_table, template,
                                                              wcorr)

REPO = Path(__file__).resolve().parents[2]
FOLDER = "peaks_lags-1_target-IPS0IPS12IPS345"


def empirical_mappings():
    from abstract_values.behavior.data import get_all_behavioral_data
    d = get_all_behavioral_data()
    d = d[d.event_type == "feedback"].reset_index()
    d["response"] = pd.to_numeric(d.response, errors="coerce")
    out = {}
    for (s, m), g in d.dropna(subset=["response"]).groupby(["subject", "mapping"]):
        mb = g.groupby("orientation").response.mean().sort_index()
        iso = IsotonicRegression(increasing=True).fit(mb.index, mb.values)
        y = iso.predict(mb.index) + np.linspace(0, 1e-3, len(mb))     # strictly increasing
        out[(f"{int(s):02d}", m)] = pd.Series(y, index=mb.index)
    return out


def model_mappings(trace):
    import arviz as az
    from bauer.efficient_coding import linear_value_np
    post = az.from_netcdf(trace).posterior
    maps = mapping_table()
    lam = post.mapping_shrinkage.mean(("chain", "draw")).to_pandas()
    off = post.mapping_offset.mean(("chain", "draw"))
    out = {}
    for s in off.subject.values:
        for m in off.offset_mapping.values:
            G = maps[str(m)]
            o = off.sel(subject=s, offset_mapping=m).to_pandas()
            o.index = o.index.astype(float)
            v = ((1 - lam[s]) * G + lam[s] * linear_value_np(G.index.values)
                 + o.reindex(G.index).fillna(0.0))
            iso = IsotonicRegression(increasing=True).fit(v.index, v.values)
            y = iso.predict(v.index) + np.linspace(0, 1e-3, len(v))
            out[(f"{int(s):02d}", str(m))] = pd.Series(y, index=v.index)
    return out


def load_maps():
    d = pd.concat(pd.read_csv(f, sep="\t", dtype={"subject": str})
                  for f in (CF / FOLDER).glob("sub-*/sub-*_desc-maps.tsv"))
    d["subject"] = d.subject.replace({"pil01": "01", "pil02": "02"})
    values = np.sort(d.value_bin.unique())
    oris = np.sort(d.orientation.unique())
    out = {}
    for (s, c), g in d.groupby(["subject", "condition"]):
        M = g.pivot(index="value_bin", columns="orientation", values="cf").reindex(
            index=values, columns=oris)
        n = g.groupby("value_bin").n.first().reindex(values).fillna(0).to_numpy()
        ok = np.isfinite(M.to_numpy()).all(1) & (n > 0)
        out[(s, c)] = (M.to_numpy()[ok].ravel(), values[ok],
                       np.repeat(n[ok, None], len(oris), 1).ravel())
    return out, oris


def scores(cmaps, subj_maps, true_maps, oris, assign=None):
    """assign: {(s, c): s_template} -- whose subjective mapping to use as 'own'."""
    rows = []
    for (s, c), (y, vals, w) in cmaps.items():
        others = [k for k in subj_maps if k[1] == c and k[0] != s and k in cmaps or
                  (k[1] == c and k[0] != s)]
        own_key = (assign or {}).get((s, c), (s, c))
        if own_key not in subj_maps:
            continue
        r = lambda mp: wcorr(y, template(mp, vals, oris).ravel(), w)
        r_own = r(subj_maps[own_key])
        r_oth = np.mean([r(subj_maps[k]) for k in others if k != own_key])
        rows.append(dict(subject=s, condition=c, r_own=r_own, r_others=r_oth,
                         r_true=r(true_maps[c])))
    d = pd.DataFrame(rows)
    d["own_minus_others"] = d.r_own - d.r_others
    d["own_minus_true"] = d.r_own - d.r_true
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="empirical", choices=["empirical", "model"])
    p.add_argument("--trace", default=None)
    p.add_argument("--n-perm", type=int, default=500)
    a = p.parse_args()
    cmaps, oris = load_maps()
    true_maps = mapping_table()
    subj = empirical_mappings() if a.source == "empirical" else model_mappings(a.trace)
    d = scores(cmaps, subj, true_maps, oris)
    per = d.groupby("subject")[["own_minus_others", "own_minus_true"]].mean()
    obs = per.own_minus_others.mean()
    print(f"source = {a.source}: n = {len(per)} participants, {len(d)} sessions")
    for k in per:
        t, pv = stats.ttest_1samp(per[k], 0)
        print(f"  {k:17s} {per[k].mean():+.4f} ± {per[k].sem():.4f}  t = {t:+.2f}  p = {pv:.3g}")
    # permutation: shuffle whose subjective mapping counts as 'own', within condition
    rng = np.random.default_rng(0)
    null = []
    keys = list(cmaps)
    for _ in range(a.n_perm):
        assign = {}
        for c in CONDS:
            ks = [k for k in keys if k[1] == c and k in subj]
            perm = rng.permutation(len(ks))
            assign.update({ks[i]: ks[j] for i, j in enumerate(perm)})
        dd = scores(cmaps, subj, true_maps, oris, assign)
        null.append(dd.groupby("subject").own_minus_others.mean().mean())
    null = np.array(null)
    print(f"  permutation p (own - others) = {(np.sum(null >= obs) + 1) / (len(null) + 1):.3f}")
    out = REPO / f"notes/data/cf_subjective_mapping_{a.source}.tsv"
    d.to_csv(out, sep="\t", index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
