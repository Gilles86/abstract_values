"""Posterior predictive check of the SHAPE of the bid distribution.

The bias/SD checks (compare_ppc) cannot tell a sharp peak with heavy shoulders
from a wider Gaussian of the same SD. Here every bid is taken relative to its
participant x mapping x orientation cell median, observed and simulated alike
(simulated bids rounded to the 0.5 CHF slider resolution of the real ones), and
per posterior-predictive draw we keep

  hist   the pooled residual histogram (per mapping)
  cell   per mapping x orientation: the share of residuals beyond 1, 3 and 6 CHF,
         the robust SD (1.4826 MAD) and the SD, averaged over participants

    python -m abstract_values.cogmodels.ppc_shape --paradigm-tsv <tsv> \\
        --fit "label=<trace>:flags" ... --out-dir <dir>
    -> <dir>/ppc_shape_hist.tsv, <dir>/ppc_shape_cells.tsv
Flags as in compare_ppc. Run in the ``bauer`` env.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from abstract_values.cogmodels.compare_ppc import GRID, LAPSE, parse_fit
from abstract_values.cogmodels.fit_efficient_coding import make_model
from abstract_values.cogmodels.ppc_efficient_coding import (load_paradigm, run_ppc,
                                                           subset_paradigm_to_trace)

EDGES = np.arange(-20.25, 20.26, 0.5)          # 0.5 CHF bins centred on the slider steps
CUTS = (1, 3, 6)


def residuals(df):
    cell = ["subject", "mapping", "orientation"]
    return df.assign(res=df.response - df.groupby(cell).response.transform("median"))


def summaries(df):
    r = residuals(df)
    hist = pd.concat([pd.DataFrame({"mapping": m, "bin": 0.5 * (EDGES[1:] + EDGES[:-1]),
                                    "density": np.histogram(np.clip(g.res, EDGES[0] + 1e-6,
                                                                    EDGES[-1] - 1e-6),
                                                            EDGES)[0] / len(g)})
                      for m, g in r.groupby("mapping")])
    a = r.res.abs()
    r = r.assign(**{f"p{c}": a > c for c in CUTS}, absres=a)
    per = r.groupby(["subject", "mapping", "orientation"]).agg(
        **{f"p{c}": (f"p{c}", "mean") for c in CUTS},
        mad_sd=("absres", lambda x: 1.4826 * np.median(x)), sd=("res", "std"))
    return hist, per.groupby(["mapping", "orientation"]).mean().reset_index()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paradigm-tsv", required=True)
    p.add_argument("--fit", action="append", required=True)
    p.add_argument("--n-draws", type=int, default=200)
    p.add_argument("--out-dir", default="notes/data/ppc_shape")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    H, C = [], []
    obs_done = False
    for spec in a.fit:
        label, path, kw, params = parse_fit(spec)
        print(f"== {label}: {path}", flush=True)
        idata = az.from_netcdf(path)
        par = subset_paradigm_to_trace(load_paradigm(a.paradigm_tsv), idata)
        kw = dict(kw)
        model = make_model(par, kw.pop("model_name"), GRID, lapse_rate=LAPSE,
                           perceptual_prior="long_term", **kw)
        if not obs_done:
            h, c = summaries(par.reset_index())
            H.append(h.assign(fit="Observed", draw=-1))
            C.append(c.assign(fit="Observed", draw=-1))
            obs_done = True
        for i, sim in enumerate(run_ppc(model, par, idata, params, a.n_draws)):
            sim = sim.assign(response=np.round(sim.response * 2) / 2)
            h, c = summaries(sim)
            H.append(h.assign(fit=label, draw=i))
            C.append(c.assign(fit=label, draw=i))
    pd.concat(H).to_csv(out / "ppc_shape_hist.tsv", sep="\t", index=False)
    pd.concat(C).to_csv(out / "ppc_shape_cells.tsv", sep="\t", index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
