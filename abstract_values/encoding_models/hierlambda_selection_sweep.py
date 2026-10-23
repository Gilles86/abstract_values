"""How much does voxel selection matter for the hierarchical lambda estimate?

Runs ``hierarchical_lambda`` for each voxel gate (std / neutral / shift, see
``hierarchical_lambda.SELECT``) at increasingly strict cvR^2-over-null
thresholds, then summarises per (gate, threshold, ROI group): voxels and
participants kept, mu and tau, value-vs-quantile random-effects BMS, and plain
Spearman correlations of the per-participant posterior means with the left/right
hemisphere estimates, the cvR^2-profile lambda-hat, behavioural lambda and the
cognitive model's value noise and mapping shrinkage (shrink + motor-tail fit).

    python -m abstract_values.encoding_models.hierlambda_selection_sweep [--run]
    -> notes/data/hierlambda_selection/summary.tsv
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from abstract_values.encoding_models.lambda_model_comparison import rfx_bms

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "notes/data"
OUT = DATA / "hierlambda_selection"
SELECTS = ["std", "neutral", "shift"]
GAINS = [0., .002, .005, .01, .02]
GROUPS = ["IPS", "IPS0", "IPS12", "IPS345"]


def tag(sel, gain):
    return f"lognormal_sel-{sel}{gain:g}"


def run_one(sel, gain):
    if (OUT / f"hierlambda_{tag(sel, gain)}_group.tsv").exists():
        return
    subprocess.run([sys.executable, "-m", "abstract_values.encoding_models.hierarchical_lambda",
                    "--select", sel, "--min-gain", str(gain), "--groups", *GROUPS],
                   check=True, stdout=subprocess.DEVNULL)
    print(f"done {sel} {gain}", flush=True)


def sid(s):
    return int(str(s).replace("pil", ""))


def externals():
    lh = pd.read_csv(DATA / "lambda_hat.tsv", sep="\t", dtype={"subject": str})
    lh = lh[lh.smoothed.astype(str) == "False"]
    prof = {r: d.set_index(d.subject.map(sid)).lam_hat for r, d in lh.groupby("roi")}
    ips = lh[lh.roi.isin(["IPS0", "IPS12", "IPS345"])]
    w = ips.assign(w=ips.lam_hat * ips.n_voxels).groupby("subject")[["w", "n_voxels"]].sum()
    prof["IPS"] = pd.Series((w.w / w.n_voxels).to_numpy(), index=w.index.map(sid))
    lb = pd.read_csv(DATA / "lambda_behaviour.tsv", sep="\t", dtype={"subject": str})
    cog = pd.read_csv(DATA / "cogmodel_shrink_mtail_subject_params.tsv", sep="\t")
    cog = cog.set_index(cog.subject.astype(int))
    return dict(profile=prof, lam_b=lb.set_index(lb.subject.map(sid)).lam_b,
                sigma_rep=cog.sigma_rep, shrinkage=cog.mapping_shrinkage)


def rho(a, b):
    b = b.reindex(a.index)
    ok = b.notna()
    r, p = stats.spearmanr(a[ok], b[ok])
    return r, p


def summarise():
    ext = externals()
    rows = []
    for sel, gain in itertools.product(SELECTS, GAINS):
        g = pd.read_csv(OUT / f"hierlambda_{tag(sel, gain)}_group.tsv", sep="\t").set_index("roi")
        s = pd.read_csv(OUT / f"hierlambda_{tag(sel, gain)}_subjects.tsv", sep="\t",
                        dtype={"subject": str})
        for roi, d in s.groupby("roi", sort=False):
            d = d.set_index(d.subject.map(sid))
            m = d["mean"]
            _, freq, xp, pxp, bor = rfx_bms(d[["logev_lam0", "logev_lam1"]].to_numpy())
            row = dict(select=sel, min_gain=gain, roi=roi, n_subjects=len(d),
                       n_voxels=int(g.loc[roi, "n_voxels"]),
                       median_voxels=d.n_voxels.median(),
                       mu=g.loc[roi, "mu"], mu_lo=g.loc[roi, "mu_lo"], mu_hi=g.loc[roi, "mu_hi"],
                       tau=g.loc[roi, "tau"], temperature=g.loc[roi, "temperature"],
                       freq_quantile=freq[1], xp_quantile=xp[1], pxp_quantile=pxp[1], bor=bor,
                       n_bf3_quantile=int((d.logev_lam1 - d.logev_lam0 > np.log(3)).sum()),
                       n_bf3_value=int((d.logev_lam0 - d.logev_lam1 > np.log(3)).sum()))
            row["rho_LR"], row["p_LR"] = stats.spearmanr(d.L_mean, d.R_mean)
            for k in ("lam_b", "sigma_rep", "shrinkage"):
                row[f"rho_{k}"], row[f"p_{k}"] = rho(m, ext[k])
            row["rho_profile"], row["p_profile"] = rho(m, ext["profile"][roi if roi in ext["profile"]
                                                                         else "IPS"])
            rows.append(row)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "summary.tsv", sep="\t", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    print(r.round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="run the missing fits first")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.run:
        OUT.mkdir(exist_ok=True)
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(lambda x: run_one(*x), itertools.product(SELECTS, GAINS)))
    summarise()
