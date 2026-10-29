#!/usr/bin/env python3
"""A1 statistics: what does M1 encode, does it add noise, and does it track behavioural noise?

Reads the tables written on the cluster by
abstract_values/visualize/collect_motor_tuning.py (rsynced to the local BIDS root):

    derivatives/followup_2026-10/a1_motor_stage/tables/
        motor_tuning_trials.tsv          ROI-mean single-trial betas, both epochs
        motor_tuning_linear_cv.tsv       closed-form LORO cvR² (value, bid, travel, RT, ...)
        motor_tuning_prf_cv.tsv          PRF-model cvR² (encoding_models/fit_motor_cv.py)
        motor_tuning_decoding.tsv        LORO ridge decoding summaries
        motor_tuning_decoding_trials.tsv trial-level decoded bids

and writes ``stats_*.tsv`` next to them. Every effect is reported as the mean over
subjects with a t-based 95% CI (or, for correlations, Spearman ρ with a Fisher-z
95% CI, plus Pearson partial r controlling mean FD). No resampling.

    ~/mambaforge/bin/python analyze_motor_stage.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
TAB = Path("/data/ds-abstractvalue/derivatives/followup_2026-10/a1_motor_stage/tables")
SLIDER = 42.0
ROIS = ["M1lhand", "M1rhand", "M1l", "M1r", "preSMA", "V1", "IPS12", "IPS345"]


def extent(b):
    return (b ** 2 + (SLIDER - b) ** 2) / (2 * SLIDER)


def summ(v, **keys):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(.975, n - 1) * se
    t = m / se
    return dict(**keys, mean=m, ci_lo=m - h, ci_hi=m + h, t=t,
                p=2 * stats.t.sf(abs(t), n - 1), dz=m / v.std(ddof=1), n=n)


def spearman_ci(x, y):
    rho, p = stats.spearmanr(x, y)
    n = len(x)
    z, se = np.arctanh(rho), 1.06 / np.sqrt(n - 3)          # Fieller et al. SE for Spearman
    return rho, p, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def partial_r(x, y, z):
    Z = np.column_stack([np.ones(len(z)), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r = np.corrcoef(rx, ry)[0, 1]
    n = len(x)
    t = r * np.sqrt((n - 3) / (1 - r ** 2))
    zz, se = np.arctanh(r), 1 / np.sqrt(n - 4)
    return r, 2 * stats.t.sf(abs(t), n - 3), np.tanh(zz - 1.96 * se), np.tanh(zz + 1.96 * se)


def trial_coefs(t, cols, epoch):
    rows = []
    for (roi, sub), s in t[t.epoch == epoch].groupby(["roi", "subject"]):
        s = s.copy()
        for c in ["beta"] + cols:
            s[c] = s[c] - s.groupby(["session", "run"])[c].transform("mean")
        X = s[cols].to_numpy()
        X = (X - X.mean(0)) / X.std(0)
        y = (s.beta / s.beta.std()).to_numpy()
        b = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0][1:]
        rows.append(dict(roi=roi, subject=sub, epoch=epoch, **dict(zip(cols, b))))
    return pd.DataFrame(rows)


def main():
    out = []
    # ── 1. trial-wise regression on ROI-mean betas ──────────────────────────
    t = pd.read_csv(TAB / "motor_tuning_trials.tsv", sep="\t", dtype={"subject": str})
    t = t[t.bid.notna() & t.rt.notna()].copy()
    t["travel"] = extent(t.bid)
    t["travel_value"] = extent(t.value)
    t["cos2theta"] = np.cos(np.deg2rad(t.orientation) * 2)
    coefs = {}
    for name, cols in [("main", ["travel", "bid", "cos2theta", "rt"]),
                       ("bid_vs_value_travel", ["travel", "travel_value", "rt", "cos2theta"])]:
        for ep in ("gabor", "response"):
            c = trial_coefs(t, cols, ep)
            coefs[(name, ep)] = c
            for roi in ROIS:
                for k in cols:
                    out.append(summ(c.loc[c.roi == roi, k], analysis=f"regression:{name}",
                                    epoch=ep, roi=roi, term=k))
            w = c.pivot(index="subject", columns="roi", values="travel")
            out.append(summ(w.M1lhand - w.M1rhand, analysis=f"regression:{name}", epoch=ep,
                            roi="M1lhand-M1rhand", term="travel"))
            out.append(summ(w.M1lhand - w.preSMA, analysis=f"regression:{name}", epoch=ep,
                            roi="M1lhand-preSMA", term="travel"))

    # ── 2. linear cvR² (×10⁻³ over the intercept-only null) ────────────────
    lin = pd.read_csv(TAB / "motor_tuning_linear_cv.tsv", sep="\t", dtype={"subject": str},
                      keep_default_na=False)
    w = lin.pivot_table(index=["subject", "epoch", "roi"], columns="model", values="cvr2") * 1e3
    for m in ["value", "bid", "extent", "rt", "orientation", "rank"]:
        w[f"{m}-null"] = w[m] - w["null"]
    w["bid+rt-rt"] = w["bid+rt"] - w["rt"]
    w["bid+extent-bid"] = w["bid+extent"] - w["bid"]
    for (ep, roi), g in w.groupby(["epoch", "roi"]):
        for k in [c for c in w.columns if "-" in c]:
            out.append(summ(g[k], analysis="linear_cvr2_x1e3", epoch=ep, roi=roi, term=k))

    # ── 3. PRF cvR² (×10⁻³) ────────────────────────────────────────────────
    prf = pd.read_csv(TAB / "motor_tuning_prf_cv.tsv", sep="\t", dtype={"subject": str},
                      keep_default_na=False)
    # ROIs without PRF fits (V1, IPS) have empty cvr2 cells, read as "" under keep_default_na
    prf = prf[prf.n_vox > 0].astype({"cvr2": float, "cvr2_null": float})
    prf["gain"] = (prf.cvr2 - prf.cvr2_null) * 1e3
    w = prf.pivot_table(index=["subject", "epoch", "roi"], columns="model", values="gain")
    w["wrapped-unwrapped"] = w["quantile-wrapped"] - w["quantile"]
    w["quantile-value"] = w["quantile"] - w["gaussian"]
    w["bid-value (same trials)"] = w["gaussian-bid"] - w["gaussian-bidmatched"]
    w["linear bid-value (same trials)"] = w["linear-bid"] - w["linear-bidmatched"]
    for (ep, roi), g in w.groupby(["epoch", "roi"]):
        for k in w.columns:
            out.append(summ(g[k], analysis="prf_cvr2_minus_null_x1e3", epoch=ep, roi=roi,
                            term=k))

    # ── 4. decoding ─────────────────────────────────────────────────────────
    dec = pd.read_csv(TAB / "motor_tuning_decoding.tsv", sep="\t", dtype={"subject": str})
    dec["z"] = np.arctanh(dec.r.clip(-.999, .999))
    for (ep, roi, tg), g in dec.groupby(["epoch", "roi", "target"]):
        out.append(summ(g.z, analysis="decoding_fisher_z", epoch=ep, roi=roi, term=tg))
    dz = dec[dec.target == "bid"].pivot_table(index=["subject", "epoch"], columns="roi",
                                              values="z")
    for ep in ("gabor", "response"):
        g = dz.xs(ep, level="epoch")
        for a, b in [("M1lhand", "IPS12"), ("M1lhand", "IPS345"), ("M1lhand", "M1rhand"),
                     ("M1lhand", "preSMA")]:
            out.append(summ(g[a] - g[b], analysis="decoding_fisher_z_diff", epoch=ep,
                            roi=f"{a}-{b}", term="bid"))
    # extra noise: does an ROI's decoded bid track the bid error (bid − value), and does
    # M1 explain the bid beyond IPS? Per subject: partial r(decoded, bid | value) and
    # the unique R² of M1 in bid ~ value + IPS-decoded + M1-decoded.
    dt = pd.read_csv(TAB / "motor_tuning_decoding_trials.tsv", sep="\t", dtype={"subject": str})
    rows = []
    for (sub, ep), g in dt.groupby(["subject", "epoch"]):
        wide = g.pivot_table(index=["session", "run", "trial_nr"], columns="roi",
                             values="decoded_bid")
        base = g.drop_duplicates(["session", "run", "trial_nr"]).set_index(
            ["session", "run", "trial_nr"]).loc[wide.index]
        err = (base.bid - base.value).to_numpy()
        val = base.value.to_numpy()
        for roi in wide.columns:
            r, *_ = partial_r(wide[roi].to_numpy(), base.bid.to_numpy(), val)
            rows.append(dict(subject=sub, epoch=ep, roi=roi, partial_r_bid_given_value=r))
        for ips in ("IPS12", "IPS345"):
            X0 = np.column_stack([np.ones(len(val)), val, wide[ips]])
            X1 = np.column_stack([X0, wide["M1lhand"]])
            y = base.bid.to_numpy()
            r0 = 1 - np.var(y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]) / np.var(y)
            r1 = 1 - np.var(y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]) / np.var(y)
            rows.append(dict(subject=sub, epoch=ep, roi=f"M1lhand_beyond_{ips}",
                             unique_r2_x1e3=(r1 - r0) * 1e3))
        _ = err
    er = pd.DataFrame(rows)
    for (ep, roi), g in er.groupby(["epoch", "roi"]):
        col = "unique_r2_x1e3" if roi.startswith("M1lhand_beyond") else "partial_r_bid_given_value"
        out.append(summ(np.arctanh(g[col].clip(-.999, .999)) if col.startswith("partial") else g[col],
                        analysis=f"decoding_error_tracking:{col}", epoch=ep, roi=roi, term="bid"))

    res = pd.DataFrame(out)
    res.to_csv(TAB / "stats_group_effects.tsv", sep="\t", index=False, float_format="%.5g")

    # ── 5. individual differences (exploratory) ─────────────────────────────
    c = coefs[("main", "gabor")]
    cr = coefs[("main", "response")]
    ind = pd.DataFrame({
        "travel_M1lhand_gabor": c[c.roi == "M1lhand"].set_index("subject").travel,
        "travel_lat_gabor": (c[c.roi == "M1lhand"].set_index("subject").travel
                             - c[c.roi == "M1rhand"].set_index("subject").travel),
        "travel_M1lhand_response": cr[cr.roi == "M1lhand"].set_index("subject").travel,
    })
    d_m1 = dec[(dec.target == "bid") & (dec.roi == "M1lhand")]
    for ep in ("gabor", "response"):
        ind[f"decode_bid_M1lhand_{ep}"] = d_m1[d_m1.epoch == ep].set_index("subject").z
    ind.index = ind.index.to_series().replace({"pil01": "01", "pil02": "02"})
    p = pd.read_csv(REPO / "notes/data/cogmodel_shrink_mtail_subject_params.tsv", sep="\t")
    p["subject"] = p.subject.astype(str).str.zfill(2)
    p = p.set_index("subject")
    p["perceptual_sd"] = 40 / (2 * np.pi) / np.sqrt(p.kappa_r)
    fd = pd.read_csv(REPO / "notes/data/motion_summary.tsv", sep="\t", dtype={"subject": str})
    fd = fd.groupby("subject").mean_fd.mean()
    fd.index = fd.index.to_series().replace({"pil01": "01", "pil02": "02"})
    d = ind.join(p[["sigma_motor", "sigma_rep", "perceptual_sd", "total_noise", "motor_tail"]]
                 ).join(fd)
    rows = []
    for k in ind.columns:
        for q in ("sigma_motor", "sigma_rep", "perceptual_sd", "total_noise", "motor_tail"):
            e = d[[k, q, "mean_fd"]].dropna()
            rho, prho, lo, hi = spearman_ci(e[k], e[q])
            r, pr, rlo, rhi = partial_r(e[k].to_numpy(), e[q].to_numpy(), e.mean_fd.to_numpy())
            rows.append(dict(neural=k, behaviour=q, n=len(e), spearman_rho=rho, rho_ci_lo=lo,
                             rho_ci_hi=hi, rho_p=prho, partial_r_fd=r, r_ci_lo=rlo, r_ci_hi=rhi,
                             r_p=pr))
    corr = pd.DataFrame(rows)
    corr.to_csv(TAB / "stats_individual_differences.tsv", sep="\t", index=False,
                float_format="%.4g")
    d.to_csv(TAB / "stats_subject_measures.tsv", sep="\t", float_format="%.5g")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(res.round(3).to_string())
    print(corr.round(3).to_string())


if __name__ == "__main__":
    main()
