#!/usr/bin/env python3
"""A6 (exploratory, post hoc): does left-M1 hand-area activity track MOTOR noise?

A1 showed that the left-M1 hand area encodes the expected slider travel
(b^2 + (42 - b)^2) / 84 = E|b - s|, s ~ U[0, 42] the unlogged marker start. Across
people the test is not specific (sigma_motor and sigma_rep correlate .66), so every
test here takes its leverage WITHIN person.

Bid error is taken relative to the cogmodel's predictive mean for the trial's cell
(subject x mapping x orientation; model_cell_predictions.py, the
categorical/free-prior/shrink/motor-tail fit). Its motor stage is a stationary
kernel: the same for every trial of a subject, independent of the bid.

T1  signed, trial level. Cell-demeaned error ~ cell-demeaned decoded bid of each ROI
    (+ RT). Then jointly with IPS12, IPS345, V1 decoded bids, V1-decoded orientation
    error pushed through the mapping (a perceptual-error proxy) and the NPCr decoded
    value. Unique R^2 of M1 vs of IPS. NB an M1 that executes the bid carries every
    error source, so signed tracking is NOT motor-specific by itself; only tracking
    beyond the upstream ROIs is suggestive.
T2  unsigned, trial level (Fitts-like). Within cell, |error| ~ ROI beta, controlling
    travel(b), b, RT. If actual movement distance (unknown start) drives both M1
    amplitude and execution error, the M1 beta left over after travel(b) predicts
    |error|. Robustness: a 5-df spline of b in place of travel(b) + b; adding the
    decoded value uncertainty (NPCr posterior SD).
T3  cell level (Fitts-like). Observed mean |bid - predicted mean| minus the model's
    predicted mean absolute deviation, regressed on the cell's expected travel, per
    subject. Then: does a person's distance-variance slope track their M1 travel
    sensitivity (A1)?
T4  overshoot. Signed cell-demeaned error ~ ROI beta x expected direction of
    movement, dir = (2c - 42)/42 = P(right) - P(left) for the uniform start. Weak by
    construction: direction is never observed. And travel(b) - travel(c) = dir*e +
    e^2/42, so the naive interaction is what the A1 travel code alone predicts; only
    the version with beta residualised on travel(b) is a test of extra M1 activity.

Per-subject OLS coefficients, then the mean over subjects with a t-based 95% CI.
No resampling.

    ~/mambaforge/bin/python analyze_m1_motor_noise.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
A1 = Path("/data/ds-abstractvalue/derivatives/followup_2026-10/a1_motor_stage/tables")
OUT = Path("/data/ds-abstractvalue/derivatives/followup_2026-10/a6_m1_motor_noise")
SLIDER = 42.0
ROIS = ["M1lhand", "M1rhand", "M1l", "M1r", "preSMA", "V1", "IPS12", "IPS345"]
KEY = ["subject", "session", "run", "trial_nr"]


def travel(b):
    return (b ** 2 + (SLIDER - b) ** 2) / (2 * SLIDER)


def sub_label(s):
    s = int(s)
    return {1: "pil01", 2: "pil02"}.get(s, f"{s:02d}")


def summ(v, **keys):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    m, sd = v.mean(), v.std(ddof=1)
    se = sd / np.sqrt(n)
    h = stats.t.ppf(.975, n - 1) * se
    t = m / se
    return dict(**keys, mean=m, ci_lo=m - h, ci_hi=m + h, t=t,
                p=2 * stats.t.sf(abs(t), n - 1), dz=m / sd, n=n)


def spearman_ci(x, y):
    rho, p = stats.spearmanr(x, y)
    z, se = np.arctanh(rho), 1.06 / np.sqrt(len(x) - 3)
    return rho, p, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def partial_spearman(x, y, Z):
    """Spearman partial correlation: Pearson on ranks after regressing out ranked Z."""
    rk = lambda a: stats.rankdata(a)
    Z = np.column_stack([np.ones(len(x))] + [rk(z) for z in np.atleast_2d(Z)])
    rx = rk(x) - Z @ np.linalg.lstsq(Z, rk(x), rcond=None)[0]
    ry = rk(y) - Z @ np.linalg.lstsq(Z, rk(y), rcond=None)[0]
    r = np.corrcoef(rx, ry)[0, 1]
    n, k = len(x), Z.shape[1] - 1
    t = r * np.sqrt((n - 2 - k) / (1 - r ** 2))
    se = 1.06 / np.sqrt(n - 3 - k)
    return r, 2 * stats.t.sf(abs(t), n - 2 - k), np.tanh(np.arctanh(r) - 1.96 * se), \
        np.tanh(np.arctanh(r) + 1.96 * se)


def demean(df, cols, by):
    return df[cols] - df.groupby(by)[cols].transform("mean")


def ols(y, X):
    """Coefficients (no intercept: everything is demeaned) and R^2."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b, 1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2)


def cv_r2(y, X, folds):
    """Leave-one-fold-out R^2 of a no-intercept OLS (y and X already demeaned)."""
    pred = np.empty_like(y, dtype=float)
    for f in np.unique(folds):
        te = folds == f
        b = np.linalg.lstsq(X[~te], y[~te], rcond=None)[0]
        pred[te] = X[te] @ b
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def zs(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / a.std()


def spline_basis(b, df=5):
    """Truncated-power cubic spline of b with knots at quantiles (df columns)."""
    knots = np.quantile(b, np.linspace(0, 1, df - 1)[1:-1])
    cols = [b, b ** 2, b ** 3] + [np.clip(b - k, 0, None) ** 3 for k in knots]
    return np.column_stack(cols)[:, :df + 1]


def mapping_value_fn(trials):
    """orientation -> value for each mapping, linearly interpolated, clamped at edges."""
    tab = trials.groupby(["mapping", "orientation"]).value.first()
    fns = {}
    for mp, s in tab.groupby(level=0):
        o, v = s.index.get_level_values(1).to_numpy(), s.to_numpy()
        fns[mp] = (lambda th, o=o, v=v: np.interp(th, o, v))
    return fns


def load():
    t = pd.read_csv(A1 / "motor_tuning_trials.tsv", sep="\t", dtype={"subject": str})
    t = t[t.roi.isin(ROIS)]
    base = t.drop_duplicates(KEY)[KEY + ["mapping", "value", "orientation", "bid", "rt"]]
    beta = t.pivot_table(index=KEY, columns=["epoch", "roi"], values="beta")
    beta.columns = [f"beta_{e}_{r}" for e, r in beta.columns]
    d = pd.read_csv(A1 / "motor_tuning_decoding_trials.tsv", sep="\t", dtype={"subject": str})
    d = d[d.roi.isin(ROIS)]
    dec = d.pivot_table(index=KEY, columns=["epoch", "roi"], values="decoded_bid")
    dec.columns = [f"dec_{e}_{r}" for e, r in dec.columns]
    df = base.set_index(KEY).join(beta).join(dec).reset_index()
    df = df[df.bid.notna() & df.rt.notna()].copy()

    tt = pd.read_csv(REPO / "notes/data/trial_table.tsv", sep="\t")
    tt["subject"] = tt.subject.map(sub_label)
    tt = tt[KEY + ["v1_ori_decode_error", "npcr_val_post_mean", "npcr_val_post_sd"]]
    df = df.merge(tt, on=KEY, how="left")

    mp = pd.read_csv(OUT / "model_cell_predictions.tsv", sep="\t")
    mp["subject"] = mp.subject.map(sub_label)
    df = df.merge(mp.drop(columns="value"), on=["subject", "mapping", "orientation"],
                  how="left")
    assert df.pred_mean_full.notna().all(), "cells without model predictions"

    fn = mapping_value_fn(df)
    th = (df.orientation + df.v1_ori_decode_error) % 180
    df["perc_pred"] = [fn[m](x) for m, x in zip(df.mapping, th)] - df.value
    df["err"] = df.bid - df.pred_mean_full
    df["abserr"] = df.err.abs()
    df["travel_b"] = travel(df.bid)
    df["travel_c"] = travel(df.pred_mean_full)
    df["dir"] = (2 * df.pred_mean_full - SLIDER) / SLIDER
    df["cell"] = df.subject + "_" + df.mapping + "_" + df.orientation.astype(str)
    # run-demean ROI betas (slow drifts / run offsets), as in A1
    bcols = [c for c in df.columns if c.startswith("beta_")]
    df[bcols] = demean(df, bcols, ["subject", "session", "run"])
    return df


# ── T1 ─────────────────────────────────────────────────────────────────────────
def t1(df):
    rows, joint = [], []
    for (sub, ep), _ in [((s, e), None) for s in df.subject.unique()
                         for e in ("gabor", "response")]:
        g = df[df.subject == sub]
        need = [f"dec_{ep}_{r}" for r in ROIS] + ["err", "rt", "perc_pred",
                                                    "npcr_val_post_mean"]
        g = g.dropna(subset=need)
        c = demean(g, need, "cell")
        y = c.err.to_numpy()
        for r in ROIS:
            X = np.column_stack([zs(c[f"dec_{ep}_{r}"]), zs(c.rt)])
            b, _ = ols(y, X)
            rows.append(dict(subject=sub, epoch=ep, roi=r, coef=b[0]))
        up = ["dec_%s_IPS12" % ep, "dec_%s_IPS345" % ep, "dec_%s_V1" % ep, "perc_pred",
              "npcr_val_post_mean", "rt"]
        for m1 in ("M1lhand", "M1rhand", "preSMA"):
            cols = [f"dec_{ep}_{m1}"] + up
            X = np.column_stack([zs(c[k]) for k in cols])
            b, r2 = ols(y, X)
            # in-sample unique R^2 is positive by construction (~1/n per regressor), so
            # the unique contributions are leave-one-run-out cross-validated
            folds = (g.session.astype(str) + "_" + g.run.astype(str)).to_numpy()
            r2cv = cv_r2(y, X, folds)
            r2cv_no_m1 = cv_r2(y, X[:, 1:], folds)
            r2cv_no_ips = cv_r2(y, np.delete(X, [1, 2], 1), folds)
            rec = dict(subject=sub, epoch=ep, motor_roi=m1, r2=r2, r2cv_x1e3=r2cv * 1e3,
                       uniq_motor_x1e3=(r2cv - r2cv_no_m1) * 1e3,
                       uniq_ips_x1e3=(r2cv - r2cv_no_ips) * 1e3)
            rec.update({f"coef_{k.replace(f'dec_{ep}_', '')}": v for k, v in zip(cols, b)})
            joint.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(joint)


def interaction(err_cd, x_cd, basis, d, extra):
    """Coefficient of (x_r * dir) on the signed within-cell error, x_r = x residualised
    on ``basis`` (functions of the bid). Every input is already cell-demeaned and dir is
    constant within a cell, so x_r * dir is too."""
    x = np.asarray(x_cd, float)
    xr = x - basis @ np.linalg.lstsq(basis, x, rcond=None)[0]
    cols = [zs(xr), zs(xr * d)] + ([] if extra is None else [extra])
    return ols(np.asarray(err_cd, float), np.column_stack(cols))[0][1]


# ── T2 / T4 ────────────────────────────────────────────────────────────────────
def t2_t4(df):
    rows = []
    for sub in df.subject.unique():
        for ep in ("gabor", "response"):
            for r in ROIS:
                bc = f"beta_{ep}_{r}"
                g = df[df.subject == sub].dropna(subset=[bc, "npcr_val_post_sd"]).copy()
                g["beta_x_dir"] = g[bc] * g.dir
                g["rt_x_dir"] = g.rt * g.dir
                S = spline_basis(g.bid.to_numpy())
                for j in range(S.shape[1]):
                    g[f"sp{j}"] = S[:, j]
                cols = [bc, "abserr", "err", "travel_b", "bid", "rt", "npcr_val_post_sd",
                        "beta_x_dir", "rt_x_dir"] + [f"sp{j}" for j in range(S.shape[1])]
                c = demean(g, cols, "cell")
                sp = [zs(c[f"sp{j}"]) for j in range(S.shape[1])]
                ctrl = [zs(c.travel_b), zs(c.bid), zs(c.rt)]
                xb = zs(c[bc])
                rec = dict(subject=sub, epoch=ep, roi=r, n=len(g))
                rec["t2_main"] = ols(c.abserr, np.column_stack([xb] + ctrl))[0][0]
                rec["t2_spline"] = ols(c.abserr, np.column_stack([xb] + sp + [zs(c.rt)]))[0][0]
                rec["t2_uncert"] = ols(c.abserr, np.column_stack(
                    [xb] + ctrl + [zs(c.npcr_val_post_sd)]))[0][0]
                rec["t2_beta_on_travel"] = ols(zs(c[bc]), np.column_stack(ctrl))[0][0]
                # behavioural Fitts proxy: RT ~ |err| within cell, same controls
                rec["t2_rt_behav"] = ols(c.abserr, np.column_stack(
                    [zs(c.rt), zs(c.travel_b), zs(c.bid)]))[0][0]
                # T4: overshoot in the expected direction of movement. Within a cell
                # b = c + e, so any control that is a function of b (b, travel(b))
                # contains e itself and must stay out. And travel(b) - travel(c) =
                # dir*e + e^2/42 exactly, so an M1 that only encodes travel(b) already
                # produces beta x dir ~ e: the naive coefficient is NOT motor-specific.
                # 't4_naive': no b-controls. 't4_travel_resid': beta first residualised
                # on travel(b) (within cell), which removes the dir*e + e^2/42 term with
                # its fitted weight; what is left of beta x dir ~ e can come only from
                # M1 activity beyond the expected-travel code.
                rec["t4_naive"] = ols(c.err, np.column_stack(
                    [xb, zs(c.beta_x_dir), zs(c.rt)]))[0][1]
                d_ = g.dir.to_numpy()
                tb = zs(c.travel_b)[:, None]
                # any smooth function of the bid, with its own shape per session
                ses = g.session.to_numpy()
                spS = np.column_stack([np.column_stack(sp) * (ses == s_)[:, None]
                                       for s_ in np.unique(ses)])
                rtc = zs(c.rt)
                rec["t4_travel_resid"] = interaction(c.err, c[bc], tb, d_, rtc)
                rec["t4_spline_resid"] = interaction(c.err, c[bc], spS, d_, rtc)
                rec["t4_rt_x_dir"] = ols(c.err, np.column_stack(
                    [zs(c.rt), zs(c.rt_x_dir)]))[0][1]
                rec["t4_rt_x_dir_spline_resid"] = interaction(c.err, c.rt, spS, d_, None)
                rows.append(rec)
    return pd.DataFrame(rows)


# ── T3 ─────────────────────────────────────────────────────────────────────────
def t3(df):
    cell = df.groupby(["subject", "cell"]).agg(
        obs_absdev=("abserr", "mean"), n=("abserr", "size"),
        pred_absdev_full=("pred_absdev_full", "first"),
        pred_absdev_nomotor=("pred_absdev_nomotor", "first"),
        pred_sd_full=("pred_sd_full", "first"), travel_c=("travel_c", "first"),
        value=("value", "first"), mapping=("mapping", "first"),
        orientation=("orientation", "first")).reset_index()
    # mapping slope |dv/dtheta| at the cell (CHF / deg), from neighbouring orientations
    sl = {}
    for mp, s in df.groupby("mapping"):
        tab = s.groupby("orientation").value.first()
        sl[mp] = pd.Series(np.abs(np.gradient(tab.to_numpy(), tab.index.to_numpy())),
                           index=tab.index)
    cell["log_slope"] = np.log([sl[m][o] for m, o in zip(cell.mapping, cell.orientation)])
    rows = []
    for sub, g in cell.groupby("subject"):
        w = np.sqrt(g.n.to_numpy())
        for null in ("full", "nomotor"):
            y = (g.obs_absdev - g[f"pred_absdev_{null}"]).to_numpy()
            X = np.column_stack([np.ones(len(g)), zs(g.travel_c)])
            b = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
            X2 = np.column_stack([X, zs(g.log_slope)])
            b2 = np.linalg.lstsq(X2 * w[:, None], y * w, rcond=None)[0]
            # the model's own (stationary-motor) prediction vs travel, for reference
            bp = np.linalg.lstsq(X * w[:, None], g[f"pred_absdev_{null}"].to_numpy() * w,
                                 rcond=None)[0]
            bo = np.linalg.lstsq(X * w[:, None], g.obs_absdev.to_numpy() * w, rcond=None)[0]
            rows.append(dict(subject=sub, null=null, misfit_mean=b[0], slope_travel=b[1],
                             slope_travel_ctrl_slope=b2[1], pred_slope_travel=bp[1],
                             obs_slope_travel=bo[1]))
    return pd.DataFrame(rows), cell


def main():
    df = load()
    print(f"{df.subject.nunique()} subjects, {len(df)} trials")
    out = []
    t1s, t1j = t1(df)
    for (ep, roi), g in t1s.groupby(["epoch", "roi"]):
        out.append(summ(g.coef, test="T1_signed_single", epoch=ep, roi=roi,
                        term="decoded_bid (CHF/SD)"))
    w = t1s.pivot_table(index=["subject", "epoch"], columns="roi", values="coef")
    for ep in ("gabor", "response"):
        x = w.xs(ep, level="epoch")
        for ctl in ("M1rhand", "preSMA", "V1", "IPS345", "IPS12"):
            out.append(summ(x.M1lhand - x[ctl], test="T1_signed_single", epoch=ep,
                            roi=f"M1lhand-{ctl}", term="decoded_bid (CHF/SD)"))
    for (ep, m1), g in t1j.groupby(["epoch", "motor_roi"]):
        for k in [c for c in g.columns if c.startswith(("coef_", "uniq_"))]:
            out.append(summ(g[k], test="T1_signed_joint", epoch=ep, roi=m1, term=k))
        out.append(summ(g.uniq_motor_x1e3 - g.uniq_ips_x1e3, test="T1_signed_joint",
                        epoch=ep, roi=m1, term="uniq_motor-uniq_ips_x1e3"))

    t24 = t2_t4(df)
    terms = ["t2_main", "t2_spline", "t2_uncert", "t2_beta_on_travel", "t4_naive",
             "t4_travel_resid", "t4_spline_resid"]
    for (ep, roi), g in t24.groupby(["epoch", "roi"]):
        for k in terms:
            out.append(summ(g[k], test=k.split("_")[0].upper(), epoch=ep, roi=roi, term=k))
    g = t24.drop_duplicates(["subject", "epoch"])
    for ep in ("gabor", "response"):
        gg = g[g.epoch == ep]
        out.append(summ(gg.t2_rt_behav, test="T2", epoch=ep, roi="behaviour", term="rt"))
        out.append(summ(gg.t4_rt_x_dir, test="T4", epoch=ep, roi="behaviour",
                        term="rt_x_dir"))
        out.append(summ(gg.t4_rt_x_dir_spline_resid, test="T4", epoch=ep, roi="behaviour",
                        term="rt_x_dir_spline_resid"))
    w2 = t24.pivot_table(index=["subject", "epoch"], columns="roi",
                         values=["t2_main", "t4_travel_resid", "t4_spline_resid"])
    for k in ("t2_main", "t4_travel_resid", "t4_spline_resid"):
        for ep in ("gabor", "response"):
            x = w2[k].xs(ep, level="epoch")
            for ctl in ("M1rhand", "preSMA", "V1", "IPS345"):
                out.append(summ(x.M1lhand - x[ctl], test=k.split("_")[0].upper(), epoch=ep,
                                roi=f"M1lhand-{ctl}", term=k))

    t3s, cell = t3(df)
    for null, g in t3s.groupby("null"):
        for k in ("misfit_mean", "slope_travel", "slope_travel_ctrl_slope",
                  "pred_slope_travel", "obs_slope_travel"):
            out.append(summ(g[k], test="T3", epoch="-", roi=f"null={null}", term=k))

    # individual differences: Fitts slope vs M1 travel sensitivity (A1), exploratory
    sm = pd.read_csv(A1 / "stats_subject_measures.tsv", sep="\t", dtype={"subject": str})
    sm["subject"] = sm.subject.replace({"01": "pil01", "02": "pil02"})
    ind = t3s[t3s.null == "full"].set_index("subject")[["slope_travel"]].join(
        sm.set_index("subject"))
    t2m = t24[(t24.roi == "M1lhand") & (t24.epoch == "gabor")].set_index("subject")
    ind = ind.join(t2m[["t2_main"]])
    ind_rows = []
    for neural in ("travel_M1lhand_gabor", "travel_lat_gabor"):
        for beh in ("slope_travel", "t2_main"):
            e = ind[[neural, beh, "mean_fd", "sigma_rep"]].dropna()
            rho, p, lo, hi = spearman_ci(e[neural], e[beh])
            pr, pp, plo, phi = partial_spearman(e[neural].to_numpy(), e[beh].to_numpy(),
                                                [e.mean_fd.to_numpy(), e.sigma_rep.to_numpy()])
            ind_rows.append(dict(neural=neural, behaviour=beh, n=len(e), rho=rho,
                                 rho_ci_lo=lo, rho_ci_hi=hi, rho_p=p,
                                 partial_rho_fd_sigmarep=pr, prho_ci_lo=plo, prho_ci_hi=phi,
                                 prho_p=pp))
    e = ind[["slope_travel", "sigma_motor", "sigma_rep", "mean_fd"]].dropna()
    for beh in ("sigma_motor", "sigma_rep"):
        rho, p, lo, hi = spearman_ci(e.slope_travel, e[beh])
        ind_rows.append(dict(neural="slope_travel", behaviour=beh, n=len(e), rho=rho,
                             rho_ci_lo=lo, rho_ci_hi=hi, rho_p=p))

    OUT.mkdir(parents=True, exist_ok=True)
    tabs = OUT / "tables"
    tabs.mkdir(exist_ok=True)
    res = pd.DataFrame(out)
    res.to_csv(tabs / "a6_group_effects.tsv", sep="\t", index=False, float_format="%.5g")
    pd.DataFrame(ind_rows).to_csv(tabs / "a6_individual_differences.tsv", sep="\t",
                                  index=False, float_format="%.4g")
    t1s.to_csv(tabs / "a6_t1_subject_single.tsv", sep="\t", index=False, float_format="%.5g")
    t1j.to_csv(tabs / "a6_t1_subject_joint.tsv", sep="\t", index=False, float_format="%.5g")
    t24.to_csv(tabs / "a6_t2_t4_subject.tsv", sep="\t", index=False, float_format="%.5g")
    t3s.to_csv(tabs / "a6_t3_subject.tsv", sep="\t", index=False, float_format="%.5g")
    cell.to_csv(tabs / "a6_t3_cells.tsv", sep="\t", index=False, float_format="%.5g")
    df.to_csv(tabs / "a6_trials.tsv", sep="\t", index=False, float_format="%.5g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 1000)
    print(res.round(4).to_string())
    print(pd.DataFrame(ind_rows).round(3).to_string())


if __name__ == "__main__":
    main()
