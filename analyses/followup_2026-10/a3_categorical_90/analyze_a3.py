"""A3: is the 90 deg category built in IPS or inherited from V1?  (local; reads the
per-subject RDMs from compute_rdms.py / compute_controls.py and the behaviour tables
from behavior_90.py)

Distances per subject x ROI (x mapping), from a 23 x 23 distance matrix d:
  d90    mean d(82.5, 90), d(90, 97.5)                     -- 90 vs its two neighbours
  dadj   mean d over the other 20 adjacent (7.5 deg) pairs  -- typical adjacent step
  d90all mean d(90, theta) over all 22 other orientations   -- 90 vs everything
  drest  mean d over the 231 pairs not involving 90
  dall   mean d over all 253 pairs                          -- the ROI's overall signal

Indices
  ratio  d90 / dadj: the prompt's local index. Reported at the group level only, as a
         ratio of group means with a participant-bootstrap CI, because per participant
         dadj is often <= 0 (crossnobis is unbiased; with weak signal it scatters
         around 0, so a per-participant ratio explodes).
  LCI    local categorical index per participant = (d90 - dadj) / dall
  GDI    global distinctiveness per participant  = (d90all - drest) / dall
         (planned after seeing that d(90, theta) is roughly flat in theta: 90 sits far
         from every other orientation, not only from its neighbours)
  Both are scale-free (units of the ROI's mean pairwise distance), so individual
  differences in overall SNR do not make V1 and IPS indices correlate trivially.
  NaN when dall <= 0. Pooled = d averaged over both sessions first. Split-half: the
  same on odd vs even runs (crossnobis within each half), Spearman-Brown corrected.

Value space: in this design the value neighbours of 22 CHF ARE the orientation
neighbours, so the question becomes how the distance at 90 scales with the value step
under each mapping. |dv| from 22 CHF to its neighbours is 3.5 CHF under Misaligned
(cdf: steepest point of that mapping) and 0.5 CHF under Aligned (inverse_cdf: flattest
point). A code whose distances follow value (d ~ dv^2) predicts the cdf / inverse_cdf
ratio of (d90/dadj) to equal the design's (dv90^2/dvadj^2) ratio; a mapping-invariant
(orientation) code predicts 1.

    python analyze_a3.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

OUT = Path("/data/ds-abstractvalue/derivatives/followup_2026-10/a3_categorical_90")
ROIS = ["V1", "V2V3", "IPS0", "IPS12", "IPS345", "IPS", "NPCr"]
RNG = np.random.default_rng(1)
NBOOT = 5000
IDX = ["LCI", "GDI"]


def subj_num(s):
    s = str(s)
    return int(s.replace("pil", "")) if s.startswith("pil") else int(s)


def load(pattern):
    fs = sorted(OUT.glob(pattern))
    if not fs:
        return None
    d = pd.concat(pd.read_csv(f, sep="\t", dtype={"subject": str}) for f in fs)
    d["subj"] = d.subject.map(subj_num)
    return d


def summarise(d, keys):
    step = (d.ori_j - d.ori_i).round(2)
    has90 = (d.ori_i == 90) | (d.ori_j == 90)
    d = d.assign(nb90=(step == 7.5) & has90, adj=(step == 7.5) & ~has90, has90=has90,
                 dv2=(d.val_j - d.val_i) ** 2)
    g = d.groupby(keys)
    out = pd.DataFrame({
        "d90": g.apply(lambda x: x.d[x.nb90].mean()),
        "dadj": g.apply(lambda x: x.d[x.adj].mean()),
        "d90all": g.apply(lambda x: x.d[x.has90].mean()),
        "drest": g.apply(lambda x: x.d[~x.has90].mean()),
        "dall": g.d.mean(),
        "dv2_90": g.apply(lambda x: x.dv2[x.nb90].mean()),
        "dv2_adj": g.apply(lambda x: x.dv2[x.adj].mean()),
    })
    pos = out.dall > 0
    out["LCI"] = np.where(pos, (out.d90 - out.dadj) / out.dall, np.nan)
    out["GDI"] = np.where(pos, (out.d90all - out.drest) / out.dall, np.nan)
    return out


def boot_ratio(num, den, n=NBOOT):
    num, den = np.asarray(num), np.asarray(den)
    idx = RNG.integers(0, len(num), (n, len(num)))
    r = num[idx].mean(1) / den[idx].mean(1)
    return num.mean() / den.mean(), *np.percentile(r, [2.5, 97.5])


def boot_mean(x, n=NBOOT):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    idx = RNG.integers(0, len(x), (n, len(x)))
    return x.mean(), *np.percentile(x[idx].mean(1), [2.5, 97.5])


def spearman_ci(x, y, n=2000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho, p = stats.spearmanr(x, y)
    idx = RNG.integers(0, len(x), (n, len(x)))
    bs = [stats.spearmanr(x[i], y[i])[0] for i in idx]
    return rho, *np.nanpercentile(bs, [2.5, 97.5]), p, len(x)


def skipped(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    r = pg.corr(x[ok], y[ok], method="skipped")
    return float(r["r"].iloc[0]), float(r["p-val"].iloc[0]), int(r["outliers"].iloc[0])


def reliability(a, b, n=2000):
    """Split-half Spearman, Spearman-Brown corrected, bootstrap CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    sb = lambda r: 2 * r / (1 + r)
    idx = RNG.integers(0, len(a), (n, len(a)))
    bs = [sb(stats.spearmanr(a[i], b[i])[0]) for i in idx]
    return sb(stats.spearmanr(a, b)[0]), *np.nanpercentile(bs, [2.5, 97.5]), len(a)


def main():
    d = load("sub-*/sub-*_desc-a3_rdm.tsv")
    dm = load("sub-*/sub-*_desc-a3demeaned_rdm.tsv")
    if dm is not None:
        d = pd.concat([d, dm])
    lines = []

    def P(*a):
        print(*a)
        lines.append(" ".join(str(x) for x in a))

    # ---- per subject x roi x kind x split (pooled over mappings) and per mapping
    pooled = d.groupby(["subj", "roi", "kind", "split", "ori_i", "ori_j"], as_index=False).agg(
        d=("d", "mean"), val_i=("val_i", "mean"), val_j=("val_j", "mean"))
    S = summarise(pooled, ["subj", "roi", "kind", "split"]).reset_index()
    M = summarise(d, ["subj", "roi", "kind", "split", "mapping"]).reset_index()
    S.to_csv(OUT / "index_subject_pooled.tsv", sep="\t", index=False)
    M.to_csv(OUT / "index_subject_mapping.tsv", sep="\t", index=False)
    P(f"n subjects with RDMs: {d.subj.nunique()}  ({sorted(d.subj.unique())})")

    # ---- adjacency profile (group, for the figure): d per adjacent pair / median
    st = (d.ori_j - d.ori_i).round(2)
    a = d[(st == 7.5) & (d.split == "all")].assign(mid=lambda x: (x.ori_i + x.ori_j) / 2)
    prof = a.groupby(["roi", "kind", "mid"]).d.mean().reset_index()
    prof["d_rel"] = prof.d / prof.groupby(["roi", "kind"]).d.transform("median")
    row = pd.concat([d, d.rename(columns={"ori_i": "ori_j", "ori_j": "ori_i"})])
    row = row[row.split == "all"].groupby(["roi", "kind", "subj", "ori_i"]).d.mean().reset_index()
    row = row.groupby(["roi", "kind", "ori_i"]).d.mean().reset_index()
    row["d_rel"] = row.d / row.groupby(["roi", "kind"]).d.transform("mean")
    prof.to_csv(OUT / "profile_adjacent.tsv", sep="\t", index=False)
    row.to_csv(OUT / "profile_rowmean.tsv", sep="\t", index=False)
    P("\n== Row-mean distance d(theta, .) / mean, at 90 (crossnobis): ",
      row[(row.ori_i == 90) & (row.kind == "crossnobis")][["roi", "d_rel"]].round(2).to_string(index=False))

    # ---- group ratios per ROI, kind, mapping
    rows = []
    for (roi, kind, m), g in M[M.split == "all"].groupby(["roi", "kind", "mapping"]):
        g = g.dropna(subset=["d90", "dadj"])
        r, lo, hi = boot_ratio(g.d90, g.dadj)
        rg, rglo, rghi = boot_ratio(g.d90all, g.drest)
        s = g.dv2_adj.mean() / g.dv2_90.mean()
        rows.append(dict(roi=roi, kind=kind, mapping=m, n=len(g), ratio=r, ratio_lo=lo, ratio_hi=hi,
                         global_ratio=rg, global_lo=rglo, global_hi=rghi,
                         **{f"{k}{suf}": v for k in IDX for suf, v in zip(["", "_lo", "_hi"], boot_mean(g[k]))},
                         value_FI_ratio=r * s, value_FI_lo=lo * s, value_FI_hi=hi * s,
                         value_pred_ratio=1 / s))
    for (roi, kind), g in S[S.split == "all"].groupby(["roi", "kind"]):
        g = g.dropna(subset=["d90", "dadj"])
        r, lo, hi = boot_ratio(g.d90, g.dadj)
        rg, rglo, rghi = boot_ratio(g.d90all, g.drest)
        rows.append(dict(roi=roi, kind=kind, mapping="pooled", n=len(g), ratio=r, ratio_lo=lo,
                         ratio_hi=hi, global_ratio=rg, global_lo=rglo, global_hi=rghi,
                         **{f"{k}{suf}": v for k in IDX for suf, v in zip(["", "_lo", "_hi"], boot_mean(g[k]))}))
    G = pd.DataFrame(rows)
    G.to_csv(OUT / "index_group.tsv", sep="\t", index=False)
    cols = ["roi", "kind", "mapping", "n", "ratio", "ratio_lo", "ratio_hi", "global_ratio", "global_lo",
            "global_hi", "LCI", "LCI_lo", "LCI_hi", "GDI", "GDI_lo", "GDI_hi", "value_FI_ratio",
            "value_FI_lo", "value_FI_hi"]
    P("\n== Group: ratio = d90/dadj, global_ratio = d90all/drest (ratio of means, 95% bootstrap CI)")
    for k in G.kind.unique():
        P(G[G.kind == k][cols].round(3).to_string(index=False))

    # ---- mapping comparison (value-space test)
    P("\n== Mapping test: (d90/dadj)_cdf / (d90/dadj)_inverse_cdf vs value-code prediction")
    mt = []
    for roi in ROIS:
        for kind in G.kind.unique():
            g = M[(M.roi == roi) & (M.kind == kind) & (M.split == "all")]
            w = g.pivot(index="subj", columns="mapping", values="d90").dropna()
            a_ = g.pivot(index="subj", columns="mapping", values="dadj").reindex(w.index)
            if len(w) < 5:
                continue
            rr = lambda i: (w.cdf.values[i].mean() / a_.cdf.values[i].mean()) / (
                w.inverse_cdf.values[i].mean() / a_.inverse_cdf.values[i].mean())
            bs = np.array([rr(i) for i in RNG.integers(0, len(w), (NBOOT, len(w)))])
            gi = g.pivot(index="subj", columns="mapping", values="GDI").dropna()
            gd = boot_mean(gi.cdf - gi.inverse_cdf)
            gc, gi_ = g[g.mapping == "cdf"], g[g.mapping == "inverse_cdf"]
            pred = (gc.dv2_90.mean() / gc.dv2_adj.mean()) / (gi_.dv2_90.mean() / gi_.dv2_adj.mean())
            mt.append(dict(roi=roi, kind=kind, n=len(w), ratio_cdf_over_inv=rr(np.arange(len(w))),
                           lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5),
                           value_code_pred=pred, GDI_cdf_minus_inv=gd[0], GDI_diff_lo=gd[1],
                           GDI_diff_hi=gd[2], GDI_diff_wilcoxon_p=stats.wilcoxon(gi.cdf - gi.inverse_cdf).pvalue))
    MT = pd.DataFrame(mt)
    MT.to_csv(OUT / "mapping_test.tsv", sep="\t", index=False)
    P(MT.round(3).to_string(index=False))

    # ---- univariate control
    rm = load("sub-*/sub-*_desc-a3roimean.tsv")
    if rm is not None:
        rm["is90"] = rm.orientation == 90
        u = rm.groupby(["subj", "roi", "is90"]).mean_beta.mean().unstack()
        u["diff"] = u[True] - u[False]
        uu = []
        for roi, g in u.reset_index().groupby("roi"):
            m_, lo_, hi_ = boot_mean(g["diff"])
            uu.append(dict(roi=roi, n=len(g), beta90_minus_other=m_, lo=lo_, hi=hi_,
                           wilcoxon_p=stats.wilcoxon(g["diff"]).pvalue))
        U = pd.DataFrame(uu)
        U.to_csv(OUT / "univariate_control.tsv", sep="\t", index=False)
        P("\n== Univariate control: ROI-mean beta at 90 minus other orientations")
        P(U.round(4).to_string(index=False))

    # ---- reliability of the per-subject indices
    P("\n== Split-half reliability (pooled over mappings), Spearman-Brown, 95% bootstrap CI")
    rel, rr_rows = {}, []
    for roi in ROIS:
        for k in IDX + ["d90", "dall"]:
            g = S[(S.roi == roi) & (S.kind == "crossnobis")].pivot(index="subj", columns="split", values=k)
            r, lo, hi, n = reliability(g.odd, g.even)
            rel[(roi, k)] = r
            rr_rows.append(dict(measure=k, roi=roi, n=n, rel=r, lo=lo, hi=hi))
    B = pd.read_csv(OUT / "behavior_subject.tsv", sep="\t")
    bp = B.groupby("subject")[["sd_dip_data", "sd_dip_model", "sd_dip_data_odd", "sd_dip_data_even"]].mean()
    r, lo, hi, n = reliability(bp.sd_dip_data_odd, bp.sd_dip_data_even)
    rel[("behav", "sd_dip_data")] = r
    rr_rows.append(dict(measure="sd_dip_data", roi="behaviour", n=n, rel=r, lo=lo, hi=hi))
    R = pd.DataFrame(rr_rows)
    R.to_csv(OUT / "reliability.tsv", sep="\t", index=False)
    P(R.pivot(index="roi", columns="measure", values="rel").round(2).to_string())
    P(R.round(2).to_string(index=False))

    # ---- IPS vs V1
    P("\n== Does IPS go beyond V1? (crossnobis, pooled over mappings)")
    reg_rows, wides = [], {}
    for k in IDX:
        wide = S[(S.kind == "crossnobis") & (S.split == "all")].pivot(index="subj", columns="roi", values=k)
        for roi in ["IPS", "IPS0", "IPS12", "IPS345", "NPCr", "V2V3"]:
            w = wide[["V1", roi]].dropna()
            diff = boot_mean(w[roi] - w.V1)
            rho = spearman_ci(w.V1.values, w[roi].values)
            sk = skipped(w.V1.values, w[roi].values)
            lr = pg.linear_regression(w.V1, w[roi])
            r_p = stats.pearsonr(w.V1, w[roi])[0]
            rv, ri = rel[("V1", k)], rel[(roi, k)]
            dis = r_p / np.sqrt(rv * ri) if rv > 0 and ri > 0 else np.nan
            reg_rows.append(dict(index=k, roi=roi, n=len(w), mean_diff=diff[0], diff_lo=diff[1],
                                 diff_hi=diff[2], wilcoxon_p=stats.wilcoxon(w[roi] - w.V1).pvalue,
                                 rho=rho[0], rho_lo=rho[1], rho_hi=rho[2], rho_p=rho[3],
                                 skipped_r=sk[0], skipped_p=sk[1], slope=lr.coef[1],
                                 slope_lo=lr["CI[2.5%]"][1], slope_hi=lr["CI[97.5%]"][1],
                                 intercept=lr.coef[0], int_lo=lr["CI[2.5%]"][0],
                                 int_hi=lr["CI[97.5%]"][0], pearson_r=r_p, rel_V1=rv, rel_roi=ri,
                                 ceiling_r=np.sqrt(max(rv, 0) * max(ri, 0)), disattenuated_r=dis))
            wide[f"{roi}_resid_on_V1"] = w[roi] - lr.coef[0] - lr.coef[1] * w.V1
        wides[k] = wide.add_prefix(f"{k}_")
    RG = pd.DataFrame(reg_rows)
    RG.to_csv(OUT / "ips_vs_v1.tsv", sep="\t", index=False)
    P(RG.round(3).to_string(index=False))

    # ---- brain <-> behaviour (exploratory)
    P("\n== Brain-behaviour (exploratory): index vs SD dip (more negative = sharper dip at 90)")
    W = pd.concat(wides.values(), axis=1).join(bp, how="inner")
    bb = []
    for k in IDX:
        for col in ROIS + ["IPS_resid_on_V1"]:
            for b in ["sd_dip_data", "sd_dip_model"]:
                rho = spearman_ci(W[f"{k}_{col}"].values, W[b].values)
                sk = skipped(W[f"{k}_{col}"].values, W[b].values)
                bb.append(dict(index=k, neural=col, behaviour=b, n=rho[4], rho=rho[0], lo=rho[1],
                               hi=rho[2], p=rho[3], skipped_r=sk[0], skipped_p=sk[1], n_outliers=sk[2]))
    BB = pd.DataFrame(bb)
    BB.to_csv(OUT / "brain_behaviour.tsv", sep="\t", index=False)
    W.to_csv(OUT / "subject_table.tsv", sep="\t")
    P(BB.round(3).to_string(index=False))

    # ---- model-implied vs crossnobis per subject
    P("\n== Model-implied vs crossnobis index across subjects (Spearman)")
    mv = []
    for k in IDX:
        for roi in ROIS:
            x = S[(S.roi == roi) & (S.split == "all")].pivot(index="subj", columns="kind", values=k)
            for mk in ["vonmises", "aprf", "crossnobis_demeaned"]:
                if mk in x:
                    rho = spearman_ci(x.crossnobis.values, x[mk].values)
                    mv.append(dict(index=k, roi=roi, other=mk, n=rho[4], rho=rho[0], lo=rho[1],
                                   hi=rho[2], p=rho[3]))
    MV = pd.DataFrame(mv)
    MV.to_csv(OUT / "model_vs_crossnobis.tsv", sep="\t", index=False)
    P(MV.round(3).to_string(index=False))
    (OUT / "analyze_a3_log.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
