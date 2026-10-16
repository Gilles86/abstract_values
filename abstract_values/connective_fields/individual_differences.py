"""Connectivity shift (V1 -> parietal coupling that follows the current mapping):
group test on all participants, and individual differences against behaviour.

Score per participant = observed - mean(label-shuffled) mapping score, target
<- V1 direction (test_coupling.py). Specifications, fixed before looking at
the n = 35 results (2026-10-16):

  PRIMARY    target Wang IPS0-5 (IPS0+IPS12+IPS345), V1 channels from the
             inverted von Mises encoding model (24 basis functions, kappa 16),
             the neighbouring trials' orientations (+-1) removed. Changed from
             +-2 on 2026-10-16: gabors are ~13.5 s apart, so only the adjacent
             trials' responses can overlap; +-2 only spends degrees of freedom.
  secondary  lag 0, +-2, IPS3-5 alone, and the earlier NPCr target

Individual differences (primary first):
  1. behavioural mapping fidelity: r(bid, value under THIS session's mapping)
     - r(bid, value the same orientation has under the OTHER mapping), averaged
     over sessions -- the behavioural twin of the coupling score
  2. value precision (-log sigma_rep) and perceptual precision from the
     two-stage model
  3. (added after the first look, at Gilles's suggestion, 2026-10-16) mapping
     error: RMS distance between the mean bid per orientation and the
     session's actual orientation -> CHF mapping, averaged over sessions.
     Correlates rho = -0.82 with value precision: it is mostly the same
     individual difference seen through the bias rather than the noise.
Correlations are Pearson across participants with a participant bootstrap.

    python -m abstract_values.connective_fields.individual_differences
    -> notes/data/cf_individual_differences.tsv, notes/figures/fig_cf_individual.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from abstract_values.utils.data import BIDS_FOLDER, Subject

REPO = Path(__file__).resolve().parents[2]
CF = Path(BIDS_FOLDER) / "derivatives" / "connective_fields"
PIL = {"pil01": "01", "pil02": "02"}
SPECS = {
    "IPS0–5, ±1 lag (primary)": "coupling_lags-1_iem-k24-kappa16_target-IPS0IPS12IPS345",
    "IPS0–5, lag 0": "coupling_iem-k24-kappa16_target-IPS0IPS12IPS345",
    "IPS0–5, ±2 lags": "coupling_lags-2_iem-k24-kappa16_target-IPS0IPS12IPS345",
    "IPS3–5, ±1 lag": "coupling_lags-1_iem-k24-kappa16_target-IPS345",
    "NPC R, lag 0": "coupling_iem-k24-kappa16",
}
PARAMS = Path("/data/ds-abstractvalue/derivatives/cogmodels_n35/"
              "efficient_coding_sequential_grid161_hn_total-share_erf_subject_params.tsv")


def scores(folder):
    rows = []
    for f in (CF / folder).glob("sub-*/sub-*_desc-scores.tsv"):
        d = pd.read_csv(f, sep="\t", dtype={"subject": str})
        d = d[d.direction == "npc_from_v1"]
        obs = d[d.kind == "observed"].score.mean()
        sh = d[d.kind == "label-shuffle"].score.mean()
        s = d.subject.iloc[0]
        rows.append(dict(subject=PIL.get(s, s), score=obs - sh))
    return pd.DataFrame(rows).set_index("subject").score


def mapping_fidelity_all():
    """Per participant: r(bid, value this mapping) - r(bid, value under the other
    mapping for the same orientation), averaged over the two sessions.
    One row per trial via the canonical behaviour recipe (frame-1 confirms out)."""
    from abstract_values.behavior.data import get_all_behavioral_data
    df = get_all_behavioral_data()
    df = df[df["event_type"] == "feedback"].copy()
    df["response"] = pd.to_numeric(df["response"], errors="coerce")
    df = df.reset_index()
    out, err = {}, {}
    for s, g in df.groupby("subject"):
        maps = {ses: h.groupby("orientation").value.first() for ses, h in g.groupby("session")}
        if len(maps) != 2:
            continue
        err[f"{int(s):02d}"] = float(np.mean([
            np.sqrt(np.mean((h.groupby("orientation").response.mean()
                             - maps[ses]).dropna() ** 2)) for ses, h in g.groupby("session")]))
        rs = []
        for ses, h in g.groupby("session"):
            other = maps[[x for x in maps if x != ses][0]]
            h = h[h.response.notna()]
            rs.append(np.corrcoef(h.response, h.value)[0, 1]
                      - np.corrcoef(h.response, h.orientation.map(other))[0, 1])
        out[f"{int(s):02d}"] = float(np.mean(rs))
    return pd.DataFrame({"fidelity": pd.Series(out), "mapping_error": pd.Series(err)})


def behaviour(subjects):
    par = pd.read_csv(PARAMS, sep="\t")
    par.index = par.subject.map(lambda s: f"{int(s):02d}")
    b = mapping_fidelity_all().reindex(subjects)
    b["value_precision"] = -np.log(par.sigma_rep)
    b["perceptual_precision"] = -np.log(par.total_noise * np.sqrt(par.perceptual_share))
    return b


def boot_r(x, y, n=5000):
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(x), (n, len(x)))
    r = [np.corrcoef(x[i], y[i])[0, 1] for i in idx]
    return np.quantile(r, [0.025, 0.975])


def main():
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 7, "axes.labelsize": 7.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    S = pd.DataFrame({k: scores(v) for k, v in SPECS.items()})
    print("Group test, observed - shuffled (one-sample t over participants):")
    for k in S:
        x = S[k].dropna()
        t, p = stats.ttest_1samp(x, 0)
        print(f"  {k:28s} {x.mean():+.4f} ± {x.sem():.4f}  t{len(x) - 1} = {t:+.2f}  p = {p:.4f}")
    b = behaviour(list(S.index))
    rows = []
    for k in S:
        for m in ("fidelity", "mapping_error", "value_precision", "perceptual_precision"):
            j = pd.concat([S[k], b[m]], axis=1).dropna()
            r, p = stats.pearsonr(j.iloc[:, 0], j.iloc[:, 1])
            lo, hi = boot_r(j.iloc[:, 0].to_numpy(), j.iloc[:, 1].to_numpy())
            rho, prho = stats.spearmanr(j.iloc[:, 0], j.iloc[:, 1])
            rows.append(dict(spec=k, measure=m, n=len(j), r=r, lo=lo, hi=hi, p=p,
                             spearman=rho, p_spearman=prho))
    res = pd.DataFrame(rows)
    res.to_csv(REPO / "notes/data/cf_individual_differences.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))

    prim = list(SPECS)[0]
    fig, axes = plt.subplots(1, 4, figsize=(7.25, 2.2), constrained_layout=True)
    for ax, m, lab in zip(axes, ("fidelity", "mapping_error", "value_precision",
                                 "perceptual_precision"),
                          ("Behavioural mapping fidelity\n(r this − r other mapping)",
                           "Mapping error (CHF)\nmean bid vs actual mapping",
                           "Value precision (−log σ_rep)",
                           "Perceptual precision")):
        j = pd.concat([S[prim], b[m]], axis=1).dropna()
        sns.regplot(x=j.iloc[:, 1], y=j.iloc[:, 0], ax=ax, color="#D9730D",
                    scatter_kws=dict(s=12, edgecolor="white", linewidths=0.4), line_kws=dict(lw=1.2))
        rr = res[(res.spec == prim) & (res.measure == m)].iloc[0]
        ax.set_title(f"r = {rr.r:+.2f} [{rr.lo:+.2f}, {rr.hi:+.2f}], ρ = {rr.spearman:+.2f}",
                     fontsize=7, loc="left")
        ax.set_xlabel(lab)
        ax.set_ylabel("Coupling follows mapping\n(observed − shuffled)" if m == "fidelity" else "")
        sns.despine(ax=ax, offset=4, trim=True)
    out = REPO / "notes/figures/fig_cf_individual.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
