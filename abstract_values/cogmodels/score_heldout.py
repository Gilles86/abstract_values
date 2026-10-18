"""Split-half validation of efficient-coding fits: fit on odd trials, score even trials.

Trials are split within each subject x mapping x orientation cell (1st, 3rd, ...
occurrence -> odd; 2nd, 4th, ... -> even), so both halves cover every cell. Each
fit (``--fit label=path:flags``, flags as in compare_ppc) was fitted on the odd
half; here it predicts the even half:

  elpd      held-out log predictive density per trial, log mean_draws p(y | theta),
            summed; differences against --baseline with the SE of the paired
            per-trial difference (and per subject)
  cell      RMSE between the predicted and observed even-half cell means of the
            bid error (predicted mean averaged over posterior draws), and the
            correlation of the cell-mean RESIDUALS (observed - baseline
            prediction) with each fit's predicted improvement over the baseline
            -- whether the odd-half offsets carry over to unseen trials

    python -m abstract_values.cogmodels.score_heldout \\
        --train-tsv notes/data/efficient_coding_paradigm_n35_split-odd.tsv \\
        --test-tsv  notes/data/efficient_coding_paradigm_n35_split-even.tsv \\
        --fit "No offsets=<..>_trace.nc:trunc,prior_weight,shrink,freelapse" ... \\
        --baseline "No offsets" --out-dir /scratch/gdehol/heldout
Run in the ``bauer`` env.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from abstract_values.cogmodels.compare_ppc import GRID, LAPSE, parse_fit
from abstract_values.cogmodels.fit_efficient_coding import make_model
from abstract_values.cogmodels.loo_efficient_coding import pointwise_loglik
from abstract_values.cogmodels.ppc_efficient_coding import (load_paradigm,
                                                           posterior_subject_draws)


def align_offsets(idata, model):
    """Reorder mapping_offset's mapping axis to the test model's condition order."""
    post = idata.posterior
    if "mapping_offset" not in post:
        return idata
    order = list(model.value_on_ori_grid)
    post["mapping_offset"] = post.mapping_offset.sel(offset_mapping=order)
    return idata


def predicted_means(model, par, idata, params, n_draws):
    subjects = list(par.index.get_level_values("subject").unique())
    acc = None
    for k, pars in enumerate(posterior_subject_draws(idata, subjects, params, n_draws, seed=2)):
        p = model.predict(par, pars).predicted_mean.to_numpy()
        acc = p if acc is None else acc + p
    return acc / (k + 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-tsv", required=True)
    p.add_argument("--test-tsv", required=True)
    p.add_argument("--fit", action="append", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--n-draws", type=int, default=400)
    p.add_argument("--n-pred", type=int, default=100)
    p.add_argument("--out-dir", default="notes/data/heldout")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    test = load_paradigm(a.test_tsv)
    del a.train_tsv                                  # documented provenance only

    trial, cellpred = {}, {}
    for spec in a.fit:
        label, path, kw, params = parse_fit(spec)
        print(f"== {label}: {path}", flush=True)
        kw = dict(kw)
        model = make_model(test, kw.pop("model_name"), GRID, lapse_rate=LAPSE,
                           perceptual_prior="long_term", **kw)
        model._setup_grids(test)
        idata = align_offsets(az.from_netcdf(path), model)
        ll = pointwise_loglik(model, test, idata, params, n_draws=a.n_draws)   # (D, T)
        trial[label] = logsumexp(ll, axis=0) - np.log(ll.shape[0])
        cellpred[label] = predicted_means(model, test, idata, params, a.n_pred)
        print(f"   held-out elpd {trial[label].sum():.1f}", flush=True)

    t = test.reset_index()
    T = pd.DataFrame(trial).assign(subject=t.subject.values)
    rows = []
    for lab in trial:
        dif = T[lab] - T[a.baseline]
        subj = T.groupby("subject")[lab].sum() - T.groupby("subject")[a.baseline].sum()
        rows.append(dict(fit=lab, elpd=T[lab].sum(), diff_vs_baseline=dif.sum(),
                         se_diff=dif.std() * np.sqrt(len(dif)),
                         subjects_better=int((subj > 0).sum()), n_subjects=len(subj)))
    C = t.assign(err=t.response - t.value)
    for lab, pm_ in cellpred.items():
        C[f"pred_{lab}"] = pm_ - t.value.values
    cells = C.groupby(["subject", "mapping", "orientation"]).mean(numeric_only=True)
    base = cells[f"pred_{a.baseline}"]
    for r in rows:
        pr = cells[f"pred_{r['fit']}"]
        r["cell_rmse"] = float(np.sqrt(((cells.err - pr) ** 2).mean()))
        r["r_residual_vs_improvement"] = (np.nan if r["fit"] == a.baseline else
                                          float(np.corrcoef(cells.err - base, pr - base)[0, 1]))
    res = pd.DataFrame(rows)
    res.to_csv(out / "heldout_scores.tsv", sep="\t", index=False)
    T.to_csv(out / "heldout_trial_elpd.tsv", sep="\t", index=False)
    cells.reset_index().to_csv(out / "heldout_cells.tsv", sep="\t", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
