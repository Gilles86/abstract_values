"""Gated permutation FDR for the model-neutral voxel selection (pure numpy).

Copied unchanged from retinonumeral (retinonumeral/analyses/permfdr.py);
design and calibration: retinonumeral/notes/permutation_fdr_selection.md.
Used here by ``permfdr_select_models.py``.

Copies c = 0 (real labels) and 1..K (shuffles; one relabelling per shuffle,
seen by all six models). For every copy the per-voxel statistic is

    T_c(v) = max_m -log10 p_m,c(v),
    p_m,c(v) = (1 + #{pool_m >= R2_m,c(v)}) / (1 + |pool_m|),

with model m's null R² pooled over voxels and over the copies NOT excluded
(the real copy is never pooled; a null copy never sits in its own pool; in the
leave-one-shuffle-out calibration the data copy is removed from every pool).
T is capped at the floor of the smallest pool compared, so a copy with a
larger pool cannot reach a larger T by construction.

Selection within an ROI:

* SAM / Storey-Tibshirani step-up: FDR̂(t) = pi0 * mean_j N_j(t) / R(t);
  t* = smallest observed T_0 with FDR̂ <= alpha.
* Shuffle-level gate: D_c = number of SAM discoveries of copy c as data;
  p_gate = (1 + #{null copies k: D_k >= D_0}) / (1 + K); keep the selection only
  if p_gate <= alpha. Under the complete null the real copy is exchangeable
  with the shuffles, so P(any discovery) <= alpha whatever the voxel
  dependence (needs K >= 19).
"""
from __future__ import annotations

import numpy as np

ALPHA = 0.05


# ------------------------------------------------------------------ basics
def ge_count(sorted_pool, x):
    """#{pool >= x} for each x (pool sorted ascending, finite)."""
    return len(sorted_pool) - np.searchsorted(sorted_pool, x, side="left")


def neglog_p(count_ge, n_pool):
    return -np.log10((1.0 + count_ge) / (1.0 + n_pool))


def tcap(n_pool_min):
    """Largest attainable T for the smallest pool among the compared copies."""
    return float(-np.log10(1.0 / (1.0 + n_pool_min)))


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


def pi0_storey(T0, Tnull):
    """pi0 = min(1, #{T0 < median pooled null T} / (0.5 n))."""
    med = np.median(np.asarray(Tnull).ravel())
    return float(min(1.0, np.mean(T0 < med) / 0.5))


def sam_select(T0, Tnull, alpha=ALPHA, pi0=1.0, null_scale=1.0):
    """SAM step-up within one ROI.

    T0 : (n,) data statistic. Tnull : (J, n_b) null copies (n_b may differ from
    n; ``null_scale`` = n / n_b rescales their counts to the ROI size).
    Returns dict(keep (n,) bool, t_star, fdr_hat, n_sel).
    """
    T0 = np.nan_to_num(np.asarray(T0, float), nan=0.0, neginf=0.0)
    Tn = np.nan_to_num(np.asarray(Tnull, float), nan=0.0, neginf=0.0)
    J = Tn.shape[0]
    pooled = np.sort(Tn.ravel())
    desc = np.sort(T0)[::-1]
    # R(t) at t = desc[r] counts ties: #{T0 >= desc[r]}
    R = len(desc) - np.searchsorted(np.sort(T0), desc, side="left")
    Nmean = ge_count(pooled, desc) / J * null_scale
    fdr = pi0 * Nmean / np.maximum(R, 1)
    ok = np.flatnonzero(fdr <= alpha)
    if len(ok) == 0:
        return dict(keep=np.zeros(len(T0), bool), t_star=np.inf, fdr_hat=np.nan, n_sel=0)
    r = ok.max()
    t = desc[r]
    keep = T0 >= t
    return dict(keep=keep, t_star=float(t), fdr_hat=float(fdr[r]), n_sel=int(keep.sum()))


def gate_p(D0, Dnull):
    Dnull = np.asarray(Dnull)
    return float((1 + np.sum(Dnull >= D0)) / (1 + len(Dnull)))


# ----------------------------------------------------- matched (per fold)
def fold_statistics(real, null):
    """Matched per-fold banks.

    real : (M, n) training R² of the ROI voxels; null : (M, K, nb) bank.
    Returns (T_real (n,), arg_real (n,), T_null (K, nb), arg_null (K, nb)):
    real against all K shuffles; shuffle k against the other K - 1
    (leave-own-out); both capped at the (K - 1)-row pool floor.
    """
    real = np.where(np.isfinite(real), real, -np.inf)
    null = np.where(np.isfinite(null), null, -np.inf)
    M, K, nb = null.shape
    Tr = np.full((M, real.shape[1]), 0.0)
    Tn = np.full((M, K, nb), 0.0)
    nmin = None
    for m in range(M):
        rows = [np.sort(null[m, k][np.isfinite(null[m, k])]) for k in range(K)]
        pool = np.sort(np.concatenate(rows))
        N = len(pool)
        Tr[m] = neglog_p(ge_count(pool, real[m]), N)
        for k in range(K):
            c = ge_count(pool, null[m, k]) - ge_count(rows[k], null[m, k])
            Tn[m, k] = neglog_p(c, N - len(rows[k]))
        nk = N - max(len(r) for r in rows)
        nmin = nk if nmin is None else min(nmin, nk)
    cap = tcap(nmin)
    Tr, Tn = np.minimum(Tr, cap), np.minimum(Tn, cap)
    return Tr.max(0), Tr.argmax(0), Tn.max(0), Tn.argmax(0)


def matched_roi_permfdr(folds, alpha=ALPHA, pi0_adjust=False):
    """Gated permutation FDR for one participant x ROI of the matched pipeline.

    folds : list of (real (M, n), null (M, K, nb), bank_ids (nb,)) per LORO fold;
        ``bank_ids`` index the ROI voxels (columns of ``real``).
    Null counts pool ALL folds' null copies (scaled n / nb); the gate compares
    the real fold's discoveries ON THE BANK VOXELS with every null copy's
    discoveries (each null copy as data against all other copies).

    Returns (per-fold list of dict(keep, n_sel, t_star, p_gate, D_bank, arg)),
    calibration dict for the null copies (ungated / gated any-discovery rates).
    """
    stats = [fold_statistics(r, nl) for r, nl, _ in folds]
    Tnull_all = np.concatenate([s[2] for s in stats], 0)          # (F*K, nb)
    n_copies = Tnull_all.shape[0]
    # each null copy as data against all other copies (same bank voxels)
    D_null = np.empty(n_copies, int)
    for c in range(n_copies):
        rest = np.delete(Tnull_all, c, 0)
        D_null[c] = sam_select(Tnull_all[c], rest, alpha)["n_sel"]
    p_gate_null = np.array([gate_p(D_null[c], np.delete(D_null, c)) for c in range(n_copies)])
    calib = dict(n_copies=int(n_copies),
                 any_disc_ungated=float(np.mean(D_null > 0)),
                 any_disc_gated=float(np.mean((D_null > 0) & (p_gate_null <= alpha))),
                 mean_disc_frac_ungated=float(np.mean(D_null) / Tnull_all.shape[1]))
    out = []
    for (real, null, ids), (Tr, ar, _, _) in zip(folds, stats):
        nb = null.shape[2]
        n = real.shape[1]
        pi0 = pi0_storey(Tr, Tnull_all) if pi0_adjust else 1.0
        sel = sam_select(Tr, Tnull_all, alpha, pi0=pi0, null_scale=n / nb)
        Db = sam_select(Tr[ids], Tnull_all, alpha, pi0=pi0)["n_sel"]
        pg = gate_p(Db, D_null)
        keep = sel["keep"] & (pg <= alpha)
        out.append(dict(keep=keep, n_sel=int(keep.sum()), n_sel_ungated=sel["n_sel"],
                        t_star=sel["t_star"], p_gate=pg, D_bank=int(Db), arg=ar, T=Tr, pi0=pi0))
    return out, calib


__all__ = ["ge_count", "neglog_p", "tcap", "bh", "pi0_storey", "sam_select", "gate_p",
           "fold_statistics", "matched_roi_permfdr", "ALPHA"]
