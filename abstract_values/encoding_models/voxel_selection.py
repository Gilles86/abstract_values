#!/usr/bin/env python3
"""Shared voxel-selection logic for the leave-one-run-out decoders.

Both ``decode_gabor.py`` and ``decode_value.py`` pick, inside every outer CV
fold, which voxels to decode from.  Three criteria are supported:

``n_voxels > 0``
    top-N by *training* R².  Handled at the call site — it can never be empty.

``n_voxels == 0``
    every voxel whose *nested* CV R² clears a floor.  The floor is 0 by
    default, or the per-voxel null model's nested CV R² when ``cv_r2_null``
    is given (see ``project_cvr2_null_baseline``).  **This is the one that
    can legitimately select nothing.**

``fdr_alpha`` / ``p_signal_thr``
    threshold from the cached whole-brain logit-Gaussian R² mixture, with a
    top-N fallback whenever the mixture is degenerate or leaves < 10 voxels.
    Also never empty.

Before this module the empty case was handled five different ways across the
two scripts — ``SystemExit``, a silent top-1 fallback, and twice not at all.
``select_voxels`` is now the single implementation, and an empty selection is
reported as a *status* the caller records rather than as a crash or a silent
substitution.  See ``notes/decode_zero_voxel_fix_proposal.md``.
"""

import numpy as np
import pandas as pd

#: Selection produced voxels by its primary criterion.
STATUS_OK = 'ok'
#: Whole-brain mixture was degenerate; fell back to top-N by cv-R².
STATUS_MIXTURE_DEGENERATE = 'mixture_degenerate'
#: Mixture threshold left < 10 voxels; fell back to top-N by cv-R².
STATUS_FDR_FALLBACK = 'fdr_fallback'
#: No voxel cleared the nested-CV floor. The fold is not decodable.
STATUS_EMPTY = 'empty'

#: Statuses whose folds carry real decoded posteriors.
USABLE_STATUSES = (STATUS_OK, STATUS_MIXTURE_DEGENERATE, STATUS_FDR_FALLBACK)


def select_voxels(cv_r2, *, mixture_model, subject, bids_folder, smoothed,
                  fdr_alpha=None, p_signal_thr=None,
                  fdr_fallback_n_voxels=100, cv_r2_null=None,
                  cv_r2_rival=None):
    """Pick decoding voxels from nested-CV R².

    Parameters
    ----------
    cv_r2 : pd.Series
        Nested cross-validated R² per voxel, indexed by masker column.
    mixture_model : str
        Model label for the cached whole-brain R² mixture, e.g. ``'vonmises'``,
        ``'vonmises-linear'``, ``'aprf'``, ``'aprf-weighted'``, ``'aprf-linear'``.
        Only consulted for the FDR / P(signal) criteria.
    subject, bids_folder, smoothed
        Passed through to the mixture lookup.
    fdr_alpha, p_signal_thr : float or None
        Mutually exclusive mixture criteria.  When both are ``None`` the
        nested-CV floor is used.
    fdr_fallback_n_voxels : int
        Top-N by cv-R² to fall back on when the mixture cannot be used.
    cv_r2_null : pd.Series or None
        Per-voxel null-model nested CV R².  When given, the floor becomes
        ``cv_r2 > cv_r2_null`` instead of ``cv_r2 > 0``.
    cv_r2_rival : pd.Series or None
        Per-voxel nested CV R² of a *competing* model, on the same inner
        folds.  When given, a voxel must also beat it — the floor becomes the
        elementwise max of the null and the rival.  This is what turns "voxels
        that carry signal" into "voxels this model wins", the selection the
        value-vs-orientation contrast needs: within a session value is a
        deterministic function of orientation, so a voxel tuned to either fits
        both, and only "which model wins across the mapping flip" separates
        them.  Computed per fold from the training runs only, so it does not
        leak the held-out run.

    Returns
    -------
    (sel, status, message)
        ``sel`` is a possibly-empty ``pd.Index`` of voxel columns, ``status``
        one of the ``STATUS_*`` constants, ``message`` a line to print.
    """
    if fdr_alpha is not None and p_signal_thr is not None:
        raise ValueError('fdr_alpha and p_signal_thr are mutually exclusive.')

    if fdr_alpha is None and p_signal_thr is None:
        return _select_by_cv_floor(cv_r2, cv_r2_null, cv_r2_rival)

    if cv_r2_rival is not None:
        raise ValueError(
            'cv_r2_rival needs the nested-CV floor criterion (n_voxels=0 and '
            'no fdr_alpha / p_signal_thr): the mixture criteria threshold one '
            "model's R² distribution and have no notion of a rival.")

    return _select_by_mixture(
        cv_r2, mixture_model=mixture_model, subject=subject,
        bids_folder=bids_folder, smoothed=smoothed, fdr_alpha=fdr_alpha,
        p_signal_thr=p_signal_thr,
        fdr_fallback_n_voxels=fdr_fallback_n_voxels)


def _select_by_cv_floor(cv_r2, cv_r2_null, cv_r2_rival=None):
    """``cv_r2 > 0`` (or ``> cv_r2_null``) — the only criterion that can be empty."""
    if cv_r2_rival is not None:
        return _select_by_winner(cv_r2, cv_r2_null, cv_r2_rival)

    if cv_r2_null is None:
        sel = cv_r2[cv_r2 > 0.0].index
        if len(sel) == 0:
            return sel, STATUS_EMPTY, (
                f'    0/{len(cv_r2)} voxels with nested CV R² > 0 '
                f'(max={float(cv_r2.max()):.4f}) — fold is not decodable')
        return sel, STATUS_OK, (
            f'    {len(sel)} voxels selected  '
            f'(nested CV R² > 0, mean={float(cv_r2.loc[sel].mean()):.3f})')

    sel = cv_r2[cv_r2 > cv_r2_null].index
    if len(sel) == 0:
        return sel, STATUS_EMPTY, (
            f'    0/{len(cv_r2)} voxels with nested CV R² > nested CV R²_null '
            f'(max Δ={float((cv_r2 - cv_r2_null).max()):.4f}) — '
            f'fold is not decodable')
    delta = float((cv_r2.loc[sel] - cv_r2_null.loc[sel]).mean())
    return sel, STATUS_OK, (
        f'    {len(sel)} voxels selected  (nested CV R² > nested CV R²_null, '
        f'mean Δ={delta:.3f})')


def _select_by_mixture(cv_r2, *, mixture_model, subject, bids_folder, smoothed,
                       fdr_alpha, p_signal_thr, fdr_fallback_n_voxels):
    """Whole-brain R²-mixture threshold, with a top-N fallback. Never empty."""
    if fdr_alpha is not None:
        from abstract_values.encoding_models.compute_r2_mixture \
            import get_brain_fdr_threshold
        res = get_brain_fdr_threshold(
            subject, model=mixture_model, bids_folder=bids_folder,
            alpha=fdr_alpha, smoothed=smoothed)
        crit_label = f'FDR≤{fdr_alpha:.2f}'
    else:
        from abstract_values.encoding_models.compute_r2_mixture \
            import get_brain_p_signal_threshold
        res = get_brain_p_signal_threshold(
            subject, model=mixture_model, bids_folder=bids_folder,
            p=p_signal_thr, smoothed=smoothed)
        crit_label = f'P(signal)≥{p_signal_thr:.2f}'

    if res is None:
        raise RuntimeError(
            f'Whole-brain {mixture_model} mixture missing and auto-fit failed '
            f'for sub-{subject}. Run `python -m '
            f'abstract_values.encoding_models.compute_r2_mixture '
            f'--models {mixture_model}` first.')

    thr = res['threshold']
    top_n = cv_r2.sort_values(ascending=False).index[:fdr_fallback_n_voxels]

    if res['degenerate'] or not np.isfinite(thr):
        return top_n, STATUS_MIXTURE_DEGENERATE, (
            f'    {len(top_n)} voxels selected  '
            f'(mixture degenerate ⇒ fallback to top-{fdr_fallback_n_voxels} by cv-R²)')

    sel = cv_r2[cv_r2 > thr].index
    if len(sel) < 10:
        return top_n, STATUS_FDR_FALLBACK, (
            f'    {len(top_n)} voxels selected  '
            f'(only {len(sel)} passed {crit_label} → R² > {thr:.3f}; '
            f'fallback to top-{fdr_fallback_n_voxels} by cv-R²)')

    return sel, STATUS_OK, (
        f'    {len(sel)} voxels selected  '
        f'(whole-brain mixture {crit_label} → R² > {thr:.3f})')


def empty_posterior(stimulus_range, index_names):
    """A header-only posterior frame, for when no fold was decodable.

    Keeps the on-disk contract identical (same columns, same index names) so
    downstream loaders read an empty result rather than hitting a missing file.
    """
    return pd.DataFrame(
        np.empty((0, len(stimulus_range))),
        columns=pd.Index(stimulus_range),
        index=pd.MultiIndex.from_arrays(
            [[] for _ in index_names], names=list(index_names)))


def concat_posteriors(all_pdfs, stimulus_range, index_names):
    """``pd.concat`` that tolerates every fold having been undecodable."""
    if not all_pdfs:
        return empty_posterior(stimulus_range, index_names)
    return pd.concat(all_pdfs).sort_index()


def warn_if_degraded(fold_meta, subject, mask_desc):
    """Print a loud summary when folds were dropped. Never raises."""
    n_empty = sum(1 for f in fold_meta if f.get('status') == STATUS_EMPTY)
    if n_empty == 0:
        return
    n = len(fold_meta)
    if n_empty == n:
        print(f'\n  *** WARNING: all {n} folds undecodable for sub-{subject} '
              f'(mask {mask_desc}) — no voxel cleared the nested-CV floor in '
              f'any fold. Writing an empty result; downstream code must drop '
              f'this cell (status column in the _meta sidecar). ***')
    else:
        print(f'\n  *** WARNING: {n_empty}/{n} folds undecodable for '
              f'sub-{subject} (mask {mask_desc}) — those folds contribute no '
              f'trials. See the status column in the _meta sidecar. ***')


def usable_folds(meta):
    """Filter a ``_meta.tsv`` frame to folds that carry real posteriors.

    Sidecars written before the ``status`` column existed are treated as all
    usable — at that time an empty selection could not produce output at all.
    """
    if 'status' not in meta.columns:
        return meta
    return meta[meta['status'].isin(USABLE_STATUSES)]


def _select_by_winner(cv_r2, cv_r2_null, cv_r2_rival):
    """``cv_r2`` beats both the null and a rival model, per voxel."""
    rival = cv_r2_rival.reindex(cv_r2.index)
    floor = rival if cv_r2_null is None else \
        pd.concat([rival, cv_r2_null.reindex(cv_r2.index)], axis=1).max(axis=1)
    sel = cv_r2[cv_r2 > floor].index
    beats_null = (len(cv_r2[cv_r2 > cv_r2_null.reindex(cv_r2.index)])
                  if cv_r2_null is not None else len(cv_r2[cv_r2 > 0.0]))
    if len(sel) == 0:
        return sel, STATUS_EMPTY, (
            f'    0/{len(cv_r2)} voxels win against the rival model '
            f'({beats_null} beat the null) — fold is not decodable')
    margin = float((cv_r2.loc[sel] - rival.loc[sel]).mean())
    return sel, STATUS_OK, (
        f'    {len(sel)} voxels selected  (beat the null AND the rival model; '
        f'{beats_null} beat the null alone, mean margin over rival '
        f'={margin:.3f})')
