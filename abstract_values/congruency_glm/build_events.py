"""Build nilearn-format events for the congruency GLM.

Per trial: unmodulated `gabor` (stimulus onset) and `response_bar` regressors
are always present. Value modulators are parametric modulators on the
`gabor` onset only (value is a stimulus property, known at stimulus onset --
not on `response_bar`).

`value_congruent` is the CHF value of the presented orientation under the
session's TRUE/active mapping; `value_incongruent` is the value the same
orientation would have under the other (counterfactual) mapping. Both are
z-scored using fixed per-mapping constants (see mappings.zscore_params).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from abstract_values.congruency_glm.mappings import (
    MAPPING_NAMES, value_lookup, value_under_mapping, zscore_params)

MODEL_MODULATORS = {
    'congruent': ('value_congruent',),
    'incongruent': ('value_incongruent',),
    'both': ('value_congruent', 'value_incongruent'),
}


def build_run_events(sub, session, run, model, lut=None):
    """One run's nilearn events DataFrame: onset, duration, trial_type, modulation.

    `model` in {'congruent', 'incongruent', 'both'} selects which value
    modulator(s) to include; `gabor`/`response_bar` are always included.
    """
    if model not in MODEL_MODULATORS:
        raise ValueError(f'model must be one of {list(MODEL_MODULATORS)}, got {model!r}')
    lut = lut or value_lookup()
    true_mapping = sub.get_mapping(session)
    other_mapping = [m for m in MAPPING_NAMES if m != true_mapping][0]

    ev = sub.get_events(session, [run]).reset_index()
    gabor = ev[ev['event_type'] == 'gabor'].copy()
    resp = ev[ev['event_type'] == 'response_bar'].copy()
    if gabor.empty or resp.empty:
        raise ValueError(f'sub-{sub.subject_id} ses-{session} run-{run}: '
                         f'missing gabor or response_bar events')

    rows = [
        pd.DataFrame({'onset': gabor['onset'], 'duration': gabor['duration'],
                      'trial_type': 'gabor', 'modulation': 1.0}),
        pd.DataFrame({'onset': resp['onset'], 'duration': resp['duration'],
                      'trial_type': 'response_bar', 'modulation': 1.0}),
    ]

    value_true = value_under_mapping(gabor['orientation'], true_mapping, lut)
    # Sanity check: recomputed congruent value must exactly match
    # Subject.get_events()'s own 'value' column (already the true-mapping
    # value) -- catches any mapping/orientation-lookup mismatch immediately.
    if not np.allclose(value_true, gabor['value'].to_numpy(dtype=float)):
        raise AssertionError(
            f'sub-{sub.subject_id} ses-{session} run-{run}: recomputed '
            f'value_under_mapping(orientation, {true_mapping!r}) does not match '
            f"Subject.get_events()'s 'value' column -- mapping/orientation mismatch.")

    modulators = MODEL_MODULATORS[model]
    if 'value_congruent' in modulators:
        mean_t, sd_t = zscore_params(true_mapping)
        rows.append(pd.DataFrame({
            'onset': gabor['onset'], 'duration': gabor['duration'],
            'trial_type': 'value_congruent', 'modulation': (value_true - mean_t) / sd_t}))
    if 'value_incongruent' in modulators:
        value_other = value_under_mapping(gabor['orientation'], other_mapping, lut)
        mean_o, sd_o = zscore_params(other_mapping)
        rows.append(pd.DataFrame({
            'onset': gabor['onset'], 'duration': gabor['duration'],
            'trial_type': 'value_incongruent', 'modulation': (value_other - mean_o) / sd_o}))

    events = pd.concat(rows, ignore_index=True).sort_values('onset').reset_index(drop=True)
    return events


def build_subject_events(sub, model, sessions=None):
    """List of per-run events DataFrames across all sessions/runs, and the
    matching (session, run) keys in the same order -- callers zip this
    against `sub.get_preprocessed_bold` per (session, run)."""
    sessions = sessions or sub.get_sessions()
    lut = value_lookup()
    events_list, keys = [], []
    for session in sessions:
        for run in sub.get_runs(session):
            events_list.append(build_run_events(sub, session, run, model, lut))
            keys.append((session, run))
    return events_list, keys
