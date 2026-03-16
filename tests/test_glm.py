"""
Tests for GLMsingle design matrix construction.
No BOLD data required — all tests use synthetic inputs.
"""

import numpy as np
import pandas as pd
import pytest

from abstract_values.glm.fit_glmsingle import (
    build_condition_index, build_design_matrix, TR
)


def make_events(n_trials=5, onset_start=30.0, isi=12.0):
    """Synthetic events DataFrame mimicking one fMRI run.

    Each trial has a gabor onset and a response_bar onset ~6 s later.
    Orientations and response values are unique per trial to keep the
    condition count predictable in tests.
    """
    orientations = [22.5 * t for t in range(1, n_trials + 1)]
    responses    = [10.0 * t for t in range(1, n_trials + 1)]
    rows = []
    for t in range(1, n_trials + 1):
        gabor_onset = onset_start + (t - 1) * isi
        rows.append({'trial_nr': t, 'onset': gabor_onset,
                     'event_type': 'gabor',
                     'orientation': orientations[t - 1],
                     'value': 20.0, 'response': responses[t - 1],
                     'duration': 0.0})
        rows.append({'trial_nr': t, 'onset': gabor_onset + 6.0,
                     'event_type': 'response_bar',
                     'orientation': orientations[t - 1],
                     'value': 20.0, 'response': responses[t - 1],
                     'duration': 0.0})
    return pd.DataFrame(rows).set_index(['trial_nr'])


def make_cond_idx(events):
    """Build a condition_to_idx from a single-run events DataFrame."""
    return build_condition_index([events.reset_index()])


# ── build_design_matrix ────────────────────────────────────────────────────────

class TestBuildDesignMatrix:

    def setup_method(self):
        self.n_trials = 5
        self.n_vols = 400
        self.events = make_events(n_trials=self.n_trials)
        self.cond_idx = make_cond_idx(self.events)

    def test_shape(self):
        dm, trial_order = build_design_matrix(self.events, self.n_vols, self.cond_idx)
        # unique orientations + unique response values (one each per trial here)
        assert dm.shape == (self.n_vols, self.n_trials * 2)

    def test_trial_types_length(self):
        _, trial_order = build_design_matrix(self.events, self.n_vols, self.cond_idx)
        assert len(trial_order) == self.n_trials * 2

    def test_trial_type_labels(self):
        _, trial_order = build_design_matrix(self.events, self.n_vols, self.cond_idx)
        orientations = [t for t in trial_order if t.startswith('orientation')]
        responses    = [t for t in trial_order if t.startswith('response')]
        assert len(orientations) == self.n_trials
        assert len(responses) == self.n_trials

    def test_binary(self):
        dm, _ = build_design_matrix(self.events, self.n_vols, self.cond_idx)
        assert set(np.unique(dm)).issubset({0.0, 1.0})

    def test_one_event_per_column(self):
        # With unique orientations/responses each column has exactly one 1
        dm, _ = build_design_matrix(self.events, self.n_vols, self.cond_idx)
        assert np.all(dm.sum(axis=0) == 1), \
            "Every column must have exactly one 1"

    def test_onset_snapping(self):
        """Gabor at onset 30 s should land at TR index round(30 / TR)."""
        dm, _ = build_design_matrix(self.events, self.n_vols, self.cond_idx)
        # First trial orientation is 22.5
        col = self.cond_idx['orientation_22.5']
        expected_tr = int(np.round(30.0 / TR))
        assert dm[expected_tr, col] == 1.0

    def test_onset_at_last_tr_clamped(self):
        """Onsets beyond the last TR must be clamped to n_vols - 1."""
        late_events = make_events(n_trials=1, onset_start=(self.n_vols + 10) * TR)
        cond_idx = make_cond_idx(late_events)
        dm, _ = build_design_matrix(late_events, self.n_vols, cond_idx)
        assert dm[-1, :].sum() >= 1  # clamped, not dropped

    def test_gabor_response_split_indices(self):
        """Indices used to split betas must cover all trials without overlap."""
        _, trial_order = build_design_matrix(self.events, self.n_vols, self.cond_idx)
        gabor_idx    = [i for i, t in enumerate(trial_order) if t.startswith('orientation')]
        response_idx = [i for i, t in enumerate(trial_order) if t.startswith('response')]
        assert len(gabor_idx) == self.n_trials
        assert len(response_idx) == self.n_trials
        assert set(gabor_idx).isdisjoint(set(response_idx))
        assert sorted(gabor_idx + response_idx) == list(range(self.n_trials * 2))

    def test_multi_run_trial_nr_independence(self):
        """Same condition label must appear in both runs (orientation repeats across runs)."""
        ev2 = make_events(n_trials=3, onset_start=30.0)
        shared_cond_idx = build_condition_index(
            [self.events.reset_index(), ev2.reset_index()]
        )
        _, tt1 = build_design_matrix(self.events, self.n_vols, shared_cond_idx)
        _, tt2 = build_design_matrix(ev2, self.n_vols, shared_cond_idx)
        # First trial orientation (22.5) appears in both runs
        assert 'orientation_22.5' in tt1
        assert 'orientation_22.5' in tt2

    def test_global_condition_index_consistent_across_runs(self):
        """Same condition label must map to the same column index in all runs."""
        ev2 = make_events(n_trials=3, onset_start=30.0)
        shared_cond_idx = build_condition_index(
            [self.events.reset_index(), ev2.reset_index()]
        )
        dm1, _ = build_design_matrix(self.events, self.n_vols, shared_cond_idx)
        dm2, _ = build_design_matrix(ev2, self.n_vols, shared_cond_idx)
        # Both design matrices must have the same number of columns
        assert dm1.shape[1] == dm2.shape[1] == len(shared_cond_idx)
