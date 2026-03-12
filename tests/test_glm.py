"""
Tests for GLMsingle design matrix construction.
No BOLD data required — all tests use synthetic inputs.
"""

import numpy as np
import pandas as pd
import pytest

from abstract_values.glm.fit_glmsingle import build_design_matrix, TR


def make_events(n_trials=5, onset_start=30.0, isi=12.0):
    """Synthetic events DataFrame mimicking one fMRI run.

    Each trial has a gabor onset and a response_bar onset ~6 s later.
    """
    rows = []
    for t in range(1, n_trials + 1):
        gabor_onset = onset_start + (t - 1) * isi
        rows.append({'trial_nr': t, 'onset': gabor_onset,
                     'event_type': 'gabor'})
        rows.append({'trial_nr': t, 'onset': gabor_onset + 6.0,
                     'event_type': 'response_bar'})
    df = pd.DataFrame(rows).set_index(['trial_nr'])
    # add dummy columns present in real events
    df['orientation'] = 45.0
    df['value'] = 20.0
    df['response'] = 100.0
    df['duration'] = 0.0
    return df


# ── build_design_matrix ────────────────────────────────────────────────────────

class TestBuildDesignMatrix:

    def setup_method(self):
        self.n_trials = 5
        self.n_vols = 400
        self.events = make_events(n_trials=self.n_trials)

    def test_shape(self):
        dm, trial_types = build_design_matrix(self.events, self.n_vols)
        # 2 events per trial (gabor + response)
        assert dm.shape == (self.n_vols, self.n_trials * 2)

    def test_trial_types_length(self):
        _, trial_types = build_design_matrix(self.events, self.n_vols)
        assert len(trial_types) == self.n_trials * 2

    def test_trial_type_labels(self):
        _, trial_types = build_design_matrix(self.events, self.n_vols)
        gabors    = [t for t in trial_types if t.startswith('gabor')]
        responses = [t for t in trial_types if t.startswith('response')]
        assert len(gabors) == self.n_trials
        assert len(responses) == self.n_trials

    def test_binary(self):
        dm, _ = build_design_matrix(self.events, self.n_vols)
        assert set(np.unique(dm)).issubset({0.0, 1.0})

    def test_one_event_per_column(self):
        dm, _ = build_design_matrix(self.events, self.n_vols)
        assert np.all(dm.sum(axis=0) == 1), \
            "Every column must have exactly one 1"

    def test_onset_snapping(self):
        """Gabor at onset 30 s should land at TR index round(30 / TR)."""
        dm, trial_types = build_design_matrix(self.events, self.n_vols)
        col = trial_types.index('gabor_1')
        expected_tr = int(np.round(30.0 / TR))
        assert dm[expected_tr, col] == 1.0

    def test_onset_at_last_tr_clamped(self):
        """Onsets beyond the last TR must be clamped to n_vols - 1."""
        late_events = make_events(n_trials=1, onset_start=(self.n_vols + 10) * TR)
        dm, _ = build_design_matrix(late_events, self.n_vols)
        assert dm[-1, :].sum() >= 1  # clamped, not dropped

    def test_gabor_response_split_indices(self):
        """Indices used to split betas must cover all trials without overlap."""
        _, trial_types = build_design_matrix(self.events, self.n_vols)
        gabor_idx    = [i for i, t in enumerate(trial_types) if t.startswith('gabor')]
        response_idx = [i for i, t in enumerate(trial_types) if t.startswith('response')]
        assert len(gabor_idx) == self.n_trials
        assert len(response_idx) == self.n_trials
        assert set(gabor_idx).isdisjoint(set(response_idx))
        assert sorted(gabor_idx + response_idx) == list(range(self.n_trials * 2))

    def test_multi_run_trial_nr_independence(self):
        """trial_nr resets each run, so gabor_1 must appear once per run."""
        ev2 = make_events(n_trials=3, onset_start=30.0)
        _, tt1 = build_design_matrix(self.events, self.n_vols)
        _, tt2 = build_design_matrix(ev2, self.n_vols)
        # Both runs have gabor_1 — that is expected and correct
        assert 'gabor_1' in tt1
        assert 'gabor_1' in tt2
