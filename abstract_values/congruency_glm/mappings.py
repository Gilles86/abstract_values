"""Orientation -> CHF value lookup for the two-mapping (cdf / inverse_cdf)
congruency design.

The 0deg/180deg endpoints in sns_multisubject.yml are examples-phase-only
and never appear as gabor stimuli in the main estimate task (see
experiment/README.md), so they're excluded here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_YML = REPO_ROOT / 'experiment' / 'settings' / 'sns_multisubject.yml'

MAPPING_NAMES = ('cdf', 'inverse_cdf')


def load_mappings(settings_yml=SETTINGS_YML):
    """Return {mapping_name: (orientations, values)} arrays (endpoints dropped)."""
    with open(settings_yml) as f:
        cfg = yaml.safe_load(f)['mappings']
    orientations = np.array(cfg['orientations'], dtype=float)
    keep = (orientations > 0) & (orientations < 180)
    return {
        name: (orientations[keep], np.array(cfg[name], dtype=float)[keep])
        for name in MAPPING_NAMES
    }


def value_lookup(mappings=None):
    """Return {mapping_name: {orientation: value}}."""
    mappings = mappings or load_mappings()
    return {name: dict(zip(np.round(o, 4), v)) for name, (o, v) in mappings.items()}


def value_under_mapping(orientation, mapping_name, lut=None):
    """Vectorized orientation(deg) -> CHF value under `mapping_name`.

    Raises if any orientation isn't one of the known task orientations --
    fail loudly rather than silently interpolate/guess.
    """
    lut = lut or value_lookup()
    table = lut[mapping_name]
    orientation = np.round(np.asarray(orientation, dtype=float), 4)
    missing = sorted(set(orientation.tolist()) - set(table))
    if missing:
        raise ValueError(
            f'Orientation(s) {missing} not in {mapping_name} mapping '
            f'(known: {sorted(table)})')
    return np.array([table[o] for o in orientation])


def zscore_params(mapping_name, mappings=None):
    """Fixed (mean, sd) for z-scoring a mapping's value regressor.

    Computed once from the mapping's own 23-orientation value set rather
    than per-run empirical stats -- deterministic, and every orientation
    appears exactly once per run so the difference is negligible anyway.
    """
    mappings = mappings or load_mappings()
    _, values = mappings[mapping_name]
    return float(values.mean()), float(values.std())
