import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from abstract_values.utils.data import (BIDS_FOLDER,  # main study root
                                        MIN_VALID_RT, flag_invalid_responses,
                                        warn_invalid_responses)

PILOT_BIDS_FOLDER = '/data/ds-abstract_values_pilot'


# ---------------------------------------------------------------------------
# Main study  (sub-01, ses-1 dirs, ses-1 in filenames)
# ---------------------------------------------------------------------------

def get_all_subject_ids(bids_folder=BIDS_FOLDER):
    behavior_root = Path(bids_folder) / 'sourcedata' / 'behavior'
    if not behavior_root.exists():
        return []
    subject_ids = []
    for d in sorted(behavior_root.glob('sub-*')):
        try:
            subject_ids.append(int(d.name.split('-')[1]))
        except (IndexError, ValueError):
            pass
    return subject_ids


def get_all_subjects(bids_folder=BIDS_FOLDER):
    subject_ids = get_all_subject_ids(bids_folder=bids_folder)
    return [Subject(subject_id, bids_folder=bids_folder) for subject_id in subject_ids]


def get_all_behavioral_data(bids_folder=BIDS_FOLDER, min_rt=MIN_VALID_RT):
    subjects = get_all_subjects(bids_folder=bids_folder)
    df = []
    for subject in subjects:
        d = subject.get_behavioral_data(min_rt=min_rt)
        if not d.empty:
            df.append(d)
    if not df:
        return pd.DataFrame()
    return pd.concat(df)


class Subject:
    """Behavioral data for a single study participant.

    File organisation::

        sourcedata/behavior/sub-{id:02d}/ses-{session}/
            sub-{id:02d}_ses-{session}_run-{run:02d}_task-{task}_events.tsv
    """

    def __init__(self, subject_id, bids_folder=BIDS_FOLDER):
        self.subject_id = int(subject_id)
        self.bids_folder = Path(bids_folder)

    @property
    def _behavior_root(self):
        return self.bids_folder / 'sourcedata' / 'behavior' / f'sub-{self.subject_id:02d}'

    def _session_dir(self, session):
        return self._behavior_root / f'ses-{session}'

    def _filename_prefix(self, session, run):
        return f'sub-{self.subject_id:02d}_ses-{session}_run-{run:02d}'

    def get_sessions(self):
        if not self._behavior_root.exists():
            return []
        sessions = []
        for d in sorted(self._behavior_root.glob('ses-*')):
            try:
                sessions.append(int(d.name.split('-')[1]))
            except (IndexError, ValueError):
                pass
        return sessions

    def get_runs(self, session=1):
        session_dir = self._session_dir(session)
        if not session_dir.exists():
            return []
        runs = set()
        for f in session_dir.glob(f'sub-{self.subject_id:02d}_ses-{session}_run-*_task-estimate.*_events.tsv'):
            for part in f.stem.split('_'):
                if part.startswith('run-'):
                    try:
                        runs.add(int(part.split('-')[1]))
                    except (IndexError, ValueError):
                        pass
        return sorted(runs)

    def get_mapping(self, session=1):
        if self.subject_id % 2 == 0:
            return "cdf" if session == 1 else "inverse_cdf"
        return "inverse_cdf" if session == 1 else "cdf"

    def get_behavioral_data(self, min_rt=MIN_VALID_RT):
        """One row per event, with `response`/`rt` on the trial's rows.

        Parameters
        ----------
        min_rt : float or None
            Trials whose response-bar RT falls below this (see
            :data:`~abstract_values.utils.data.MIN_VALID_RT`) get
            `response = rt = NaN` and `invalid_response = True`: the slider
            re-randomises every trial, so a frame-1 confirm records a uniform
            draw from the value range, not a bid.  `None` keeps them.
        """
        df = []
        for session in self.get_sessions():
            session_dir = self._session_dir(session)
            mapping = self.get_mapping(session=session)
            task = f'estimate.{mapping}'

            for run in self.get_runs(session):
                prefix = self._filename_prefix(session, run)
                behavioral_file = session_dir / f'{prefix}_task-{task}_events.tsv'
                if behavioral_file.exists():
                    d = pd.read_csv(behavioral_file, sep='\t')
                    d['subject'] = self.subject_id
                    d['session'] = session
                    d['mapping'] = mapping
                    d['run'] = run
                    responded = d[d['event_type'] == 'feedback'].dropna(subset=['response'])['trial_nr']
                    rt = d[d['event_type'] == 'response_bar'].set_index('trial_nr')['duration']
                    d['rt'] = d['trial_nr'].map(rt[rt.index.isin(responded)])

                    bad = flag_invalid_responses(d['rt'], min_rt)
                    warn_invalid_responses(bad.groupby(d['trial_nr']).first(), 'responses',
                                  f'sub-{self.subject_id} ses-{session} run-{run:02d}')
                    d['invalid_response'] = bad
                    d.loc[bad, ['response', 'rt']] = np.nan
                    # File mtime is the run's acquisition timestamp (preserved by rsync -a end to end)
                    d['run_datetime'] = datetime.fromtimestamp(behavioral_file.stat().st_mtime)
                    df.append(d)
                else:
                    print(f"Warning: Behavioral file {behavioral_file} does not exist")
        if not df:
            return pd.DataFrame()
        return pd.concat(df, ignore_index=True).set_index(['subject', 'session', 'mapping', 'run', 'trial_nr'])


# ---------------------------------------------------------------------------
# Pilot dataset  (sub-1, session-01 dirs, ses-01 in filenames)
# ---------------------------------------------------------------------------

def get_all_pilot_subject_ids(bids_folder=PILOT_BIDS_FOLDER):
    behavior_root = Path(bids_folder) / 'sourcedata' / 'behavior'
    if not behavior_root.exists():
        return []
    subject_ids = []
    for d in sorted(behavior_root.glob('sub-*')):
        try:
            subject_ids.append(int(d.name.split('-')[1]))
        except (IndexError, ValueError):
            pass
    return subject_ids


def get_all_pilot_subjects(bids_folder=PILOT_BIDS_FOLDER):
    subject_ids = get_all_pilot_subject_ids(bids_folder=bids_folder)
    return [PilotSubject(subject_id, bids_folder=bids_folder) for subject_id in subject_ids]


def get_all_pilot_behavioral_data(bids_folder=PILOT_BIDS_FOLDER, min_rt=MIN_VALID_RT):
    subjects = get_all_pilot_subjects(bids_folder=bids_folder)
    df = []
    for subject in subjects:
        d = subject.get_behavioral_data(min_rt=min_rt)
        if not d.empty:
            df.append(d)
    if not df:
        return pd.DataFrame()
    return pd.concat(df)


class PilotSubject:
    """Behavioral data for a single pilot subject.

    File organisation::

        sourcedata/behavior/sub-{id}/session-{session:02d}/
            sub-{id}_ses-{session:02d}_run-{run:02d}_task-{task}_events.tsv
    """

    def __init__(self, subject_id, bids_folder=PILOT_BIDS_FOLDER):
        self.subject_id = int(subject_id)
        self.bids_folder = Path(bids_folder)

    @property
    def _behavior_root(self):
        return self.bids_folder / 'sourcedata' / 'behavior' / f'sub-{self.subject_id}'

    def _session_dir(self, session):
        return self._behavior_root / f'session-{session:02d}'

    def _filename_prefix(self, session, run):
        return f'sub-{self.subject_id}_ses-{session:02d}_run-{run:02d}'

    def get_sessions(self):
        if not self._behavior_root.exists():
            return []
        sessions = []
        for d in sorted(self._behavior_root.glob('session-*')):
            try:
                sessions.append(int(d.name.split('-')[1]))
            except (IndexError, ValueError):
                pass
        return sessions

    def get_runs(self, session=1):
        session_dir = self._session_dir(session)
        if not session_dir.exists():
            return []
        runs = set()
        for f in session_dir.glob(f'sub-{self.subject_id}_ses-{session:02d}_run-*_task-estimate.*_events.tsv'):
            for part in f.stem.split('_'):
                if part.startswith('run-'):
                    try:
                        runs.add(int(part.split('-')[1]))
                    except (IndexError, ValueError):
                        pass
        return sorted(runs)

    def get_mapping(self, session=1):
        if self.subject_id % 2 == 0:
            return "cdf" if session == 1 else "inverse_cdf"
        return "inverse_cdf" if session == 1 else "cdf"

    def get_behavioral_data(self, min_rt=MIN_VALID_RT):
        """One row per event, with `response`/`rt` on the trial's rows.

        Parameters
        ----------
        min_rt : float or None
            Trials whose response-bar RT falls below this (see
            :data:`~abstract_values.utils.data.MIN_VALID_RT`) get
            `response = rt = NaN` and `invalid_response = True`: the slider
            re-randomises every trial, so a frame-1 confirm records a uniform
            draw from the value range, not a bid.  `None` keeps them.
        """
        df = []
        for session in self.get_sessions():
            session_dir = self._session_dir(session)
            mapping = self.get_mapping(session=session)
            task = f'estimate.{mapping}'

            for run in self.get_runs(session):
                prefix = self._filename_prefix(session, run)
                behavioral_file = session_dir / f'{prefix}_task-{task}_events.tsv'
                if behavioral_file.exists():
                    d = pd.read_csv(behavioral_file, sep='\t')
                    d['subject'] = self.subject_id
                    d['session'] = session
                    d['mapping'] = mapping
                    d['run'] = run
                    responded = d[d['event_type'] == 'feedback'].dropna(subset=['response'])['trial_nr']
                    rt = d[d['event_type'] == 'response_bar'].set_index('trial_nr')['duration']
                    d['rt'] = d['trial_nr'].map(rt[rt.index.isin(responded)])

                    bad = flag_invalid_responses(d['rt'], min_rt)
                    warn_invalid_responses(bad.groupby(d['trial_nr']).first(), 'responses',
                                  f'sub-{self.subject_id} ses-{session} run-{run:02d}')
                    d['invalid_response'] = bad
                    d.loc[bad, ['response', 'rt']] = np.nan
                    df.append(d)
                else:
                    print(f"Warning: Behavioral file {behavioral_file} does not exist")
        if not df:
            return pd.DataFrame()
        return pd.concat(df, ignore_index=True).set_index(['subject', 'session', 'mapping', 'run', 'trial_nr'])
