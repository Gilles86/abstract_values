import pandas as pd
from pathlib import Path


PILOT_BIDS_FOLDER = '/data/ds-abstract_values_pilot'
FMRI_BIDS_FOLDER = '/data/ds-abstractvalue'


# ---------------------------------------------------------------------------
# Pilot dataset  (sub-1, session-01 dirs, ses-01 in filenames)
# ---------------------------------------------------------------------------

def get_all_subject_ids(bids_folder=PILOT_BIDS_FOLDER):
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


def get_all_subjects(bids_folder=PILOT_BIDS_FOLDER):
    subject_ids = get_all_subject_ids(bids_folder=bids_folder)
    return [Subject(subject_id, bids_folder=bids_folder) for subject_id in subject_ids]


def get_all_behavioral_data(bids_folder=PILOT_BIDS_FOLDER):
    subjects = get_all_subjects(bids_folder=bids_folder)
    df = []
    for subject in subjects:
        d = subject.get_behavioral_data()
        if not d.empty:
            df.append(d)
    return pd.concat(df)


class Subject():
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
            mapping = "cdf" if session == 1 else "inverse_cdf"
        else:
            mapping = "inverse_cdf" if session == 1 else "cdf"
        return mapping

    def get_behavioral_data(self):
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
                    df.append(d)
                else:
                    print(f"Warning: Behavioral file {behavioral_file} does not exist")
        if not df:
            return pd.DataFrame()
        return pd.concat(df, ignore_index=True).set_index(['subject', 'session', 'mapping', 'run', 'trial_nr'])


# ---------------------------------------------------------------------------
# fMRI dataset  (sub-01, ses-1 dirs, ses-1 in filenames)
# ---------------------------------------------------------------------------

def get_all_fmri_subject_ids(bids_folder=FMRI_BIDS_FOLDER):
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


def get_all_fmri_subjects(bids_folder=FMRI_BIDS_FOLDER):
    subject_ids = get_all_fmri_subject_ids(bids_folder=bids_folder)
    return [FMRISubject(subject_id, bids_folder=bids_folder) for subject_id in subject_ids]


def get_all_fmri_behavioral_data(bids_folder=FMRI_BIDS_FOLDER):
    subjects = get_all_fmri_subjects(bids_folder=bids_folder)
    df = []
    for subject in subjects:
        d = subject.get_behavioral_data()
        if not d.empty:
            df.append(d)
    return pd.concat(df)


class FMRISubject():
    """Behavioral data for a single fMRI subject.

    File organisation::

        sourcedata/behavior/sub-{id:02d}/ses-{session}/
            sub-{id:02d}_ses-{session}_run-{run:02d}_task-{task}_events.tsv
    """

    def __init__(self, subject_id, bids_folder=FMRI_BIDS_FOLDER):
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
            mapping = "cdf" if session == 1 else "inverse_cdf"
        else:
            mapping = "inverse_cdf" if session == 1 else "cdf"
        return mapping

    def get_behavioral_data(self):
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
                    df.append(d)
                else:
                    print(f"Warning: Behavioral file {behavioral_file} does not exist")
        if not df:
            return pd.DataFrame()
        return pd.concat(df, ignore_index=True).set_index(['subject', 'session', 'mapping', 'run', 'trial_nr'])
