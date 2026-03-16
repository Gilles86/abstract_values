import re
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image

BIDS_FOLDER = Path('/data/ds-abstractvalue')


class Subject:
    """Data access for a single abstract-values fMRI subject.

    Parameters
    ----------
    subject_id : str
        Subject label without the 'sub-' prefix, e.g. ``'pil01'`` or ``'01'``.
    bids_folder : str or Path
    fmriprep_deriv : str
        Which fmriprep derivative folder to use.
        One of ``'fmriprep'``, ``'fmriprep-flair'``, ``'fmriprep-noflair'``.
    """

    def __init__(self, subject_id, bids_folder=BIDS_FOLDER,
                 fmriprep_deriv='fmriprep-flair'):
        self.subject_id = str(subject_id)
        self.bids_folder = Path(bids_folder)
        self.fmriprep_deriv = fmriprep_deriv

    # ── private helpers ────────────────────────────────────────────────────────

    @property
    def _fmriprep_dir(self):
        return self.bids_folder / 'derivatives' / self.fmriprep_deriv

    def _func_dir(self, session):
        return (self._fmriprep_dir / f'sub-{self.subject_id}'
                / f'ses-{session}' / 'func')

    def _behavior_dir(self, session):
        return (self.bids_folder / 'sourcedata' / 'behavior'
                / f'sub-{self.subject_id}' / f'ses-{session}')

    # ── sessions ───────────────────────────────────────────────────────────────

    def get_sessions(self):
        """Return sorted list of session numbers available in fmriprep output."""
        sub_dir = self._fmriprep_dir / f'sub-{self.subject_id}'
        sessions = []
        for d in sub_dir.iterdir():
            m = re.match(r'ses-(\d+)$', d.name)
            if d.is_dir() and m:
                sessions.append(int(m.group(1)))
        sessions = sorted(sessions)
        if not sessions:
            raise FileNotFoundError(f'No sessions found in {sub_dir}')
        return sessions

    # ── runs ───────────────────────────────────────────────────────────────────

    def get_runs(self, session):
        """Return sorted list of run numbers available in fmriprep output."""
        func_dir = self._func_dir(session)
        pattern = (f'sub-{self.subject_id}_ses-{session}'
                   f'_task-abstractvalue_run-*_space-T1w_*desc-preproc_bold.nii.gz')
        runs = sorted({
            int(re.search(r'run-(\d+)', f.name).group(1))
            for f in func_dir.glob(pattern)
        })
        if not runs:
            raise FileNotFoundError(
                f'No preprocessed BOLD found in {func_dir}\n'
                f'(fmriprep_deriv={self.fmriprep_deriv!r})')
        return runs

    # ── BOLD ───────────────────────────────────────────────────────────────────

    def get_preprocessed_bold(self, session, runs=None):
        """Return list of preprocessed BOLD Paths (T1w space)."""
        if runs is None:
            runs = self.get_runs(session)
        func_dir = self._func_dir(session)
        paths = []
        for run in runs:
            matches = sorted(func_dir.glob(
                f'sub-{self.subject_id}_ses-{session}'
                f'_task-abstractvalue_run-{run}_space-T1w_*desc-preproc_bold.nii.gz'
            ))
            if not matches:
                raise FileNotFoundError(
                    f'No BOLD file for run-{run} in {func_dir}')
            paths.append(matches[0])
        return paths

    # ── events ─────────────────────────────────────────────────────────────────

    def get_events(self, session, runs=None):
        """Return gabor and response_bar events for all runs.

        Returns a DataFrame indexed by (run, trial_nr) with columns:
        onset, event_type, orientation, value, response, duration.
        """
        if runs is None:
            runs = self.get_runs(session)

        dfs = []
        for run in runs:
            behavior_dir = self._behavior_dir(session)
            candidates = sorted(behavior_dir.glob(
                f'*_run-{run:02d}_task-estimate.*_events.tsv'))
            if not candidates:
                raise FileNotFoundError(
                    f'No events file for sub-{self.subject_id} '
                    f'ses-{session} run-{run:02d} in {behavior_dir}')
            df = pd.read_csv(candidates[0], sep='\t')

            # The participant's bid is stored in the feedback event, not in
            # response_bar. Join it onto response_bar rows by trial_nr.
            bids = (df[df['event_type'] == 'feedback']
                    .set_index('trial_nr')['response']
                    .rename('bid'))
            df = df[df['event_type'].isin(['gabor', 'response_bar'])].copy()
            df = df.join(bids, on='trial_nr')
            # For response_bar events use the bid; for gabor events it is NaN
            # (intentionally — make_condition_label only uses bid for response_bar).
            df['run'] = run
            dfs.append(df)

        events = pd.concat(dfs, ignore_index=True)
        events = events.set_index(['run', 'trial_nr'])
        return events[['onset', 'event_type', 'orientation', 'value',
                        'bid', 'duration']]

    # ── confounds ──────────────────────────────────────────────────────────────

    def get_confounds(self, session, runs=None,
                      columns=('cosine00', 'cosine01', 'cosine02',
                               'trans_x', 'trans_y', 'trans_z',
                               'rot_x', 'rot_y', 'rot_z')):
        """Return confound timeseries for all runs.

        Returns a DataFrame indexed by (run, timepoint).
        """
        if runs is None:
            runs = self.get_runs(session)
        func_dir = self._func_dir(session)
        dfs = []
        for run in runs:
            fn = (func_dir /
                  f'sub-{self.subject_id}_ses-{session}'
                  f'_task-abstractvalue_run-{run}'
                  f'_desc-confounds_timeseries.tsv')
            if not fn.exists():
                raise FileNotFoundError(f'No confounds file: {fn}')
            df = pd.read_csv(fn, sep='\t')
            available = [c for c in columns if c in df.columns]
            dfs.append(df[available])
        return pd.concat(dfs, keys=runs, names=['run'])

    # ── brain mask ─────────────────────────────────────────────────────────────

    def get_brain_mask(self, session, run=1):
        """Return brain mask NIfTI image (T1w space) from a given run."""
        func_dir = self._func_dir(session)
        fn = (func_dir /
              f'sub-{self.subject_id}_ses-{session}'
              f'_task-abstractvalue_run-{run}_space-T1w_desc-brain_mask.nii.gz')
        if not fn.exists():
            raise FileNotFoundError(f'No brain mask: {fn}')
        return image.load_img(str(fn))

    # ── GLMsingle outputs ──────────────────────────────────────────────────────

    def get_single_trial_estimates(self, sessions, desc='gabor', smoothed=False,
                                   zscore_sessions=False):
        """Return single-trial beta image from GLMsingle.

        Parameters
        ----------
        sessions : int or list of int
            Session number(s) that were fitted jointly. Pass a single int for
            a single-session fit, or a list for a multi-session fit.
        desc : {'gabor', 'response', 'R2'}
        smoothed : bool
            Load from ``glmsingle.smoothed`` instead of ``glmsingle``.
        zscore_sessions : bool
            Z-score betas within each session independently before returning.
            Only valid when multiple sessions were fitted jointly.
        """
        if isinstance(sessions, int):
            sessions = [sessions]
        ses_label = f'ses-{sessions[0]}' if len(sessions) == 1 else 'ses-all'

        glmsingle_deriv = 'glmsingle.smoothed' if smoothed else 'glmsingle'
        fn = (self.bids_folder / 'derivatives' / glmsingle_deriv
              / self.fmriprep_deriv / f'sub-{self.subject_id}'
              / ses_label / 'func'
              / f'sub-{self.subject_id}_{ses_label}'
                f'_task-abstractvalue_space-T1w_desc-{desc}_pe.nii.gz')
        if not fn.exists():
            raise FileNotFoundError(f'No GLMsingle output ({desc}): {fn}')

        im = image.load_img(str(fn))

        if zscore_sessions:
            if len(sessions) < 2:
                raise ValueError('zscore_sessions requires multiple sessions')
            # Determine trial count per session to split the 4-D image.
            session_sizes = []
            for session in sessions:
                runs = self.get_runs(session)
                events = self.get_events(session, runs)
                n_trials = sum(len(events.loc[run]) for run in runs)
                session_sizes.append(n_trials)
            boundaries = np.cumsum([0] + session_sizes)
            zscored = []
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                block = image.index_img(im, slice(start, stop))
                zscored.append(image.clean_img(block, detrend=False,
                                               standardize='zscore'))
            im = image.concat_imgs(zscored)

        return im

    def get_glmsingle_betas(self, sessions, desc='gabor'):
        """Alias for get_single_trial_estimates (smoothed=False)."""
        return self.get_single_trial_estimates(sessions, desc=desc)
