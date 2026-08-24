import os
import re
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image

# Default to the local mac path; override with $BIDS_FOLDER (e.g. the cluster
# share /shares/zne.uzh/gdehol/ds-abstractvalue) so scripts that build paths
# from this constant — not just the ones taking --bids-folder — work remotely.
BIDS_FOLDER = Path(os.environ.get('BIDS_FOLDER', '/data/ds-abstractvalue'))

# Minimum plausible response-bar RT, in seconds.  The BDM slider re-randomises
# its marker position on every trial (experiment/response_slider.py ::
# random_init_marker), so a trial that is confirmed on the very first frame
# records a uniform draw from [0, 42] CHF rather than a bid — a missed response
# wearing the costume of a fast one.  Cohort-wide the RT distribution is
# cleanly bimodal: a handful of trials at ~17 ms (one 60 Hz frame) and then
# nothing at all until 534 ms, so any threshold in [0.02, 0.5] selects exactly
# the same trials.  Pass ``min_rt=None`` to the data getters to keep them.
MIN_VALID_RT = 0.25


def flag_invalid_responses(rt, min_rt=MIN_VALID_RT):
    """Boolean mask of trials whose response-bar RT is too fast to be a bid.

    Parameters
    ----------
    rt : pandas.Series
        Response-bar reaction times in seconds.  NaN (no response) is never
        flagged — a miss is already a miss.
    min_rt : float or None
        Threshold in seconds.  ``None`` disables the check (all-False mask).

    Returns
    -------
    pandas.Series of bool, aligned to ``rt``.
    """
    if min_rt is None:
        return pd.Series(False, index=rt.index)
    rt = pd.to_numeric(rt, errors='coerce')
    return rt.notna() & (rt < min_rt)


def warn_invalid_responses(invalid, what, who):
    """Emit a one-line warning naming the frame-1 trials that were blanked."""
    n = int(invalid.sum())
    if n:
        warnings.warn(f'{who}: blanked {n} frame-1 {what} '
                      f'(RT < {MIN_VALID_RT}s; randomised slider position, '
                      f'not a bid)', stacklevel=3)


class Subject:
    """Data access for a single abstract-values fMRI subject.

    Parameters
    ----------
    subject_id : str
        Subject label without the 'sub-' prefix, e.g. ``'pil01'`` or ``'01'``.
    bids_folder : str or Path
    fmriprep_deriv : str
        Which fmriprep derivative folder to use for locating preprocessed BOLD,
        confounds, and brain masks.  Defaults to ``'fmriprep'`` (T1w + T2w).
    """

    def __init__(self, subject_id, bids_folder=BIDS_FOLDER,
                 fmriprep_deriv='fmriprep'):
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

    def get_mapping(self, session):
        """Return value-orientation mapping name for this subject and session.

        The assignment alternates by subject parity:
          Even subject number → ses-1 = 'cdf',         ses-2 = 'inverse_cdf'
          Odd  subject number → ses-1 = 'inverse_cdf', ses-2 = 'cdf'

        Returns
        -------
        str
            ``'cdf'`` or ``'inverse_cdf'``
        """
        num = int(''.join(c for c in self.subject_id if c.isdigit()))
        if num % 2 == 0:
            return 'cdf' if session == 1 else 'inverse_cdf'
        return 'inverse_cdf' if session == 1 else 'cdf'

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

    # Study subjects (sub-NN) are expected to have 2 MRI sessions; pilots (sub-pilNN)
    # have 1. Encoding / decoding / FI scripts rely on multi-session aggregation in
    # GLMsingle output paths, so they break silently if started before all sessions
    # are present. Hard-fail loudly instead.
    DEFAULT_EXPECTED_SESSIONS = 2
    PILOT_EXPECTED_SESSIONS = 1

    def expected_sessions(self):
        return (self.PILOT_EXPECTED_SESSIONS
                if self.subject_id.startswith('pil')
                else self.DEFAULT_EXPECTED_SESSIONS)

    def require_complete_sessions(self, expected=None):
        """Raise if fewer than `expected` MRI sessions are present in fmriprep output.

        Defaults: 2 for study subjects, 1 for pilots (`sub-pil*`). Pass `expected`
        to override (e.g. for legitimate single-session debug runs).
        """
        expected = expected if expected is not None else self.expected_sessions()
        sessions = self.get_sessions()
        if len(sessions) < expected:
            raise RuntimeError(
                f'sub-{self.subject_id} has only {len(sessions)} of {expected} '
                f'expected MRI sessions in fmriprep ({sessions}). Refusing to '
                f'run downstream analysis on an incomplete subject — ingest the '
                f'remaining session(s) first, or pass --allow-incomplete.'
            )

    # ── runs ───────────────────────────────────────────────────────────────────

    def get_runs(self, session):
        """Return sorted list of run numbers found in the behavior directory."""
        behavior_dir = self._behavior_dir(session)
        runs = sorted({
            int(re.search(r'run-(\d+)', f.name).group(1))
            for f in behavior_dir.glob(f'*_run-*_task-estimate.*_events.tsv')
        })
        if not runs:
            raise FileNotFoundError(f'No events files found in {behavior_dir}')
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

    def get_events(self, session, runs=None, min_rt=MIN_VALID_RT):
        """Return gabor and response_bar events for all runs.

        Returns a DataFrame indexed by (run, trial_nr) with columns:
        onset, event_type, orientation, value, bid, duration, invalid_response.

        Parameters
        ----------
        min_rt : float or None
            Trials whose response-bar RT falls below this (see
            :data:`MIN_VALID_RT`) get ``bid = NaN`` and
            ``invalid_response = True``; ``duration`` is left alone, since the
            bar really was on screen for that long and the GLM should model it
            as such.  ``None`` keeps every bid as recorded.
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

            # Set the first scanner trigger (pulse) as t=0.
            # Raw onsets are relative to Psychopy script start; the BOLD
            # acquisition starts at the first pulse, so we must subtract it.
            pulse_onsets = df.loc[df['event_type'] == 'pulse', 'onset']
            if pulse_onsets.empty:
                raise ValueError(
                    f'No pulse events found in {candidates[0]}')
            first_pulse = float(pulse_onsets.min())
            df = df.copy()
            df['onset'] = df['onset'] - first_pulse

            # The participant's bid is stored in the feedback event, not in
            # response_bar. Join it onto response_bar rows by trial_nr.
            bids = (df[df['event_type'] == 'feedback']
                    .set_index('trial_nr')['response']
                    .rename('bid'))

            # The response_bar event's duration IS the reaction time.
            rt = (df[df['event_type'] == 'response_bar']
                  .set_index('trial_nr')['duration'])
            bad = flag_invalid_responses(rt, min_rt)
            warn_invalid_responses(bad, 'bids',
                          f'sub-{self.subject_id} ses-{session} run-{run:02d}')
            bids = bids.mask(bad.reindex(bids.index, fill_value=False))

            df = df[df['event_type'].isin(['gabor', 'response_bar'])].copy()
            df = df.join(bids, on='trial_nr')
            # For response_bar events use the bid; for gabor events it is NaN
            # (intentionally — make_condition_label only uses bid for response_bar).
            # make_condition_label already falls back to response_<angle> on a
            # NaN bid, so a blanked trial degrades gracefully there.
            df['invalid_response'] = (df['trial_nr']
                                      .map(bad).fillna(False).astype(bool))
            df['run'] = run
            dfs.append(df)

        events = pd.concat(dfs, ignore_index=True)
        events = events.set_index(['run', 'trial_nr'])
        return events[['onset', 'event_type', 'orientation', 'value',
                        'bid', 'duration', 'invalid_response']]

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
        """Return brain mask NIfTI image (T1w space) from a given run.

        Returned as float32 so that NiftiMasker.inverse_transform produces
        float images rather than inheriting the on-disk uint8 dtype.
        """
        func_dir = self._func_dir(session)
        fn = (func_dir /
              f'sub-{self.subject_id}_ses-{session}'
              f'_task-abstractvalue_run-{run}_space-T1w_desc-brain_mask.nii.gz')
        if not fn.exists():
            raise FileNotFoundError(f'No brain mask: {fn}')
        mask = image.load_img(str(fn))
        return nib.Nifti1Image(mask.get_fdata().astype(np.float32), mask.affine)

    # ── ROI masks ──────────────────────────────────────────────────────────────

    def get_roi_mask(self, roi, hemi='LR'):
        """Return a volumetric ROI mask (T1w space).

        Masks are created by ``get_surface_roi_mask.py`` and stored under
        ``derivatives/masks/sub-{subject}/anat/``.

        Parameters
        ----------
        roi : str
            ROI label, e.g. ``'BensonV1'``, ``'NPC'``.
        hemi : str or None
            Hemisphere entity: ``'LR'`` (bilateral, default), ``'L'``, ``'R'``,
            or ``None`` to omit the hemi entity (used by some ROIs such as NPC).
        """
        mask_dir = (self.bids_folder / 'derivatives' / 'masks'
                    / f'sub-{self.subject_id}' / 'anat')
        if hemi:
            fn = mask_dir / f'sub-{self.subject_id}_space-T1w_hemi-{hemi}_desc-{roi}_mask.nii.gz'
        else:
            fn = mask_dir / f'sub-{self.subject_id}_space-T1w_desc-{roi}_mask.nii.gz'
        if not fn.exists():
            raise FileNotFoundError(f'No ROI mask: {fn}')
        return image.load_img(str(fn))

    # ── GLMsingle outputs ──────────────────────────────────────────────────────

    def get_single_trial_estimates(self, sessions, desc='gabor', smoothed=False,
                                   zscore_sessions=False):
        """Return single-trial beta image from GLMsingle.

        GLMsingle is fitted once across all sessions. Output path::

            derivatives/glmsingle[.smoothed]/sub-{subject}/func/
                sub-{subject}_task-abstractvalue_space-T1w_desc-{desc}_pe.nii.gz

        When ``sessions`` is a strict subset of all sessions, the full image is
        loaded and only the trials belonging to the requested sessions are
        returned (trial order: session → run → gabor event sorted by onset).

        Parameters
        ----------
        sessions : int or list of int
        desc : {'gabor', 'response', 'R2'}
        smoothed : bool
            Load from ``glmsingle.smoothed`` instead of ``glmsingle``.
        zscore_sessions : bool
            Z-score betas within each session before returning.
            Requires multiple sessions.
        """
        if isinstance(sessions, int):
            sessions = [sessions]
        sessions = sorted(sessions)

        glmsingle_deriv = 'glmsingle.smoothed' if smoothed else 'glmsingle'
        sub_dir = (self.bids_folder / 'derivatives' / glmsingle_deriv
                   / f'sub-{self.subject_id}')

        fn = (sub_dir / 'func'
              / f'sub-{self.subject_id}_task-abstractvalue'
                f'_space-T1w_desc-{desc}_pe.nii.gz')

        if not fn.exists():
            raise FileNotFoundError(
                f'No GLMsingle output ({desc}) for sub-{self.subject_id}: {fn}\n'
                f'Run fit_glmsingle.sh ({"smoothed" if smoothed else "unsmoothed"}) '
                f'for all sessions.')

        im = image.load_img(str(fn))

        # ── sanity-check volume count ─────────────────────────────────────────
        # Compute expected trial count for *all* sessions so we can warn early
        # when a stale single-session file sits at the canonical all-sessions path.
        all_sessions = sorted(self.get_sessions())
        expected_total = 0
        for ses in all_sessions:
            runs = self.get_runs(ses)
            events = self.get_events(ses, runs)
            for run in runs:
                run_ev = events.loc[run].reset_index().sort_values('onset')
                expected_total += len(run_ev[run_ev['event_type'] == 'gabor'])
        if im.shape[3] < expected_total and sessions == all_sessions:
            raise ValueError(
                f'Loaded {im.shape[3]} volumes from {fn} but expected '
                f'{expected_total} (all {len(all_sessions)} sessions). '
                f'The file may be a stale single-session output. '
                f'Re-run GLMsingle ({"smoothed" if smoothed else "unsmoothed"}) '
                f'for all sessions and it will overwrite this file.')

        # ── session subsetting ────────────────────────────────────────────────
        # When loading the all-sessions image but only a subset is requested,
        # select the matching trial indices.
        if sessions != all_sessions:
            trial_indices = []
            cumulative = 0
            for ses in all_sessions:
                runs = self.get_runs(ses)
                events = self.get_events(ses, runs)
                for run in runs:
                    run_ev = events.loc[run].reset_index().sort_values('onset')
                    n = len(run_ev[run_ev['event_type'] == 'gabor'])
                    if ses in sessions:
                        trial_indices.extend(range(cumulative, cumulative + n))
                    cumulative += n
            im = image.index_img(im, trial_indices)

        # ── per-session z-scoring ─────────────────────────────────────────────
        if zscore_sessions:
            if len(sessions) < 2:
                raise ValueError('zscore_sessions requires multiple sessions')
            session_sizes = []
            for ses in sessions:
                runs = self.get_runs(ses)
                events = self.get_events(ses, runs)
                n = sum(
                    len(events.loc[run].reset_index()
                        .query("event_type == 'gabor'"))
                    for run in runs
                )
                session_sizes.append(n)
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

    # ── encoding model outputs ─────────────────────────────────────────────────

    def get_fisher_information(self, session=None, roi='BensonV1', hemi='LR',
                               n_voxels=250, smoothed=False):
        """Return Fisher information DataFrame computed by compute_fisher_information.py.

        Index is orientation in **degrees**; column is ``fisher_information``.

        Parameters
        ----------
        session : int or None
            Session number.  ``None`` loads the across-session fit (no ses- dir).
        roi : str
        hemi : str or None
            ``'LR'``, ``'L'``, ``'R'``, or ``None`` (omit hemi entity).
        n_voxels : int
        smoothed : bool
        """
        hemi_arg = None if hemi == 'None' else hemi
        mask_desc = f'{roi}{"_hemi-" + hemi if hemi_arg else ""}'
        smooth_label = '_smoothed' if smoothed else ''

        ses_dir    = f'ses-{session}' if session is not None else ''
        ses_entity = f'_ses-{session}' if session is not None else ''

        out_dir = (self.bids_folder / 'derivatives' / 'encoding_models' / 'vonmises'
                   / f'sub-{self.subject_id}')
        if ses_dir:
            out_dir = out_dir / ses_dir
        out_dir = out_dir / 'func'

        fn = (out_dir /
              f'sub-{self.subject_id}{ses_entity}_task-abstractvalue'
              f'_mask-{mask_desc}_nvoxels-{n_voxels}{smooth_label}_desc-fisherinfo_pe.tsv')

        if not fn.exists():
            raise FileNotFoundError(f'No Fisher information file: {fn}')

        df = pd.read_csv(fn, sep='\t', index_col=0)
        df.index = np.rad2deg(df.index)
        return df

    def get_prf_parameters(self, sessions=None, smoothed=False):
        """Return dict of NIfTI images for fitted aPRF parameters.

        Keys: ``'mode'``, ``'fwhm'``, ``'amplitude'``, ``'baseline'``, ``'r2'``.

        The aPRF (LogGaussianPRF) is always fitted jointly across all of a
        subject's sessions; outputs live at
        ``derivatives/encoding_models/aprf/sub-<subject>/func/`` with no
        ``ses-*`` entity in the path or filename.

        Parameters
        ----------
        sessions : ignored (kept for backwards compatibility with old callers).
            Per-session aPRF fits were dropped; the joint fit is now the only
            output and this parameter has no effect on the path returned.
        smoothed : bool
        """
        del sessions  # legacy: per-session fits no longer exist
        smooth_label = '_smoothed' if smoothed else ''

        out_dir = (self.bids_folder / 'derivatives' / 'encoding_models' / 'aprf'
                   / f'sub-{self.subject_id}' / 'func')

        result = {}
        for param in ['mode', 'fwhm', 'amplitude', 'baseline', 'r2']:
            fn = (out_dir / f'sub-{self.subject_id}_task-abstractvalue'
                            f'_space-T1w_desc-{param}{smooth_label}_pe.nii.gz')
            if not fn.exists():
                raise FileNotFoundError(f'No aPRF parameter file: {fn}')
            result[param] = image.load_img(str(fn))
        return result

    # ── surface-sampled encoding maps ───────────────────────────────────────────

    def get_encoding_surface(self, model, desc, hemi,
                             space='fsaverage', smoothed=False):
        """Load one surface-sampled encoding-model map for one hemisphere.

        Reads the GIfTI written by ``surface/sample_aprf_to_surface.py`` /
        ``sample_r2_to_surface.py``:
        ``derivatives/encoding_models/<model>/sub-<s>/func/
        sub-<s>_task-abstractvalue_hemi-<L|R>_space-<space>_desc-<desc>[_smoothed]_pe.func.gii``

        Parameters
        ----------
        model : str
            Encoding-model dir, e.g. ``'aprf'``, ``'aprf.cv'``, ``'aprf-null.cv'``.
        desc : str
            Map to load, e.g. ``'mode'``, ``'r2'``, ``'cvr2'``, ``'gabor-r2'``.
        hemi : {'L', 'R'}
        space : str
            ``'fsaverage'`` (default) or ``'fsnative'``.
        smoothed : bool

        Returns
        -------
        np.ndarray, shape (n_vertices,), float32. Raises FileNotFoundError if
        the file is missing.
        """
        smooth = '_smoothed' if smoothed else ''
        fn = (self.bids_folder / 'derivatives' / 'encoding_models' / model
              / f'sub-{self.subject_id}' / 'func'
              / f'sub-{self.subject_id}_task-abstractvalue'
                f'_hemi-{hemi}_space-{space}_desc-{desc}{smooth}_pe.func.gii')
        if not fn.exists():
            raise FileNotFoundError(f'No surface map: {fn}')
        return nib.load(str(fn)).darrays[0].data.astype(np.float32)

    def get_encoding_surface_bilateral(self, model, desc,
                                       space='fsaverage', smoothed=False):
        """L+R concatenated surface map (pycortex convention), or ``None`` if
        either hemisphere is missing."""
        try:
            return np.concatenate([
                self.get_encoding_surface(model, desc, h, space, smoothed)
                for h in ('L', 'R')])
        except FileNotFoundError:
            return None


# ── cross-validated R² "signal" helpers ─────────────────────────────────────────
#
# The right per-voxel test for "does this encoding model carry signal" is not
# ``cvR² > 0`` but ``cvR² > cvR²_null``, where the null model predicts the
# *training-set* mean for every voxel. A genuinely silent voxel can only reach
# the training mean, which scores a slightly NEGATIVE cvR² on held-out data
# (≈ −0.03 here) because of train/test mean mismatch — so ``> 0`` is far too
# strict and discards real-but-modest voxels (empirically ~11× fewer survive).
# cvR² is also parameter-count-fair, so the same criterion is comparable across
# models with different numbers of parameters. These helpers are deliberately
# light (numpy + nibabel only) so they run in the pycortex2 env too.

DEFAULT_NULL_MODEL = 'aprf-null.cv'


def cvr2_signal(subject, model, baseline_model=DEFAULT_NULL_MODEL, cv_thr=0.0,
                space='fsaverage', smoothed=False, bids_folder=BIDS_FOLDER):
    """Per-vertex "this model beats the null" mask for one subject.

    Compares ``model`` cvR² against, per vertex, the ``baseline_model`` cvR²
    (the null-null "predict training mean" model). If ``baseline_model`` is
    None — or its surface map is missing — falls back to the scalar ``cv_thr``.

    Returns ``(signal, delta)``: a boolean ndarray (model wins) and the cvR²
    margin ``model_cvr2 - reference``. Returns ``(None, None)`` if the model's
    own cvR² surface is missing.
    """
    sub = Subject(subject, bids_folder=bids_folder)
    cvr2 = sub.get_encoding_surface_bilateral(model, 'cvr2', space, smoothed)
    if cvr2 is None:
        return None, None
    ref = None
    if baseline_model:
        ref = sub.get_encoding_surface_bilateral(
            baseline_model, 'cvr2', space, smoothed)
    if ref is None:
        ref = np.full_like(cvr2, cv_thr, dtype=np.float32)
    delta = cvr2 - ref
    return (np.isfinite(delta) & (delta > 0)), delta


def cvr2_prevalence(subjects, model, baseline_model=DEFAULT_NULL_MODEL,
                    cv_thr=0.0, space='fsaverage', smoothed=False,
                    bids_folder=BIDS_FOLDER):
    """Per-vertex prevalence: fraction of subjects where ``model`` beats the
    null (see :func:`cvr2_signal`).

    Returns a dict with ``count`` (positive subjects per vertex), ``n`` (number
    of subjects with data), ``prop`` (count / n, float32), ``p0`` (pooled base
    rate of "wins" — a sanity reference for a binomial prevalence test), and
    ``subjects`` (labels actually used). Returns ``None`` if no subject had data.
    """
    masks, used = [], []
    for s in subjects:
        sig, _ = cvr2_signal(s, model, baseline_model=baseline_model,
                             cv_thr=cv_thr, space=space, smoothed=smoothed,
                             bids_folder=bids_folder)
        if sig is None:
            continue
        masks.append(sig)
        used.append(str(s))
    if not masks:
        return None
    stack = np.vstack(masks)                       # (n_subjects, n_vertices) bool
    count = stack.sum(axis=0)
    n = stack.shape[0]
    return {
        'count': count,
        'n': n,
        'prop': (count / n).astype(np.float32),
        'p0': float(stack.mean()),
        'subjects': used,
    }
