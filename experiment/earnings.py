from exptools2.core import Session
from pathlib import Path
from psychopy import event
from psychopy.visual import TextStim
import argparse
import os
from utils import InstructionTrial


# Default archive location: the project folder on the department share. The drive
# letter differs per stim PC (T:\ on the old one, Z:\ on the current one), so it
# can be overridden with the same environment variable _common.ps1 uses.
DEFAULT_BACKUP_DIR = r'Z:\Department\projects\2026\dehollander_bedi_ruff_abstract_values\data\sourcedata\behavior'


def default_backup_dir():
    """Archive root to fall back on, from $ABSTRACT_VALUES_BACKUP or the default."""
    return Path(os.environ.get('ABSTRACT_VALUES_BACKUP', DEFAULT_BACKUP_DIR))


class EarningsSession(Session):
    def __init__(self, subject, session, output_str, output_dir=None, settings_file=None,
                 backup_dir=None):
        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file)

        self.subject = subject
        self.session = session
        self.backup_dir = Path(backup_dir) if backup_dir is not None else None
        self.mouse = event.Mouse(visible=False)

    def _is_this_subject(self, token):
        """Does a reward filename's subject token belong to this subject?

        Tolerant about zero-padding: '5' and '05' are the same subject.
        """
        if token == str(self.subject):
            return True
        try:
            return int(token) == int(self.subject)
        except (TypeError, ValueError):
            return False

    def _reward_files_in(self, directory, session):
        """Reward files for this subject/session in `directory`, keyed by run number.

        Keying by run (rather than by filename) means the same run can never be
        counted twice, whatever the zero-padding in the filename.
        """
        if directory is None:
            return {}

        try:
            if not directory.is_dir():
                return {}
            candidates = sorted(directory.glob('reward_*.txt'))
        except OSError as e:
            # An unmapped or unreachable network drive must never break payment.
            print(f'Could not read {directory}: {e}')
            return {}

        files = {}
        for reward_file in candidates:
            parts = reward_file.stem.split('_')  # reward, subject, session, run
            if len(parts) != 4:
                continue
            _, subject_token, session_token, run_token = parts
            if not self._is_this_subject(subject_token):
                continue
            try:
                if int(session_token) != int(session):
                    continue
                run = int(run_token)
            except ValueError:
                continue
            files[run] = reward_file

        return files

    def _locate_session_rewards(self, session):
        """Find this subject's reward files for `session`, local copy preferred.

        Returns (files_by_run, n_from_backup). The local `logs` folder is the
        primary source: the runs of the session being finished right now have not
        been copied to the share yet. The share fills the gaps -- which is what
        rescues a second session whose first session was recorded on another PC.
        """
        local_dir = self.output_dir.parent / f'ses-{session}'

        backup_dir = None
        if self.backup_dir is not None:
            backup_dir = (self.backup_dir / f'sub-{str(self.subject).zfill(2)}'
                          / f'ses-{session}')

        local = self._reward_files_in(local_dir, session)
        backup = self._reward_files_in(backup_dir, session)

        files = dict(backup)
        files.update(local)  # local wins for runs present in both
        n_from_backup = len(set(backup) - set(local))

        if n_from_backup:
            print(f'Session {session}: {n_from_backup} run(s) read from {backup_dir}')

        return files, n_from_backup

    def _sum_reward_files(self, reward_files):
        """Read and sum a list of reward files. Returns (total, errors)."""
        total = 0.0
        errors = []
        for reward_file in sorted(reward_files):
            try:
                reward = float(reward_file.read_text().strip())
                total += reward
                print(f'{reward_file.name}: {reward:.2f} CHF')
            except (IOError, ValueError) as e:
                error_msg = f'Error reading {reward_file.name}: {e}'
                print(error_msg)
                errors.append(error_msg)
        return total, errors

    def _sum_session(self, session):
        """Total earnings for one session, from local logs plus the share."""
        files, n_from_backup = self._locate_session_rewards(session)
        total, errors = self._sum_reward_files(list(files.values()))

        n_expected = self.settings.get('main_task', {}).get('n_blocks')
        if n_expected is not None and len(files) != n_expected:
            missing = sorted(set(range(1, n_expected + 1)) - set(files))
            print(f'WARNING: session {session} has {len(files)}/{n_expected} runs; '
                  f'missing run(s): {missing}')

        return total, len(files), errors, n_from_backup

    def run(self):
        self.start_experiment()

        # Earnings of the session that just finished
        current_earnings, n_current, read_errors, n_backup = self._sum_session(self.session)
        n_current_files = n_current

        # For session 2+, also load previous session earnings
        prev_earnings = 0.0
        for prev_ses in range(1, int(self.session)):
            ses_earnings, _, ses_errors, ses_backup = self._sum_session(prev_ses)
            prev_earnings += ses_earnings
            read_errors.extend(ses_errors)
            n_backup += ses_backup

        total_earnings = prev_earnings + current_earnings

        if n_current_files == 0 and prev_earnings == 0.0:
            message = f'No reward files found for subject {self.subject}, session {self.session}.'
        else:
            print(f'\nTotal variable reward: {total_earnings:.2f} CHF')
            n_sessions = int(self.session)
            show_up_total = 30.0 * n_sessions
            total_payment = show_up_total + total_earnings
            error_note = f'\n\nNote: {len(read_errors)} file(s) could not be read.' if read_errors else ''

            if prev_earnings > 0.0:
                message = (
                    f'Congratulations!\n\n'
                    f'You have completed the experiment.\n\n'
                    f'Show-up fee: {show_up_total:.0f} CHF ({n_sessions} sessions)\n'
                    f'Session 1 reward: {prev_earnings:.2f} CHF\n'
                    f'Session 2 reward: {current_earnings:.2f} CHF\n'
                    f'Total variable reward: {total_earnings:.2f} CHF\n\n'
                    f'Your total earnings are:\n\n'
                    f'{total_payment:.2f} CHF\n\n'
                    f'({n_current_files} runs completed this session)'
                    f'{error_note}\n\n'
                    f'Thank you for participating!\n\n'
                    f'Please wait for the experimenter.'
                )
            else:
                message = (
                    f'Congratulations!\n\n'
                    f'You have completed the experiment.\n\n'
                    f'Show-up fee: 30.00 CHF\n'
                    f'Variable reward: {total_earnings:.2f} CHF\n\n'
                    f'Your total earnings are:\n\n'
                    f'{total_payment:.2f} CHF\n\n'
                    f'({n_current_files} runs completed)'
                    f'{error_note}\n\n'
                    f'Thank you for participating!\n\n'
                    f'Please wait for the experimenter.'
                )

        # Display earnings on screen
        earnings_trial = InstructionTrial(
            self,
            trial_nr=0,
            txt=message,
            keys=None
        )
        earnings_trial.run()

        self.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Display total earnings for a subject/session')
    argparser.add_argument('subject', type=str, help='Subject identifier')
    argparser.add_argument('session', type=int, help='Session number')
    argparser.add_argument('--settings', type=str, default='default', help='Name of settings file (default by default)')
    argparser.add_argument('--backup-dir', type=str, default=None,
                           help='Archive root to fall back on for runs missing from the local '
                                'logs folder (default: $ABSTRACT_VALUES_BACKUP, else the '
                                'department share). Mainly there so a second session can find '
                                'the first session recorded on another PC.')
    argparser.add_argument('--no-backup', action='store_true',
                           help='Only use the local logs folder, never the share.')
    args = argparser.parse_args()

    output_dir = Path(__file__).parent / 'logs' / f'sub-{args.subject.zfill(2)}' / f'ses-{args.session}'

    if args.no_backup:
        backup_dir = None
    elif args.backup_dir is not None:
        backup_dir = Path(args.backup_dir)
    else:
        backup_dir = default_backup_dir()

    session = EarningsSession(
        subject=args.subject,
        session=args.session,
        output_str=f'sub-{args.subject.zfill(2)}_ses-{args.session}_earnings',
        output_dir=output_dir,
        settings_file=Path(__file__).parent / 'settings' / f'{args.settings}.yml',
        backup_dir=backup_dir,
    )

    session.run()
