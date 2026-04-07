from exptools2.core import Session
from pathlib import Path
from psychopy import event
from psychopy.visual import TextStim
import argparse
from utils import InstructionTrial


class EarningsSession(Session):
    def __init__(self, subject, session, output_str, output_dir=None, settings_file=None):
        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file)
        
        self.subject = subject
        self.session = session
        self.mouse = event.Mouse(visible=False)

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

    def run(self):
        self.start_experiment()

        # Find reward files for the current session
        current_files = list(self.output_dir.glob(f'reward_{self.subject}_{self.session}_*.txt'))
        current_earnings, read_errors = self._sum_reward_files(current_files)

        # For session 2+, also load previous session earnings
        prev_earnings = 0.0
        if int(self.session) >= 2:
            for prev_ses in range(1, int(self.session)):
                prev_dir = self.output_dir.parent / f'ses-{prev_ses}'
                prev_files = list(prev_dir.glob(f'reward_{self.subject}_{prev_ses}_*.txt'))
                ses_earnings, ses_errors = self._sum_reward_files(prev_files)
                prev_earnings += ses_earnings
                read_errors.extend(ses_errors)

        total_earnings = prev_earnings + current_earnings

        if not current_files and prev_earnings == 0.0:
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
                    f'({len(current_files)} runs completed this session)'
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
                    f'({len(current_files)} runs completed)'
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
    args = argparser.parse_args()

    output_dir = Path(__file__).parent / 'logs' / f'sub-{args.subject.zfill(2)}' / f'ses-{args.session}'

    session = EarningsSession(
        subject=args.subject,
        session=args.session,
        output_str=f'sub-{args.subject.zfill(2)}_ses-{args.session}_earnings',
        output_dir=output_dir,
        settings_file=Path(__file__).parent / 'settings' / f'{args.settings}.yml'
    )
    
    session.run()
