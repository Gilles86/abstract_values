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

    def run(self):
        self.start_experiment()
        
        # Find all reward files for this subject/session
        reward_files = list(self.output_dir.glob(f'reward_{self.subject}_{self.session}_*.txt'))
        
        if not reward_files:
            total_earnings = 0.0
            message = f'No reward files found for subject {self.subject}, session {self.session}.'
        else:
            # Read and sum all rewards
            total_earnings = 0.0
            read_errors = []
            for reward_file in sorted(reward_files):
                try:
                    with open(reward_file, 'r') as f:
                        reward = float(f.read().strip())
                        total_earnings += reward
                        print(f'{reward_file.name}: {reward:.2f} CHF')
                except (IOError, ValueError) as e:
                    error_msg = f'Error reading {reward_file.name}: {e}'
                    print(error_msg)
                    read_errors.append(error_msg)
            
            print(f'\nTotal variable reward: {total_earnings:.2f} CHF')
            total_payment = 30.0 + total_earnings

            error_note = f'\n\nNote: {len(read_errors)} file(s) could not be read.' if read_errors else ''
            message = f'Congratulations!\n\nYou have completed the experiment.\n\nShow-up fee: 30.00 CHF\nVariable reward: {total_earnings:.2f} CHF\n\nYour total earnings are:\n\n{total_payment:.2f} CHF\n\n({len(reward_files)} runs completed){error_note}\n\nThank you for participating!\n\nPlease wait for the experimenter.'
        
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
