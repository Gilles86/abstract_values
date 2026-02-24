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
            for reward_file in sorted(reward_files):
                with open(reward_file, 'r') as f:
                    reward = float(f.read().strip())
                    total_earnings += reward
                    print(f'{reward_file.name}: {reward:.2f} CHF')
            
            print(f'\nTotal earnings: {total_earnings:.2f} CHF')
            message = f'Congratulations!\n\nYou have completed the experiment.\n\nYour total earnings are:\n\n{total_earnings:.2f} CHF\n\n({len(reward_files)} runs completed)\n\nThank you for participating!\n\nPlease wait for the experimenter.'
        
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

    output_dir = Path(__file__).parent / 'logs' / f'sub-{args.subject}' / f'session-{args.session:02d}'
    
    session = EarningsSession(
        subject=args.subject,
        session=args.session,
        output_str=f'sub-{args.subject}_ses-{args.session:02d}_earnings',
        output_dir=output_dir,
        settings_file=Path(__file__).parent / 'settings' / f'{args.settings}.yml'
    )
    
    session.run()
