from exptools2.core import Session, Trial
from psychopy import event
import yaml
from pathlib import Path
from psychopy.visual import TextStim
import numpy as np
from utils import get_value
from stimuli import AnnulusGrating, FixationCross
import argparse

class ExampleSession(Session):
    def __init__(self, subject, mapping, output_str, output_dir=None, settings_file=None):
        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file)
        self.mouse = event.Mouse(visible=False)
        self.settings['subject'] = subject
        self.settings['mapping'] = mapping
        self.bottom_text = TextStim(self.win, text='Press left/right to change orientation\nPress SPACE BAR to quit.',
                                    pos=(0, -1.5 * self.settings['grating']['size'] / 2),
                                    height=.75, color='white', wrapWidth=60)
        self.orientations = self.settings['mappings']['orientations'][1:-1]
        self.current_orientation_idx = 0
        self.fixation_cross = FixationCross(self.win, 
                                                size=self.settings['fixation_cross']['size'],
                                                color=self.settings['fixation_cross']['color'],
                                                line_width=self.settings['fixation_cross']['line_width'])

    def run(self):
        self.start_experiment()
        self.create_trials()
        self.trials[0].run()  # Only one trial
        self.close()

    def create_trials(self):
        self.trials = [ExampleTrial(self, trial_nr=1, orientation=self.orientations[0])]

    def update_orientation(self, step):
        self.current_orientation_idx += step
        self.current_orientation_idx = np.clip(self.current_orientation_idx, 0, len(self.orientations) - 1)
        self.trials[0].update_stimuli(self.orientations[self.current_orientation_idx])

class ExampleTrial(Trial):
    def __init__(self, session, trial_nr, orientation=0):
        phase_names = ['show_gabor']
        phase_durations = [np.inf]
        super().__init__(session, trial_nr, phase_durations, phase_names=phase_names)
        self.orientation = orientation
        self.value = get_value(orientation, self.session.settings['mapping'])
        size_grating = self.session.settings['grating']['size']
        # self.grating = GratingStim(session.win, tex='sin', mask='circle',
        #                            sf=session.settings['grating']['spatial_freq'], ori=orientation, size=size_grating)
        self.grating = AnnulusGrating(session.win,
                                      clock=self.session.clock,
                                      size=size_grating,
                                      hole_deg=self.session.settings['grating']['hole_size'],
                                      sf=self.session.settings['grating']['spatial_freq'],
                                      ori=orientation,
                                      contrast=1.0,
                                      tf=self.session.settings['grating']['temporal_frequency'])

        self.value_instruction_text = TextStim(session.win, text=f'Value: ',
                                   pos=(0, 1.7 * size_grating / 2),
                                   height=1, color=(1, 1, 1))
        self.value_text = TextStim(session.win, text=f'{self.value:.2f}',
                                   pos=(0, 1.35 * size_grating / 2),
                                   height=1, color=(-1, 1, -1))

    def update_stimuli(self, orientation):
        self.orientation = orientation
        self.value = get_value(orientation, self.session.settings['mapping'])

        if self.grating.drift_direction == 1:
            self.grating.drift_direction = -1
        else:
            self.grating.drift_direction = 1

        self.grating.set_ori(orientation)
        self.value_text.text = f'{self.value:.2f}'

    def draw(self):
        self.grating.draw()
        self.value_instruction_text.draw()
        self.session.bottom_text.draw()
        self.session.fixation_cross.draw()
        self.value_text.draw()

    def get_events(self):
        events = super().get_events()
        for key, t in events:
            if key == 'left':
                self.session.update_orientation(-1)
            elif key == 'right':
                self.session.update_orientation(1)
            elif key == 'space':
                self.session.close()
                self.session.quit()

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('subject', type=str, help='Subject identifier')
    argparser.add_argument('session', type=int, help='Session number')
    argparser.add_argument('mapping', type=str, choices=['linear', 'cdf', 'inverse_cdf'], help='Mapping type')
    argparser.add_argument('--settings', type=str, default='default', help='Name of settings file (default by default)')
    
    args = argparser.parse_args()
    session = ExampleSession(subject=args.subject,
                             mapping=args.mapping,
                             output_str=f'sub-{args.subject}_ses-{args.session:02d}_task-examples.{args.mapping}',
                             output_dir=Path(__file__).parent / 'logs' / f'sub-{args.subject}' / f'session-{args.session:02d}',
                             settings_file=Path(__file__).parent / 'settings' / f'{args.settings}.yml')
    session.run()
