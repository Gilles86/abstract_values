from exptools2.core import Session, Trial
from psychopy import event
import yaml
from pathlib import Path
from psychopy.visual import GratingStim, TextStim
import numpy as np
from utils import InstructionTrial, get_value
from response_slider import ResponseSlider
from stimuli import AnnulusGrating, FixationCross
import argparse

class TrainingSession(Session):
    def __init__(self, subject, run, mapping, output_str, output_dir=None, settings_file=None):
        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file)

        self.instructions_file = Path(__file__).parent / 'instructions.yml'
        with open(self.instructions_file, 'r') as f:
            self.instructions = yaml.safe_load(f)['phase_2']

        self.mouse = event.Mouse(visible=False)
        self.settings['subject'] = subject
        self.settings['run'] = run
        self.settings['mapping'] = mapping
        self.settings['range'] = [2, 42]
        self.fixation_stimulus = FixationCross(self.win, 
                                                size=self.settings['fixation_cross']['size'],
                                                color=self.settings['fixation_cross']['color'],
                                                line_width=self.settings['fixation_cross']['line_width'])

        self.too_late_stimulus = TextStim(self.win, text='Too late!', pos=(0, 0), color=(1, -1, -1), height=0.5)

        self._setup_response_slider()

    def _setup_response_slider(self):
        position_slider = (0, -1.2*self.settings['grating']['size']/2)
        max_range = self.settings['slider'].get('max_range')[1] - self.settings['slider'].get('max_range')[0]
        length_line = self.settings['slider'].get('max_length')

        self.response_slider = ResponseSlider(self.win,
                                         position_slider,
                                         length_line,
                                         self.settings['slider'].get('height'),
                                         self.settings['slider'].get('color'),
                                         self.settings['slider'].get('borderColor'),
                                         self.settings['range'],
                                         marker_position=None,
                                         markerColor=self.settings['slider'].get('markerColor'),
                                         borderWidth=self.settings['slider'].get('borderWidth'),
                                         text_height=self.settings['slider'].get('text_height'),
                                         precision=0.5)
    def run(self):
        self.start_experiment()
        for trial in self.trials:
            trial.run()

        self.close()

    def create_trials(self, n_trials=None):


        self.trials = []

        self.trials.append(InstructionTrial(self, trial_nr=0,
                                             txt=self.instructions['instructions'],
                                             bottom_txt='Press SPACE BAR to continue.',
                                             keys=['space'],
                                             phase_durations=[np.inf],
                                             phase_names=['instruction']))

        # Randomly sample orientations:
        # possible orientations
        self.orientations = self.settings['mappings']['orientations'][1:-1]

        # Repeat orientations and shuffle
        self.orientations = np.tile(self.orientations, self.settings['training']['n_repeats'])
        np.random.shuffle(self.orientations)

        # Split orientations into blocks
        n_blocks = self.settings['training']['n_blocks']
        block_size = len(self.orientations) // n_blocks  # Integer division

        if n_trials is not None:
            n_blocks = 1
            self.orientations = self.orientations[:n_trials]

        # Add InstructionTrial at the start of each block
        for i in range(n_blocks):
            # Calculate start and end indices for the current block
            start_idx = i * block_size
            end_idx = (i + 1) * block_size if i < n_blocks - 1 else len(self.orientations)  # Handle last block

            # Add instruction trial
            block_start_trial = InstructionTrial(
                self,
                trial_nr=len(self.trials) + 1,
                txt=f'Block {i+1} of {n_blocks}\n\nIn this block, you will continue to receive feedback on your performance.\n\nRemember to respond as accurately as possible using the slider.\n\nPress any button to start.',
            )
            self.trials.append(block_start_trial)

            # Add trials for this block
            for ori in self.orientations[start_idx:end_idx]:
                self.trials.append(TrainingTrial(self, trial_nr=len(self.trials) + 1, orientation=ori))


class TrainingTrial(Trial):
    def __init__(self, session, trial_nr, orientation=0):
        phase_names = ['green_fixation',
                       'white_fixation','gabor',
                       'response_bar', 'feedback',
                       'iti']

        self.session = session

        phase_durations = [self.session.settings['durations_training']['green_fixation'],
                            self.session.settings['durations_training']['white_fixation'],
                            self.session.settings['durations_training']['gabor'],
                            self.session.settings['durations_training']['response_bar'],
                            self.session.settings['durations_training']['feedback'],
                            self.session.settings['durations_training']['iti']]

        super().__init__(session, trial_nr, phase_durations, phase_names=phase_names)

        self.parameters['orientation'] = orientation
        self.parameters['value'] = get_value(orientation, self.session.settings['mapping'])
        size_grating = self.session.settings['grating']['size']

        # self.grating = GratingStim(session.win, tex='sin', mask='circle',
        #                            sf=session.settings['grating']['spatial_freq'], ori=orientation, size=size_grating)

        self.grating = AnnulusGrating(session.win,
                                      clock=self.session.clock,
                                      size=size_grating,
                                      hole_deg=self.session.settings['grating']['hole_size'],
                                      sf=self.session.settings['grating']['spatial_freq'],
                                      ori=self.parameters['orientation'],
                                      contrast=1.0,
                                      tf=self.session.settings['grating']['temporal_frequency'])

        self.feedback_text = TextStim(session.win, text=f'{self.parameters["value"]:.2f}',
                                   pos=(0, 1.2 * self.session.settings['grating']['size'] / 2.),
                                   height=.75, color=(-1, 1, -1))

        self.response_phase = self.phase_names.index('response_bar')
        self.total_duration = np.sum(phase_durations)

    def draw(self):

        draw_fixation = True

        if self.phase_names[self.phase] == 'green_fixation':
            self.session.fixation_stimulus.set_color(( -1, 1, -1))  # green
        elif self.phase_names[self.phase] == 'white_fixation':
            self.session.fixation_stimulus.set_color((1, 1, 1))  # white
        if self.phase_names[self.phase] == 'gabor':
            self.grating.draw()

        elif self.phase_names[self.phase] == 'response_bar':
            self.grating.draw()
            self.session.response_slider.marker.inner_color = self.session.settings['slider'].get('color')
            self.session.response_slider.draw()

        elif self.phase_names[self.phase] == 'feedback':
            self.grating.draw()
            if hasattr(self, 'response_onset'):
                self.session.response_slider.marker.inner_color = self.session.settings['slider'].get('feedbackColor')
                self.session.response_slider.draw()
            else:
                self.session.too_late_stimulus.draw()
            self.feedback_text.draw()

        if draw_fixation:
            self.session.fixation_stimulus.draw()


    def get_events(self):
        events = super().get_events()

        response_slider = self.session.response_slider

        if self.phase == (self.response_phase - 1):

            if (not self.session.mouse.getPressed()[0]) and (self.session.mouse.getPos()[0] != response_slider.marker.pos[0]):
                try:
                    self.session.mouse.setPos((response_slider.marker.pos[0],0))
                    self.last_mouse_pos = response_slider.marker.pos[0]
                except Exception as e:
                    print(f'Warning: Could not set mouse position: {e}')
                    self.last_mouse_pos = self.session.mouse.getPos()[0]/self.session.settings['interface']['mouse_multiplier']

            self.last_mouse_pos = self.session.mouse.getPos()[0]/self.session.settings['interface']['mouse_multiplier']

        elif self.phase == self.response_phase:

            if not hasattr(self, 'response_onset'):
                current_mouse_pos = self.session.mouse.getPos()[0]/self.session.settings['interface']['mouse_multiplier']
                if np.abs(self.last_mouse_pos - current_mouse_pos) > 0.0:
                    marker_position = response_slider.mouseToMarkerPosition(current_mouse_pos)
                    response_slider.setMarkerPosition(marker_position)
                    self.last_mouse_pos  = current_mouse_pos
                    response_slider.show_marker = True
                
                if self.session.mouse.getPressed()[0]:
                    self.response_onset = self.session.clock.getTime()
                    self.parameters['response_time'] = self.response_onset - self.session.global_log.iloc[-1]['onset']
                    self.parameters['response'] = response_slider.marker_position
                    print(f"Recorded response: {self.parameters['response']:.2f}")

                    if response_slider.marker_position == self.parameters['value']:
                        self.feedback_text.color = ( -1, 1, -1)  # green
                    else:
                        self.feedback_text.color = (1, -1, -1)  # red

                    self.stop_phase()



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('subject', type=str, help='Subject identifier')
    argparser.add_argument('session', type=int, help='Session number')
    argparser.add_argument('mapping', type=str, choices=['linear', 'cdf', 'inverse_cdf'], help='Mapping type')
    argparser.add_argument('--settings', type=str, default='default', help='Name of settings file (default by default)')
    argparser.add_argument('--n_trials', type=int, default=None, help='Number of trials')
    args = argparser.parse_args()

    session = TrainingSession(subject=args.subject,
                             run=args.session,
                              mapping=args.mapping,
                              output_str=f'sub-{args.subject}_ses-{args.session:02d}_task-training.{args.mapping}',
                              output_dir=Path(__file__).parent / 'logs' / f'sub-{args.subject}' / f'session-{args.session:02d}',
                              settings_file=Path(__file__).parent / 'settings' / f'{args.settings}.yml')
    
    session.create_trials(n_trials=args.n_trials)
    session.run()