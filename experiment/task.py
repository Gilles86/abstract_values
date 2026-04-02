from exptools2.core import Session, Trial
try:
    from exptools2.core import PylinkEyetrackerSession as _TaskBase
except ImportError:
    _TaskBase = Session
from psychopy import event
import yaml
from pathlib import Path
from psychopy.visual import GratingStim, TextStim
import numpy as np
from utils import get_value, InstructionTrial, DummyWaiterTrial
from response_slider import ResponseSlider
from stimuli import AnnulusGrating, FixationCross
import argparse

class TaskSession(_TaskBase):
    def __init__(self, subject, session, run, mapping, output_str, output_dir=None, settings_file=None, feedback=False, eyetracker_on=False):
        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file, eyetracker_on=eyetracker_on)
        if not hasattr(self, 'eyetracker_on'):
            self.eyetracker_on = False

        self.instructions_file = Path(__file__).parent / 'instructions.yml'
        with open(self.instructions_file, 'r') as f:
            self.instructions = yaml.safe_load(f)['phase_3']

        self.mouse = event.Mouse(visible=False)

        self.settings['subject'], self.settings['session'], self.settings['run'] = subject, session, run
        self.settings['mapping'] = mapping
        self.settings['feedback'] = feedback
        self.settings['range'] = [0, 42]
        self.fixation_stimulus = FixationCross(self.win, 
                                                size=self.settings['fixation_cross']['size'],
                                                color=self.settings['fixation_cross']['color'],
                                                line_width=self.settings['fixation_cross']['line_width'])

        self.too_late_stimulus = TextStim(self.win, text='Too late!', pos=(0, 0), color=(1, -1, -1), height=0.5)

        self._setup_response_slider()

    def _setup_response_slider(self):
        position_slider = (0, 0)
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
        if self.eyetracker_on and self.settings['run'] == 1:
            self.calibrate_eyetracker()

        self.start_experiment()

        if self.eyetracker_on:
            self.start_recording_eyetracker()

        for trial in self.trials:
            self.response_slider.random_init_marker()
            self.response_slider.show_marker = False
            print('Current marker slider position:', self.response_slider.marker_position)
            trial.run()

        total_reward = sum([trial.parameters.get('reward', 0.0) for trial in self.trials]) / self.settings.get('reward_scaling', 184.0)

        print(f'Total reward for this session: {total_reward:.2f} points')

        # write total_reward to a file:
        reward_file = Path(self.output_dir) / f'reward_{self.settings["subject"]}_{self.settings["session"]}_{self.settings["run"]}.txt'
        with open(reward_file, 'w') as f:
            f.write(f'{total_reward:.2f}\n')

        # sum earnings across all runs completed so far this session
        all_reward_files = sorted(Path(self.output_dir).glob(
            f'reward_{self.settings["subject"]}_{self.settings["session"]}_*.txt'))
        cumulative_reward = sum(float(rf.read_text().strip()) for rf in all_reward_files)

        reward_trial = InstructionTrial(self,
                                        trial_nr=self.n_trials + 1,
                                        txt=(f'This run: {total_reward:.2f} CHF\n\n'
                                             f'Total so far ({len(all_reward_files)}/8 runs): {cumulative_reward:.2f} CHF'),
                                        bottom_txt='',
                                        keys=['space', 'q'])
        reward_trial.run()

        self.close()

    def create_trials(self, n_trials=None):

        self.trials = []

        n_dummy_scans = self.settings.get('mri', {}).get('n_dummy_scans', 0)
        if n_dummy_scans > 0:
            self.trials.append(DummyWaiterTrial(self, trial_nr=-5, n_triggers=n_dummy_scans))
        else:
            self.trials.append(InstructionTrial(self, trial_nr=0,
                                                txt=self.instructions['instructions'].format(block_nr=self.settings['run'], n_blocks=self.settings['main_task'].get('n_blocks')),
                                                bottom_txt='Press SPACE BAR to continue.',
                                                keys=['space'],
                                                phase_durations=[np.inf],
                                                phase_names=['instruction']))

        # Randomly sample orientations:
        # 23 possible orientations
        self.orientations = self.settings['mappings']['orientations'][1:-1]
        self.orientations = self.orientations.copy() * self.settings['main_task'].get('n_repeats')
        self.n_trials = len(self.orientations) * self.settings['main_task'].get('n_repeats')

        if n_trials is not None:
            self.n_trials = n_trials
        else:
            self.n_trials = len(self.orientations) * self.settings['main_task'].get('n_repeats')

        n_trials_effective = self.n_trials

        isis = list(np.random.choice(self.settings['main_task'].get('isis'), n_trials_effective))

        np.random.shuffle(self.orientations)

        baseline_duration = self.settings['main_task'].get('baseline_duration', 0)
        pre_baseline  = self.settings['main_task'].get('pre_baseline_duration',  baseline_duration)
        post_baseline = self.settings['main_task'].get('post_baseline_duration', baseline_duration)
        wait_duration = self.settings['main_task'].get('wait_duration', 0)

        if pre_baseline > 0:
            self.trials.append(FixationTrial(self, trial_nr=-1, duration=pre_baseline))

        task_trials = []
        for i, (isi, ori) in enumerate(zip(isis, self.orientations[:n_trials_effective])):
            task_trials.append(TaskTrial(self, trial_nr=(self.settings['run']-1)*n_trials_effective + i + 1, orientation=ori, isi=isi))

        if wait_duration > 0:
            third = n_trials_effective // 3
            two_thirds = 2 * n_trials_effective // 3
            # Insert in reverse order so the earlier index is not shifted
            task_trials.insert(two_thirds, FixationTrial(self, trial_nr=-4, duration=wait_duration))
            task_trials.insert(third, FixationTrial(self, trial_nr=-3, duration=wait_duration))

        self.trials.extend(task_trials)

        if post_baseline > 0:
            self.trials.append(FixationTrial(self, trial_nr=-2, duration=post_baseline))


class FixationTrial(Trial):
    def __init__(self, session, trial_nr, duration):
        super().__init__(session, trial_nr,
                         phase_durations=[duration],
                         phase_names=['fixation'])

    def draw(self):
        self.session.fixation_stimulus.set_color((1, 1, 1))
        self.session.fixation_stimulus.draw()



class TaskTrial(Trial):
    def __init__(self, session, trial_nr, orientation=0, isi=4.0):
        phase_names = ['green_fixation',
                       'white_fixation','gabor',
                       'isi', 'response_bar', 'feedback',
                       'iti']

        self.session = session

        phase_durations = [self.session.settings['durations']['green_fixation'],
                            self.session.settings['durations']['white_fixation'],
                            self.session.settings['durations']['gabor'],
                            isi,
                            self.session.settings['durations']['response_bar'],
                            self.session.settings['durations']['feedback'],
                            self.session.settings['durations']['iti']]

        super().__init__(session, trial_nr, phase_durations, phase_names=phase_names)

        self.parameters['orientation'] = orientation
        self.parameters['value'] = get_value(self.parameters['orientation'], self.session.settings['mapping'])
        self.parameters['reward'] = 0.0
        size_grating = self.session.settings['grating']['size']

        # self.grating = GratingStim(session.win, tex='sin', mask='circle',
        #                            sf=session.settings['grating']['spatial_freq'], ori=orientation, size=size_grating)

        self.grating = AnnulusGrating(session.win,
                                      clock=self.session.clock,
                                      size=size_grating,
                                      hole_deg=self.session.settings['grating']['hole_size'],
                                      sf=self.session.settings['grating']['spatial_freq'],
                                      ori=self.parameters['orientation'],
                                      tf=self.session.settings['grating']['temporal_frequency'])

        self.response_phase = self.phase_names.index('response_bar')
        self.total_duration = np.sum(phase_durations)

        if self.session.settings.get('feedback') is True:
            self.feedback_text = TextStim(session.win, text=f'{self.parameters["value"]:.2f}',
                                    pos=(0, 2. * self.session.settings['slider']['height'] / 2.),
                                    height=.5, color=(-1, 1, -1))
    def draw(self):

        draw_fixation = True

        if self.phase_names[self.phase] == 'green_fixation':
            self.session.fixation_stimulus.set_color(( -1, 1, -1))  # green
        elif self.phase_names[self.phase] == 'white_fixation':
            self.session.fixation_stimulus.set_color((1, 1, 1))  # white
        if self.phase_names[self.phase] == 'gabor':
            self.grating.draw()

        elif self.phase_names[self.phase] == 'response_bar':
            self.session.response_slider.marker.inner_color = self.session.settings['slider'].get('color')
            self.session.response_slider.draw()
            draw_fixation = False

        elif self.phase_names[self.phase] == 'feedback':
            draw_fixation = False
            if hasattr(self, 'response_onset'):
                self.session.response_slider.marker.inner_color = self.session.settings['slider'].get('feedbackColor')
                self.session.response_slider.draw()
            else:
                self.session.too_late_stimulus.draw()
            
            if hasattr(self, 'feedback_text'):
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
                    print(e)

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

                    # implement random bid
                    precision = response_slider.precision
                    bid = round(np.random.uniform(self.session.settings['range'][0], self.session.settings['range'][1]) / precision) * precision

                    print('bid: ', bid)
                    print('Response: ', self.parameters['response'])
                    print("value: ", self.parameters['value'])
                    
                    if self.parameters['response'] > bid:
                        self.parameters['reward'] = 42 + self.parameters['value'] - bid
                    else:
                        self.parameters['reward'] = 42.0

                    print('reward: ', self.parameters['reward'])

                    time_so_far = self.session.clock.getTime() - self.start_trial
                    self.phase_durations[self.phase_names.index('iti')] = self.total_duration - time_so_far - self.phase_durations[self.phase_names.index('feedback')]
                    self.stop_phase()

                    if hasattr(self, 'feedback_text'):
                        if response_slider.marker_position == self.parameters['value']:
                            self.feedback_text.color = ( -1, 1, -1)  # green
                        else:
                            self.feedback_text.color = (1, -1, -1)  # red



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('subject', type=str, help='Subject identifier')
    argparser.add_argument('session', type=int, help='Session number')
    argparser.add_argument('run', type=int, help='Run number')
    argparser.add_argument('mapping', type=str, choices=['linear', 'cdf', 'inverse_cdf'], help='Mapping type')
    argparser.add_argument('--settings', type=str, default='default', help='Name of settings file (default by default)')
    argparser.add_argument('--feedback', action='store_true', help='Whether to provide feedback or not')
    argparser.add_argument('--n_trials', type=int, default=None, help='Number of trials')
    argparser.add_argument('--eyetracker', action='store_true', help='Enable Eyelink eyetracker')
    args = argparser.parse_args()

    session = TaskSession(subject=args.subject,
                             session=args.session,
                             run=args.run,
                              mapping=args.mapping,
                              output_str=f'sub-{args.subject.zfill(2)}_ses-{args.session}_run-{args.run:02d}_task-estimate.{args.mapping}',
                              output_dir=Path(__file__).parent / 'logs' / f'sub-{args.subject.zfill(2)}' / f'ses-{args.session}',
                              feedback=args.feedback,
                              eyetracker_on=args.eyetracker,
                              settings_file=Path(__file__).parent / 'settings' / f'{args.settings}.yml')
    session.create_trials(n_trials=args.n_trials)
    session.run()
