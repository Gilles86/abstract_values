import numpy as np
import yaml
from pathlib import Path
from functools import lru_cache
from exptools2.core import Trial
from psychopy.visual import TextStim

@lru_cache(maxsize=None)
def load_settings():
    with open(Path(__file__).parent / 'settings' / 'default.yml') as f:
        return yaml.safe_load(f)

def get_value(orientation, mapping):

    assert mapping in ['linear', 'cdf', 'inverse_cdf'], "Mapping must be 'linear', 'cdf' or 'inverse_cdf'"

    settings = load_settings()

    orientations = settings['mappings']['orientations']
    values = settings['mappings'][mapping]

    # interpolate if necessary
    if orientation not in orientations:
        value = np.interp(orientation, orientations, values)
    else:
        value = values[orientations.index(orientation)]

    return value


class DummyWaiterTrial(Trial):
    """Waits for n_triggers MRI sync pulses before starting.

    Shows a 'Waiting for scanner' message with a live trigger counter.
    Uses a large finite duration instead of np.inf to avoid corrupting
    the session timer on exit.
    """

    def __init__(self, session, trial_nr, n_triggers):
        self.n_triggers = n_triggers
        self.n_triggers_received = 0
        super().__init__(session, trial_nr,
                         phase_durations=[3600],  # 1-hour cap; always exited via stop_phase()
                         phase_names=['dummy_scans'])

        txt_height = self.session.settings['various'].get('text_height', .75)
        txt_color = self.session.settings['various'].get('text_color', (1, 1, 1))
        self.waiting_text = TextStim(
            session.win,
            text='',
            pos=(0.0, 0.0), height=txt_height, color=txt_color)

    def _run_label(self):
        run = self.session.settings.get('run')
        n_runs = self.session.settings.get('main_task', {}).get('n_blocks')
        if run is not None and n_runs is not None:
            return f'Run {run}/{n_runs}'
        return None

    def draw(self):
        if self.n_triggers_received == 0:
            label = self._run_label()
            if label is not None:
                self.waiting_text.text = label
                self.waiting_text.draw()
                return
        self.session.fixation_stimulus.set_color((1, 1, 1))
        self.session.fixation_stimulus.draw()

    def get_events(self):
        events = super().get_events()
        for key, t in events:
            if key == self.session.mri_trigger:
                self.n_triggers_received += 1
                print(f'Dummy scan {self.n_triggers_received}/{self.n_triggers}')
                if self.n_triggers_received >= self.n_triggers:
                    self.stop_phase()
            elif key == 'c' and getattr(self.session, 'eyetracker_on', False):
                if hasattr(self.session, 'calibrate_eyetracker'):
                    print('Calibration triggered by operator (c key).')
                    self.session.calibrate_eyetracker()


class InstructionTrial(Trial):

    def __init__(self, session, trial_nr, txt, bottom_txt=None, keys=None, phase_durations=None, 
                 phase_names=None, **kwargs):

        self.keys = keys

        if phase_durations is None:
            phase_durations = [.5, np.inf]

        if phase_names is None:
            phase_names = ['instruction'] * len(phase_durations)

        super().__init__(session, trial_nr, phase_durations=phase_durations, phase_names=phase_names, **kwargs)

        txt_height = self.session.settings['various'].get('text_height', .75)
        txt_width = self.session.settings['various'].get('text_width', 60)
        txt_color = self.session.settings['various'].get('text_color', (1, 1, 1))

        self.text = TextStim(session.win, txt,
                             pos=(0.0, 0.0), height=txt_height, wrapWidth=txt_width, color=txt_color)

        if bottom_txt is None:
            bottom_txt = ""

        self.text2 = TextStim(session.win, bottom_txt, pos=(
            0.0, -6.0), height=txt_height, wrapWidth=txt_width,
            color=txt_color)

    def get_events(self):

        events = Trial.get_events(self)

        if self.keys is None:
            if events:
                self.stop_phase()
        else:
            for key, t in events:
                if key in self.keys:
                    self.stop_phase()

        if self.phase > 0:
            if self.session.mouse.getPressed()[0]:
                self.stop_phase()

    def draw(self):

        if self.session.win.mouseVisible:
            self.session.win.mouseVisible = False

        # self.session.fixation_stimulus.draw()
        self.text.draw()
        self.text2.draw()