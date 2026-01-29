import numpy as np
from psychopy import visual

class AnnulusGrating:
    def __init__(self, win, size, clock, hole_deg=1.0, sf=2, ori=45, tf=1.0, contrast=0.1, drift_direction=None, **kwargs):
        self.win = win
        self.size = size
        self.hole_deg = hole_deg
        self.sf = sf
        self.ori = ori
        self.tf = tf
        self.contrast = contrast
        self.clock = clock
        self.drift_direction = np.random.choice([-1, 1]) if drift_direction is None else drift_direction

        # Create outer and inner masks
        outer_radius = size / 2
        inner_radius = hole_deg / 2

        # Outer mask: full circle
        outer_mask = visual.filters.makeMask(
            matrixSize=256,
            shape='raisedCosine',
            fringeWidth=.5/7.5,
            radius=1.0,  # Normalized to 1.0 (full radius)
        )

        # Inner mask: circle for the hole (scaled to match the outer mask)
        inner_mask = visual.filters.makeMask(
            matrixSize=256,
            shape='raisedCosine',
            fringeWidth=.5 / 7.5 / (inner_radius / outer_radius),
            radius=inner_radius / outer_radius,  # Normalized inner radius
        )

        # Combine masks: outer_mask - inner_mask = annulus
        annulus_mask = outer_mask * -inner_mask

        # Outer grating with the combined annulus mask
        self.outer_stim = visual.GratingStim(
            win,
            tex='sin',
            mask=annulus_mask,
            size=size,
            sf=sf,
            ori=ori,
            contrast=contrast,
            interpolate=False,  # Crisp edges
            **kwargs
        )

    def draw(self):
        """Draw the annulus grating with updated phase for drifting."""
        self.outer_stim.phase = self.drift_direction * self.clock.getTime() * self.tf
        self.outer_stim.draw()

    def set_ori(self, ori):
        """Set the orientation of the grating."""
        self.outer_stim.ori = ori

    def set_sf(self, sf):
        """Set the spatial frequency of the grating."""
        self.outer_stim.sf = sf

    def set_tf(self, tf):
        """Set the temporal frequency of the grating."""
        self.tf = tf



class FixationCross:
    def __init__(self, win, size=0.5, color='red', line_width=2, **kwargs):
        """
        Initialize a fixation cross stimulus.

        Parameters:
            win: PsychoPy window
            size: Size of the cross arms (degrees)
            color: Color of the cross (default: red)
            line_width: Width of the cross lines (pixels)
            **kwargs: Additional arguments for PsychoPy Line
        """
        self.win = win
        self.size = size
        self.color = color
        self.line_width = line_width

        # Create vertical and horizontal lines
        self.vertical = visual.Line(
            win,
            start=(0, -size/2),
            end=(0, size/2),
            lineColor=color,
            lineWidth=line_width,
            **kwargs
        )
        self.horizontal = visual.Line(
            win,
            start=(-size/2, 0),
            end=(size/2, 0),
            lineColor=color,
            lineWidth=line_width,
            **kwargs
        )

    def set_color(self, color):
        """Set the color of the fixation cross."""
        self.vertical.lineColor = color
        self.horizontal.lineColor = color 

    def draw(self):
        """Draw the fixation cross."""
        self.vertical.draw()
        self.horizontal.draw()