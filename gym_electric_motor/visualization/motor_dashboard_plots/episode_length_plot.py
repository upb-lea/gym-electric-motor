from .base_plots import EpisodePlot


class EpisodeLengthPlot(EpisodePlot):
    """Plot to display the lengths of all episodes."""

    def __init__(self):
        super().__init__()
        # data container for episode lengths
        self._episode_lengths = []
        self._episode_length = 0
        self._label = 'Episode Length'
        self._axis = None
        # Flag, that is true, if an episode has ended before the rendering.
        self._reset = False
        self._ymax = 1.0
        self._update_lim = False
        self._y_data.append(self._episode_lengths)

    def initialize(self, axis):
        super().initialize(axis)
        self._lines.append(self._axis.plot([], [], color=self._colors[0])[0])

    def on_step_end(self, k, state, reference, reward, done):
        self._episode_length = k

    def reset_data(self):
        super().reset_data()
        self._episode_lengths = []
        self._ymax = 1
        self._update_lim = True
        self._y_data.append(self._episode_lengths)

    def _set_y_data(self):
        self._episode_lengths.append(self._episode_length)
        if self._ymax < self._episode_length:
            self._ymax = self._episode_length
            self._update_lim = True
        self._episode_length = 0

    def _scale_y_axis(self):
        if self._update_lim:
            self._axis.set_ylim(-0.1 * self._ymax, 1.1 * self._ymax)
