from .base_plots import EpisodePlot


class EpisodeLengthPlot(EpisodePlot):
    """Plot to display the lengths of all episodes."""

    def __init__(self):
        super().__init__()
        # data container for episode lengths
        self._episode_lengths = []
        self._episode_length = 0
        self._episode_no = 0
        self._label = 'Episode Length'
        self._axis = None
        # Flag, that is true, if an episode has ended before the rendering.
        self._reset = False
        self._y_data.append(self._episode_lengths)

    def initialize(self, axis):
        """
            Args:
             axis (object): the subplot axis for plotting the action variable
        """

        super().initialize(axis)
        self._lines.append(self._axis.plot([], self._episode_lengths, color=self._colors[0]))

    def on_step_end(self, k, state, reference, reward, done):
        self._episode_length = k

    def _set_y_data(self):
        self._episode_lengths.append(self._episode_length)
        self._episode_length = 0
