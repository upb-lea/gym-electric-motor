from .base_plots import EpisodePlot


class MeanEpisodeRewardPlot(EpisodePlot):
    """Class to plot the mean episode reward"""

    def __init__(self):
        super().__init__()

        # data container for mean reward
        self._reward_data = []
        self._reward_sum = 0
        self._episode_length = 0
        self._label = 'Mean Reward Per Step'

    def initialize(self, axis):
        """
            Args:
             axis (object): the subplot axis for plotting the action variable
        """

        super().initialize(axis)
        self._reward_data = []
        self._y_data.append(self._reward_data)
        self._lines.append(self._axis.plot([], self._reward_data, color=self._colors[0])[0])

    def on_step_end(self, k, state, reference, reward, done):
        super().on_step_end(k, state, reference, reward, done)
        self._reward_sum += reward
        self._episode_length = k

    def _set_y_data(self):
        self._reward_data.append(self._reward_sum / self._episode_length)
        self._reward_sum = 0
        self._episode_length = 0
        self._reset = True
