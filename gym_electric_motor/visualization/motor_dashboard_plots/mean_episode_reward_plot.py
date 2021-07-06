import numpy as np

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
        self._reward_range = [np.inf, -np.inf]

    def initialize(self, axis):
        super().initialize(axis)
        self._reward_data = []
        self._y_data.append(self._reward_data)
        self._lines.append(self._axis.plot([], self._reward_data, color=self._colors[0])[0])

    def on_step_end(self, k, state, reference, reward, done):
        super().on_step_end(k, state, reference, reward, done)
        self._reward_sum += reward
        self._episode_length = k

    def reset_data(self):
        super().reset_data()
        self._reward_data = []
        self._reward_range = [np.inf, -np.inf]

    def _set_y_data(self):
        mean_reward = self._reward_sum / self._episode_length
        self._reward_data.append(mean_reward)
        if self._reward_range[0] > mean_reward:
            self._reward_range[0] = mean_reward
        if self._reward_range[1] < mean_reward:
            self._reward_range[1] = mean_reward
        self._reward_sum = 0
        self._episode_length = 0
        self._reset = True

    def _scale_y_axis(self):
        if len(self._reward_data) > 1 and self._axis.get_ylim() != tuple(self._reward_range):
            spacing = 0.1 * (self._reward_range[1] - self._reward_range[0])
            self._axis.set_ylim(self._reward_range[0]-spacing, self._reward_range[1]+spacing)
