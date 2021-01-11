import numpy as np

from .base_plots import TimePlot


class RewardPlot(TimePlot):
    """Plot to display the instantaneous reward during the episode"""

    def __init__(self):
        super().__init__()
        self._reward_range = None
        self._reward_line = None
        self._reward_data = None
        self._reward_line_cfg = self._default_time_line_cfg.copy()
        self._reward_line_cfg['color'] = self._colors[-1]

    def initialize(self, axis):
        super().initialize(axis)
        self._reward_line, = self._axis.plot(self._x_data, self._reward_data, **self._reward_line_cfg)
        self._lines.append(self._reward_line)

    def set_env(self, env):
        super().set_env(env)
        self._reward_range = env.reward_range
        self._reward_data = np.full(shape=self._x_data.shape, fill_value=np.nan)
        self._y_data = [self._reward_data]
        min_limit = self._reward_range[0]
        max_limit = self._reward_range[1]
        spacing = 0.1 * (max_limit - min_limit)
        self._y_lim = (min_limit - spacing, max_limit + spacing)
        self._label = 'reward'

    def reset_data(self):
        super().reset_data()
        self._reward_data = np.full(shape=self._x_data.shape, fill_value=np.nan)

    def on_step_end(self, k, state, reference, reward, done):
        idx = self.data_idx
        self._x_data[idx] = self._t
        self._reward_data[idx] = reward
        super().on_step_end(k, state, reference, reward, done)
