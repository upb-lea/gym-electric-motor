from gym.spaces import Box, Discrete, MultiDiscrete
import numpy as np

from .base_plots import TimePlot


class ActionPlot(TimePlot):
    """ Class to plot the instantaneous actions applied on the environment"""

    def __init__(self, action=0):
        """
        Args:
            action(int): Index of the action to be plotted, if there are multiple continuous actions.
                Select 0, if the environment has a Discrete action space. Default: 0
        """
        super().__init__()

        # the action space of the environment. can be Box() or Discrete()
        self._action_space = None
        self._action = action

        # The matplotlib line object for the actions
        self._action_line = None
        # Data containers
        self._action_data = None
        # the range of the action values
        self._action_range_min = None
        self._action_range_max = None
        # the type of action space: Discrete or Continuous
        self._action_type = None

        self._action_line_config = self._default_time_line_cfg.copy()
        self._action_line_config['color'] = self._colors[-2]

    def initialize(self, axis):
        # Docstring of superclass
        super().initialize(axis)
        self._action_line, = self._axis.plot(
            self._x_data, self._action_data, **self._action_line_config,
        )
        self._lines.append(self._action_line)

    def reset_data(self):
        super().reset_data()
        self._action_data = np.full(shape=self._x_data.shape, fill_value=np.nan)
        self._y_data.append(self._action_data)

    def set_env(self, env):
        # Docstring of superclass
        super().set_env(env)
        ps = env.physical_system
        # fetch the action space from the physical system
        self._action_space = ps.action_space
        self.reset_data()

        # check for the type of action space: Discrete or Continuous
        if type(self._action_space) is Box:  # for continuous action space
            self._action_type = 'Continuous'
            # fetch the action range of continuous type actions
            self._action_range_min = self._action_space.low[self._action]
            self._action_range_max = self._action_space.high[self._action]

        elif type(self._action_space) is Discrete:
            self._action_type = 'Discrete'
            # lower bound of discrete action = 0
            self._action_range_min = 0
            # fetch the action range of discrete type actions
            self._action_range_max = self._action_space.n
        elif type(self._action_space) is MultiDiscrete:
            self._action_type = 'MultiDiscrete'
            # lower bound of discrete action = 0
            self._action_range_min = 0
            # fetch the action range of discrete type actions
            self._action_range_max = self._action_space.nvec[self._action]

        spacing = 0.1 * (self._action_range_max - self._action_range_min)
        self._y_lim = self._action_range_min - spacing, self._action_range_max + spacing
        self._label = f'Action {self._action}'

    def on_step_begin(self, k, action):
        # Docstring of superclass
        super().on_step_begin(k, action)
        idx = self.data_idx
        self._x_data[idx] = self._t

        if action is not None:
            if self._action_type == 'Discrete':
                self._action_data[idx] = action
            else:
                self._action_data[idx] = action[self._action]
