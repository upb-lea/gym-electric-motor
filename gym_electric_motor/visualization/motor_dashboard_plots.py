import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from gym_electric_motor.core import Callback


class MotorDashboardPlot(Callback):
    """Base Plot class that all plots in the MotorDashboard have to derive from."""

    _default_line_cfg = {
        'linestyle': '',
        'linewidth': 0.75,
        'marker': '.',
        'markersize': .75
    }

    def __init__(self, line_config=None):
        super().__init__()
        self._axis = None
        line_config = line_config or {}
        assert type(line_config) is dict, f'The line_config of the plots needs to be a dict but is {type(line_config)}.'

        self._line_cfg = self._default_line_cfg.copy()
        self._line_cfg.update(line_config)
        self._lines = None
        self._labels = None
        self._colors = [cycle['color'] for cycle in plt.rcParams['axes.prop_cycle']]

    def initialize(self, axis):
        """Initialization of the plot. Set labels, legends,... here.

        Args:
            axis(matplotlib.pyplot.axis): Axis to plot in
        """
        self._axis = axis
        self._axis.grid(True)

    def render(self):
        raise NotImplementedError


class StepPlot(MotorDashboardPlot):

    _default_violation_line_cfg = {
        'color': 'red',
        'linewidth': 1,
        'linestyle': '-'
    }

    _default_reset_line_cfg = {
        'color': 'blue',
        'linewidth': 1,
        'linestyle': '-'
    }

    def __init__(self, violation_line_cfg=None, reset_line_cfg=None, **kwargs):
        super().__init__(**kwargs)
        reset_line_cfg = reset_line_cfg or {}
        violation_line_cfg = violation_line_cfg or {}

        assert type(reset_line_cfg) is dict
        assert type(violation_line_cfg) is dict

        self._reset_line_cfg = self._default_reset_line_cfg.copy()
        self._reset_line_cfg.update(reset_line_cfg)

        self._violation_line_cfg = self._default_violation_line_cfg.copy()
        self._violation_line_cfg.update(violation_line_cfg)

        self._t = 0
        self._t_data = []
        self._y_data = []
        self._tau = None
        self._done = None
        self._x_width = 10000
        self.y_lim = (-np.inf, np.inf)
        self._k = 0

    def set_width(self, width):
        self._x_width = width

    def set_env(self, env):
        super().set_env(env)
        self._tau = env.physical_system.tau
        self._t_data = np.linspace(0, self._x_width * self._tau, self._x_width)
        self._y_data = np.nan * np.ones_like(self._t_data)

    def render(self):
        """Called by the MotorDashboard each time before the figure is updated."""
        # configure x-axis properties
        x_lim = self._axis.get_xlim()
        upper_lim = max(self._t, x_lim[1])
        lower_lim = upper_lim - self._x_width * self._tau
        self._axis.set_xlim(lower_lim, upper_lim)

    def initialize(self, axis):
        super().initialize(axis)
        self._axis.set_xlim(0, self._x_width * self._tau)
        if self.y_lim == (-np.inf, np.inf):
            self._axis.autoscale(True, axis='Y')
        else:
            min_limit, max_limit = self.y_lim
            spacing = 0.1 * (max_limit - min_limit)
            self._axis.set_ylim(min_limit - spacing, max_limit + spacing)


class StatePlot(StepPlot):
    """Class to plot any motor state and its reference."""

    limit_line_cfg = {
        'color': 'red',
        'linestyle': '--',
        'linewidth': 1
    }

    # Labels for each state variable.
    state_labels = {
        'omega': r'$\omega$/(rad/s)',
        'torque': '$T$/Nm',
        'i': '$i$/A',
        'i_a': '$i_{a}$/A$',
        'i_e': '$i_{e}$/A$',
        'i_b': '$i_{b}$/A$',
        'i_c': '$i_{c}$/A',
        'i_sq': '$i_{sq}$/A',
        'i_sd': '$i_{sd}$/A',
        'u': '$u$/V',
        'u_a': '$u_{a}$/V',
        'u_b': '$u_{b}$/V',
        'u_c': '$u_{c}$/V',
        'u_sq': '$u_{sq}$/V',
        'u_sd': '$u_{sd}$/V',
        'u_e': '$u_{e}$/V',
        'u_sup': '$u_{sup}$/V',
        'epsilon': r'$\epsilon$/rad'
    }

    def __init__(self, state, limit_line_cfg=None, **kwargs):
        """
        Args:
            state(str): Name of the state to plot
        """
        super().__init__(**kwargs)
        limit_line_cfg = limit_line_cfg or {}
        assert type(limit_line_cfg) is dict

        #: State space of the plotted variable
        self._state_space = None
        # State name of the plotted variable
        self._state = state
        # Index in the state array of the plotted variable
        self._state_idx = None
        # Maximal value of the plotted variable
        self._limits = None
        # Bool: Flag if the plotted variable is referenced.
        self._referenced = None

        # matplotlib-Lines for the state and reference
        self._state_line = None
        self._reference_line = None

        # Data containers
        self._state_data = []
        self._ref_data = []

        # Flag, if the passed data is normalized
        self._normalized = True

    def set_env(self, env):
        # Docstring of superclass
        super().set_env(env)
        ps = env.physical_system
        rg = env.reference_generator
        self._state_idx = ps.state_positions[self._state]
        self._limits = ps.limits[self._state_idx]
        self._state_space = ps.state_space.low[self._state_idx], ps.state_space.high[self._state_idx]
        self._referenced = rg.referenced_states[self._state_idx]
        if self._limits == self._state_space[1]:
            self._normalized = False
        self._t_data = np.linspace(0, self._x_width * self._tau, self._x_width, endpoint=False)
        self._state_data = np.ones(self._x_width) * np.nan
        self._ref_data = np.ones(self._x_width) * np.nan

    def initialize(self, axis):
        min_limit = self._limits * self._state_space[0] if self._normalized else self._state_space[0]
        max_limit = self._limits * self._state_space[1] if self._normalized else self._state_space[1]
        self.y_lim = min_limit, max_limit
        super().initialize(axis)
        self._state_line, = self._axis.plot(self._t_data, self._state_data, **self._line_cfg)
        if self._referenced:
            self._reference_line, = self._axis.plot(self._t_data, self._ref_data, **self._line_cfg)
            # Plot state line in front
            axis.lines = axis.lines[::-1]
        if self._state_space[0] < 0:
            self._axis.axhline(min_limit, **self.limit_line_cfg)
        lim = self._axis.axhline(max_limit, **self.limit_line_cfg)

        y_label = self.state_labels.get(self._state, self._state)
        self._axis.set_ylabel(y_label)
        limit_label = y_label + r'$_{\mathrm{max}}$'
        if self._referenced:
            self._axis.legend(
                (self._state_line, self._reference_line, lim), (y_label, y_label + '*', limit_label), loc='upper left',
                numpoints=20
            )
        else:
            self._axis.legend((self._state_line, lim), (y_label, limit_label), loc='upper left', numpoints=20)

    def on_step_end(self, k, state, reference, reward, done):
        self._t += self._tau
        state_ = state[self._state_idx]
        ref = reference[self._state_idx]
        idx = int((self._t / self._tau) % self._x_width)
        self._t_data[idx] = self._t
        self._state_data[idx] = state_
        if self._referenced:
            self._ref_data[idx] = ref
        if done:
            self._done = True

    def on_reset_begin(self):
        if self._done is not None:
            if self._done:
                self._axis.axvline(self._t, **self._violation_line_cfg)
            else:
                self._axis.axvline(self._t, **self._reset_line_cfg)
        self._done = False

    def render(self):
        state_data = self._state_data
        ref_data = self._ref_data

        if self._normalized:
            state_data = state_data * self._limits
            if self._referenced:
                ref_data = ref_data * self._limits
        if self._referenced:
            self._reference_line.set_data(self._t_data, ref_data)
        self._state_line.set_data(self._t_data, state_data)

        super().render()


class RewardPlot(StepPlot):
    """ Class used to plot the instantaneous reward during the episode
    """

    def __init__(self, line_config=None):
        self._reward_range = None
        self._reward_line = None
        self._reward_data = None
        self._reward_line_cfg = self._default_line_cfg.copy()
        line_config = line_config or {}
        assert type(line_config) is dict
        self._reward_line_cfg.update(line_config)
        super().__init__()

    def initialize(self, axis):
        super().initialize(axis)
        self._reward_line, = self._axis.plot(self._t_data, self._reward_data, color=self._colors[-1],
                                             **self._reward_line_cfg)
        min_limit = self._reward_range[0]
        max_limit = self._reward_range[1]
        spacing = 0.1 * (max_limit - min_limit)
        self._axis.set_ylim(min_limit - spacing, max_limit + spacing)
        y_label = 'reward'
        self._axis.set_ylabel(y_label)

    def set_env(self, env):
        super().set_env(env)
        self._reward_range = env.reward_range
        self._t_data = np.linspace(0, self._x_width * self._tau, self._x_width, endpoint=False)
        self._reward_data = np.zeros_like(self._t_data, dtype=float) * np.nan

    def on_step_end(self, k, state, reference, reward, done):
        idx = int(self._t / self._tau) % self._x_width

        self._t_data[idx] = self._t
        self._reward_data[idx] = reward
        if done:
            self._axis.axvline(self._t, color='red', linewidth=1)
        self._t += self._tau

    def render(self):
        self._reward_line.set_data(self._t_data, self._reward_data)
        super().render()


class ActionPlot(StepPlot):
    """ Class to plot the instantaneous actions applied on the environment
    """

    def __init__(self, action):
        super().__init__()
        # the action space of the environment. can be Box() or Discrete()
        self._action_space = None
        self._action = action

        self._action_line = None
        # Data containers
        self._action_data = None
        # the range of the action values
        self._action_range_min = None
        self._action_range_max = None
        # the type of action space: Discrete or Continuous
        self._action_type = None

    def initialize(self, axis):
        """
        Args:
            axis (object): the subplot axis for plotting the action variable
        """
        super().initialize(axis)
        self._action_line, = self._axis.plot(self._t_data, self._action_data, color=self._colors[-2], **self._line_cfg)

        # set the layout of the subplot
        act_min = self._action_range_min
        act_max = self._action_range_max
        spacing = (act_max - act_min) * 0.1
        self._axis.set_ylim(act_min - spacing, act_max + spacing)
        label = f'Action {self._action}'
        self._axis.set_ylabel(label)
        self._axis.legend((self._action_line,), (label,), loc='upper left', numpoints=20)

    def set_env(self, env):
        super().set_env(env)
        ps = env.physical_system
        # fetch the action space from the physical system
        self._action_space = ps.action_space
        self._t_data = np.linspace(0, self._x_width * self._tau, self._x_width, endpoint=False)
        self._action_data = np.zeros_like(self._t_data, dtype=float)

        # check for the type of action space: Discrete or Continuous
        if type(self._action_space) is Box:  # for continuous action space
            self._action_type = 'Continuous'
            # fetch the action range of continuous type actions
            self._action_range_min = self._action_space.low[self._action]
            self._action_range_max = self._action_space.high[self._action]

        else:
            self._action_type = 'Discrete'
            # lower bound of discrete action = 0
            self._action_range_min = 0
            # fetch the action range of discrete type actions
            self._action_range_max = self._action_space.n

    def on_step_begin(self, k, action):
        self._t += self._tau
        idx = int((self._t / self._tau) % self._x_width)

        self._t_data[idx] = self._t


        if action is not None:
            if self._action_type == 'Discrete':
                self._action_data[idx] = action
            else:
                self._action_data[idx] = action[self._action]

    def render(self):
        self._action_line.set_data(self._t_data, self._action_data)
        super().render()


class EpisodicPlot(MotorDashboardPlot):
    """Base Plot class that all episode based plots ."""

    def initialize(self, axis):
        super().initialize(axis)
        axis.autoscale(True)

    def render(self):
        raise NotImplementedError


class MeanEpisodeRewardPlot(EpisodicPlot):
    """Class to plot the mean episode reward"""

    def __init__(self):
        super().__init__()

        self._reward_line = None
        # data container for mean reward
        self._reward_data = None
        self._reward_sum = 0
        self._episode_length = 0
        self._episode_count = 0

        self._axis = None
        self._reward_range = (-np.inf, np.inf)
        # Flag, that is true, if an episode has ended before the rendering.
        self._reset = False

    def initialize(self, axis):
        """
            Args:
             axis (object): the subplot axis for plotting the action variable
        """

        super().initialize(axis)
        self._reward_data = []
        self._reward_line, = self._axis.plot([], self._reward_data, color=self._colors[0])
        self._axis.set_ylabel('mean reward per episode')

    def set_env(self, env):
        # fetch reward range from reward function module
        self._reward_range = env.reward_function.reward_range

    def on_step_end(self, k, state, reference, reward, done):
        self._reward_sum += reward
        self._episode_length = k

    def on_reset_begin(self):
        if self._episode_count > 0:
            self._reward_data.append(self._reward_sum / self._episode_length)
            self._reward_sum = 0
            self._episode_length = 0
            self._reset = True
        self._episode_count += 1

    def render(self):
        if self._reset:
            # plot the data on the canvas
            self._reward_line.set_data(range(1, self._episode_count), self._reward_data)
            self._reset = False
            self._axis.set_xlim(0, self._episode_count)
            self._axis.set_ylim(min(self._reward_data), max(self._reward_data))


class EpisodeLengthPlot(EpisodicPlot):
    """
    class to plot the mean episode reward
    """

    def __init__(self):
        super().__init__()

        self._line = None
        # data container for episode lengths
        self._episode_lengths = []
        self._episode_length = 0
        self._episode_count = 0

        self._axis = None
        # Flag, that is true, if an episode has ended before the rendering.
        self._reset = False

    def initialize(self, axis):
        """
            Args:
             axis (object): the subplot axis for plotting the action variable
        """

        super().initialize(axis)
        self._line, = self._axis.plot([], self._episode_lengths, color=self._colors[0])
        self._axis.set_ylabel('Steps per Episode')

    def on_step_end(self, k, state, reference, reward, done):
        self._episode_length = k

    def on_reset_begin(self):
        if self._episode_count > 0:
            self._episode_lengths.append(self._episode_length)
            self._episode_length = 0
            self._reset = True
        self._episode_count += 1

    def render(self):
        if self._reset:
            # plot the data on the canvas
            self._line.set_data(range(1, self._episode_count), self._episode_lengths)
            self._reset = False
            self._axis.set_xlim(0, self._episode_count)
            self._axis.set_ylim(min(self._episode_lengths), max(self._episode_lengths))


class IntervalPlot(MotorDashboardPlot):

    def on_step_begin(self, k, action):
        pass

    def on_step_end(self, k, state, reference, reward, done):
        pass

    def on_reset_end(self, state, reference):
        pass

    def render(self):
        pass
