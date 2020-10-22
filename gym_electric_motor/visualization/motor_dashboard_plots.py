import numpy as np
import matplotlib.lines as lin
from collections import deque
from gym.spaces import Box


class MotorDashboardPlot:
    """Base Plot class that all plots in the MotorDashboard have to derive from."""

    def __init__(self):
        self._axis = None
        self._tau = None
        self._x_points = None
        self._t = 0
        self._t_data = []

    def initialize(self, axis):
        """Initialization of the plot. Set labels, legends,... here.

        Args:
            axis(matplotlib.pyplot.axis): Axis to plot in
        """
        self._axis = axis
        self._axis.grid(True)
        self._axis.set_xlim(-self.x_width, 0)

    def set_modules(self, ps, rg, rf):
        """Interconnection of the environments modules.
        Save all relevant information from other modules here (e.g. state_names, references,...)
        Args:
            ps(PhysicalSystem): The PhysicalSystem of the environment
            rg(ReferenceGenerator): The ReferenceGenerator of the environment.
            rf(RewardFunction): The RewardFunction of the environment
        """
        self._tau = ps.tau
        # Number of points on the x-axis in a plot (= x_width / tau)
        self._x_points = int(self.x_width / self._tau)

    def step(self, k, state, reference, action, reward, done):
        """Passing of current environmental information..

        Args:
            k(int): Current episode step.
            state(ndarray(float)): State of the system
            reference(ndarray(float)): Reference array of the system
            action(ndarray(float)): Last taken action. (None after reset)
            reward(ndarray(float)): Last received reward. (None after reset)
            done(bool): Flag if the current state is terminal
        """
        raise NotImplementedError

    def update(self):
        """Called by the MotorDashboard each time before the figure is updated."""
        # configure x-axis properties
        if self.mode == 'continuous':
            x_lim = self._axis.get_xlim()
            upper_lim = max(self._t, x_lim[1])
            lower_lim = upper_lim - self.x_width
            self._axis.set_xlim(lower_lim, upper_lim)

    def reset(self):
        """Called by the MotorDashboard each time the environment is reset."""
        pass


class StatePlot(MotorDashboardPlot):
    """Class to plot any motor state and its reference."""

    # Width of the Plot in seconds
    x_width = 1

    # Either "continuous" or "repeating"
    mode = 'continuous'

    # Configurations of the lines
    state_line_cfg = {
        'color': 'blue',
        'linestyle': '',
        'linewidth': 0.75,
        'marker': '.',
        'markersize': .5
    }
    reference_line_cfg = {
        'color': 'green',
        'linewidth': 0.75,
        'linestyle': '',
        'marker': '.',
        'markersize': .5
    }
    limit_line_cfg = {
        'color': 'red',
        'linestyle': '--',
        'linewidth': 1
    }

    # Labels for each state variable.
    state_labels = {
        'omega': r'$\omega/(rad/s)$',
        'torque': '$T/Nm$',
        'i': '$i/A$',
        'i_a': '$i_{a}/A$',
        'i_e': '$i_{e}/A$',
        'i_b': '$i_{b}/A$',
        'i_c': '$i_{c}/A$',
        'i_sq': '$i_{sq}/A$',
        'i_sd': '$i_{sd}/A$',
        'u': '$u/V$',
        'u_a': '$u_{a}/V$',
        'u_b': '$u_{b}/V$',
        'u_c': '$u_{c}/V$',
        'u_sq': '$u_{sq}/V$',
        'u_sd': '$u_{sd}/V$',
        'u_e': '$u_{e}/V$',
        'u_sup': '$u_{sup}/V$',
        'epsilon': r'$\epsilon/rad$'
    }

    def __init__(self, state):
        """
        Args:
            state(str): Name of the state to plot
        """
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
        super().__init__()

    def set_modules(self, ps, rg, rf):
        # Docstring of superclass
        super(StatePlot, self).set_modules(ps, rg, rf)
        self._state_idx = ps.state_positions[self._state]
        self._limits = ps.limits[self._state_idx]
        self._state_space = ps.state_space.low[self._state_idx], ps.state_space.high[self._state_idx]
        self._referenced = rg.referenced_states[self._state_idx]
        if self._limits == self._state_space[1]:
            self._normalized = False

    def initialize(self, axis):
        super().initialize(axis)
        if self._referenced:
            self._reference_line, = self._axis.plot(self._t_data, self._ref_data, **self.reference_line_cfg)
        self._state_line, = self._axis.plot(self._t_data, self._state_data, **self.state_line_cfg)
        min_limit = self._limits * self._state_space[0] if self._normalized else self._state_space[0]
        max_limit = self._limits * self._state_space[1] if self._normalized else self._state_space[1]

        if self._state_space[0] < 0:
            self._axis.axhline(min_limit, **self.limit_line_cfg)
        lim = self._axis.axhline(max_limit, **self.limit_line_cfg)

        self._axis.set_ylim(min_limit - 0.1 * (max_limit - min_limit), max_limit + 0.1 * (max_limit - min_limit))
        y_label = self.state_labels.get(self._state, self._state)
        self._axis.set_ylabel(y_label)

        self._t_data = np.linspace(0, self.x_width, self._x_points, endpoint=False).tolist()
        self._state_data = np.array([0.0] * self._x_points)
        self._ref_data = np.array([0.0] * self._x_points)
        dummy_state_line = lin.Line2D([], [], color=self.state_line_cfg['color'])
        if self._referenced:
            dummy_ref_line = lin.Line2D([], [], color=self.reference_line_cfg['color'])
            self._axis.legend(
                (dummy_state_line, dummy_ref_line, lim), (y_label, y_label + '*', 'limit'), loc='upper left'
            )
        else:
            self._axis.legend((dummy_state_line, lim), (y_label, 'limit'), loc='upper left')

    def step(self, k, state, reference, action, reward, done):
        self._t += self._tau
        state_ = state[self._state_idx]
        ref = reference[self._state_idx]
        idx = int((self._t % self.x_width) / self._tau) - 1
        if self.mode == 'continuous':
            self._t_data[idx] = self._t
        self._state_data[idx] = state_
        if self._referenced:
            self._ref_data[idx] = ref
        if done:
            self._axis.axvline(self._t, color='red', linewidth=1)

    def update(self):
        state_data = self._state_data
        ref_data = self._ref_data

        if self._normalized:
            state_data = state_data * self._limits
            if self._referenced:
                ref_data = ref_data * self._limits
        if self._referenced:
            self._reference_line.set_data(self._t_data, ref_data)
        self._state_line.set_data(self._t_data, state_data)

        super(StatePlot, self).update()


class RewardPlot(MotorDashboardPlot):
    """ Class used to plot the instantaneous reward during the episode
    """

    x_width = 1
    mode = 'continuous'
    reward_line_cfg = {
        'color': 'gray',
        'linestyle': '',
        'linewidth': 0.75,
        'marker': '.',
        'markersize': .5
    }

    def __init__(self):
        self._reward_range = None
        self._reward_line = None
        self._reward_data = None
        super().__init__()

    def initialize(self, axis):
        super().initialize(axis)
        self._t_data = np.linspace(0, self.x_width, self._x_points, endpoint=False)
        self._reward_data = np.zeros_like(self._t_data, dtype=float)
        self._reward_line, = self._axis.plot(self._t_data, self._reward_data, **self.reward_line_cfg)
        min_limit = self._reward_range[0]
        max_limit = self._reward_range[1]
        spacing = 0.1 * (max_limit - min_limit)
        self._axis.set_ylim(min_limit - spacing, max_limit + spacing)
        y_label = 'reward'
        self._axis.set_ylabel(y_label)
        # adds a constant line at 0 which is eventually updated by the plot variable values. legend can be set here.
        dummy_rew_line = lin.Line2D([], [], color=self.reward_line_cfg['color'])
        self._axis.legend((dummy_rew_line,), ('reward',), loc='upper left')

    def set_modules(self, ps, rg, rf):
        super(RewardPlot, self).set_modules(ps, rg, rf)
        self._reward_range = rf.reward_range

    def step(self, k, state, reference, action, reward, done):
        self._t += self._tau
        idx = int((self._t % self.x_width) / self._tau) - 1
        if self.mode == 'continuous':
            self._t_data[idx] = self._t
        self._reward_data[idx] = reward
        if done:
            self._axis.axvline(self._t, color='red', linewidth=1)

    def update(self):
        self._reward_line.set_data(self._t_data, self._reward_data)
        super(RewardPlot, self).update()


class ActionPlot(MotorDashboardPlot):
    """ Class to plot the instantaneous actions applied on-to the environment
    """

    x_width = 1
    mode = 'continuous'
    action_line_cfg = {
        'color': 'magenta',
        'linestyle': '',
        'linewidth': 0.75,
        'marker': '.',
        'markersize': .5
    }

    def __init__(self, action):
        super().__init__()
        # the action space of the environment. can be Box() or Discrete()
        self._action_space = None
        self._action = action
        # the index for the actions.
        self._action_idx = None
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
        self._t_data = np.linspace(0, self.x_width, self._x_points, endpoint=False)
        self._action_data = np.zeros_like(self._t_data, dtype=float)
        self._action_line, = self._axis.plot(self._t_data, self._action_data, **self.action_line_cfg)
        # set the layout of the subplot
        act_min = self._action_range_min
        act_max = self._action_range_max
        spacing = (act_max - act_min) * 0.1
        self._axis.set_ylim(act_min - spacing, act_max + spacing)
        self._axis.set_ylabel(self._action)
        base_action_line = lin.Line2D([], [], color=self.action_line_cfg['color'])
        self._axis.legend((base_action_line,), (self._action,), loc='upper left')

    def set_modules(self, ps, rg, rf):
        super(ActionPlot, self).set_modules(ps, rg, rf)
        # fetch the action space from the physical system
        self._action_space = ps.action_space
        # extract the action index from the action name
        self._action_idx = int(self._action.split('_')[1])
        # check for the type of action space: Discrete or Continuous
        if type(self._action_space) is Box:  # for continuous action space
            self._action_type = 'Continuous'
            # fetch the action range of continuous type actions
            self._action_range_min = self._action_space.low[self._action_idx]
            self._action_range_max = self._action_space.high[self._action_idx]

        else:
            self._action_type = 'Discrete'
            # lower bound of discrete action = 0
            self._action_range_min = 0
            # fetch the action range of discrete type actions
            self._action_range_max = self._action_space.n

    def step(self, k, state, reference, action, reward, done):
        self._t += self._tau
        idx = int((self._t % self.x_width) / self._tau) - 1
        if self.mode == 'continuous':
            self._t_data[idx] = self._t
        # the first action at the start of the simulation is None. Add a check.
        if action is not None:
            if self._action_type == 'Discrete':
                self._action_data[idx] = action
            else:
                self._action_data[idx] = action[self._action_idx]

        if done:
            self._axis.axvline(self._t, color='red', linewidth=1)

    def update(self):
        self._action_line.set_data(self._t_data, self._action_data)
        super(ActionPlot, self).update()


class EpisodeBasedPlot:
    """Base Plot class that all episode based plots ."""

    def __init__(self):
        pass

    def initialize(self, axis):
        """Initialization of the plot. Set labels, legends,... here.

        Args:
            axis(matplotlib.pyplot.axis): Axis to plot in
        """
        pass

    def set_modules(self, ps, rg, rf):
        """Interconnection of the environments modules.
        Save all relevant information from other modules here (e.g. state_names, references,...)
        Args:
            ps(PhysicalSystem): The PhysicalSystem of the environment
            rg(ReferenceGenerator): The ReferenceGenerator of the environment.
            rf(RewardFunction): The RewardFunction of the environment
        """
        pass

    def step(self, k, state, reference, action, reward, done):
        """Passing of current environmental information..

        Args:
            k(int): Current episode step.
            state(ndarray(float)): State of the system
            reference(ndarray(float)): Reference array of the system
            action(ndarray(float)): Last taken action. (None after reset)
            reward(ndarray(float)): Last received reward. (None after reset)
            done(bool): Flag if the current state is terminal
        """
        raise NotImplementedError

    def update(self):
        """Called by the MotorDashboard each time before the figure is updated."""
        pass

    def reset(self):
        """Called by the MotorDashboard each time the environment is reset."""
        pass


class MeanEpisodeRewardPlot(EpisodeBasedPlot):
    """
    class to plot the mean episode reward
    """
    x_width = 100
    mode = 'continuous'
    reward_line_cfg = {
        'color': 'blue',
        'linestyle': '-',
        'linewidth': 0.75,
        'marker': 'o',
        'markersize': 1,

    }

    def __init__(self):
        super().__init__()
        # range of the rewards
        self._reward_range = None
        self._reward_line = None
        # data container for mean reward
        self._reward_data = None
        self._reward_sum = 0
        self._episode_length = 0
        self.episode_count = 0
        # data container for x-axis(episode count)
        self.x = None
        self._axis = None

    def initialize(self, axis):
        """
            Args:
             axis (object): the subplot axis for plotting the action variable
        """

        self._axis = axis
        self._axis.grid(True)
        # create empty deques for x and y data
        self.x = deque(np.zeros(self.x_width), self.x_width)
        self._reward_data = deque(np.zeros(self.x_width), self.x_width)
        self._reward_line, = self._axis.plot(self.x, self._reward_data, **self.reward_line_cfg)
        min_limit = self._reward_range[0]
        max_limit = self._reward_range[1]
        spacing = 0.1 * (max_limit - min_limit)
        self._axis.set_xlim(0, self.x_width)
        self._axis.set_ylim(min_limit - spacing, max_limit + spacing)
        self._axis.set_ylabel('mean episodic reward')

    def set_modules(self, ps, rg, rf):
        # fetch reward range from reward function module
        self._reward_range = rf.reward_range

    def step(self, k, state, reference, action, reward, done):

        self._reward_sum += reward
        self._episode_length = k

    def update(self):

        self.x.append(self.episode_count)
        self._reward_data.append(self._reward_sum / self._episode_length)
        # plot the data on the canvas
        self._reward_line.set_data(self.x, self._reward_data)
        # configure x-axis properties
        if self.mode == 'continuous':
            self._axis.set_xlim(max(0, self.episode_count - self.x_width), self.episode_count + 0.2 * self.x_width)

    def reset(self):
        # end of episode is identified via reset
        if self._episode_length > 0:
            self.update()

        self.episode_count += 1
        self._reward_sum = 0
