import numpy as np
import matplotlib.lines as lin


class MotorDashboardPlot:

    def __init__(self):
        self._axis = None

    def initialize(self, axis):
        self._axis = axis

    def set_modules(self, ps, rg, rf):
        pass

    def step(self, k, state, reference, action, reward, done):
        raise NotImplementedError

    def update(self):
        pass

    def reset(self):
        pass


class StatePlot(MotorDashboardPlot):

    x_width = 3
    mode = 'continuous'
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
        self._state_space = None
        self._state = state
        self._state_idx = None
        self._limits = None
        self._referenced = None
        self._state_line = None
        self._reference_line = None
        self._state_data = []
        self._ref_data = []
        self._t_data = []
        self._tau = None
        self._normalized = True
        self._x_points = None
        self._t = 0
        super().__init__()

    def set_modules(self, ps, rg, rf):
        self._state_idx = ps.state_positions[self._state]
        self._limits = ps.limits[self._state_idx]
        self._state_space = ps.state_space.low[self._state_idx], ps.state_space.high[self._state_idx]
        self._referenced = rg.referenced_states[self._state_idx]
        self._tau = ps.tau
        self._x_points = int(self.x_width / self._tau)
        if self._limits == self._state_space[1]:
            self._normalized = False

    def initialize(self, axis):
        super().initialize(axis)
        axis.grid()
        if self._referenced:
            self._reference_line, = self._axis.plot(self._t_data, self._ref_data, **self.reference_line_cfg)
        self._state_line, = self._axis.plot(self._t_data, self._state_data, **self.state_line_cfg)
        min_limit = self._limits * self._state_space[0] if self._normalized else self._state_space[0]
        max_limit = self._limits * self._state_space[1] if self._normalized else self._state_space[1]

        if self._state_space[0] < 0:
            self._axis.axhline(min_limit, **self.limit_line_cfg)
        lim = self._axis.axhline(max_limit, **self.limit_line_cfg)

        self._axis.set_ylim(min_limit - 0.1 * (max_limit - min_limit), max_limit + 0.1*(max_limit - min_limit))
        self._axis.set_xlim(0, self.x_width)
        self._axis.set_xlabel('t/s')
        y_label = self.state_labels.get(self._state, self._state)
        self._axis.set_ylabel(y_label)

        self._t_data = np.linspace(0, self.x_width, self._x_points, endpoint=False).tolist()
        self._state_data = np.array([0.0] * self._x_points)
        self._ref_data = np.array([0.0] * self._x_points)
        dummy_state_line = lin.Line2D([], [], color=self.state_line_cfg['color'])
        if self._referenced:
            dummy_ref_line = lin.Line2D([], [], color=self.reference_line_cfg['color'])
            self._axis.legend(
                (dummy_state_line, dummy_ref_line, lim), (y_label, y_label+'*', 'limit'), loc='upper left'
            )
        else:
            self._axis.legend((dummy_state_line, lim), (y_label, 'limit'), loc='upper left')

    def step(self, k, state, reference, action, reward, done):
        self._t += self._tau
        state_ = state[self._state_idx]
        ref = reference[self._state_idx]
        idx = int((self._t % self.x_width) / self._tau)
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
        if self.mode == 'continuous':
            x_lim = self._axis.get_xlim()
            upper_lim = max(self._t, x_lim[1])
            lower_lim = upper_lim - self.x_width
            self._axis.set_xlim(lower_lim, upper_lim)


class RewardPlot(MotorDashboardPlot):

    x_width = 3
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
        self._tau = None
        self._x_points = None
        self._t = 0
        self._t_data = None
        super().__init__()

    def initialize(self, axis):
        super().initialize(axis)
        axis.grid()
        self._t_data = np.linspace(0, self.x_width, self._x_points, endpoint=False)
        self._reward_data = np.zeros_like(self._t_data, dtype=float)
        self._reward_line, = self._axis.plot(self._t_data, self._reward_data, **self.reward_line_cfg)
        min_limit = self._reward_range[0]
        max_limit = self._reward_range[1]
        spacing = 0.1 * (max_limit - min_limit)
        self._axis.set_ylim(min_limit - spacing, max_limit + spacing)
        self._axis.set_xlim(0, self.x_width)
        self._axis.set_xlabel('t/s')
        y_label = 'reward'
        self._axis.set_ylabel(y_label)
        dummy_rew_line = lin.Line2D([], [], color=self.reward_line_cfg['color'])
        self._axis.legend((dummy_rew_line,), ('reward',), loc='upper left')

    def set_modules(self, ps, rg, rf):
        self._reward_range = rf.reward_range
        self._tau = ps.tau
        self._x_points = int(self.x_width / self._tau)

    def step(self, k, state, reference, action, reward, done):
        self._t += self._tau
        idx = int((self._t % self.x_width) / self._tau)
        if self.mode == 'continuous':
            self._t_data[idx] = self._t
        self._reward_data[idx] = reward
        if done:
            self._axis.axvline(self._t, color='red', linewidth=1)

    def update(self):
        self._reward_line.set_data(self._t_data, self._reward_data)
        if self.mode == 'continuous':
            x_lim = self._axis.get_xlim()
            upper_lim = max(self._t, x_lim[1])
            lower_lim = upper_lim - self.x_width
            self._axis.set_xlim(lower_lim, upper_lim)


class ActionPlot(MotorDashboardPlot):
    def __init__(self, plot):
        super().__init__()
        self._plot = plot

    def step(self, k, state, reference, action, reward, done):
        pass


class MeanEpisodeRewardPlot(MotorDashboardPlot):
    def step(self, k, state, reference, action, reward, done):
        pass
