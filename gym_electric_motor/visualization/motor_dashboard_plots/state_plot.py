import numpy as np

from .base_plots import TimePlot


class StatePlot(TimePlot):
    """Class to plot any motor state and its reference."""

    _default_limit_line_cfg = {
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

    def __init__(self, state, state_line_config=None, ref_line_config=None, limit_line_config=None, **kwargs):
        """
        Args:
            state(str): Name of the state to plot
        """
        super().__init__(**kwargs)
        limit_line_config = limit_line_config or {}
        assert type(limit_line_config) is dict
        state_line_config = state_line_config or {}
        assert type(state_line_config) is dict
        ref_line_config = ref_line_config or {}
        assert type(ref_line_config) is dict

        self._state_line_config = self._default_time_line_cfg.copy()
        self._ref_line_config = self._default_time_line_cfg.copy()
        self._state_line_config.update(state_line_config)
        self._ref_line_config.update(ref_line_config)
        self._limit_line_config = self._default_limit_line_cfg
        self._limit_line_config.update(limit_line_config)

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
        self._state_data = np.ones(self._x_width) * np.nan
        self._ref_data = np.ones(self._x_width) * np.nan
        min_limit = self._limits * self._state_space[0] if self._normalized else self._state_space[0]
        max_limit = self._limits * self._state_space[1] if self._normalized else self._state_space[1]
        spacing = 0.1 * (max_limit - min_limit)
        self._y_lim = (min_limit - spacing, max_limit + spacing)
        self._label = self.state_labels.get(self._state, self._state)

    def initialize(self, axis):
        super().initialize(axis)
        self._state_line, = self._axis.plot(self._t_data, self._state_data, **self._state_line_config)
        self._lines = [self._state_line]
        if self._referenced:
            self._reference_line, = self._axis.plot(self._t_data, self._ref_data, **self._ref_line_config)
            # Plot state line in front
            axis.lines = axis.lines[::-1]
            self._lines.append(self._reference_line)
        min_limit = self._limits * self._state_space[0] if self._normalized else self._state_space[0]
        max_limit = self._limits * self._state_space[1] if self._normalized else self._state_space[1]
        if self._state_space[0] < 0:
            self._axis.axhline(min_limit, **self._limit_line_config)
        lim = self._axis.axhline(max_limit, **self._limit_line_config)

        y_label = self._label
        limit_label = y_label + r'$_{\mathrm{max}}$'

        if self._referenced:
            self._axis.legend(
                (self._state_line, self._reference_line, lim), (y_label, y_label + '*', limit_label), loc='upper left',
                numpoints=20
            )
        else:
            self._axis.legend((self._state_line, lim), (y_label, limit_label), loc='upper left', numpoints=20)

        self._y_data = [self._state_data, self._ref_data]

    def on_step_end(self, k, state, reference, reward, done):
        super().on_step_end(k, state, reference, reward, done)
        state_ = state[self._state_idx]
        ref = reference[self._state_idx]
        idx = self.data_idx
        self._t_data[idx] = self._t
        self._state_data[idx] = state_ * self._limits
        if self._referenced:
            self._ref_data[idx] = ref * self._limits
