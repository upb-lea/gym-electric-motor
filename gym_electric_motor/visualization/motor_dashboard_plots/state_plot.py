import numpy as np

from .base_plots import TimePlot


class StatePlot(TimePlot):
    """Plot to display the environments states and their references."""

    _default_limit_line_cfg = {
        'color': 'red',
        'linestyle': '--',
        'linewidth': .75
    }

    # Labels for each state variable.
    state_labels = {
        'omega': r'$\omega$/(1/s)',
        'torque': '$T$/Nm',
        'i': '$i$/A',
        'i_a': '$i_{a}$/A',
        'i_e': '$i_{e}$/A',
        'i_b': '$i_{b}$/A',
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

    @property
    def state(self):
        return self._state

    def __init__(self, state):
        """
        Args:
            state(str): Name of the state to plot
        """
        super().__init__()

        self._state_line_config = self._default_time_line_cfg.copy()
        self._ref_line_config = self._default_time_line_cfg.copy()
        self._limit_line_config = self._default_limit_line_cfg.copy()

        #: State space of the plotted variable
        self._state_space = None
        #: State name of the plotted variable
        self._state = state
        #: Index in the state array of the plotted variable
        self._state_idx = None
        #: Maximal value of the plotted variable
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
        # Save the index of the state.
        self._state_idx = ps.state_positions[self._state]
        # The maximal values of the state.
        self._limits = ps.limits[self._state_idx]
        self._state_space = ps.state_space.low[self._state_idx], ps.state_space.high[self._state_idx]
        # Bool: if the state is referenced.
        self._referenced = rg.referenced_states[self._state_idx]
        # Bool: if the data is already normalized to an interval of [-1, 1]
        self._normalized = self._limits != self._state_space[1]
        self.reset_data()

        min_limit = self._limits * self._state_space[0] if self._normalized else self._state_space[0]
        max_limit = self._limits * self._state_space[1] if self._normalized else self._state_space[1]
        spacing = 0.1 * (max_limit - min_limit)

        # Set the y-axis limits to fixed initital values
        self._y_lim = (min_limit - spacing, max_limit + spacing)

        # Set the y-axis label
        self._label = self.state_labels.get(self._state, self._state)

    def reset_data(self):
        super().reset_data()
        # Initialize the data containers
        self._state_data = np.full(shape=self._x_data.shape, fill_value=np.nan)
        self._ref_data = np.full(shape=self._x_data.shape, fill_value=np.nan)

    def initialize(self, axis):
        # Docstring of superclass
        super().initialize(axis)

        # Line to plot the state data
        self._state_line, = self._axis.plot(self._x_data, self._state_data, **self._state_line_config)
        self._lines = [self._state_line]

        # If the state is referenced plot also the reference line
        if self._referenced:
            self._reference_line, = self._axis.plot(self._x_data, self._ref_data, **self._ref_line_config)
            # Plot state line in front
            axis.lines = axis.lines[::-1]
            self._lines.append(self._reference_line)
        min_limit = self._limits * self._state_space[0] if self._normalized else self._state_space[0]
        max_limit = self._limits * self._state_space[1] if self._normalized else self._state_space[1]
        if self._state_space[0] < 0:
            self._axis.axhline(min_limit, **self._limit_line_config)
        lim = self._axis.axhline(max_limit, **self._limit_line_config)

        y_label = self._label
        unit_split = y_label.find('/')
        if unit_split == -1:
            unit_split = len(y_label)
        limit_label = y_label[:unit_split] + r'$_{\mathrm{max}}$' + y_label[unit_split:]

        if self._referenced:
            ref_label = y_label[:unit_split] + r'$^*$' + y_label[unit_split:]
            self._axis.legend(
                (self._state_line, self._reference_line, lim), (y_label, ref_label, limit_label), loc='upper left',
                numpoints=20
            )
        else:
            self._axis.legend((self._state_line, lim), (y_label, limit_label), loc='upper left', numpoints=20)

        self._y_data = [self._state_data, self._ref_data]

    def on_step_end(self, k, state, reference, reward, done):
        super().on_step_end(k, state, reference, reward, done)
        # Write the data to the data containers
        state_ = state[self._state_idx]
        ref = reference[self._state_idx]
        idx = self.data_idx
        self._x_data[idx] = self._t
        self._state_data[idx] = state_ * self._limits if self._normalized else state_
        if self._referenced:
            self._ref_data[idx] = ref * self._limits if self._normalized else ref
