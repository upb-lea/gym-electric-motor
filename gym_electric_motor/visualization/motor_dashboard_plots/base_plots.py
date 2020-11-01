import matplotlib.pyplot as plt
import numpy as np
from gym_electric_motor.core import Callback


class MotorDashboardPlot(Callback):
    """Base Plot class that all plots in the MotorDashboard have to derive from."""

    def __init__(self):
        super().__init__()
        self._axis = None

        self._lines = []

        self._label = ''

        self._t_data = []

        self._y_data = []

        self._x_lim = None
        self._y_lim = None

        self._colors = [cycle['color'] for cycle in plt.rcParams['axes.prop_cycle']]

    def initialize(self, axis):
        """Initialization of the plot. Set labels, legends,... here.

        Args:
            axis(matplotlib.pyplot.axis): Axis to plot in
        """
        self._axis = axis
        self._axis.grid(True)
        self._axis.set_ylabel(self._label)
        self._axis.autoscale(False)
        if self._x_lim is not None:
            self._axis.set_xlim(self._x_lim)
        if self._y_lim is not None:
            self._axis.set_ylim(self._y_lim)

    def render(self):
        for line, data in zip(self._lines, self._y_data):
            line.set_data(self._t_data, data)
        self._scale_x_axis()
        self._scale_y_axis()

    def _scale_x_axis(self):
        pass

    def _scale_y_axis(self):
        pass


class TimePlot(MotorDashboardPlot):

    _default_time_line_cfg = {
        'linestyle': '',
        'marker': '.',
        'markersize': 0.25
    }

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

    @property
    def data_idx(self):
        """Returns the current index to access the time and data arrays."""
        return self._k % self._x_width

    def __init__(self, violation_line_cfg=None, reset_line_cfg=None):
        super().__init__()
        reset_line_cfg = reset_line_cfg or {}
        violation_line_cfg = violation_line_cfg or {}

        assert type(reset_line_cfg) is dict
        assert type(violation_line_cfg) is dict

        self._reset_line_cfg = self._default_reset_line_cfg.copy()
        self._reset_line_cfg.update(reset_line_cfg)

        self._violation_line_cfg = self._default_violation_line_cfg.copy()
        self._violation_line_cfg.update(violation_line_cfg)

        self._t = 0
        self._tau = None
        self._done = None
        self._x_width = 10000
        self._k = 0

    def set_width(self, width):
        """Sets the width of the plot in data points.

        Args:
            width(int > 0): The width of the plot
        """
        self._x_width = width

    def set_env(self, env):
        super().set_env(env)
        self._tau = env.physical_system.tau
        self._t_data = np.linspace(0, self._x_width * self._tau, self._x_width, endpoint=False)

    def on_reset_begin(self):
        if self._done is not None:
            if self._done:
                self._axis.axvline(self._t, **self._violation_line_cfg)
            else:
                self._axis.axvline(self._t, **self._reset_line_cfg)
        self._done = False

    def on_step_end(self, k, state, reference, reward, done):
        self._k += 1
        self._t += self._tau
        self._done = done

    def _scale_x_axis(self):
        x_lim = self._axis.get_xlim()
        upper_lim = max(self._t, x_lim[1])
        lower_lim = upper_lim - self._x_width * self._tau
        self._axis.set_xlim(lower_lim, upper_lim)


class EpisodicPlot(MotorDashboardPlot):
    """Base Plot class that all episode based plots ."""

    def __init__(self, **kwargs):
        super().__init__()
        self._episode_no = -1

    def on_reset_begin(self):
        if self._episode_no > -1:
            self._t_data.append(self._episode_no)
            self._set_y_data()
        self._episode_no += 1

    def _set_y_data(self):
        pass


class StepPlot(MotorDashboardPlot):

    def __init__(self, interval=10000, **kwargs):
        super().__init__()
        self._interval = interval
        self._k = 0

    def on_step_begin(self, k, action):
        self._k += 1
        if self._k % self._interval == 0:
            self._t_data.append(self._k)
            self._set_y_data()

    def _set_y_data(self):
        raise NotImplementedError
