import matplotlib.pyplot as plt
import numpy as np
from gym_electric_motor.core import Callback


class MotorDashboardPlot(Callback):
    """Base Plot class that all plots in the MotorDashboard have to derive from.

    Attributes to set by subclasses:
        _axis: The matplotlib.pyplot.axis that this instance plots the lines into.
        _lines: The matplotlib lines that have to be updated.
        _label: Write the y-label of the axis here.
        _x_data(list/ndarray): Write the x-axis data for all lines here.
        _y_data(list(list/ndarray)): list of all y-axis data from all lines. Write the y-values for all lines into
            different lists here.
        _x_lim(2-tuple(float)/None): Initial limits of the x-axis. None for matplotlib-default.
        _y_lim(2-tuple(float)/None): Initial limits of the y-axis. None for matplotlib-default.
        _colors(list(matplotlib-colors)): The list of all colors for lines in the axis from the selected mpl-style.
            Choose one of these for the lines and the color will fit to the overall design.

    """

    def __init__(self):
        super().__init__()

        # The axis to plot into
        self._axis = None
        # A list of all lines in the plot
        self._lines = []
        # The y-axis Label
        self._label = ''
        # The x-axis data (common to all lines)
        self._x_data = []
        # List of all y-axis data of the lines
        self._y_data = []

        # Initial limits for the x and y axes view
        self._x_lim = None
        self._y_lim = None

        # All colors of the current matplotlib style. It is recommended to select one of these for plotting the lines.
        self._colors = [cycle['color'] for cycle in plt.rcParams['axes.prop_cycle']]

    def initialize(self, axis):
        """Initialization of the plot.

        It is called by the MotorDashboard. Set labels, legends... when overriding this method.

        Args:
            axis(matplotlib.pyplot.axis): Axis to plot in
        """
        self._lines = []
        self._axis = axis
        self._axis.grid(True)
        self._axis.set_ylabel(self._label)
        self._axis.autoscale(False)
        if self._x_lim is not None:
            self._axis.set_xlim(self._x_lim)
        if self._y_lim is not None:
            self._axis.set_ylim(self._y_lim)

    def render(self):
        """Update of the plots axis.

        The current x and y-data are written onto the respective lines in this methods. Furthermore the x- and y-axes
        are scaled dynamically."""
        for line, data in zip(self._lines, self._y_data):
            line.set_data(self._x_data, data)
        self._scale_x_axis()
        self._scale_y_axis()

    def _scale_x_axis(self):
        """Override this function to dynamically scale the plots x-axis.

        Call *self._axis.set_xlim(lower, upper)* within this method to set the x-axis boundaries.
        """
        pass

    def _scale_y_axis(self):
        """Override this function to dynamically scale the plots y-axis.

        Call *self._axis.set_ylim(lower, upper)* within this method to set the y-axis boundaries.
        """
        pass

    def reset_data(self):
        """Called by the dashboard, when the figures are reset to generate a new figure."""
        self._x_data = []
        self._y_data = []


class TimePlot(MotorDashboardPlot):
    """Base class for all MotorDashboardPlots that have the cumulative simulated time on the x-Axis.

    These use fixed-size numpy-arrays as x and y data. The plot is moved along the time axis and old data is cut out.
    Furthermore, if the environment is reset manually or a limit violation occurs, a blue or red vertical line is
    plotted to indicate these cases in the timeline.

    Attributes:
        _t(float): The cumulative simulation time.
        _k(int): The cumulative no of taken steps.
        _x_width(int): The width of the x-axis plot. (Set automatically by the dashboard)

    """

    _default_time_line_cfg = {
        'linestyle': '',
        'marker': 'o',
        'markersize': 0.75
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

    def __init__(self):
        super().__init__()

        self._reset_line_cfg = self._default_reset_line_cfg.copy()

        self._violation_line_cfg = self._default_violation_line_cfg.copy()

        self._t = 0
        self._tau = None
        self._done = None
        self._x_width = 10000
        self._k = 0
        self._reset_memory = []
        self._violation_memory = []

    def set_width(self, width):
        """Sets the width of the plot in data points.

        Args:
            width(int > 0): The width of the plot
        """
        self._x_width = width

    def set_env(self, env):
        super().set_env(env)
        self._tau = env.physical_system.tau
        self.reset_data()

    def reset_data(self):
        super().reset_data()
        self._k = 0
        self._t = 0
        self._reset_memory = []
        self._violation_memory = []
        self._x_data = np.linspace(0, self._x_width * self._tau, self._x_width, endpoint=False)
        self._x_lim = (0, self._x_data[-1])

    def on_reset_begin(self):
        # self._done is None at initial reset.
        if self._done is not None:
            if self._done:
                self._violation_memory.append(self._t)
            else:
                self._reset_memory.append(self._t)
        self._done = False

    def on_step_end(self, k, state, reference, reward, done):
        self._k += 1
        self._t += self._tau
        self._done = done

    def render(self):
        super().render()

        for violation in self._violation_memory:
            self._axis.axvline(violation, **self._violation_line_cfg)
        self._violation_memory = []

        for reset in self._reset_memory:
            self._axis.axvline(reset, **self._reset_line_cfg)
        self._reset_memory = []

    def _scale_x_axis(self):
        """The x-axis is modeled as a sliding window in this plot."""
        x_lim = self._axis.get_xlim()
        upper_lim = max(self._t, x_lim[1])
        lower_lim = upper_lim - self._x_width * self._tau
        self._axis.set_xlim(lower_lim, upper_lim)


class EpisodePlot(MotorDashboardPlot):
    """Base Plot class that all episode based plots ."""

    def __init__(self):
        super().__init__()
        self._episode_no = -1

    def on_reset_begin(self):
        if self._episode_no > -1:
            self._x_data.append(self._episode_no)
            self._set_y_data()
        self._episode_no += 1

    def _set_y_data(self):
        pass

    def reset_data(self):
        super().reset_data()
        self._y_data = []
        self._episode_no = -1

    def _scale_x_axis(self):
        if len(self._x_data) > 0 and self._axis.get_xlim() != (-1, self._x_data[-1]):
            self._axis.set_xlim(-1, self._x_data[-1])


class StepPlot(MotorDashboardPlot):

    def __init__(self):
        super().__init__()
        self._k = 0

    def on_step_begin(self, k, action):
        self._k += 1

    def reset_data(self):
        super().reset_data()
        self._y_data = []
        self._k = 0

    def _scale_x_axis(self):
        if self._axis.get_xlim() != (-1, self._x_data[-1]):
            self._axis.set_xlim(-1, self._x_data[-1])

    def _scale_y_axis(self):
        min_, max_ = min(self._y_data[0]), max(self._y_data[0])
        if self._axis.get_ylim() != (min_, max_):
            self._axis.set_ylim(min_, max_)
