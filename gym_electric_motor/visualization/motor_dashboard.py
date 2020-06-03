from ..core import ElectricMotorVisualization
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import warnings


class MotorDashboard(ElectricMotorVisualization):

    """
    Visualization of the variables of the motor in graphs. This can be the angular velocity omega, torque
    phase voltages, phase currents, supply voltage and the rotor angle at the PMSM.

    In the table are all variables given that can be shown for a specific motor.

    +--------+-------+-------+--------+-------+-------+--------------------------------------------+
    |variable| extex | shunt | permex |series | PMSM, | Description                                |
    |        |       |       |        |       | SynRM |                                            |
    +========+=======+=======+========+=======+=======+============================================+
    |omega   | x     | x     | x      | x     | x     | mechanical angular velocity                |
    +--------+-------+-------+--------+-------+-------+--------------------------------------------+
    |torque  | x     | x     | x      | x     | x     | Motor generated torque                     |
    +--------+-------+-------+--------+-------+-------+--------------------------------------------+
    |u_sup   | x     | x     | x      | x     | x     | supply voltage of the converter            |
    +--------+-------+-------+--------+-------+-------+--------------------------------------------+
    |i       |       |       | x      | x     |       | current in permex and series motors        |
    +--------+-------+-------+--------+-------+-------+--------------------------------------------+
    |i_e     | x     | x     |        |       |       | current in excitation circuit              |
    +--------+-------+-------+--------+-------+-------+--------------------------------------------+
    |u       |       | x     | x      | x     |       | voltage at shunt, permex and series motors |
    +--------+-------+-------+--------+-------+-------+--------------------------------------------+
    |u_e     | x     |       |        |       |       | voltage at the excitation circuit          |
    +--------+-------+-------+--------+-------+-------+--------------------------------------------+
    |u_a     | x     |       |        |       | x     | armature voltage or voltage of phase a     |
    +--------+-------+-------+--------+-------+-------+--------------------------------------------+
    |u_b     |       |       |        |       | x     | voltage of phase b                         |
    +--------+-------+-------+--------+-------+-------+--------------------------------------------+
    |u_c     |       |       |        |       | x     | voltage of phase c                         |
    +--------+-------+-------+--------+-------+-------+--------------------------------------------+
    |u_d     |       |       |        |       | x     | d-axis voltage                             |
    +--------+-------+-------+--------+-------+-------+--------------------------------------------+
    |u_q     |       |       |        |       | x     | q-axis voltage                             |
    +--------+-------+-------+--------+-------+-------+--------------------------------------------+
    |i_a     | x     | x     |        |       | x     | armature current or current in phase a     |
    +--------+-------+-------+--------+-------+-------+--------------------------------------------+
    |i_b     |       |       |        |       | x     | current in phase a                         |
    +--------+-------+-------+--------+-------+-------+--------------------------------------------+
    |i_c     |       |       |        |       | x     | current in phase b                         |
    +--------+-------+-------+--------+-------+-------+--------------------------------------------+
    |i_d     |       |       |        |       | x     | d-axis current                             |
    +--------+-------+-------+--------+-------+-------+--------------------------------------------+
    |i_q     |       |       |        |       | x     | q-axis current                             |
    +--------+-------+-------+--------+-------+-------+--------------------------------------------+
    |epsilon |       |       |        |       | x     | electrical rotational angle                |
    +--------+-------+-------+--------+-------+-------+--------------------------------------------+


    Remark: In the current MotorDashboard version an overlap can occur in the visualization at short episodes.

    """

    def __init__(self, update_period=5e-2, visu_period=5, plotted_variables='all', **_):
        """
              Constructor of the dashboard.

              Args:
                  plotted_variables: Names of the variables that shall be shown on the dashboard
                        | Shortcut: ['all']/['none'] for all/no visualized variables
                  update_period: Number of seconds after that dashboard will be updated
                                | Updating with tiny periods lead to very low speed.
                  visu_period: Time period shown on the dashboard
        """

        self._update_period = update_period
        self._visu_period = visu_period
        self._plotted_variables = plotted_variables
        self._physical_system = None
        self._figure = None
        plt.ion()
        self._tau = None
        self._update_cycle = None
        self._episode_length = np.Inf

        self.dash_vars = None
        self._referenced_states = None
        self._limits = None
        self._nominal_state = None
        self._observation_space = None
        self._labels = None
        self._plotted_state_index = []
        self._k = 0
        self.initialized = False

        # If available use the Qt5 Backend and the update function to update the plot (faster)
        try:
            matplotlib.use('Qt5Agg')
        # Otherwise stick to the default backend and use the draw function (slower)
        except ImportError:
            warnings.warn('Cannot use Qt5Agg matplotlib backend. Plotting will be slower.')

    def set_modules(self, physical_system, reference_generator, _):
        self._physical_system = physical_system
        self._referenced_states = reference_generator.referenced_states

    def reset(self, reference_trajectories=None, *_, **__):
        """
        Function to call when a new episode has started

        Args:
            reference_trajectories: the references for the new episode
        """
        self._k = 0
        if not self.initialized:
            return
        if reference_trajectories is None:
            for var in self.dash_vars:
                var.reset()
        else:
            self._episode_length = reference_trajectories.shape[1]
            self._visu_period = min(self._visu_period, self._episode_length * self._tau)
            for index, var in enumerate(self.dash_vars):
                if self._referenced_states[index]:
                    var.reset(reference_trajectories[index, :])
                else:
                    var.reset()
        plt.pause(0.05)

    def step(self, state, reference=None, reward=None, done=False, *_, **__):
        """
        Function to call in every step of the environment

        Args:
            state: current state to plot
            reference: current reference in this step
            reward: current reward in this step
            done: Flag, if the episode has terminated
        """
        if not self.initialized:
            self.initialized = True
            self._update_physical_system_data()
            self._order_plotted_variables()
            self._set_up_plots()
            for var in self.dash_vars:
                var.reset()
            plt.pause(0.05)
        k = self._k
        state_denorm = state * self._limits
        if reference is not None:
            reference_denorm = reference * self._limits
            for var, data, ref in zip(
                    self.dash_vars,
                    state_denorm[self._plotted_state_index],
                    reference_denorm[self._plotted_state_index]
            ):
                var.step(data, k, ref)
        else:
            for var, data in zip(self.dash_vars, state_denorm[self._plotted_state_index]):
                var.step(data, k)

        if (k + 1) % self._update_cycle == 0 or k == self._episode_length - 1 or done:
            for dashVar in self.dash_vars:
                dashVar.scatter(k)
                self._update_figure()
            self._figure.canvas.draw_idle()
            self._figure.canvas.flush_events()
        self._k += 1

    def close(self, *_, **__):
        """
        Call this function when the environment is closed to close all windows.
        """
        plt.close(self._figure)

    def _update_physical_system_data(self):
        """
        update the limits, nominal values and sampling time
        """
        self._limits = self._physical_system.limits
        self._nominal_state = self._physical_system.nominal_state
        self._tau = self._physical_system.tau
        self._update_cycle = int(self._update_period / self._tau)
        try:
            self._labels = self._physical_system.labels
        except AttributeError:
            self._labels = {
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
        self._state_space = self._physical_system.state_space

    def _set_up_plots(self):
        """
        This function handles the setup of all plots.
        """
        plt.close()
        self._figure, axes = plt.subplots(len(self._plotted_variables))
        self._figure.subplots_adjust(wspace=0.0, hspace=0.4)

        if matplotlib.get_backend() == 'Qt5Agg':
            self._update_figure = self._figure.canvas.update
            # use for full screen
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.showMaximized()
        else:
            self._update_figure = self._figure.canvas.draw
        low = self._limits * self._state_space.low
        high = self._limits * self._state_space.high

        low_nominal = self._nominal_state * self._state_space.low
        high_nominal = self._nominal_state * self._state_space.high

        if len(self._plotted_state_index) > 1:
            self.dash_vars = np.array(
                [
                    _DashboardVariable(
                        var,
                        ax,
                        lo,
                        hi,
                        lo_nominal,
                        hi_nominal,
                        self._episode_length,
                        self._tau,
                        self._update_period,
                        self._labels[var],
                        self._visu_period,
                        with_ref
                    )
                    for var, ax, lo, hi, lo_nominal, hi_nominal, with_ref in zip(
                        self._plotted_variables, axes,
                        low[self._plotted_state_index],
                        high[self._plotted_state_index],
                        low_nominal[self._plotted_state_index],
                        high_nominal[self._plotted_state_index],
                        self._referenced_states[self._plotted_state_index]
                    )
                ]
            )
        elif len(self._plotted_state_index) == 1:
            self.dash_vars = np.array(
                [
                    _DashboardVariable(
                        self._plotted_variables,
                        axes,
                        low[self._plotted_state_index][0],
                        high[self._plotted_state_index][0],
                        low_nominal[self._plotted_state_index][0],
                        high_nominal[self._plotted_state_index][0],
                        self._episode_length,
                        self._tau,
                        self._update_period,
                        self._labels[self._plotted_variables[0]],
                        self._visu_period,
                        self._referenced_states[self._plotted_state_index])])
        else:
            self.dash_vars = np.array([])
            warnings.warn("Nothing to plot", Warning, stacklevel=2)

        self._figure.show()
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()

    def _order_plotted_variables(self):
        """
            This functions checks if the variables that should be plotted exists. The order is always as in the motor
            parameter specified.

            The value [True] is the shortcut for plotting all variables.
        """
        self._plotted_state_index = []
        if self._plotted_variables == 'all':
            self._plotted_variables = list(self._physical_system.state_names)
            self._plotted_state_index = list(range(len(self._plotted_variables)))
        elif self._plotted_variables == 'none':
            self._plotted_variables = []
            warnings.warn("No valid variables for visualization", Warning, stacklevel=2)
        else:
            temp = self._plotted_variables
            self._plotted_variables = []
            for variable in self._labels.keys():
                if variable in temp and variable in self._physical_system.state_names:
                    self._plotted_variables.append(variable)
                    self._plotted_state_index.append(self._physical_system.state_names.index(variable))
        if len(self._plotted_variables) < 1:
            warnings.warn("No valid variables for visualization", Warning, stacklevel=2)


class _DashboardVariable:
    """
        Class to manage a single axis for a state variable on the dashboard with trajectory (and reference)
    """
    def __init__(self, name, ax, min_limit, max_limit, min_nominal, max_nominal, episode_length, tau, update_period,
                 label, visu_period, referenced):
        """
        Args:
            name: Name of the state variable
            ax: the axis generated by the Dashboard
            min_limit: minimal possible value for the variable
            max_limit: maximal possible value for the variable
            min_nominal: minimal nominal value for the variable
            max_nominal: maximal nominal value for the variable
            episode_length: Episode length of the environment
            tau: Sampling time of the environment
            update_period: updating period of the dashboard in s
            label: labels of the variables in the plots
            visu_period: time period shown on the dashboard
        """
        self.name = name
        self.ax = ax
        self._tau = tau
        self.max_limit = max_limit
        self.min_limit = min_limit
        self.max_nominal = max_nominal
        self.min_nominal = min_nominal
        self.episode_length = episode_length
        self._update_period = update_period
        self.update_cycle = int(update_period / self._tau)
        self._visu_period = visu_period
        self._visu_steps = int(visu_period / self._tau)
        self._past_data_dash = 0.75
        self.points = np.zeros(self.update_cycle)
        self.reference_points = np.zeros(self.update_cycle)
        self.current_point = 0
        self._remaining_steps = None
        self._x = None
        self.point_list = []
        self.reference_list = []
        self._label = label
        self._with_ref = referenced

    def reset(self, ref=None):
        """
        Reset the axis when a new episode has started

        Args:
            ref: Reference to display for the state variable
        """
        self.ax.clear()

        if ref is not None:
            self._with_ref = True
            self.episode_length = len(ref)
            x_ref = np.linspace(0, len(ref) * self._tau, len(ref))
            line, = self.ax.plot(x_ref, ref, 'g-')
            self.ax.draw_artist(line)
            # if the given reference is shorter than the specified visualization length, the visualization length is
            # adapted
            # self._visu_period = min(self._visu_period, self.episode_length * self._tau)

        self._visu_steps = int(self._visu_period / self._tau)

        self._remaining_steps = int(self._visu_period / self._update_period * self._past_data_dash) + 1

        self.ax.set_xlim(0, self._visu_period)
        self.ax.set_ylim(self.min_limit - 0.1*(self.max_limit - self.min_limit),
                         self.max_limit + 0.1*(self.max_limit - self.min_limit))
        self.ax.set_xlabel('t/s')

        # set y ticks, display min and max for nominal and limit and two values between the nominal values
        axis_scale = np.linspace(self.min_nominal, self.max_nominal, 5)
        axis_scale = np.append(axis_scale, np.array([self.max_limit, self.min_limit, 0]))
        self.ax.set_yticks(axis_scale)

        self.ax.set_ylabel(self._label, fontsize=13)
        self.ax.draw_artist(self.ax.patch)
        self._x = np.linspace(0, self._update_period, self.update_cycle+1)

        self.plot_nominal_state()  # show nominal values on dashboard
        if 'u' == self.name[0]:
            self.ax.set_ylim(self.min_nominal - 0.1 * (self.max_nominal - self.min_nominal),
                             self.max_nominal + 0.1 * (self.max_nominal - self.min_nominal))
        else:
            self.plot_limits()  # show limits on dashboard

        self.ax.grid()
        self.point_list = []
        self.reference_list = []

    def step(self, data, k, ref=None):
        """
        Function to call every cycle in the render function of the environment

        Args:
            data: the new data point to store
            k: the current step of the environment
            ref: reference value of this step
        """

        # Store the points in an array and plot them only when the update cycle is reached or the episode ends
        self.points[k % self.update_cycle] = data
        self.reference_points[k % self.update_cycle] = ref

    def scatter(self, k):
        """
        Update function for the dashboard. Plots all the points

        Args:
            k: Current step of the environment
        """
        start = max(k * self._tau - self._past_data_dash * self._visu_period, 0)
        end = start + self._visu_period
        self.ax.set_xlim(start, end)
        if self.reference_points[0] is not None and self._with_ref:
            reference = self.ax.scatter(
                self._x[0:self.update_cycle] + (k - self.update_cycle) * self._tau, self.reference_points, s=1, c='g'
            )
            self.ax.draw_artist(reference)
            self.reference_list.append(reference)
            if len(self.reference_list) > self._remaining_steps:
                self.reference_list[0].remove()
                del self.reference_list[0]
        points = self.ax.scatter(
            self._x[0:self.update_cycle] + (k - self.update_cycle) * self._tau, self.points, s=1, c='b'
        )
        self.ax.draw_artist(points)
        self.point_list.append(points)
        if len(self.point_list) > self._remaining_steps:
            self.point_list[0].remove()
            del self.point_list[0]

    def plot_nominal_state(self):
        """
        Plots a dotted line at the nominal values
        """
        if self.min_nominal < 0:
            self.ax.axhline(self.min_nominal, color='y', linestyle=':', linewidth=2)
        self.ax.axhline(self.max_nominal, color='y', linestyle=':', linewidth=2)

    def plot_limits(self):
        """
        Plots a dashed read line at minimal and maximal values
        """
        if self.min_limit < 0:
            self.ax.axhline(self.min_limit, color='r', linestyle='--', linewidth=2)
        self.ax.axhline(self.max_limit, color='r', linestyle='--', linewidth=2)
