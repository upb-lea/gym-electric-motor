from matplotlib import pyplot as plt
import numpy as np


class MotorDashboard(object):
    """
    Visualization of the variables of the motor in graphs. This can be the angular velocity omega, torque
    phase voltages, phase currents, supply voltage and the rotor angle at the PMSM.

    In the table are all variables given that can be shown for a specific motor.

    +--------+-------+-------+--------+-------+-------+
    |variable| extex | shunt | permex |series | PMSM  |
    +========+=======+=======+========+=======+=======+
    |omega   | x     | x     | x      | x     | x     |
    +--------+-------+-------+--------+-------+-------+
    |torque  | x     | x     | x      | x     | x     |
    +--------+-------+-------+--------+-------+-------+
    |u_sup   | x     | x     | x      | x     | x     |
    +--------+-------+-------+--------+-------+-------+
    |i       |       |       | x      | x     |       |
    +--------+-------+-------+--------+-------+-------+
    |i_e     | x     | x     |        |       |       |
    +--------+-------+-------+--------+-------+-------+
    |u       |       | x     | x      | x     |       |
    +--------+-------+-------+--------+-------+-------+
    |u_e     | x     |       |        |       |       |
    +--------+-------+-------+--------+-------+-------+
    |u_a     | x     |       |        |       | x     |
    +--------+-------+-------+--------+-------+-------+
    |u_b     |       |       |        |       | x     |
    +--------+-------+-------+--------+-------+-------+
    |u_c     |       |       |        |       | x     |
    +--------+-------+-------+--------+-------+-------+
    |i_a     | x     | x     |        |       | x     |
    +--------+-------+-------+--------+-------+-------+
    |i_b     |       |       |        |       | x     |
    +--------+-------+-------+--------+-------+-------+
    |i_c     |       |       |        |       | x     |
    +--------+-------+-------+--------+-------+-------+
    |epsilon |       |       |        |       | x     |
    +--------+-------+-------+--------+-------+-------+


    """

    def __init__(self, variables, tau, low, high, episode_length, safety_margin, with_ref, update_cycle=None):
        """
        Constructor of the dashboard.

        Args:
            variables: Names of the variables that shall be shown on the dashboard
            tau: system time constant
            low: Array of lowest possible values for the variables in the same order as the variables
            high: Array of highest possible values for the variables in the same order as the variables
            episode_length: episode length of the system
            safety_margin: Ratio between maximal and nominal power of the motor parameters.
            with_ref: Array of Booleans indicating if the variables have a reference that has to be plotted
            update_cycle: Number of steps after which the dashboard shall be updated.
                          | Updating every cycle leads to a very low speed.

        """

        # Generate window and graphs
        self._update_cycle = update_cycle or episode_length // 20
        self._figure, axes = plt.subplots(len(variables))
        self._figure.subplots_adjust(wspace=0.0, hspace=0.4)  # Adjust distance between subplots

        try:
            self.dash_vars = np.array([_DashboardVariable(var, ax, lo, hi, episode_length, safety_margin, tau, ref,
                                                          self._update_cycle)
                                       for var, ax, lo, hi, ref in zip(variables, axes, low, high, with_ref)])
        except TypeError:
            self.dash_vars = np.array([_DashboardVariable(variables[0], axes, low, high, episode_length, safety_margin,
                                                          tau, with_ref, self._update_cycle)])
        self._tau = tau
        self._episode_length = episode_length
        self._figure.show()
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()
        self._update_figure = self._figure.canvas.update

    def reset(self, references):
        """
        Function to call when a new episode has started

        Args:
            references: the references for the new episode
        """
        for var, ref in zip(self.dash_vars, references):
            var.reset(ref)
        plt.pause(0.05)

    def step(self, state, k):
        """
        Function to call in every step of the environment

        Args:
            state: current state to plot
            k: current step of the environment
        """
        for var, data in zip(self.dash_vars, state):
            var.step(data, k)
        if (k + 1) % self._update_cycle == 0 or k == self._episode_length - 1:
            for dashVar in self.dash_vars:
                dashVar.scatter(k)
            try:
                self._update_figure()
            except:
                self._update_figure = self._figure.canvas.draw
                self._update_figure()
            self._figure.canvas.flush_events()

    def close(self):
        """
        Call this function when the environment is closed to close all windows.
        """
        plt.close(self._figure)


class _DashboardVariable:
    """
        Class to manage a single axis for a state variable on the dashboard with trajectory (and reference)
    """
    def __init__(self, name, ax, min_limit, max_limit, episode_length, safety_margin, tau, with_ref, update_cycle):
        """
        Args:
            name: Name of the state variable
            ax: the axis generated by the Dashboard
            min_limit: minimal possible value for the variable
            max_limit: maximal possible value for the variable
            episode_length: Episode length of the environment
            safety_margin: Ratio between maximal and nominal power of the motor parameters.
            tau: Time constant of the environment
            with_ref: Flag, indicating if the variable has got a reference
            update_cycle: updating cycle of the dashboard
        """
        self.name = name
        self.ax = ax
        self._tau = tau
        self._with_ref = with_ref
        self.max_limit = max_limit
        self.min_limit = min_limit
        self.episode_length = episode_length
        self.safety_margin = safety_margin
        self.update_cycle = update_cycle
        self.points = np.zeros(update_cycle)
        self.current_point = 0
        self._x = None

        self._labels = {'omega': r'$\omega$/(1/s)',
                        'torque': r'$T$/Nm',
                        'i': r'$i$/A',
                        'i_A': r'$i_A$/A',
                        'i_E': r'$i_E$/A',
                        'i_e': r'$i_e$/A',
                        'u': r'$u$/V',
                        'u_A': r' $u_A$/V',
                        'u_E': r'$u_E$/V',
                        'u_e': r'$u_e$/V',
                        'u_sup': r'$u_{{sup}}$/V',
                        'u_a': r'$u_a$/V',
                        'u_b': r'$u_b$/V',
                        'u_c': r'$u_c$/V',
                        'i_a': r'$i_a$/A',
                        'i_b': r'$i_b$/A',
                        'i_c': r'$i_c$/A',
                        'epsilon': r'$\epsilon$/rad'}

    def reset(self, ref=None, episode_length=None):
        """
        Reset the axis when a new episode has started

        Args:
            ref: Reference to display for the state variable
            episode_length: new episode length if it has changed
        """

        if episode_length is not None:
            self.episode_length = episode_length
        self.ax.clear()
        self.ax.set_xlim(0, self._tau * self.episode_length)
        self.ax.set_ylim(self.min_limit - 0.1*(self.max_limit - self.min_limit),
                         self.max_limit + 0.1*(self.max_limit - self.min_limit))
        self.ax.set_xlabel('t/s')

        if self.name in self._labels.keys():
            name = self._labels[self.name]
        else:
            name = 'Error'

        self.ax.set_ylabel(name, fontsize=13)
        self.ax.draw_artist(self.ax.patch)
        if self._with_ref:
            self._x = np.linspace(0, len(ref) * self._tau, len(ref))
            line, = self.ax.plot(self._x, ref, 'g-')
            self.ax.draw_artist(line)
        else:
            self._x = np.linspace(0, self.episode_length * self._tau, self.episode_length)

        self.plot_nominal_values()  # show nominal values on dashboard
        if 'u' == self.name[0]:
            self.ax.set_ylim(self.min_limit / self.safety_margin - 0.1 * (self.max_limit - self.min_limit)
                             / self.safety_margin,
                             self.max_limit / self.safety_margin + 0.1 * (self.max_limit - self.min_limit)
                             / self.safety_margin)
        else:
            self.plot_limits()  # show limits on dashboard
        self.ax.grid()


    def step(self, data, k):
        """
        Function to call every cycle in the render function of the environment

        Args:
            data: the new data point to store
            k: the current step of the environment
        """

        # Store the points in an array and plot them only when the update cycle is reached or the episode ends
        self.points[k % self.update_cycle] = data

    def scatter(self, k):
        """
        Update function for the dashboard. Plots all the points

        Args:
            k: Current step of the environment
        """
        points = self.ax.scatter(self._x[k+1-self.update_cycle:k+1], self.points, s=1, c='b')
        self.ax.draw_artist(points)

    def plot_nominal_values(self):
        """
        Plots a dotted line at the nominal values
        """
        self.ax.plot([0, self.episode_length * self._tau],
                     np.ones([2, 2]) * np.array([self.min_limit, self.max_limit]).transpose() / self.safety_margin,
                     color='y', linestyle=':', linewidth=2)

    def plot_limits(self):
        """
        Plots a dashed read line at minimal and maximal values
        """
        self.ax.plot([0, self.episode_length * self._tau],
                     np.ones([2, 2]) * np.array([self.min_limit, self.max_limit]).transpose(),
                     color='r', linestyle='--', linewidth=2)

