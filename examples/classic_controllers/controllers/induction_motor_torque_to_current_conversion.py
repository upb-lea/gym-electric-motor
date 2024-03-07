from .pi_controller import PIController
from matplotlib import pyplot as plt
import numpy as np


class InductionMotorTorqueToCurrentConversion:
    """
    This class represents the torque controller for the cascaded control of induction motors. The torque controller
    uses LUT to find an appropriate operating point for the flux and torque.  The flux is limited by a modulation
    controller. A reference value for the i_sd current is then determined using the operating point of the flux and
    a PI controller. In addition, a reference for the i_sq current is calculated based on the current flux and the
    operating point of the torque.
    Predefined plots are available for visualization of the operating points (plot_torque: default True). Also the
    operation of the modulation controller can be plotted (plot_modulation: default False).
    Further information can be found at https://ieeexplore.ieee.org/document/7203404.
    """

    def __init__(
        self,
        environment,
        stages,
        plot_torque=True,
        plot_modulation=False,
        update_interval=1000,
    ):
        self.env = environment
        self.nominal_values = self.env.physical_system.nominal_state
        self.state_space = self.env.physical_system.state_space

        # Calculate parameters of the motor
        mp = self.env.physical_system.electrical_motor.motor_parameter
        self.l_m = mp["l_m"]
        self.l_r = self.l_m + mp["l_sigr"]
        self.l_s = self.l_m + mp["l_sigs"]
        self.r_r = mp["r_r"]
        self.r_s = mp["r_s"]
        self.p = mp["p"]
        self.tau = self.env.physical_system.tau
        tau_s = self.l_s / self.r_s

        self.i_sd_idx = self.env.state_names.index("i_sd")
        self.i_sq_idx = self.env.state_names.index("i_sq")
        self.torque_idx = self.env.state_names.index("torque")
        self.u_sa_idx = environment.state_names.index("u_sa")
        self.u_sd_idx = environment.state_names.index("u_sd")
        self.u_sq_idx = environment.state_names.index("u_sq")
        self.omega_idx = environment.state_names.index("omega")
        self.limits = self.env.physical_system.limits

        p_gain = stages[0][1]["p_gain"] * 2 * tau_s**2  # flux controller p gain
        i_gain = p_gain / self.tau  # flux controller i gain
        self.psi_controller = PIController(
            self.env, p_gain=p_gain, i_gain=i_gain
        )  # flux controller

        self.torque_count = 1001

        self.i_sd_count = 500
        self.i_sd = np.linspace(0, self.limits[self.i_sd_idx], self.i_sd_count)

        self.i_sq_count = 1001
        self.i_sq = np.linspace(0, self.limits[self.i_sq_idx], self.i_sq_count)

        self.t_maximum = self.limits[self.torque_idx]
        self.t_minimum = -self.limits[self.torque_idx]

        self.psi_opt_t = self.psi_opt()
        self.psi_max = np.max(self.psi_opt_t[1])
        self.psi_count = 1000

        self.t_max_psi = self.t_max()

        self.a_max = 1  # maximum modulation level
        self.k_ = 0.8
        d = 2  # damping of the modulation controller
        alpha = d / (d - np.sqrt(d**2 - 1))
        self.i_gain = 1 / (self.l_s / (1.25 * self.r_s)) * (alpha - 1) / alpha**2

        self.u_dc = np.sqrt(3) * self.limits[self.u_sa_idx]
        self.limited = False
        self.integrated = 0
        self.psi_high = 0.1 * self.psi_max
        self.psi_low = -self.psi_max
        self.integrated_reset = (
            0.5 * self.psi_low
        )  # Reset value of the modulation controller

        self.plot_torque = plot_torque
        self.plot_modulation = plot_modulation
        self.update_interval = update_interval
        self.k = 0

    def intitialize_torque_plot(self):
        plt.ion()

        self.fig_torque = plt.figure("Torque Controller")
        self.psi_opt_plot = plt.subplot2grid((1, 2), (0, 0))
        self.t_max_plot = plt.subplot2grid((1, 2), (0, 1))
        self.psi_opt_plot.plot(
            self.psi_opt_t[0], self.psi_opt_t[1], label="$\Psi^*_{r, opt}(T^*)$"
        )
        self.psi_opt_plot.grid()
        self.psi_opt_plot.set_xlabel("T / Nm")
        self.psi_opt_plot.set_ylabel("$\Psi$ / Vs")
        self.psi_opt_plot.legend()

        self.t_max_plot.plot(
            self.t_max_psi[1], self.t_max_psi[0], label="$T_{max}(\Psi_{max})$"
        )
        self.t_max_plot.grid()
        self.t_max_plot.set_xlabel("$\Psi$ / Vs")
        self.t_max_plot.set_ylabel("T / Nm")
        self.t_max_plot.legend()

    def psi_opt(self):
        # Calculate the optimal flux for a given torque
        psi_opt_t = []
        i_sd = np.linspace(0, self.limits[self.i_sd_idx], self.i_sd_count)
        for t in np.linspace(self.t_minimum, self.t_maximum, self.torque_count):
            if t != 0:
                i_sq = t / (3 / 2 * self.p * self.l_m**2 / self.l_r * i_sd[1:])
                pv = (
                    3
                    / 2
                    * (
                        self.r_s * np.power(i_sd[1:], 2)
                        + (self.r_s + self.r_r * self.l_m**2 / self.l_r**2)
                        * np.power(i_sq, 2)
                    )
                )  # Calculate losses

                i_idx = np.argmin(pv)  # Minimize losses
                i_sd_opt = i_sd[i_idx]
                i_sq_opt = i_sq[i_idx]
            else:
                i_sq_opt = 0
                i_sd_opt = 0

            psi_opt = self.l_m * i_sd_opt
            psi_opt_t.append([t, psi_opt, i_sd_opt, i_sq_opt])
        return np.array(psi_opt_t).T

    def t_max(self):
        # All flux values to calculate the corresponding torque and currents for
        psi = np.linspace(self.psi_max, 0, self.psi_count)
        # The resulting torque and currents lists
        t_val = []
        i_sd_val = []
        i_sq_val = []

        for psi_ in psi:
            i_sd = psi_ / self.l_m
            i_sq = np.sqrt(
                self.nominal_values[self.u_sd_idx] ** 2
                / (self.nominal_values[self.omega_idx] ** 2 * self.l_s**2)
                - i_sd**2
            )

            t = 3 / 2 * self.p * self.l_m / self.l_r * psi_ * i_sq
            t_val.append(t)
            i_sd_val.append(i_sd)
            i_sq_val.append(i_sq)

        # The characteristic is symmetrical for positive and negative torques.
        t_val.extend(list(-np.array(t_val[::-1])))
        psi = np.append(psi, psi[::-1])
        i_sd_val.extend(i_sd_val[::-1])
        i_sq_val.extend(list(-np.array(i_sq_val[::-1])))

        return np.array([t_val, psi, i_sd_val, i_sq_val])

    # Methods to get the indices of the lists for maximum torque and optimal flux

    def get_psi_opt(self, torque):
        torque = np.clip(torque, self.t_minimum, self.t_maximum)
        return int(
            round(
                (torque - self.t_minimum)
                / (self.t_maximum - self.t_minimum)
                * (self.torque_count - 1)
            )
        )

    def get_t_max(self, psi):
        psi = np.clip(psi, 0, self.psi_max)
        return int(round(psi / self.psi_max * (self.psi_count - 1)))

    def control(self, state, torque, psi_abs):
        """
        This main method is called by the CascadedFieldOrientedControllerRotorFluxObserver to calculate reference
        values for the i_sd and i_sq currents from a given torque reference.

        Args:
            state: state of the gym-electric-motor environment
            torque: reference value for the torque
            psi_abs: amount of the estimated flux

        Returns:
            Reference values for the currents i_sq and i_sd, optimal flux
        """

        # Calculate the optimal flux
        psi_opt = self.psi_opt_t[1, self.get_psi_opt(torque)]
        psi_max = self.modulation_control(state)
        psi_opt = min(psi_opt, psi_max)

        # Limit the torque
        t_max = self.t_max_psi[0, self.psi_count - self.get_t_max(psi_opt)]
        torque = np.clip(torque, -np.abs(t_max), np.abs(t_max))

        # Calculate the reference for i_sd
        i_sd_ = self.psi_controller.control(psi_abs, psi_opt)
        i_sd = np.clip(
            i_sd_,
            -0.9 * self.nominal_values[self.i_sd_idx],
            0.9 * self.nominal_values[self.i_sd_idx],
        )
        if i_sd_ == i_sd:
            self.psi_controller.integrate(psi_abs, psi_opt)

        # Calculate the reference for i_sq
        i_sq = np.clip(
            torque / max(psi_abs, 0.001) * 2 / 3 / self.p * self.l_r / self.l_m,
            -self.nominal_values[self.i_sq_idx],
            self.nominal_values[self.i_sq_idx],
        )
        if self.nominal_values[self.i_sq_idx] < np.sqrt(i_sq**2 + i_sd**2):
            i_sq = np.sign(i_sq) * np.sqrt(
                self.nominal_values[self.i_sq_idx] ** 2 - i_sd**2
            )

        # Update plots
        if self.plot_torque:
            if self.k == 0:
                self.intitialize_torque_plot()
                self.k_list = []
                self.torque_list = []
                self.psi_list = []

            self.k_list.append(self.k * self.tau)
            self.torque_list.append(torque)
            self.psi_list.append(psi_opt)

            if self.k % self.update_interval == 0:
                self.psi_opt_plot.scatter(
                    self.torque_list, self.psi_list, c="tab:blue", s=3
                )
                self.t_max_plot.scatter(
                    self.psi_list, self.torque_list, c="tab:blue", s=3
                )

                self.fig_torque.canvas.draw()
                self.fig_torque.canvas.flush_events()
                self.k_list = []
                self.torque_list = []
                self.psi_list = []

        self.k += 1
        return i_sq / self.limits[self.i_sq_idx], i_sd / self.limits[
            self.i_sd_idx
        ], psi_opt

    def modulation_control(self, state):
        # Calculate modulation
        a = (
            2
            * np.sqrt(
                (state[self.u_sd_idx] * self.limits[self.u_sd_idx]) ** 2
                + (state[self.u_sq_idx] * self.limits[self.u_sq_idx]) ** 2
            )
            / self.u_dc
        )

        #
        if a > 1.01 * self.a_max:
            self.integrated = self.integrated_reset

        a_delta = self.k_ * self.a_max - a

        omega = max(np.abs(state[self.omega_idx]) * self.limits[self.omega_idx], 0.0001)

        # Calculate i gain
        k_i = 2 * np.abs(omega) * self.p / self.u_dc
        i_gain = self.i_gain * k_i

        psi_delta = i_gain * (
            a_delta * self.tau + self.integrated
        )  # Calculate Flux delta

        # Check, if limits are violated
        if self.psi_low <= psi_delta <= self.psi_high:
            self.integrated += a_delta * self.tau
        else:
            psi_delta = np.clip(psi_delta, self.psi_low, self.psi_high)

        psi_max = self.u_dc / (np.sqrt(3) * np.abs(omega) * self.p)

        psi = max(psi_max + psi_delta, 0)

        # Update plot
        if self.plot_modulation:
            if self.k == 0:
                self.initialize_modulation_plot()
            self.k_list_a.append(self.k * self.tau)
            self.a_list.append(a)
            self.psi_delta_list.append(psi_delta)

            if self.k % self.update_interval == 0:
                self.a_plot.scatter(self.k_list_a, self.a_list, c="tab:blue", s=3)
                self.psi_delta_plot.scatter(
                    self.k_list_a, self.psi_delta_list, c="tab:blue", s=3
                )
                self.a_plot.set_xlim(
                    max(self.k * self.tau, 1) - 1, max(self.k * self.tau, 1)
                )
                self.psi_delta_plot.set_xlim(
                    max(self.k * self.tau, 1) - 1, max(self.k * self.tau, 1)
                )
                self.k_list_a = []
                self.a_list = []
                self.psi_delta_list = []

        return psi

    def initialize_modulation_plot(self):
        if self.plot_modulation:
            plt.ion()
            self.fig_modulation = plt.figure("Modulation Controller")
            self.a_plot = plt.subplot2grid((1, 2), (0, 0))
            self.psi_delta_plot = plt.subplot2grid((1, 2), (0, 1))

            # Define modulation plot
            self.a_plot.set_title("Modulation")
            self.a_plot.axhline(self.k_ * self.a_max, c="tab:orange", label=r"$a^*$")
            self.a_plot.plot([], [], c="tab:blue", label="a")
            self.a_plot.set_xlabel("t / s")
            self.a_plot.set_ylabel("a")
            self.a_plot.grid(True)
            self.a_plot.set_xlim(0, 1)
            self.a_plot.legend(loc=2)

            # Define the delta flux plot
            self.psi_delta_plot.set_title(r"$\Psi_\mathrm{\Delta}$")
            self.psi_delta_plot.axhline(
                self.psi_low, c="tab:red", linestyle="dashed", label="Limit"
            )
            self.psi_delta_plot.axhline(self.psi_high, c="tab:red", linestyle="dashed")
            self.psi_delta_plot.plot(
                [], [], c="tab:blue", label=r"$\Psi_\mathrm{\Delta}$"
            )
            self.psi_delta_plot.set_xlabel("t / s")
            self.psi_delta_plot.set_ylabel(r"$\Psi_\mathrm{\Delta} / Vs$")
            self.psi_delta_plot.grid(True)
            self.psi_delta_plot.set_xlim(0, 1)
            self.psi_delta_plot.legend(loc=2)

            self.a_list = []
            self.psi_delta_list = []
            self.k_list_a = []

    def reset(self):
        # Reset the integrated value
        self.psi_controller.reset()
