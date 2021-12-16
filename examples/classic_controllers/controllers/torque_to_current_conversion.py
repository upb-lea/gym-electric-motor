import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


class TorqueToCurrentConversion:
    """
        This class represents the torque controller for cascaded control of synchronous motors.  For low speeds only the
        current limitation of the motor is important. The current vector to set a desired torque is selected so that the
        amount of the current vector is minimum (Maximum Torque per Current). For higher speeds, the voltage limitation
        of the synchronous motor or the actuator must also be taken into account. This is done by converting the
        available voltage to a speed-dependent maximum flux. An additional modulation controller is used for the flux
        control. By limiting the flux and the maximum torque per flux (MTPF), an operating point for the flux and the
        torque is obtained. This is then converted into a current operating point. The conversion can be done by
        different methods (parameter torque_control). On the one hand, maps can be determined in advance by
        interpolation or analytically, or the analytical determination can be done online.
        For the visualization of the operating points, both for the current operating points as well as the flux and
        torque operating points, predefined plots are available (plot_torque: default True). Also the values of the
        modulation controller can be visualized (plot_modulation: default False).
    """

    def __init__(self, environment, plot_torque=True, plot_modulation=False, update_interval=1000,
                 torque_control='interpolate'):

        self.mp = environment.physical_system.electrical_motor.motor_parameter
        self.limit = environment.physical_system.limits
        self.nominal_values = environment.physical_system.nominal_state
        self.torque_control = torque_control

        self.l_d = self.mp['l_d']
        self.l_q = self.mp['l_q']
        self.p = self.mp['p']
        self.psi_p = self.mp.get('psi_p', 0)
        self.invert = -1 if (self.psi_p == 0 and self.l_q < self.l_d) else 1
        self.tau = environment.physical_system.tau

        self.omega_idx = environment.state_names.index('omega')
        self.i_sd_idx = environment.state_names.index('i_sd')
        self.i_sq_idx = environment.state_names.index('i_sq')
        self.u_sd_idx = environment.state_names.index('u_sd')
        self.u_sq_idx = environment.state_names.index('u_sq')
        self.torque_idx = environment.state_names.index('torque')
        self.epsilon_idx = environment.state_names.index('epsilon')

        self.a_max = 2 / np.sqrt(3)     # maximum modulation level
        self.k_ = 0.95
        d = 1.2    # damping of the modulation controller
        alpha = d / (d - np.sqrt(d ** 2 - 1))
        self.i_gain = 1 / (self.mp['l_q'] / (1.25 * self.mp['r_s'])) * (alpha - 1) / alpha ** 2

        self.u_a_idx = environment.state_names.index('u_a')
        self.u_dc = np.sqrt(3) * self.limit[self.u_a_idx]
        self.limited = False
        self.integrated = 0
        self.psi_high = 0.2 * np.sqrt((self.psi_p + self.l_d * self.nominal_values[self.i_sd_idx]) ** 2 + (
                    self.l_q * self.nominal_values[self.i_sq_idx]) ** 2)
        self.psi_low = -self.psi_high
        self.integrated_reset = 0.01 * self.psi_low  # Reset value of the modulation controller

        self.t_count = 250
        self.psi_count = 250
        self.i_count = 500

        self.torque_list = []
        self.psi_list = []
        self.k_list = []
        self.i_d_list = []
        self.i_q_list = []

        def mtpc():
            def i_q_(i_d, torque):
                return torque / (i_d * (self.l_d - self.l_q) + self.psi_p) / (1.5 * self.p)

            def i_d_(i_q, torque):
                return -np.abs(torque / (1.5 * self.p * (self.l_d - self.l_q) * i_q))

            # calculate the maximum torque
            self.max_torque = max(
                1.5 * self.p * (self.psi_p + (self.l_d - self.l_q) * (-self.limit[self.i_sd_idx])) * self.limit[
                    self.i_sq_idx], self.limit[self.torque_idx])
            torque = np.linspace(-self.max_torque, self.max_torque, self.t_count)
            characteristic = []

            for t in torque:
                if self.psi_p != 0:
                    if self.l_d == self.l_q:
                        i_d = 0
                    else:
                        i_d = np.linspace(-2.5*self.limit[self.i_sd_idx], 0, self.i_count)
                    i_q = i_q_(i_d, t)
                else:
                    i_q = np.linspace(-2.5*self.limit[self.i_sq_idx], 2.5*self.limit[self.i_sq_idx], self.i_count)
                    if self.l_d == self.l_q:
                        i_d = 0
                    else:
                        i_d = i_d_(i_q, t)

                # Different current vectors are determined for each torque and the smallest magnitude is selected
                i = np.power(i_d, 2) + np.power(i_q, 2)
                min_idx = np.where(i == np.amin(i))[0][0]
                if self.l_d == self.l_q:
                    i_q_ret = i_q
                    i_d_ret = i_d
                else:
                    i_q_ret = np.sign((self.l_q - self.l_d) * t) * np.abs(i_q[min_idx])
                    i_d_ret = i_d[min_idx]

                # The flow is finally calculated from the currents
                psi = np.sqrt((self.psi_p + self.l_d * i_d_ret) ** 2 + (self.l_q * i_q_ret) ** 2)
                characteristic.append([t, i_d_ret, i_q_ret, psi])
            return np.array(characteristic)

        def mtpf():
            # maximum flux is calculated
            self.psi_max_mtpf = np.sqrt((self.psi_p + self.l_d * self.nominal_values[self.i_sd_idx]) ** 2 + (
                        self.l_q * self.nominal_values[self.i_sq_idx]) ** 2)
            psi = np.linspace(0, self.psi_max_mtpf, self.psi_count)
            i_d = np.linspace(-self.nominal_values[self.i_sd_idx], 0, self.i_count)
            i_d_best = 0
            i_q_best = 0
            psi_i_d_q = []

            # Iterates through all flux values to determine the maximum torque
            for psi_ in psi:
                if psi_ == 0:
                    i_d_ = -self.psi_p / self.l_d
                    i_q = 0
                    t = 0
                    psi_i_d_q.append([psi_, t, i_d_, i_q])

                else:
                    if self.psi_p == 0:
                        i_q_best = psi_ / np.sqrt(self.l_d ** 2 + self.l_q ** 2)
                        i_d_best = -i_q_best
                        t = 1.5 * self.p * (self.psi_p + (self.l_d - self.l_q) * i_d_best) * i_q_best
                    else:
                        i_d_idx = np.where(psi_ ** 2 - np.power(self.psi_p + self.l_d * i_d, 2) >= 0)
                        i_d_ = i_d[i_d_idx]

                        # calculate all possible i_q currents for i_d currents
                        i_q = np.sqrt(psi_ ** 2 - np.power(self.psi_p + self.l_d * i_d_, 2)) / self.l_q
                        i_idx = np.where(np.sqrt(np.power(i_q / self.nominal_values[self.i_sq_idx], 2) + np.power(
                            i_d_ / self.nominal_values[self.i_sd_idx], 2)) <= 1)
                        i_d_ = i_d_[i_idx]
                        i_q = i_q[i_idx]
                        torque = 1.5 * self.p * (self.psi_p + (self.l_d - self.l_q) * i_d_) * i_q

                        # choose the maximum torque
                        if np.size(torque) > 0:
                            t = np.amax(torque)
                            i_idx = np.where(torque == t)[0][0]
                            i_d_best = i_d_[i_idx]
                            i_q_best = i_q[i_idx]
                    if np.sqrt(i_d_best**2 + i_q_best**2) <= self.nominal_values[self.i_sq_idx]:
                        psi_i_d_q.append([psi_, t, i_d_best, i_q_best])

            psi_i_d_q = np.array(psi_i_d_q)
            self.psi_max_mtpf = np.max(psi_i_d_q[:, 0])
            psi_i_d_q_neg = np.rot90(np.array([psi_i_d_q[:, 0], -psi_i_d_q[:, 1], psi_i_d_q[:, 2], -psi_i_d_q[:, 3]]))
            psi_i_d_q = np.append(psi_i_d_q_neg, psi_i_d_q, axis=0)

            return np.array(psi_i_d_q)

        self.mtpc = mtpc()  # define maximum torque per current characteristic
        self.mtpf = mtpf()  # define maximum torque per flux characteristic

        # Calculate a list with the flux and the corresponding torque of the mtpc characteristic
        self.psi_t = np.sqrt(
            np.power(self.psi_p + self.l_d * self.mtpc[:, 1], 2) + np.power(self.l_q * self.mtpc[:, 2], 2))
        self.psi_t = np.array([self.mtpc[:, 0], self.psi_t])

        # define a grid for the two current components
        self.i_q_max = np.linspace(-self.nominal_values[self.i_sq_idx], self.nominal_values[self.i_sq_idx], self.i_count)
        self.i_d_max = -np.sqrt(self.nominal_values[self.i_sq_idx] ** 2 - np.power(self.i_q_max, 2))
        i_count_mgrid = self.i_count * 1j
        i_d, i_q = np.mgrid[-self.limit[self.i_sd_idx]:0:i_count_mgrid,
                            -self.limit[self.i_sq_idx]:self.limit[self.i_sq_idx]:i_count_mgrid / 2]
        i_d = i_d.flatten()
        i_q = i_q.flatten()

        # Decide between SPMSM and IPMSM
        if self.l_d != self.l_q:
            idx = np.where(np.sign(self.psi_p + i_d * self.l_d) * np.power(self.psi_p + i_d * self.l_d, 2) + np.power(
                i_q * self.l_q, 2) > 0)
        else:
            idx = np.where(self.psi_p + i_d * self.l_d > 0)

        i_d = i_d[idx]
        i_q = i_q[idx]

        # Calculate torque and flux for the grid of the currents
        t = self.p * 1.5 * (self.psi_p + (self.l_d - self.l_q) * i_d) * i_q
        psi = np.sqrt(np.power(self.l_d * i_d + self.psi_p, 2) + np.power(self.l_q * i_q, 2))

        self.t_min = np.amin(t)
        self.t_max = np.amax(t)

        self.psi_min = np.amin(psi)
        self.psi_max = np.amax(psi)

        if torque_control == 'analytical':
            res = []
            for psi in np.linspace(self.psi_min, self.psi_max, self.psi_count):
                ret = []
                for T in np.linspace(self.t_min, self.t_max, self.t_count):
                    i_d_, i_q_ = self.solve_analytical(T, psi)
                    ret.append([T, psi, i_d_, i_q_])
                res.append(ret)
            res = np.array(res)
            self.t_grid = res[:, :, 0]
            self.psi_grid = res[:, :, 1]
            self.i_d_inter = res[:, :, 2].T
            self.i_q_inter = res[:, :, 3].T
            self.i_d_inter_plot = self.i_d_inter.T
            self.i_q_inter_plot = self.i_q_inter.T

        elif torque_control == 'interpolate':
            # Interpolate the torque and flux to get lists for the optimal currents
            self.t_grid, self.psi_grid = np.mgrid[np.amin(t):np.amax(t):np.complex(0, self.t_count),
                                                  self.psi_min:self.psi_max:np.complex(self.psi_count)]
            self.i_q_inter = griddata((t, psi), i_q, (self.t_grid, self.psi_grid), method='linear')
            self.i_d_inter = griddata((t, psi), i_d, (self.t_grid, self.psi_grid), method='linear')
            self.i_d_inter_plot = self.i_d_inter
            self.i_q_inter_plot = self.i_q_inter

        elif torque_control != 'online':
            raise NotImplementedError

        self.k = 0
        self.update_interval = update_interval
        self.plot_torque = plot_torque
        self.plot_modulation = plot_modulation

    def intitialize_torque_plot(self):
        if self.plot_torque:
            plt.ion()
            self.fig_torque = plt.figure('Torque Controller')

            # Check if current, torque, flux characteristics could be plotted
            if self.torque_control in ['interpolate', 'analytical']:
                self.i_d_q_characteristic_ = plt.subplot2grid((2, 3), (0, 0), rowspan=2)
                self.psi_plot = plt.subplot2grid((2, 3), (0, 1))
                self.i_d_plot = plt.subplot2grid((2, 3), (0, 2), projection='3d')
                self.torque_plot = plt.subplot2grid((2, 3), (1, 1))
                self.i_q_plot = plt.subplot2grid((2, 3), (1, 2), projection='3d')

            elif self.torque_control == 'online':
                self.i_d_q_characteristic_ = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
                self.psi_plot = plt.subplot2grid((2, 2), (0, 1))
                self.torque_plot = plt.subplot2grid((2, 2), (1, 1))

            mtpc_i_idx = np.where(
                np.sqrt(np.power(self.mtpc[:, 1], 2) + np.power(self.mtpc[:, 2], 2)) <= self.nominal_values[
                    self.i_sd_idx])

            # Define the plot for the current characteristics
            self.i_d_q_characteristic_.set_title('$i_\mathrm{d,q_{ref}}$')
            self.i_d_q_characteristic_.plot(self.mtpc[mtpc_i_idx, 1][0], self.mtpc[mtpc_i_idx, 2][0], label='MTPC', c='tab:orange')
            self.i_d_q_characteristic_.plot(self.mtpf[:, 2], self.mtpf[:, 3], label=r'MTPF', c='tab:green')
            self.i_d_q_characteristic_.plot(self.i_d_max, self.i_q_max, label=r'$i_\mathrm{max}$', c='tab:red')
            self.i_d_q_characteristic_.plot([], [], label=r'$i_\mathrm{d,q}$', c='tab:blue')
            self.i_d_q_characteristic_.grid(True)
            self.i_d_q_characteristic_.legend(loc=2)
            self.i_d_q_characteristic_.axis('equal')
            self.i_d_q_characteristic_.set_xlabel(r'$i_\mathrm{d}$ / A')
            self.i_d_q_characteristic_.set_ylabel(r'$i_\mathrm{q}$ / A')

            # Define the plot for the flux characteristic
            self.psi_plot.set_title(r'$\Psi^*_\mathrm{max}(T^*)$')
            self.psi_plot.plot(self.psi_t[0], self.psi_t[1], label=r'$\Psi^*_\mathrm{max}(T^*)$', c='tab:orange')
            self.psi_plot.plot([], [], label=r'$\Psi(T)$', c='tab:blue')
            self.psi_plot.grid(True)
            self.psi_plot.set_xlabel(r'T / Nm')
            self.psi_plot.set_ylabel(r'$\Psi$ / Vs')
            self.psi_plot.set_ylim(bottom=0)
            self.psi_plot.legend(loc=2)

            # Define the plot for the torque characteristic
            torque = self.mtpf[:, 1]
            torque[0:np.where(torque == np.min(torque))[0][0]] = np.min(torque)
            torque[np.where(torque == np.max(torque))[0][0]:] = np.max(torque)
            self.torque_plot.set_title(r'$T_\mathrm{max}(\Psi_\mathrm{max})$')
            self.torque_plot.plot(self.mtpf[:, 0], torque, label=r'$T_\mathrm{max}(\Psi)$', c='tab:orange')
            self.torque_plot.plot([], [], label=r'$T(\Psi)$', c='tab:blue')
            self.torque_plot.set_xlabel(r'$\Psi$ / Vs')
            self.torque_plot.set_ylabel(r'$T_\mathrm{max}$ / Nm')
            self.torque_plot.grid(True)
            self.torque_plot.legend(loc=2)

            # Define the plot of currents
            if self.torque_control in ['interpolate', 'analytical']:
                self.i_q_plot.plot_surface(self.t_grid, self.psi_grid, self.i_q_inter_plot, cmap=cm.jet, linewidth=0,
                                           vmin=np.nanmin(self.i_q_inter_plot), vmax=np.nanmax(self.i_q_inter_plot))
                self.i_q_plot.set_ylabel(r'$\Psi / Vs$')
                self.i_q_plot.set_xlabel(r'$T / Nm$')
                self.i_q_plot.set_title(r'$i_\mathrm{q}(T, \Psi)$')

                self.i_d_plot.plot_surface(self.t_grid, self.psi_grid, self.i_d_inter_plot, cmap=cm.jet, linewidth=0,
                                           vmin=np.nanmin(self.i_d_inter_plot), vmax=np.nanmax(self.i_d_inter_plot))
                self.i_d_plot.set_ylabel(r'$\Psi / Vs$')
                self.i_d_plot.set_xlabel(r'$T / Nm$')
                self.i_d_plot.set_title(r'$i_\mathrm{d}(T, \Psi)$')

    def solve_analytical(self, torque, psi):
        """
           Assuming linear magnetization characteristics, the optimal currents for given torque and flux can be obtained
           by solving the torque and flux equations. These lead to a fourth degree polynomial which can be solved
           analytically.  There are two ways to use this analytical solution for control. On the one hand, the currents
           can be determined in advance as in the case of interpolation for different torques and fluxes and stored in a
           LUT (torque_control='analytical'). On the other hand, the solution can be calculated at runtime with the
           given torque and flux (torque_control='online').
        """

        poly = [self.l_d ** 2 * (self.l_d - self.l_q) ** 2,
                2 * self.l_d ** 2 * (self.l_d - self.l_q) * self.psi_p + 2 * self.l_d * self.psi_p * (
                        self.l_d - self.l_q) ** 2,
                self.l_d ** 2 * self.psi_p ** 2 + 4 * self.l_d * self.psi_p ** 2 * (self.l_d - self.l_q) + (
                        self.psi_p ** 2 - psi ** 2) * (
                        self.l_d - self.l_q) ** 2,
                2 * self.l_q * self.psi_p ** 3 + 2 * (self.psi_p ** 2 - psi ** 2) * self.psi_p * (self.l_d - self.l_q),
                (self.psi_p ** 2 - psi ** 2) * self.psi_p ** 2 + (self.l_q * 2 * torque / (3 * self.p)) ** 2]

        sol = np.roots(poly)    # Solve polynomial
        i_d = np.real(sol[-1])  # Select the appropriate solution for i_d
        i_q = 2 * torque / (3 * self.p * (self.psi_p + (self.l_d - self.l_q) * i_d))   # Calculate the corresponding i_q
        return i_d, i_q

    def get_i_d_q(self, torque, psi, psi_idx):
        """Method to solve the control online and check, if current is on mtpc characteristic"""
        i_d, i_q = self.solve_analytical(torque, psi)
        if i_d > self.mtpc[psi_idx, 1]:
            i_d = self.mtpc[psi_idx, 1]
            i_q = self.mtpc[psi_idx, 2]
        return i_d, i_q

    # Methods to get the indices of the calculated characteristics

    def get_t_idx(self, torque):
        torque = np.clip(torque, self.t_min, self.t_max)
        return int(round((torque - self.t_min) / (self.t_max - self.t_min) * (self.t_count - 1)))

    def get_psi_idx(self, psi):
        psi = np.clip(psi, self.psi_min, self.psi_max)
        return int(round((psi - self.psi_min) / (self.psi_max - self.psi_min) * (self.psi_count - 1)))

    def get_psi_idx_mtpf(self, psi):
        return np.clip(int((self.psi_count - 1) - round(psi / self.psi_max_mtpf * (self.psi_count - 1))), 0,
                       self.psi_count)

    def get_t_idx_mtpc(self, torque):
        return np.clip(int(round((torque + self.max_torque) / (2 * self.max_torque) * (self.t_count - 1))), 0,
                       self.t_count)

    def control(self, state, torque):
        """
            This main method is called by the CascadedFieldOrientedController to calculate reference values for the i_d
            and i_q currents from a given torque reference.
        """

        # get the optimal psi for a given torque from the mtpc characteristic
        psi_idx_ = self.get_t_idx_mtpc(torque)
        psi_opt = self.mtpc[psi_idx_, 3]

        # limit the flux to keep the voltage limit using the modulation controller
        psi_max_ = self.modulation_control(state)
        psi_max = min(psi_opt, psi_max_)

        # get the maximum torque for a given flux from the mtpf characteristic
        psi_max_idx = self.get_psi_idx_mtpf(psi_max)
        t_max = np.abs(self.mtpf[psi_max_idx, 1])
        if np.abs(torque) > t_max:
            torque = np.sign(torque) * t_max

        # calculate the currents online
        if self.torque_control == 'online':
            i_d, i_q = self.get_i_d_q(torque, psi_max, psi_idx_)

        # get the currents from a LUT
        else:
            t_idx = self.get_t_idx(torque)
            psi_idx = self.get_psi_idx(psi_max)

            if self.i_d_inter[t_idx, psi_idx] <= self.mtpf[psi_max_idx, 2]:
                i_d = self.mtpf[psi_max_idx, 2]
                i_q = np.sign(torque) * np.abs(self.mtpf[psi_max_idx, 3])
                torque = np.sign(torque) * t_max
            else:
                i_d = self.i_d_inter[t_idx, psi_idx]
                i_q = self.i_q_inter[t_idx, psi_idx]
                if i_d > self.mtpc[psi_idx_, 1]:
                    i_d = self.mtpc[psi_idx_, 1]
                    i_q = np.sign(torque) * np.abs(self.mtpc[psi_idx_, 2])

        # ensure that the mtpf characteristic curve is observed
        if i_d < self.mtpf[psi_max_idx, 2]:
            i_d = self.mtpf[psi_max_idx, 2]
            i_q = np.sign(torque) * np.abs(self.mtpf[psi_max_idx, 3])

        # invert the i_q if necessary
        i_q = self.invert * i_q

        # plot all calculated quantities
        if self.plot_torque:
            if self.k == 0:
                self.intitialize_torque_plot()

            self.k_list.append(self.k * self.tau)
            self.i_d_list.append(i_d)
            self.i_q_list.append(i_q)
            self.torque_list.append(torque)
            self.psi_list.append(psi_max)

            if self.k % self.update_interval == 0:
                self.psi_plot.scatter(self.torque_list, self.psi_list, c='tab:blue', s=3)
                self.torque_plot.scatter(self.psi_list, self.torque_list, c='tab:blue', s=3)
                self.i_d_q_characteristic_.scatter(self.i_d_list, self.i_q_list, c='tab:blue', s=3)

                self.fig_torque.canvas.draw()
                self.fig_torque.canvas.flush_events()
                self.k_list = []
                self.i_d_list = []
                self.i_q_list = []
                self.torque_list = []
                self.psi_list = []

        # clipping and normalizing the currents
        i_q = np.clip(i_q, -self.nominal_values[self.i_sq_idx], self.nominal_values[self.i_sq_idx]) / self.limit[self.i_sq_idx]
        i_d = np.clip(i_d, -self.nominal_values[self.i_sd_idx], self.nominal_values[self.i_sd_idx]) / self.limit[self.i_sd_idx]

        self.k += 1

        return i_q, i_d

    def modulation_control(self, state):
        """
            To ensure the functionality of the current control, a small dynamic manipulated variable reserve to the
            voltage limitation must be kept available. This control is performed by this modulation controller. Further
            information can be found at https://ieeexplore.ieee.org/document/7409195.
        """

        # Calculate modulation
        a = 2 * np.sqrt((state[self.u_sd_idx] * self.limit[self.u_sd_idx]) ** 2 + (
                    state[self.u_sq_idx] * self.limit[self.u_sq_idx]) ** 2) / self.u_dc

        # Check, if integral part should be reset
        if a > 1.1 * self.a_max:
            self.integrated = self.integrated_reset

        a_delta = self.k_ * self.a_max - a
        omega = max(np.abs(state[self.omega_idx]) * self.limit[self.omega_idx], 0.0001)

        # Calculate maximum flux for a given speed
        psi_max_ = self.u_dc / (np.sqrt(3) * omega * self.p)

        # Calculate gain
        k_i = 2 * omega * self.p / self.u_dc
        i_gain = self.i_gain / k_i

        psi_delta = i_gain * (a_delta * self.tau + self.integrated)

        # Check, if limits are violated
        if self.psi_low <= psi_delta <= self.psi_high:
            if self.limited:
                self.integrated = self.integrated_reset
                self.limited = False
            self.integrated += a_delta * self.tau

        else:
            psi_delta = np.clip(psi_delta, self.psi_low, self.psi_high)
            self.limited = True

        # Calculate output flux of the modulation controller
        psi = psi_max_ + psi_delta

        # Plot the operation of the modulation controller
        if self.plot_modulation:
            if self.k == 0:
                self.initialize_modulation_plot()
            self.k_list_a.append(self.k * self.tau)
            self.a_list.append(a)
            self.psi_delta_list.append(psi_delta)

            if self.k % self.update_interval == 0:
                    self.a_plot.scatter(self.k_list_a, self.a_list, c='tab:blue', s=3)
                    self.psi_delta_plot.scatter(self.k_list_a, self.psi_delta_list, c='tab:blue', s=3)
                    self.a_plot.set_xlim(max(self.k * self.tau, 1) - 1, max(self.k * self.tau, 1))
                    self.psi_delta_plot.set_xlim(max(self.k * self.tau, 1) - 1, max(self.k * self.tau, 1))
                    self.k_list_a = []
                    self.a_list = []
                    self.psi_delta_list = []

        return psi

    def initialize_modulation_plot(self):
        if self.plot_modulation:
            plt.ion()
            self.fig_modulation = plt.figure('Modulation Controller')
            self.a_plot = plt.subplot2grid((1, 2), (0, 0))
            self.psi_delta_plot = plt.subplot2grid((1, 2), (0, 1))

            # Define the modulation plot
            self.a_plot.set_title('Modulation')
            self.a_plot.axhline(self.k_ * self.a_max, c='tab:orange', label=r'$a^*$')
            self.a_plot.plot([], [], c='tab:blue', label='a')
            self.a_plot.set_xlabel('t / s')
            self.a_plot.set_ylabel('a')
            self.a_plot.grid(True)
            self.a_plot.set_xlim(0, 1)
            self.a_plot.legend(loc=2)

            # Define the delta flux plot
            self.psi_delta_plot.set_title(r'$\Psi_\mathrm{\Delta}$')
            self.psi_delta_plot.axhline(self.psi_low, c='tab:red', linestyle='dashed', label='Limit')
            self.psi_delta_plot.axhline(self.psi_high, c='tab:red', linestyle='dashed')
            self.psi_delta_plot.plot([], [], c='tab:blue', label=r'$\Psi_\mathrm{\Delta}$')
            self.psi_delta_plot.set_xlabel('t / s')
            self.psi_delta_plot.set_ylabel(r'$\Psi_\mathrm{\Delta} / Vs$')
            self.psi_delta_plot.grid(True)
            self.psi_delta_plot.set_xlim(0, 1)
            self.psi_delta_plot.legend(loc=2)

            self.a_list = []
            self.psi_delta_list = []
            self.k_list_a = []

    def reset(self):
        # Reset the integrated value
        self.integrated = self.integrated_reset
