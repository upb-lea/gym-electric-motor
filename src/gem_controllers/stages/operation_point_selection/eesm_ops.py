import numpy as np
import scipy.interpolate as sp_interpolate

from .foc_operation_point_selection import FieldOrientedControllerOperationPointSelection


class EESMOperationPointSelection(FieldOrientedControllerOperationPointSelection):
    """
    This class represents the operation point selection of the torque controller for cascaded control of an
    externally synchronous motor. The operating point is selected in the analog to that of the PMSM and SynRM, but
    the excitation current is also included in the optimization of the operating point.
    """

    def __init__(self, max_modulation_level: float = 2 / np.sqrt(3), modulation_damping: float = 1.2):
        """
        Args:
            max_modulation_level(float): Maximum value for the modulation controller.
            modulation_damping(float): Damping of the gain of the modulation controller.
        """

        super().__init__(max_modulation_level, modulation_damping)
        self.l_d = None
        self.l_q = None
        self.l_m = None
        self.l_e = None
        self.r_s = None
        self.r_e = None
        self.i_e_lim = None
        self.t_lim = None
        self.i_q_lim = None

        self.t_count = None
        self.psi_count = None
        self.i_e_count = None

        self.psi_opt = None
        self.i_d_opt = None
        self.i_q_opt = None
        self.i_e_opt = None
        self.t_max = None
        self.psi_max = None
        self.t_max_psi = None

        self.t_grid_count = None
        self.psi_grid_count = None

        self.torque_equation = None
        self.loss = None
        self.poly = None

    def tune(self, env, env_id, current_safety_margin=0.2):
        """
        Tune the operation point selcetion stage.

        Args:
            env(gym_electric_motor.ElectricMotorEnvironment): The environment to be controlled.
            env_id(str): The id of the environment.
            current_safety_margin(float): Percentage of the current margin to the current limit.
        """

        super().tune(env, env_id, current_safety_margin)
        self.l_d = self.mp["l_d"]
        self.l_q = self.mp["l_q"]
        self.l_m = self.mp["l_m"]
        self.l_e = self.mp["l_e"]
        self.r_s = self.mp["r_s"]
        self.r_e = self.mp["r_e"]
        self.p = self.mp["p"]
        self.i_e_lim = env.limits[env.state_names.index("i_e")] * (1 - current_safety_margin)
        self.i_q_lim = env.limits[env.state_names.index("i_sq")] * (1 - current_safety_margin)
        self.t_lim = env.limits[env.state_names.index("torque")]

        self.t_count = 50
        self.psi_count = 100
        self.i_e_count = 150

        self.t_grid_count = 200
        self.psi_grid_count = 200

        self.k_ = 0.953
        self.i_gain = 1 / (self.l_q / (1.25 * self.r_s)) * (self.alpha - 1) / self.alpha**2

        self.psi_high = 0.2 * np.sqrt(
            (self.l_m * self.i_e_lim * current_safety_margin + self.l_d * self.i_sq_limit * current_safety_margin) ** 2
        )

        self.psi_low = -self.psi_high
        self.integrated_reset = 0.01 * self.psi_low

        self.torque_equation = (
            lambda i_d, i_q, i_e: 3 / 2 * self.p * (self.l_m * i_e + (self.l_d - self.l_q) * i_d) * i_q
        )

        self.loss = lambda i_d, i_q, i_e: np.abs(i_d) * self.r_s + np.abs(i_q) * self.r_s + np.abs(i_e) * self.r_e

        self.poly = lambda i_e, psi, torque: [
            self.l_d**2 * (self.l_d - self.l_q) ** 2,
            2 * self.l_d**2 * (self.l_d - self.l_q) * self.l_m * i_e
            + 2 * self.l_d * self.l_m * i_e * (self.l_d - self.l_q) ** 2,
            self.l_d**2 * (self.l_m * i_e) ** 2
            + 4 * self.l_d * (self.l_m * i_e) ** 2 * (self.l_d - self.l_q)
            + ((self.l_m * i_e) ** 2 - psi**2) * (self.l_d - self.l_q) ** 2,
            2 * self.l_q * (self.l_m * i_e) ** 3
            + 2 * ((self.l_m * i_e) ** 2 - psi**2) * self.l_m * i_e * (self.l_d - self.l_q),
            ((self.l_m * i_e) ** 2 - psi**2) * (self.l_m * i_e) ** 2 + (self.l_q * torque / (3 * self.p)) ** 2,
        ]

        self._calculate_luts()

    def solve_analytical(self, torque, psi, i_e):
        """
        Assuming linear magnetization characteristics, the optimal currents for given reference, flux and exitation
        current can be obtained by solving the reference and flux equations. These lead to a fourth degree polynomial
        which can be solved analytically.

        Args:
            torque(float): The torque reference value.
            psi(float): The optimal flux value.
            i_e(float): The excitation current.

        Returns:
            i_d(float): optimal i_sd current
            i_q(flaot): optimal i_sq current
        """
        if torque == 0 and i_e == 0:
            return 0, 0
        else:
            i_d = np.real(np.roots(self.poly(i_e, psi, torque))[-1])
            i_q = 2 * torque / (3 * self.p * (self.l_m * i_e + (self.l_d - self.l_q) * i_d))
            return i_d, i_q

    def _calculate_luts(self):
        """
        Calculates the lookup tables for the maximum torque and the optimal currents for a given torque reference.
        """
        minimum_loss = []
        best_params = []

        minimum_loss_psi = []
        best_params_psi = []

        self.psi_max = self.l_m * self.i_e_lim + self.l_d * self.i_q_lim
        torque = np.linspace(0, self.t_lim, self.t_count)

        self.t_max_psi = np.zeros(self.psi_count)

        for t in torque:
            losses = []
            parameter = []
            for idx, psi in enumerate(np.linspace(0, self.psi_max, self.psi_count)):
                losses_psi = []
                parameter_psi = []
                for i_e in np.linspace(0, self.i_e_lim, self.i_e_count):
                    i_d, i_q = self.solve_analytical(t, psi, i_e)
                    if np.sqrt(i_d**2 + i_q**2) < self.i_q_lim:
                        loss = self.loss(i_d, i_q, i_e)
                        params = np.array([t, psi, i_d, i_q, i_e])
                        losses.append(loss)
                        losses_psi.append(loss)
                        parameter.append(params)
                        parameter_psi.append(params)
                        self.t_max_psi[idx] = t
                if len(losses_psi) > 0:
                    minimum_loss_psi.append(min(losses_psi))
                    best_params_psi.append(parameter_psi[losses_psi.index(minimum_loss_psi[-1])])
            if len(losses) > 0:
                minimum_loss.append(min(losses))
                best_params.append(parameter[losses.index(minimum_loss[-1])])

        best_params = np.array(best_params)
        best_params_psi = np.array(best_params_psi)

        self.t_max_psi = sp_interpolate.interp1d(
            np.linspace(0, self.psi_max, self.psi_count), 0.99 * self.t_max_psi, kind="linear"
        )

        self.t_max = np.max(best_params[:, 0])
        self.psi_opt = sp_interpolate.interp1d(best_params[:, 0], best_params[:, 1], kind="cubic")
        self.i_d_opt = sp_interpolate.interp1d(best_params[:, 0], best_params[:, 2], kind="cubic")
        self.i_q_opt = sp_interpolate.interp1d(best_params[:, 0], best_params[:, 3], kind="cubic")
        self.i_e_opt = sp_interpolate.interp1d(best_params[:, 0], best_params[:, 4], kind="cubic")

        self.t_grid, self.psi_grid = np.mgrid[
            0 : self.t_max : np.complex(0, self.t_grid_count), 0 : self.psi_max : np.complex(self.psi_grid_count)
        ]

        self.i_d_inter = sp_interpolate.griddata(
            (best_params_psi[:, 0], best_params_psi[:, 1]),
            best_params_psi[:, 2],
            (self.t_grid, self.psi_grid),
            method="linear",
        )
        self.i_q_inter = sp_interpolate.griddata(
            (best_params_psi[:, 0], best_params_psi[:, 1]),
            best_params_psi[:, 3],
            (self.t_grid, self.psi_grid),
            method="linear",
        )
        self.i_e_inter = sp_interpolate.griddata(
            (best_params_psi[:, 0], best_params_psi[:, 1]),
            best_params_psi[:, 4],
            (self.t_grid, self.psi_grid),
            method="linear",
        )

    def _get_psi_idx(self, psi):
        """
        Get the index of the lookup tables for a given flux.

        Args:
            psi(float): optimal magnetic flux.

        Returns:
            index(int): index of the lookup tables of the currents
        """

        psi = np.clip(psi, 0, self.psi_max)
        return int(round(psi / self.psi_max * (self.psi_grid_count - 1)))

    def _get_t_idx(self, torque):
        """
        Get the index of the lookup tables for a given torque.

        Args:
            torque(float): The clipped torque reference value.

        Returns:
            index(int): index of the lookup tables of the currents
        """

        torque = np.clip(torque, 0, self.t_max)
        return int(round(torque / self.t_max * (self.t_grid_count - 1)))

    def _select_operating_point(self, state, reference):
        """
        Calculate the current operation point for a given torque reference value.

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference of the state.

        Returns:
            current_reference(np.ndarray): references for the current control stage
        """

        psi_max = self.modulation_control(state)
        t_ref = reference[0]

        t_ref_clip = np.abs(np.clip(t_ref, -self.t_max, self.t_max))
        psi_opt = self.psi_opt(t_ref_clip)

        psi = np.clip(psi_opt, 0, psi_max)

        t_max = self.t_max_psi(psi_opt)
        t_ref_clip = np.clip(t_ref_clip, 0, t_max)

        t_idx = self._get_t_idx(t_ref_clip)
        psi_idx = self._get_psi_idx(psi)

        i_d_ref = self.i_d_inter[t_idx, psi_idx]
        i_q_ref = np.sign(t_ref) * self.i_q_inter[t_idx, psi_idx]
        i_e_ref = self.i_e_inter[t_idx, psi_idx]

        return np.array([i_d_ref, i_q_ref, i_e_ref])

    def reset(self):
        """Reset the EESM operation point selection"""
        super().reset()
