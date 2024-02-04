import numpy as np

import gym_electric_motor as gem

from ..base_controllers import PIController
from .foc_operation_point_selection import FieldOrientedControllerOperationPointSelection


class SCIMOperationPointSelection(FieldOrientedControllerOperationPointSelection):
    """
    This class represents the operation point selection of the torque controller for the cascaded control of
    induction motors. The torque controller uses LUT to find an appropriate operating point for the flux and torque.
    The flux is limited by a modulation controller. A reference value for the i_sd current is then determined using
    the operating point of the flux and a PI controller. In addition, a reference for the i_sq current is calculated
    based on the current flux and the operating point of the torque.
    """

    def __init__(
        self,
        max_modulation_level: float = 2 / np.sqrt(3),
        modulation_damping: float = 1.2,
        psi_controller=PIController(control_task="FC"),
    ):
        """
        Args:
            max_modulation_level(float): Maximum value for the modulation controller.
            modulation_damping(float): Damping of the gain of the modulation controller.
            psi_controller(gc.stages.BaseController): Flux controller for the i_d current.
        """

        super().__init__(max_modulation_level, modulation_damping)

        self.l_m = None
        self.l_r = None
        self.l_s = None
        self.r_r = None
        self.r_s = None
        self.psi_abs_idx = None

        self.i_sd_count = None
        self.i_sq_count = None

        self.psi_controller = psi_controller

    def _psi_opt(self):
        """Calculate the optimal flux for a given torque"""
        psi_opt_t = []
        i_sd = np.linspace(0, self.limit[self.i_sd_idx], self.i_sd_count)
        for t in np.linspace(self.t_minimum, self.t_maximum, self.t_count):
            if t != 0:
                i_sq = t / (3 / 2 * self.p * self.l_m**2 / self.l_r * i_sd[1:])
                pv = (
                    3
                    / 2
                    * (
                        self.r_s * np.power(i_sd[1:], 2)
                        + (self.r_s + self.r_r * self.l_m**2 / self.l_r**2) * np.power(i_sq, 2)
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

    def _t_max(self):
        """Calculate a lookup table with the maximum torque for a given flux."""

        # The resulting torque and currents lists
        t_val = []
        i_sd_val = []
        i_sq_val = []
        psi = np.linspace(self.psi_max, 0, self.psi_count)

        # Calculate the maximum torque for a given flux
        for psi_ in psi:
            i_sd = psi_ / self.l_m
            i_sq = np.sqrt(
                self.nominal_value[self.u_sd_idx] ** 2 / (self.nominal_value[self.omega_idx] ** 2 * self.l_s**2)
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

    def tune(self, env: gem.core.ElectricMotorEnvironment, env_id: str, current_safety_margin: float = 0.2):
        """
        Tune the operation point selcetion stage.

        Args:
            env(gym_electric_motor.ElectricMotorEnvironment): The environment to be controlled.
            env_id(str): The id of the environment.
            current_safety_margin(float): Percentage of the current margin to the current limit.
        """

        super().tune(env, env_id, current_safety_margin)

        self.t_count = 1001
        self.psi_count = 1000
        self.i_sd_count = 500
        self.i_sq_count = 1000

        self.psi_abs_idx = env.state_names.index("psi_abs")

        self.t_minimum = -self.limit[self.torque_idx]
        self.t_maximum = self.limit[self.torque_idx]

        self.l_m = self.mp["l_m"]
        self.l_r = self.l_m + self.mp["l_sigr"]
        self.l_s = self.l_m + self.mp["l_sigs"]
        self.r_r = self.mp["r_r"]
        self.r_s = self.mp["r_s"]
        self.p = self.mp["p"]
        self.tau = env.physical_system.tau
        self.psi_controller.tune(env, env_id, 4, self.l_s / self.r_s)

        self.psi_opt_t = self._psi_opt()
        self.psi_max = np.max(self.psi_opt_t[1])
        self.t_max_psi = self._t_max()

        # Set the parameters of the modulation controller
        self.i_gain = 1 / (self.l_s / (1.25 * self.r_s)) * (self.alpha - 1) / self.alpha**2
        self.k_ = 0.8
        self.psi_high = 0.1 * self.psi_max
        self.psi_low = -self.psi_max
        self.integrated_reset = 0.5 * self.psi_low  # Reset value of the modulation controller

    # Methods to get the indices of the lists for maximum torque and optimal flux
    def _get_psi_opt(self, torque):
        """Get the optimal flux for a given torque"""
        torque = np.clip(torque, self.t_minimum, self.t_maximum)
        return int(round((torque - self.t_minimum) / (self.t_maximum - self.t_minimum) * (self.t_count - 1)))

    def _get_t_max(self, psi):
        """Get the maximum torque for a given flux"""
        psi = np.clip(psi, 0, self.psi_max)
        return int(round(psi / self.psi_max * (self.psi_count - 1)))

    def _select_operating_point(self, state, reference):
        """
        Calculate the current operation point for a given torque reference value.

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference of the state.

        Returns:
            current_reference(np.ndarray): references for the current control stage
        """

        psi = state[self.psi_abs_idx]

        # Get the optimal flux
        psi_opt = self.psi_opt_t[1, self._get_psi_opt(reference[0])]

        # Limit the flux by the modulation controller
        psi_max = self.modulation_control(state)
        psi_opt = min(psi_opt, psi_max)

        # Limit the torque
        t_max = self.t_max_psi[0, self.psi_count - self._get_t_max(psi_opt)]
        torque = np.clip(reference[0], -np.abs(t_max), np.abs(t_max))

        # Control the i_sd current with a PI-Controller
        i_sd_ = self.psi_controller(np.array([psi]), np.array([psi_opt]))
        i_sd = np.clip(i_sd_, -self.i_sd_limit, self.i_sd_limit)
        if i_sd_ == i_sd:
            self.psi_controller.integrate(np.array([psi]), np.array([psi_opt]))
        i_sd = i_sd[0]

        # Calculate and limit the i_sq current
        i_sq = np.clip(
            torque / max(psi, 0.001) * 2 / 3 / self.p * self.l_r / self.l_m, -self.i_sq_limit, self.i_sq_limit
        )
        if self.i_sd_limit < np.sqrt(i_sq**2 + i_sd**2):
            i_sq = np.sign(i_sq) * np.sqrt(self.i_sd_limit**2 - i_sd**2)

        return np.array([i_sd, i_sq])

    def reset(self):
        """Reset the SCIM operation point selection"""
        super().reset()
        self.psi_controller.reset()
