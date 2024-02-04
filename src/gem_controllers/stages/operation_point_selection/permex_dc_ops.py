import numpy as np

import gem_controllers as gc

from ... import parameter_reader as reader
from .operation_point_selection import OperationPointSelection


class PermExDcOperationPointSelection(OperationPointSelection):
    """This class computes the current operation point of a PermExDx Motor for a given torque reference value."""

    @property
    def magnetic_flux(self):
        """Permanent magnetic flux of the PermEx Dc motor"""
        return self._magnetic_flux

    @magnetic_flux.setter
    def magnetic_flux(self, value):
        self._magnetic_flux = np.array(value, dtype=float)

    @property
    def voltage_limit(self):
        """Voltage limit of the the PermEx Dc motor"""
        return self._voltage_limit

    @voltage_limit.setter
    def voltage_limit(self, value):
        self._voltage_limit = np.array(value, dtype=float)

    @property
    def omega_index(self):
        """Index of the rotational speeda"""
        return self._omega_index

    @omega_index.setter
    def omega_index(self, value):
        self._omega_index = int(value)

    @property
    def resistance(self):
        """Ohmic resistance of the PermEx Dc motor"""
        return self._resistance

    @resistance.setter
    def resistance(self, value):
        self._resistance = np.array(value, dtype=float)

    def __init__(self):
        super().__init__()
        self._magnetic_flux = np.array([])
        self._voltage_limit = np.array([])
        self._resistance = np.array([])
        self._omega_index = 0

    def _select_operating_point(self, state, reference):
        """
        Calculate the current refrence values.

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference of the state.

        Returns:
            np.array: current reference values
        """

        if state[self._omega_index] > 0:
            return min(reference / self._magnetic_flux, self._max_current_per_speed(state))
        else:
            return max(reference / self._magnetic_flux, -self._max_current_per_speed(state))

    def _max_current_per_speed(self, state):
        """Calculate the maximum current for a given speed."""
        return self._voltage_limit / (self._resistance + self._magnetic_flux * abs(state[self._omega_index]))

    def tune(self, env, env_id, current_safety_margin=0.2):
        """
        Tune the operation point selcetion stage.

        Args:
            env(gym_electric_motor.ElectricMotorEnvironment): The environment to be controlled.
            env_id(str): The id of the environment.
            current_safety_margin(float): Percentage of the current margin to the current limit.
        """
        super().tune(env, env_id, current_safety_margin=current_safety_margin)
        motor = gc.utils.get_motor_type(env_id)
        self._magnetic_flux = reader.psi_reader[motor](env)
        voltages = reader.voltages[motor]
        voltage_indices = [env.state_names.index(voltage) for voltage in voltages]
        self._voltage_limit = env.limits[voltage_indices]
        self._omega_index = env.state_names.index("omega")
