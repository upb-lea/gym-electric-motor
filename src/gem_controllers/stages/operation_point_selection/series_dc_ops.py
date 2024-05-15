import numpy as np

import gem_controllers as gc

from ... import parameter_reader as reader
from .operation_point_selection import OperationPointSelection


class SeriesDcOperationPointSelection(OperationPointSelection):
    """This class computes the current operation point of a SeriesDc Motor for a given torque reference value."""

    @property
    def cross_inductance(self):
        """Cross inductance of the Series Dc motor"""
        return self._cross_inductance

    @cross_inductance.setter
    def cross_inductance(self, value):
        self._cross_inductance = np.array(value, dtype=float)

    def __init__(self):
        super().__init__()
        self._cross_inductance = np.array([])

    def _select_operating_point(self, state, reference):
        """
        Calculate the current refrence values.

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference of the state.

        Returns:
            np.array: current reference values
        """
        return np.sqrt(reference / self._cross_inductance)

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
        self._cross_inductance = reader.l_prime_reader[motor](env)
