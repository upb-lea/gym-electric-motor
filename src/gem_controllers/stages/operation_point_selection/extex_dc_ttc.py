import numpy as np

import gem_controllers as gc

from ... import parameter_reader as reader
from .operation_point_selection import OperationPointSelection


class ExtExDcOperationPointSelection(OperationPointSelection):
    """This class comutes the current operation point of a ExtExDc Motor for a given torque reference value."""

    @property
    def cross_inductance(self):
        """Cross inducances of the ExtEx Dc motor"""
        return self._cross_inductance

    @cross_inductance.setter
    def cross_inductance(self, value):
        self._cross_inductance = np.array(value, dtype=float)

    @property
    def i_e_idx(self):
        """Index of the i_e current"""
        return self._i_e_idx

    @i_e_idx.setter
    def i_e_idx(self, value):
        self._i_e_idx = int(value)

    @property
    def i_a_idx(self):
        """Index of the i_a current"""
        return self._i_a_idx

    @i_a_idx.setter
    def i_a_idx(self, value):
        self._i_a_idx = int(value)

    @property
    def i_e_policy(self):
        """Policy for calculating the i_e current"""
        return self._i_e_policy

    @i_e_policy.setter
    def i_e_policy(self, value):
        assert callable(value), "The i_e_policy has to be a callable function."
        self._i_e_policy = value

    def __init__(self):
        super().__init__()
        self._cross_inductance = np.array([])
        self._i_e_idx = None
        self._i_a_idx = None
        self._r_a_sqrt = None
        self._r_e_sqrt = None
        self._l_e_prime = None
        self._i_e_policy = self.__i_e_policy

    def __i_e_policy(self, state, reference):
        """The policy for the exciting current that is used per default."""
        return np.sqrt(self._r_a_sqrt * abs(reference[0]) / (self._r_e_sqrt * self._l_e_prime))

    def _select_operating_point(self, state, reference):
        """
        Calculate the current refrence values.

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference of the state.

        Returns:
            np.array: current reference values
        """
        # Select the i_e reference
        i_e_ref = self._i_e_policy(state, reference)

        # Calculate the i_a reference
        i_a_ref = reference[0] / self._cross_inductance[0] / max(state[self._i_e_idx], 1e-4)
        return np.array([i_a_ref, i_e_ref])

    def tune(self, env, env_id, current_safety_margin=0.2):
        """
        Tune the operation point selcetion stage.

        Args:
            env(gym_electric_motor.ElectricMotorEnvironment): The environment to be controlled.
            env_id(str): The id of the environment.
            current_safety_margin(float): Percentage of the current margin to the current limit.
        """

        super().tune(env, env_id, current_safety_margin)
        motor_type = gc.utils.get_motor_type(env_id)
        self._i_e_idx = env.state_names.index("i_e")
        self._i_a_idx = env.state_names.index("i_a")
        self._cross_inductance = reader.l_prime_reader[motor_type](env)
        mp = env.physical_system.electrical_motor.motor_parameter
        self._r_a_sqrt = np.sqrt(mp["r_a"])
        self._r_e_sqrt = np.sqrt(mp["r_e"])
        self._l_e_prime = mp["l_e_prime"]
