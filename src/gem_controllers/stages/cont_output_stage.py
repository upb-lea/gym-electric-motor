import numpy as np

import gem_controllers as gc

from .. import parameter_reader as reader
from .stage import Stage


class ContOutputStage(Stage):
    """This class normalizes continuous input voltages to the volatge limits."""

    @property
    def voltage_limit(self):
        """Voltage limit of the motor"""
        return self._voltage_limit

    @voltage_limit.setter
    def voltage_limit(self, value):
        self._voltage_limit = np.array(value, dtype=float)

    def __init__(self):
        super().__init__()
        self._voltage_limit = np.array([])

    def __call__(self, state, reference):
        """ "Divide the input voltages by the limits"""
        return reference / self.voltage_limit

    def tune(self, env, env_id, **_):
        """
        Set the volatage limits.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
        """
        action_type, _, motor_type = gc.utils.split_env_id(env_id)
        voltages = reader.get_output_voltages(motor_type, action_type)
        voltage_indices = [env.state_names.index(voltage) for voltage in voltages]
        self.voltage_limit = env.limits[voltage_indices]
