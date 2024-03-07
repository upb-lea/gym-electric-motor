import numpy as np

import gem_controllers as gc

from .. import parameter_reader as reader
from .stage import Stage


class EMFFeedforward(Stage):
    """This class calculates the emf feedforward, to decouple the actions."""

    @property
    def inductance(self):
        """Inductances of the motor"""
        return self._inductance

    @inductance.setter
    def inductance(self, value):
        self._inductance = np.array(value)

    @property
    def psi(self):
        """Permanent magnet flux of the motor"""
        return self._psi

    @psi.setter
    def psi(self, value):
        self._psi = np.array(value)

    @property
    def current_indices(self):
        """Indices of the currents"""
        return self._current_indices

    @current_indices.setter
    def current_indices(self, value):
        self._current_indices = np.array(value)

    @property
    def omega_idx(self):
        """Index of the rotational speed omega"""
        return self._omega_idx

    @omega_idx.setter
    def omega_idx(self, value):
        self._omega_idx = int(value)

    @property
    def action_range(self):
        """Action range of the motor"""
        return self._action_range

    def omega_el(self, state):
        """
        Calculate the electrical speed.

        Args:
            state(np.array): state of the environment

        Returns:
            float: electrical speed
        """
        return state[self._omega_idx] * self._p

    def __init__(self):
        super().__init__()
        self._p = 1
        self._inductance = np.array([])
        self._psi = np.array([])
        self._current_indices = np.array([])
        self._omega_idx = None
        self._action_range = np.array([]), np.array([])

    def __call__(self, state, reference):
        """
        Calculate the emf feedforward voltages and add them to the actions of the current controller.

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference voltages.

        Returns:
            input voltages(np.ndarray): decoupled input voltages
        """
        action = reference + (self._inductance * state[self._current_indices] + self._psi) * self.omega_el(state)
        return action

    def tune(self, env, env_id, **_):
        """
        Set all needed motor parameters for the decoupling.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
        """

        motor_type = gc.utils.get_motor_type(env_id)
        omega_idx = env.state_names.index("omega")
        current_indices = [env.state_names.index(current) for current in reader.emf_currents[motor_type]]
        self.omega_idx = omega_idx
        self.current_indices = current_indices
        self.inductance = reader.l_emf_reader[motor_type](env)
        self.psi = reader.psi_reader[motor_type](env)
        self._p = reader.p_reader[motor_type](env)
        voltages = reader.voltages[motor_type]
        voltage_indices = [env.state_names.index(voltage) for voltage in voltages]
        voltage_limits = env.limits[voltage_indices]
        self._action_range = (
            env.observation_space[0].low[voltage_indices] * voltage_limits,
            env.observation_space[0].high[voltage_indices] * voltage_limits,
        )
