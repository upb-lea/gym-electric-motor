import numpy as np

import gem_controllers as gc

from ... import parameter_reader as reader
from .base_controller import BaseController
from .e_base_controller_task import EBaseControllerTask


class ThreePointController(BaseController):
    """This class represents a three point controller, that can be used for discrete action spaces."""

    @property
    def high_action(self):
        """High action value of the three point controller"""
        return self._high_action

    @high_action.setter
    def high_action(self, value):
        self._high_action = np.array(value)

    @property
    def low_action(self):
        """Low action value of the three point controller"""
        return self._low_action

    @low_action.setter
    def low_action(self, value):
        self._low_action = np.array(value)

    @property
    def idle_action(self):
        """Idle action value of the three point controller"""
        return self._idle_action

    @idle_action.setter
    def idle_action(self, value):
        self._idle_action = np.array(value)

    @property
    def referenced_state_indices(self):
        """Indices of the controlled states"""
        return self._referenced_state_indices

    @referenced_state_indices.setter
    def referenced_state_indices(self, value):
        self._referenced_state_indices = np.array(value)

    @property
    def hysteresis(self):
        """Value of the hysteresis level"""
        return self._hysteresis

    @hysteresis.setter
    def hysteresis(self, value):
        self._hysteresis = np.array(value)

    @property
    def action_range(self):
        """Action range of the base controller"""
        return self._action_range

    @action_range.setter
    def action_range(self, value):
        self._action_range = value

    def __init__(self, control_task):
        """
        Args:
            control_task(str): Control task of the three point controller
        """
        super().__init__(control_task)
        self._hysteresis = np.array([])
        self._referenced_state_indices = np.array([])
        self._idle_action = 0.0
        self._high_action = np.array([])
        self._low_action = np.array([])
        self._action_range = (np.array([]), np.array([]))

    def __call__(self, state, reference):
        """
        Select one of the three actions.

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference of the state.

        Returns:
            action(np.ndarray): Action or reference for the next stage
        """
        referenced_states = state[self._referenced_state_indices]
        high_actions = referenced_states + self._hysteresis < reference
        low_actions = referenced_states - self._hysteresis > reference
        return np.select([low_actions, high_actions], [self._low_action, self._high_action], default=self._idle_action)

    def tune(self, env, env_id, **base_controller_kwargs):
        """
        Tune a three point controller stage.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
        """

        if self._control_task == EBaseControllerTask.SC:
            self._tune_speed_controller(env, env_id)
        elif self._control_task == EBaseControllerTask.CC:
            self._tune_current_controller(env, env_id)
        else:
            raise Exception(f"No tuner available for control_task {self._control_task}.")

    def _tune_current_controller(self, env, env_id):
        """
        Calculate the hysteresis levels of the current control stage and set the action values.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
        """

        motor_type = gc.utils.get_motor_type(env_id)
        voltages = reader.voltages[motor_type]
        currents = reader.currents[motor_type]
        voltage_indices = [env.state_names.index(voltage) for voltage in voltages]
        current_indices = [env.state_names.index(current) for current in currents]
        self.referenced_state_indices = current_indices
        voltage_limits = env.limits[voltage_indices]

        action_range = (
            env.observation_space[0].low[voltage_indices] * voltage_limits,
            env.observation_space[0].high[voltage_indices] * voltage_limits,
        )
        self.action_range = action_range
        # Todo: Calculate Hysteresis based on the dynamics of the underlying control plant
        self.hysteresis = 0.01 * (action_range[1] - action_range[0])
        self.high_action = action_range[1]
        self.low_action = action_range[0]
        self.idle_action = np.zeros_like(action_range[1])

    def _tune_speed_controller(self, env, _env_id):
        """
        Calculate the hysteresis levels of the speed control stage and set the torque reference values.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            _env_id(str): The corresponding environment-id to specify the concrete environment.
        """

        torque_index = [env.state_names.index("reference")]
        torque_limit = env.limits[torque_index]
        self.referenced_state_indices = torque_index
        action_range = (
            env.observation_space[0].low[torque_index] * torque_limit,
            env.observation_space[0].high[torque_index] * torque_limit,
        )
        self.action_range = action_range
        # Todo: Calculate Hysteresis based on the dynamics of the underlying control plant
        self.hysteresis = 0.01 * (action_range[1] - action_range[0])
        self.high_action = action_range[1]
        self.low_action = action_range[0]
        self.idle_action = np.zeros_like(action_range[1])
