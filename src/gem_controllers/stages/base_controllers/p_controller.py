import numpy as np

import gem_controllers as gc

from ... import parameter_reader as reader
from .base_controller import BaseController, EBaseControllerTask


class PController(BaseController):
    """This class represents an proportional controller, which can be combined e.g. with a integration controller to a
    PI controller.
    """

    @property
    def p_gain(self):
        """P gain of the P controller"""
        return self._p_gain

    @p_gain.setter
    def p_gain(self, value):
        self._p_gain = value

    @property
    def state_indices(self):
        """Indices of the controlled states"""
        return self._state_indices

    @state_indices.setter
    def state_indices(self, value):
        self._state_indices = np.array(value)

    @property
    def action_range(self):
        """Action range of the base controller"""
        return self._action_range

    @action_range.setter
    def action_range(self, value):
        self._action_range = value

    def __call__(self, state, reference):
        """
        Calculate the reference for the underlying stage

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference of the state.

        Returns:
             np.array: reference values of the next stage
        """
        return self.control(state, reference)

    def __init__(self, control_task, p_gain=np.array([0.0]), action_range=(np.array([0.0]), np.array([0.0]))):
        """
        Args:
            control_task(str): Control task of the P controller.
            p_gain(np.array): Array of p gains of the P controller.
            action_range(np.array): Action range of the stage.
        """
        BaseController.__init__(self, control_task)
        self._p_gain = p_gain
        self._action_range = action_range
        self._state_indices = np.array([])

    def _control(self, state, reference):
        """Multiply the proportional gain by the current error to get the action value."""
        return self._p_gain * (reference - state)

    def control(self, state, reference):
        """
        Calculate the reference for the underlying stage

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference of the state.

        Returns:
             np.array: reference values of the next stage
        """
        return self._control(state[self._state_indices], reference)

    def tune(self, env, env_id, a=4):
        """
        Tune the controller for the desired control task.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
            a(float): Design parameter of the symmetrical optimum.
        """

        if self._control_task == EBaseControllerTask.CurrentControl:
            self._tune_current_controller(env, env_id, a)
        elif self._control_task == EBaseControllerTask.SpeedControl:
            self._tune_speed_controller(env, env_id, a)
        else:
            raise Exception(f"No Tuner available for control task{self._control_task}.")

    def _tune_current_controller(self, env, env_id, a):
        """
        Tune the P controller for the current control by the symmetrical optimum.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
            a(float): Design parameter of the symmetrical optimum.
        """

        action_type, control_task, motor_type = gc.utils.split_env_id(env_id)
        l_ = reader.l_reader[motor_type](env)
        tau = env.physical_system.tau
        currents = reader.currents[motor_type]
        voltages = reader.voltages[motor_type]
        voltage_indices = [env.state_names.index(voltage) for voltage in voltages]
        current_indices = [env.state_names.index(current) for current in currents]
        voltage_limits = env.limits[voltage_indices]
        self.p_gain = l_ / (tau * a)

        self.action_range = (
            env.observation_space[0].low[voltage_indices] * voltage_limits,
            env.observation_space[0].high[voltage_indices] * voltage_limits,
        )
        self.state_indices = current_indices

    def _tune_speed_controller(self, env, env_id, a=4, t_n=None):
        """
        Tune the P controller for the speed control by the symmetrical optimum.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
            a(float): Design parameter of the symmetrical optimum.
            t_n(float): Time constant of the underlying torque controller.
        """

        if t_n is None:
            t_n = env.physical_system.tau
        j_total = env.physical_system.mechanical_load.j_total
        torque_index = env.state_names.index("torque")
        speed_index = env.state_names.index("omega")
        torque_limit = env.limits[torque_index]
        p_gain = j_total / (a * t_n)
        self.p_gain = np.array([p_gain])
        self.state_indices = [speed_index]
        self.action_range = (
            env.observation_space[0].low[[torque_index]] * np.array([torque_limit]),
            env.observation_space[0].high[[torque_index]] * np.array([torque_limit]),
        )
