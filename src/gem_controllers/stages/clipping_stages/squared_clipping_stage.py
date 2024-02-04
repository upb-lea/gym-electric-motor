import numpy as np

import gem_controllers as gc

from .clipping_stage import ClippingStage


class SquaredClippingStage(ClippingStage):
    """This class clips multiple references together, by clipping the vector length of the references to a scalar limit."""

    @property
    def clipping_difference(self) -> np.ndarray:
        """Difference between the reference and the clipped reference"""
        return self._clipping_difference

    @property
    def limits(self):
        """Limits of the controlled states"""
        return self._limits

    @property
    def margin(self):
        """Margin of the controlled states"""
        return self._margin

    @property
    def action_range(self):
        """Action range of the controller stage"""
        return []

    def __init__(self, control_task="CC"):
        """
        Args:
            control_task(str): Control task of the controller stage.
        """

        self._clipping_difference = np.array([])
        self._margin = 0.0
        self._limits = np.array([])
        self._control_task = control_task

    def __call__(self, state, reference):
        """
        Clips a reference to the limits.

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference of the state.

        Returns:
            clipped_reference(np.ndarray): The reference of a controller stage clipped to the limit.
        """

        relative_reference_length = np.sum((reference / self._limits) ** 2)
        relative_maximum = 1 - self._margin
        clipped = (
            reference
            if relative_reference_length < relative_maximum**2
            else reference / relative_reference_length * relative_maximum
        )
        self._clipping_difference = reference - clipped
        return clipped

    def tune(self, env, env_id, margin=0.0, **kwargs):
        """
        Set the limits for the clipped states.

        Args:
            env(gym_electric_motor.ElectricMotorEnvironment): The environment to be controlled.
            env_id(str): The id of the environment.
            margin(float): Percentage, how far the value should be clipped below the limit.
        """

        motor_type = gc.utils.get_motor_type(env_id)
        state_names = []
        if self._control_task == "CC":
            state_names = gc.parameter_reader.voltages[motor_type]
        elif self._control_task == "TC":
            state_names = gc.parameter_reader.currents[motor_type]
        elif self._control_task == "SC":
            state_names = ["torque"]
        state_indices = [env.state_names.index(state_name) for state_name in state_names]
        self._limits = env.limits[state_indices]

    def reset(self):
        """Reset the squared clipping stage"""
        self._clipping_difference = np.zeros_like(self._limits)
