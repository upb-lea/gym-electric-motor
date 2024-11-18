from typing import Tuple

import numpy as np

import gem_controllers as gc

from .clipping_stage import ClippingStage


class AbsoluteClippingStage(ClippingStage):
    """This class clips a reference absolute to the limit of the corresponding limit of the state"""

    @property
    def clipping_difference(self) -> np.ndarray:
        """Difference between the reference and the clipped reference"""
        return self._clipping_difference

    @property
    def action_range(self) -> Tuple[np.ndarray, np.ndarray]:
        """Action range of the controller stage"""
        return self._action_range

    def __init__(self, control_task="CC"):
        """
        Args:
            control_task(str): Control task of the controller stage.
        """
        self._action_range = np.array([]), np.array([])
        self._clipping_difference = np.array([])
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

        clipped = np.clip(reference, self._action_range[0], self._action_range[1])
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
        if self._control_task == "CC":
            action_names = gc.parameter_reader.voltages[motor_type]
        elif self._control_task == "TC":
            action_names = gc.parameter_reader.currents[motor_type]
        elif self._control_task == "SC":
            action_names = ["torque"]
        else:
            raise AttributeError(f"Control task is {self._control_task} but has to be one of [SC, TC, CC].")
        action_indices = [env.state_names.index(action_name) for action_name in action_names]
        limits = env.limits[action_indices] * (1 - margin)
        state_space = env.observation_space[0]
        lower_action_limit = state_space.low[action_indices] * limits
        upper_action_limit = state_space.high[action_indices] * limits
        self._action_range = lower_action_limit, upper_action_limit
        self._clipping_difference = np.zeros_like(lower_action_limit)

    def reset(self):
        """Reset the absolute clipping stage"""
        self._clipping_difference = np.zeros_like(self._action_range[0])
