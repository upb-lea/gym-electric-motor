from typing import Tuple

import numpy as np

import gem_controllers as gc

from . import ClippingStage


class CombinedClippingStage(ClippingStage):
    """This clipping stage combines the absolute clipping and the squared clipping."""

    @property
    def clipping_difference(self) -> np.ndarray:
        """Difference between the reference and the clipped reference"""
        return self._clipping_difference

    @property
    def action_range(self) -> Tuple[np.ndarray, np.ndarray]:
        """Action range of the controller stage"""
        action_range_low = np.zeros(len(self._squared_clipped_states) + len(self._absolute_clipped_states))
        action_range_high = np.zeros(len(self._squared_clipped_states) + len(self._absolute_clipped_states))
        action_range_low[-1] = self._action_range_absolute[0][0]
        action_range_high[-1] = self._action_range_absolute[1][0]
        return action_range_low, action_range_high

    def __init__(self, control_task="CC"):
        self._action_range_absolute = np.array([]), np.array([])
        self._limit_squred_clipping = np.array([])
        self._clipping_difference = np.array([])
        self._control_task = control_task
        self._absolute_clipped_states = np.array([])
        self._squared_clipped_states = np.array([])
        self._margin = None

    def __call__(self, state, reference):
        clipped = np.zeros(np.size(self._squared_clipped_states) + np.size(self._absolute_clipped_states))
        relative_reference_length = np.sum((reference[self._squared_clipped_states] / self._limit_squred_clipping) ** 2)
        relative_maximum = 1 - self._margin
        clipped[self._squared_clipped_states] = (
            reference[self._squared_clipped_states]
            if relative_reference_length < relative_maximum**2
            else reference[self._squared_clipped_states] / relative_reference_length * relative_maximum
        )

        clipped[self._absolute_clipped_states] = np.clip(
            reference[self._absolute_clipped_states], self._action_range_absolute[0], self._action_range_absolute[1]
        )

        self._clipping_difference = reference - clipped
        return clipped

    def tune(
        self, env, env_id, margin=0.0, squared_clipped_state=np.array([0, 1]), absoulte_clipped_states=np.array([2])
    ):
        """
        Set the limits for the clipped states.

        Args:
            env(gym_electric_motor.ElectricMotorEnvironment): The environment to be controlled.
            env_id(str): The id of the environment.
            margin(float): Percentage, how far the value should be clipped below the limit.
            squared_clipped_state(np.ndarray): Indices of the squared clipped states.
            absoulte_clipped_states(np.ndarray): Indices of the absolute clipped states.
        """

        self._squared_clipped_states = squared_clipped_state
        self._absolute_clipped_states = absoulte_clipped_states
        self._margin = margin

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
        self._limit_squred_clipping = limits[squared_clipped_state]
        self._action_range_absolute = (
            lower_action_limit[absoulte_clipped_states],
            upper_action_limit[absoulte_clipped_states],
        )
        self._clipping_difference = np.zeros_like(lower_action_limit)

    def reset(self):
        """Reset the combined clipping stage"""
        self._clipping_difference = np.zeros(
            np.size(self._squared_clipped_states) + np.size(self._absolute_clipped_states)
        )
