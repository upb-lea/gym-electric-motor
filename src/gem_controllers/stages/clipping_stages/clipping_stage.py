import numpy as np

import gym_electric_motor.core

from ..stage import Stage


class ClippingStage(Stage):
    """This is the base class for all clipping stages."""

    @property
    def clipping_difference(self) -> np.ndarray:
        """Difference between the reference and the clipped reference"""
        raise NotImplementedError

    @property
    def clipped(self) -> np.ndarray:
        """Flag, if the references have been clipped"""
        return self.clipping_difference != 0.0

    def __call__(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Clips a reference to the limits.

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference of the state.

        Returns:
            clipped_reference(np.ndarray): The reference of a controller stage clipped to the limit.
        """
        raise NotImplementedError

    def tune(self, env: gym_electric_motor.core.ElectricMotorEnvironment, env_id: str, margin: float = 0.0, **kwargs):
        """
        Set the limits for the clipped states.

        Args:
            env(gym_electric_motor.ElectricMotorEnvironment): The environment to be controlled.
            env_id(str): The id of the environment.
            margin(float): Percentage, how far the value should be clipped below the limit.
        """
        pass
