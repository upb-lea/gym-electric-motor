import numpy as np

from .stage import Stage


class InputStage(Stage):
    """This class denormalizes the state and reference."""

    @property
    def state_limits(self):
        """Limits of the states"""
        return self._state_limits

    @state_limits.setter
    def state_limits(self, value):
        self._state_limits = np.array(value)

    @property
    def reference_limits(self):
        """Limits of the references"""
        return self._reference_limits

    @reference_limits.setter
    def reference_limits(self, value):
        self._reference_limits = np.array(value)

    def __init__(self):
        super().__init__()
        self._state_limits = np.array([])
        self._reference_limits = np.array([])

    def __call__(self, state, reference):
        """
        Denormalize the state and the references

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference values at the input.

        Returns:
            np.array: denormalized reference values
        """

        state[:] = state * self._state_limits
        return reference * self._reference_limits

    def tune(self, env, env_id, **__):
        """
        Set the limits of the state and the references.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
        """

        self._state_limits = env.limits
        reference_indices = [env.state_names.index(reference) for reference in env.reference_names]
        self.reference_limits = env.limits[reference_indices]
