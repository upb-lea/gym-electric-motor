import gym
import numpy as np

from gym_electric_motor.physical_system_wrappers import PhysicalSystemWrapper


class CosSinProcessor(PhysicalSystemWrapper):
    """Adds ``cos(angle)`` and ``sin(angle)`` states to the systems state vector that are the cosine and sine of a
    certain systems state.

    Optionally, the CosSinProcessor can also remove the angle from the state vector.
    """

    @property
    def angle(self):
        """Returns the name of the state whose cosine and sine are appended to the state vector."""
        return self._angle

    def __init__(self, angle='epsilon', physical_system=None, remove_angle=False):
        """
        Args:
            angle(string): Name of the state whose cosine and sine will be added to the systems state vector.
                Default: ``'epsilon'``
            physical_system(PhysicalSystem(optional)): Inner system of this processor.
            remove_angle(bool): Remove the angle from the state vector
        """
        self._angle = angle
        self._angle_index = None
        self._remove_angle = remove_angle
        self._remove_idx = []
        super().__init__(physical_system)

    def set_physical_system(self, physical_system):
        # Docstring of super class
        super().set_physical_system(physical_system)
        self._angle_index = physical_system.state_positions[self._angle]

        self._remove_idx = self._angle_index if self._remove_angle else []
        low = np.concatenate((np.delete(physical_system.state_space.low, self._remove_idx), [-1., -1.]))
        high = np.concatenate((np.delete(physical_system.state_space.high, self._remove_idx), [1., 1.]))
        self.state_space = gym.spaces.Box(low, high, dtype=np.float64)

        self._limits = np.concatenate((np.delete(physical_system.limits, self._remove_idx), [1., 1.]))
        self._nominal_state = np.concatenate((np.delete(physical_system.nominal_state, self._remove_idx), [1., 1.]))
        self._state_names = list(np.delete(physical_system.state_names, self._remove_idx)) \
            + [f'cos({self._angle})', f'sin({self._angle})']
        self._state_positions = {key: index for index, key in enumerate(self._state_names)}
        return self

    def reset(self):
        # Docstring of super class
        state = self._physical_system.reset()
        return self._get_cos_sin(state)

    def simulate(self, action):
        # Docstring of super class
        state = self._physical_system.simulate(action)
        if self._remove_angle:
            return self._delete_angle(self._get_cos_sin(state))
        return self._get_cos_sin(state)

    def _delete_angle(self, state):
        """Removes the angle from the state vector

        Args:
            state(numpy.ndarray[float]): The state vector of the system.

        Returns:
            numpy.ndarray[float]: The state vector removed by the angle.
                """
        return np.delete(state, self._remove_idx)

    def _get_cos_sin(self, state):
        """Appends the cosine and sine of the specified state to the state vector.

        Args:
            state(numpy.ndarray[float]): The state vector of the system.

        Returns:
            numpy.ndarray[float]: The state vector extended by cosine and sine.
        """
        return np.concatenate((state, [np.cos(state[self._angle_index]), np.sin(state[self._angle_index])]))
