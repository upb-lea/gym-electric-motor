import gym
import numpy as np

from gym_electric_motor.state_action_processors import StateActionProcessor


class CosSinProcessor(StateActionProcessor):
    """Adds ``cos(angle)`` and ``sin(angle)`` states to the systems state vector that are the cosine and sine of a
    certain systems state.
    """

    @property
    def angle(self):
        """Returns the name of the state whose cosine and sine are appended to the state vector."""
        return self._angle

    def __init__(self, angle='epsilon', physical_system=None):
        """
        Args:
            angle(string): Name of the state whose cosine and sine will be added to the systems state vector.
                Default: ``'epsilon'``
            physical_system(PhysicalSystem(optional)): Inner system of this processor.
        """
        self._angle = angle
        self._angle_index = None
        super().__init__(physical_system)

    def set_physical_system(self, physical_system):
        # Docstring of super class
        super().set_physical_system(physical_system)
        low = np.concatenate((physical_system.state_space.low, [-1., -1.]))
        high = np.concatenate((physical_system.state_space.high, [1., 1.]))
        self.state_space = gym.spaces.Box(low, high, dtype=np.float64)
        self._angle_index = physical_system.state_positions[self._angle]
        self._limits = np.concatenate((physical_system.limits, [1., 1.]))
        self._nominal_state = np.concatenate((physical_system.nominal_state, [1., 1.]))
        self._state_names = physical_system.state_names + [f'cos({self._angle})', f'sin({self._angle})']
        self._state_positions = {key: index for index, key in enumerate(self._state_names)}
        return self

    def reset(self):
        # Docstring of super class
        state = self._physical_system.reset()
        return self._get_cos_sin(state)

    def simulate(self, action):
        # Docstring of super class
        state = self._physical_system.simulate(action)
        return self._get_cos_sin(state)

    def _get_cos_sin(self, state):
        """Appends the cosine and sine of the specified state to the state vector.

        Args:
            state(numpy.ndarray[float]): The state vector of the system.

        Returns:
            numpy.ndarray[float]: The state vector extended by cosine and sine.
        """
        return np.concatenate((state, [np.cos(state[self._angle_index]), np.sin(state[self._angle_index])]))
