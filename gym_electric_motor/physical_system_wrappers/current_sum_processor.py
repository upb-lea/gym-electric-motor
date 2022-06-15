import gym
import numpy as np

from gym_electric_motor.physical_system_wrappers import PhysicalSystemWrapper


class CurrentSumProcessor(PhysicalSystemWrapper):
    """Adds an ``i_sum`` state to the systems state vector that adds up currents."""

    def __init__(self, currents, limit='max', physical_system=None):
        """
        Args:
            currents(Iterable[string]): Iterable of the names of the currents to be summed up.
            limit('max'/'sum'): Selection if the limit and nominal values of ``i_sum`` are calculated by the maximum of
                the limits / nominal values of the source currents or their summation.
            physical_system(PhysicalSystem(optional)): The inner PhysicalSystem of this processor.
        """
        self._currents = currents
        assert limit in ['max', 'sum']
        self._limit = max if limit == 'max' else np.sum
        self._current_indices = []
        super().__init__(physical_system)

    def set_physical_system(self, physical_system):
        # Docstring of superclass
        super().set_physical_system(physical_system)
        self._current_indices = [physical_system.state_positions[current] for current in self._currents]

        # Define the new state space as concatenation of the old state space and [-1,1] for i_sum
        low = np.concatenate((physical_system.state_space.low, [-1.]))
        high = np.concatenate((physical_system.state_space.high, [1.]))
        self.state_space = gym.spaces.Box(low, high, dtype=np.float64)

        # Set the new limits /  nominal values of the state vector
        current_limit = self._limit(physical_system.limits[self._current_indices])
        current_nominal_value = self._limit(physical_system.nominal_state[self._current_indices])
        self._limits = np.concatenate((physical_system.limits, [current_limit]))
        self._nominal_state = np.concatenate((physical_system.nominal_state, [current_nominal_value]))

        # Append the new state to the state name vector and the state positions dictionary
        self._state_names = physical_system.state_names + ['i_sum']
        self._state_positions = physical_system.state_positions.copy()
        self._state_positions['i_sum'] = self._state_names.index('i_sum')
        return self

    def reset(self):
        # Docstring of superclass
        state = self._physical_system.reset()
        return np.concatenate((state, [self._get_current_sum(state)]))

    def simulate(self, action):
        # Docstring of superclass
        state = self._physical_system.simulate(action)
        return np.concatenate((state, [self._get_current_sum(state)]))

    def _get_current_sum(self, state):
        """Calculates the sum of the currents from the state

        Args:
            state(numpy.ndarray[float]): The state of the inner system.

        Returns:
            float: The summation of the currents of the state.
        """
        return np.sum(state[self._current_indices])
