import gym
import numpy as np

from gym_electric_motor.state_action_processors import StateActionProcessor


class CurrentSumProcessor(StateActionProcessor):

    def __init__(self, currents, limit='max', physical_system=None):
        self._currents = currents
        assert limit in ['max', 'sum']
        self._limit = max if limit == 'max' else np.sum
        self._current_indices = []
        super().__init__(physical_system)

    def set_physical_system(self, physical_system):
        super().set_physical_system(physical_system)
        low = np.concatenate((physical_system.state_space.low, [-1.]))
        high = np.concatenate((physical_system.state_space.high, [1.]))
        self.state_space = gym.spaces.Box(low, high, dtype=np.float64)
        self._current_indices = [physical_system.state_positions[current] for current in self._currents]
        current_limit = self._limit(physical_system.limits[self._current_indices])
        current_nominal_value = self._limit(physical_system.nominal_state[self._current_indices])
        self._limits = np.concatenate((physical_system.limits, [current_limit]))
        self._nominal_state = np.concatenate((physical_system.nominal_state, [current_nominal_value]))
        self._state_names = physical_system.state_names + ['i_sum']
        self._state_positions = {key: index for index, key in enumerate(self._state_names)}
        return self

    def reset(self):
        state = self._physical_system.reset()
        return np.concatenate((state, [self._get_current_sum(state)]))

    def simulate(self, action):
        state = self._physical_system.simulate(action)
        return np.concatenate((state, [self._get_current_sum(state)]))

    def _get_current_sum(self, state):
        return np.sum(state[self._current_indices])
