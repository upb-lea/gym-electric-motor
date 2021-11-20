import gym
import numpy as np

import gym_electric_motor as gem
from gym_electric_motor.state_action_processors import StateActionProcessor


class FluxObserver(StateActionProcessor):

    def __init__(self, current_names=('i_sd', 'i_sq'), physical_system=None):
        super(FluxObserver, self).__init__(physical_system)
        self._current_indices = None
        self._current_names = current_names

    def set_physical_system(self, physical_system):
        assert isinstance(physical_system.electrical_motor, gem.physical_systems.electric_motors.InductionMotor)
        super().set_physical_system(physical_system)
        low = np.concatenate((physical_system.state_space.low, [-1., -1.]))
        high = np.concatenate((physical_system.state_space.high, [1., 1.]))
        self.state_space = gym.spaces.Box(low, high, dtype=np.float64)
        self._current_indices = [physical_system.state_positions[name] for name in self._current_names]
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
