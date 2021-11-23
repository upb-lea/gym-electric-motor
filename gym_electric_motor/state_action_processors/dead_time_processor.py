from collections import deque
import numpy as np

import gym.spaces
from gym_electric_motor.state_action_processors import StateActionProcessor


class DeadTimeProcessor(StateActionProcessor):

    @property
    def dead_time(self):
        return True

    def __init__(self, steps=1, reset_action=None, physical_system=None):
        self._reset_action = reset_action
        self._action_deque = deque(maxlen=steps)
        super().__init__(physical_system)

    def set_physical_system(self, physical_system):

        if self._reset_action is None:
            action_space = physical_system.action_space
            if isinstance(action_space, gym.spaces.Discrete):
                self._reset_action = 0
            elif isinstance(action_space, gym.spaces.MultiDiscrete):
                self._reset_action = np.zeros_like(action_space.nvec)
            elif isinstance(action_space, gym.spaces.Box):
                self._reset_action = np.zeros(action_space.shape, dtype=np.float64)
            else:
                raise AssertionError(
                    f'Action Space {action_space} of type {type(action_space)} unsupported.'
                    'Only Discrete / MultiDiscrete and Box allowed for the dead time processor.'
                )

    def reset(self):
        state = super().reset()
        self._action_deque.clear()
        self._action_deque.extend([self._reset_action] * self._action_deque.maxlen)
        return state

    def simulate(self, action):
        active_action = self._action_deque.pop()
        self._action_deque.appendleft(action)
        return self._physical_system.simulate(active_action)


