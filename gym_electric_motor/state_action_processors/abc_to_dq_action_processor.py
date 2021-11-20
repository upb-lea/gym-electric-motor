import gym
import numpy as np

from gym_electric_motor.physical_systems.physical_systems import SCMLSystem
from gym_electric_motor.physical_systems.electric_motors import ThreePhaseMotor
from gym_electric_motor.state_action_processors import StateActionProcessor


class AbcToDqActionProcessor(StateActionProcessor):

    @property
    def action_space(self):
        return gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float64)

    def __init__(self, physical_system=None):
        self._angle = 0.0
        self._angle_index = None
        super().__init__(physical_system)

    def set_physical_system(self, physical_system):
        assert isinstance(physical_system.unwrapped, SCMLSystem)
        assert isinstance(physical_system.electrical_motor, ThreePhaseMotor)
        super().set_physical_system(physical_system)
        self._angle_index = physical_system.state_names.index('epsilon')
        return self

    def reset(self):
        state = self._physical_system.reset()
        return np.concatenate((state, [self._get_current_sum(state)]))

    def simulate(self, action):
        state = self._physical_system.simulate(action)
        return np.concatenate((state, [self._get_current_sum(state)]))

    def _get_epsilon(self, state):
        return state[self._epsilon_idx]
