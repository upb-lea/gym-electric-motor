import gym
import numpy as np

import gym_electric_motor.physical_systems as ps
from gym_electric_motor.state_action_processors import StateActionProcessor


class DqToAbcActionProcessor(StateActionProcessor):

    @property
    def action_space(self):
        return gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float64)

    @staticmethod
    def _transformation(action, angle):
        return ps.ThreePhaseMotor.t_32(ps.ThreePhaseMotor.q_inv(action, angle))

    def __init__(self, angle_name=None, physical_system=None):
        self._angle = 0.0
        self._angle_index = None
        self._omega_index = None
        self._state = None
        self._angle_name = angle_name
        self._angle_advance = 0.0
        super().__init__(physical_system)

    def set_physical_system(self, physical_system):
        assert isinstance(physical_system.unwrapped, ps.SCMLSystem)
        assert isinstance(physical_system.electrical_motor, ps.ThreePhaseMotor)
        super().set_physical_system(physical_system)
        if self._angle_name is None:
            if isinstance(physical_system.electrical_motor,
                          (ps.PermanentMagnetSynchronousMotor, ps.SynchronousReluctanceMotor)):
                self._angle_name = 'epsilon'
            else:
                self._angle_name = 'psi_angle'
        assert self._angle_name in physical_system.state_names, \
            f'Angle {self._angle_name} not in the states of the physical system. Probably, a flux observer is required.'
        self._angle_index = physical_system.state_names.index(self._angle_name)
        self._omega_index = physical_system.state_names.index('omega')
        if hasattr(physical_system.converter, 'dead_time') \
                and physical_system.converter.dead_time is True:
            self._angle_advance = 1.5
        else:
            self._angle_advance = 0.5

        return self

    def simulate(self, action):
        advanced_angle = self._state[self._angle_index] * self._physical_system.limits[self._angle_index] \
            + self._angle_advance * self._physical_system.tau \
            * self._state[self._omega_index] * self._physical_system.limits[self._omega_index]
        abc_action = self._transformation(action, advanced_angle)
        self._state = self._physical_system.simulate(abc_action)
        return self._state

    def reset(self, **kwargs):
        self._state = self._physical_system.reset()
        return self._state
