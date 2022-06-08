import gym
import pytest
import numpy as np
import gym_electric_motor as gem

from tests.testing_utils import DummyPhysicalSystem
from .test_physical_system_wrapper import TestPhysicalSystemWrapper


class TestCosSinProcessor(TestPhysicalSystemWrapper):

    @pytest.fixture
    def processor(self, physical_system):
        return gem.physical_system_wrappers.CosSinProcessor(angle='dummy_state_0', physical_system=physical_system)

    def test_limits(self, processor, physical_system):
        assert all(processor.limits == np.concatenate((physical_system.limits, [1., 1.])))

    def test_nominal_state(self, processor, physical_system):
        assert all(processor.nominal_state == np.concatenate((physical_system.nominal_state, [1., 1.])))

    def test_state_space(self, processor, physical_system):
        low = np.concatenate((physical_system.state_space.low, [-1, -1]))
        high = np.concatenate((physical_system.state_space.high, [1, 1]))
        space = gym.spaces.Box(low, high)
        assert processor.state_space == space

    def test_reset(self, processor, physical_system):
        assert all(processor.reset() == np.concatenate((physical_system.state, [1., 0.])))

    @pytest.mark.parametrize('action', [1, 2, 3, 4])
    def test_simulate(self, processor, physical_system, action):
        state = processor.simulate(action)
        ps_state = physical_system.state
        assert action == physical_system.action
        cos_sin_state = ps_state[physical_system.state_positions[processor.angle]]
        assert all(state == np.concatenate((ps_state, [np.cos(cos_sin_state), np.sin(cos_sin_state)])))

