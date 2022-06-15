import gym
import pytest
import numpy as np
import gym_electric_motor as gem

from tests.testing_utils import DummyPhysicalSystem
from tests.utils import PhysicalSystemTestWrapper
from .test_physical_system_wrapper import TestPhysicalSystemWrapper


class TestFluxObserver(TestPhysicalSystemWrapper):

    @pytest.fixture
    def physical_system(self):
        ps = DummyPhysicalSystem(state_names=['omega','i_sa', 'i_sb', 'i_sc','i_sd', 'i_sq'])
        ps.unwrapped.electrical_motor = gem.physical_systems.electric_motors.SquirrelCageInductionMotor(
            limit_values=dict(i=20.0),
            motor_parameter=dict(l_m=10.0)
        )
        ps.unwrapped._limits[ps.state_names.index('i_sd')] = ps.unwrapped.electrical_motor.limits['i_sd']
        return ps

    @pytest.fixture
    def reset_physical_system(self, physical_system):
        ps.reset()
        return ps

    @pytest.fixture
    def processor(self, physical_system):
        return gem.physical_system_wrappers.FluxObserver(physical_system=physical_system)

    @pytest.fixture
    def reset_processor(self, processor):
        processor.reset()
        return processor

    def test_limits(self, processor, physical_system):
        assert all(processor.limits == np.concatenate((physical_system.limits, [200., np.pi])))

    def test_nominal_state(self, processor, physical_system):
        assert all(processor.nominal_state == np.concatenate((physical_system.nominal_state, [200., np.pi])))

    def test_state_space(self, processor, physical_system):
        psi_abs_max = 200.0
        low = np.concatenate((physical_system.state_space.low, [-psi_abs_max, -np.pi]))
        high = np.concatenate((physical_system.state_space.high, [psi_abs_max, np.pi]))
        space = gym.spaces.Box(low, high, dtype=np.float64)
        assert processor.state_space == space

    def test_reset(self, processor, physical_system):
        assert all(processor.reset() == np.concatenate((physical_system.state, [0., 0.])))

    @pytest.mark.parametrize('action', [1, 2, 3, 4])
    def test_simulate(self, reset_processor, physical_system, action):
        state = reset_processor.simulate(action)
        assert all(state[:-2] == physical_system.state)

