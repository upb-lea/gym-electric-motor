import pytest
import numpy as np

from gym_electric_motor.constraints import LimitConstraint
from ..testing_utils import DummyPhysicalSystem


class TestLimitConstraint:

    @pytest.mark.parametrize(['ps', 'observed_state_names', 'expected_state_names'], [
        [DummyPhysicalSystem(2), ['all_states'], ['dummy_state_0', 'dummy_state_1']],
        [DummyPhysicalSystem(3), ['dummy_state_0', 'dummy_state_2'], ['dummy_state_0', 'dummy_state_2']]
    ])
    def test_initialization(self, ps, observed_state_names, expected_state_names):
        lc = LimitConstraint(observed_state_names)
        lc.set_modules(ps)
        assert lc._observed_state_names == expected_state_names
        assert np.all(lc._observed_states == np.array([state in expected_state_names for state in ps.state_names]))

    @pytest.mark.parametrize(['ps', 'observed_state_names', 'state', 'expected_violation'], [
        [DummyPhysicalSystem(2), ['all_states'], np.array([0.0, 1.1]), 1.0],
        [DummyPhysicalSystem(2), ['all_states'], np.array([0.0, 0.9]), 0.0],
        [DummyPhysicalSystem(2), ['all_states'], np.array([0.0, 1.0]), 0.0],
        [DummyPhysicalSystem(2), ['all_states'], np.array([-1.1, 0.9]), 1.0],
        [DummyPhysicalSystem(2), ['all_states'], np.array([-0.1, 0.9]), 0.0],
        [DummyPhysicalSystem(2), ['all_states'], np.array([-1.1, 1.1]), 1.0],
        [DummyPhysicalSystem(2), ['all_states'], np.array([-1.0, 1.0]), 0.0],
        [DummyPhysicalSystem(3), ['dummy_state_0', 'dummy_state_2'], np.array([0.0, 1.1, 0.0]), 0.0],
        [DummyPhysicalSystem(3), ['dummy_state_0', 'dummy_state_2'], np.array([-1.0, 1.1, 0.0]), 0.0],
        [DummyPhysicalSystem(3), ['dummy_state_0', 'dummy_state_2'], np.array([-1.1, 1.1, 0.0]), 1.0],
    ])
    def test_call(self, ps, observed_state_names, state, expected_violation):
        lc = LimitConstraint(observed_state_names)
        lc.set_modules(ps)
        violation = lc(state)
        assert violation == expected_violation
