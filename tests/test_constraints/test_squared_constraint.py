import pytest
import numpy as np

from gym_electric_motor.constraints import SquaredConstraint
from ..testing_utils import DummyPhysicalSystem


class TestSquaredConstraint:

    @pytest.mark.parametrize(['ps', 'observed_state_names', 'expected_state_names'], [
        [DummyPhysicalSystem(2), ['dummy_state_1'], ['dummy_state_1']],
        [DummyPhysicalSystem(3), ['dummy_state_0', 'dummy_state_2'], ['dummy_state_0', 'dummy_state_2']]
    ])
    def test_initialization(self, ps, observed_state_names, expected_state_names):
        sc = SquaredConstraint(observed_state_names)
        sc.set_modules(ps)
        assert sc._states == expected_state_names

    @pytest.mark.parametrize(['ps', 'observed_state_names', 'state', 'expected_violation'], [
        [DummyPhysicalSystem(2), ['dummy_state_0', 'dummy_state_1'], np.array([0.8, 0.8]), 1.0],
        [DummyPhysicalSystem(2), ['dummy_state_0', 'dummy_state_1'], np.array([0.0, 0.9]), 0.0],
        [DummyPhysicalSystem(2), ['dummy_state_0', 'dummy_state_1'], np.array([0.0, 1.0]), 0.0],
        [DummyPhysicalSystem(2), ['dummy_state_0', 'dummy_state_1'], np.array([-0.1, 1.0]), 1.0],
        [DummyPhysicalSystem(2), ['dummy_state_0', 'dummy_state_1'], np.array([-1.1, 0.9]), 1.0],
        [DummyPhysicalSystem(2), ['dummy_state_0', 'dummy_state_1'], np.array([-0.1, 0.9]), 0.0],
        [DummyPhysicalSystem(2), ['dummy_state_0', 'dummy_state_1'], np.array([-1.1, 1.1]), 1.0],
        [DummyPhysicalSystem(2), ['dummy_state_0', 'dummy_state_1'], np.array([-1.0, 1.0]), 1.0],
        [DummyPhysicalSystem(3), ['dummy_state_0', 'dummy_state_2'], np.array([0.5, 1.1, 0.5]), 0.0],
        [DummyPhysicalSystem(3), ['dummy_state_0', 'dummy_state_2'], np.array([-1.0, 1.1, 0.1]), 1.0],
        [DummyPhysicalSystem(3), ['dummy_state_0', 'dummy_state_2'], np.array([-1.1, 1.1, 0.0]), 1.0],
    ])
    def test_call(self, ps, observed_state_names, state, expected_violation):
        sc = SquaredConstraint(observed_state_names)
        sc.set_modules(ps)
        violation = sc(state)
        assert violation == expected_violation
