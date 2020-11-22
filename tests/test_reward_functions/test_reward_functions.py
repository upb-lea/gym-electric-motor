import numpy as np
import pytest

from gym_electric_motor import RewardFunction
from ..testing_utils import DummyReferenceGenerator, DummyPhysicalSystem, DummyConstraintMonitor


class MockRewardFunction(RewardFunction):
    """Mock RewardFunction that only implements the basics to test the basic reward function interface."""

    def reward(self, state, reference, k=None, action=None, violation_degree=0.0):
        return 1.0


class TestRewardFunction:

    @pytest.fixture
    def reward_function(self):
        return MockRewardFunction()

    @pytest.mark.parametrize(['state', 'reference'], [
        [np.array([1.0, 2.0, 3.0, 4.0]),  np.array([0.0, 2.0, 4.0, 0.0])],
        [np.array([1.0, -2.0]), np.array([-1.0, 20.0])],
        [np.array([-24.0]), np.array([0.0])]
    ])
    @pytest.mark.parametrize('k', [0, 1, 2, 3, 4, 90999])
    @pytest.mark.parametrize('violation_degree', [0.0, 0.25, 0.5, 1.0])
    @pytest.mark.parametrize('action', [
        np.array([0.0, 1.0]),
        np.array([-1.0]),
        8,
        0
    ])
    def test_call(self, reward_function, state, reference, k, action, violation_degree):
        """Test, if the call function is equal to the reward function"""
        rew_call = reward_function(state, reference, k, action, violation_degree)
        rew_fct = reward_function.reward(state, reference, k, action, violation_degree)
        assert rew_call == rew_fct

    @pytest.mark.parametrize(['initial_state', 'initial_reference'], [
        [None, None],
        [np.array([1.0, 2.0]), np.array([0.0, -2.0])],
        [np.array([1.0, 2.0]), None],
        [None, np.array([1.0, 2.0])],
    ])
    def test_reset_interface(self, reward_function, initial_state, initial_reference):
        """Test, if the reset function accepts the correct parameters."""
        kwargs = dict()
        if initial_state is not None:
            kwargs['initial_state'] = initial_state
        if initial_reference is not None:
            kwargs['initial_reference'] = initial_reference
        # It has to run through without any error
        reward_function.reset(**kwargs)

    @pytest.mark.parametrize(['physical_system', 'reference_generator', 'constraint_monitor'], [[
        DummyPhysicalSystem(), DummyReferenceGenerator(), DummyConstraintMonitor()]
    ])
    def test_set_modules_interface(self, reward_function, physical_system, reference_generator, constraint_monitor):
        reward_function.set_modules(physical_system, reference_generator, constraint_monitor)

    def test_close_interface(self, reward_function):
        reward_function.close()

