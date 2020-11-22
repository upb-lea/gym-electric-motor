import pytest
import numpy as np

from gym_electric_motor.reward_functions.weighted_sum_of_errors import WeightedSumOfErrors
from .test_reward_functions import TestRewardFunction
from ..testing_utils import DummyPhysicalSystem, DummyReferenceGenerator, DummyConstraintMonitor


class TestWeightedSumOfErrors(TestRewardFunction):

    class_to_test = WeightedSumOfErrors

    @pytest.fixture
    def reward_function(self):
        rf = WeightedSumOfErrors()
        rf.set_modules(DummyPhysicalSystem(), DummyReferenceGenerator(), DummyConstraintMonitor())
        return rf

    @pytest.mark.parametrize(['ps', 'reward_power', 'n'], [
        [DummyPhysicalSystem(state_length=2), [1, 2], np.array([1, 2])],
        [DummyPhysicalSystem(state_length=3), 2, np.array([2, 2, 2])],
        [DummyPhysicalSystem(state_length=1), [1], np.array([1])],
        [DummyPhysicalSystem(state_length=2), dict(dummy_state_0=2), np.array([2, 0])]
    ])
    @pytest.mark.parametrize('rg', [DummyReferenceGenerator()])
    @pytest.mark.parametrize('cm', [DummyConstraintMonitor()])
    def test_reward_powers(self, ps, rg, cm, reward_power, n):
        rf = self.class_to_test(reward_power=reward_power)
        rf.set_modules(
            ps, rg, cm)
        assert all(rf._n == n)

    @pytest.mark.parametrize(['ps', 'reward_weights', 'expected_rw'], [
        [DummyPhysicalSystem(state_length=2), [1, 2], np.array([1, 2])],
        [DummyPhysicalSystem(state_length=3), 2, np.array([2, 2, 2])],
        [DummyPhysicalSystem(state_length=1), [1], np.array([1])],
        [DummyPhysicalSystem(state_length=2), dict(dummy_state_0=2), np.array([2, 0])],
        [DummyPhysicalSystem(state_length=2), None, np.array([1.0, 0.0])],
    ])
    @pytest.mark.parametrize('rg', [DummyReferenceGenerator()])
    @pytest.mark.parametrize('cm', [DummyConstraintMonitor()])
    def test_reward_weights(self, ps, rg, cm, reward_weights, expected_rw):
        rg.set_modules(ps)
        rf = self.class_to_test(reward_weights=reward_weights)
        rf.set_modules(ps, rg, cm)
        assert all(rf._reward_weights == expected_rw)

    @pytest.mark.parametrize(['ps', 'violation_reward', 'gamma', 'expected_vr'], [
        [DummyPhysicalSystem(state_length=2), 0.0, 0.9, 0.0],
        [DummyPhysicalSystem(state_length=3), -100000, 0.99, -100000],
        [DummyPhysicalSystem(state_length=1), None, 0.99, -100],
        [DummyPhysicalSystem(state_length=2), None, 0.5, -2],
        [DummyPhysicalSystem(state_length=2), 5000, 0.5, 5000],
    ])
    @pytest.mark.parametrize('rg', [DummyReferenceGenerator()])
    @pytest.mark.parametrize('cm', [DummyConstraintMonitor()])
    def test_violation_reward(self, ps, rg, cm, violation_reward, gamma, expected_vr):
        rg.set_modules(ps)
        rf = self.class_to_test(violation_reward=violation_reward, gamma=gamma)
        rf.set_modules(ps, rg, cm)
        assert np.isclose(rf._violation_reward, expected_vr, rtol=1e-4)

    @pytest.mark.parametrize(['ps', 'reward_weights', 'normed_rw', 'expected_rw'], [
        [DummyPhysicalSystem(state_length=2), [1, 2], False, np.array([1, 2])],
        [DummyPhysicalSystem(state_length=2), [1, 2], True, np.array([1/3, 2/3])],
        [DummyPhysicalSystem(state_length=3), 2, True, np.array([1/3, 1/3, 1/3])],
        [DummyPhysicalSystem(state_length=1), [1], False, np.array([1])],
        [DummyPhysicalSystem(state_length=2), dict(dummy_state_0=2), True, np.array([1, 0])],
    ])
    @pytest.mark.parametrize('rg', [DummyReferenceGenerator()])
    @pytest.mark.parametrize('cm', [DummyConstraintMonitor()])
    def test_normed_reward_weights(self, ps, rg, cm, reward_weights, normed_rw, expected_rw):
        rg.set_modules(ps)
        rf = self.class_to_test(reward_weights=reward_weights, normed_reward_weights=normed_rw)
        rf.set_modules(ps, rg, cm)
        assert all(rf._reward_weights == expected_rw)

    @pytest.mark.parametrize(['ps', 'rw', 'bias', 'expected_rr'], [
        [DummyPhysicalSystem(state_length=2), [1, 2], 0.0, (-3.0, 0.0)],
        [DummyPhysicalSystem(state_length=2), [1, 0], 1.0, (0.0, 1.0)],
        [DummyPhysicalSystem(state_length=3), [5, 4, 3], 9, (-3.0, 9.0)],
    ])
    @pytest.mark.parametrize('rg', [DummyReferenceGenerator()])
    @pytest.mark.parametrize('cm', [DummyConstraintMonitor()])
    def test_reward_range(self, ps, rg, cm, rw, bias, expected_rr):
        rg.set_modules(ps)
        rf = self.class_to_test(reward_weights=rw, bias=bias)
        rf.set_modules(ps, rg, cm)
        assert rf.reward_range == expected_rr

    @pytest.mark.parametrize(['ps', 'rw', 'bias', 'expected_bias'], [
        [DummyPhysicalSystem(state_length=2), [2.0, 0.5], 0.9, 0.9],
        [DummyPhysicalSystem(state_length=3), 100000, 'positive', 300000],
        [DummyPhysicalSystem(state_length=1), [5], 0, 0],
    ])
    @pytest.mark.parametrize('rg', [DummyReferenceGenerator()])
    @pytest.mark.parametrize('cm', [DummyConstraintMonitor()])
    def test_bias(self, ps, rg, cm, rw, bias, expected_bias):
        rg.set_modules(ps)
        rf = self.class_to_test(reward_weights=rw, bias=bias)
        rf.set_modules(ps, rg, cm)
        assert rf._bias == expected_bias

    @pytest.mark.parametrize(
        ['reward_weights', 'violation_reward', 'bias', 'violation_degree', 'state', 'reference', 'expected_rw'], [
            [[1, 2, 0], -10000, 0.0, 0.0, np.array([0, 1, 0]), np.array([0, 1, 5]), 0.0],
            [[1, 2, 0], -10000, 10.0, 0.0, np.array([0, 1, 0]), np.array([0, 1, 5]), 10.0],
            [[1, 2, 0], -10000, 10.0, 1.0, np.array([0, 1, 0]), np.array([0, 1, 5]), -10000.0],
            [[1, 2, 0], -10000, 10.0, 0.5, np.array([0, 1, 0]), np.array([0, 1, 5]), -4995.0],
    ])
    @pytest.mark.parametrize('ps', [DummyPhysicalSystem(state_length=3)])
    @pytest.mark.parametrize('rg', [DummyReferenceGenerator()])
    @pytest.mark.parametrize('cm', [DummyConstraintMonitor()])
    def test_reward(
            self, ps, rg, cm, reward_weights, bias, violation_reward, violation_degree, state, reference, expected_rw
    ):
        rg.set_modules(ps)
        rf = self.class_to_test(reward_weights=reward_weights, bias=bias, violation_reward=violation_reward)
        rf.set_modules(ps, rg, cm)
        assert rf.reward(state, reference, violation_degree=violation_degree) == expected_rw
