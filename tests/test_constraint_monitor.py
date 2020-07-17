import pytest
import numpy as np
import gym
from gym.spaces import Box
from gym_electric_motor.constraint_monitor import ConstraintMonitor


class MockTestViolation:
    """ Test class for external Monitor usage"""
    def __init__(self, test_violation, **kwargs):
        self.test_violation = test_violation

    def __call__(self, **_):
        return self.test_violation


class TestConstraintMonitor:
    """
    This test class consists of unit tests to verify modules in the constraint_monitor.py
    Following modules are included:
    ConstraintMonitor
        __init__
        set_modules
        check_constraint_violation
    """

    # Parameters used for testing
    test_box_constraints = Box(high=16, low=-16, shape=(3,), dtype=float)
    test_limit_constraints = np.asarray([16., 16., 16.])
    test_observed_states = np.array([True, True, False])
    test_limits = np.asarray([16., 16., 16.])

    def mock_external_monitor(self, test_violation, **kwargs):
        return test_violation

    def test_initialization(self,):
        """
        Tests the initialization of ConstraintMonitor object with
        defined parameters
        """
        test_con_mon = ConstraintMonitor(external_monitor='test',
                                         normalised=True)

        assert test_con_mon._external_monitor == 'test'
        assert test_con_mon._normalised == True

    @pytest.mark.parametrize('test_violation, exp_return',
                             [(True, 1.0),
                              (False, 0.0),
                              (0.5, 0.5)])
    def test_check_constraint_violation(self,
                                        test_violation,
                                        exp_return):
        """
         :param test_violation: value returned by external monitor
         :param exp_return: expected return with given system state
         :return:
         """
        test_con_mon_fct = ConstraintMonitor(external_monitor=
                                self.mock_external_monitor, test_violation=test_violation)
        test_con_mon_class = ConstraintMonitor(external_monitor=
                                MockTestViolation(test_violation))
        assert test_con_mon_fct.check_constraint_violation(state=None) == exp_return
        assert test_con_mon_class.check_constraint_violation(state=None) == exp_return
