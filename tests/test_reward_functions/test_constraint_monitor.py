import pytest
import numpy as np
import gym
from gym.spaces import Box
from gym_electric_motor.reward_functions.constraint_monitor \
    import ConstraintMonitor
import gym_electric_motor.reward_functions.constraint_monitor as cm
import tests.testing_utils as tu


class TestConstraintMonitor:
    """
    This test class consists of unit tests to verify modules in the constraint_monitor.py
    Following modules are included:
    ConstraintMonitor
        __init__
        set_attributes
        check_constraint_violation
        _normalize_constraints
    """

    # Parameters used for testing
    test_box_constraints = Box(high=16, low=-16, shape=(3,), dtype=float)
    test_limit_constraints = np.asarray([16., 16., 16.])
    test_observed_states = np.array([True, True, False])
    test_limits = np.asarray([16., 16., 16.])
    # Test counter to count function calls

    @pytest.mark.parametrize('constraint, exp_constraint',
                             [(test_box_constraints, test_box_constraints),
                              (test_limit_constraints, test_limit_constraints)])
    @pytest.mark.parametrize('disc_fact, exp_dis_fact', [(0.5, 0.5), (1, 1)])
    @pytest.mark.parametrize('pen_coeff, exp_pen_coeff', [(0, 0), (0.8, 0.8)])
    def test_initialization(self,
                            constraint, exp_constraint,
                            disc_fact, exp_dis_fact,
                            pen_coeff, exp_pen_coeff,):
        """
        Tests the initialization of ConstraintMonitor object with
        defined parameters

        :param constraint: defined system constraints
        :param exp_constraint: expected defined system constraints
        :param disc_fact: defined discount_factor
        :param exp_dis_fact: expected defined discount_factor
        :param pen_coeff: defined penalty_coefficient
        :param exp_pen_coeff: expected defined penalty_coefficient
        :return:
        """
        test_con_mon = ConstraintMonitor(constraints=constraint,
                                         discount_factor=disc_fact,
                                         penalty_coefficient=pen_coeff,
                                         normalised=True)
        if isinstance(constraint, gym.spaces.Box):
            assert test_con_mon.constraints == exp_constraint
        else:
            assert all(test_con_mon.constraints == exp_constraint)
        assert test_con_mon.discount_factor == exp_dis_fact
        assert test_con_mon.penalty_coefficient == exp_pen_coeff

    @pytest.mark.parametrize('test_disc_fact', [0.5, 1])
    def test_set_attributes(self, test_disc_fact):
        """
        test the set_attributes function

        :param :
        :param :
        :return:
        """
        normalised_constraints = Box(-1, 1, (3,))
        normalised_limits = np.asarray([1., 1., 1.])

        test_con_mon = ConstraintMonitor(discount_factor=test_disc_fact,
                                         normalised=True)
        test_con_mon.set_attributes(self.test_observed_states,
                                    normalised_limits)

        assert all(test_con_mon.constraints == normalised_limits * test_disc_fact)
        assert all(test_con_mon.observed_states == self.test_observed_states)

    @pytest.mark.parametrize('constraints, exp_result',
                             [(test_box_constraints, Box(-1, 1, (3,))),
                              (test_limit_constraints, np.ones(3, dtype=float))])
    def test_normalize_constraints(self, constraints, exp_result):
        """

         :param :
         :param :
         :param :
         :return:
         """
        test_con_mon = ConstraintMonitor()
        normalized_result = test_con_mon._normalize_constraints(
            constraint=constraints,
            denominator=self.test_limits
        )
        if isinstance(constraints, gym.spaces.Box):
            assert normalized_result == exp_result
        else:
            assert all(normalized_result == exp_result)

    @pytest.mark.parametrize('constraints', (test_box_constraints,
                                             test_limit_constraints))
    @pytest.mark.parametrize('state, exp_return',
                             [(np.array([0., 0., 0.]), 0),
                              (np.array([0.5, 0.4, 0.16]), 0),
                              (np.array([-0.5, -0.4, -0.16]), 0),
                              (np.array([1.5, 0.4, 0.16]), 1),
                              (np.array([0.5, -1.4, 0.16]), 1)])
    def test_check_constraint_violation(self,
                                        state,
                                        exp_return,
                                        constraints):
        """
         :param state: system state
         :param exp_return: expected return with given system state
         :param constraints: constraints for monitor
         :return:
         """
        test_con_mon = ConstraintMonitor(constraints=constraints)
        test_con_mon.set_attributes(self.test_observed_states,
                                    self.test_limits)

        assert test_con_mon.check_constraint_violation(state) == exp_return

