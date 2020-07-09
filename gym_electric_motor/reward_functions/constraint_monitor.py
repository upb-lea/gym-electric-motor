import numpy as np
import gym
from ..core import RewardFunction
from gym.spaces import Box

# todo docstrings, comments for discount factor


class ConstraintMonitor:
    """


    """
    @property
    def constraints(self):
        """
        Userdefined constraints for system state-space

        Returns:
            constraints(gym.Box) for system state-space

        """
        return self._constraints

    @property
    def discount_factor(self):
        """
        Userdefined value to get discounted limit values

        Returns:
             discount_factor(float): factor to discount limits
        """
        return self._discount_factor

    @property
    def penalty_coefficient(self):
        """
        Userdefined value to scale penalty function

        Returns:
             penalty_coefficient(float): factor to scale penalty function
        """
        return self._penalty_coefficient

    @property
    def observed_states(self):
        """
        States, which should be monitored

        Returns:
            observed_states(list): monitored states
        """
        return self._observed_states

    def __init__(self,
                 constraints=None,
                 discount_factor=1.0,
                 penalty_coefficient=0,
                 normalised=False,):
        """
        Args:
            constraints():
            discount_factor(float):
            penalty_coefficient(float):
            normalised(bool):

        """
        # assert isinstance(constraints, gym.spaces.Box),\
        #     'constraints have to be a gym.Box'
        # assert isinstance(penalty_coefficient, (float, int))

        self._constraints = constraints

        self._discount_factor = discount_factor
        self._penalty_coefficient = penalty_coefficient
        self._normalised = normalised
        self._observed_states = None

    def set_attributes(self, observed_states, limits):
        """

        """
        # setting system limits as constraint, when no constraints given
        if self._constraints is None:
            self._constraints = limits * self._discount_factor
            #print(self._constraints)

        # normalize constraints to limit values
        if not self._normalised:
            self._constraints = self._normalize_constraints(self._constraints,
                                                            limits)
        self._observed_states = observed_states

        #print(abs(limits))
        #print(self._constraints)

    def check_constraint_violation(self, state):
        """

        """
        if isinstance(self._constraints, gym.spaces.Box):
            print('hier')
            lower = self._constraints.low
            upper = self._constraints.high
            print(lower)
            print(state)
            print(upper)
            return int((abs(state) < lower).any() or (abs(state) > upper).any())

        elif isinstance(self._constraints, (list, np.ndarray)):
            print('oder hier')
            print(abs(state[self._observed_states]))
            print(self._constraints[self._observed_states])
            check = (abs(state[self._observed_states]) >
                     self._constraints[self._observed_states]).any()
            print(check)
            return int(check)
        else:
            raise NotImplementedError

    def _normalize_constraints(self, constraint, denominator):
        if isinstance(constraint, gym.spaces.Box):
            upper = constraint.high / abs(denominator)
            lower = constraint.low / abs(denominator)
            return Box(lower, upper)
        elif isinstance(constraint, (list, np.ndarray)):
            return constraint / abs(denominator)

    # todo in following implementations: add calculation of soft violation
    #  return from [0, 1]
    def _penalty_function(self, ):
        # eventually useful for next implementations
        raise NotImplementedError
