import numpy as np
import gym
from ..core import RewardFunction
from gym.spaces import Box

# todo docstrings, comments for discount factor


class ConstraintMonitor:
    """
    The ConstraintMonitor Class monitors the system-states and assesses whether
    they comply the given limits or violate them.
    It returns the necessary information for the RewardFunction, to calculate
    the corresponding reward-value.
    Limits, here called constraints, can be given by the physical limits from
    the environment, or user-defined constraints.
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
            constraints(gym.spaces.Box, list): user-defined constraints,
                given as a Box() or a list with maximal values for each state
            discount_factor(float): scalar to discount physical-limits by the
                given factor
            penalty_coefficient(float): factor for the penalty function
            normalised(bool): describes whether the constraints, given by the
                user, are normalised or not
        Return:
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
        Setting the necessary attributes for different class-methods

        Args:
            observed_states(list): list with boolean entries, indicating which
                states have to be monitored
            limits(np.ndarray): limits given by the physical.system
        Returns:
        """
        # setting system limits as constraint, when no constraints given
        if self._constraints is None:
            self._constraints = limits * self._discount_factor

        # normalize constraints to limit values
        if not self._normalised:
            self._constraints = self._normalize_constraints(self._constraints,
                                                            limits)
        self._observed_states = observed_states

    def check_constraint_violation(self, state):
        """
        Checks if a given system-state violates the constraints

        Args:
            state(np.ndarray): system-state of the environment
        Returns:
            integer value from [0, 1], where 0 is no violation at all and 1 is
            a hard constraint violation
        """
        if isinstance(self._constraints, gym.spaces.Box):
            lower = self._constraints.low
            upper = self._constraints.high
            return int((abs(state) < lower).any() or (abs(state) > upper).any())

        elif isinstance(self._constraints, (list, np.ndarray)):
            check = (abs(state[self._observed_states]) >
                     self._constraints[self._observed_states]).any()
            return int(check)
        else:
            # todo return for soft violation
            raise NotImplementedError

    def _normalize_constraints(self, constraint, denominator):
        """
        Args:
            constraint(gym.spaces.Box, np.ndarray): constraints to be normalised
            denominator(np.ndarray): constraint is normalised in relation to
                this argument
        Returns:
             normalised constraints
        """
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
