import numpy as np

from ..core import RewardFunction


class ConstraintMonitor(RewardFunction):
    """


    """

    def __init__(self, penalty_coeff=1):
        """
        Args:

        """

        raise NotImplementedError

    def _check_limit_violation(self, state):
        # Docstring from superclass

        return (abs(state[self._observed_states]) > self._limits[self._observed_states]).any()

    def _penalty_function(self, ):

        raise NotImplementedError
