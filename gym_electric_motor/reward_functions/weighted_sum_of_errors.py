import numpy as np

from ..core import RewardFunction
from ..utils import set_state_array
import warnings


class WeightedSumOfErrors(RewardFunction):
    """A reward function that calculates the reward as the weighted sum of errors with a certain power.

    .. math::
        r_{wse} = - \sum_i w_i ((|s_i-s^*_i|) / l_i) ^{n_i} + b

    Notation:
        - :math:`r_\mathrm{wse}`: Weighted sum of error reward
        - :math:`w_{i}`: Reward weight of state :math:`i`
        - :math:`s_{i}`: State value of state :math:`i`
        - :math:`s^*_{i}`: Reference value of state :math:`i`
        - :math:`l_{i}`: State length of state :math:`i`
        - :math:`n_{i}`: Reward power of state :math:`i`
        - :math:`b`: Bias

    | :math:`l_i = 1` for states with positive values only.
    | :math:`l_i = 2` for states with positive and negative values.

    If environments constraints are violated to a certain degree, a special violation reward is returned as follows:

    .. math::
        r_{total} = (1.0 - d_{violation})  r_{wse} + d_{violation} r_{violation}

    Notation:
        - :math:`r_{total}`: Total reward
        - :math:`r_{wse}`: Weighted sum of error reward
        - :math:`r_{violation}`: Constraint violation reward
        - :math:`d_{violation}`: Limit violation degree :math:`d_{violation} \in [0,1]`


    The violation reward can be chosen freely by the user and shall punish the agents to comply with the constraints.
    Per default, the violation reward is selected so that it is always the worst expected reward the agent could get.

    .. math::
        r_{violation} = r_{wse,min} / (1 - \gamma)

    :math:`r_{wse,min}` is the minimal :math:`r_{wse}` (=reward_range[0]) and :math:`\gamma` the agents discount factor.
    """

    def __init__(self, reward_weights=None, normed_reward_weights=False, violation_reward=None,
                 gamma=0.9, reward_power=1, bias=0.0):
        """
        Args:
            reward_weights(dict/list/ndarray(float)): Dict mapping state names to reward_weights, 0 otherwise.
                Or an array with the reward_weights on the position of the state_names.

            normed_reward_weights(bool): If True, the reward weights will be normalized to sum up to 1.

            violation_reward(None/float): The punishment reward if constraints have been violated.

                - None(default): Per default, the violation reward is calculated as described above.
                - float: This value is taken as limit violation reward.

            gamma(float in [0.0, 1.0]): Discount factor for the reward punishment.
                Should equal agents' discount factor gamma. Used only, if violation_reward=None.

            reward_power(dict/list(float)/float): Reward power for each of the systems states.

            bias(float/'positive'): Additional bias that is added to the reward.

                - float: The value that is added
                - 'positive': The bias is selected so that the minimal reward is zero and all further are positive.
        """
        super().__init__()
        self._n = reward_power
        self._reward_weights = reward_weights
        self._state_length = None
        self._normed = normed_reward_weights
        self._gamma = gamma
        self._bias = bias
        self._violation_reward = violation_reward

    def set_modules(self, physical_system, reference_generator, constraint_monitor):
        super().set_modules(physical_system, reference_generator, constraint_monitor)
        ps = physical_system
        self._state_length = ps.state_space.high - ps.state_space.low
        self._n = set_state_array(self._n, ps.state_names)
        referenced_states = reference_generator.referenced_states
        if self._reward_weights is None:
            # If there are any referenced states reward weights are equally distributed over them
            if np.any(referenced_states):
                reward_weights = dict.fromkeys(
                    np.array(physical_system.state_names)[referenced_states],
                    1 / len(np.array(physical_system.state_names)[referenced_states])
                )
            # If no referenced states and no reward weights passed, uniform reward over all states
            else:
                reward_weights = dict.fromkeys(
                    np.array(physical_system.state_names),
                    1 / len(np.array(physical_system.state_names))
                )
        else:
            reward_weights = self._reward_weights
        self._reward_weights = set_state_array(reward_weights, ps.state_names)
        if sum(self._reward_weights) == 0:
            warnings.warn("All reward weights sum up to zero", Warning, stacklevel=2)
        rw_sum = sum(self._reward_weights)
        if self._normed:
            if self._bias == 'positive':
                self._bias = 1
            self._reward_weights = self._reward_weights / rw_sum
            self.reward_range = (-1 + self._bias, self._bias)
        else:
            if self._bias == 'positive':
                self._bias = rw_sum
            self.reward_range = (-rw_sum + self._bias, self._bias)
        if self._violation_reward is None:
            self._violation_reward = min(self.reward_range[0] / (1.0 - self._gamma), 0)

    def reward(self, state, reference, k=None, action=None, violation_degree=0.0):
        return (1.0 - violation_degree) * self._wse_reward(state, reference) \
            + violation_degree * self._violation_reward

    def _wse_reward(self, state, reference):
        return -np.sum(self._reward_weights * (abs(state - reference) / self._state_length) ** self._n) + self._bias
