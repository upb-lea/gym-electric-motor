from .weighted_sum_of_errors import WeightedSumOfErrors
from ..utils import register_class
from .. import RewardFunction

register_class(WeightedSumOfErrors, RewardFunction, 'WSE')

