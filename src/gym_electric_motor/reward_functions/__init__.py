from ..core import RewardFunction
from ..utils import register_class
from .weighted_sum_of_errors import WeightedSumOfErrors

register_class(WeightedSumOfErrors, RewardFunction, "WSE")
