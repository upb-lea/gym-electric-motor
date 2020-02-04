import gym_electric_motor.envs
from ..testing_utils import *
from gym_electric_motor.reward_functions import *
from gym_electric_motor.utils import make_module
import numpy as np
import pytest


# region first version tests


# region general setup functions


def limits_testing(environment_state, lower_limits, upper_limits, observed_states_indices):
    """
    Test the limits to compare the output with the reward function

    :param environment_state: current state (ndarray)
    :param lower_limits: lower physical limits (ndarray)
    :param upper_limits: upper physical limits (ndarray)
    :param observed_states_indices: indices of the observed states for the limit violation
    :return: True if limits are violated, False otherwise
    """
    if any(environment_state[observed_states_indices] > upper_limits[observed_states_indices]) or\
            any(environment_state[observed_states_indices] < lower_limits[observed_states_indices]):
        return True
    else:
        return False
# endregion


@pytest.mark.parametrize("number_reward_weights", [1, 2, 3])
@pytest.mark.parametrize("normed", [False, True])
@pytest.mark.parametrize("reward_power", [-1, 0, 1, 2, 3])
@pytest.mark.parametrize("motor_type", ['DcSeries', 'DcShunt', 'DcPermEx', 'DcExtEx'])
@pytest.mark.parametrize("converter_type", ['Disc-1QC', 'Disc-2QC', 'Disc-4QC', 'Cont-1QC', 'Cont-2QC', 'Cont-4QC'])
def test_initialization_weighted_sum_of_errors(normed, reward_power, motor_type, converter_type, number_reward_weights):
    """
    Function to test the basic methods and functions ot the WeightedSumOfErrors class

    :param normed: specifies if the reward is normed (True/False)
    :param reward_power: specifies the used power order (int)
    :param motor_type: motor name (string)
    :param converter_type: converter name (string
    :param number_reward_weights: specifies how many states are considered for the reward
    :return:
    """
    # setup physical system and reference generator
    physical_system = setup_physical_system(motor_type, converter_type)
    reference_generator = setup_reference_generator('SinusReference', physical_system)
    reference_generator.reset()
    len_state = len(physical_system.state_names)
    state = np.ones(len_state)
    reference = physical_system.state_space.low
    # set parameters
    gamma = 0.99
    # observe the limits for all state variables
    observed_states = physical_system.state_names

    # set different types of reward weights
    reward_weights_dict = test_motor_parameter[motor_type]['reward_weights']
    # test different numbers of reward weights
    if number_reward_weights == 1:
        reward_weights_dict['torque'] = 0
    elif number_reward_weights == 3:
        reward_weights_dict['u_sup'] = 1

    reward_weights_list = list(reward_weights_dict.values())
    reward_weights_array = np.array(reward_weights_list)
    # initialize the reward function with different types of reward weights
    reward_fct_default = WeightedSumOfErrors()
    reward_fct_initial_dict = WeightedSumOfErrors(reward_weights_dict)
    reward_fct_initial_list = WeightedSumOfErrors(reward_weights_list)
    reward_fct_initial_array = WeightedSumOfErrors(reward_weights_array)
    reward_fct_make = make_module(RewardFunction, 'WSE', reward_weights=reward_weights_dict)
    # initialize a reward function where all parameters are set
    reward_fct_test = WeightedSumOfErrors(reward_weights_dict, normed, observed_states, gamma, reward_power)
    reward_fcts = [reward_fct_default,
                   reward_fct_initial_dict,
                   reward_fct_initial_list,
                   reward_fct_initial_array,
                   reward_fct_make,
                   reward_fct_test]
    for reward_fct in reward_fcts:
        reward_fct.set_modules(physical_system, reference_generator)

    # test if reward weights of default initialization are different to the parametrized initialization
    assert reward_fct_default._reward_weights is not reward_fct_initial_dict._reward_weights
    # test if all parametrized initializations result in the same reward weights
    assert all(reward_fct_initial_dict._reward_weights == reward_fct_initial_list._reward_weights) \
        and all(reward_fct_initial_dict._reward_weights == reward_fct_initial_array._reward_weights)

    # test the reward range, it should be negative and therefore, the upper limit is always zero
    assert reward_fct_test.reward_range[1] == 0
    # if the reward is normalized the minimum reward should be -1
    if normed:
        assert reward_fct_test.reward_range[0] == -1

    # test the reset, reward and close function as well as the limit violation in a basic test scenario
    for index, reward_fct in enumerate(reward_fcts):
        reward_fct.reset()
        reward_fct.close()
        rew, done = reward_fct.reward(state, reference)
        # test if reward is in the expected range
        assert reward_fct.reward_range[0] <= rew <= reward_fct.reward_range[1], "Reward out of reward range" +\
                                                                                str(reward_fct.reward_range) + " " +\
                                                                                str(reward_fct._normed) + " " +\
                                                                                str(reward_fct._reward_weights) + " " +\
                                                                                str(state) + " " + \
                                                                                str(reference) + " " + \
                                                                                str(index)
        assert done == 0
        # test if limit check does work
        exceeding_state = state * 1.2
        rew, done = reward_fct.reward(exceeding_state, reference)
        if index < 5:
            assert done == 0, "Limit violation recognized " + str([exceeding_state, physical_system.limits])
        else:
            assert done == 1, "Limit violation is not recognized " + str([exceeding_state, physical_system.limits])
            # test if the punishment is correct
            assert rew == reward_fct.reward_range[0] / (1 - gamma), "Wrong punishment"


@pytest.mark.parametrize("environment_state", np.array([[0.1, 0.2, 0.6, 0.5, 0.7, 0.8, 1],
                                                        [0.6, 0.3, 0.8, 0.5, 0.2, 0.5, 1],
                                                        [1.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1],
                                                        [0.2, 1.3, 0.4, 0.5, 0.7, 0.8, 1],
                                                        [0.2, 0.3, 1.4, 0.5, 0.7, 0.8, 1],
                                                        [0.2, 0.3, 0.4, 1.5, 0.7, 0.8, 1],
                                                        [0.2, 0.3, 0.4, 0.5, 1.7, 0.8, 1],
                                                        [0.2, 0.3, 0.4, 0.5, 0.7, 1.8, 1],
                                                        [0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.2],
                                                        [-1.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1],
                                                        [0.2, -1.3, 0.4, 0.5, 0.7, 0.8, 1],
                                                        [0.2, 0.3, -1.4, 0.5, 0.7, 0.8, 1],
                                                        [0.2, 0.3, 0.4, -1.5, 0.7, 0.8, 1],
                                                        [0.2, 0.3, 0.4, 0.5, -1.7, 0.8, 1],
                                                        [0.2, 0.3, 0.4, 0.5, 0.7, -1.8, 1],
                                                        [0.2, 0.3, 0.4, 0.5, 0.7, 0.8, -1.2]]))
@pytest.mark.parametrize("motor_type", ['DcSeries', 'DcShunt', 'DcPermEx', 'DcExtEx'])
@pytest.mark.parametrize("converter_type", ['Disc-1QC', 'Disc-2QC', 'Disc-4QC', 'Cont-1QC', 'Cont-2QC', 'Cont-4QC'])
@pytest.mark.parametrize("observed_states_basic", [['omega', 'torque', 'i', 'u', 'u_sup'],
                                                   ['omega', 'torque', 'i', 'u', 'u_sup', 'u_a', 'u_e', 'i_e', 'i_a'],
                                                   ['omega'],
                                                   ['torque'],
                                                   ['u_a', 'u_e', 'u'],
                                                   ['i', 'i_a', 'i_e']])
def test_limit_violation_check(motor_type, converter_type, environment_state, observed_states_basic):
    """
    test several combinations of states, if the limit violation check works well
    :param motor_type: motor name (string)
    :param converter_type: converter name (string)
    :param environment_state: state (ndarray)
    :param observed_states_basic: observed states for limit violation (list)
    :return:
    """
    # set parameter
    gamma = 0.99
    normed = True
    reward_power = 1
    reward_weights_dict = test_motor_parameter[motor_type]['reward_weights']
    # setup physical system and reference generator
    physical_system = setup_physical_system(motor_type, converter_type)
    reference_generator = setup_reference_generator('SinusReference', physical_system)
    reference_generator.reset()
    len_state = len(physical_system.state_names)
    reference = np.zeros(len_state)
    # set the observed states
    observed_states = []
    for variable in observed_states_basic:
        if variable in physical_system.state_names:
            observed_states.append(variable)
    observed_states_indices = []
    [observed_states_indices.append(physical_system.state_names.index(state_variable))
     for state_variable in observed_states]
    # initialize the reward function
    reward_fct = WeightedSumOfErrors(reward_weights_dict, normed, observed_states, gamma, reward_power)
    reward_fct.set_modules(physical_system, reference_generator)
    # get limits for limit violation testing
    lower_limits = physical_system.state_space.low # * physical_system.limits
    upper_limits = physical_system.state_space.high # * physical_system.limits
    # take the correct state array length
    state = np.array(environment_state[-len_state:]) * physical_system.limits
    # test limit violation function
    rew, done = reward_fct.reward(state, reference)
    done_test = limits_testing(state, lower_limits, upper_limits, observed_states_indices)
    assert done == done_test
    if not done:
        assert rew <= reward_fct.reward_range[1], "Reward out of range"


@pytest.mark.parametrize("number_reward_weights", [0, 1, 2, 3])
@pytest.mark.parametrize("normed", [False, True])
@pytest.mark.parametrize("reward_power", [-1, 0, 1, 2, 3])
@pytest.mark.parametrize("motor_type", ['DcSeries', 'DcShunt', 'DcPermEx', 'DcExtEx'])
@pytest.mark.parametrize("converter_type", ['Disc-1QC', 'Disc-2QC', 'Disc-4QC', 'Cont-1QC', 'Cont-2QC', 'Cont-4QC'])
def test_shifted_weighted_sum_of_errors(normed, reward_power, motor_type, converter_type, number_reward_weights):
    """
    Test the ShiftedWeightedSumOfErrors Class
    :param normed: specifies if the reward is normed (True/False)
    :param reward_power: specifies the used power order (int)
    :param motor_type: motor name (string)
    :param converter_type: converter name (string
    :param number_reward_weights: specifies how many states are considered for the reward
    :return:
    """
    # setup physical system and reference generator
    physical_system = setup_physical_system(motor_type, converter_type)
    reference_generator = setup_reference_generator('SinusReference', physical_system)
    reference_generator.reset()
    len_state = len(physical_system.state_names)
    state = np.ones(len_state) # * physical_system.limits
    reference = physical_system.state_space.low # * physical_system.limits
    # set parameter
    gamma = 0.99
    observed_states = physical_system.state_names
    reward_weights_dict = test_motor_parameter[motor_type]['reward_weights']
    # setup reward functions
    reward_fct_default = ShiftedWeightedSumOfErrors()
    reward_fct_1 = ShiftedWeightedSumOfErrors(reward_weights_dict, normed, observed_states, gamma, reward_power)
    reward_fct_2 = ShiftedWeightedSumOfErrors(reward_weights=reward_weights_dict, normed=normed,
                                              observed_states=observed_states, gamma=gamma, reward_power=reward_power)
    reward_fct_3 = make_module(RewardFunction, 'SWSE', observed_states=observed_states,
                               reward_weights=reward_weights_dict)
    reward_fct_list = [reward_fct_default, reward_fct_1, reward_fct_2, reward_fct_3]
    # test reset, reward, close function
    for index, reward_fct in enumerate(reward_fct_list):
        reward_fct.set_modules(physical_system, reference_generator)
        reward_fct.reset()
        assert reward_fct.reward_range[0] == 0, 'Wrong reward range lower limit'
        assert reward_fct.reward_range[1] > 0, 'Wrong reward range upper limit '
        rew, done = reward_fct.reward(state, reference)
        assert done == 0
        assert reward_fct.reward_range[0] <= rew <= reward_fct.reward_range[1], "Reward out of range"
        # test limit violation
        exceeding_state = 1.2 * state
        rew, done = reward_fct.reward(exceeding_state, reference)
        if index > 0:
            assert done == 1
            assert rew == 0
        else:
            assert done == 0
        reward_fct.close()

# endregion
