import gym_electric_motor.physical_systems.converters as cv
from tests.testing_utils import PowerElectronicConverterWrapper, DummyConverter
import gym_electric_motor as gem
from functools import reduce
import pytest
import numpy as np
import tests.conf as cf
from gym_electric_motor.utils import make_module
from random import seed, uniform, randint
from gym.spaces import Discrete, Box

# region first version tests


# region definitions

# define basic test parameter
g_taus = [1E-3, 1E-4, 1E-5]
g_dead_times = [False, True]
g_interlocking_times = np.array([0.0, 1 / 20, 1 / 3])

# define test parameter for different cases
# continuous case
g_i_ins_cont = [-1, -0.5, 0.0, 0.25, 0.5, 0.75, 1]
g_times_cont = np.arange(15)

# disc 1QC
g_times_1qc = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
g_actions_1qc = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
g_i_ins_1qc = np.array([-0.5, 0.25, 0.75, 1, -0.5, 0, 0.25, 0.35, -0.15, 0.65, 0.85])
g_test_voltages_1qc = np.array([1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0])
g_test_voltages_1qc_dead_time = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1])

# disc 2QC
g_i_ins_2qc = [0, 0.5, -0.5, 0.5, 0.5, 0, -0.5, 0.5, 0.5, 0, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
g_actions_2qc = [0, 0, 0, 0, 1, 1, 1, 0, 2, 2, 2, 1, 2, 2, 1, 2]
g_times_2qc = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
g_test_voltages_2qc = np.array([0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0])
g_test_voltages_2qc_dead_time = np.array([0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1])
g_test_voltages_2qc_interlocking = np.array([0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0])
g_test_voltages_2qc_dead_time_interlocking = np.array([0, 0, 1, 0, 0,
                                                       1, 1, 1, 0, 0,
                                                       0, 0, 1, 1, 0,
                                                       0, 0, 0, 1, 0])
# disc 4QC
g_times_4qc = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
g_i_ins_4qc = [0, 0.5, -0.5, 0.5, 0.5, -0.5, 0, 0.5, 0.5, 0.5, 0.5, -0.5, 0, 0.5, 0.5, 0.5, 0.5, -0.5, 0, 0.5, 0.5, 0.5,
               0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5]
g_actions_4qc = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 0, 0, 3, 3, 3, 3, 0, 1, 2, 3, 1, 3, 3, 2, 1])
g_test_voltages_4qc = np.array(
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 1, 0, 0, -1, 1])
g_test_voltages_4qc_dead_time = np.array(
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 1, 0, 0, -1])
g_test_voltages_4qc_interlocking = np.array(
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, 0, 0, 0, 0,
     -1, 0, 0, 1, -1, -1, -1, 0, 0, 1, 1, 0, 0, 0, -1, 1, 1])
g_test_voltages_4qc_dead_time_interlocking = np.array(
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 1, 0,
     0, 0, 0, -1, 0, 0, 1, -1, -1, -1, 0, 1, 1, 1, 0, 0, 0, -1])
# combine all test voltages in one vector for each converter
g_1qc_test_voltages = [g_test_voltages_1qc, g_test_voltages_1qc_dead_time, g_test_voltages_1qc,
                       g_test_voltages_1qc_dead_time]
g_2qc_test_voltages = [g_test_voltages_2qc, g_test_voltages_2qc_dead_time, g_test_voltages_2qc_interlocking,
                       g_test_voltages_2qc_dead_time_interlocking]
g_4qc_test_voltages = [g_test_voltages_4qc, g_test_voltages_4qc_dead_time, g_test_voltages_4qc_interlocking,
                       g_test_voltages_4qc_dead_time_interlocking]
g_disc_test_voltages = {'Finite-1QC': g_1qc_test_voltages,
                        'Finite-2QC': g_2qc_test_voltages,
                        'Finite-4QC': g_4qc_test_voltages}
g_disc_test_i_ins = {'Finite-1QC': g_i_ins_1qc,
                     'Finite-2QC': g_i_ins_2qc,
                     'Finite-4QC': g_i_ins_4qc}
g_disc_test_actions = {'Finite-1QC': g_actions_1qc,
                       'Finite-2QC': g_actions_2qc,
                       'Finite-4QC': g_actions_4qc}


# endregion


# region discrete converter


def discrete_converter_functions_testing(converter, action_space_n, times,
                                         actions,
                                         i_ins,
                                         test_voltage_ideal,
                                         test_voltage_dead_time,
                                         test_voltage_interlocking_time,
                                         test_voltage_dead_interlocking,
                                         interlocking_time=0.0, dead_time=False):
    """
    test of convert function of discrete converter
    :param converter:
    :param action_space_n: number of possible actions
    :param times: time instance for set action
    :param actions: pre defined actions for testing
    :param i_ins: used input current for testing
    :param test_voltage_ideal: expected voltages in ideal behaviour
    :param test_voltage_dead_time: expected voltages if dead time is considered
    :param test_voltage_interlocking_time: expected output voltages if interlocking time is considered
    :param test_voltage_dead_interlocking: expected output voltages if interlocking and dead time are considered
    :param interlocking_time: used interlocking time
    :param dead_time: used dead time
    :return:
    """
    action_space = converter.action_space
    assert action_space.n == action_space_n  # test if the action space has the right size
    step_counter = 0
    for time, action, i_in in zip(times, actions, i_ins):  # apply each action at different times
        time_steps = converter.set_action(action, time)  # test set action, returns switching times
        for index, time_step in enumerate(time_steps):
            converter_voltage = converter.convert([i_in], time_step)
            converter.i_sup([i_in])
            for u in converter_voltage:
                assert converter.voltages.low[0] <= u <= converter.voltages.high[0], "Voltage limits violated"
            # test for different cases of (non) ideal behaviour
            if dead_time and interlocking_time > 0:
                test_voltage = test_voltage_dead_interlocking[step_counter]
                # g_u_out_dead_time_interlocking[step_counter]
            elif dead_time:
                test_voltage = test_voltage_dead_time[step_counter]
                # g_u_out_dead_time[step_counter]
            elif interlocking_time > 0:
                test_voltage = test_voltage_interlocking_time[step_counter]
                # g_u_out_interlocking[step_counter]
            else:
                test_voltage = test_voltage_ideal[step_counter]
                # g_u_out[step_counter]
            assert test_voltage == converter_voltage, "Wrong voltage " + str(step_counter)
            step_counter += 1


@pytest.mark.parametrize("tau", g_taus)
@pytest.mark.parametrize("interlocking_time", g_interlocking_times)
@pytest.mark.parametrize("dead_time", g_dead_times)
@pytest.mark.parametrize("converter_type, action_space_n, actions, i_ins, test_voltages",
                         [('Finite-1QC', 2, g_actions_1qc, g_i_ins_1qc, g_1qc_test_voltages),
                          ('Finite-2QC', 3, g_actions_2qc, g_i_ins_2qc, g_2qc_test_voltages),
                          ('Finite-4QC', 4, g_actions_4qc, g_i_ins_4qc, g_4qc_test_voltages)])
def test_discrete_single_power_electronic_converter(converter_type, action_space_n, actions, i_ins, test_voltages,
                                                    dead_time,
                                                    interlocking_time, tau):
    """
    test the initialization of all single converters for different tau, interlocking times, dead times
    furthermore, test the other functions: reset, convert with different parameters
    :param converter_type: converter name (string)
    :param action_space_n: number of elements in the action space
    :param actions: pre defined actions for testing
    :param i_ins: used input currents for testing
    :param test_voltages: expected voltages for ideal and non ideal behaviour
    :param dead_time: specifies if a dead time is considered or not
    :param interlocking_time: the used interlocking time
    :param tau: sampling time
    :return:
    """
    # test with default initialization
    converter = make_module(cv.PowerElectronicConverter, converter_type)
    g_times = g_times_4qc
    times = g_times * converter._tau
    discrete_converter_functions_testing(converter,
                                         action_space_n,
                                         times,
                                         actions,
                                         i_ins,
                                         test_voltages[0],
                                         test_voltages[1],
                                         test_voltages[2],
                                         test_voltages[3])

    # define various constants for test
    times = g_times * tau
    interlocking_time *= tau
    # setup converter
    # initialize converter with given parameter
    converter = make_module(cv.PowerElectronicConverter, converter_type, tau=tau,
                            interlocking_time=interlocking_time, dead_time=dead_time)
    assert converter.reset() == [0.0]  # test if reset returns 0.0
    # test the conversion function of the converter
    discrete_converter_functions_testing(converter, action_space_n, times,
                                         actions,
                                         i_ins,
                                         test_voltages[0],
                                         test_voltages[1],
                                         test_voltages[2],
                                         test_voltages[3],
                                         interlocking_time=interlocking_time, dead_time=dead_time)


@pytest.mark.parametrize("convert, convert_class", [
    ('Finite-1QC', cv.FiniteOneQuadrantConverter),
    ('Finite-2QC', cv.FiniteTwoQuadrantConverter),
    ('Finite-4QC', cv.FiniteFourQuadrantConverter)
])
@pytest.mark.parametrize("tau", g_taus)
@pytest.mark.parametrize("interlocking_time", g_interlocking_times)
@pytest.mark.parametrize("dead_time", g_dead_times)
def test_discrete_single_initializations(convert, convert_class, tau, interlocking_time, dead_time):
    """
    test of both ways of initialization lead to the same result
    :param convert: string name of the converter
    :param convert_class: class name of the converter
    :return:
    """
    # test default initialization
    converter_1 = make_module(cv.PowerElectronicConverter, convert)
    converter_2 = convert_class()
    assert converter_1._tau == converter_2._tau
    # test with different parameters
    interlocking_time *= tau
    # initialize converters
    converter_1 = make_module(cv.PowerElectronicConverter, convert, tau=tau,
                              interlocking_time=interlocking_time, dead_time=dead_time)
    converter_2 = convert_class(
        tau=tau, interlocking_time=interlocking_time, dead_time=dead_time)
    parameter = str(tau) + " " + str(dead_time) + " " + str(interlocking_time)
    # test if they are equal
    assert converter_1.reset() == converter_2.reset()
    assert converter_1.action_space == converter_2.action_space
    assert converter_1._tau == converter_2._tau == tau, "Error (tau): " + parameter
    assert converter_1._dead_time == converter_2._dead_time == dead_time, "Dead time Error"
    assert converter_1._interlocking_time == converter_2._interlocking_time == interlocking_time, \
        "Error (interlocking): " + parameter


@pytest.mark.parametrize("tau", g_taus)
@pytest.mark.parametrize("interlocking_time", g_interlocking_times)
@pytest.mark.parametrize("dead_time", g_dead_times)
def test_discrete_multi_converter_initializations(tau, interlocking_time, dead_time):
    """
    tests different initializations of the converters
    :return:
    """
    # define all converter
    all_single_disc_converter = ['Finite-1QC', 'Finite-2QC', 'Finite-4QC', 'Finite-B6C']
    interlocking_time *= tau
    # chose every combination of single converters
    for conv_1 in all_single_disc_converter:
        for conv_2 in all_single_disc_converter:
            converter = make_module(
                cv.PowerElectronicConverter, 'Finite-Multi', tau=tau,
                interlocking_time=interlocking_time, dead_time=dead_time,
                subconverters=[conv_1, conv_2]
            )
            # test if both converter have the same parameter
            assert converter._subconverters[0]._tau == converter._subconverters[1]._tau == tau
            assert converter._subconverters[0]._interlocking_time == converter._subconverters[1]._interlocking_time \
                   == interlocking_time
            assert converter._subconverters[0]._dead_time == converter._subconverters[1]._dead_time == dead_time


@pytest.mark.parametrize("tau", g_taus)
@pytest.mark.parametrize("interlocking_time", g_interlocking_times)
@pytest.mark.parametrize("dead_time", g_dead_times)
def test_discrete_multi_power_electronic_converter(tau, interlocking_time, dead_time):
    """
    setup all combinations of single converters and test the convert function if no error is raised
    :return:
    """
    # define all converter
    all_single_disc_converter = ['Finite-1QC', 'Finite-2QC', 'Finite-4QC', 'Finite-B6C']
    interlocking_time *= tau

    for conv_0 in all_single_disc_converter:
        for conv_1 in all_single_disc_converter:
            converter = make_module(
                cv.PowerElectronicConverter, 'Finite-Multi', tau=tau,
                interlocking_time=interlocking_time, dead_time=dead_time,
                subconverters=[conv_0, conv_1]
            )
            comparable_converter_0 = make_module(cv.PowerElectronicConverter, conv_0, tau=tau,
                                                 interlocking_time=interlocking_time, dead_time=dead_time)
            comparable_converter_1 = make_module(cv.PowerElectronicConverter, conv_1, tau=tau,
                                                 interlocking_time=interlocking_time, dead_time=dead_time)
            action_space_n = converter.action_space.nvec
            assert np.all(
                converter.reset() ==
                np.concatenate([[-0.5, -0.5, -0.5] if ('Finite-B6C' == conv) else [0] for conv in [conv_0, conv_1]])
            )  # test if reset returns 0.0
            actions = [[np.random.randint(0, upper_bound) for upper_bound in action_space_n] for _ in range(100)]
            times = np.arange(100) * tau
            for action, t in zip(actions, times):
                time_steps = converter.set_action(action, t)
                time_steps_1 = comparable_converter_0.set_action(action[0], t)
                time_steps_2 = comparable_converter_1.set_action(action[1], t)
                for time_step in time_steps_1 + time_steps_2:
                    assert time_step in time_steps
                for time_step in time_steps:
                    i_in_0 = np.random.uniform(-1, 1, 3) if conv_0 == 'Finite-B6C' else [np.random.uniform(-1, 1)]
                    i_in_1 = np.random.uniform(-1, 1, 3) if conv_1 == 'Finite-B6C' else [np.random.uniform(-1, 1)]
                    i_in = np.concatenate([i_in_0, i_in_1])
                    voltage = converter.convert(i_in, time_step)
                    # test if the single phase converters work independent and correct for singlephase subsystems
                    if 'Finite-B6C' not in [conv_0, conv_1]:
                        voltage_0 = comparable_converter_0.convert(i_in_0, time_step)
                        voltage_1 = comparable_converter_1.convert(i_in_1, time_step)
                        converter.i_sup(i_in)
                        assert voltage[0] == voltage_0[0], "First converter is wrong"
                        assert voltage[1] == voltage_1[0], "Second converter is wrong"


# endregion


# region continuous converter


@pytest.mark.parametrize("converter_type", ['Cont-1QC', 'Cont-2QC', 'Cont-4QC'])
@pytest.mark.parametrize("tau", g_taus)
@pytest.mark.parametrize("interlocking_time", g_interlocking_times)
@pytest.mark.parametrize("dead_time", g_dead_times)
def test_continuous_power_electronic_converter(converter_type, tau, interlocking_time, dead_time):
    """
    test the functions and especially the conversion of continuous single converter
    :param converter_type:
    :return:
    """
    interlocking_time *= tau
    # setup converter
    converter = make_module(cv.PowerElectronicConverter, converter_type, tau=tau,
                            interlocking_time=interlocking_time, dead_time=dead_time)
    assert converter.reset() == [0.0], "Error reset function"
    action_space = converter.action_space
    # take 100 random actions
    seed(123)
    actions = [[uniform(action_space.low, action_space.high)] for _ in range(len(g_times_cont))]
    times = g_times_cont * tau

    continuous_converter_functions_testing(converter, times, interlocking_time, dead_time, actions, converter_type)


def continuous_converter_functions_testing(converter, times, interlocking_time, dead_time, actions, converter_type):
    """
    test function for conversion
    :param converter: instantiated converter
    :param times: times for set actions
    :param interlocking_time: used interlocking time
    :param dead_time: used dead time
    :param actions: random actions
    :param converter_type: converter name (string)
    :return:
    """
    tau = converter._tau
    last_action = [np.zeros_like(actions[0])]
    for index, time in enumerate(times):
        action = actions[index]
        parameters = " Error during set action " + str(dead_time) + "  " \
                     + str(interlocking_time) + "  " + str(action) + "  " + str(time)
        time_steps = converter.set_action(action, time)
        for time_step in time_steps:
            for i_in in g_i_ins_cont:
                # test if conversion works
                if converter_type == 'Cont-1QC':
                    i_in = abs(i_in)
                conversion = converter.convert([i_in], time_step)
                voltage = comparable_voltage(converter_type, action[0], i_in, tau, interlocking_time, dead_time,
                                             last_action[0])
                assert abs((voltage[0] - conversion[0])) < 1E-5, "Wrong voltage: " + \
                                                                 str([tau, dead_time, interlocking_time, action,
                                                                      last_action, time_step, i_in, conversion,
                                                                      voltage])
                params = parameters + "  " + str(i_in) + "  " + str(time_step) + "  " + str(conversion)
                assert (converter.action_space.low.tolist() <= conversion) and \
                       (converter.action_space.high.tolist() >= conversion), \
                    "Error, does not hold limits:" + str(params)
        last_action = action


def comparable_voltage(converter_type, action, i_in, tau, interlocking_time, dead_time, last_action):
    if dead_time:
        voltage = np.array([last_action])
    else:
        voltage = np.array([action])
    error = np.array([- np.sign(i_in) / tau * interlocking_time])
    if converter_type == 'Cont-2QC':
        voltage += error
        voltage = max(min(voltage, np.array([1])), np.array([0]))
    elif converter_type == 'Cont-4QC':
        voltage_1 = (1 + voltage) / 2 + error
        voltage_2 = (1 - voltage) / 2 + error
        voltage_1 = max(min(voltage_1, 1), 0)
        voltage_2 = max(min(voltage_2, 1), 0)
        voltage = voltage_1 - voltage_2
        voltage = max(min(voltage[0], np.array([1])), np.array([-1]))
    return voltage


@pytest.mark.parametrize("tau", g_taus)
@pytest.mark.parametrize("interlocking_time", g_interlocking_times)
@pytest.mark.parametrize("dead_time", g_dead_times)
def test_continuous_multi_power_electronic_converter(tau, interlocking_time, dead_time):
    """
    test functions of continuous double converter
    :return:
    """
    # define all converter
    all_single_cont_converter = ['Cont-1QC', 'Cont-2QC', 'Cont-4QC', 'Cont-B6C']
    interlocking_time *= tau
    times = g_times_cont * tau
    for conv_1 in all_single_cont_converter:
        for conv_2 in all_single_cont_converter:
            # setup converter with all possible combinations
            converter = make_module(cv.PowerElectronicConverter, 'Cont-Multi', tau=tau,
                                    interlocking_time=interlocking_time, dead_time=dead_time,
                                    subconverters=[conv_1, conv_2])
            assert all(converter.reset() ==
                       np.concatenate([[-0.5, -0.5, -0.5] if ('Cont-B6C' == conv) else [0] for conv in [conv_1, conv_2]]))
            action_space = converter.action_space
            seed(123)
            actions = [uniform(action_space.low, action_space.high) for _ in range(0, 100)]
            continuous_multi_converter_functions_testing(converter, times, interlocking_time, dead_time, actions,
                                                          [conv_1, conv_2])


def continuous_multi_converter_functions_testing(converter, times, interlocking_time, dead_time, actions,
                                                  converter_type):
    """
    test function for conversion
    :param converter: instantiated converter
    :param times: times for set actions
    :param interlocking_time: used interlocking time
    :param dead_time: used dead time
    :param actions: random actions
    :param converter_type: converter name (string)
    :return:
    """
    tau = converter._tau
    last_action = np.zeros_like(actions[0])
    for index, time in enumerate(times):
        action = actions[index]
        parameters = " Error during set action " + str(dead_time) + "  " \
                     + str(interlocking_time) + "  " + str(action) + "  " + str(time)
        time_steps = converter.set_action(action, time)
        for time_step in time_steps:
            for i_in in g_i_ins_cont:
                # test if conversion works
                i_in_0 = [i_in] * 3 if converter_type[0] == 'Cont-B6C' else [i_in]
                i_in_1 = [-i_in] * 3 if converter_type[1] == 'Cont-B6C' else [i_in]
                if converter_type[0] == 'Cont-1QC':
                    i_in_0 = np.abs(i_in_0)
                if converter_type[1] == 'Cont-1QC':
                    i_in_1 = np.abs(i_in_1)
                conversion = converter.convert(np.concatenate([i_in_0, i_in_1]), time_step)
                params = parameters + "  " + str(i_in) + "  " + str(time_step) + "  " + str(conversion)
                # test if the limits are hold
                assert (converter.action_space.low.tolist() <= conversion) and \
                       (converter.action_space.high.tolist() >= conversion), \
                    "Error, does not hold limits:" + str(params)
                # test if the single phase converters work independent and correct for singlephase subsystems
                if 'Cont-B6C' not in converter_type:
                    voltage_0 = comparable_voltage(converter_type[0], action[0], i_in_0[0], tau, interlocking_time, dead_time,
                                                   last_action[0])
                    voltage_1 = comparable_voltage(converter_type[1], action[1], i_in_1[0], tau, interlocking_time, dead_time,
                                                   last_action[1])
                    assert abs(voltage_0 - conversion[0]) < 1E-5, "Wrong converter value for armature circuit"
                    assert abs(voltage_1 - conversion[1]) < 1E-5, "Wrong converter value for excitation circuit"
        last_action = action


# endregion


# region three phase converter


def test_discrete_b6_bridge():
    """
    test discrete b6 bridge
    :return:
    """

    tau = cf.converter_parameter['tau']
    # test default initializations
    converter_default_init_1 = make_module(cv.PowerElectronicConverter, 'Finite-B6C')
    converter_default_init_2 = cv.FiniteB6BridgeConverter()
    assert converter_default_init_1._tau == 1E-5
    for subconverter in converter_default_init_1._subconverters:
        assert subconverter._tau == 1E-5
        assert subconverter._interlocking_time == 0
        assert subconverter._dead_time is False

    # test default initialized converter
    converters_default = [converter_default_init_1, converter_default_init_2]
    for converter in converters_default:
        assert all(converter.reset() == -0.5 * np.ones(3))
        assert converter._subconverters[0].action_space.n == 3
        assert converter.action_space.n == 8

        # 1  1  1  1  2  2  2  1  2  1  # Action for the converter
        actions_1 = [4, 5, 6, 7, 0, 1, 2, 5, 3, 6]
        actions_2 = [2, 3, 6, 7, 0, 1, 4, 2, 5, 6]
        actions_3 = [1, 3, 5, 7, 0, 2, 4, 3, 6, 5]
        actions = [actions_1, actions_2, actions_3]
        times = np.arange(len(actions[0])) * tau
        i_ins = [0.5, 0, -0.5, 0.5, 0.5, 0, -0.5, -0.5, 0.5, 0.5]
        u_out = np.array([1, 1, 1, 1, -1, -1, -1, 1, -1, 1])
        # test each subconverter individually
        for k in range(3):
            converter.reset()
            step_counter = 0
            i_in = [[0.5], [0], [-0.5]]
            for time, action, i_in_ in zip(times, actions[k], i_ins):
                time_steps = converter.set_action(action, time)
                for time_step in time_steps:
                    i_in[k] = [i_in_]
                    voltage = converter.convert(i_in, time_step)
                    assert voltage[k] == 0.5 * u_out[step_counter], "Wrong action without dead time " + str(
                        step_counter)
                    step_counter += 1

    # test parametrized converter
    converter_init_1 = make_module(cv.PowerElectronicConverter, 'Finite-B6C', **cf.converter_parameter)
    converter_init_2 = cv.FiniteB6BridgeConverter(**cf.converter_parameter)
    assert converter_init_1._tau == cf.converter_parameter['tau']
    for subconverter in converter_init_1._subconverters:
        assert subconverter._tau == cf.converter_parameter['tau']
        assert subconverter._interlocking_time == cf.converter_parameter['interlocking_time']
        assert subconverter._dead_time == cf.converter_parameter['dead_time']
    # set parameter
    actions = [6, 6, 4, 5, 1, 2, 3, 7, 0, 4]
    i_ins = [[[0.5], [0.5], [-0.5]],
             [[0], [0.5], [0]],
             [[-0.5], [0.5], [-0.5]],
             [[0.5], [0.5], [0.5]],
             [[0.5], [0.5], [-0.5]],
             [[0], [-0.5], [0]],
             [[-0.5], [-0.5], [0.5]],
             [[-0.5], [-0.5], [-0.5]],
             [[0.5], [-0.5], [0]],
             [[0.5], [-0.5], [0.5]]]

    expected_voltage = np.array([[-1, -1, 1],
                                 [1, 1, -1],
                                 [1, 1, -1],
                                 [1, -1, -1],
                                 [1, -1, -1],
                                 [1, -1, 1],
                                 [1, -1, 1],
                                 [-1, -1, 1],
                                 [-1, -1, 1],
                                 [-1, 1, -1],
                                 [-1, 1, -1],
                                 [-1, 1, 1],
                                 [-1, 1, 1],
                                 [-1, 1, 1],
                                 [1, 1, 1],
                                 [-1, 1, -1],
                                 [-1, -1, -1]]) / 2

    times = np.arange(len(actions)) * tau
    converters_init = [converter_init_1, converter_init_2]
    # test every initialized converter for a given action sequence
    for converter in converters_init:
        converter.reset()
        step_counter = 0
        for time, action, i_in in zip(times, actions, i_ins):
            time_steps = converter.set_action(action, time)
            i_in = np.array(i_in)
            for time_step in time_steps:
                voltage = np.array(converter.convert(i_in, time_step))
                converter.i_sup(i_in)
                assert all(voltage == expected_voltage[step_counter]), "Wrong voltage calculated " + str(step_counter)
                step_counter += 1


def test_continuous_b6_bridge():
    converter_default_init_1 = cv.ContB6BridgeConverter()
    converter_default_init_2 = make_module(cv.PowerElectronicConverter, 'Cont-B6C')
    converters_default = [converter_default_init_1, converter_default_init_2]
    actions = np.array([[1, -1, 0.65],
                        [0.75, -0.95, -0.3],
                        [-0.25, 0.98, -1],
                        [0.65, 0.5, -0.95],
                        [-0.3, 0.5, 0.98],
                        [-1, 1, 0.5],
                        [-0.95, 0.75, 0.5],
                        [0.98, -0.25, 1],
                        [0.5, 0.65, 0.75],
                        [0.5, -0.3, -0.25]])
    i_ins = [[[0.5], [-0.2], [1]],
             [[-0.6], [-0.5], [0.8]],
             [[0.3], [0.4], [0.9]],
             [[-0.2], [0.7], [0.5]],
             [[-0.5], [1], [-0.6]],
             [[0.4], [0.8], [0.3]],
             [[0.7], [0.9], [-0.2]],
             [[1], [0.5], [-0.5]],
             [[0.8], [-0.6], [0.4]],
             [[0.9], [0.75], [0.7]]]

    times = np.arange(len(actions)) * 1E-4
    for converter in converters_default:
        # parameter testing
        assert converter._tau == 1E-4
        assert converter._dead_time is False
        assert converter._interlocking_time == 0
        assert all(converter.reset() == -0.5 * np.ones(3))
        assert converter.action_space.shape == (3,)
        assert all(converter.action_space.low == -1 * np.ones(3))
        assert all(converter.action_space.high == 1 * np.ones(3))
        for subconverter in converter._subconverters:
            assert subconverter._tau == 1E-4
            assert subconverter._dead_time is False
            assert subconverter._interlocking_time == 0
        # conversion testing
        for time, action, i_in in zip(times, actions, i_ins):
            i_in = np.array(i_in)
            i_sup = converter.i_sup(i_in)
            time_step = converter.set_action(action, time)
            voltages = converter.convert(i_in, time_step)
            for voltage, single_action in zip(voltages, action):
                assert abs(voltage[0] - single_action / 2) < 1E-9

    # testing parametrized converter
    expected_voltages = np.array([[-5, -4.95, -5],
                                  [5, -4.95, 3.2],
                                  [3.7, -4.8, -1.55],
                                  [-1.2, 4.85, -5],
                                  [3.3, 2.45, -4.7],
                                  [-1.55, 2.45, 4.85],
                                  [-5, 4.95, 2.55],
                                  [-4.8, 3.7, 2.55],
                                  [4.85, -1.2, 4.95],
                                  [2.45, 3.2, 3.7]]) / 10

    converter_init_1 = cv.ContB6BridgeConverter(**cf.converter_parameter)
    converter_init_2 = make_module(cv.PowerElectronicConverter, 'Cont-B6C', **cf.converter_parameter)
    converters = [converter_init_1, converter_init_2]
    for converter in converters:
        # parameter testing
        assert converter._tau == cf.converter_parameter['tau']
        assert converter._dead_time == cf.converter_parameter['dead_time']
        assert converter._interlocking_time == cf.converter_parameter['interlocking_time']
        assert all(converter.reset() == -0.5 * np.ones(3))
        assert converter.action_space.shape == (3,)
        assert all(converter.action_space.low == -1 * np.ones(3))
        assert all(converter.action_space.high == 1 * np.ones(3))
        for subconverter in converter._subconverters:
            assert subconverter._tau == cf.converter_parameter['tau']
            assert subconverter._dead_time == cf.converter_parameter['dead_time']
            assert subconverter._interlocking_time == cf.converter_parameter['interlocking_time']
        # conversion testing
        for time, action, i_in, expected_voltage in zip(times, actions, i_ins, expected_voltages):
            i_in = np.array(i_in)
            time_step = converter.set_action(action.tolist(), time)
            voltages = converter.convert(i_in, time_step)
            for voltage, test_voltage in zip(voltages, expected_voltage):
                assert abs(voltage - test_voltage) < 1E-9

# endregion


# endregion


# region second version tests


class TestPowerElectronicConverter:

    class_to_test = cv.PowerElectronicConverter
    key = ''

    @pytest.fixture
    def converter(self):
        return self.class_to_test(tau=0, dead_time=False, interlocking_time=0)

    @pytest.mark.parametrize("tau, dead_time, interlocking_time, kwargs", [
        (1, True, 0.1, {}),
        (0.1, False, 0.0, {}),
    ])
    def test_initialization(self, tau, dead_time, interlocking_time, kwargs):
        converter = self.class_to_test(tau=tau, dead_time=dead_time, interlocking_time=interlocking_time, **kwargs)
        assert converter._tau == tau
        assert converter._dead_time == dead_time
        assert converter._interlocking_time == interlocking_time

    def test_registered(self):
        if self.key != '':
            conv = gem.utils.instantiate(cv.PowerElectronicConverter, self.key)
            assert type(conv) == self.class_to_test

    def test_reset(self, converter):
        assert converter._dead_time_action == converter._reset_action == converter._current_action
        assert converter._action_start_time == 0
        assert np.all(converter.reset() == np.array([0]))

    @pytest.mark.parametrize("action_space", [Discrete(3), Box(-1, 1, shape=(1,))])
    def test_set_action(self, monkeypatch, converter, action_space):
        monkeypatch.setattr(converter, "_set_switching_pattern", lambda: [0.0])
        monkeypatch.setattr(converter, "_dead_time", False)
        monkeypatch.setattr(converter, "action_space", converter.action_space or action_space)
        next_action = converter.action_space.sample()
        converter.set_action(next_action, 0)
        assert np.all(converter._current_action == next_action)
        assert converter._action_start_time == 0
        monkeypatch.setattr(converter, "_dead_time", True)
        action_2 = converter.action_space.sample()
        action_3 = converter.action_space.sample()
        while np.all(action_3 == action_2):
            action_3 = converter.action_space.sample()
        converter.set_action(action_2, 1)
        converter.set_action(action_3, 2)
        assert np.all(converter._current_action == action_2)
        assert np.all(converter._dead_time_action == action_3)
        assert converter._action_start_time == 2


class TestContDynamicallyAveragedConverter(TestPowerElectronicConverter):

    class_to_test = cv.ContDynamicallyAveragedConverter

    @pytest.fixture
    def converter(self, monkeypatch):
        converter = self.class_to_test(tau=0.1, dead_time=False, interlocking_time=0)
        converter.action_space = converter.action_space or Box(-1, 1, shape=(1,))
        if converter.voltages and converter.currents:
            converter.voltages = Box(converter.voltages.low[0], converter.voltages.high[0], shape=(1,)) #(converter.voltages.low[0] or -1, converter.voltages.high[0] or -1)
            converter.currents = Box(converter.voltages.low[0], converter.voltages.high[0], shape=(1,)) #(converter.currents.low[0] or  0, converter.currents.high[0] or  1)
        else:
            converter.voltages = Box(-1, -1, shape=(1,))
            converter.currents = Box(0, 1 ,shape=(1,))
        monkeypatch.setattr(converter, "_convert", lambda i_in, t: i_in[0])
        return converter

    @pytest.mark.parametrize('i_out', [[-2], [-0.5], [0], [0.5], [2]])
    def test_convert(self, monkeypatch, converter, i_out):
        monkeypatch.setattr(converter, '_interlocking_time', converter._tau / 10)
        u = converter.convert(i_out, 0)
        assert u[0] == min(
            max(
                converter._convert(i_out, 0) - converter._interlock(i_out),
                converter.voltages.low[0]
            ),
            converter.voltages.high[0]
        )

    @pytest.mark.parametrize("action_space", [Box(-1, 1, shape=(1,))])
    def test_set_action(self, monkeypatch, converter, action_space):
        super().test_set_action(monkeypatch, converter, action_space)

    def test_action_clipping(self, converter):
        low_action = converter.action_space.low - np.ones_like(converter.action_space.low)
        converter.set_action(low_action, 0)
        assert np.all(converter._current_action == converter.action_space.low)
        high_action = converter.action_space.high[0] + np.ones_like(converter.action_space.high)
        converter.set_action(high_action, 0)
        assert np.all(converter._current_action == converter.action_space.high)
        fitting_action = converter.action_space.sample()
        converter.set_action(fitting_action, 0)
        assert np.all(converter._current_action == fitting_action)

    @pytest.mark.parametrize("interlocking_time", [0.0, 0.2, 1.0])
    def test_interlock(self, monkeypatch, converter, interlocking_time):
        monkeypatch.setattr(converter, "_interlocking_time", interlocking_time)
        assert converter._interlock([-1]) == - interlocking_time / converter._tau
        assert converter._interlock([1]) == interlocking_time / converter._tau


class TestFiniteConverter(TestPowerElectronicConverter):

    class_to_test = cv.FiniteConverter

    @pytest.mark.parametrize("action_space", [1, 2, 3, 4])
    def test_set_action(self, monkeypatch, action_space):
        converter = self.class_to_test()
        monkeypatch.setattr(converter, "action_space", Discrete(action_space))
        time = 0
        with pytest.raises(AssertionError) as assertText:
            converter.set_action(-1, time)
        assert "-1" in str(assertText.value) and "Discrete(" + str(action_space) + ")" in str(assertText.value)

        with pytest.raises(AssertionError) as assertText:
            converter.set_action(int(1e9), time)
        assert str(int(1e9)) in str(assertText.value) and "Discrete(" + str(action_space) + ")" in str(assertText.value)

        with pytest.raises(AssertionError) as assertText:
            converter.set_action(np.pi, time)
        assert str(np.pi) in str(assertText.value) and "Discrete(" + str(action_space) + ")" in str(assertText.value)

    def test_default_init(self):
        converter = self.class_to_test()
        assert converter._tau == 1e-5


class TestFiniteOneQuadrantConverter(TestFiniteConverter):

    class_to_test = cv.FiniteOneQuadrantConverter
    key = 'Finite-1QC'

    def test_convert(self, converter):
        action = converter.action_space.sample()
        converter.set_action(action, 0)
        assert converter.convert([-1], 0) == [1]
        assert converter.convert([1], 0) == [action]

    @pytest.mark.parametrize("i_sup", [-12, 12])
    def test_i_sup(self, converter, i_sup):
        converter.set_action(1, 0)
        assert converter.i_sup([i_sup]) == i_sup
        converter.set_action(0, 0)
        assert converter.i_sup([i_sup]) == 0


class TestFiniteTwoQuadrantConverter(TestFiniteConverter):
    class_to_test = cv.FiniteTwoQuadrantConverter
    key = 'Finite-2QC'

    @pytest.mark.parametrize("interlocking_time", [0.0, 0.1])
    def test_set_switching_pattern(self, monkeypatch, converter, interlocking_time):
        monkeypatch.setattr(converter, "_dead_time", False)
        monkeypatch.setattr(converter, "_interlocking_time", interlocking_time)

        # Test if no interlocking step required after switching state 0 (
        monkeypatch.setattr(converter, "_switching_state", 0)
        assert converter._set_switching_pattern() == [converter._tau]
        assert converter._switching_pattern == [converter._current_action]
        # Test if no interlocking step required, if the same action is set again
        monkeypatch.setattr(converter, "_switching_state", 1)
        monkeypatch.setattr(converter, "_current_action", 1)
        monkeypatch.setattr(converter, "_action_start_time", 1)
        assert converter._set_switching_pattern() == [converter._tau + 1]
        assert converter._switching_pattern == [converter._current_action]
        # Test if interlocking step is required, if action changes to another <> 0
        monkeypatch.setattr(converter, "_switching_state", 1)
        monkeypatch.setattr(converter, "_current_action", 2)
        monkeypatch.setattr(converter, "_action_start_time", 2)
        switching_times = converter._set_switching_pattern()
        if interlocking_time > 0:
            assert switching_times == [interlocking_time + 2, converter._tau + 2]
            assert converter._switching_pattern == [0, converter._current_action]
        else:
            assert switching_times == [converter._tau + 2]
            assert converter._switching_pattern == [converter._current_action]

    @pytest.mark.parametrize('i_out', [[-1], [0], [1]])
    def test_i_sup(self, monkeypatch, converter, i_out):
        monkeypatch.setattr(converter, "_switching_state", 0)
        assert converter.i_sup(i_out) == min(i_out[0], 0)
        monkeypatch.setattr(converter, "_switching_state", 1)
        assert converter.i_sup(i_out) == i_out[0]
        monkeypatch.setattr(converter, "_switching_state", 2)
        assert converter.i_sup(i_out) == 0

    @pytest.mark.parametrize('i_out', [[-1], [0], [1]])
    def test_convert(self, monkeypatch, converter, i_out):
        monkeypatch.setattr(converter, '_interlocking_time', 0.2)
        monkeypatch.setattr(converter, '_action_start_time', 0)
        monkeypatch.setattr(converter, '_switching_pattern', [0, 1])
        for t in np.linspace(0, 1, 10):
            u = converter.convert(i_out, t)
            assert converter._switching_state == int(t > converter._interlocking_time)
            if t < converter._interlocking_time:
                assert u == [int(i_out[0] < 0)]
            else:
                assert u == [1]
        monkeypatch.setattr(converter, '_switching_pattern', [-1])
        with pytest.raises(Exception):
            converter.convert(i_out, 0)


class TestFiniteFourQuadrantConverter(TestFiniteConverter):

    class_to_test = cv.FiniteFourQuadrantConverter
    key = 'Finite-4QC'

    @pytest.fixture
    def converter(self):
        converter = self.class_to_test()
        tau = converter._tau
        converter._subconverters = [
            PowerElectronicConverterWrapper(converter._subconverters[0], tau=tau),
            PowerElectronicConverterWrapper(converter._subconverters[1], tau=tau)
        ]
        return converter

    def test_set_action(self, converter, *_):
        for action in range(converter.action_space.n):
            t = np.random.rand()
            converter.set_action(action, t)
            assert converter._subconverters[0].last_t == t
            assert converter._subconverters[1].last_t == t
            assert converter._subconverters[0].last_action == action // 2 + 1
            assert converter._subconverters[1].last_action == action % 2 + 1

        converter = self.class_to_test()
        time = 0
        with pytest.raises(AssertionError) as assertText:
            converter.set_action(-1, time)
        assert "-1" in str(assertText.value) and "Discrete(4)" in str(assertText.value)

        with pytest.raises(AssertionError) as assertText:
            converter.set_action(int(1e9), time)
        assert str(int(1e9)) in str(assertText.value) and "Discrete(4)" in str(assertText.value)

        with pytest.raises(AssertionError) as assertText:
            converter.set_action(np.pi, time)
        assert str(np.pi) in str(assertText.value) and "Discrete(4)" in str(assertText.value)

    @pytest.mark.parametrize('i_out', [[-12], [0], [12]])
    def test_convert(self, converter, i_out):
        t = np.random.rand()
        converter.set_action(converter.action_space.sample(), t)
        u = converter.convert(i_out, t)
        assert converter._subconverters[0].last_t == t
        assert converter._subconverters[0].last_i_out == i_out
        assert converter._subconverters[1].last_t == t
        assert converter._subconverters[1].last_i_out == [-i_out[0]]
        assert u == [converter._subconverters[0].last_u[0] - converter._subconverters[1].last_u[0]]

    def test_reset(self, converter):
        reset_calls = [conv.reset_calls for conv in converter._subconverters]
        super().test_reset(converter)
        assert np.all([conv.reset_calls == reset_calls[0] + 1 for conv in converter._subconverters])

    @pytest.mark.parametrize('i_out', [[-1], [0], [1]])
    def test_i_sup(self, converter, i_out):
        for action in range(converter.action_space.n):
            converter.set_action(action, 0)
            converter.convert(i_out, 0)
            i_sup = converter.i_sup(i_out)
            assert converter._subconverters[0].last_i_out == i_out
            assert converter._subconverters[1].last_i_out == [-i_out[0]]
            assert i_sup == converter._subconverters[0].last_i_sup + converter._subconverters[1].last_i_sup


class TestContOneQuadrantConverter(TestContDynamicallyAveragedConverter):
    key = 'Cont-1QC'
    class_to_test = cv.ContOneQuadrantConverter

    @pytest.fixture
    def converter(self):
        return cv.ContOneQuadrantConverter()

    def test_interlock(self, monkeypatch, converter, *_):
        assert converter._interlock(0) == 0

    @pytest.mark.parametrize('i_in', [[-1], [0], [1]])
    @pytest.mark.parametrize('action', [[0], [1]])
    def test__convert(self, monkeypatch, converter, i_in, action):
        monkeypatch.setattr(converter, '_current_action', action)
        u = converter._convert(i_in, 0)
        if i_in[0] >= 0:
            assert u == action[0]
        else:
            assert u == 1

    def test_set_action(self, monkeypatch, converter, *_):
        super().test_set_action(monkeypatch, converter, converter.action_space)

    @pytest.mark.parametrize('i_out', [[-1], [1], [0]])
    def test_i_sup(self, monkeypatch, converter, i_out):
        monkeypatch.setattr(converter, '_current_action', converter.action_space.sample())
        assert converter.i_sup(i_out) == converter._current_action[0] * i_out[0]


class TestContTwoQuadrantConverter(TestContDynamicallyAveragedConverter):
    class_to_test = cv.ContTwoQuadrantConverter
    key = 'Cont-2QC'

    @pytest.mark.parametrize('interlocking_time', [0.0, 0.1, 1])
    @pytest.mark.parametrize('i_out', [[0.0], [0.1], [-1]])
    @pytest.mark.parametrize('tau', [1, 2])
    @pytest.mark.parametrize('action', [[0], [0.5], [1]])
    def test_i_sup(self, monkeypatch, converter, interlocking_time, i_out, tau, action):
        monkeypatch.setattr(converter, '_interlocking_time', min(tau, interlocking_time))
        monkeypatch.setattr(converter, '_tau', tau)
        monkeypatch.setattr(converter, '_current_action', action)
        i_sup = converter.i_sup(i_out)
        assert abs(i_sup) <= abs(i_out[0])
        if interlocking_time == 0:
            assert i_sup == action[0] * i_out[0]
        if i_out == [0]:
            assert i_sup == 0
        if action == [0]:
            assert i_sup <= 0


class TestContFourQuadrantConverter(TestContDynamicallyAveragedConverter):
    class_to_test = cv.ContFourQuadrantConverter
    key = 'Cont-4QC'

    @pytest.fixture
    def converter(self, monkeypatch):
        converter = self.class_to_test()
        subconverters = [
            PowerElectronicConverterWrapper(subconverter, tau=converter._tau)
            for subconverter in converter._subconverters
        ]
        monkeypatch.setattr(converter, '_subconverters', subconverters)
        return converter

    @pytest.mark.parametrize('i_out', [[-2], [-0.5], [0], [0.5], [2]])
    def test_convert(self, monkeypatch, converter, i_out):
        converter.set_action(converter.action_space.sample(), 0)
        for _ in np.linspace(0, converter._tau):
            u = converter.convert(i_out, 0)
            assert u[0] == converter._subconverters[0].last_u[0] - converter._subconverters[1].last_u[0]

    def test_reset(self, converter):
        u = converter.reset()
        assert converter._subconverters[0].reset_calls == 1
        assert converter._subconverters[1].reset_calls == 1
        assert u == [0.0]

    def test_set_action(self, monkeypatch, converter, *_):
        for _ in range(10):
            action = converter.action_space.sample()
            t = np.random.rand()
            converter.set_action(action, t)
            sc1, sc2 = converter._subconverters
            assert sc1.last_action[0] == 1 - sc2.last_action[0]
            assert action[0] == 2 * sc1.last_action[0] - 1
            assert action[0] == -2 * sc2.last_action[0] + 1

    @pytest.mark.parametrize('i_out', [[0], [1], [-2]])
    def test_i_sup(self, converter, i_out):
        sc1, sc2 = converter._subconverters
        converter.set_action(converter.action_space.sample(), np.random.rand())
        i_sup = converter.i_sup(i_out)
        assert sc1.last_i_out == i_out
        assert -sc2.last_i_out[0] == i_out[0]
        assert i_sup == sc1.last_i_sup + sc2.last_i_sup


class TestFiniteMultiConverter(TestFiniteConverter):
    class_to_test = cv.FiniteMultiConverter
    key = 'Finite-Multi'

    @pytest.fixture
    def converter(self):
        return self.class_to_test(subconverters=[
            DummyConverter(action_space=Discrete(3), voltages=Box(-1, 1, shape=(1,)), currents=Box(0, 1, shape=(1,))),
            DummyConverter(action_space=Discrete(8), voltages=Box(-1, 1, shape=(3,)), currents=Box(-1, 1, shape=(3,))),
            DummyConverter(action_space=Discrete(4), voltages=Box(-1, 1, shape=(1,)), currents=Box(-1, 1, shape=(1,))),
        ])

    @pytest.mark.parametrize("tau, dead_time, interlocking_time, kwargs", [
        (1, True, 0.1, {'subconverters': ['Finite-1QC', 'Finite-B6C', 'Finite-4QC']}),
        (0.1, False, 0.0, {'subconverters': ['Finite-1QC', 'Finite-B6C', 'Finite-4QC']}),
    ])
    def test_initialization(self, tau, dead_time, interlocking_time, kwargs):
        super().test_initialization(tau, dead_time, interlocking_time, kwargs)
        conv = self.class_to_test(tau=tau, dead_time=dead_time, interlocking_time=interlocking_time, **kwargs)
        assert np.all(
            conv.subsignal_voltage_space_dims ==
            np.array([(np.squeeze(subconv.voltages.shape) or 1) for subconv in conv._subconverters])
        ), "Voltage space dims in the multi converter do not fit the subconverters."
        assert np.all(
            conv.subsignal_current_space_dims ==
            np.array([(np.squeeze(subconv.currents.shape) or 1) for subconv in conv._subconverters])
        ), "Current space dims in the multi converter do not fit the subconverters."
        assert np.all(conv.action_space.nvec == [subconv.action_space.n for subconv in conv._subconverters]
                      ), "Action space of the multi converter does not fit the subconverters."
        for sc in conv._subconverters:
            assert sc._interlocking_time == interlocking_time
            assert sc._dead_time == dead_time
            assert sc._tau == tau

    def test_registered(self):
        dummy_converters = [DummyConverter(), DummyConverter(), DummyConverter()]
        conv = gem.utils.instantiate(cv.PowerElectronicConverter, self.key, subconverters=dummy_converters)
        assert type(conv) == self.class_to_test

    def test_reset(self, converter):
        u_in = converter.reset()
        assert u_in == [0.0] * converter.voltages.shape[0]
        assert np.all([subconv.reset_counter == 1 for subconv in converter._subconverters])

    def test_set_action(self, monkeypatch, converter, **_):
        for action in np.ndindex(tuple(converter.action_space.nvec)):
            sc0 = converter._subconverters[0]
            sc1 = converter._subconverters[1]
            sc2 = converter._subconverters[2]
            t = (action[0] * sc1.action_space.n * sc2.action_space.n +
                 action[1] * sc2.action_space.n + action[2])* converter._tau
            converter.set_action(action, t)
            assert np.all((sc0.action,  sc1.action, sc2.action) == action)
            assert sc0.action_set_time == t
            assert sc1.action_set_time == t
            assert sc2.action_set_time == t

    def test_default_init(self):
        converter = self.class_to_test(subconverters=['Finite-1QC', 'Finite-B6C', 'Finite-2QC'])
        assert converter._tau == 1e-5

    @pytest.mark.parametrize('i_out', [[0, 6, 2, 7, 9], [1, 0.5, 2], [-1, 1]])
    def test_convert(self, converter, i_out):
        # Setting of the demanded output voltages from the dummy converters
        for subconverter in converter._subconverters:
            subconverter.action = subconverter.action_space.sample()
        u = converter.convert(i_out, 0)
        assert np.all(
            u == [subconv.action for subconv in converter._subconverters]
        )

    @pytest.mark.parametrize('i_out',  [[0, 6, 2, 7, 9], [1, 0.5, 2, -7, 50], [-1, 1, 0.01, 16, -42]])
    def test_i_sup(self, converter, i_out):
        sc0, sc1, sc2 = converter._subconverters
        converter.set_action(converter.action_space.sample(), np.random.rand())
        i_sup = converter.i_sup(i_out)
        assert sc0.last_i_out + sc1.last_i_out + sc2.last_i_out == i_out
        assert i_sup == sc0.last_i_out[0] + sc1.last_i_out[0] + sc2.last_i_out[0]


class TestContMultiConverter(TestContDynamicallyAveragedConverter):
    class_to_test = cv.ContMultiConverter
    key = 'Cont-Multi'

    @pytest.fixture
    def converter(self):
        return self.class_to_test(subconverters=[
            DummyConverter(
                action_space=Box(-1, 1, shape=(1,)), voltages=Box(-1, 1, shape=(1,)), currents=Box(-1, 1, shape=(1,))
            ),
            DummyConverter(
                action_space=Box(-1, 1, shape=(3,)), voltages=Box(-1, 1, shape=(3,)), currents=Box(-1, 1, shape=(3,))
            ),
            DummyConverter(
                action_space=Box(0, 1, shape=(1,)), voltages=Box(0, 1, shape=(1,)), currents=Box(0, 1, shape=(1,))
            ),
        ])

    def test_registered(self):
        dummy_converters = [DummyConverter(), DummyConverter(), DummyConverter()]
        dummy_converters[0].action_space = Box(-1, 1, shape=(1,))
        dummy_converters[1].action_space = Box(-1, 1, shape=(3,))
        dummy_converters[2].action_space = Box(0, 1, shape=(1,))
        conv = gem.utils.instantiate(cv.PowerElectronicConverter, self.key, subconverters=dummy_converters)
        assert type(conv) == self.class_to_test
        assert conv._subconverters == dummy_converters

    @pytest.mark.parametrize("tau, dead_time, interlocking_time, kwargs", [
        (1, True, 0.1, {'subconverters': ['Cont-1QC', 'Cont-B6C', 'Cont-4QC']}),
        (0.1, False, 0.0, {'subconverters': ['Cont-1QC', 'Cont-B6C', 'Cont-4QC']}),
    ])
    def test_initialization(self, tau, dead_time, interlocking_time, kwargs):
        super().test_initialization(tau, dead_time, interlocking_time, kwargs)
        conv = self.class_to_test(tau=tau, dead_time=dead_time, interlocking_time=interlocking_time, **kwargs)
        assert np.all(
            conv.action_space.low == np.concatenate([subconv.action_space.low for subconv in conv._subconverters])
        ), "Action space lower boundaries in the multi converter do not fit the subconverters."
        assert np.all(
            conv.action_space.high == np.concatenate([subconv.action_space.high for subconv in conv._subconverters])
        ), "Action space upper boundaries in the multi converter do not fit the subconverters."
        assert np.all(
            conv.subsignal_voltage_space_dims ==
            np.array([(np.squeeze(subconv.voltages.shape) or 1) for subconv in conv._subconverters])
        ), "Voltage space dims in the multi converter do not fit the subconverters."
        assert np.all(
            conv.subsignal_current_space_dims ==
            np.array([(np.squeeze(subconv.currents.shape) or 1) for subconv in conv._subconverters])
        ), "Current space dims in the multi converter do not fit the subconverters."
        for sc in conv._subconverters:
            assert sc._interlocking_time == interlocking_time
            assert sc._dead_time == dead_time
            assert sc._tau == tau

    def test_reset(self, converter):
        u_in = converter.reset()
        assert u_in == [0.0] * converter.voltages.shape[0]
        assert np.all([subconv.reset_counter == 1 for subconv in converter._subconverters])

    def test_action_clipping(self, converter):
        # Done by the subconverters
        pass

    @pytest.mark.parametrize('action', [[0, 0, 0, 0, 0], [0, 0, 1, 1, 1], [-1, 1, -1, 1, -1], [1, 1, 1, 1, 1], []])
    def test_set_action(self, monkeypatch, converter, action):
        t = np.random.randint(10) * converter._tau
        converter.set_action(action, t)
        sc0 = converter._subconverters[0]
        sc1 = converter._subconverters[1]
        sc2 = converter._subconverters[2]
        assert np.all(np.concatenate((sc0.action, sc1.action, sc2.action)) == action)
        assert sc0.action_set_time == t
        assert sc1.action_set_time == t

    @pytest.mark.parametrize('i_out', [[-2, 2, 0, -1, 1], [-0.5, 5, -7, 0, 2], [0, 1, 3, -0.7, -4],
                                       [], [2, 1, -2], [2, 9, 7, -4, 9, 41, 17]])
    def test_convert(self, monkeypatch, converter, i_out):
        t = np.random.rand()
        action_space_size = [1, 3, 1]
        converter.set_action([0.2, 0.4, 0.5, -0.8, 0.3], 0)
        u = converter.convert(i_out, t)
        sub_u = []
        start_idx = 0
        for subconverter, subaction_space_size in zip(converter._subconverters, action_space_size):
            end_idx = start_idx + subaction_space_size
            assert subconverter.i_out == i_out[start_idx:end_idx]
            start_idx = end_idx
            assert subconverter.t == t
            sub_u += subconverter.action
        assert u == sub_u


class TestFiniteB6BridgeConverter(TestFiniteConverter):
    class_to_test = cv.FiniteB6BridgeConverter
    key = 'Finite-B6C'

    @pytest.fixture
    def converter(self):
        conv = self.class_to_test()
        subconverters = [
            PowerElectronicConverterWrapper(subconverter, tau=conv._tau) for subconverter in conv._subconverters
        ]
        conv._subconverters = subconverters
        return conv

    @pytest.mark.parametrize("tau, dead_time, interlocking_time, kwargs", [
        (1, True, 0.1, {}),
        (0.1, False, 0.0, {}),
    ])
    def test_subconverter_initialization(self, tau, dead_time, interlocking_time, kwargs):
        conv = self.class_to_test(tau=tau, dead_time=dead_time, interlocking_time=interlocking_time, **kwargs)
        for sc in conv._subconverters:
            assert sc._interlocking_time == interlocking_time
            assert sc._dead_time == dead_time
            assert sc._tau == tau

    def test_reset(self, converter):
        u_init = converter.reset()
        assert np.all(
            subconv.reset_counter == 1 for subconv in converter._subconverters
        )
        assert u_init == [-0.5] * 3

    def test_set_action(self, converter, *_):
        for action in range(converter.action_space.n):
            t = np.random.rand()
            converter.set_action(action, t)
            assert converter._subconverters[0].last_t == t
            assert converter._subconverters[1].last_t == t
            assert converter._subconverters[2].last_t == t
            subactions = [sc.last_action % 2 for sc in converter._subconverters]
            assert action == reduce(lambda x, y: 2*x+y, subactions)

        time = 0
        with pytest.raises(AssertionError) as assertText:
            converter.set_action(-1, time)
        assert "-1" in str(assertText.value) and "Discrete(8)" in str(assertText.value)

        with pytest.raises(AssertionError) as assertText:
            converter.set_action(int(1e9), time)
        assert str(int(1e9)) in str(assertText.value) and "Discrete(8)" in str(assertText.value)

        with pytest.raises(AssertionError) as assertText:
            converter.set_action(np.pi, time)
        assert str(np.pi) in str(assertText.value) and "Discrete(8)" in str(assertText.value)

    @pytest.mark.parametrize('i_out', [[-1, -1, 0], [1, 1, -2], [0, 0, 1]])
    def test_convert(self, converter, i_out):
        t = np.random.rand()
        sc1, sc2, sc3 = converter._subconverters
        for action in range(converter.action_space.n):
            converter.set_action(action, t)
            u_out = converter.convert(i_out, t)
            assert sc1.last_i_out + sc2.last_i_out + sc3.last_i_out == i_out
            assert sc1.last_t == sc2.last_t == sc3.last_t == t
            assert sc1.last_u[0] - 0.5 == u_out[0]
            assert sc2.last_u[0] - 0.5 == u_out[1]
            assert sc3.last_u[0] - 0.5 == u_out[2]

    @pytest.mark.parametrize('i_out', [[-1, -1, 0], [1, 1, -2], [0, 0, 1]])
    def test_i_sup(self, converter, i_out):
        sc1, sc2, sc3 = converter._subconverters
        for action in range(converter.action_space.n):
            converter.set_action(action, converter._tau * action)
            i_sup = converter.i_sup(i_out)
            assert sc1.last_i_out + sc2.last_i_out + sc3.last_i_out == i_out
            assert i_sup == sc1.last_i_sup + sc2.last_i_sup + sc3.last_i_sup


class TestContB6BridgeConverter(TestContDynamicallyAveragedConverter):
    key = 'Cont-B6C'
    class_to_test = cv.ContB6BridgeConverter

    @pytest.fixture
    def converter(self):
        conv = self.class_to_test()
        subconverters = [
            PowerElectronicConverterWrapper(subconverter, tau=conv._tau) for subconverter in conv._subconverters
        ]
        conv._subconverters = subconverters
        return conv

    def test_action_clipping(self, converter):
        # Done by subconverters
        pass

    def test_reset(self, converter):
        u_init = converter.reset()
        assert np.all(
            subconv.reset_counter == 1 for subconv in converter._subconverters
        )
        assert u_init == [-0.5] * 3

    @pytest.mark.parametrize('i_out', [[-1, -1, 0], [1, 1, -2], [0, 0, 1]])
    def test_convert(self, converter, i_out):
        t = np.random.rand()
        sc1, sc2, sc3 = converter._subconverters
        for _ in range(10):
            action = converter.action_space.sample()
            converter.set_action(action, t)
            u_out = converter.convert(i_out, t)
            assert sc1.last_i_out + sc2.last_i_out + sc3.last_i_out == i_out
            assert sc1.last_t == sc2.last_t == sc3.last_t == t
            assert sc1.last_u[0] - 0.5 == u_out[0]
            assert sc2.last_u[0] - 0.5 == u_out[1]
            assert sc3.last_u[0] - 0.5 == u_out[2]

    def test_set_action(self, monkeypatch, converter, *_):
        t = np.random.randint(10) * converter._tau
        for _ in range(10):
            action = converter.action_space.sample()
            converter.set_action(action, t)
            sc0, sc1, sc2 = converter._subconverters
            assert sc0.last_action == action[0] * 0.5 + 0.5
            assert sc1.last_action == action[1] * 0.5 + 0.5
            assert sc2.last_action == action[2] * 0.5 + 0.5
            assert sc0.last_t == sc1.last_t == sc2.last_t == t

    @pytest.mark.parametrize('i_out', [[-1, -1, 0], [1, 1, -2], [0, 0, 1]])
    def test_i_sup(self, converter, i_out):
        sc1, sc2, sc3 = converter._subconverters
        for n in range(10):
            action = converter.action_space.sample()
            converter.set_action(action, converter._tau * n)
            i_sup = converter.i_sup(i_out)
            assert sc1.last_i_out + sc2.last_i_out + sc3.last_i_out == i_out
            assert i_sup == sc1.last_i_sup + sc2.last_i_sup + sc3.last_i_sup


# endregion
