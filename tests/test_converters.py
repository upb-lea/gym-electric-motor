import pytest
from gym_electric_motor.physical_systems.converters import *
from .conf import *
from gym_electric_motor.utils import make_module
from random import seed, uniform, randint
import numpy as np

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
g_disc_test_voltages = {'Disc-1QC': g_1qc_test_voltages,
                        'Disc-2QC': g_2qc_test_voltages,
                        'Disc-4QC': g_4qc_test_voltages}
g_disc_test_i_ins = {'Disc-1QC': g_i_ins_1qc,
                     'Disc-2QC': g_i_ins_2qc,
                     'Disc-4QC': g_i_ins_4qc}
g_disc_test_actions = {'Disc-1QC': g_actions_1qc,
                       'Disc-2QC': g_actions_2qc,
                       'Disc-4QC': g_actions_4qc}


# endregion


@pytest.fixture(scope='session')
def preparation():
    pass
    yield
    pass


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
                assert converter.voltages[0] <= u <= converter.voltages[1], "Voltage limits violated"
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
                         [('Disc-1QC', 2, g_actions_1qc, g_i_ins_1qc, g_1qc_test_voltages),
                          ('Disc-2QC', 3, g_actions_2qc, g_i_ins_2qc, g_2qc_test_voltages),
                          ('Disc-4QC', 4, g_actions_4qc, g_i_ins_4qc, g_4qc_test_voltages)])
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
    converter = make_module(PowerElectronicConverter, converter_type)
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
    converter = make_module(PowerElectronicConverter, converter_type, tau=tau,
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
    ('Disc-1QC', DiscOneQuadrantConverter),
    ('Disc-2QC', DiscTwoQuadrantConverter),
    ('Disc-4QC', DiscFourQuadrantConverter)
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
    converter_1 = make_module(PowerElectronicConverter, convert)
    converter_2 = convert_class()
    assert converter_1._tau == converter_2._tau
    # test with different parameters
    interlocking_time *= tau
    # initialize converters
    converter_1 = make_module(PowerElectronicConverter, convert, tau=tau,
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
def test_discrete_double_converter_initializations(tau, interlocking_time, dead_time):
    """
    tests different initializations of the converters
    :return:
    """
    # define all converter
    all_single_disc_converter = ['Disc-1QC', 'Disc-2QC', 'Disc-4QC']
    interlocking_time *= tau
    # chose every combination of single converters
    for conv_1 in all_single_disc_converter:
        for conv_2 in all_single_disc_converter:
            converter = make_module(
                PowerElectronicConverter, 'Disc-Double', tau=tau,
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
def test_discrete_double_power_electronic_converter(tau, interlocking_time, dead_time):
    """
    setup all combinations of single converters and test the convert function if no error is raised
    :return:
    """
    # define all converter
    all_single_disc_converter = ['Disc-1QC', 'Disc-2QC', 'Disc-4QC']
    interlocking_time *= tau

    for conv_1 in all_single_disc_converter:
        for conv_2 in all_single_disc_converter:
            converter = make_module(
                PowerElectronicConverter, 'Disc-Double', tau=tau,
                interlocking_time=interlocking_time, dead_time=dead_time,
                subconverters=[conv_1, conv_2]
            )
            comparable_converter_1 = make_module(PowerElectronicConverter, conv_1, tau=tau,
                                                 interlocking_time=interlocking_time, dead_time=dead_time)
            comparable_converter_2 = make_module(PowerElectronicConverter, conv_2, tau=tau,
                                                 interlocking_time=interlocking_time, dead_time=dead_time)
            action_space_n = converter.action_space.n
            assert converter.reset() == [0.0, 0.0]  # test if reset returns 0.0
            actions = [randint(0, action_space_n) for _ in range(100)]
            times = np.arange(100) * tau
            action_space_1_n = comparable_converter_1.action_space.n
            action_space_2_n = comparable_converter_2.action_space.n
            for action, t in zip(actions, times):
                time_steps = converter.set_action(action, t)
                time_steps_1 = comparable_converter_1.set_action(action % action_space_1_n, t)
                time_steps_2 = comparable_converter_2.set_action((action // action_space_1_n) % action_space_2_n, t)
                for time_step in time_steps_1 + time_steps_2:
                    assert time_step in time_steps
                for time_step in time_steps:
                    i_in_1 = uniform(-1, 1)
                    i_in_2 = uniform(-1, 1)
                    i_in = [i_in_1, i_in_2]
                    voltage = converter.convert(i_in, time_step)
                    voltage_1 = comparable_converter_1.convert([i_in_1], time_step)
                    voltage_2 = comparable_converter_2.convert([i_in_2], time_step)
                    converter.i_sup(i_in)
                    assert voltage[0] == voltage_1[0], "First converter is wrong"
                    assert voltage[1] == voltage_2[0], "Second converter is wrong"


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
    converter = make_module(PowerElectronicConverter, converter_type, tau=tau,
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
def test_continuous_double_power_electronic_converter(tau, interlocking_time, dead_time):
    """
    test functions of continuous double converter
    :return:
    """
    # define all converter
    all_single_cont_converter = ['Cont-1QC', 'Cont-2QC', 'Cont-4QC']
    interlocking_time *= tau
    times = g_times_cont * tau
    for conv_1 in all_single_cont_converter:
        for conv_2 in all_single_cont_converter:
            # setup converter with all possible combinations
            converter = make_module(PowerElectronicConverter, 'Cont-Double', tau=tau,
                                    interlocking_time=interlocking_time, dead_time=dead_time,
                                    subconverters=[conv_1, conv_2])
            assert all(converter.reset() == np.zeros(2))
            action_space = converter.action_space
            seed(123)
            actions = [uniform(action_space.low, action_space.high) for _ in range(0, 100)]
            continuous_double_converter_functions_testing(converter, times, interlocking_time, dead_time, actions,
                                                          [conv_1, conv_2])


def continuous_double_converter_functions_testing(converter, times, interlocking_time, dead_time, actions,
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
                i_in_0 = i_in
                i_in_1 = -i_in
                if converter_type[0] == 'Cont-1QC':
                    i_in_0 = abs(i_in_0)
                if converter_type[1] == 'Cont-1QC':
                    i_in_1 = abs(i_in_1)
                conversion = converter.convert([i_in_0, i_in_1], time_step)
                params = parameters + "  " + str(i_in) + "  " + str(time_step) + "  " + str(conversion)
                # test if the limits are hold
                assert (converter.action_space.low.tolist() <= conversion) and \
                       (converter.action_space.high.tolist() >= conversion), \
                    "Error, does not hold limits:" + str(params)
                # test if the converters work independent and correct
                voltage_0 = comparable_voltage(converter_type[0], action[0], i_in_0, tau, interlocking_time, dead_time,
                                               last_action[0])
                voltage_1 = comparable_voltage(converter_type[1], action[1], i_in_1, tau, interlocking_time, dead_time,
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

    tau = converter_parameter['tau']
    # test default initializations
    converter_default_init_1 = make_module(PowerElectronicConverter, 'Disc-B6C')
    converter_default_init_2 = DiscB6BridgeConverter()
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
    converter_init_1 = make_module(PowerElectronicConverter, 'Disc-B6C', **converter_parameter)
    converter_init_2 = DiscB6BridgeConverter(**converter_parameter)
    assert converter_init_1._tau == converter_parameter['tau']
    for subconverter in converter_init_1._subconverters:
        assert subconverter._tau == converter_parameter['tau']
        assert subconverter._interlocking_time == converter_parameter['interlocking_time']
        assert subconverter._dead_time == converter_parameter['dead_time']
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
    converter_default_init_1 = ContB6BridgeConverter()
    converter_default_init_2 = make_module(PowerElectronicConverter, 'Cont-B6C')
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

    converter_init_1 = ContB6BridgeConverter(**converter_parameter)
    converter_init_2 = make_module(PowerElectronicConverter, 'Cont-B6C', **converter_parameter)
    converters = [converter_init_1, converter_init_2]
    for converter in converters:
        # parameter testing
        assert converter._tau == converter_parameter['tau']
        assert converter._dead_time == converter_parameter['dead_time']
        assert converter._interlocking_time == converter_parameter['interlocking_time']
        assert all(converter.reset() == -0.5 * np.ones(3))
        assert converter.action_space.shape == (3,)
        assert all(converter.action_space.low == -1 * np.ones(3))
        assert all(converter.action_space.high == 1 * np.ones(3))
        for subconverter in converter._subconverters:
            assert subconverter._tau == converter_parameter['tau']
            assert subconverter._dead_time == converter_parameter['dead_time']
            assert subconverter._interlocking_time == converter_parameter['interlocking_time']
        # conversion testing
        for time, action, i_in, expected_voltage in zip(times, actions, i_ins, expected_voltages):
            i_in = np.array(i_in)
            time_step = converter.set_action(action.tolist(), time)
            voltages = converter.convert(i_in, time_step)
            for voltage, test_voltage in zip(voltages, expected_voltage):
                assert abs(voltage - test_voltage) < 1E-9

# endregion
