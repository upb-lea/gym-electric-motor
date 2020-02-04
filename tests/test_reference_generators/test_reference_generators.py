import gym_electric_motor.envs
from gym_electric_motor.utils import make_module
from numpy.random import seed
import numpy.random as rd
import pytest
from gym_electric_motor.reference_generators.multi_reference_generator import MultiReferenceGenerator
from gym_electric_motor.reference_generators.sawtooth_reference_generator import SawtoothReferenceGenerator
from gym_electric_motor.reference_generators.sinusoidal_reference_generator import SinusoidalReferenceGenerator
from gym_electric_motor.reference_generators.step_reference_generator import StepReferenceGenerator
from gym_electric_motor.reference_generators.triangle_reference_generator import TriangularReferenceGenerator
from gym_electric_motor.reference_generators.wiener_process_reference_generator import WienerProcessReferenceGenerator
from gym_electric_motor.reference_generators.subepisoded_reference_generator import SubepisodedReferenceGenerator
from gym_electric_motor.reference_generators.zero_reference_generator import ZeroReferenceGenerator
import gym_electric_motor.reference_generators.multi_reference_generator as mrg
import gym_electric_motor.reference_generators.subepisoded_reference_generator as srg
import gym_electric_motor.reference_generators.sawtooth_reference_generator as sawrg
import gym_electric_motor.reference_generators.wiener_process_reference_generator as wrg
from gym_electric_motor.core import ReferenceGenerator
from ..testing_utils import *
import numpy as np


# region first version

# region expected references
g_wiener_process_expect_1 = [0.1315388322520193, 0.14048739807899743, 0.1, 0.1, 0.1522230086844815, 0.1, 0.1,
                             0.14003241950101647, 0.11262368138853361, 0.1]
g_wiener_process_expect_2 = [-0.002791821788105553, 0.0061567440388726035, -0.04147647719565641, -0.05977342370533539,
                             -0.0075504150208539, -0.08428875061861613, -0.09785215886257594, -0.05781973936155946,
                             -0.08522847747404232, -0.10669674258476337]
g_wiener_process_expect_3 = [0.03153883225201929, 0.04048739807899745, 0.0, 0.0, 0.052223008684481494, 0.0, 0.0,
                             0.04003241950101648, 0.012623681388533623, 0.0]

g_step_expect_1 = [0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175]
g_step_expect_2 = [-0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3]
g_step_expect_3 = [0.14999999999999994, 0.14999999999999994, 0.14999999999999994, 0.14999999999999994,
                   0.14999999999999994, 0.14999999999999994, 0.14999999999999994, 0.14999999999999994,
                   0.14999999999999994, 0.14999999999999994]

g_sinus_expect_1 = [0.1892341087797782, 0.16826374935675312, 0.15165796462287975, 0.14105516621316472,
                    0.13750148043578975, 0.1413475321012121, 0.15221385008545588, 0.1690283078916173,
                    0.19013190513233447, 0.21344245299125145]
g_sinus_expect_2 = [-0.09306356488088728, -0.1769450025729876, -0.24336814150848105, -0.2857793351473412,
                    -0.2999940782568411, -0.2846098715951517, -0.24114459965817656, -0.1738867684335309,
                    -0.08947237947066215, 0.0037698119650057403]
g_sinus_expect_3 = [0.17846821755955633, 0.13652749871350617, 0.10331592924575946, 0.08211033242632937,
                    0.07500296087157943, 0.08269506420242412, 0.1044277001709117, 0.13805661578323453,
                    0.1802638102646689, 0.22688490598250285]

g_sinus_epis_expect_1 = [0.1892341087797782, 0.16826374935675312, 0.15165796462287975, 0.14105516621316472,
                         0.13750148043578975, 0.1413475321012121, 0.15221385008545588, 0.1690283078916173,
                         0.19013190513233447, 0.21344245299125145]
g_sinus_epis_expect_3 = [0.17846821755955633, 0.13652749871350617, 0.10331592924575946, 0.08211033242632937,
                         0.07500296087157943, 0.08269506420242412, 0.1044277001709117, 0.13805661578323453,
                         0.1802638102646689, 0.22688490598250285]
g_sinus_epis_expect_2 = [-0.09306356488088728, -0.1769450025729876, -0.24336814150848105, -0.2857793351473412,
                         -0.2999940782568411, -0.2846098715951517, -0.24114459965817656, -0.1738867684335309,
                         -0.08947237947066215, 0.0037698119650057403]

g_multi_expected_3 = [0.0, 0.054972090373755, 0.06782529748676631, 0.07801118572791686, 0.0763820523476228,
                      0.0699246508609987,
                      0.13251714416762556, 0.08131037995987651, 0.04608373206022021, 0.03193441412870947]
g_multi_expected_2 = [-0.05368087336940529, 0.0012912170043497123, 0.014144424117361032, 0.02433031235851158,
                      0.02270117897821752, 0.016243777491593423, 0.07883627079822027, 0.027629506590471226,
                      -0.007597141309185075, -0.021746459240695817]
g_multi_expected_1 = [0.1, 0.154972090373755, 0.16782529748676633, 0.17801118572791688, 0.17638205234762283,
                      0.16992465086099873, 0.2325171441676256, 0.18131037995987653, 0.14608373206022024,
                      0.1319344141287095]

g_sawtooth_expected_1 = [0.22003000000000003, 0.22756000000000004, 0.23509000000000002, 0.24262000000000003, 0.25015,
                         0.25768, 0.26521000000000006, 0.27274000000000004, 0.28027, 0.1378]

g_sawtooth_expected_2 = [0.03012000000000001, 0.06024000000000002, 0.09035999999999997, 0.12048000000000005,
                         0.15059999999999993, 0.18072000000000002, 0.21084000000000003, 0.24096000000000004,
                         0.27108000000000004, -0.2988]

g_sawtooth_expected_3 = [0.24006, 0.25512, 0.27018, 0.28524, 0.30029999999999996, 0.31536, 0.33042, 0.34548,
                         0.36053999999999997, 0.07559999999999997]

g_sawtooth_epis_expected_3 = [0.24006, 0.25512, 0.27018, 0.28524, 0.30029999999999996, 0.31536, 0.33042, 0.34548,
                              0.36053999999999997, 0.07559999999999997]

g_sawtooth_epis_expected_1 = [0.22003000000000003, 0.22756000000000004, 0.23509000000000002, 0.24262000000000003,
                              0.25015, 0.25768, 0.26521000000000006, 0.27274000000000004, 0.28027, 0.1378]

g_sawtooth_epis_expected_2 = [0.03012000000000001, 0.06024000000000002, 0.09035999999999997, 0.12048000000000005,
                              0.15059999999999993, 0.18072000000000002, 0.21084000000000003, 0.24096000000000004,
                              0.27108000000000004, -0.2988]

g_triangle_expect_1 = [0.27244, 0.25738000000000005, 0.24232000000000004, 0.22726000000000002, 0.21220000000000003,
                       0.19714000000000004, 0.18208000000000002, 0.16702, 0.15195999999999998, 0.1381]
g_triangle_expect_2 = [0.23975999999999992, 0.17951999999999999, 0.11928, 0.059039999999999954, -0.0011999999999999253,
                       -0.061439999999999974, -0.12168000000000004, -0.18192000000000008, -0.24216000000000013, -0.2976]
g_triangle_expect_3 = [0.34487999999999996, 0.31476, 0.28464, 0.25451999999999997, 0.22440000000000002,
                       0.19427999999999998, 0.16415999999999997, 0.13403999999999994, 0.10391999999999992,
                       0.07619999999999999]

g_triangle_epis_expect_1 = [0.27244, 0.25738000000000005, 0.24232000000000004, 0.22726000000000002, 0.21220000000000003,
                            0.19714000000000004, 0.18208000000000002, 0.16702, 0.15195999999999998, 0.1381]
g_triangle_epis_expect_2 = [0.23975999999999992, 0.17951999999999999, 0.11928, 0.059039999999999954,
                            -0.0011999999999999253, -0.061439999999999974, -0.12168000000000004, -0.18192000000000008,
                            -0.24216000000000013, -0.2976]
g_triangle_epis_expect_3 = [0.34487999999999996, 0.31476, 0.28464, 0.25451999999999997, 0.22440000000000002,
                            0.19427999999999998, 0.16415999999999997, 0.13403999999999994, 0.10391999999999992,
                            0.07619999999999999]


# endregion

# region used functions


def setup_sub_generator(generator, **kwargs):
    return make_module(ReferenceGenerator, generator, **kwargs)


def set_limit_margins(limit_margin):
    """
    set the limit margins for testing
    :param limit_margin: limit margin value used for the initialization
    :return:
    """
    if type(limit_margin) is tuple:
        limit_margin_low = limit_margin[0]
        limit_margin_high = limit_margin[1]
    else:
        limit_margin_high = limit_margin
        limit_margin_low = -limit_margin
    return limit_margin_low, limit_margin_high


def reset_testing(reference_generator, length_state):
    """
    tests the reset function
    :param reference_generator: instantiated reference generator
    :param length_state: number of states of the physical system
    :return:
    """
    ref, obs, temp = reference_generator.reset()
    assert temp is None
    assert obs.size == 1
    assert all(ref == np.zeros(length_state))


def get_reference_testing(reference_generator, index, limit_margin_low, limit_margin_high,
                          expected_reference=None):
    """
    tests the get_reference and get_reference_observation function
    :param reference_generator: instantiated reference generator
    :param index: index of referenced state
    :param limit_margin_low: lower limit margin
    :param limit_margin_high: upper limit margin
    :param expected_reference: list of expected reference
    :return:
    """
    list_ref = []
    for k in range(10):
        # test get reference observation and set next reference
        assert len(reference_generator.get_reference_observation(None)) == 1
        ref = reference_generator.get_reference(None)
        for ind, state in enumerate(reference_generator.referenced_states):
            if ind is not index:
                assert ref[ind] == 0, "Error in this state: " + str(state)
            else:
                assert limit_margin_low <= ref[ind] <= limit_margin_high
        list_ref.append(ref[index])
    assert list_ref == expected_reference, " Reference is not equal the expected one"


def state_space_testing(reference_generator, physical_system, limit_margin_low, limit_margin_high, state_names, index):
    """
    tests the state space of the instantiated reference generators
    :param reference_generator: instantiated reference generator
    :param physical_system: instantiated physical system
    :param limit_margin_low: lower limit margin
    :param limit_margin_high: upper limit margin
    :param state_names: state names of physical system (dict)
    :param index: index of referenced state
    :return:
    """
    for i in range(len(reference_generator.referenced_states)):
        if i == index:
            assert reference_generator.referenced_states[i], "Error state: " + str(state_names[i])
        else:
            assert not reference_generator.referenced_states[i], "Error state: " + str(state_names[i])
    # test the state space
    low = physical_system.state_space.low[index]
    high = physical_system.state_space.high[index]
    test_reference_space = Box(low, high, (1,))
    ref_space = reference_generator.reference_space
    assert ref_space.shape == test_reference_space.shape
    assert all(ref_space.low == max(test_reference_space.low, limit_margin_low))
    assert all(ref_space.high == min(limit_margin_high, test_reference_space.high))


def default_initialization_testing(reference_generator, physical_system):
    reference_generator.set_modules(physical_system)
    reference_generator.reset()
    reference_generator.get_reference_observation()
    reference_generator.get_reference()
    reference_generator.close()


def monkey_rand_function():
    return 0.5


def monkey_rand_triangular(min, mean, max):
    return 0.3


# endregion


def test_zero_reference_generator():
    reference_generator = ZeroReferenceGenerator()
    physical_system = setup_physical_system('DcPermEx', 'Disc-1QC')
    length_state = len(physical_system.state_names)
    reference_generator.set_modules(physical_system)
    # test reset function
    ref, obs, temp = reference_generator.reset()
    assert temp is None
    assert obs.size == 0
    assert all(ref == np.zeros(length_state))
    # check referenced states
    ref_state = np.zeros(length_state)
    ref_state[:] = False
    assert all(reference_generator.referenced_states == ref_state)
    # test get reference function
    assert all(np.zeros(length_state) == reference_generator.get_reference(np.zeros_like(physical_system.state_names)))
    # test get reference observation function
    assert reference_generator.get_reference_observation(np.ones(length_state)).size == 0
    reference_generator.close()


@pytest.mark.parametrize("motor_type", ['DcPermEx', 'DcSeries', 'DcExtEx', 'DcShunt'])
@pytest.mark.parametrize("conv", ['Disc-1QC', 'Disc-2QC', 'Disc-4QC', 'Cont-1QC', 'Cont-2QC', 'Cont-4QC'])
@pytest.mark.parametrize("limit_margin", [0.3, (0.1, 0.4), ])
@pytest.mark.parametrize("episode_length", [10, (5, 20)])
def test_wiener_process_reference_generator(monkeypatch, motor_type, conv, limit_margin, episode_length):
    monkeypatch.setattr(rd, "rand", monkey_rand_function)
    # setup physical system
    physical_system = setup_physical_system(motor_type, conv)
    state_names = physical_system.state_names
    length_state = len(state_names)
    sigma_range = (1e-3, 1e0)
    # set parameter
    limit_margin_low, limit_margin_high = set_limit_margins(limit_margin)
    reference_generator = WienerProcessReferenceGenerator()
    default_initialization_testing(reference_generator, physical_system)
    # initialize Wiener Process Reference Generator
    for index, reference_state in enumerate(state_names):
        seed(123)
        reference_generator = WienerProcessReferenceGenerator(sigma_range=sigma_range,
                                                              reference_state=reference_state,
                                                              episode_lengths=episode_length,
                                                              limit_margin=limit_margin)
        # set the physical system
        reference_generator.set_modules(physical_system)
        assert reference_generator._episode_len_range == episode_length, "Wrong episode length"
        # test reset function
        reset_testing(reference_generator, length_state)
        # test if referenced state is correct
        state_space_testing(reference_generator, physical_system, limit_margin_low, limit_margin_high, state_names,
                            index)
        # test the reference generator
        if type(limit_margin) in [float, int] and physical_system.state_space.low[index] == -1:
            expected_reference = g_wiener_process_expect_2
        elif type(limit_margin) in [float, int] and physical_system.state_space.low[index] == 0:
            expected_reference = g_wiener_process_expect_3
        else:
            expected_reference = g_wiener_process_expect_1

        get_reference_testing(reference_generator, index, limit_margin_low,
                              limit_margin_high, expected_reference)
        reference_generator.close()


@pytest.mark.parametrize("motor_type", ['DcPermEx', 'DcSeries', 'DcExtEx', 'DcShunt'])
@pytest.mark.parametrize("conv", ['Disc-1QC', 'Disc-2QC', 'Disc-4QC', 'Cont-1QC', 'Cont-2QC', 'Cont-4QC'])
@pytest.mark.parametrize("limit_margin", [0.6, (0.1, 0.4), ])
@pytest.mark.parametrize("episode_length", [10, (5, 20)])
def test_step_reference_generator(motor_type, conv, limit_margin, episode_length, monkeypatch):
    monkeypatch.setattr(rd, "rand", monkey_rand_function)
    monkeypatch.setattr(rd, "triangular", monkey_rand_triangular)
    # setup physical system
    physical_system = setup_physical_system(motor_type, conv)
    state_names = physical_system.state_names
    length_state = len(state_names)

    # amplitude_range=(0, 1), frequency_range=(1, 10), offset_range=(-1, 1)
    amplitude_range = (-1, 2)
    frequency_range = (2, 9)
    offset_range = (-0.8, 0.7)
    reference_generator = StepReferenceGenerator()
    default_initialization_testing(reference_generator, physical_system)
    # general reference parameter
    seed(123)
    for reference_state in state_names:
        index = state_names.index(reference_state)
        limit_range = 0.5
        limit_margin_low, limit_margin_high = set_limit_margins(limit_margin)
        reference_generator = StepReferenceGenerator(amplitude_range=amplitude_range,
                                                     frequncy_range=frequency_range,
                                                     offset_range=offset_range,
                                                     episode_lengths=episode_length,
                                                     limit_range=limit_range,
                                                     reference_state=reference_state,
                                                     limit_margin=limit_margin
                                                     )
        # set the physical system
        reference_generator.set_modules(physical_system)
        # test reset function
        reset_testing(reference_generator, length_state)
        assert reference_generator._episode_len_range == episode_length, "Wrong episode length"
        # test if referenced state is correct
        state_space_testing(reference_generator, physical_system, limit_margin_low, limit_margin_high, state_names,
                            index)
        # test the reference generator
        if type(limit_margin) in [float, int] and physical_system.state_space.low[index] == -1:
            expected_reference = g_step_expect_2
        elif type(limit_margin) in [float, int] and physical_system.state_space.low[index] == 0:
            expected_reference = g_step_expect_3
        else:
            expected_reference = g_step_expect_1
        get_reference_testing(reference_generator, index, limit_margin_low, limit_margin_high,
                              expected_reference)
    reference_generator.close()


@pytest.mark.parametrize("motor_type", ['DcPermEx', 'DcSeries', 'DcExtEx', 'DcShunt'])
@pytest.mark.parametrize("conv", ['Disc-1QC', 'Disc-2QC', 'Disc-4QC', 'Cont-1QC', 'Cont-2QC', 'Cont-4QC'])
@pytest.mark.parametrize("limit_margin", [0.6, (0.1, 0.4), ])
@pytest.mark.parametrize("episode_length", [20, (15, 20)])
def test_sinusoidal_reference_generator(motor_type, conv, limit_margin, episode_length, monkeypatch):
    monkeypatch.setattr(rd, "rand", monkey_rand_function)
    # setup physical system
    physical_system = setup_physical_system(motor_type, conv)
    state_names = physical_system.state_names
    length_state = len(state_names)

    reference_generator = SinusoidalReferenceGenerator()
    default_initialization_testing(reference_generator, physical_system)
    amplitude_range = (-1, 2)
    frequency_range = (2, 500)
    offset_range = (-0.8, 0.7)
    # general reference parameter
    seed(123)
    for reference_state in state_names:
        index = state_names.index(reference_state)
        limit_margin_low, limit_margin_high = set_limit_margins(limit_margin)
        reference_generator = SinusoidalReferenceGenerator(amplitude_range=amplitude_range,
                                                           frequency_range=frequency_range,
                                                           offset_range=offset_range,
                                                           reference_state=reference_state,
                                                           episode_lengths=episode_length,
                                                           limit_margin=limit_margin)
        # set the physical system
        reference_generator.set_modules(physical_system)
        assert reference_generator._episode_len_range == episode_length, "Wrong episode length"
        # test reset function
        reset_testing(reference_generator, length_state)
        # test if referenced state is correct
        state_space_testing(reference_generator, physical_system, limit_margin_low, limit_margin_high, state_names,
                            index)
        # test the reference generator
        if type(episode_length) is int:
            if type(limit_margin) in [float, int] and physical_system.state_space.low[index] == -1:
                expected_reference = g_sinus_expect_2
            elif type(limit_margin) in [float, int] and physical_system.state_space.low[index] == 0:
                expected_reference = g_sinus_expect_3
            else:
                expected_reference = g_sinus_expect_1
        else:
            if type(limit_margin) in [float, int] and physical_system.state_space.low[index] == -1:
                expected_reference = g_sinus_epis_expect_2
            elif type(limit_margin) in [float, int] and physical_system.state_space.low[index] == 0:
                expected_reference = g_sinus_epis_expect_3
            else:
                expected_reference = g_sinus_epis_expect_1
        get_reference_testing(reference_generator, index, limit_margin_low,
                              limit_margin_high, expected_reference)
    reference_generator.close()


@pytest.mark.parametrize("motor_type", ['DcPermEx', 'DcSeries', 'DcExtEx', 'DcShunt'])
@pytest.mark.parametrize("conv", ['Disc-1QC', 'Disc-2QC', 'Disc-4QC', 'Cont-1QC', 'Cont-2QC', 'Cont-4QC'])
@pytest.mark.parametrize("limit_margin", [0.6, (0.1, 0.4), ])
@pytest.mark.parametrize("episode_length", [10, (5, 20)])
def test_multi_reference_generator(motor_type, conv, limit_margin, episode_length, monkeypatch):
    monkeypatch.setattr(rd, "rand", monkey_rand_function)
    monkeypatch.setattr(rd, "triangular", monkey_rand_triangular)
    # setup physical system
    physical_system = setup_physical_system(motor_type, conv)
    state_names = physical_system.state_names
    length_state = len(state_names)
    #
    # SinusoidalReferenceGenerator  SinusReference
    # StepReferenceGenerator  StepReference
    # WienerProcessReferenceGenerator  WienerProcessReference

    amplitude_range = (-1, 1)
    frequency_range = (2, 9)
    offset_range = (-0.8, 0.7)
    probabilities = [0.2, 0.5, 0.3]
    sigma_range = (1e-3, 1e0)
    super_episode_length = 35
    limit_margin_low, limit_margin_high = set_limit_margins(limit_margin)
    for reference_state in state_names:
        index = state_names.index(reference_state)
        sub_generators_string = ['SinusReference', 'StepReference', 'WienerProcessReference']
        sub_generators_class = [SinusoidalReferenceGenerator, StepReferenceGenerator, WienerProcessReferenceGenerator]
        sinus_generator = setup_sub_generator('SinusReference',
                                              amplitude_range=amplitude_range,
                                              offset_range=offset_range,
                                              episode_lengths=episode_length,
                                              frequency_range=frequency_range,
                                              limit_margin=limit_margin,
                                              reference_state=reference_state)
        step_generator = setup_sub_generator('StepReference', amplitude_range=amplitude_range,
                                             offset_range=offset_range,
                                             episode_lengths=episode_length,
                                             frequency_range=frequency_range,
                                             limit_margin=limit_margin,
                                             reference_state=reference_state)
        wiener_generator = setup_sub_generator('WienerProcessReference',
                                               sigma_range=sigma_range,
                                               amplitude_range=amplitude_range,
                                               offset_range=offset_range,
                                               episode_lengths=episode_length,
                                               frequency_range=frequency_range,
                                               limit_margin=limit_margin,
                                               reference_state=reference_state)
        sub_generators_instance = [sinus_generator, step_generator, wiener_generator]

        for sub_generators in [sub_generators_string, sub_generators_class, sub_generators_instance]:
            seed(123)
            reference_generator = MultiReferenceGenerator(sub_generators=sub_generators,
                                                          probabilities=probabilities,
                                                          super_episode_length=super_episode_length,
                                                          reference_state=reference_state,
                                                          episode_lengths=episode_length,
                                                          limit_margin=limit_margin,
                                                          amplitude_range=amplitude_range,
                                                          frequency_range=frequency_range,
                                                          offset_range=offset_range,
                                                          sigma_range=sigma_range
                                                          )
            for ref_generator in reference_generator._sub_generators:
                assert limit_margin == ref_generator._limit_margin
                assert ref_generator._episode_len_range == episode_length, "Wrong episode length"
                if type(ref_generator) == SinusoidalReferenceGenerator:
                    assert ref_generator._amplitude_range == amplitude_range
                    assert ref_generator._offset_range == offset_range
                    assert ref_generator._frequency_range == frequency_range

                elif type(ref_generator) == StepReferenceGenerator:
                    assert ref_generator._amplitude_range == amplitude_range
                    assert ref_generator._offset_range == offset_range
                    assert ref_generator._frequency_range == frequency_range
                elif type(ref_generator) == WienerProcessReferenceGenerator:
                    assert ref_generator._sigma_range == sigma_range
                else:
                    print("No valid reference generator")

            # set the physical system
            reference_generator.set_modules(physical_system)
            # test reset function
            reset_testing(reference_generator, length_state)
            # test if referenced state is correct
            state_space_testing(reference_generator, physical_system, limit_margin_low, limit_margin_high, state_names,
                                index)
            # test the reference generator
            if type(limit_margin) in [float, int] and physical_system.state_space.low[index] == -1:
                expected_reference = g_multi_expected_2
            elif type(limit_margin) in [float, int] and physical_system.state_space.low[index] == 0:
                expected_reference = g_multi_expected_3
            else:
                expected_reference = g_multi_expected_1
            get_reference_testing(reference_generator, index, limit_margin_low,
                                  limit_margin_high, expected_reference)
            reference_generator.close()


@pytest.mark.parametrize("motor_type", ['DcPermEx', 'DcSeries', 'DcExtEx', 'DcShunt'])
@pytest.mark.parametrize("conv", ['Disc-1QC', 'Disc-2QC', 'Disc-4QC', 'Cont-1QC', 'Cont-2QC', 'Cont-4QC'])
@pytest.mark.parametrize("limit_margin", [0.6, (0.1, 0.4), ])
@pytest.mark.parametrize("episode_length", [20, (15, 20)])
def test_sawtooth_reference_generator(motor_type, conv, limit_margin, episode_length, monkeypatch):
    monkeypatch.setattr(rd, "rand", monkey_rand_function)
    # setup physical system
    physical_system = setup_physical_system(motor_type, conv)
    state_names = physical_system.state_names
    length_state = len(state_names)

    reference_generator = SawtoothReferenceGenerator()
    default_initialization_testing(reference_generator, physical_system)
    amplitude_range = (-1, 2)
    frequency_range = (2, 500)
    offset_range = (-0.8, 0.7)
    seed(123)
    # general reference parameter
    for reference_state in state_names:
        index = state_names.index(reference_state)
        limit_margin_low, limit_margin_high = set_limit_margins(limit_margin)
        reference_generator = SawtoothReferenceGenerator(amplitude_range=amplitude_range,
                                                         frequency_range=frequency_range,
                                                         offset_range=offset_range,
                                                         reference_state=reference_state,
                                                         episode_lengths=episode_length,
                                                         limit_margin=limit_margin)
        # set the physical system
        reference_generator.set_modules(physical_system)
        assert reference_generator._episode_len_range == episode_length, "Wrong episode length"
        # test reset function
        reset_testing(reference_generator, length_state)
        # test if referenced state is correct
        state_space_testing(reference_generator, physical_system, limit_margin_low, limit_margin_high, state_names,
                            index)
        # test the reference generator

        if type(episode_length) is int:
            if type(limit_margin) in [float, int] and physical_system.state_space.low[index] == -1:
                expected_reference = g_sawtooth_expected_2
            elif type(limit_margin) in [float, int] and physical_system.state_space.low[index] == 0:
                expected_reference = g_sawtooth_expected_3
            else:
                expected_reference = g_sawtooth_expected_1
        else:
            if type(limit_margin) in [float, int] and physical_system.state_space.low[index] == -1:
                expected_reference = g_sawtooth_epis_expected_2
            elif type(limit_margin) in [float, int] and physical_system.state_space.low[index] == 0:
                expected_reference = g_sawtooth_epis_expected_3
            else:
                expected_reference = g_sawtooth_epis_expected_1

        get_reference_testing(reference_generator, index, limit_margin_low,
                              limit_margin_high, expected_reference)
    reference_generator.close()


@pytest.mark.parametrize("motor_type", ['DcPermEx', 'DcSeries', 'DcExtEx', 'DcShunt'])
@pytest.mark.parametrize("conv", ['Disc-1QC', 'Disc-2QC', 'Disc-4QC', 'Cont-1QC', 'Cont-2QC', 'Cont-4QC'])
@pytest.mark.parametrize("limit_margin", [0.6, (0.1, 0.4), ])
@pytest.mark.parametrize("episode_length", [20, (15, 20)])
def test_triangular_reference_generator(motor_type, conv, limit_margin, episode_length, monkeypatch):
    monkeypatch.setattr(rd, "rand", monkey_rand_function)
    # setup physical system
    physical_system = setup_physical_system(motor_type, conv)
    state_names = physical_system.state_names
    length_state = len(state_names)

    reference_generator = TriangularReferenceGenerator()
    default_initialization_testing(reference_generator, physical_system)
    amplitude_range = (-1, 2)
    frequency_range = (2, 500)
    offset_range = (-0.8, 0.7)
    seed(123)
    # general reference parameter
    for reference_state in state_names:
        index = state_names.index(reference_state)
        limit_margin_low, limit_margin_high = set_limit_margins(limit_margin)
        reference_generator = TriangularReferenceGenerator(amplitude_range=amplitude_range,
                                                           frequency_range=frequency_range,
                                                           offset_range=offset_range,
                                                           reference_state=reference_state,
                                                           episode_lengths=episode_length,
                                                           limit_margin=limit_margin)
        # set the physical system
        reference_generator.set_modules(physical_system)
        assert reference_generator._episode_len_range == episode_length, "Wrong episode length"
        # test reset function
        reset_testing(reference_generator, length_state)
        # test if referenced state is correct
        state_space_testing(reference_generator, physical_system, limit_margin_low, limit_margin_high, state_names,
                            index)
        # test the reference generator
        if type(episode_length) is int:
            if type(limit_margin) in [float, int] and physical_system.state_space.low[index] == -1:
                expected_reference = g_triangle_expect_2
            elif type(limit_margin) in [float, int] and physical_system.state_space.low[index] == 0:
                expected_reference = g_triangle_expect_3
            else:
                expected_reference = g_triangle_expect_1
        else:
            if type(limit_margin) in [float, int] and physical_system.state_space.low[index] == -1:
                expected_reference = g_triangle_epis_expect_2
            elif type(limit_margin) in [float, int] and physical_system.state_space.low[index] == 0:
                expected_reference = g_triangle_epis_expect_3
            else:
                expected_reference = g_triangle_epis_expect_1

        get_reference_testing(reference_generator, index, limit_margin_low,
                              limit_margin_high, expected_reference)
    reference_generator.close()


# endregion


# region second version


class TestMultiReferenceGenerator:
    """
    class for testing the multi reference generator
    """
    _reference_generator = []
    _physical_system = None
    _sub_generator = []
    # pre defined test values and expected results
    _kwargs = {'test': 42}
    _reference = 0.8
    _reference_observation = np.array([0.5, 0.6, 0.8, 0.15])
    _trajectory = np.ones((4, 15))
    _initial_state = np.array([0.1, 0.25, 0.6, 0.78])
    _initial_reference = 0.8

    # counter
    _monkey_super_init_counter = 0
    _monkey_instantiate_counter = 0
    _monkey_super_set_modules_counter = 0
    _monkey_dummy_set_modules_counter = 0
    _monkey_reset_reference_counter = 0
    _monkey_dummy_reset_counter = 0

    @pytest.fixture(scope='function')
    def setup(self):
        """
        fixture to reset the counter and _reference_generator
        :return:
        """
        self._reference_generator = []

    def monkey_super_init(self):
        """
        mock function for super().__init__()
        :return:
        """
        self._monkey_super_init_counter += 1

    def monkey_instantiate(self, superclass, instance, **kwargs):
        """
        mock function for utils.instantiate
        Function tests if the instance and superclass are as expected.
        A dummy reference generator is instantiated.
        :param superclass:
        :param instance:
        :param kwargs:
        :return: DummyReferenceGenerator
        """
        assert superclass == ReferenceGenerator, 'superclass is not ReferenceGenerator as expected'
        assert instance == self._sub_generator[self._monkey_instantiate_counter], \
            'Instance is not the expected reference generator'
        dummy = DummyReferenceGenerator()
        self._reference_generator.append(dummy)
        self._monkey_instantiate_counter += 1
        return dummy

    def monkey_super_set_modules(self, physical_system):
        """
        mock function for super().set_modules()
        :param physical_system:
        :return:
        """
        self._monkey_super_set_modules_counter += 1
        assert physical_system == self._physical_system, 'physical system is not the expected instance'

    def monkey_dummy_set_modules(self, physical_system):
        """
        mock function for set_modules()
        :param physical_system:
        :return:
        """
        self._monkey_dummy_set_modules_counter += 1
        assert self._physical_system == physical_system, 'physical system is not the expected instance'

    def monkey_reset_reference(self):
        """
        mock function for reset_reference()
        :return:
        """
        self._monkey_reset_reference_counter += 1

    def monkey_dummy_reset(self, initial_state, initial_reference):
        """
        mock function for DummyReferenceGenerator.reset()
        :param initial_state:
        :param initial_reference:
        :return:
        """
        if type(initial_state == self._initial_state) is bool:
            assert initial_state == self._initial_state, 'passed initial state is not the expected one'
        else:
            assert all(initial_state == self._initial_state), 'passed initial state is not the expected one'
        assert initial_reference == self._initial_reference, 'passed initial reference is not the expected one'
        self._monkey_dummy_reset_counter += 1
        return self._reference, self._reference_observation, self._trajectory

    def monkey_dummy_get_reference(self, state, **kwargs):
        """
        mock function for DummyReferenceGenerator.get_reference()
        :param state:
        :param kwargs:
        :return:
        """
        assert all(state == self._initial_state), 'passed state is not the expected one'
        assert self._kwargs == kwargs, 'Different additional arguments. Keep in mind None and {}.'
        return self._reference

    def monkey_dummy_get_reference_observation(self, state, **kwargs):
        """
        mock function for DummyReferenceGenerator.get_reference_observation()
        :param state:
        :param kwargs:
        :return:
        """
        assert all(state == self._initial_state), 'passed state is not the expected one'
        assert self._kwargs == kwargs, 'Different additional arguments. Keep in mind None and {}.'
        return self._reference_observation

    @pytest.mark.parametrize("sub_generator",
                             [['SinusReference'], ['WienerProcessReference'], ['StepReference'], ['TriangleReference'],
                              ['SawtoothReference'],
                              ['SinusReference', 'WienerProcessReference', 'StepReference', 'TriangleReference',
                               'SawtoothReference'],
                              ['SinusReference', 'WienerProcessReference'],
                              ['StepReference', 'TriangleReference', 'SawtoothReference']])
    @pytest.mark.parametrize("sub_args", [None])
    @pytest.mark.parametrize("p", [None, [0.1, 0.2, 0.3, 0.2, 0.1]])
    @pytest.mark.parametrize("super_episode_length, expected_sel",
                             [((200, 500), (200, 500)), (100, (100, 101)), (500, (500, 501))])
    def test_init(self, monkeypatch, setup, sub_generator, sub_args, p, super_episode_length, expected_sel):
        """
        test function for the initialization of a multi reference generator with different combinations of reference
        generators
        :param monkeypatch:
        :param setup: fixture to reset the counters and _reference_generators
        :param sub_generator: list of sub generators
        :param sub_args: additional arguments for sub generators
        :param p: probabilities for the sub generators
        :param super_episode_length: range of teh episode length of the multi reference generator
        :param expected_sel: expected multi reference generator episode length
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(ReferenceGenerator, '__init__', self.monkey_super_init)
        monkeypatch.setattr(mrg, 'instantiate', self.monkey_instantiate)
        self._sub_generator = sub_generator
        self._kwargs = sub_args
        # call function to test
        test_object = MultiReferenceGenerator(sub_generator, sub_args=sub_args, p=p,
                                              super_episode_length=super_episode_length)
        # verify the expected results
        assert len(test_object._sub_generators) == len(sub_generator), 'unexpected number of sub generators'
        assert test_object._current_episode_length == 0, 'The current episode length is not 0.'
        assert test_object._super_episode_length == expected_sel, 'super episode length is not as expected'
        assert test_object._current_ref_generator == self._reference_generator[0], \
            'current reference generator is not the first in reference_generator'
        assert test_object._sub_generators == self._reference_generator, 'Other sub generators than expected'

    def test_set_modules(self, monkeypatch, setup):
        """
        test set_modules()
        :param monkeypatch:
        :param setup: fixture to reset the counters and _reference_generators
        :return:
        """
        # setup test scenario
        sub_generator = ['SinusReference', 'WienerProcessReference']
        reference_states = [1, 0, 0, 0, 0, 0, 0]
        monkeypatch.setattr(ReferenceGenerator, '__init__', self.monkey_super_init)
        monkeypatch.setattr(ReferenceGenerator, 'set_modules', self.monkey_super_set_modules)
        monkeypatch.setattr(DummyReferenceGenerator, 'set_modules', self.monkey_dummy_set_modules)
        monkeypatch.setattr(DummyReferenceGenerator, '_referenced_states', reference_states)
        self._sub_generator = sub_generator
        monkeypatch.setattr(mrg, 'instantiate', self.monkey_instantiate)
        test_object = MultiReferenceGenerator(sub_generator)
        self._physical_system = DummyPhysicalSystem()
        # call function to test
        test_object.set_modules(self._physical_system)
        # verify the expected results
        assert self._monkey_dummy_set_modules_counter == 2, 'dummy set_modules() not called twice'
        assert self._monkey_super_set_modules_counter == 1, 'super().set_modules() not called once'
        assert test_object.reference_space.low == 0, 'Lower limit of the reference space is not 0'
        assert test_object.reference_space.high == 1, 'Upper limit of the reference space is not 1'
        assert test_object._referenced_states == reference_states, 'referenced states are not the expected ones'

    @pytest.mark.parametrize("initial_state", [None, [0.8, 0.6, 0.4, 0.7]])
    @pytest.mark.parametrize("initial_reference", [None, 0.42])
    def test_reset(self, monkeypatch, setup, initial_state, initial_reference):
        """
        test reset()
        :param monkeypatch:
        :param setup: fixture to reset the counters and _reference_generators
        :param initial_state: tested initial state
        :param initial_reference: tested initial reference
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(MultiReferenceGenerator, '_reset_reference', self.monkey_reset_reference)
        sub_generator = ['SinusReference', 'WienerProcessReference']
        self._sub_generator = sub_generator
        monkeypatch.setattr(mrg, 'instantiate', self.monkey_instantiate)
        test_object = MultiReferenceGenerator(sub_generator)
        monkeypatch.setattr(test_object._sub_generators[0], 'reset', self.monkey_dummy_reset)
        self._initial_state = initial_state
        self._initial_reference = initial_reference
        # call function to test
        res_0, res_1, res_2 = test_object.reset(initial_state, initial_reference)
        # verify the expected results
        assert self._monkey_dummy_reset_counter == 1, 'reset of sub generators is not called once'
        assert res_0 == self._reference, 'reference is not the expected one'
        assert all(res_1 == self._reference_observation), 'observation is not the expected one '
        assert sum(sum(abs(res_2 - self._trajectory))) < 1E-6, \
            'absolute difference of reference trajectory to the expected is larger than 1e-6'

    def test_get_reference(self, monkeypatch):
        """
        test get_reference()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        sub_generator = ['SinusReference', 'WienerProcessReference']
        self._sub_generator = sub_generator
        monkeypatch.setattr(mrg, 'instantiate', self.monkey_instantiate)
        test_object = MultiReferenceGenerator(sub_generator)
        monkeypatch.setattr(DummyReferenceGenerator, 'get_reference', self.monkey_dummy_get_reference)
        # call function to test
        reference = test_object.get_reference(self._initial_state, **self._kwargs)
        # verify the expected results
        assert reference == self._reference, 'reference is not the expected one'

    @pytest.mark.parametrize("k, current_episode_length, k_new, reset_counter", [(10, 15, 11, 0), (10, 10, 11, 1)])
    def test_get_reference_observation(self, monkeypatch, k, current_episode_length, k_new, reset_counter):
        """
        test get_reference_observation()
        :param monkeypatch:
        :param k: used time step
        :param current_episode_length: value of the current episode length
        :param k_new: expected next time step
        :param reset_counter: expected values for reset_counter
        :return:
        """
        # setup test scenario
        sub_generator = ['SinusReference', 'WienerProcessReference']
        self._sub_generator = sub_generator
        monkeypatch.setattr(mrg, 'instantiate', self.monkey_instantiate)
        test_object = MultiReferenceGenerator(sub_generator)
        monkeypatch.setattr(test_object, '_k', k)
        monkeypatch.setattr(test_object, '_current_episode_length', current_episode_length)
        monkeypatch.setattr(test_object, '_reset_reference', self.monkey_reset_reference)
        monkeypatch.setattr(test_object, '_reference', self._initial_reference)
        monkeypatch.setattr(test_object._current_ref_generator, 'reset', self.monkey_dummy_reset)
        monkeypatch.setattr(test_object._current_ref_generator, 'get_reference_observation',
                            self.monkey_dummy_get_reference_observation)
        # call function to test
        observation = test_object.get_reference_observation(self._initial_state, **self._kwargs)
        # verify the expected results
        assert all(observation == self._reference_observation), 'observation is not the expected one'
        assert test_object._k == k_new, 'unexpected new step in the reference'
        assert self._monkey_reset_reference_counter == reset_counter, 'reset_reference called unexpected often'

    def test_reset_reference(self, monkeypatch):
        sub_reference_generators = [DummyReferenceGenerator(), DummyReferenceGenerator(), DummyReferenceGenerator()]
        probabilities = [0.2, 0.5, 0.3]
        super_episode_length = (1, 10)
        dummy_random = DummyRandom(exp_values=sub_reference_generators, exp_probabilities=probabilities,
                                   exp_low=super_episode_length[0], exp_high=super_episode_length[1])
        monkeypatch.setattr(mrg.np.random, 'randint', dummy_random.monkey_random_randint)
        monkeypatch.setattr(mrg.np.random, 'choice', dummy_random.monkey_random_choice)
        test_object = MultiReferenceGenerator(sub_generators=sub_reference_generators,
                                              super_episode_length=super_episode_length,
                                              p=probabilities)
        test_object._reset_reference()
        assert test_object._k == 0
        assert test_object._current_episode_length == 7
        assert test_object._current_ref_generator == sub_reference_generators[0]


class TestWienerProcessReferenceGenerator:
    """
    class for testing the wiener process reference generator
    """
    _kwargs = None
    _current_value = None

    # counter
    _monkey_super_init_counter = 0
    _monkey_get_current_value_counter = 0

    def monkey_super_init(self, **kwargs):
        """
        mock function for super().__init__()
        """
        self._monkey_super_init_counter += 1
        assert self._kwargs == kwargs, 'Different additional arguments. Keep in mind None and {}.'

    def monkey_get_current_value(self, value):
        if self._current_value is not None:
            assert value == self._current_value
        self._monkey_get_current_value_counter += 1
        return value

    @pytest.mark.parametrize('kwargs, expected_result', [({}, {}), ({'test': 42}, {'test': 42})])
    @pytest.mark.parametrize('sigma_range', [(2e-3, 2e-1)])
    def test_init(self, monkeypatch, kwargs, expected_result, sigma_range):
        """
        test init()
        :param monkeypatch:
        :param kwargs: additional arguments
        :param expected_result: expected result of additional arguments
        :param sigma_range: used range of sigma
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(SubepisodedReferenceGenerator, '__init__', self.monkey_super_init)
        self._kwargs = kwargs
        # call function to test
        test_object = WienerProcessReferenceGenerator(sigma_range=sigma_range, **self._kwargs)
        # verify the expected results
        assert self._monkey_super_init_counter == 1, 'super().__init__() is not called once'
        assert test_object._sigma_range == sigma_range, 'sigma range is not passed correctly'

    def test_reset_reference(self, monkeypatch):
        sigma_range = 1e-2
        episode_length = 10
        limit_margin = (-1, 1)
        reference_value = 0.5
        expected_reference = np.array([0.6, 0.4, 1, 1, 0.5, 0.2, -1, -0.9, -1, -0.6])
        monkeypatch.setattr(wrg.np.random, 'normal',
                            DummyRandom(exp_loc=0, exp_scale=sigma_range, exp_size=episode_length).monkey_random_normal)
        monkeypatch.setattr(SubepisodedReferenceGenerator, '_get_current_value', self.monkey_get_current_value)
        test_object = WienerProcessReferenceGenerator(sigma_range=sigma_range, episode_lengths=episode_length,
                                                      limit_margin=limit_margin)
        test_object._reference_value = reference_value
        self._monkey_get_current_value_counter = 0
        self._current_value = np.log10(sigma_range)
        test_object._reset_reference()
        assert sum(abs(test_object._reference - expected_reference)) < 1E-6, 'unexpected reference array'
        assert self._monkey_get_current_value_counter == 1, 'get_current_value() not called once'


class TestFurtherReferenceGenerator:
    """
    class for testing SawtoothReferenceGenerator, SinusoidalReferenceGenerator, StepReferenceGenerator,
    TriangularReferenceGenerator
    """
    # defined values for tests
    _kwargs = {}
    _physical_system = None
    _limit_margin = (0.0, 0.9)

    # counter
    _monkey_super_init_counter = 0
    _monkey_super_set_modules_counter = 0
    _monkey_get_current_value_counter = 0

    def monkey_super_init(self, **kwargs):
        """
        mock function for super().__init()__
        :param kwargs:
        :return:
        """
        assert self._kwargs == kwargs, 'Different additional arguments. Keep in mind None and {}.'
        self._monkey_super_init_counter += 1

    def monkey_super_set_modules(self, physical_system):
        """
        mock function for super().__set_modules()
        :param physical_system:
        :return:
        """
        self._monkey_super_set_modules_counter += 1
        assert physical_system == self._physical_system, 'physical system is not the expected instance'

    def monkey_get_current_value(self, value):
        self._monkey_get_current_value_counter += 1
        return value

    @pytest.mark.parametrize('amplitude_range', [(0.1, 0.8)])
    @pytest.mark.parametrize('frequency_range', [(2, 150)])
    @pytest.mark.parametrize('offset_range', [(-0.8, 0.5)])
    @pytest.mark.parametrize('kwargs', [{}])
    @pytest.mark.parametrize("reference_class",
                             [SawtoothReferenceGenerator, SinusoidalReferenceGenerator, StepReferenceGenerator,
                              TriangularReferenceGenerator])
    def test_init(self, monkeypatch, reference_class, amplitude_range, frequency_range, offset_range, kwargs):
        """
        test initialization of different reference generators
        :param monkeypatch:
        :param reference_class: class name of tested reference generator
        :param amplitude_range: range of the amplitude
        :param frequency_range: range of the frequency
        :param offset_range: range of the offset
        :param kwargs: further arguments
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(SubepisodedReferenceGenerator, '__init__', self.monkey_super_init)
        self._kwargs = kwargs
        # call function to test
        test_object = reference_class(amplitude_range=amplitude_range, frequency_range=frequency_range,
                                      offset_range=offset_range, **kwargs)
        # verify the expected results
        assert test_object._amplitude_range == amplitude_range, 'amplitude range is not passed correctly'
        assert test_object._frequency_range == frequency_range, 'frequency range is not passed correctly'
        assert test_object._offset_range == offset_range, 'offset range is not passed correctly'
        assert self._monkey_super_init_counter == 1, 'super().__init__() is not called once'

    @pytest.mark.parametrize("reference_class",
                             [SawtoothReferenceGenerator, SinusoidalReferenceGenerator, StepReferenceGenerator,
                              TriangularReferenceGenerator])
    @pytest.mark.parametrize('amplitude_range, expected_amplitude',
                             [((0.1, 0.8), (0.1, 0.45)), ((-0.5, 0.35), (0.0, 0.35))])
    @pytest.mark.parametrize('offset_range, expected_offset', [((-0.8, 0.5), (0.0, 0.5)), ((0.1, 0.96), (0.1, 0.9))])
    def test_set_modules(self, monkeypatch, reference_class, amplitude_range, offset_range,
                         expected_offset, expected_amplitude):
        """
        test set_modules()
        :param monkeypatch:
        :param reference_class: class name of tested reference generator
        :param amplitude_range: range of the amplitude
        :param offset_range: range of the offset
        :param expected_offset: expected result of the offset
        :param expected_amplitude: expected result of the amplitude
        :return:
        """
        # setup test scenario
        self._physical_system = DummyPhysicalSystem()
        monkeypatch.setattr(SubepisodedReferenceGenerator, 'set_modules', self.monkey_super_set_modules)
        test_object = reference_class(amplitude_range=amplitude_range, offset_range=offset_range)
        monkeypatch.setattr(test_object, '_limit_margin', self._limit_margin)
        # call function to test
        test_object.set_modules(self._physical_system)
        # verify the expected results
        assert all(test_object._amplitude_range == expected_amplitude), 'amplitude range is not as expected'
        assert all(test_object._offset_range == expected_offset), 'offset range is not as expected'
        assert self._monkey_super_set_modules_counter == 1, 'super().set_modules() is not called once'

    @pytest.mark.parametrize('reference_class, expected_reference, frequency_range', [
        (SawtoothReferenceGenerator, np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, -0.4, -0.3, -0.2, -0.1]), 1 / 8),
        (SinusoidalReferenceGenerator,
         np.array([0.4,  0.2828427, 0.0, -0.2828427, -0.4, -0.2828427, 0.0, 0.2828427, 0.4, 0.2828427]), 1 / 8),
        (StepReferenceGenerator, np.array([-0.4, -0.4, -0.4, 0.4, 0.4, 0.4, -0.4, -0.4, -0.4, 0.4]), 6),
        (TriangularReferenceGenerator, np.array([0.4, 0.2666667, 0.133333, 0.0, -0.133333, -0.2666667, -0.4, 0.0, 0.4,
                                                 0.2666667]), 1 / 8)])
    def test_reset_reference(self, monkeypatch, reference_class, expected_reference, frequency_range):
        # setup test scenario
        amplitude_range = 0.8
        offset_range = 0.5
        limit_margin = 0.4
        episode_length = 10
        dummy_physical_system = DummyPhysicalSystem()
        monkeypatch.setattr(SubepisodedReferenceGenerator, '_get_current_value', self.monkey_get_current_value)
        monkeypatch.setattr(sawrg.np.random, 'rand', DummyRandom().monkey_random_rand)
        monkeypatch.setattr(sawrg.np.random, 'triangular', DummyRandom().monkey_random_triangular)
        test_object = reference_class(amplitude_range=amplitude_range, frequency_range=frequency_range,
                                      offset_range=offset_range, limit_margin=limit_margin,
                                      episode_lengths=episode_length, reference_state='dummy_state_0')
        test_object.set_modules(dummy_physical_system)
        # call function to test
        test_object._reset_reference()
        # verify expected results
        assert sum(abs(expected_reference - test_object._reference)) < 1E-6, 'unexpected reference'


class TestSubepisodedReferenceGenerator:
    """
    class to the SubepisodedReferenceGenerator
    """
    # defined values for tests
    _episode_length = (10, 50)
    _reference_state = 'dummy_1'
    _referenced_states = np.array([0, 1, 0])
    _referenced_states = _referenced_states.astype(bool)
    _value_range = None
    _current_value = 35
    _initial_state = None
    _physical_system = DummyPhysicalSystem()
    _state_names = ['dummy_0', 'dummy_1', 'dummy_2']
    _nominals = np.array([1, 2, 3])
    _limits = np.array([5, 7, 6])
    _reference = np.array([0, 1, 0])
    _reference_trajectory = np.array([0.4, 0, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0])
    _state_space_low = np.zeros(3)
    _state_space_high = np.ones(3)
    _physical_system.state_space.low = _state_space_low
    _physical_system.state_space.high = _state_space_high
    _physical_system._limits = _limits
    _physical_system._nominal_values = _nominals
    _physical_system._state_names = _state_names
    _length_state = 3

    # counter
    _monkey_super_init_counter = 0
    _monkey_super_set_modules_counter = 0
    _monkey_super_reset_counter = 0
    _monkey_reset_reference_counter = 0

    def monkey_super_init(self):
        """
        mock function for super().__init__()
        :return:
        """
        self._monkey_super_init_counter += 1

    def monkey_get_current_value(self, value_range):
        """
        mock function for get_current_value()
        :param value_range:
        :return:
        """
        assert value_range == self._value_range, 'value range is not as expected'
        return self._current_value

    def monkey_reset_reference(self):
        """
        mock function for reset_reference()
        :return:
        """
        self._monkey_reset_reference_counter += 1

    def monkey_super_reset(self, initial_state):
        """
        mock function for super().reset()
        :param initial_state:
        :return:
        """
        assert initial_state == self._initial_state, 'initial state is not as expected'
        self._monkey_super_reset_counter += 1
        return self._reference

    def monkey_state_array(self, input_values, state_names):
        """
        mock function for utils.set_state_array()
        :param input_values:
        :param state_names:
        :return:
        """
        assert input_values == {self._reference_state: 1}, \
            'the input values are not a dict with the reference state and value 1'
        assert state_names == self._state_names, 'state names are not as expected'
        return np.array([0, 1, 0])

    def monkey_super_set_modules(self, physical_system):
        """
        mock function for super().set_modules()
        :param physical_system:
        :return:
        """
        self._monkey_super_set_modules_counter += 1
        assert physical_system == self._physical_system, 'physical system is not the expected instance'

    @pytest.mark.parametrize("limit_margin", [None, 0.3, (-0.1, 0.8)])
    def test_init(self, monkeypatch, limit_margin):
        """
        test __init__()
        :param monkeypatch:
        :param limit_margin: possible values for limit margin
        :return:
        """
        # setup test scenario
        self._value_range = self._episode_length
        monkeypatch.setattr(ReferenceGenerator, '__init__', self.monkey_super_init)
        monkeypatch.setattr(SubepisodedReferenceGenerator, '_get_current_value', self.monkey_get_current_value)
        # call function to test
        test_object = SubepisodedReferenceGenerator(reference_state=self._reference_state,
                                                    episode_lengths=self._episode_length, limit_margin=limit_margin)
        # verify the expected results
        assert self._monkey_super_init_counter == 1, 'super().__init__() is not called once'
        assert test_object._limit_margin == limit_margin, 'limit margin is not passed correctly'
        assert test_object._reference_value == 0.0, 'the reference value is not 0'
        assert test_object._reference_state == self._reference_state, 'reference state is not passed correctly'
        assert test_object._episode_len_range == self._episode_length, 'episode length is not passed correctly'
        assert test_object._current_episode_length == self._current_value, 'current episode length is not as expected'
        assert test_object._k == 0, 'current reference step is not 0'

    @pytest.mark.parametrize("limit_margin, expected_low, expected_high",
                             [(None, 0, 2 / 7), (0.3, 0.0, 0.3), ((-0.1, 0.8), -0.1, 0.8)])
    def test_set_modules(self, monkeypatch, limit_margin, expected_low, expected_high):
        """
        test set_modules()
        :param monkeypatch:
        :param limit_margin: possible values for limit margin
        :param expected_low: expected value for lower limit margin
        :param expected_high: expected value for upper limit margin
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(ReferenceGenerator, 'set_modules', self.monkey_super_set_modules)
        monkeypatch.setattr(srg, 'set_state_array', self.monkey_state_array)
        test_object = SubepisodedReferenceGenerator(reference_state=self._reference_state, limit_margin=limit_margin)
        # call function to test
        test_object.set_modules(self._physical_system)
        # verify the expected results
        assert self._monkey_super_set_modules_counter == 1, 'super().set_modules() is not called once'
        assert all(test_object.reference_space.low == expected_low), 'lower reference space not as expected'
        assert all(test_object.reference_space.high == expected_high), 'upper reference space not as expected'
        test_object._limit_margin = ['test']
        # call function to test
        with pytest.raises(Exception):
            test_object.set_modules(self._physical_system)

    @pytest.mark.parametrize('initial_state', [None, _initial_state])
    @pytest.mark.parametrize('initial_reference, expected_reference', [(np.array([0.2, 0.4, 0.7]), 0.4), (None, 0.0)])
    def test_reset(self, monkeypatch, initial_reference, initial_state, expected_reference):
        """
        test reset()
        :param monkeypatch:
        :param initial_reference: possible values for the initial reference
        :param initial_state: possible values for the initial state
        :param expected_reference: expected value for the reference
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(ReferenceGenerator, 'reset', self.monkey_super_reset)
        test_object = SubepisodedReferenceGenerator(reference_state=self._reference_state)
        monkeypatch.setattr(test_object, '_referenced_states', self._referenced_states)
        # call function to test
        reference = test_object.reset(initial_state, initial_reference)
        # verify the expected results
        assert all(reference == self._reference), 'reference not as expected'
        assert test_object._current_episode_length == -1, 'current episode length is not -1'
        assert test_object._reference_value == expected_reference, 'unexpected reference value'
        assert self._monkey_super_reset_counter == 1, 'super().reset() not called once'

    def test_get_reference(self, monkeypatch):
        """
        test get_reference()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        test_object = SubepisodedReferenceGenerator()
        monkeypatch.setattr(test_object, '_referenced_states', self._referenced_states)
        monkeypatch.setattr(test_object, '_reference_value', 0.4)
        # call function to test
        reference = test_object.get_reference()
        # verify the expected results
        assert all(reference == np.array([0, 0.4, 0])), 'unexpected reference'

    @pytest.mark.parametrize('k, expected_reference, expected_parameter', [(8, 0.6, (10, 0)), (10, 0.4, (35, 1))])
    # setup test scenario
    def test_get_reference_observation(self, monkeypatch, k, expected_reference, expected_parameter):
        """
        test get_reference_observation()
        :param monkeypatch:
        :param k: current time step for testing
        :param expected_reference: expected value for the reference
        :param expected_parameter: expected counter for reset reference
        :return:
        """
        monkeypatch.setattr(SubepisodedReferenceGenerator, '_get_current_value', self.monkey_get_current_value)
        monkeypatch.setattr(SubepisodedReferenceGenerator, '_reset_reference', self.monkey_reset_reference)
        self._value_range = self._episode_length
        test_object = SubepisodedReferenceGenerator(episode_lengths=self._episode_length)
        monkeypatch.setattr(test_object, '_reference', self._reference_trajectory)
        monkeypatch.setattr(test_object, '_k', k)
        monkeypatch.setattr(test_object, '_current_episode_length', 10)
        # call function to test
        reference = test_object.get_reference_observation()
        # verify the expected results
        assert reference == np.array([expected_reference]), 'unexpected reference'
        assert test_object._current_episode_length == expected_parameter[0], 'unexpected current episode length'
        assert self._monkey_reset_reference_counter == expected_parameter[1], \
            'unexpected number of calls of reset_reference, depends on the setting'

    @pytest.mark.parametrize('value_range, expected_value', [(12, 12),
                                                             (1.0365, 1.0365),
                                                             ([0, 1.6], 0.4),
                                                             ((-0.12, 0.4), 0.01),
                                                             (np.array([-0.5, 0.6]), -0.225)])
    def test_get_current_value(self, monkeypatch, value_range, expected_value):
        # setup test scenario
        monkeypatch.setattr(srg.np.random, 'rand', DummyRandom().monkey_random_rand)
        # call function to test
        val = SubepisodedReferenceGenerator._get_current_value(value_range)
        # verify expected results
        assert abs(val - expected_value) < 1E-6, 'unexpected value from get_current_value().'

# endregion

