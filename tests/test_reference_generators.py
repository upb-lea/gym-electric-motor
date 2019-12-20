import gym_electric_motor.envs
from .functions_used_for_testing import *
from gym_electric_motor.physical_systems.converters import *
from gym_electric_motor.utils import make_module
from numpy.random import seed
import numpy.random as rd
from gym.spaces import Box
import pytest

g_wiener_process_expect_1 = [0.1315388322520193, 0.14048739807899743, 0.1, 0.1, 0.1522230086844815, 0.1, 0.1,
                             0.14003241950101647, 0.11262368138853361, 0.1, 0.1, 0.14716188097299707,
                             0.12695802586146862, 0.1129180835364033, 0.1, 0.16975763420496093, 0.2389098821722488,
                             0.27066085428092235, 0.2828731405086375, 0.30619078225730606]
g_wiener_process_expect_2 = [-0.002791821788105553, 0.0061567440388726035, -0.04147647719565641, -0.05977342370533539,
                             -0.0075504150208539, -0.08428875061861613, -0.09785215886257594, -0.05781973936155946,
                             -0.08522847747404232, -0.10669674258476337, -0.10969170315163049, -0.06252982217863343,
                             -0.08273367729016186, -0.09677361961522718, -0.11050901297076794, -0.040751378765807006,
                             0.02840086920148087, 0.06015184131015443, 0.07236412753786958, 0.09568176928653817]
g_wiener_process_expect_3 = [0.03153883225201929, 0.04048739807899745, 0.0, 0.0, 0.052223008684481494, 0.0, 0.0,
                             0.04003241950101648, 0.012623681388533623, 0.0, 0.0, 0.04716188097299707,
                             0.026958025861468633, 0.012918083536403316, 0.0, 0.06975763420496094, 0.1389098821722488,
                             0.17066085428092237, 0.1828731405086375, 0.20619078225730608]

g_step_expect_1 = [0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175,
                   0.175, 0.175, 0.175, 0.175, 0.175, 0.175]
g_step_expect_2 = [-0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3,
                   -0.3, -0.3, -0.3]
g_step_expect_3 = [0.14999999999999994, 0.14999999999999994, 0.14999999999999994, 0.14999999999999994,
                   0.14999999999999994, 0.14999999999999994, 0.14999999999999994, 0.14999999999999994,
                   0.14999999999999994, 0.14999999999999994, 0.14999999999999994, 0.14999999999999994,
                   0.14999999999999994, 0.14999999999999994, 0.14999999999999994, 0.14999999999999994,
                   0.14999999999999994, 0.14999999999999994, 0.14999999999999994, 0.14999999999999994]

g_sinus_expect_1 = [0.18805372832646555, 0.1662776295356353, 0.14955022372492416, 0.13969858154094128,
                    0.1377987602312324, 0.1440582700211158, 0.15779340856435342, 0.1775039391390388, 0.2010369557745974,
                    0.22582203694795816, 0.24915200290882902, 0.2684786105837857, 0.2816908884342029,
                    0.2873457098145918, 0.2848254201840843, 0.27440530115280953, 0.25722350252305337,
                    0.23515672653350222, 0.21061524284174973, 0.21250000000000002]
g_sinus_expect_2 = [-0.0977850866941379, -0.18488948185745882, -0.2517991051003034, -0.29120567383623486,
                    -0.29880495907507043, -0.27376691991553687, -0.21882636574258643, -0.13998424344384486,
                    -0.045852176901610454, 0.05328814779183252, 0.14660801163531598, 0.2239144423351426,
                    0.27676355373681155, 0.29938283925836695, 0.28930168073633705, 0.247621204611238,
                    0.17889401009221326, 0.09062690613400883, -0.007539028633001196, 3.6739403974420595e-17]
g_sinus_expect_3 = [0.17610745665293104, 0.13255525907127058, 0.09910044744984828, 0.07939716308188255,
                    0.07559752046246476, 0.08811654004223154, 0.11558681712870676, 0.15500787827807755,
                    0.20207391154919474, 0.25164407389591625, 0.298304005817658, 0.33695722116757126,
                    0.3633817768684058, 0.37469141962918345, 0.36965084036816853, 0.348810602305619, 0.3144470050461066,
                    0.2703134530670044, 0.22123048568349937, 0.225]

g_sinus_epis_expect_1 = [0.1878331472243922, 0.16591083898597708, 0.149172249502306, 0.13947978922134457,
                         0.1379118846921974, 0.14464298799634628, 0.15892416642759086, 0.17916643209985167,
                         0.20311753985738898, 0.22811258215996472, 0.25137049863346106, 0.27030350930511504,
                         0.2828050425954793, 0.28748412190892886, 0.2838201319206939, 0.27222074457003725,
                         0.21250000000000002, 0.1878331472243922, 0.16591083898597708, 0.149172249502306]
g_sinus_epis_expect_3 = [0.17566629444878434, 0.13182167797195415, 0.09834449900461201, 0.07895957844268914,
                         0.07582376938439475, 0.0892859759926925, 0.11784833285518166, 0.15833286419970333,
                         0.2062350797147779, 0.25622516431992937, 0.3027409972669221, 0.3406070186102299,
                         0.3656100851909585, 0.3749682438178576, 0.3676402638413877, 0.34444148914007444, 0.225,
                         0.17566629444878434, 0.13182167797195415, 0.09834449900461201]
g_sinus_epis_expect_2 = [-0.09866741110243127, -0.18635664405609167, -0.25331100199077594, -0.2920808431146217,
                         -0.29835246123121045, -0.27142804801461495, -0.21430333428963663, -0.13333427160059333,
                         -0.03752984057044419, 0.06245032863985878, 0.15548199453384423, 0.2312140372204599,
                         0.28122017038191705, 0.29993648763571523, 0.2852805276827755, 0.23888297828014893,
                         3.6739403974420595e-17, -0.09866741110243127, -0.18635664405609167, -0.25331100199077594]

g_multi_expected_3 = [0.0, 0.054972090373755, 0.06782529748676631, 0.07801118572791686, 0.0763820523476228,
                      0.0699246508609987,
                      0.13251714416762556, 0.08131037995987651, 0.04608373206022021, 0.03193441412870947,
                      0.08469390558323936,
                      0.08016006986305074, 0.06057953433550248, 0.036247911555891604, 0.05448622214800124,
                      0.05848732305527147,
                      0.017330627963620712, 0.08713561261090365, 0.10366618093872126, 0.11839116102699339]
g_multi_expected_2 = [-0.05368087336940529, 0.0012912170043497123, 0.014144424117361032, 0.02433031235851158,
                      0.02270117897821752, 0.016243777491593423, 0.07883627079822027, 0.027629506590471226,
                      -0.007597141309185075, -0.021746459240695817, 0.031013032213834075, 0.02647919649364545,
                      0.006898660966097186, -0.01743296181351369, 0.0008053487785959455, 0.0048064496858661755,
                      -0.03635024540578458, 0.03345473924149835, 0.04998530756931597, 0.0647102876575881]
g_multi_expected_1 = [0.1, 0.154972090373755, 0.16782529748676633, 0.17801118572791688, 0.17638205234762283,
                      0.16992465086099873,
                      0.2325171441676256, 0.18131037995987653, 0.14608373206022024, 0.1319344141287095,
                      0.1846939055832394,
                      0.18016006986305078, 0.1605795343355025, 0.13624791155589164, 0.15448622214800128,
                      0.15848732305527152,
                      0.11733062796362076, 0.1871356126109037, 0.2036661809387213, 0.21839116102699344]

g_sawtooth_expected_1 = [0.22042631578947372, 0.2283526315789474, 0.2362789473684211, 0.24420526315789476,
                         0.2521315789473685, 0.26005789473684215, 0.2679842105263158, 0.2759105263157895,
                         0.28383684210526317, 0.14176315789473687, 0.14968947368421054, 0.15761578947368424,
                         0.16554210526315793, 0.1734684210526316, 0.1813947368421053, 0.189321052631579,
                         0.19724736842105267, 0.20517368421052634, 0.2131, 0.21250000000000002]
g_sawtooth_expected_2 = [0.03170526315789474, 0.06341052631578949, 0.09511578947368424, 0.12682105263157897,
                         0.15852631578947374, 0.19023157894736842, 0.22193684210526315, 0.2536421052631579,
                         0.2853473684210526, -0.2829473684210526, -0.2512421052631578, -0.2195368421052631,
                         -0.18783157894736835, -0.15612631578947359, -0.12442105263157882, -0.0927157894736841,
                         -0.06101052631578939, -0.029305263157894678, 0.002400000000000002, 0.0]
g_sawtooth_expected_3 = [0.24085263157894735, 0.2567052631578947, 0.2725578947368421, 0.28841052631578945,
                         0.30426315789473685, 0.3201157894736842, 0.3359684210526316, 0.3518210526315789,
                         0.36767368421052626, 0.08352631578947367, 0.09937894736842107, 0.11523157894736842,
                         0.1310842105263158, 0.1469368421052632, 0.16278947368421057, 0.17864210526315794,
                         0.19449473684210528, 0.21034736842105264, 0.22619999999999998, 0.22499999999999998]

g_sawtooth_epis_expected_3 = [0.24100125, 0.2570025, 0.27300375, 0.28900499999999996, 0.30500625, 0.3210075,
                              0.33700874999999997, 0.35301, 0.36901125, 0.08501250000000002, 0.10101374999999997,
                              0.11701500000000002, 0.13301625000000003, 0.14901750000000002, 0.16501875000000008,
                              0.18102000000000001, 0.22499999999999998, 0.24100125, 0.2570025, 0.27300375]
g_sawtooth_epis_expected_1 = [0.22050062500000003, 0.22850125000000004, 0.23650187500000003, 0.24450250000000004,
                              0.25250312500000005, 0.26050375000000003, 0.268504375, 0.27650500000000006,
                              0.28450562500000004, 0.14250625000000003, 0.150506875, 0.15850750000000002,
                              0.16650812500000003, 0.17450875000000005, 0.18250937500000006, 0.19051000000000004,
                              0.21250000000000002, 0.22050062500000003, 0.22850125000000004, 0.23650187500000003]
g_sawtooth_epis_expected_2 = [0.032002500000000024, 0.06400500000000005, 0.0960075, 0.12801, 0.16001250000000003,
                              0.19201500000000007, 0.2240175, 0.25602, 0.2880225, -0.2799749999999999, -0.2479725,
                              -0.2159699999999999, -0.1839674999999999, -0.15196499999999988, -0.11996249999999982,
                              -0.0879599999999999, 0.0, 0.032002500000000024, 0.06400500000000005, 0.0960075]

g_triangle_expect_1 = [0.27164736842105264, 0.2557947368421053, 0.23994210526315793, 0.2240894736842105,
                       0.20823684210526316, 0.1923842105263158, 0.17653157894736843, 0.16067894736842103,
                       0.1448263157894737, 0.1460263157894737, 0.1618789473684211, 0.17773157894736846,
                       0.19358421052631586, 0.2094368421052632, 0.22528947368421062, 0.24114210526315796,
                       0.25699473684210533, 0.27284736842105267, 0.2863, 0.28750000000000003]
g_triangle_expect_2 = [0.2365894736842105, 0.17317894736842096, 0.10976842105263156, 0.046357894736841966,
                       -0.017052631578947448, -0.08046315789473686, -0.1438736842105263, -0.20728421052631588,
                       -0.2706947368421053, -0.26589473684210524, -0.20248421052631568, -0.13907368421052627,
                       -0.07566315789473667, -0.012252631578947237, 0.05115789473684235, 0.11456842105263178,
                       0.1779789473684212, 0.24138947368421063, 0.29519999999999996, 0.3]
g_triangle_expect_3 = [0.3432947368421052, 0.31158947368421047, 0.27988421052631574, 0.24817894736842097,
                       0.21647368421052626, 0.18476842105263155, 0.15306315789473685, 0.12135789473684204,
                       0.08965263157894732, 0.09205263157894736, 0.12375789473684214, 0.15546315789473686,
                       0.18716842105263165, 0.21887368421052636, 0.2505789473684212, 0.28228421052631586,
                       0.3139894736842106, 0.3456947368421053, 0.37259999999999993, 0.375]

g_triangle_epis_expect_1 = [0.27149875, 0.25549750000000004, 0.23949625000000002, 0.223495, 0.20749375, 0.1914925,
                            0.17549125, 0.15949, 0.14348875000000003, 0.14751250000000005, 0.16351375,
                            0.17951500000000006, 0.19551625000000006, 0.21151750000000008, 0.2275187500000001,
                            0.24352000000000007, 0.28750000000000003, 0.27149875, 0.25549750000000004,
                            0.23949625000000002]
g_triangle_epis_expect_2 = [0.23599499999999998, 0.17198999999999998, 0.10798499999999996, 0.04397999999999994,
                            -0.020025000000000067, -0.08403000000000009, -0.14803500000000008, -0.2120400000000001,
                            -0.27604499999999993, -0.25994999999999985, -0.19594500000000004, -0.1319399999999998,
                            -0.06793499999999981, -0.0039299999999998, 0.06007500000000034, 0.12408000000000019, 0.3,
                            0.23599499999999998, 0.17198999999999998, 0.10798499999999996]
g_triangle_epis_expect_3 = [0.34299749999999996, 0.31099499999999997, 0.2789925, 0.24698999999999996,
                            0.21498749999999994, 0.18298499999999993, 0.15098249999999994, 0.11897999999999993,
                            0.08697750000000001, 0.09502500000000005, 0.12702749999999996, 0.15903000000000006,
                            0.19103250000000008, 0.22303500000000007, 0.25503750000000014, 0.2870400000000001, 0.375,
                            0.34299749999999996, 0.31099499999999997, 0.2789925]


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
    for k in range(20):
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
