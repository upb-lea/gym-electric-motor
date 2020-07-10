from gym_electric_motor.physical_systems.electric_motors import *
from ..conf import *
from gym_electric_motor.utils import make_module
import pytest

# region first version tests

# global parameter
g_omega = [0, 15, 12.2, 13]
g_i_a = [0, 3, 40, 35.3, -3]  # also used for permanently excited and series dc motor
g_i_e = [0, 0.2, 0.5, -1, 1]
g_u_in = [0, 400, -400, 325.2, -205.4]
g_t_load = [0, 1, 5, -5, 4.35]

# default/ global initializer
default_test_initializer = {'states': {},
                    'interval': None,
                    'random_init': None,
                    'random_params': None}
test_initializer = {'states': {},
                    'interval': None,
                    'random_init': None,
                    'random_params': None}


# region general test functions


def motor_testing(motor_1_default, motor_2_default, motor_1, motor_2,
                  motor_state, limit_values, nominal_values, motor_parameter,
                  state_positions, state_space):
    motor_reset_testing(motor_1_default, motor_2_default, motor_state,
                        state_positions, state_space, nominal_values)
    motor_reset_testing(motor_1, motor_2, motor_state, state_positions,
                        state_space, nominal_values)

    limit_nominal_testing(motor_1_default, motor_2_default, motor_1, motor_2, limit_values, nominal_values)
    motor_parameter_testing(motor_1_default, motor_2_default, motor_1, motor_2, motor_parameter)


def limit_nominal_testing(motor_1_default, motor_2_default, motor_1, motor_2, limit_values, nominal_values):
    # test if limits and nominal values were taken correct
    assert motor_1.nominal_values == motor_2.nominal_values, "Nominal Values: " + str(motor_1.nominal_values)
    assert motor_1.limits == motor_2.limits
    assert motor_1_default.nominal_values == motor_2_default.nominal_values
    assert motor_1_default.limits == motor_2_default.limits
    assert motor_1.nominal_values is not motor_1_default.nominal_values, " Nominal value error, no difference between" \
                                                                         "default and parametrized values"
    assert motor_1.limits is not motor_1_default.limits
    assert motor_1.limits == limit_values, "Limit Error " + str([motor_1.limits, limit_values])
    assert motor_1.nominal_values == nominal_values, "Nominal Error " + str([motor_1.nominal_values, nominal_values])


def motor_parameter_testing(motor_1_default, motor_2_default, motor_1, motor_2, motor_parameter):
    # tests if all motor parameter are the same
    assert motor_1_default.motor_parameter == motor_2_default.motor_parameter
    assert motor_1.motor_parameter == motor_2.motor_parameter == motor_parameter, "Different Motor Parameter"
    assert motor_1_default.motor_parameter is not motor_1.motor_parameter, "Failed motor parameter test"


def motor_reset_testing(motor_1, motor_2, motor_state, state_positions,
                        state_space, nominal_values):
    # tests if the reset function works correctly
    assert motor_1.initializer == motor_2.initializer
    # test random initialization
    if motor_1.initializer['states'] is None:
        for idx, state in enumerate(motor_state):
            assert np.abs(motor_1.reset(state_space, state_positions)[idx] \
                     <= nominal_values[state])
            assert np.abs(motor_2.reset(state_space, state_positions)[idx] \
                     <= nominal_values[state])
    # test constant initialization
    else:
        assert all(motor_1.reset(state_space, state_positions) ==
                motor_2.reset(state_space, state_positions))
        init_values = list(motor_1.initializer['states'].values())
        assert all(motor_1.reset(state_space, state_positions) == init_values)

# endregion

# region DcSeriesMotor
@pytest.mark.parametrize(
    'states, interval, random_init, random_params',
    [({'i': 16}, None, None, (None, None)),
     (None, None, 'uniform', (None, None)),
     (None, [[10, 20]], 'uniform', (None, None)),
     (None, None, 'gaussian', (None, None)),
     (None, None, 'gaussian', (16, 2)),
     (None, [[10, 20]], 'gaussian', (None, None))])
def test_dc_series_motor(states, interval, random_init, random_params):
    #motor_state = ['i', 'i']  # list of state names
    motor_state = ['i']  # list of state names
    motor_parameter = test_motor_parameter['DcSeries']['motor_parameter']
    nominal_values = test_motor_parameter['DcSeries']['nominal_values']  # dict
    limit_values = test_motor_parameter['DcSeries']['limit_values']  # dict
    # set initializer parameters
    test_initializer['states'] = states
    test_initializer['interval'] = interval
    test_initializer['random_init'] = random_init
    test_initializer['random_params'] = random_params
    # default initialization
    motor_1_default = make_module(ElectricMotor, 'DcSeries')
    motor_2_default = DcSeriesMotor()
    # initialization parameters as dicts
    motor_1 = make_module(ElectricMotor, 'DcSeries',
                          motor_parameter=motor_parameter,
                          nominal_values=nominal_values,
                          limit_values=limit_values,
                          motor_initializer=test_initializer)
    motor_2 = DcSeriesMotor(motor_parameter, nominal_values,
                            limit_values, test_initializer)
    motor_testing(motor_1_default, motor_2_default, motor_1, motor_2,
                  motor_state, limit_values, nominal_values, motor_parameter,
                  series_state_positions, series_state_space)

    series_motor_state_space_testing(motor_1_default, motor_2_default)
    series_motor_state_space_testing(motor_1, motor_2)

    series_motor_electrical_ode_testing(motor_1_default)
    series_motor_electrical_ode_testing(motor_1)


def series_motor_state_space_testing(motor_1, motor_2):
    # u,i>0
    state_space = ({'omega': 0, 'torque': 0, 'i': 0, 'u': 0},
                   {'omega': 1, 'torque': 1, 'i': 1, 'u': 1})
    box01 = Box(0, 1, shape=(1,))
    box_11 = Box(-1, 1, shape=(1,))
    assert motor_1.get_state_space(box01, box01) == motor_2.get_state_space(box01, box01)
    assert motor_1.get_state_space(box01, box01) == state_space

    # u>0
    state_space = (
        {'omega': 0, 'torque': 0, 'i': -1, 'u': 0}, {'omega': 1, 'torque': 1, 'i': 1, 'u': 1})
    assert motor_1.get_state_space(box_11, box01) == motor_2.get_state_space(box_11, box01)
    assert motor_1.get_state_space(box_11, box01) == state_space

    # i>0
    state_space = (
        {'omega': 0, 'torque': 0, 'i': 0, 'u': -1}, {'omega': 1, 'torque': 1, 'i': 1, 'u': 1})
    assert motor_1.get_state_space(box01, box_11) == motor_2.get_state_space(box01, box_11)
    assert motor_1.get_state_space(box01, box_11) == state_space

    # u,i>&<0
    state_space = (
        {'omega': 0, 'torque': 0, 'i': -1, 'u': -1}, {'omega': 1, 'torque': 1, 'i': 1, 'u': 1})
    assert motor_1.get_state_space(box_11, box_11) == motor_2.get_state_space(box_11, box_11)
    assert motor_1.get_state_space(box_11, box_11) == state_space


def series_motor_electrical_ode_testing(motor_1):
    # test set j load
    # test motor electrical_ode, i_in, torque
    I_IDX = 0
    mp = motor_1.motor_parameter
    for omega in g_omega:
        for i in g_i_a:
            state = np.array([i, i])
            assert motor_1.i_in(state) == np.array([i]), "Input current is wrong"  # i_in
            assert abs(motor_1.torque(state) - mp['l_e_prime'] * i ** 2) < 1E-10, "Wrong torque"  # torque
            for u_in in g_u_in:
                result = np.array([(u_in - mp['l_e_prime'] * state[I_IDX] * omega -
                                    (mp['r_a'] + mp['r_e']) * state[I_IDX]) / (mp['l_a'] + mp['l_e'])])
                assert sum(abs(motor_1.electrical_ode(state, [u_in], omega) - result)) < 1E-10, \
                    "Used Parameter set (state, u_in): " + str([state, u_in])


# endregion

# region DcShuntMotor

@pytest.mark.parametrize(
    'states, interval, random_init, random_params',
    [({'i_a': 16, 'i_e': 1.6}, None, None, (None, None)),
     (None, None, 'uniform', (None, None)),
     (None, [[10, 20], [1, 2]], 'uniform', (None, None)),
     (None, None, 'gaussian', (None, None)),
     (None, None, 'gaussian', (None, 2)),
     (None, [[10, 20], [1, 2]], 'gaussian', (None, None))])
def test_dc_shunt_motor(states, interval, random_init, random_params):
    """
    tests the dc shunt motor class
    :return:
    """
    # set up test parameter
    motor_state = ['i_a', 'i_e']  # list of state names
    motor_parameter = test_motor_parameter['DcShunt']['motor_parameter']
    nominal_values = test_motor_parameter['DcShunt']['nominal_values']  # dict
    limit_values = test_motor_parameter['DcShunt']['limit_values']  # dict
    # set initializer parameters
    test_initializer['states'] = states
    test_initializer['interval'] = interval
    test_initializer['random_init'] = random_init
    test_initializer['random_params'] = random_params
    # default initialization of electric motor
    motor_1_default = make_module(ElectricMotor, 'DcShunt')
    motor_2_default = DcShuntMotor()

    motor_1 = make_module(ElectricMotor, 'DcShunt',
                          motor_parameter=motor_parameter,
                          nominal_values=nominal_values,
                          limit_values=limit_values,
                          motor_initializer=test_initializer)
    motor_2 = DcShuntMotor(motor_parameter, nominal_values,
                           limit_values, test_initializer)
    # test if both initializations work correctly for limits and nominal values
    motor_testing(motor_1_default, motor_2_default, motor_1, motor_2,
                  motor_state, limit_values, nominal_values, motor_parameter,
                  shunt_state_positions, shunt_state_space)

    shunt_motor_state_space_testing(motor_1_default, motor_2_default)
    shunt_motor_state_space_testing(motor_1, motor_2)
    # test electrical_ode function
    shunt_motor_electrical_ode_testing(motor_1_default)
    shunt_motor_electrical_ode_testing(motor_1)


def shunt_motor_state_space_testing(motor_1, motor_2):
    # test the motor state space for different converters
    # u,i>0
    state_space = ({'omega': 0, 'torque': 0, 'i_a': 0, 'i_e': 0, 'u': 0},
                   {'omega': 1, 'torque': 1, 'i_a': 1, 'i_e': 1, 'u': 1})
    box01 = Box(0, 1, shape=(1,))
    box_11 = Box(-1, 1, shape=(1,))
    assert motor_1.get_state_space(box01, box01) == motor_2.get_state_space(box01, box01)
    assert motor_1.get_state_space(box01, box01) == state_space

    # u>0
    state_space = (
        {'omega': 0, 'torque': -1, 'i_a': -1, 'i_e': -1, 'u': 0}, {'omega': 1, 'torque': 1, 'i_a': 1, 'i_e': 1, 'u': 1})
    assert motor_1.get_state_space(box_11, box01) == motor_2.get_state_space(box_11, box01)
    assert motor_1.get_state_space(box_11, box01) == state_space

    # i>0
    state_space = (
        {'omega': 0, 'torque': 0, 'i_a': 0, 'i_e': 0, 'u': -1}, {'omega': 1, 'torque': 1, 'i_a': 1, 'i_e': 1, 'u': 1})
    assert motor_1.get_state_space(box01, box_11) == motor_2.get_state_space(box01, box_11)
    assert motor_1.get_state_space(box01, box_11) == state_space

    # u,i>&<0
    state_space = (
        {'omega': 0, 'torque': -1, 'i_a': -1, 'i_e': -1, 'u': -1},
        {'omega': 1, 'torque': 1, 'i_a': 1, 'i_e': 1, 'u': 1})
    assert motor_1.get_state_space(box_11, box_11) == motor_2.get_state_space(box_11, box_11)
    assert motor_1.get_state_space(box_11, box_11) == state_space


def shunt_motor_electrical_ode_testing(motor_1):
    # test motor electrical_ode, i_in, torque
    I_A_IDX = 0
    I_E_IDX = 1
    # setup electrical_ode parameter
    mp = motor_1.motor_parameter
    # setup different test parameter
    for omega in g_omega:
        for i_a in g_i_a:
            for i_e in g_i_e:
                state = np.array([i_a, i_e])
                assert motor_1.i_in(state) == [i_a + i_e], "Input current is wrong"  # i_in
                assert motor_1.torque(state) == mp['l_e_prime'] * i_a * i_e, "Wrong torque"  # torque
                for u_in in g_u_in:
                    # test electrical_ode function
                    result = np.array([(u_in - mp['l_e_prime'] * state[I_E_IDX] * omega - mp['r_a']
                                        * state[I_A_IDX]) / mp['l_a'],
                                       (u_in - mp['r_e'] * state[I_E_IDX]) / mp['l_e']])
                    assert sum(abs(motor_1.electrical_ode(state, [u_in], omega) - result)) < 1E-10, \
                        "Used Parameter set (state, u_in, t_load): " + str([state, u_in])


# endregion

# region DcPermExMotor

@pytest.mark.parametrize(
    'states, interval, random_init, random_params',
    [({'i': 16}, None, None, (None, None)),
     (None, None, 'uniform', (None, None)),
     (None, [[10, 20]], 'uniform', (None, None)),
     (None, None, 'gaussian', (None, None)),
     (None, None, 'gaussian', (None, 2)),
     (None, [[10, 20]], 'gaussian', (None, None))])
def test_dc_permex_motor(states, interval, random_init, random_params):
    motor_state = ['i']  # list of state names
    motor_parameter = test_motor_parameter['DcPermEx']['motor_parameter']
    nominal_values = test_motor_parameter['DcPermEx']['nominal_values']  # dict
    limit_values = test_motor_parameter['DcPermEx']['limit_values']  # dict
    # set initializer parameters
    test_initializer['states'] = states
    test_initializer['interval'] = interval
    test_initializer['random_init'] = random_init
    test_initializer['random_params'] = random_params
    # default initialization without parameters
    motor_1_default = make_module(ElectricMotor, 'DcPermEx')
    motor_2_default = DcPermanentlyExcitedMotor()
    # initialization parameters as dicts
    motor_1 = make_module(ElectricMotor, 'DcPermEx',
                          motor_parameter=motor_parameter,
                          nominal_values=nominal_values,
                          limit_values=limit_values,
                          motor_initializer=test_initializer)
    motor_2 = DcPermanentlyExcitedMotor(motor_parameter, nominal_values,
                                        limit_values, test_initializer)

    motor_testing(motor_1_default, motor_2_default, motor_1, motor_2,
                  motor_state, limit_values, nominal_values, motor_parameter,
                  permex_state_positions, permex_state_space)

    permex_motor_state_space_testing(motor_1_default, motor_2_default)
    permex_motor_state_space_testing(motor_1, motor_2)

    permex_motor_electrical_ode_testing(motor_1_default)
    permex_motor_electrical_ode_testing(motor_1)


def permex_motor_state_space_testing(motor_1, motor_2):
    # u,i>0
    state_space = ({'omega': 0, 'torque': 0, 'i': 0, 'u': 0},
                   {'omega': 1, 'torque': 1, 'i': 1, 'u': 1})
    box01 = Box(0, 1, shape=(1,))
    box_11 = Box(-1, 1, shape=(1,))
    assert motor_1.get_state_space(box01, box01) == motor_2.get_state_space(box01, box01)
    assert motor_1.get_state_space(box01, box01) == state_space

    # u>0
    state_space = (
        {'omega': 0, 'torque': -1, 'i': -1, 'u': 0}, {'omega': 1, 'torque': 1, 'i': 1, 'u': 1})
    assert motor_1.get_state_space(box_11, box01) == motor_2.get_state_space(box_11, box01)
    assert motor_1.get_state_space(box_11, box01) == state_space

    # i>0
    state_space = (
        {'omega': -1, 'torque': 0, 'i': 0, 'u': -1}, {'omega': 1, 'torque': 1, 'i': 1, 'u': 1})
    assert motor_1.get_state_space(box01, box_11) == motor_2.get_state_space(box01, box_11)
    assert motor_1.get_state_space(box01, box_11) == state_space

    # u,i>&<0
    state_space = (
        {'omega': -1, 'torque': -1, 'i': -1, 'u': -1}, {'omega': 1, 'torque': 1, 'i': 1, 'u': 1})
    assert motor_1.get_state_space(box_11, box_11) == motor_2.get_state_space(box_11, box_11)
    assert motor_1.get_state_space(box_11, box_11) == state_space


def permex_motor_electrical_ode_testing(motor_1):
    # test set j load
    # test motor electrical_ode, i_in, torque
    I_IDX = 0
    mp = motor_1.motor_parameter
    for omega in g_omega:
        for i in g_i_a:
            state = np.array([i])
            assert motor_1.i_in(state) == [i], "Input current is wrong"  # i_in
            assert abs(motor_1.torque(state) - mp['psi_e'] * i) < 1E-10, "Wrong torque"  # torque
            for u_in in g_u_in:
                result = np.array([(u_in - mp['psi_e'] * omega - mp['r_a'] * state[I_IDX]) / mp['l_a']])
                assert sum(abs(motor_1.electrical_ode(state, [u_in], omega) - result)) < 1E-10, \
                    "Used Parameter set (state, u_in, t_load): " + str(state) + " " + str(u_in)


# endregion

# region DcExtExMotor

@pytest.mark.parametrize(
    'states, interval, random_init, random_params',
    [({'i_a': 16, 'i_e': 1.6}, None, None, (None, None)),
     (None, None, 'uniform', (None, None)),
     (None, [[10, 20], [1, 2]], 'uniform', (None, None)),
     (None, None, 'gaussian', (None, None)),
     (None, None, 'gaussian', (None, 2)),
     (None, [[10, 20], [1, 2]], 'gaussian', (None, None))])
def test_dc_extex_motor(states, interval, random_init, random_params):
    motor_state = ['i_a', 'i_e']  # list of state names
    motor_parameter = test_motor_parameter['DcExtEx']['motor_parameter']
    nominal_values = test_motor_parameter['DcExtEx']['nominal_values']  # dict
    limit_values = test_motor_parameter['DcExtEx']['limit_values']  # dict
    test_initializer['states'] = states
    test_initializer['interval'] = interval
    test_initializer['random_init'] = random_init
    test_initializer['random_params'] = random_params
    # default initialization without parameters
    motor_1_default = make_module(ElectricMotor, 'DcExtEx')
    motor_2_default = DcExternallyExcitedMotor()
    # initialization parameters as dicts
    motor_1 = make_module(ElectricMotor, 'DcExtEx',
                          motor_parameter=motor_parameter,
                          nominal_values=nominal_values,
                          limit_values=limit_values,
                          motor_initializer=test_initializer)
    motor_2 = DcExternallyExcitedMotor(motor_parameter, nominal_values,
                                       limit_values, test_initializer)
    motor_testing(motor_1_default, motor_2_default, motor_1, motor_2,
                  motor_state, limit_values, nominal_values, motor_parameter,
                  extex_state_positions, extex_state_space)

    extex_motor_state_space_testing(motor_1_default, motor_2_default)
    extex_motor_state_space_testing(motor_1, motor_2)

    extex_motor_electrical_ode_testing(motor_1_default)
    extex_motor_electrical_ode_testing(motor_1)


def extex_motor_state_space_testing(motor_1, motor_2):
    # u,i>0
    state_space = ({'omega': 0, 'torque': 0, 'i_a': 0, 'i_e': 0, 'u_a': 0, 'u_e': 0}, {'omega': 1, 'torque': 1,
                                                                                       'i_a': 1, 'i_e': 1,
                                                                                       'u_a': 1, 'u_e': 1})
    voltage_limits = Box(np.array([0, 1]), np.array([0, 1])) # Box(np.array([0, 0]), np.array([1, 1])) # [[0, 1], [0, 1]]
    current_limits = Box(np.array([0, 0]), np.array([1, 1])) # [[0, 1], [0, 1]]
    assert motor_1.get_state_space(current_limits, voltage_limits) == motor_2.get_state_space(current_limits,
                                                                                              voltage_limits)
    assert motor_1.get_state_space(current_limits, voltage_limits) == state_space


def extex_motor_electrical_ode_testing(motor_1):
    # test motor electrical_ode, i_in, torque
    I_A_IDX = 0
    I_E_IDX = 1
    mp = motor_1.motor_parameter
    for omega in g_omega:
        for i_a in g_i_a:
            for i_e in g_i_e:
                state = np.array([i_a, i_e])
                assert all(motor_1.i_in(state) == np.array([i_a, i_e])), "Input current is wrong"  # i_in
                assert motor_1.torque(state) == mp['l_e_prime'] * i_a * i_e, "Wrong torque"  # torque
                for u_a in g_u_in:
                    for u_e in g_u_in:
                        result = np.array(
                            [(u_a - mp['l_e_prime'] * state[I_E_IDX] * omega
                              - mp['r_a'] * state[I_A_IDX]) / mp['l_a'],
                             (u_e - mp['r_e'] * state[I_E_IDX]) / mp['l_e']])
                        assert sum(abs(motor_1.electrical_ode(state, [u_a, u_e],
                                                              omega) - result)) < 1E-10, \
                            "Used Parameter set (state, u_a, u_e): " + str(state) + " " + str([u_a, u_e])


# endregion

# region PMSM and SynRM

def torque_testing(state, mp):
    """
    use this function as benchmark for the torque
    :param state: [i_d, i_q, epsilon]
    :param mp: motor parameter (dict)
    :return: generated torque
    """
    return 1.5 * mp['p'] * (mp['psi_p'] + (mp['l_d'] - mp['l_q']) * state[0]) * state[1]


def synchronous_motor_ode_testing(state, voltage, omega, mp):
    """
    use this function as benchmark for the ode
    :param state: [i_d, i_q, epsilon]
    :param voltage: [u_d, u_q]
    :param omega: angular velocity
    :param mp: motor parameter (dict)
    :return: ode system
    """
    u_d = voltage[0]
    u_q = voltage[1]
    i_d = state[0]
    i_q = state[1]
    return np.array([(u_d - mp['r_s'] * i_d + mp['l_q'] * omega * mp['p'] * i_q) / mp['l_d'],
                     (u_q - mp['r_s'] * i_q - omega * mp['p'] * (mp['l_d'] * i_d + mp['psi_p'])) / mp['l_q'],
                     omega * mp['p']])

@pytest.mark.parametrize(
    'states, interval, random_init, random_params',
    [({'i_sd': 16, 'i_sq': 16, 'epsilon': 1.6}, None, None, (None, None)),
     (None, None, 'uniform', (None, None)),
     (None, [[10, 20], [10, 20], [1, 2]], 'uniform', (None, None)),
     (None, None, 'gaussian', (None, None)),
     (None, None, 'gaussian', (None, 2)),
     (None, [[10, 20], [10, 20], [1, 2]], 'gaussian', (None, None))])
@pytest.mark.parametrize("motor_type, motor_class", [('SynRM', SynchronousReluctanceMotor),
                                                     ('PMSM', PermanentMagnetSynchronousMotor)])
def test_synchronous_motor_testing(motor_type, motor_class,
                                   states, interval, random_init, random_params):
    """
    testing the synrm and pmsm
    consider that it uses dq coordinates and the state is [i_d, i_q, epsilon]!!.
    voltage u_dq
    :return:
    """
    parameter = test_motor_parameter[motor_type]
    state = np.array([0.3, 0.5, 0.68])  # i_d, i_q, epsilon
    u_dq = np.array([50, 200])
    omega = 25
    # set initializer parameters
    test_initializer['states'] = states
    test_initializer['interval'] = interval
    test_initializer['random_init'] = random_init
    test_initializer['random_params'] = random_params
    # test default initialization first
    default_init_1 = make_module(ElectricMotor, motor_type)
    default_init_2 = motor_class()
    # test parameters
    assert default_init_1.motor_parameter == default_init_2.motor_parameter
    assert default_init_1.nominal_values == default_init_2.nominal_values
    assert default_init_1.limits == default_init_2.limits
    # test functions
    mp = default_init_1.motor_parameter
    if motor_type == 'SynRM':
        state_positions = synrm_state_positions
        state_space = synrm_state_space
        mp.update({'psi_p': 0})
    else:
        state_positions = pmsm_state_positions
        state_space = pmsm_state_space
    for motor in [default_init_1, default_init_2]:
        assert abs(motor.torque(state) - torque_testing(state, mp)) < 1E-8
        assert all(motor.i_in(state) == state[0:2])
        ode = motor.electrical_ode(state, u_dq, omega)
        test_ode = synchronous_motor_ode_testing(state, u_dq, omega, mp)
        assert sum(abs(ode - test_ode)) < 1E-8, "Motor ode is wrong: " + str([ode, test_ode])
    # test parametrized motors
    motor_init_1 = make_module(ElectricMotor, motor_type,
                               motor_parameter=parameter['motor_parameter'],
                               nominal_values=parameter['nominal_values'],
                               limit_values=parameter['limit_values'],
                               motor_initializer = test_initializer)

    motor_init_2 = motor_class(motor_parameter=parameter['motor_parameter'],
                               nominal_values=parameter['nominal_values'],
                               limit_values=parameter['limit_values'],
                               motor_initializer=test_initializer)
    for motor in [motor_init_1, motor_init_2]:
        # test motor parameter

        assert motor.motor_parameter == parameter['motor_parameter'], "Different Parameter: " + \
                                                                      str([motor.motor_parameter,
                                                                           parameter['motor_parameter']])
        mp = motor.motor_parameter
        if motor_type == 'SynRM':
            mp.update({'psi_p': 0})
        # test limits
        factor_abc_to_alpha_beta = 1
        for limit_key in motor_init_1.limits.keys():
            if 'i_' in limit_key:
                assert motor_init_1.limits[limit_key] == parameter['limit_values']['i'] or \
                       motor_init_1.limits[limit_key] == factor_abc_to_alpha_beta * parameter['limit_values']['i']
            if 'u_' in limit_key:
                assert motor_init_1.limits[limit_key] == parameter['limit_values']['u'] or \
                       motor_init_1.limits[limit_key] == .5 * factor_abc_to_alpha_beta * parameter['limit_values']['u'] \
                       or motor_init_1.limits[limit_key] == .5 * parameter['limit_values']['u']

            if limit_key in parameter['limit_values'].keys():
                assert motor_init_1.limits[limit_key] == parameter['limit_values'][limit_key]

            if 'i_' in limit_key:
                assert motor_init_1.nominal_values[limit_key] == parameter['nominal_values']['i'] or \
                       motor_init_1.nominal_values[limit_key] == factor_abc_to_alpha_beta \
                       * parameter['nominal_values']['i']
            if 'u_' in limit_key:
                assert motor_init_1.nominal_values[limit_key] == 0.5 * factor_abc_to_alpha_beta * \
                       parameter['nominal_values']['u'] or \
                       motor_init_1.nominal_values[limit_key] == parameter['nominal_values']['u'] \
                       or motor_init_1.nominal_values[limit_key] == .5 * parameter['nominal_values']['u']
            if limit_key in parameter['nominal_values'].keys():
                assert motor_init_1.nominal_values[limit_key] == parameter['nominal_values'][limit_key]
        # test functions
        assert motor.torque(state) == torque_testing(state, mp)
        assert all(motor.i_in(state) == state[0:2])
        ode = motor.electrical_ode(state, u_dq, omega)
        test_ode = synchronous_motor_ode_testing(state, u_dq, omega, mp)
        assert sum(abs(ode - test_ode)) < 1E-8, "Motor ode is wrong: " + str([ode, test_ode])
        #motor_state = ['i_sd', 'i_sq', 'epsilon']
        motor_state = ['i', 'i', 'epsilon']
        nominal_values = parameter['nominal_values']
        motor_reset_testing(motor_init_1, motor_init_2, motor_state,
                            state_positions, state_space, nominal_values)


# endregion

# endregion  tests

# region second version tests

# region ElectricMotor

class TestElectricMotor:
    """
    class for testing ElectricMotor
    """
    # defined values for testing
    _CURRENTS = [1, 2]
    _length_Currents = 2
    _motor_parameter = test_motor_parameter['DcPermEx']['motor_parameter']
    _nominal_values = test_motor_parameter['DcPermEx']['nominal_values']
    _limits = test_motor_parameter['DcPermEx']['limit_values']
    _initializer = test_motor_initializer['DcPermEx']
    state_position = permex_state_positions
    state_space = permex_state_space

    @pytest.mark.parametrize("motor_parameter, result_motor_parameter",
                             [(None, {}), (_motor_parameter, _motor_parameter)])
    @pytest.mark.parametrize("nominal_values, result_nominal_values",
                             [(None, {}), (_nominal_values, _nominal_values)])
    @pytest.mark.parametrize("limits, result_limit_values",
                             [(None, {}), (_limits, _limits)])
    @pytest.mark.parametrize("motor_initializer, result_motor_initializer",
                             [(None, default_test_initializer), (_initializer, _initializer)])
    def test_init(self, motor_parameter, nominal_values, limits, motor_initializer,
                  result_motor_parameter, result_nominal_values,
                  result_limit_values, result_motor_initializer):
        """
        test initialization of ElectricMotor
        :param motor_parameter: possible values for motor parameter
        :param nominal_values: possible values for nominal values
        :param limits: possible values for limit values
        :param motor_initializer: possible values for initializer
        :param result_motor_parameter: expected motor parameter
        :param result_nominal_values: expected nominal values
        :param result_limit_values: expected limit values
        :param result_motor_initializer: expected motor initializer values
        :return:
        """
        # call function to test
        test_object = ElectricMotor(motor_parameter,
                                    nominal_values,
                                    limits,
                                    motor_initializer)
        # verify the expected results
        assert test_object._motor_parameter == test_object.motor_parameter == result_motor_parameter, \
            'unexpected initialization of motor parameter'
        assert test_object._limits == test_object.limits == result_limit_values, 'unexpected initialization of limits'
        assert test_object._nominal_values == test_object.nominal_values == result_nominal_values, \
            'unexpected initialization of nominal values'
        assert test_object._initializer == test_object.initializer == result_motor_initializer, \
            'unexpected initialization of motor initializer'

    def test_reset(self, monkeypatch):
        """
        test reset()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        test_object = ElectricMotor()
        monkeypatch.setattr(ElectricMotor, "CURRENTS", self._CURRENTS)
        # call function to test
        result = test_object.reset(self.state_space, self.state_position)
        # verify the expected results
        assert all(result == np.zeros(self._length_Currents)), 'unexpected state after reset()'


# endregion

# region DcMotor

class TestDcMotor:
    # defined test values
    class_to_test = DcMotor
    _motor_parameter = test_motor_parameter['DcExtEx']['motor_parameter']
    _nominal_values = test_motor_parameter['DcExtEx']['nominal_values']
    _limits = test_motor_parameter['DcExtEx']['limit_values']
    _initializer = test_motor_initializer['DcExtEx']
    state_position = extex_state_positions
    state_space = extex_state_space
    s_motor_parameter = test_motor_parameter['DcExtEx']['motor_parameter']
    default_motor_parameter = class_to_test._default_motor_parameter
    default_nominal_values = class_to_test._default_nominal_values
    default_limits = class_to_test._default_limits
    _currents = np.array([15, 37])

    # counter
    monkey_update_model_counter = 0
    monkey_update_limits_counter = 0
    monkey_super_init_counter = 0

    # test cases for the get_state_space test
    _test_cases = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [-1, 0, 0, 0, -1, 0, 0, 0, -1, 0],
                            [0, -1, 0, 0, -1, 0, 0, 0, 0, -1],
                            [0, 0, -1, 0, 0, -1, -1, 0, 0, 0],
                            [0, 0, 0, -1, 0, -1, 0, -1, 0, 0],
                            [-1, -1, 0, 0, -1, 0, 0, 0, -1, -1],
                            [-1, 0, -1, 0, -1, -1, -1, 0, -1, 0],
                            [-1, 0, 0, -1, -1, -1, 0, -1, -1, 0],
                            [0, -1, -1, 0, -1, -1, -1, 0, 0, -1],
                            [0, -1, 0, -1, -1, -1, 0, -1, 0, -1],
                            [0, 0, -1, -1, 0, -1, -1, -1, 0, 0],
                            [-1, -1, -1, 0, -1, -1, -1, 0, -1, -1],
                            [-1, -1, 0, -1, -1, -1, 0, -1, -1, -1],
                            [-1, 0, -1, -1, -1, -1, -1, -1, -1, 0],
                            [0, -1, -1, -1, -1, -1, -1, -1, 0, -1],
                            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

    def _monkey_update_model(self):
        """
        mock function for _update_module()
        :return:
        """
        self.monkey_update_model_counter += 1

    def _monkey_update_limits(self):
        """
        mock function for _update_limits()
        :return:
        """
        self.monkey_update_limits_counter += 1

    def _monkey_super_init(self, motor_parameter, nominal_values,
                           limits, motor_initializer):
        """
        mock function for super().__init__()
        :param motor_parameter:
        :param nominal_values:
        :param limits:
        :param motor_initializer:
        :return:
        """
        self.monkey_super_init_counter += 1
        assert motor_parameter == self._motor_parameter, 'motor parameter are not passed correctly'
        assert nominal_values == self._nominal_values, 'nominal values are not passed correctly'
        assert limits == self._limits, 'limits are not passed correctly'
        assert motor_initializer == self._initializer, 'initializers are not matching'

    @pytest.mark.parametrize("motor_parameter, result_motor_parameter",
                             [(None, {}), (_motor_parameter, _motor_parameter)])
    @pytest.mark.parametrize("nominal_values, result_nominal_values",
                             [(None, {}), (_nominal_values, _nominal_values)])
    @pytest.mark.parametrize("limits, result_limit_values",
                             [(None, {}), (_limits, _limits)])
    @pytest.mark.parametrize("motor_initializer, result_motor_initializer",
                             [(None, {}), (_initializer, _initializer)])
    def test_init(self, monkeypatch, motor_parameter, nominal_values, limits,
                  motor_initializer, result_motor_parameter, result_nominal_values,
                  result_limit_values, result_motor_initializer):
        """
        test initialization of DcMotor
        :param monkeypatch:
        :param motor_parameter: possible motor parameters
        :param nominal_values: possible nominal values
        :param limits: possible limit values
        :param motor_initializer: possible values for initializer
        :param result_motor_parameter: expected resulting motor parameter
        :param result_nominal_values: expected resulting nominal values
        :param result_limit_values: expected resulting limit values
        :param result_motor_initializer: expected motor initializer values
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(self.class_to_test, "_update_model", self._monkey_update_model)
        monkeypatch.setattr(self.class_to_test, "_update_limits", self._monkey_update_limits)
        monkeypatch.setattr(ElectricMotor, "__init__", self._monkey_super_init)

        self._motor_parameter = motor_parameter
        self._nominal_values = nominal_values
        self._limits = limits
        self._initializer = motor_initializer

        # call function to test
        test_object = self.class_to_test(motor_parameter, nominal_values,
                                         limits, motor_initializer)

        # verify the expected results
        assert test_object._model_constants is None
        assert self.monkey_super_init_counter == 1, 'super().__init__() is not called once'
        assert self.monkey_update_limits_counter == 1, 'update_limits() is not called once'
        assert self.monkey_update_model_counter == 1, 'update_model() is not called once'

    def test_torque(self, monkeypatch):
        """
        test torque()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        test_object = self.class_to_test()
        monkeypatch.setattr(test_object, "_motor_parameter", self._motor_parameter)
        # call function to test
        torque = test_object.torque(self._currents)
        # verify the expected results
        assert torque == 0.95 * 15 * 37, 'unexpected torque value'

    def test_i_in(self):
        """
        test i_in()
        :return:
        """
        # setup test scenario
        test_object = self.class_to_test()
        # call function to test
        i_in = test_object.i_in(self._currents)
        # verify the expected results
        assert i_in == list(self._currents), 'unexpected current in the motor'

    def test_electrical_ode(self):
        # test electrical_ode()
        # setup test scenario
        test_object = self.class_to_test(motor_parameter=self.s_motor_parameter)
        u_in = np.array([10, 15])
        omega = 38
        # call function to test
        state = test_object.electrical_ode(self._currents, u_in, omega)
        # verify the expected results
        assert all(abs(state - np.array([-219428.5714286, -8000.0])) < np.ones(2) * 1E-7), 'unexpected ode value'

    @pytest.mark.parametrize("test_case", _test_cases)
    def test_get_state_space(self, test_case):
        """
        test get_state_space()
        :param test_case: definition of state space limits for different test cases
        :return:
        """
        # setup test scenario
        u_1_min = test_case[0]
        u_2_min = test_case[1]
        i_1_min = test_case[2]
        i_2_min = test_case[3]
        result_high = dict(omega=1, torque=1, i_a=1, i_e=1, u_a=1, u_e=1)
        result_low = dict(omega=test_case[4], torque=test_case[5], i_a=test_case[6], i_e=test_case[7], u_a=test_case[8],
                          u_e=test_case[9])

        input_currents = Box(np.array([i_1_min, i_2_min]), np.array([1, 1]))
        input_voltages = Box(np.array([u_1_min, u_2_min]), np.array([1, 1]))
        test_object = self.class_to_test()

        # call function to test
        low, high = test_object.get_state_space(input_currents, input_voltages)
        # verify the expected results
        assert low == result_low, "Different lower limits: " + str([low, result_low])
        assert high == result_high, "Different higher limits: " + str([high, result_high])

    def test_update_model(self):
        """
        test _update_model()
        :return:
        """
        # setup test scenario
        test_object = self.class_to_test()
        test_object._motor_parameter = extex_motor_parameter['motor_parameter']
        # call function to test
        test_object._update_model()
        # verify results
        expected_constants = np.array(
            [[-3.78 / 6.3e-3, 0, -0.95 / 6.3e-3, 1 / 6.3e-3, 0], [0, -35 / 160e-3, 0, 0, 1 / 160e-3]])
        assert sum(sum(abs(test_object._model_constants - expected_constants))) < 1E-6, 'unexpected model constants'


class TestDcShuntMotor:
    """
    class for testing DcShuntMotor
    """
    class_to_test = DcShuntMotor
    # defined test values
    _currents = np.array([15, 37])
    u_in = [10]
    omega = 38
    state = np.array([42, 36])
    # test cases for the get_state_space test
    _test_cases = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [-1, 0, 0, 0, 0, 0, -1],
                            [0, -1, 0, -1, -1, -1, 0],
                            [-1, -1, 0, -1, -1, -1, -1]])

    # counter
    monkey_super_electrical_ode_counter = 0

    def monkey_super_electrical_ode(self, currents, u_in, omega):
        """
        mock function for super().electrical_ode()
        :param currents: currents in the motor
        :param u_in: applied voltage
        :param omega: speed of the motor
        :return:
        """
        self.monkey_super_electrical_ode_counter += 1
        assert all(currents == self._currents), 'unexpected currents passed'
        assert type(u_in) is tuple, 'type of u_in is not tuple'
        assert u_in[0] == u_in[1] == self.u_in[0], 'unexpected u_in passed'
        assert omega == self.omega, 'unexpected omega passed'
        return self.state

    def test_i_in(self):
        """
        test i_in()
        :return:
        """
        # setup test scenario
        test_object = DcShuntMotor()
        # call function to test
        i_in = test_object.i_in(self._currents)
        # verify the expected results
        assert i_in == [self._currents[0] + self._currents[1]], 'unexpected current in the motor'

    def test_electrical_ode(self, monkeypatch):
        """
         Testing the whole ode system, if the result is correct
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        motor_parameter = test_motor_parameter['DcShunt']['motor_parameter']
        test_object = DcShuntMotor(motor_parameter=motor_parameter)
        # call function to test
        state = test_object.electrical_ode(self._currents, self.u_in, self.omega)
        # verify the expected results
        resulting_state = np.array([-219428.5714286, -8031.25])
        assert all(abs(resulting_state - state) < 1E-7 * np.ones(2)), 'unexpected ode result'

        # Test if the internal function is called with the correct arguments
        # setup test scenario
        monkeypatch.setattr(DcMotor, "electrical_ode", self.monkey_super_electrical_ode)
        # call function to test
        resulting_state = test_object.electrical_ode(self._currents, self.u_in, self.omega)
        # verify the expected results
        assert all(resulting_state == self.state), 'unexpected resulting state'
        assert self.monkey_super_electrical_ode_counter == 1, 'super()._electrical_ode() is not called once'

    @pytest.mark.parametrize("test_case", _test_cases)
    def test_get_state_space(self, test_case):
        """
        test get_state_space()
        :param test_case: definition of state space limits for different test cases
        :return:
        """
        # setup test scenario
        u_converter_min = test_case[0]
        i_converter_min = test_case[1]
        omega = test_case[2]
        torque = test_case[3]
        i_a_min = test_case[4]
        i_e_min = test_case[5]
        u_min = test_case[6]
        result_low = dict(omega=omega, torque=torque, i_a=i_a_min, i_e=i_e_min, u=u_min)
        result_high = dict(omega=1, torque=1, i_a=1, i_e=1, u=1)
        test_object = DcShuntMotor()
        converter_currents = Box(i_converter_min, 1, shape=(1,))
        converter_voltages = Box(u_converter_min, 1, shape=(1,))
        # call function to test
        low, high = test_object.get_state_space(converter_currents, converter_voltages)
        # verify the expected results
        assert low == result_low, 'unexpected lower state space'
        assert high == result_high, 'unexpected upper state space'

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result',[
            (np.array([0, 1]), [0], 0, dict(r_a=10, l_a=2, l_e_prime=0.1, r_e=0.5, l_e=0.5),
             np.array([[-5, 0], [0, -1]])),
            (np.array([5, 7]), [4], -2, dict(r_a=10, l_a=2, l_e_prime=0.1, r_e=0.5, l_e=0.5),
             np.array([[-5, 0.1], [0, -1]])),
            (np.array([5, 7]), [4], 2, dict(r_a=0, l_a=2, l_e_prime=0.1, r_e=2, l_e=0.5),
             np.array([[0, -0.1], [0, -4]])),
        ]
    )
    def test_el_jac_0(self, state, u_in, omega, motor_parameter, result):
        # Test first return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        el_jac, _, _ = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(el_jac == result)

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result',[
            (np.array([0, 2]), [0], 0, dict(r_a=10, l_a=0.5, l_e_prime=0.1, r_e=0.5, l_e=0.5),
             np.array([-0.4, 0])),
            (np.array([5, 0]), [4], -2, dict(r_a=10, l_a=2, l_e_prime=0.1, r_e=0.5, l_e=0.5),
             np.array([0, 0])),
            (np.array([-2, -2]), [8], 2, dict(r_a=0, l_a=0.2, l_e_prime=2, r_e=2, l_e=0.5),
             np.array([20, 0])),
        ]
    )
    def test_el_jac_1(self, state, u_in, omega, motor_parameter, result):
        # Test second return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        _, el_over_omega, _ = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(el_over_omega == result)

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result',[
            (np.array([0, 1]), [0], 0, dict(r_a=10, l_a=0.5, l_e_prime=0.1, r_e=0.5, l_e=0.5),
             np.array([0.1, 0])),
            (np.array([5, 0]), [4], -2, dict(r_a=10, l_a=2, l_e_prime=0.1, r_e=0.5, l_e=0.5),
             np.array([0, 0.5])),
            (np.array([-2, -2]), [8], 2, dict(r_a=0, l_a=0.2, l_e_prime=2, r_e=2, l_e=0.5),
             np.array([-4, -4])),
        ]
    )
    def test_el_jac_2(self, state, u_in, omega, motor_parameter, result):
        # Test second return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        _, _, torque_over_el = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(torque_over_el == result)


class TestDcSeriesMotor:
    """
    class for testing DcSeriesMotor
    """
    # defined test values
    _current = 15
    _u_in = 10
    _omega = 38
    _torque = 150
    _state = np.array([_current])
    _motor_parameter = test_motor_parameter['DcSeries']['motor_parameter']
    class_to_test = DcSeriesMotor
    # counter
    monkey_super_torque_counter = 0

    # test cases for get_state_space_testing
    _test_cases = np.array([[0, 0, 0, 0, 0, 0],
                            [-1, 0, 0, 0, 0, -1],
                            [0, -1, 0, 0, -1, 0],
                            [-1, -1, 0, 0, -1, -1]])

    def monkey_super_torque(self, currents):
        """
        mock function for super().torque()
        :param currents:
        :return:
        """
        self.monkey_super_torque_counter += 1
        assert currents[0] == currents[1] == self._current, 'unexpected currents passed'
        return self._torque

    def test_torque(self, monkeypatch):
        """
        test torque()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        test_object = DcSeriesMotor(motor_parameter=self._motor_parameter)
        # call function to test
        torque = test_object.torque([self._current])
        # verify the expected results
        assert torque == 0.95 * 15 * 15, 'unexpected torque calculated'

        # setup test scenario
        monkeypatch.setattr(DcMotor, "torque", self.monkey_super_torque)
        # call function to test
        torque = test_object.torque(([self._current]))
        # verify the expected results
        assert torque == self._torque, 'unexpected torque calculated'
        assert self.monkey_super_torque_counter == 1, 'super().torque() is not called once'

    def test_electrical_ode(self):
        """
        test electrical_ode()
        :return:
        """
        # setup test scenario
        test_object = DcSeriesMotor(motor_parameter=self._motor_parameter)
        # call function to test
        state = test_object.electrical_ode([self._current], [self._u_in], self._omega)
        # verify the expected results
        expected_state = np.array([-6693.926639])
        assert abs(state - expected_state) < 1E-6, 'unexpected result of the electrical ode'

    def test_i_in(self):
        """
        test i_in()
        :return:
        """
        # setup test scenario
        test_object = DcSeriesMotor()
        # call function to test
        i_in = test_object.i_in(self._state)
        # verify the expected results
        assert i_in == self._current, 'unexpected current in the motor'

    @pytest.mark.parametrize("test_case", _test_cases)
    def test_get_state_space(self, test_case):
        """
        test get_state_space()
        :param test_case: definition of state space limits for different test cases
        :return:
        """
        # setup test scenario
        u_converter_min = test_case[0]
        i_converter_min = test_case[1]
        omega = test_case[2]
        torque = test_case[3]
        i = test_case[4]
        u = test_case[5]
        expected_low = dict(omega=omega, torque=torque, i=i, u=u)
        expected_high = dict(omega=1, torque=1, i=1, u=1)

        converter_currents = Box(i_converter_min, 1, shape=(1,))
        converter_voltages = Box(u_converter_min, 1, shape=(1,))
        test_object = DcSeriesMotor()
        # call function to test
        low, high = test_object.get_state_space(converter_currents, converter_voltages)
        # verify the expected results
        assert low == expected_low, 'unexpected lower state space'
        assert high == expected_high, 'unexpected upper state space'

    def test_update_model(self):
        """
        test _update_model()
        :return:
        """
        # setup test scenario
        test_object = DcSeriesMotor()
        test_object._motor_parameter = series_motor_parameter['motor_parameter']
        # call function to test
        test_object._update_model()
        # verify results
        expected_constants = np.array(
            [-(3.78 + 35), -0.95, 1]) / (6.3e-3 + 160e-3)
        assert sum(sum(abs(test_object._model_constants - expected_constants))) < 1E-6, 'unexpected model constants'

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result', [
            (np.array([1]), [0], 60, dict(r_a=10, l_a=3.5, l_e_prime=0.1, r_e=2, l_e=2.5),
             np.array([[-3]])),
            (np.array([7]), [4], -30, dict(r_a=10, l_a=3.5, l_e_prime=0.1, r_e=2, l_e=2.5),
             np.array([[-1.5]])),
            (np.array([5]), [8], 0, dict(r_a=10, l_a=3.5, l_e_prime=0.1, r_e=2, l_e=2.5),
             np.array([[-2]])),
        ]
    )
    def test_el_jac_0(self, state, u_in, omega, motor_parameter, result):
        # Test first return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        el_jac, _, _ = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(el_jac == result)

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result', [
            (np.array([2]), [0], 0, dict(r_a=10, l_a=1.5, l_e_prime=0.1, r_e=0.5, l_e=0.5),
             np.array([-0.1])),
            (np.array([0]), [4], -2, dict(r_a=10, l_a=2, l_e_prime=0.1, r_e=0.5, l_e=0.5),
             np.array([0])),
            (np.array([-3]), [8], 2, dict(r_a=0, l_a=4, l_e_prime=2, r_e=2, l_e=8),
             np.array([0.5])),
        ]
    )
    def test_el_jac_1(self, state, u_in, omega, motor_parameter, result):
        # Test second return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        _, el_over_omega, _ = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(el_over_omega == result)

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result', [
            (np.array([10]), [2], 0, dict(r_a=10, l_a=0.5, l_e_prime=0.2, r_e=0.5, l_e=0.5),
             np.array([4])),
            (np.array([-30]), [8], -2, dict(r_a=10, l_a=2, l_e_prime=0.1, r_e=0.5, l_e=0.5),
             np.array([-6])),
            (np.array([0]), [4], 3, dict(r_a=0, l_a=0.2, l_e_prime=2, r_e=2, l_e=0.5),
             np.array([0])),
        ]
    )
    def test_el_jac_2(self, state, u_in, omega, motor_parameter, result):
        # Test second return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        _, _, torque_over_el = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(torque_over_el == result)


class TestDcPermanentlyExcitedMotor:
    """
    class for testing DcPermanentlyExcitedMotor
    """
    # defined test values
    _motor_parameter = test_motor_parameter['DcPermEx']['motor_parameter']
    _current = 15
    _u_in = 10
    _omega = 38
    _state = np.array([_current])
    class_to_test = DcPermanentlyExcitedMotor
    # test cases for get_state_space test
    _test_cases = np.array([[0, 0, 0, 0, 0, 0],
                            [-1, 0, -1, 0, 0, -1],
                            [0, -1, 0, -1, -1, 0],
                            [-1, -1, -1, -1, -1, -1]])

    def test_torque(self):
        """
        test torque()
        :return:
        """
        # setup test scenario
        test_object = DcPermanentlyExcitedMotor(motor_parameter=self._motor_parameter)
        # call function to test
        torque = test_object.torque([self._current])
        # verify the expected results
        assert torque == 0.16 * 15, 'unexpected torque calculated'

    def test_i_in(self):
        """
        test i_in()
        :return:
        """
        # call function to test
        i_in = DcPermanentlyExcitedMotor().i_in(self._state)
        # verify the expected results
        assert i_in == self._current, 'unexpected currents in teh motor'

    def test_electrical_ode(self):
        """
        test electrical_ode()
        :return:
        """
        # setup test scenario
        test_object = DcPermanentlyExcitedMotor(motor_parameter=self._motor_parameter)
        # call function to test
        state = test_object.electrical_ode(self._state, [self._u_in], self._omega)
        # verify the expected results
        expected_state = np.array([-8377.777778])
        assert abs(state - expected_state) < 1E-6, 'unexpected result of the electrical ode'

    @pytest.mark.parametrize("test_case", _test_cases)
    def test_get_state_space(self, test_case):
        """
        test get_state_space()
        :param test_case: definition of state space limits for different test cases
        :return:
        """
        # setup test scenario
        u_converter_min = test_case[0]
        i_converter_min = test_case[1]
        omega = test_case[2]
        torque = test_case[3]
        i = test_case[4]
        u = test_case[5]
        expected_low = dict(omega=omega, torque=torque, i=i, u=u)
        expected_high = dict(omega=1, torque=1, i=1, u=1)

        converter_currents = Box(i_converter_min, 1, shape=(1,))
        converter_voltages = Box(u_converter_min, 1, shape=(1,))
        test_object = DcPermanentlyExcitedMotor()
        # call function to test
        low, high = test_object.get_state_space(converter_currents, converter_voltages)
        # verify the expected results
        assert low == expected_low, 'unexpected lower state space'
        assert high == expected_high, 'unexpected upper state space'

    def test_update_model(self):
        """
        test _update_model()
        :return:
        """
        # setup test scenario
        test_object = DcPermanentlyExcitedMotor()
        test_object._motor_parameter = permex_motor_parameter['motor_parameter']
        # call function to test
        test_object._update_model()
        # verify results
        expected_constants = np.array([-160e-3, -3.78, 1]) / 6.3e-3
        assert sum(sum(abs(test_object._model_constants - expected_constants))) < 1E-6, 'unexpected model constants'

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result', [
            (np.array([1]), [0], 60, dict(r_a=10, l_a=0.5, psi_e=0.2, r_e=0.5, l_e=0.5),
             np.array([[-20]])),
            (np.array([7]), [4], -30, dict(r_a=0, l_a=0.5, psi_e=0.2, r_e=0.5, l_e=0.5),
             np.array([[0]])),
            (np.array([5]), [8], 0, dict(r_a=30, l_a=0.5, psi_e=0.2, r_e=0.5, l_e=0.5),
             np.array([[-60]])),
        ]
    )
    def test_el_jac_0(self, state, u_in, omega, motor_parameter, result):
        # Test first return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        el_jac, _, _ = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(el_jac == result)

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result', [
            (np.array([2]), [0], 0, dict(r_a=10, l_a=2, psi_e=0.2, r_e=0.5, l_e=0.5),
             np.array([-0.1])),
            (np.array([0]), [4], -2, dict(r_a=10, l_a=4.5, psi_e=0, r_e=0.5, l_e=0.5),
             np.array([0])),
            (np.array([-3]), [8], 2, dict(r_a=10, l_a=0.5, psi_e=5, r_e=0.5, l_e=0.5),
             np.array([-10])),
        ]
    )
    def test_el_jac_1(self, state, u_in, omega, motor_parameter, result):
        # Test second return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        _, el_over_omega, _ = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(el_over_omega == result)

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result', [
            (np.array([10]), [2], 0, dict(r_a=10, l_a=0.5, psi_e=0.2, r_e=0.5, l_e=0.5),
             np.array([0.2])),
            (np.array([-30]), [8], -2, dict(r_a=10, l_a=0.5, psi_e=0.0, r_e=0.5, l_e=0.5),
             np.array([0])),
            (np.array([0]), [4], 3, dict(r_a=10, l_a=0.5, psi_e=30, r_e=0.5, l_e=0.5),
             np.array([30])),
        ]
    )
    def test_el_jac_2(self, state, u_in, omega, motor_parameter, result):
        # Test second return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        _, _, torque_over_el = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(torque_over_el == result)


class TestDcExternallyExcitedMotor(TestDcMotor):
    class_to_test = DcExternallyExcitedMotor

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result',[
            (np.array([0, 1]), [0, 2], 0, dict(r_a=10, l_a=2, l_e_prime=0.1, r_e=0.5, l_e=0.5),
             np.array([[-5, 0], [0, -1]])),
            (np.array([5, 7]), [4, 8], -2, dict(r_a=10, l_a=2, l_e_prime=0.1, r_e=0.5, l_e=0.5),
             np.array([[-5, 0.1], [0, -1]])),
            (np.array([5, 7]), [4, 8], 2, dict(r_a=0, l_a=2, l_e_prime=0.1, r_e=2, l_e=0.5),
             np.array([[0, -0.1], [0, -4]])),
        ]
    )
    def test_el_jac_0(self, state, u_in, omega, motor_parameter, result):
        # Test first return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        el_jac, _, _ = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(el_jac == result)

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result',[
            (np.array([0, 2]), [0, 2], 0, dict(r_a=10, l_a=0.5, l_e_prime=0.1, r_e=0.5, l_e=0.5),
             np.array([-0.4, 0])),
            (np.array([5, 0]), [4, 8], -2, dict(r_a=10, l_a=2, l_e_prime=0.1, r_e=0.5, l_e=0.5),
             np.array([0, 0])),
            (np.array([-2, -2]), [4, 8], 2, dict(r_a=0, l_a=0.2, l_e_prime=2, r_e=2, l_e=0.5),
             np.array([20, 0])),
        ]
    )
    def test_el_jac_1(self, state, u_in, omega, motor_parameter, result):
        # Test second return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        _, el_over_omega, _ = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(el_over_omega == result)

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result',[
            (np.array([0, 1]), [0, 2], 0, dict(r_a=10, l_a=0.5, l_e_prime=0.1, r_e=0.5, l_e=0.5),
             np.array([0.1, 0])),
            (np.array([5, 0]), [4, 8], -2, dict(r_a=10, l_a=2, l_e_prime=0.1, r_e=0.5, l_e=0.5),
             np.array([0, 0.5])),
            (np.array([-2, -2]), [4, 8], 2, dict(r_a=0, l_a=0.2, l_e_prime=2, r_e=2, l_e=0.5),
             np.array([-4, -4])),
        ]
    )
    def test_el_jac_2(self, state, u_in, omega, motor_parameter, result):
        # Test second return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        _, _, torque_over_el = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(torque_over_el == result)





# endregion

# region Three Phase Motors

class TestThreePhaseMotor:
    """
        class for testing ThreePhaseMotor
    """
    # transformation from abc to dq
    _forward_quantities_abc = np.array([60, 45, 150])
    _forward_quantities_alpha_beta = np.array([-25, -60.621778])
    _forward_quantities_dq = np.array([60.54374003, 25.18840097])
    _forward_epsilon = 3 / 4 * np.pi
    _forward_epsilon_me = _forward_epsilon / 3

    # transformation from dq to abc
    _backward_quantities_abc = np.array([3.535533906, -17.0770778, 13.54154394])
    _backward_quantities_alpha_beta = np.array([3.535533906, -17.6776695])
    _backward_quantities_dq = np.array([-15, 10])
    _backward_epsilon = 5 / 4 * np.pi
    _backward_epsilon_me = _backward_epsilon / 3

    # define expected values
    _expected_epsilon = None
    _expected_quantities = None
    _expected_result = None
    _expected_parameter = None
    #_expected_initializer = None

    # defined test values
    _motor_parameter = pmsm_motor_parameter['motor_parameter']

    # counter
    _monkey_q_counter = 0

    def monkey_q(self, quantities, epsilon):
        """
        mock function for q()
        :param quantities:
        :param epsilon:
        :return:
        """
        self._monkey_q_counter += 1
        assert epsilon == self._expected_epsilon, 'unexpected angle epsilon. It must be the electrical one.'
        assert all(quantities == self._expected_quantities), 'unexpected quantities. Alpha and beta are needed.'
        return self._expected_result

    def test_t_23(self):
        """
        test t_23()
        :return:
        """
        assert sum(
            abs(ThreePhaseMotor.t_23(self._forward_quantities_abc) - self._forward_quantities_alpha_beta)) < 1E-6, \
            'unexpected calculation from abc to alpha-beta'

    def test_t_32(self):
        """
        test t_32()
        :return:
        """
        assert sum(
            abs(ThreePhaseMotor.t_32(self._backward_quantities_alpha_beta) - self._backward_quantities_abc)) < 1E-6, \
            'unexpected calculation from alpha-beta to abc'

    def test_q(self):
        """
        test q()
        :return:
        """
        assert sum(abs(ThreePhaseMotor.q(self._forward_quantities_alpha_beta,
                                          self._forward_epsilon) - self._forward_quantities_dq)) < 1E-6, \
            'unexpected calculation from alpha-beta to dq'

    def test_q_inv(self, monkeypatch):
        """
        test q_inv()
        :param monkeypatch:
        :return:
        """
        # test if the resulting value is correct
        assert sum(abs(ThreePhaseMotor.q_inv(self._backward_quantities_dq,
                                              self._backward_epsilon) - self._backward_quantities_alpha_beta)) < 1E-6, \
            'unexpected calculation from dq to alpha-beta'
        # test if the internal function is called correctly
        monkeypatch.setattr(ThreePhaseMotor, "q", self.monkey_q)
        # setup test scenario
        self._expected_epsilon = -self._backward_epsilon
        self._expected_quantities = self._backward_quantities_dq
        self._expected_result = self._backward_quantities_alpha_beta
        # call function to test
        ThreePhaseMotor.q_inv(self._backward_quantities_dq, self._backward_epsilon)
        # verify the expected results
        assert self._monkey_q_counter == 1, "q function was not called correctly"

    def test_q_me(self, monkeypatch):
        """
        test q_me()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        test_object = ThreePhaseMotor(motor_parameter=self._motor_parameter)
        # call function to test and verify the expected results
        assert sum(abs(test_object.q_me(self._forward_quantities_alpha_beta,
                                        self._forward_epsilon_me) - self._forward_quantities_dq)) < 1E-6, \
            'unexpected result from alpha-beta to dq. Mechanical angle needed.'
        # setup test scenario
        monkeypatch.setattr(ThreePhaseMotor, 'q', self.monkey_q)
        self._expected_epsilon = self._forward_epsilon
        self._expected_quantities = self._forward_quantities_alpha_beta
        self._expected_result = self._forward_quantities_dq
        # call function to test
        test_object.q_me(self._forward_quantities_alpha_beta, self._forward_epsilon_me)
        # verify the expected results
        assert self._monkey_q_counter == 1, "q function was not called correctly"

    def test_q_inv_me(self, monkeypatch):
        """
        test q_inv_me()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        test_object = ThreePhaseMotor(motor_parameter=self._motor_parameter)
        # call function to test verify the expected results
        assert sum(abs(test_object.q_inv_me(self._forward_quantities_dq,
                                            self._forward_epsilon_me) - self._forward_quantities_alpha_beta)) < 1E-6, \
            'unexpected result from dq to alpha-beta. Mechanical angle needed.'
        # setup test scenario
        monkeypatch.setattr(ThreePhaseMotor, 'q', self.monkey_q)
        self._expected_epsilon = -self._forward_epsilon
        self._expected_quantities = self._forward_quantities_dq
        self._expected_result = self._forward_quantities_alpha_beta
        # call function to test
        test_object.q_inv_me(self._forward_quantities_dq, self._forward_epsilon_me)
        # verify the expected results
        assert self._monkey_q_counter == 1, "q function was not called correctly"


class TestSynchronousMotor(TestThreePhaseMotor):
    """
    class for testing SynchronousMotor
    """
    # defined test values
    _motor_parameter = pmsm_motor_parameter['motor_parameter']
    _p = _motor_parameter['p']  # pole pair number for testing
    _nominal_values = pmsm_motor_parameter['nominal_values']
    _limit_values = pmsm_motor_parameter['limit_values']
    _initializer = pmsm_initializer
    state_position = pmsm_state_positions
    state_space = pmsm_state_space
    _CURRENTS = ['i_a', 'i_b', 'i_c']
    _CURRENTS_IDX = [2, 3, 5]
    _number_states = 4

    # counter
    _monkey_update_model_counter = 0
    _monkey_update_limits_counter = 0
    _monkey_super_init_counter = 0

    def monkey_update_model(self):
        """
        mock function for _update_model()
        :return:
        """
        self._monkey_update_model_counter += 1

    def monkey_update_limits(self):
        """
        mock function for _update_limits()
        :return:
        """
        self._monkey_update_limits_counter += 1

    def monkey_super_init(self, motor_parameter=None, nominal_values=None,
                          limit_values=None, motor_initializer=None):
        """
        mock function for super().__init__()
        :param motor_parameter:
        :param nominal_values:
        :param limit_values:
        :param motor_initializer:
        :return:
        """
        self._monkey_super_init_counter += 1
        assert self._expected_parameter['motor_parameter'] == motor_parameter, 'unexpected motor parameter passed'
        assert self._expected_parameter['nominal_values'] == nominal_values, 'unexpected nominal values passed'
        assert self._expected_parameter['limit_values'] == limit_values, 'unexpected limit values passed'
        assert self._expected_parameter['motor_initializer'] == motor_initializer, 'unexpected initializer'

    @pytest.mark.parametrize("motor_parameter", [_motor_parameter, None])
    @pytest.mark.parametrize("nominal_values, expected_nv", [(None, {}), (_nominal_values, _nominal_values)])
    @pytest.mark.parametrize("limit_values, expected_lv", [(None, {}), (_limit_values, _limit_values)])
    @pytest.mark.parametrize("motor_initializer", [_initializer, {}])
    def test_init(self, monkeypatch, motor_parameter, nominal_values,
                  limit_values, expected_nv, expected_lv, motor_initializer):
        """
        test initialization of SynchronousMotor
        :param monkeypatch:
        :param setup: fixture that is called before the function
        :param motor_parameter: possible motor parameters
        :param nominal_values: possible nominal values
        :param limit_values: possible limit values
        :param expected_nv: expected nominal values
        :param expected_lv: expected limit values
        :param motor_initializer: possible motor initializers
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(SynchronousMotor, "_update_model", self.monkey_update_model)
        monkeypatch.setattr(SynchronousMotor, "_update_limits", self.monkey_update_limits)
        monkeypatch.setattr(ElectricMotor, '__init__', self.monkey_super_init)

        self._expected_parameter = dict(motor_parameter=motor_parameter, nominal_values=expected_nv,
                                        limit_values=expected_lv, motor_initializer=motor_initializer)
        # call function to test
        test_object = SynchronousMotor(motor_parameter, nominal_values,
                                       limit_values, motor_initializer)
        # verify the expected results
        assert self._monkey_update_limits_counter == 1, 'update_limits() is not called once'
        assert self._monkey_super_init_counter == 1, 'super().__init__() is not called once'
        assert self._monkey_update_model_counter == 1, 'update_model() is not called once'

    def test_reset(self, monkeypatch):
        """
        test reset()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(SynchronousMotor, '__init__', self.monkey_update_model)
        monkeypatch.setattr(SynchronousMotor, 'CURRENTS', self._CURRENTS)
        #monkeypatch.setattr(SynchronousMotor, '_initializer', self._initializer)
        test_object = SynchronousMotor()
        # call function to test
        result = test_object.reset(self.state_position, self.state_space)
        # verify the expected results
        assert all(result == np.zeros(self._number_states)), 'unexpected state after reset()'

    def test_electrical_ode(self, monkeypatch):
        """
        test electrical_ode()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(SynchronousMotor, '__init__', self.monkey_update_model)
        # use the identity matrix. Therefore, only the states and their multiplications are returned
        monkeypatch.setattr(SynchronousMotor, '_model_constants', np.identity(7))
        test_object = SynchronousMotor()
        state = np.array([25, 36, -15])
        u_qd = np.array([-400, 325])
        omega = 42
        expected_result = np.array([42, 25, 36, -400, 325, 42 * 25, 42 * 36])
        # call function to test
        result = test_object.electrical_ode(state, u_qd, omega)
        # verify the expected results
        assert all(result == expected_result), 'unexpected result of the electrical_ode()'

    def test_i_in(self, monkeypatch):
        """
        test i_in()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(SynchronousMotor, '__init__', self.monkey_update_model)
        monkeypatch.setattr(SynchronousMotor, 'CURRENTS_IDX', self._CURRENTS_IDX)
        test_object = SynchronousMotor()
        state = np.array([25, 10, 24, 36, 25, 50, 12])
        expected_result = np.array([24, 36, 50])
        # call function to test
        result = test_object.i_in(state)
        # verify the expected results
        assert all(result == expected_result), 'unexpected currents in the motor'


class TestSynchronousReluctanceMotor:
    """
    class for testing SynchronousReluctanceMotor
    """
    class_to_test = SynchronousReluctanceMotor

    def test_torque(self, monkeypatch):
        """
        test torque()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        _motor_parameter = synrm_motor_parameter['motor_parameter']
        test_object = SynchronousReluctanceMotor()
        monkeypatch.setattr(test_object, '_motor_parameter', _motor_parameter)
        currents = np.array([15, 10])
        # call function to test
        torque = test_object.torque(currents)
        # verify the expected results
        expected_torque = 41.85
        assert abs(torque - expected_torque) < 1E-6, 'unexpected torque calculated'

    def test_update_model(self):
        """
        test _update_model()
        :return:
        """
        # setup test scenario
        test_object = SynchronousReluctanceMotor()
        test_object._motor_parameter = synrm_motor_parameter['motor_parameter']
        # call function to test
        test_object._update_model()
        # verify results
        expected_constants = np.array([
            [0, -0.5 / 70E-3,           0, 1 / 70E-3,        0,           0, 3 * 8 / 70],
            [0,            0, -0.5 / 8E-3,         0, 1 / 8E-3, -3 * 70 / 8,          0],
            [3,            0,           0,         0,        0,           0,          0]
        ])
        print(test_object._model_constants)
        print(expected_constants)
        assert sum(sum(abs(test_object._model_constants - expected_constants))) < 1E-6, 'unexpected model constants'

    @pytest.mark.parametrize(
        'state,                  u_in, omega,                                  motor_parameter,                                            result',[
        (np.array([1, 0, 2]),  [2, 0],     0, dict(p=2, l_d=10, l_q=2, j_rotor=2.45e-3, r_s=5), np.array([[-0.5, 0, 0], [0, -2.5, 0], [0, 0, 0]])),
        (np.array([7, 5, -3]), [8, 4],    -2, dict(p=2, l_d=10, l_q=2, j_rotor=2.45e-3, r_s=2), np.array([[-0.2, -0.8, 0], [20, -1, 0], [0, 0, 0]])),
        (np.array([7, 5, -2]), [8, 4],     2, dict(p=2, l_d=10, l_q=2, j_rotor=2.45e-3, r_s=5), np.array([[-0.5, 0.8, 0], [-20, -2.5, 0], [0, 0, 0]])),
        ]
    )
    def test_el_jac_0(self, state, u_in, omega, motor_parameter, result):
        # Test first return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        el_jac, _, _ = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(el_jac == result)

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result',[
            (np.array([2, 0, 0]), [2, 0], 0, dict(p=2, l_d=10, l_q=2, j_rotor=2.45e-3, r_s=5), np.array([0, -20, 2])),
            (np.array([0, 5, 2]), [8, 4], -2, dict(p=1, l_d=5, l_q=2, j_rotor=2.45e-3, r_s=5), np.array([2, 0, 1])),
            (np.array([2, -2, 9]), [4, 8], 2, dict(p=4, l_d=10, l_q=2, j_rotor=2.45e-3, r_s=5), np.array([-1.6, -40, 4])),
        ]
    )
    def test_el_jac_1(self, state, u_in, omega, motor_parameter, result):
        # Test second return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        _, el_over_omega, _ = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(el_over_omega == result)

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result',[
            (np.array([2, 0, 0]), [2, 0], 0, dict(p=2, l_d=10, l_q=2, j_rotor=2.45e-3, r_s=5), np.array([0, 48, 0])),
            (np.array([0, 5, 2]), [8, 4], -2, dict(p=2, l_d=5, l_q=2, j_rotor=2.45e-3, r_s=5), np.array([45, 0, 0])),
            (np.array([2, -2, 9]), [8, 4], 2, dict(p=2, l_d=10, l_q=2, j_rotor=2.45e-3, r_s=5), np.array([-48, 48, 0])),
        ]
    )
    def test_el_jac_2(self, state, u_in, omega, motor_parameter, result):
        # Test second return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        _, _, torque_over_el = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(torque_over_el == result)


class TestPermanentMagnetSynchronousMotor:
    """
    class for testing PermanentMagnetSynchronousMotor
    """
    class_to_test = PermanentMagnetSynchronousMotor

    def test_torque(self, monkeypatch):
        """
        test torque()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        _motor_parameter = pmsm_motor_parameter['motor_parameter']
        test_object = PermanentMagnetSynchronousMotor()
        monkeypatch.setattr(test_object, '_motor_parameter', _motor_parameter)
        currents = np.array([10, 15])
        # call function to test
        torque = test_object.torque(currents)
        # verify the expected results
        expected_torque = -16.1325
        assert abs(torque - expected_torque) < 1E-6, 'unexpected torque calculated'

    def test_update_model(self):
        """
        test _update_model()
        :return:
        """
        # setup test scenario
        test_object = PermanentMagnetSynchronousMotor()
        test_object._motor_parameter = pmsm_motor_parameter['motor_parameter']
        # call function to test
        test_object._update_model()
        # verify results
        expected_constants = np.array([
            [                0, -5 / 84E-3,            0, 1 / 84E-3,          0,                    0, 3 * 125 / 84],
            [-3 * 0.171/125E-3,           0, -5 / 125E-3,         0, 1 / 125E-3, -84E-3 * 3 / 125E-3,             0],
            [                3,           0,           0,         0,          0,                   0,             0]
        ])
        assert sum(sum(abs(test_object._model_constants - expected_constants))) < 1E-6

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result',[
            (np.array([0, 1, 2]), [0, 2], 0, dict(p=2, l_d=10, l_q=2, j_rotor=2.45e-3, r_s=5, psi_p=0.165), np.array([[-0.5, 0, 0], [0, -2.5, 0], [0, 0, 0]])),
            (np.array([5, 7, -3]), [4, 8], -2, dict(p=2, l_d=10, l_q=2, j_rotor=2.45e-3, r_s=2, psi_p=0.165), np.array([[-0.2, -0.8, 0], [20, -1, 0], [0, 0, 0]])),
            (np.array([5, 7, -2]), [4, 8], 2, dict(p=2, l_d=10, l_q=2, j_rotor=2.45e-3, r_s=5, psi_p=0.165), np.array([[-0.5, 0.8, 0], [-20, -2.5, 0], [0, 0, 0]])),
        ]
    )
    def test_el_jac_0(self, state, u_in, omega, motor_parameter, result):
        # Test first return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        el_jac, _, _ = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(el_jac == result)

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result',[
            (np.array([2, 0, 0]), [2, 0], 0, dict(p=2, l_d=10, l_q=2, j_rotor=2.45e-3, r_s=5, psi_p=0), np.array([0, -20, 2])),
            (np.array([0, 5, 2]), [8, 4], -2, dict(p=1, l_d=5, l_q=2, j_rotor=2.45e-3, r_s=5, psi_p=10), np.array([2, -5, 1])),
            (np.array([2, -2, 9]), [8, 4], 2, dict(p=2, l_d=10, l_q=2, j_rotor=2.45e-3, r_s=5, psi_p=10), np.array([-0.8, -30, 2])),
        ]
    )
    def test_el_jac_1(self, state, u_in, omega, motor_parameter, result):
        # Test second return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        _, el_over_omega, _ = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(el_over_omega == result)

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result',[
            (np.array([2, 0, 0]), [0, 2], 0, dict(p=2, l_d=10, l_q=2, j_rotor=2.45e-3, r_s=5, psi_p=0), np.array([0, 48, 0])),
            (np.array([0, 5, 2]), [4, 8], -2, dict(p=2, l_d=5, l_q=2, j_rotor=2.45e-3, r_s=5, psi_p=10), np.array([45, 30, 0])),
            (np.array([2, -2, 9]), [4, 8], 2, dict(p=2, l_d=10, l_q=2, j_rotor=2.45e-3, r_s=5, psi_p=10), np.array([-48, 78, 0])),
        ]
    )
    def test_el_jac_2(self, state, u_in, omega, motor_parameter, result):
        # Test second return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        _, _, torque_over_el = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(torque_over_el == result)


class TestInductionMotor:
    """
    class for testing InductionMotor
    """
    # defined test values
    class_to_test = InductionMotor
    _motor_parameter = sci_motor_parameter['motor_parameter']
    _p = _motor_parameter['p']  # pole pair number for testing
    _nominal_values = sci_motor_parameter['nominal_values']
    _limit_values = sci_motor_parameter['limit_values']
    _initializer = sci_initializer
    state_position = sci_state_positions
    state_space = sci_state_space
    _CURRENTS = ['i_salpha', 'i_sbeta']
    _CURRENTS_IDX = [2, 3, 5]
    _number_states = 5

    # counter
    _monkey_update_model_counter = 0
    _monkey_update_limits_counter = 0
    _monkey_super_init_counter = 0

    def monkey_update_model(self):
        """
        mock function for _update_model()
        :return:
        """
        self._monkey_update_model_counter += 1

    def monkey_update_limits(self, *args):
        """
        mock function for _update_limits()
        :return:
        """
        self._monkey_update_limits_counter += 1

    def monkey_super_init(self, motor_parameter=None, nominal_values=None,
                          limit_values=None, motor_initializer=None, initial_limits=None):
        """
        mock function for super().__init__()
        :param motor_parameter:
        :param nominal_values:
        :param limit_values:
        :return:
        """
        self._monkey_super_init_counter += 1
        assert self._expected_parameter['motor_parameter'] == motor_parameter, 'unexpected motor parameter passed'
        assert self._expected_parameter['nominal_values'] == nominal_values, 'unexpected nominal values passed'
        assert self._expected_parameter['limit_values'] == limit_values, 'unexpected limit values passed'
        assert self._expected_parameter['motor_initializer'] == motor_initializer, 'unexpected initializer'
        assert self._expected_parameter['initial_limits'] == initial_limits

    @pytest.mark.parametrize("motor_parameter", [_motor_parameter, None])
    @pytest.mark.parametrize("nominal_values, expected_nv", [({}, {}), (_nominal_values, _nominal_values)])
    @pytest.mark.parametrize("limit_values, expected_lv", [({}, {}), (_limit_values, _limit_values)])
    @pytest.mark.parametrize("motor_initializer", [_initializer, {}])
    @pytest.mark.parametrize("initial_limits", [{},{}])
    def test_init(self, monkeypatch, motor_parameter, nominal_values,
                  limit_values, expected_nv, expected_lv, motor_initializer,
                  initial_limits):
        """
        test initialization of InductionMotor
        :param monkeypatch:
        :param setup: fixture that is called before the function
        :param motor_parameter: possible motor parameters
        :param nominal_values: possible nominal values
        :param limit_values: possible limit values
        :param expected_nv: expected nominal values
        :param expected_lv: expected limit values
        :param motor_initializer: possible motor initializers
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(InductionMotor, "_update_model", self.monkey_update_model)
        monkeypatch.setattr(InductionMotor, "_update_limits", self.monkey_update_limits)
        monkeypatch.setattr(ElectricMotor, '__init__', self.monkey_super_init)
        self._expected_parameter = dict(motor_parameter=motor_parameter, nominal_values=expected_nv,
                                        limit_values=expected_lv, motor_initializer=motor_initializer,
                                        initial_limits=initial_limits)
        # call function to test
        test_object = InductionMotor(motor_parameter, nominal_values,
                                     limit_values, motor_initializer, initial_limits)
        # verify the expected results
        assert self._monkey_update_limits_counter == 1, 'update_limits() is not called once'
        assert self._monkey_super_init_counter == 1, 'super().__init__() is not called once'
        assert self._monkey_update_model_counter == 1, 'update_model() is not called once'

    def test_reset(self, monkeypatch):
        """
        test reset()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(InductionMotor, '__init__', self.monkey_update_model)
        monkeypatch.setattr(InductionMotor, 'CURRENTS', self._CURRENTS)
        test_object = InductionMotor()
        # call function to test
        result = test_object.reset(self.state_position, self.state_space)
        # verify the expected results
        assert all(result == np.zeros(self._number_states)), 'unexpected state after reset()'

    def test_electrical_ode(self, monkeypatch):
        """
        test electrical_ode()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(InductionMotor, '__init__', self.monkey_update_model)
        # use the identity matrix. Therefore, only the states and their multiplications are returned
        monkeypatch.setattr(InductionMotor, '_model_constants', np.identity(11))
        test_object = InductionMotor()
        state = np.array([25, 36, -15, 28])
        u_alphabeta = np.array([[-400, 325], [140, -222]])
        omega = 42
        expected_result = np.array([42, 25, 36, -15, 28, 42 * -15, 42 * 28, -400, 325, 140, -222])
        # call function to test
        result = test_object.electrical_ode(state, u_alphabeta, omega)
        # verify the expected results
        assert all(result == expected_result), 'unexpected result of the electrical_ode()'

    def test_i_in(self, monkeypatch):
        """
        test i_in()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(InductionMotor, '__init__', self.monkey_update_model)
        monkeypatch.setattr(InductionMotor, 'CURRENTS_IDX', self._CURRENTS_IDX)
        test_object = InductionMotor()
        state = np.array([25, 10, 24, 36, 25, 50, 12])
        expected_result = np.array([24, 36, 50])
        # call function to test
        result = test_object.i_in(state)
        # verify the expected results
        assert all(result == expected_result), 'unexpected currents in the motor'

    def test_torque(self, monkeypatch):
        """
        test torque()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        _motor_parameter = sci_motor_parameter['motor_parameter']
        test_object = InductionMotor()
        monkeypatch.setattr(test_object, '_motor_parameter', _motor_parameter)
        currents_fluxes = np.array([15, 10, 0.8, -0.2])
        # call function to test
        torque = test_object.torque(currents_fluxes)
        # verify the expected results
        expected_torque = 31.862069
        assert abs(torque - expected_torque) < 1E-6, 'unexpected torque calculated'

    def test_update_model(self):
        """
        test _update_model()
        :return:
        """
        # setup test scenario
        test_object = InductionMotor()
        test_object._motor_parameter = sci_motor_parameter['motor_parameter']
        # call function to test
        test_object._update_model()
        # verify results
        l_s = 145E-3
        l_r = 145E-3
        tau_r = l_r / 1.5
        sigma = (l_s * l_r - 140e-3 ** 2) / (l_s * l_r)
        tau_sig = sigma * l_s / (3 + 1.5 * (140e-3 ** 2) / (l_r ** 2))

        assert abs(l_s - (test_object._motor_parameter['l_m'] + test_object._motor_parameter['l_sigs'])) < 1E-6, 'unexpected stator inductance'
        assert abs(l_r - (test_object._motor_parameter['l_m'] + test_object._motor_parameter['l_sigr'])) < 1E-6, 'unexpected rotor inductance'
        assert abs(sigma - ((l_s * l_r - test_object._motor_parameter['l_m'] ** 2) / (l_s * l_r))) < 1E-6, 'unexpected leakage coefficient'
        assert abs(tau_r - (l_r / test_object._motor_parameter['r_r'])) < 1E-6, 'unexpected rotor time constant'
        assert abs(tau_sig - (sigma * l_s / (test_object._motor_parameter['r_s'] + test_object._motor_parameter['r_r'] * (test_object._motor_parameter['l_m'] ** 2) / (l_r ** 2)))) < 1E-6, 'unexpected leakage time constant'

        expected_constants = np.array([
            [0, -1 / tau_sig, 0, 140E-3 * 1.5 / (sigma * l_s * l_r ** 2), 0, 0, 140E-3 * 2 / (sigma * l_r * l_s), 1 / (sigma * l_s), 0, -140E-3 / (sigma * l_r * l_s), 0, ],  # i_ralpha_dot
            [0, 0, -1 / tau_sig, 0, 140E-3 * 1.5 / (sigma * l_s * l_r ** 2), -140E-3 * 2 / (sigma * l_r * l_s), 0, 0, 1 / (sigma * l_s), 0, -140E-3 / (sigma * l_r * l_s), ],  # i_rbeta_dot
            [0, 140E-3 / tau_r, 0, -1 / tau_r, 0, 0, -2, 0, 0, 1, 0, ],  # psi_ralpha_dot
            [0, 0, 140E-3 / tau_r, 0, -1 / tau_r, 2, 0, 0, 0, 0, 1, ],  # psi_rbeta_dot
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # epsilon_dot
        ])
        assert sum(sum(abs(test_object._model_constants - expected_constants))) < 1E-6

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result',[
            (np.array([0, 1, 2, 3, 4]), [0, 1, 2, 3], 0,
             dict(p=2, l_m=10, l_sigs=2,  l_sigr=3 ,j_rotor=4, r_s=5, r_r=6),
             np.array([[-1.9848901098901102, 0, 0.08241758241758242, 0.0, 0],
                       [0, -1.9848901098901102, 0.0, 0.08241758241758242, 0],
                       [4.615384615384616, 0, -0.46153846153846156, 0, 0],
                       [0, 4.615384615384616, 0, -0.46153846153846156, 0],
                       [0, 0, 0, 0, 0]])),
        ]
    )
    def test_el_jac_0(self, state, u_in, omega, motor_parameter, result):
        # Test first return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        el_jac, _, _ = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(el_jac == result)

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result',[
            (np.array([0, 1, 2, 3, 4]), [0, 1, 2, 3], 0,
             dict(p=2, l_m=10, l_sigs=2,  l_sigr=3, j_rotor=4, r_s=5, r_r=6),
             np.array([10 * 2 / 56 * 3,
                       - 10 * 2 / 56 * 2,
                       - 2 * 3,
                       2 * 2,
                       2])),
        ]
    )
    def test_el_jac_1(self, state, u_in, omega, motor_parameter, result):
        # Test second return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        _, el_over_omega, _ = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(el_over_omega == result)

    @pytest.mark.parametrize(
        'state, u_in, omega, motor_parameter, result',[
            (np.array([5, 1, 2, 3, 4]), [0, 1, 2, 3], 0,
             dict(p=2, l_m=10, l_sigs=2,  l_sigr=3, j_rotor=4, r_s=5, r_r=6),
             np.array([- 3 * 3 / 2 * 2 * 10 / 13,
                2 * 3 / 2 * 2 * 10 / 13,
                1 * 3 / 2 * 2 * 10 / 13,
                - 5 * 3 / 2 * 2 * 10 / 13,
            0])),
        ]
    )
    def test_el_jac_2(self, state, u_in, omega, motor_parameter, result):
        # Test second return of electrical jacobian function
        motor = self.class_to_test(motor_parameter=motor_parameter)
        _, _, torque_over_el = motor.electrical_jacobian(state, u_in, omega)
        assert np.all(torque_over_el == result)


class TestSquirrelCageInductionMotor:
    """
    class for testing SquirrelCageInductionMotor
    """

    _monkey_super_electrical_ode_counter = 0
    _monkey_passed_voltage = []

    def monkey_init(self):
        """
        mock function for __init__()
        :return:
        """
        pass

    def monkey_super_electrical_ode(self, state, u_sr_alphabeta, omega, *_):
        """
        mock function for _update_model()
        :param state: electrical state of the system
        :param u_sr_alphabeta: input voltage of the system
        :param omega: meachnical velocity
        :return:
        """
        self._monkey_passed_voltage = u_sr_alphabeta
        self._monkey_super_electrical_ode_counter += 1


    def test_electrical_ode(self, monkeypatch):
        """
        test electrical_ode()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(InductionMotor, '__init__', self.monkey_init)
        # use the identity matrix. Therefore, only the states and their multiplications are returned
        monkeypatch.setattr(InductionMotor, '_model_constants', np.identity(11))
        monkeypatch.setattr(InductionMotor, 'electrical_ode', self.monkey_super_electrical_ode)
        test_object = SquirrelCageInductionMotor()
        state = np.array([25, 36, -15, 28])
        u_salphabeta = np.array([-400, 325])
        omega = 42
        # call function to test
        test_object.electrical_ode(state, u_salphabeta, omega)
        expected_voltage = np.array([[-400, 325], [0, 0]])
        # verify the expected results
        assert np.all(self._monkey_passed_voltage == expected_voltage), 'unexpected voltage passed to super().electrical_ode()'
        assert self._monkey_super_electrical_ode_counter == 1, 'super().electrical_ode() is not called once'


class TestDoublyFedInductionMotor:
    """
    class for testing DoublyFedInductionMotor
    """

    _motor_parameter = sci_motor_parameter['motor_parameter']
    _nominal_values = sci_motor_parameter['nominal_values']
    _limit_values = sci_motor_parameter['limit_values']

    _monkey_super_update_limits_counter = 0

    def monkey_init(self):
        """
        mock function for super().__init__()
        :return:
        """
        pass

    def monkey_super_update_limits(self, *args):
        """
        mock function for super()._update_limits()
        :return:
        """
        self._monkey_super_update_limits_counter += 1

    def test_update_limits(self, monkeypatch):
        """
        test update limits of DFIM
        :param monkeypatch:
        :return:
        """
        # call function to test
        monkeypatch.setattr(DoublyFedInductionMotor, '__init__', self.monkey_init)

        test_object = DoublyFedInductionMotor()
        test_object._motor_parameter = self._motor_parameter
        test_object._limits = self._limit_values
        test_object._nominal_values = self._nominal_values
        # verify the expected results
        test_object._update_limits()

        assert test_object._limits['u_rbeta'] == 0.5 * self._limit_values['u']
        assert test_object._nominal_values['i_ralpha'] == (test_object._nominal_values.get('i', None) or test_object._nominal_values['u_ralpha'] / test_object._motor_parameter['r_r'])

        monkeypatch.setattr(InductionMotor, '_update_limits', self.monkey_super_update_limits)
        test_object._update_limits()
        assert self._monkey_super_update_limits_counter == 1, 'super().update_limits() is not called once'
# endregion

# endregion

