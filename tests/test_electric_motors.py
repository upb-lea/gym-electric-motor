from gym_electric_motor.physical_systems.electric_motors import *
from .conf import *
from gym_electric_motor.utils import make_module
import pytest


# region first Version

# global parameter
g_omega = [0, 15, 12.2, 13]
g_i_a = [0, 3, 40, 35.3, -3]  # also used for permanently excited and series dc motor
g_i_e = [0, 0.2, 0.5, -1, 1]
g_u_in = [0, 400, -400, 325.2, -205.4]
g_t_load = [0, 1, 5, -5, 4.35]


# region general test functions


def motor_testing(motor_1_default, motor_2_default, motor_1, motor_2, motor_state, limit_values, nominal_values,
                  motor_parameter):
    motor_reset_testing(motor_1_default, motor_2_default, motor_state)
    motor_reset_testing(motor_1, motor_2, motor_state)

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


def motor_reset_testing(motor_1, motor_2, motor_state):
    # tests if the reset function works correctly
    assert (motor_1.reset() == motor_2.reset()).all()
    assert all(motor_1.reset() == np.zeros(len(motor_state))), "Reset is not zero"


# endregion

# region DcSeriesMotor


def test_dc_series_motor():
    motor_state = ['i', 'i']  # list of state names
    motor_parameter = test_motor_parameter['DcSeries']['motor_parameter']
    nominal_values = test_motor_parameter['DcSeries']['nominal_values']  # dict
    limit_values = test_motor_parameter['DcSeries']['limit_values']  # dict
    # default initialization
    motor_1_default = make_module(ElectricMotor, 'DcSeries')
    motor_2_default = DcSeriesMotor()
    # initialization parameters as dicts
    motor_1 = make_module(ElectricMotor, 'DcSeries', motor_parameter=motor_parameter, nominal_values=nominal_values,
                          limit_values=limit_values)
    motor_2 = DcSeriesMotor(motor_parameter, nominal_values, limit_values)
    motor_testing(motor_1_default, motor_2_default, motor_1, motor_2, motor_state, limit_values, nominal_values,
                  motor_parameter)

    series_motor_state_space_testing(motor_1_default, motor_2_default)
    series_motor_state_space_testing(motor_1, motor_2)

    series_motor_electrical_ode_testing(motor_1_default)
    series_motor_electrical_ode_testing(motor_1)


def series_motor_state_space_testing(motor_1, motor_2):
    # u,i>0
    state_space = ({'omega': 0, 'torque': 0, 'i': 0, 'u': 0}, {'omega': 1, 'torque': 1, 'i': 1, 'u': 1})
    assert motor_1.get_state_space([0, 1], [0, 1]) == motor_2.get_state_space([0, 1], [0, 1])
    assert motor_1.get_state_space([0, 1], [0, 1]) == state_space

    # u>0
    state_space = (
        {'omega': 0, 'torque': 0, 'i': -1, 'u': 0}, {'omega': 1, 'torque': 1, 'i': 1, 'u': 1})
    assert motor_1.get_state_space([-1, 1], [0, 1]) == motor_2.get_state_space([-1, 1], [0, 1])
    assert motor_1.get_state_space([-1, 1], [0, 1]) == state_space

    # i>0
    state_space = (
        {'omega': 0, 'torque': 0, 'i': 0, 'u': -1}, {'omega': 1, 'torque': 1, 'i': 1, 'u': 1})
    assert motor_1.get_state_space([0, 1], [-1, 1]) == motor_2.get_state_space([0, 1], [-1, 1])
    assert motor_1.get_state_space([0, 1], [-1, 1]) == state_space

    # u,i>&<0
    state_space = (
        {'omega': 0, 'torque': 0, 'i': -1, 'u': -1}, {'omega': 1, 'torque': 1, 'i': 1, 'u': 1})
    assert motor_1.get_state_space([-1, 1], [-1, 1]) == motor_2.get_state_space([-1, 1], [-1, 1])
    assert motor_1.get_state_space([-1, 1], [-1, 1]) == state_space


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


def test_dc_shunt_motor():
    """
    tests the dc shunt motor class
    :return:
    """
    # set up test parameter
    motor_state = ['i_a', 'i_e']  # list of state names
    motor_parameter = test_motor_parameter['DcShunt']['motor_parameter']
    nominal_values = test_motor_parameter['DcShunt']['nominal_values']  # dict
    limit_values = test_motor_parameter['DcShunt']['limit_values']  # dict
    # default initialization of electric motor
    motor_1_default = make_module(ElectricMotor, 'DcShunt')
    motor_2_default = DcShuntMotor()

    motor_1 = make_module(ElectricMotor, 'DcShunt', motor_parameter=motor_parameter, nominal_values=nominal_values,
                          limit_values=limit_values)
    motor_2 = DcShuntMotor(motor_parameter, nominal_values, limit_values)
    # test if both initializations work correctly for limits and nominal values
    motor_testing(motor_1_default, motor_2_default, motor_1, motor_2, motor_state, limit_values, nominal_values,
                  motor_parameter)
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
    assert motor_1.get_state_space([0, 1], [0, 1]) == motor_2.get_state_space([0, 1], [0, 1])
    assert motor_1.get_state_space([0, 1], [0, 1]) == state_space

    # u>0
    state_space = (
        {'omega': 0, 'torque': -1, 'i_a': -1, 'i_e': -1, 'u': 0}, {'omega': 1, 'torque': 1, 'i_a': 1, 'i_e': 1, 'u': 1})
    assert motor_1.get_state_space([-1, 1], [0, 1]) == motor_2.get_state_space([-1, 1], [0, 1])
    assert motor_1.get_state_space([-1, 1], [0, 1]) == state_space

    # i>0
    state_space = (
        {'omega': 0, 'torque': 0, 'i_a': 0, 'i_e': 0, 'u': -1}, {'omega': 1, 'torque': 1, 'i_a': 1, 'i_e': 1, 'u': 1})
    assert motor_1.get_state_space([0, 1], [-1, 1]) == motor_2.get_state_space([0, 1], [-1, 1])
    assert motor_1.get_state_space([0, 1], [-1, 1]) == state_space

    # u,i>&<0
    state_space = (
        {'omega': 0, 'torque': -1, 'i_a': -1, 'i_e': -1, 'u': -1},
        {'omega': 1, 'torque': 1, 'i_a': 1, 'i_e': 1, 'u': 1})
    assert motor_1.get_state_space([-1, 1], [-1, 1]) == motor_2.get_state_space([-1, 1], [-1, 1])
    assert motor_1.get_state_space([-1, 1], [-1, 1]) == state_space


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


def test_dc_permex_motor():
    motor_state = ['i']  # list of state names
    motor_parameter = test_motor_parameter['DcPermEx']['motor_parameter']
    nominal_values = test_motor_parameter['DcPermEx']['nominal_values']  # dict
    limit_values = test_motor_parameter['DcPermEx']['limit_values']  # dict
    # default initialization without parameters
    motor_1_default = make_module(ElectricMotor, 'DcPermEx')
    motor_2_default = DcPermanentlyExcitedMotor()
    # initialization parameters as dicts
    motor_1 = make_module(ElectricMotor, 'DcPermEx', motor_parameter=motor_parameter, nominal_values=nominal_values,
                          limit_values=limit_values)
    motor_2 = DcPermanentlyExcitedMotor(motor_parameter, nominal_values, limit_values)

    motor_testing(motor_1_default, motor_2_default, motor_1, motor_2, motor_state, limit_values, nominal_values,
                  motor_parameter)

    permex_motor_state_space_testing(motor_1_default, motor_2_default)
    permex_motor_state_space_testing(motor_1, motor_2)

    permex_motor_electrical_ode_testing(motor_1_default)
    permex_motor_electrical_ode_testing(motor_1)


def permex_motor_state_space_testing(motor_1, motor_2):
    # u,i>0
    state_space = ({'omega': 0, 'torque': 0, 'i': 0, 'u': 0}, {'omega': 1, 'torque': 1, 'i': 1, 'u': 1})
    assert motor_1.get_state_space([0, 1], [0, 1]) == motor_2.get_state_space([0, 1], [0, 1])
    assert motor_1.get_state_space([0, 1], [0, 1]) == state_space

    # u>0
    state_space = (
        {'omega': 0, 'torque': -1, 'i': -1, 'u': 0}, {'omega': 1, 'torque': 1, 'i': 1, 'u': 1})
    assert motor_1.get_state_space([-1, 1], [0, 1]) == motor_2.get_state_space([-1, 1], [0, 1])
    assert motor_1.get_state_space([-1, 1], [0, 1]) == state_space

    # i>0
    state_space = (
        {'omega': -1, 'torque': 0, 'i': 0, 'u': -1}, {'omega': 1, 'torque': 1, 'i': 1, 'u': 1})
    assert motor_1.get_state_space([0, 1], [-1, 1]) == motor_2.get_state_space([0, 1], [-1, 1])
    assert motor_1.get_state_space([0, 1], [-1, 1]) == state_space

    # u,i>&<0
    state_space = (
        {'omega': -1, 'torque': -1, 'i': -1, 'u': -1}, {'omega': 1, 'torque': 1, 'i': 1, 'u': 1})
    assert motor_1.get_state_space([-1, 1], [-1, 1]) == motor_2.get_state_space([-1, 1], [-1, 1])
    assert motor_1.get_state_space([-1, 1], [-1, 1]) == state_space


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


def test_dc_extex_motor():
    motor_state = ['i_a', 'i_e']  # list of state names
    motor_parameter = test_motor_parameter['DcExtEx']['motor_parameter']
    nominal_values = test_motor_parameter['DcExtEx']['nominal_values']  # dict
    limit_values = test_motor_parameter['DcExtEx']['limit_values']  # dict
    # default initialization without parameters
    motor_1_default = make_module(ElectricMotor, 'DcExtEx')
    motor_2_default = DcExternallyExcitedMotor()
    # initialization parameters as dicts
    motor_1 = make_module(ElectricMotor, 'DcExtEx', motor_parameter=motor_parameter, nominal_values=nominal_values,
                          limit_values=limit_values)
    motor_2 = DcExternallyExcitedMotor(motor_parameter, nominal_values, limit_values)
    motor_testing(motor_1_default, motor_2_default, motor_1, motor_2, motor_state, limit_values, nominal_values,
                  motor_parameter)

    extex_motor_state_space_testing(motor_1_default, motor_2_default)
    extex_motor_state_space_testing(motor_1, motor_2)

    extex_motor_electrical_ode_testing(motor_1_default)
    extex_motor_electrical_ode_testing(motor_1)


def extex_motor_state_space_testing(motor_1, motor_2):
    # u,i>0
    state_space = ({'omega': 0, 'torque': 0, 'i_a': 0, 'i_e': 0, 'u_a': 0, 'u_e': 0}, {'omega': 1, 'torque': 1,
                                                                                       'i_a': 1, 'i_e': 1,
                                                                                       'u_a': 1, 'u_e': 1})
    voltage_limits = [[0, 1], [0, 1]]
    current_limits = [[0, 1], [0, 1]]
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
    :param state: [i_q, i_d, epsilon]
    :param mp: motor parameter (dict)
    :return: generated torque
    """
    return 1.5 * mp['p'] * (mp['psi_p'] + (mp['l_d'] - mp['l_q']) * state[1]) * state[0]


def synchronous_motor_ode_testing(state, voltage, omega, mp):
    """
    use this function as benchmark for the ode
    :param state: [i_q, i_d, epsilon]
    :param voltage: [u_q, u_d]
    :param omega: angular velocity
    :param mp: motor parameter (dict)
    :return: ode system
    """
    u_d = voltage[1]
    u_q = voltage[0]
    i_d = state[1]
    i_q = state[0]
    return np.array([(u_q - mp['r_s'] * i_q - omega * mp['p'] * (mp['l_d'] * i_d + mp['psi_p'])) / mp['l_q'],
                     (u_d - mp['r_s'] * i_d + mp['l_q'] * omega * mp['p'] * i_q) / mp['l_d'],
                     omega * mp['p']])


@pytest.mark.parametrize("motor_type, motor_class", [('SynRM', SynchronousReluctanceMotor),
                                                     ('PMSM', PermanentMagnetSynchronousMotor)])
def test_synchronous_motor_testing(motor_type, motor_class):
    """
    testing the synrm and pmsm
    consider that it uses dq coordinates and the state is [i_q, i_d, epsilon]!!.
    The same goes with the voltages u_qd and not as usual dq!!
    :return:
    """
    parameter = test_motor_parameter[motor_type]
    # use this values for testing
    state = np.array([0.5, 0.3, 0.68])  # i_q, i_d, epsilon
    u_qd = np.array([200, 50])
    omega = 25
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
        mp.update({'psi_p': 0})
    for motor in [default_init_1, default_init_2]:
        assert motor.torque(state) == torque_testing(state, mp)
        assert all(motor.i_in(state) == state[0:2])
        ode = motor.electrical_ode(state, u_qd, omega)
        test_ode = synchronous_motor_ode_testing(state, u_qd, omega, mp)
        assert sum(abs(ode - test_ode)) < 1E-8, "Motor ode is wrong: " + str([ode, test_ode])
    # test parametrized motors
    motor_init_1 = make_module(ElectricMotor, motor_type,
                               motor_parameter=parameter['motor_parameter'],
                               nominal_values=parameter['nominal_values'],
                               limit_values=parameter['limit_values'])

    motor_init_2 = motor_class(motor_parameter=parameter['motor_parameter'],
                               nominal_values=parameter['nominal_values'],
                               limit_values=parameter['limit_values'])
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
                       motor_init_1.limits[limit_key] == .5 * factor_abc_to_alpha_beta * parameter['limit_values']['u']\
                       or motor_init_1.limits[limit_key] == .5 * parameter['limit_values']['u']

            if limit_key in parameter['limit_values'].keys():
                assert motor_init_1.limits[limit_key] == parameter['limit_values'][limit_key]

            if 'i_' in limit_key:
                assert motor_init_1.nominal_values[limit_key] == parameter['nominal_values']['i'] or \
                       motor_init_1.nominal_values[limit_key] == factor_abc_to_alpha_beta\
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
        ode = motor.electrical_ode(state, u_qd, omega)
        test_ode = synchronous_motor_ode_testing(state, u_qd, omega, mp)
        assert sum(abs(ode - test_ode)) < 1E-8, "Motor ode is wrong: " + str([ode, test_ode])

# endregion
