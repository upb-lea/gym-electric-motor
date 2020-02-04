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
        ode = motor.electrical_ode(state, u_qd, omega)
        test_ode = synchronous_motor_ode_testing(state, u_qd, omega, mp)
        assert sum(abs(ode - test_ode)) < 1E-8, "Motor ode is wrong: " + str([ode, test_ode])


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

    @pytest.mark.parametrize("motor_parameter, result_motor_parameter",
                             [(None, {}), (_motor_parameter, _motor_parameter)])
    @pytest.mark.parametrize("nominal_values, result_nominal_values", [(None, {}), (_nominal_values, _nominal_values)])
    @pytest.mark.parametrize("limits, result_limit_values", [(None, {}), (_limits, _limits)])
    def test_init(self, motor_parameter, nominal_values, limits, result_motor_parameter, result_nominal_values,
                  result_limit_values):
        """
        test initialization of ElectricMotor
        :param motor_parameter: possible values for motor parameter
        :param nominal_values: possible values for nominal values
        :param limits: possible values for limit values
        :param result_motor_parameter: expected motor parameter
        :param result_nominal_values: expected nominal values
        :param result_limit_values: expected limit values
        :return:
        """
        # call function to test
        test_object = ElectricMotor(motor_parameter,
                                    nominal_values,
                                    limits)
        # verify the expected results
        assert test_object._motor_parameter == test_object.motor_parameter == result_motor_parameter, \
            'unexpected initialization of motor parameter'
        assert test_object._limits == test_object.limits == result_limit_values, 'unexpected initialization of limits'
        assert test_object._nominal_values == test_object.nominal_values == result_nominal_values, \
            'unexpected initialization of nominal values'

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
        result = test_object.reset()
        # verify the expected results
        assert all(result == np.zeros(self._length_Currents)), 'unexpected state after reset()'


# endregion

# region DcMotor

class TestDcMotor:
    # defined test values
    _motor_parameter = test_motor_parameter['DcExtEx']['motor_parameter']
    _nominal_values = test_motor_parameter['DcExtEx']['nominal_values']
    _limits = test_motor_parameter['DcExtEx']['limit_values']
    s_motor_parameter = test_motor_parameter['DcExtEx']['motor_parameter']
    defult_motor_parameter = DcMotor._default_motor_parameter
    default_nominal_values = DcMotor._default_nominal_values
    default_limits = DcMotor._default_limits
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

    def _monkey_super_init(self, motor_parameter, nominal_values, limits):
        """
        mock function for super().__init__()
        :param motor_parameter:
        :param nominal_values:
        :param limits:
        :return:
        """
        self.monkey_super_init_counter += 1
        assert motor_parameter == self._motor_parameter, 'motor parameter are not passed correctly'
        assert nominal_values == self._nominal_values, 'nominal values are not passed correctly'
        assert limits == self._limits, 'limits are not passed correctly'

    @pytest.mark.parametrize("motor_parameter, result_motor_parameter",
                             [(None, {}), (_motor_parameter, _motor_parameter)])
    @pytest.mark.parametrize("nominal_values, result_nominal_values", [(None, {}), (_nominal_values, _nominal_values)])
    @pytest.mark.parametrize("limits, result_limit_values", [(None, {}), (_limits, _limits)])
    def test_init(self, monkeypatch, motor_parameter, nominal_values, limits, result_motor_parameter,
                  result_nominal_values,
                  result_limit_values):
        """
        test initialization of DcMotor
        :param monkeypatch:
        :param motor_parameter: possible motor parameters
        :param nominal_values: possible nominal values
        :param limits: possible limit values
        :param result_motor_parameter: expected resulting motor parameter
        :param result_nominal_values: expected resulting nominal values
        :param result_limit_values: expected resulting limit values
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(DcMotor, "_update_model", self._monkey_update_model)
        monkeypatch.setattr(DcMotor, "_update_limits", self._monkey_update_limits)
        monkeypatch.setattr(ElectricMotor, "__init__", self._monkey_super_init)

        self._motor_parameter = motor_parameter
        self._nominal_values = nominal_values
        self._limits = limits

        # call function to test
        test_object = DcMotor(motor_parameter, nominal_values, limits)

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
        test_object = DcMotor()
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
        test_object = DcMotor()
        # call function to test
        i_in = test_object.i_in(self._currents)
        # verify the expected results
        assert i_in == list(self._currents), 'unexpected current in the motor'

    def test_electrical_ode(self):
        # test electrical_ode()
        # setup test scenario
        test_object = DcMotor(motor_parameter=self.s_motor_parameter)
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
        input_currents = ((i_1_min, 1), (i_2_min, 1))
        input_voltages = ((u_1_min, 1), (u_2_min, 1))
        test_object = DcMotor()
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
        test_object = DcMotor()
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
        converter_currents = (i_converter_min, 1)
        converter_voltages = (u_converter_min, 1)
        # call function to test
        low, high = test_object.get_state_space(converter_currents, converter_voltages)
        # verify the expected results
        assert low == result_low, 'unexpected lower state space'
        assert high == result_high, 'unexpected upper state space'


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

        converter_currents = (i_converter_min, 1)
        converter_voltages = (u_converter_min, 1)
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

        converter_currents = (i_converter_min, 1)
        converter_voltages = (u_converter_min, 1)
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


# endregion

# region Three Phase Motors


class TestSynchronousMotor:
    """
    class for testing SynchronousMotor
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

    # defined test values
    _motor_parameter = pmsm_motor_parameter['motor_parameter']
    _p = _motor_parameter['p']  # pole pair number for testing
    _nominal_values = pmsm_motor_parameter['nominal_values']
    _limit_values = pmsm_motor_parameter['limit_values']
    _CURRENTS = ['i_a', 'i_b', 'i_c']
    _CURRENTS_IDX = [2, 3, 5]
    _number_states = 4

    # counter
    _monkey_q_counter = 0
    _monkey_update_model_counter = 0
    _monkey_update_limits_counter = 0
    _monkey_super_init_counter = 0

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

    def monkey_super_init(self, motor_parameter=None, nominal_values=None, limit_values=None):
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

    def test_t_23(self):
        """
        test t_23()
        :return:
        """
        assert sum(
            abs(SynchronousMotor.t_23(self._forward_quantities_abc) - self._forward_quantities_alpha_beta)) < 1E-6, \
            'unexpected calculation from abc to alpha-beta'

    def test_t_32(self):
        """
        test t_32()
        :return:
        """
        assert sum(
            abs(SynchronousMotor.t_32(self._backward_quantities_alpha_beta) - self._backward_quantities_abc)) < 1E-6, \
            'unexpected calculation from alpha-beta to abc'

    def test_q(self):
        """
        test q()
        :return:
        """
        assert sum(abs(SynchronousMotor.q(self._forward_quantities_alpha_beta,
                                          self._forward_epsilon) - self._forward_quantities_dq)) < 1E-6, \
            'unexpected calculation from alpha-beta to dq'

    def test_q_inv(self, monkeypatch):
        """
        test q_inv()
        :param monkeypatch:
        :return:
        """
        # test if the resulting value is correct
        assert sum(abs(SynchronousMotor.q_inv(self._backward_quantities_dq,
                                              self._backward_epsilon) - self._backward_quantities_alpha_beta)) < 1E-6, \
            'unexpected calculation from dq to alpha-beta'
        # test if the internal function is called correctly
        monkeypatch.setattr(SynchronousMotor, "q", self.monkey_q)
        # setup test scenario
        self._expected_epsilon = -self._backward_epsilon
        self._expected_quantities = self._backward_quantities_dq
        self._expected_result = self._backward_quantities_alpha_beta
        # call function to test
        SynchronousMotor.q_inv(self._backward_quantities_dq, self._backward_epsilon)
        # verify the expected results
        assert self._monkey_q_counter == 1, "q function was not called correctly"

    def test_q_me(self, monkeypatch):
        """
        test q_me()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(SynchronousMotor, "_update_model", self.monkey_update_model)
        monkeypatch.setattr(SynchronousMotor, "_update_limits", self.monkey_update_limits)
        test_object = SynchronousMotor(motor_parameter=self._motor_parameter)
        # call function to test and verify the expected results
        assert sum(abs(test_object.q_me(self._forward_quantities_alpha_beta,
                                        self._forward_epsilon_me) - self._forward_quantities_dq)) < 1E-6, \
            'unexpected result from alpha-beta to dq. Mechanical angle needed.'
        # setup test scenario
        monkeypatch.setattr(SynchronousMotor, 'q', self.monkey_q)
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
        monkeypatch.setattr(SynchronousMotor, "_update_model", self.monkey_update_model)
        monkeypatch.setattr(SynchronousMotor, "_update_limits", self.monkey_update_limits)
        test_object = SynchronousMotor(motor_parameter=self._motor_parameter)
        # call function to test verify the expected results
        assert sum(abs(test_object.q_inv_me(self._forward_quantities_dq,
                                            self._forward_epsilon_me) - self._forward_quantities_alpha_beta)) < 1E-6, \
            'unexpected result from dq to alpha-beta. Mechanical angle needed.'
        # setup test scenario
        monkeypatch.setattr(SynchronousMotor, 'q', self.monkey_q)
        self._expected_epsilon = -self._forward_epsilon
        self._expected_quantities = self._forward_quantities_dq
        self._expected_result = self._forward_quantities_alpha_beta
        # call function to test
        test_object.q_inv_me(self._forward_quantities_dq, self._forward_epsilon_me)
        # verify the expected results
        assert self._monkey_q_counter == 1, "q function was not called correctly"

    @pytest.mark.parametrize("motor_parameter", [_motor_parameter, None])
    @pytest.mark.parametrize("nominal_values, expected_nv", [(None, {}), (_nominal_values, _nominal_values)])
    @pytest.mark.parametrize("limit_values, expected_lv", [(None, {}), (_limit_values, _limit_values)])
    def test_init(self, monkeypatch, motor_parameter, nominal_values, limit_values, expected_nv, expected_lv):
        """
        test initialization of SynchronousMotor
        :param monkeypatch:
        :param setup: fixture that is called before the function
        :param motor_parameter: possible motor parameters
        :param nominal_values: possible nominal values
        :param limit_values: possible limit values
        :param expected_nv: expected nominal values
        :param expected_lv: expected limit values
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(SynchronousMotor, "_update_model", self.monkey_update_model)
        monkeypatch.setattr(SynchronousMotor, "_update_limits", self.monkey_update_limits)
        monkeypatch.setattr(ElectricMotor, '__init__', self.monkey_super_init)
        self._expected_parameter = dict(motor_parameter=motor_parameter, nominal_values=expected_nv,
                                        limit_values=expected_lv)
        # call function to test
        test_object = SynchronousMotor(motor_parameter, nominal_values, limit_values)
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
        test_object = SynchronousMotor()
        # call function to test
        result = test_object.reset()
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
            [0, -0.5 / 8E-3, 0, 1 / 8E-3, 0, 0, -3 * 70 / 8],
            [0, 0, -0.5 / 70E-3, 0, 1 / 70E-3, 8e-3 * 3 / 70E-3, 0],
            [3, 0, 0, 0, 0, 0, 0]
        ])
        assert sum(sum(abs(test_object._model_constants - expected_constants))) < 1E-6, 'unexpected model constants'


class TestPermanentMagnetSynchronousMotor:
    """
    class for testing PermanentMagnetSynchronousMotor
    """

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
        currents = np.array([15, 10])
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
            [-3*0.171/125E-3, -5 / 125E-3, 0, 1 / 125E-3, 0, 0, -3 * 84 / 125],
            [0, 0, -5 / 84E-3, 0, 1 / 84E-3, 125E-3 * 3 / 84E-3, 0],
            [3, 0, 0, 0, 0, 0, 0]
        ])
        assert sum(sum(abs(test_object._model_constants - expected_constants))) < 1E-6

# endregion

# endregion
