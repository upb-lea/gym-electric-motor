from gym_electric_motor.physical_systems import *
from gym_electric_motor.physical_systems.electric_motors import *
from .conf import *
from .functions_used_for_testing import *
from gym_electric_motor.physical_systems.mechanical_loads import PolynomialStaticLoad
from gym_electric_motor.physical_systems.voltage_supplies import *
from gym_electric_motor.physical_systems.solvers import *
from gym_electric_motor.physical_systems.converters import *
from gym_electric_motor.utils import make_module

from numpy.random import randint, seed, uniform
import pytest

g_solver_classes = ['euler', 'scipy.ode', 'scipy.solve_ivp', 'scipy.odeint']
g_disc_converter = ['Disc-1QC', 'Disc-2QC', 'Disc-4QC']
g_cont_converter = ['Cont-1QC', 'Cont-2QC', 'Cont-4QC']
g_double_converter = ['Disc-Double', 'Cont-Double']
g_motors = ['DcSeries', 'DcShunt', 'DcExtEx', 'DxPermEx']


# region discrete  dc systems


@pytest.mark.parametrize("motor_type, motor_class", [('DcExtEx', DcExternallyExcitedMotor),
                                                     ('DcPermEx', DcPermanentlyExcitedMotor),
                                                     ('DcSeries', DcSeriesMotor),
                                                     ('DcShunt', DcShuntMotor)])
@pytest.mark.parametrize("conv", g_disc_converter)
def test_dc_disc(motor_type, motor_class, conv):
    """
    test all combinations of motors and converters in a physical system for the discrete case
    :param motor_type: motor name (string)
    :param motor_class: class name of motor
    :param conv: converter name (string)
    :return:
    """
    dc_motor_system_1, dc_motor_system_2, kwargs, nominal_values, limit_values, motor_parameter, converter_parameter = \
        setup_dc_tests(motor_type, motor_class, conv)
    compare_dc_motor_constants(dc_motor_system_1, dc_motor_system_2, kwargs, nominal_values, limit_values,
                               motor_parameter, converter_parameter)
    compare_discrete_dc_motor_simulation(dc_motor_system_1, dc_motor_system_2)


# endregion

# region continuous dc systems


@pytest.mark.parametrize("motor_type, motor_class", [('DcExtEx', DcExternallyExcitedMotor),
                                                     ('DcPermEx', DcPermanentlyExcitedMotor),
                                                     ('DcSeries', DcSeriesMotor),
                                                     ('DcShunt', DcShuntMotor)])
@pytest.mark.parametrize("conv", g_cont_converter)
def test_dc_series_cont(motor_type, motor_class, conv):
    """
    test all combinations of motors and converters in a physical system for the continuous case
    :param motor_type: motor name (string)
    :param motor_class: motor name (class)
    :param conv: converter name (string)
    :return:
    """
    dc_motor_system_1, dc_motor_system_2, kwargs, nominal_values, limit_values, motor_parameter, converter_parameter = \
        setup_dc_tests(motor_type, motor_class, conv)
    compare_dc_motor_constants(dc_motor_system_1, dc_motor_system_2, kwargs, nominal_values, limit_values,
                               motor_parameter, converter_parameter)
    compare_continuous_dc_motor_simulation(dc_motor_system_1, dc_motor_system_2)


# endregion

# region general test functions

def setup_dc_tests(motor_type, motor_class, conv):
    """
    This function initializes the physical system that should be tested afterwards.
    :param motor_type: motor name (string)
    :param motor_class: motor name (class)
    :param conv: converter name (string)
    :return:    dc_motor_system_1 (string initialization),
                dc_motor_system_2 (modularized initialization),
                params (all parameter),
                nominal_values (dict),
                limit_values (dict),
                motor_parameter (dict),
                converter_parameter
    """
    # load test parameter
    u_sup = test_motor_parameter[motor_type]['motor_parameter']['u_sup']
    tau = converter_parameter['tau']
    motor_parameter = test_motor_parameter[motor_type]['motor_parameter']  # dict
    nominal_values = test_motor_parameter[motor_type]['nominal_values']  # dict
    limit_values = test_motor_parameter[motor_type]['limit_values']  # dict
    solver_kwargs = {'method': 'RK45'}
    # sum up all parameters in one dict
    params = motor_parameter
    params.update(converter_parameter)
    params.update(load_parameter)
    params.update(solver_kwargs)
    params.update({'motor_type': motor_type})
    params.update({'converter_type': conv})
    params.update({'u_sup': u_sup})
    # setup motor system with strings and set each parameter
    if motor_type == 'DcExtEx':
        if 'Disc' in conv:
            double_converter = 'Disc-Double'
        else:
            double_converter = 'Cont-Double'
        dc_motor_system_1 = DcMotorSystem(converter=double_converter,
                                          motor=motor_type,
                                          ode_solver='scipy.solve_ivp',
                                          load='PolyStaticLoad',
                                          solver_kwargs=solver_kwargs,
                                          subconverters=[conv, conv],
                                          motor_parameter=motor_parameter,
                                          nominal_values=nominal_values,
                                          limit_values=limit_values,
                                          tau=tau,
                                          load_parameter=load_parameter,
                                          dead_time=converter_parameter['dead_time'],
                                          interlocking_time=converter_parameter['interlocking_time'],
                                          u_sup=u_sup)
    else:
        dc_motor_system_1 = DcMotorSystem(converter=conv,
                                          motor=motor_type,
                                          ode_solver='scipy.solve_ivp',
                                          load='PolyStaticLoad',
                                          solver_kwargs=solver_kwargs,
                                          motor_parameter=motor_parameter,
                                          nominal_values=nominal_values,
                                          limit_values=limit_values,
                                          tau=tau,
                                          load_parameter=load_parameter,
                                          dead_time=converter_parameter['dead_time'],
                                          interlocking_time=converter_parameter['interlocking_time'],
                                          u_sup=u_sup)
    # setup the motor
    motor = motor_class(motor_parameter=motor_parameter, nominal_values=nominal_values, limit_values=limit_values)
    # setup the converter
    converter = setup_dc_converter(conv, motor_type)
    # setup the voltage supply
    voltage_supply = IdealVoltageSupply(u_sup)
    # setup the load
    load = PolynomialStaticLoad(load_parameter)
    # setup the integrator
    solver = ScipySolveIvpSolver(method='RK45')
    # setup motor system with pre instantiated models
    dc_motor_system_2 = DcMotorSystem(converter=converter, motor=motor, ode_solver=solver,
                                      supply=voltage_supply, load=load, tau=tau)
    return dc_motor_system_1, dc_motor_system_2, params, nominal_values, limit_values, motor_parameter, \
           converter_parameter


def compare_discrete_dc_motor_simulation(dc_motor_system_1, dc_motor_system_2):
    """
    This function applies random actions to the system and simulates the behaviour for discrete systems.
    It compares if both initializations result in the same states
    :param dc_motor_system_1: motor system form string initialization
    :param dc_motor_system_2: motor system from modularized initialization
    :return:
    """
    seed(123)
    actions = [randint(0, dc_motor_system_1.action_space.n - 1) for _ in range(100)]
    for action in actions:
        assert all(dc_motor_system_1.simulate(action) == dc_motor_system_2.simulate(action)), \
            "The simulated states of both systems are different. Maybe the limits are different: " + \
            str([dc_motor_system_1.limits, dc_motor_system_2.limits])


def compare_continuous_dc_motor_simulation(dc_motor_system_1, dc_motor_system_2):
    """
    This function applies random actions to the system and simulates the behaviour for continuous systems.
    It compares if both initializations result in the same states
    :param dc_motor_system_1: motor system form string initialization
    :param dc_motor_system_2: motor system from modularized initialization
    :return:
    """
    seed(123)
    actions = [uniform(dc_motor_system_1.action_space.low, dc_motor_system_1.action_space.high) for _ in range(100)]
    for action in actions:
        assert all(dc_motor_system_1.simulate(action) == dc_motor_system_2.simulate(action)), \
            "The simulated states of both systems are different. Maybe the limits are different: " + \
            str([dc_motor_system_1.limits, dc_motor_system_2.limits])


def compare_dc_motor_constants(dc_motor_system_1, dc_motor_system_2, kwargs, nominal_values_test, limit_values_test,
                               motor_parameter, converter_parameter):
    """
    This function compares the constants of both initialized motors with the given parameters.
    :param dc_motor_system_1: motor system form string initialization
    :param dc_motor_system_2: motor system from modularized initialization
    :param kwargs: all parameters used for the initialization (dict)
    :param nominal_values_test: nominal values (dict)
    :param limit_values_test: limit values (dict)
    :param motor_parameter: motor parameter (dict)
    :param converter_parameter: converter parameter (dict)
    :return:
    """
    # reset function if the returned state has the correct size
    reset_value = np.zeros(len(dc_motor_system_2.state_names))
    reset_value[-1] = 1
    assert all(dc_motor_system_1.reset() == dc_motor_system_2.reset()) \
           and all(dc_motor_system_2.reset() == reset_value)
    # test if sampling time is the correct
    assert dc_motor_system_1.tau == dc_motor_system_2.tau == kwargs['tau'], 'Tau is wrong' + str([dc_motor_system_1.tau,
                                                                                                  dc_motor_system_2.tau,
                                                                                                  kwargs['tau']])
    # test the state variables
    assert dc_motor_system_1.state_names == dc_motor_system_2.state_names, 'Different state variables'
    # test the shape of the action space
    assert dc_motor_system_1.action_space == dc_motor_system_2.action_space, 'Different dimensions of action spaces'
    # test the values of the state space
    assert all(dc_motor_system_1.state_space.high == dc_motor_system_2.state_space.high) \
           and all(
        dc_motor_system_1.state_space.low == dc_motor_system_2.state_space.low), 'Different action space limits'

    # test limit values
    limit_values_test.update({'u_sup': kwargs['u_sup']})
    limit_values = []
    [limit_values.append(limit_values_test[state]) for state in dc_motor_system_1.state_names]
    assert all(dc_motor_system_1.limits == dc_motor_system_2.limits) and \
           all(dc_motor_system_1.limits == limit_values), "Different limits " + str([dc_motor_system_1.limits,
                                                                                     dc_motor_system_2.limits,
                                                                                     limit_values])
    # test nominal values
    nominal_values_test.update({'u_sup': kwargs['u_sup']})
    nominal_values = []
    [nominal_values.append(nominal_values_test[state]) for state in dc_motor_system_1.state_names]
    assert all(dc_motor_system_1.nominal_state == dc_motor_system_2.nominal_state) \
           and all(dc_motor_system_1.nominal_state == nominal_values), \
        "Nominal value error:" + str([dc_motor_system_1.nominal_state, dc_motor_system_2.nominal_state, nominal_values])
    # test motor parameters
    for key in dc_motor_system_1.electrical_motor.motor_parameter.keys():
        if key in dc_motor_system_2.electrical_motor.motor_parameter.keys():
            assert dc_motor_system_1.electrical_motor.motor_parameter[key] == \
                   dc_motor_system_2.electrical_motor.motor_parameter[key] == motor_parameter[key]
        else:
            f'different parameters in system 1 and system 2'
    # test converter parameter
    assert dc_motor_system_1.converter._interlocking_time == dc_motor_system_2.converter._interlocking_time == \
           converter_parameter['interlocking_time']
    assert dc_motor_system_1.converter._dead_time == dc_motor_system_2.converter._dead_time == \
           converter_parameter['dead_time']
    # test load parameter
    # test voltage supply parameters
    assert dc_motor_system_1.supply.u_nominal == dc_motor_system_2.supply.u_nominal == kwargs['u_sup']


# endregion


# region three phase systems


"""
SynchronousMotorSystem
__init__(representation( aplhabeta/ abc/dq), converter, motor, load, supply, ode_solver,  solver_kwargs, noise_generator, tau, **kwargs)
limits
nominal_state
supply
converter
electrical_motor
mechanical_load
k
tau
state_names
state_position
action_space
state_space


simulate
reset
close
"""


@pytest.mark.parametrize("motor_type", ['PMSM', 'SynRM'])
def test_disc_three_phase(motor_type):
    """
    test the physical system for discrete three phase motors
    :param motor_type: motor name (string)
    :return:
    """
    converter_type = 'Disc-B6C'
    # test default initializations
    default_motor = make_module(ElectricMotor, motor_type)
    default_converter = make_module(PowerElectronicConverter, converter_type)
    default_mechanical_load = make_module(MechanicalLoad, 'PolyStaticLoad')
    default_supply = make_module(VoltageSupply, 'IdealVoltageSupply')
    default_solver = make_module(OdeSolver, 'scipy.solve_ivp')

    default_scml_1 = SynchronousMotorSystem(converter=converter_type,
                                            motor=motor_type)

    default_scml_2 = SynchronousMotorSystem(motor=default_motor,
                                            converter=default_converter,
                                            load=default_mechanical_load,
                                            supply=default_supply,
                                            ode_solver=default_solver)

    default_scml_3 = SynchronousMotorSystem(motor=motor_type,
                                            converter=converter_type,
                                            load='PolyStaticLoad',
                                            supply='IdealVoltageSupply',
                                            ode_solver='scipy.solve_ivp')
    # test parameters for different default initializations
    assert all(default_scml_2.limits == default_scml_3.limits), "Different default limits"
    assert all(default_scml_2.nominal_state == default_scml_3.nominal_state), "Different default nominal values"
    assert default_scml_2.tau == default_scml_3.tau == default_scml_1.tau, "Different sampling times"
    assert default_scml_1.state_names == default_scml_2.state_names == default_scml_3.state_names
    assert default_scml_1.state_positions == default_scml_2.state_positions == default_scml_3.state_positions == \
           dict(omega=0, torque=1, i_a=2, i_b=3, i_c=4, u_a=5, u_b=6, u_c=7, epsilon=8, u_sup=9)
    assert default_scml_1.state_space.shape == default_scml_2.state_space.shape == \
           default_scml_2.state_space.shape == (10,)
    assert default_scml_1.action_space.n == default_scml_2.action_space.n == default_scml_3.action_space.n == 8

    # test functions as reset, simulate, close
    for scml in [default_scml_1, default_scml_2, default_scml_3]:
        assert all(scml.reset() == np.array([0, 0, 0, 0, 0, -1, -1, -1, 0, 1]))
        for action in range(8):
            state = scml.simulate(action)
            assert all(scml.state_space.low <= state / scml.limits) and all(
                state / scml.limits <= scml.state_space.high)
        scml.close()

    # test parametrized initializations
    mp = test_motor_parameter[motor_type]['motor_parameter']
    cp = converter_parameter
    lp = load_parameter['parameter']
    u_sup = mp['u_sup']
    tau = cp['tau']
    # set up instantiated modules
    motor = make_module(ElectricMotor, motor_type, motor_parameter=mp)
    converter = make_module(PowerElectronicConverter, converter_type, converter_parameter=cp)
    mechanical_load = make_module(MechanicalLoad, 'PolyStaticLoad', load_parameter=lp)
    supply = make_module(VoltageSupply, 'IdealVoltageSupply', u_nominal=u_sup, tau=tau)
    solver = make_module(OdeSolver, 'scipy.solve_ivp')
    # intialize parametrized SCML systems
    scml_1 = SynchronousMotorSystem(motor=motor,
                                    converter=converter,
                                    load=mechanical_load,
                                    supply=supply,
                                    ode_solver=solver,
                                    tau=tau)

    scml_2 = SynchronousMotorSystem(motor=motor_type,
                                    converter=converter_type,
                                    load='PolyStaticLoad',
                                    supply='IdealVoltageSupply',
                                    ode_solver='scipy.solve_ivp',
                                    motor_parameter=mp,
                                    converter_parameter=cp,
                                    load_parameter=lp,
                                    tau=tau,
                                    u_sup=u_sup)
    # test parameter
    assert all(scml_1.limits == scml_2.limits), "Different default limits"
    assert all(scml_1.nominal_state == scml_2.nominal_state), "Different default nominal values"
    assert scml_1.tau == scml_2.tau == tau, "Different sampling times"
    assert scml_1.state_names == scml_2.state_names
    assert scml_1.state_positions == scml_2.state_positions == \
           dict(omega=0, torque=1, i_a=2, i_b=3, i_c=4, u_a=5, u_b=6, u_c=7, epsilon=8, u_sup=9)
    assert scml_1.state_space.shape == scml_2.state_space.shape == (10,)
    assert scml_1.action_space.n == scml_2.action_space.n == 8

    # test functions reset, simulate, close
    for scml in [scml_1, scml_2]:
        assert all(scml.reset() == np.array([0, 0, 0, 0, 0, -.75, -.75, -.75, 0, 1]))

        assert all(scml.state_space.high == np.ones(10))
        assert all(scml.state_space.low == np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, 0])), "Wrong state space"

    for action in range(8):
        state_1 = scml_1.simulate(action)
        state_2 = scml_2.simulate(action)
        assert all(state_1 == state_2)
        assert all(scml.state_space.low <= state_1 / scml.limits) and all(
            state_1 / scml.limits <= scml.state_space.high)

    scml_1.close()
    scml_2.close()


@pytest.mark.parametrize("motor_type", ['PMSM', 'SynRM'])
def test_cont_three_phase(motor_type):
    """
    test the physical system for continuous-time three phase motors
    :param motor_type: motor name (string)
    :return:
    """
    converter_type = 'Cont-B6C'
    actions = [[1, 1, 1],
               [0, 0, 0],
               [-1, -1, -1],
               [0.5, -0.5, 0.3],
               [-0.6, 0.5, 0.3]]
    # test default initializations
    default_motor = make_module(ElectricMotor, motor_type)
    default_converter = make_module(PowerElectronicConverter, converter_type)
    default_mechanical_load = make_module(MechanicalLoad, 'PolyStaticLoad')
    default_supply = make_module(VoltageSupply, 'IdealVoltageSupply')
    default_solver = make_module(OdeSolver, 'scipy.solve_ivp')

    default_scml_1 = SynchronousMotorSystem(converter=converter_type,
                                            motor=motor_type)

    default_scml_2 = SynchronousMotorSystem(motor=default_motor,
                                            converter=default_converter,
                                            load=default_mechanical_load,
                                            supply=default_supply,
                                            ode_solver=default_solver)

    default_scml_3 = SynchronousMotorSystem(motor=motor_type,
                                            converter=converter_type,
                                            load='PolyStaticLoad',
                                            supply='IdealVoltageSupply',
                                            ode_solver='scipy.solve_ivp')
    # test parameters for different default initializations
    assert all(default_scml_2.limits == default_scml_3.limits), "Different default limits"
    assert all(default_scml_2.nominal_state == default_scml_3.nominal_state), "Different default nominal values"
    assert default_scml_2.tau == default_scml_3.tau == default_scml_1.tau, "Different sampling times"
    assert default_scml_1.state_names == default_scml_2.state_names == default_scml_3.state_names
    assert default_scml_1.state_positions == default_scml_2.state_positions == default_scml_3.state_positions == \
           dict(omega=0, torque=1, i_a=2, i_b=3, i_c=4, u_a=5, u_b=6, u_c=7, epsilon=8, u_sup=9)

    assert default_scml_1.state_space.shape == default_scml_2.state_space.shape == \
           default_scml_2.state_space.shape == (10,)

    # test functions as reset, simulate, close
    for scml in [default_scml_1, default_scml_2, default_scml_3]:
        assert all(scml.reset() == np.array([0, 0, 0, 0, 0, -1, -1, -1, 0, 1]))

        assert all(default_scml_1.action_space.low == -1 * np.ones(3)) and all(scml.action_space.high == np.ones(3))

        for action in actions:
            state = scml.simulate(action)
            assert all(scml.state_space.low <= state / scml.limits) and all(
                state / scml.limits <= scml.state_space.high)
        scml.close()

    # test parametrized initializations
    mp = test_motor_parameter[motor_type]['motor_parameter']
    cp = converter_parameter
    lp = load_parameter['parameter']
    u_sup = mp['u_sup']
    tau = cp['tau']
    # set up instantiated modules
    motor = make_module(ElectricMotor, motor_type, motor_parameter=mp)
    converter = make_module(PowerElectronicConverter, converter_type, converter_parameter=cp)
    mechanical_load = make_module(MechanicalLoad, 'PolyStaticLoad', load_parameter=lp)
    supply = make_module(VoltageSupply, 'IdealVoltageSupply', u_nominal=u_sup, tau=tau)
    solver = make_module(OdeSolver, 'scipy.solve_ivp')
    # initialize parametrized SCML systems
    scml_1 = SynchronousMotorSystem(motor=motor,
                                    converter=converter,
                                    load=mechanical_load,
                                    supply=supply,
                                    ode_solver=solver,
                                    tau=tau)

    scml_2 = SynchronousMotorSystem(motor=motor_type,
                                    converter=converter_type,
                                    load='PolyStaticLoad',
                                    supply='IdealVoltageSupply',
                                    ode_solver='scipy.solve_ivp',
                                    motor_parameter=mp,
                                    converter_parameter=cp,
                                    load_parameter=lp,
                                    tau=tau,
                                    u_sup=u_sup)
    # test parameter
    assert all(scml_1.limits == scml_2.limits), "Different default limits"
    assert all(scml_1.nominal_state == scml_2.nominal_state), "Different default nominal values"
    assert scml_1.tau == scml_2.tau == tau, "Different sampling times"
    assert scml_1.state_names == scml_2.state_names
    assert scml_1.state_positions == scml_2.state_positions == \
           dict(omega=0, torque=1, i_a=2, i_b=3, i_c=4, u_a=5, u_b=6, u_c=7, epsilon=8, u_sup=9)
    assert scml_1.state_space.shape == scml_2.state_space.shape == (10,)

    # test functions reset, simulate, close
    for scml in [scml_1, scml_2]:
        assert all(scml.reset() == np.array([0, 0, 0, 0, 0, -.75, -.75, -.75, 0, 1]))
        assert all(default_scml_1.action_space.low == -1 * np.ones(3)) and all(scml.action_space.high == np.ones(3))

        assert all(scml.state_space.high == np.ones(10))
        assert all(scml.state_space.low == np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, 0])), "Wrong state space"

    for index, action in enumerate(actions):
        state_1 = scml_1.simulate(action)
        state_2 = scml_2.simulate(action)
        assert all(state_1 == state_2)
        assert all(scml.state_space.low <= state_1 / scml.limits) and all(
            state_1 / scml.limits <= scml.state_space.high)
        assert scml.k == index + 1

    scml_1.close()
    scml_2.close()
# endregion
