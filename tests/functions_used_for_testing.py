from .conf import *
from gym_electric_motor.physical_systems import *
from gym_electric_motor.reward_functions import *
from gym_electric_motor.utils import make_module
from gym_electric_motor.reference_generators import *


def setup_physical_system(motor_type, converter_type, three_phase=False):
    """
    Function to set up a physical system with test parameters
    :param motor_type: motor name (string)
    :param converter_type: converter name (string)
    :param three_phase: if True, than a synchronous motor system will be instantiated
    :return: instantiated physical system
    """
    # get test parameter
    tau = converter_parameter['tau']
    u_sup = test_motor_parameter[motor_type]['motor_parameter']['u_sup']
    motor_parameter = test_motor_parameter[motor_type]['motor_parameter']  # dict
    nominal_values = test_motor_parameter[motor_type]['nominal_values']  # dict
    limit_values = test_motor_parameter[motor_type]['limit_values']  # dict
    # setup load
    load = PolynomialStaticLoad(load_parameter=load_parameter['parameter'])
    # setup voltage supply
    voltage_supply = IdealVoltageSupply(u_sup)
    # setup converter
    if motor_type == 'DcExtEx':
        if 'Disc' in converter_type:
            double_converter = 'Disc-Double'
        else:
            double_converter = 'Cont-Double'
        converter = make_module(PowerElectronicConverter, double_converter,
                                subconverters=[converter_type, converter_type],
                                tau=converter_parameter['tau'],
                                dead_time=converter_parameter['dead_time'],
                                interlocking_time=converter_parameter['interlocking_time'])
    else:
        converter = make_module(PowerElectronicConverter, converter_type, tau=converter_parameter['tau'],
                                dead_time=converter_parameter['dead_time'],
                                interlocking_time=converter_parameter['interlocking_time'])
    # setup motor
    motor = make_module(ElectricMotor, motor_type, motor_parameter=motor_parameter, nominal_values=nominal_values,
                        limit_values=limit_values)
    # setup solver
    solver = ScipySolveIvpSolver(method='RK45')
    # combine all modules to a physical system
    if three_phase:
        physical_system = SynchronousMotorSystem(converter=converter, motor=motor, ode_solver=solver,
                                        supply=voltage_supply, load=load, tau=tau)
    else:
        physical_system = DcMotorSystem(converter=converter, motor=motor, ode_solver=solver,
                                        supply=voltage_supply, load=load, tau=tau)
    return physical_system


def setup_reference_generator(reference_type, physical_system, reference_state='omega'):
    """
    Function to setup the reference generator
    :param reference_type: name of reference generator
    :param physical_system: instantiated physical system
    :param reference_state: referenced state name (string)
    :return: instantiated reference generator
    """
    reference_generator = make_module(ReferenceGenerator, reference_type, reference_state=reference_state)
    reference_generator.set_modules(physical_system)
    reference_generator.reset()
    return reference_generator


def setup_reward_function(reward_function_type, physical_system, reference_generator, reward_weights, observed_states):
    reward_function = make_module(RewardFunction, reward_function_type, observed_states=observed_states,
                                  reward_weights=reward_weights)
    reward_function.set_modules(physical_system, reference_generator)
    return reward_function


def setup_dc_converter(conv, motor_type):
    """
    This function initializes the converter.
    It differentiates between single and double converter and can be used for discrete and continuous converters.
    :param conv: converter name (string)
    :param motor_type: motor name (string)
    :return: initialized converter
    """
    if motor_type == 'DcExtEx':
        # setup double converter
        if 'Disc' in conv:
            double_converter = 'Disc-Double'
        else:
            double_converter = 'Cont-Double'
        converter = make_module(PowerElectronicConverter, double_converter,
                                interlocking_time=converter_parameter['interlocking_time'],
                                dead_time=converter_parameter['dead_time'],
                                subconverters=[make_module(PowerElectronicConverter, conv,
                                                           tau=converter_parameter['tau'],
                                                           dead_time=converter_parameter['dead_time'],
                                                           interlocking_time=converter_parameter['interlocking_time']),
                                               make_module(PowerElectronicConverter, conv,
                                                           tau=converter_parameter['tau'],
                                                           dead_time=converter_parameter['dead_time'],
                                                           interlocking_time=converter_parameter['interlocking_time'])])
    else:
        # setup single converter
        converter = make_module(PowerElectronicConverter, conv,
                                tau=converter_parameter['tau'],
                                dead_time=converter_parameter['dead_time'],
                                interlocking_time=converter_parameter['interlocking_time'])
    return converter




