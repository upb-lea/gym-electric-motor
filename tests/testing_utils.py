from .conf import *
from gym_electric_motor.physical_systems import *
from gym_electric_motor.utils import make_module, set_state_array
from gym_electric_motor import ReferenceGenerator, RewardFunction, PhysicalSystem, ElectricMotorVisualization, \
    ConstraintMonitor
from gym_electric_motor.physical_systems import PowerElectronicConverter, MechanicalLoad, ElectricMotor, OdeSolver, \
    VoltageSupply, NoiseGenerator
import gym_electric_motor.physical_systems.converters as cv
from gym_electric_motor.physical_systems.physical_systems import SCMLSystem
import numpy as np
from gym.spaces import Box, Discrete
from scipy.integrate import ode
from tests.conf import system, jacobian, permex_motor_parameter
from gym_electric_motor.utils import instantiate
from gym_electric_motor.core import Callback


# region first version


def setup_physical_system(motor_type, converter_type, subconverters=None, three_phase=False):
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
            double_converter = 'Disc-Multi'
        else:
            double_converter = 'Cont-Multi'
        converter = make_module(PowerElectronicConverter, double_converter,
                                subconverters=[converter_type, converter_type],
                                tau=converter_parameter['tau'],
                                dead_time=converter_parameter['dead_time'],
                                interlocking_time=converter_parameter['interlocking_time'])
    else:
        converter = make_module(PowerElectronicConverter, converter_type,
                                subconverters=subconverters,
                                tau=converter_parameter['tau'],
                                dead_time=converter_parameter['dead_time'],
                                interlocking_time=converter_parameter['interlocking_time'])
    # setup motor
    motor = make_module(ElectricMotor, motor_type, motor_parameter=motor_parameter, nominal_values=nominal_values,
                        limit_values=limit_values)
    # setup solver
    solver = ScipySolveIvpSolver(method='RK45')
    # combine all modules to a physical system
    if three_phase:
        if motor_type == "SCIM":
            physical_system = SquirrelCageInductionMotorSystem(converter=converter, motor=motor, ode_solver=solver,
                                                                supply=voltage_supply, load=load, tau=tau)
        elif motor_type == "DFIM":
            physical_system = DoublyFedInductionMotor(converter=converter, motor=motor, ode_solver=solver,
                                                               supply=voltage_supply, load=load, tau=tau)
        else:
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


def setup_dc_converter(conv, motor_type, subconverters=None):
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
            double_converter = 'Disc-Multi'
        else:
            double_converter = 'Cont-Multi'
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
                                subconverters=subconverters,
                                tau=converter_parameter['tau'],
                                dead_time=converter_parameter['dead_time'],
                                interlocking_time=converter_parameter['interlocking_time'])
    return converter


# endregion

# region second version

instantiate_dict = {}


def mock_instantiate(superclass, key, **kwargs):
    # Instantiate the object and log the passed and returned values to validate correct function calls
    instantiate_dict[superclass] = {}
    instantiate_dict[superclass]['key'] = key
    inst = instantiate(superclass, key, **kwargs)
    instantiate_dict[superclass]['instance'] = inst
    return inst


class DummyReferenceGenerator(ReferenceGenerator):
    _reset_counter = 0

    def __init__(self, reference_observation=np.array([1.]), reference_state='dummy_state_0', **kwargs):
        super().__init__()
        self.reference_space = Box(0., 1., shape=(1,), dtype=np.float64)
        self.kwargs = kwargs
        self._reference_names = [reference_state]
        self.closed = False
        self.physical_system = None
        self.get_reference_state = None
        self.get_reference_obs_state = None
        self.trajectory = np.sin(np.linspace(0, 50, 100))
        self._reference_state = reference_state
        self.reference_observation = reference_observation
        self.reference_array = None
        self.kwargs = kwargs

    def set_modules(self, physical_system):
        self.physical_system = physical_system
        self.reference_array = np.ones_like(physical_system.state_names).astype(float)
        super().set_modules(physical_system)
        self._referenced_states = set_state_array(
            {self._reference_state: 1}, physical_system.state_names
        ).astype(bool)

    def reset(self, initial_state=None, initial_reference=None):
        self._reset_counter += 1
        res = super().reset(initial_state, initial_reference)
        return res[0], res[1], self.trajectory

    def get_reference(self, state, *_, **__):
        self.get_reference_state = state
        return self.reference_array

    def get_reference_observation(self, state, *_, **__):
        self.get_reference_obs_state = state
        return self.reference_observation

    def close(self):
        self.closed = True
        super().close()


class DummyRewardFunction(RewardFunction):

    def __init__(self, **kwargs):
        self.last_state = None
        self.last_reference = None
        self.last_action = None
        self.last_time_step = None
        self.closed = False
        self.done = False
        self.kwargs = kwargs
        super().__init__()

    def reset(self, initial_state=None, initial_reference=None):
        self.last_state = initial_state
        self.last_reference = initial_reference
        super().reset(initial_state, initial_reference)

    def reward(self, state, reference, k=None, action=None, violation_degree=0.0):
        self.last_state = state
        self.last_reference = reference
        self.last_action = action
        self.last_time_step = k
        return -1 if violation_degree == 1 else 1

    def close(self):
        self.closed = True
        super().close()

    def _limit_violation_reward(self, state):
        pass

    def _reward(self, state, reference, action):
        pass


class DummyPhysicalSystem(PhysicalSystem):

    @property
    def limits(self):
        """
        Returns:
             ndarray(float): An array containing the maximum allowed physical values for each state variable.
        """
        return self._limits

    @property
    def nominal_state(self):
        """
        Returns:
             ndarray(float): An array containing the nominal values for each state variable.
        """
        return self._nominal_values

    def __init__(self, state_length=1, state_names='dummy_state', **kwargs):
        super().__init__(
            Box(-1, 1, shape=(1,), dtype=np.float64), Box(-1, 1, shape=(state_length,), dtype=np.float64),
            [f'{state_names}_{i}' for i in range(state_length)], 1
        )
        self._limits = np.array([10 * (i + 1) for i in range(state_length)])
        self._nominal_values = np.array([(i + 1) for i in range(state_length)])
        self.action = None
        self.state = None
        self.closed = False
        self.kwargs = kwargs

    def reset(self, initial_state=None):
        self.state = np.array([0.] * len(self._state_names))
        return self.state

    def simulate(self, action):
        self.action = action
        self.state = np.array([action * (i + 1) for i in range(len(self._state_names))])
        return self.state

    def close(self):
        self.closed = True
        super().close()


class DummyVisualization(ElectricMotorVisualization):

    def __init__(self, **kwargs):
        self.closed = False
        self.state = None
        self.reference = None
        self.reward = None
        self.reference_trajectory = None
        self.physical_system = None
        self.reference_generator = None
        self.reward_function = None
        self.kwargs = kwargs
        super().__init__()

    def step(self, state, reference, reward, *_, **__):
        self.state = state
        self.reference = reference
        self.reward = reward

    def reset(self, reference_trajectories=None, *_, **__):
        self.reference_trajectory = reference_trajectories

    def set_modules(self, physical_system, reference_generator, reward_function):
        self.physical_system = physical_system
        self.reference_generator = reference_generator
        self.reward_function = reward_function


class DummyVoltageSupply(VoltageSupply):

    def __init__(self, u_nominal=560, tau=1e-4, **kwargs):
        super().__init__(u_nominal)
        self.i_sup = None
        self.t = None
        self.reset_counter = 0
        self.args = None
        self.kwargs = kwargs
        self.get_voltage_counter = 0

    def reset(self):
        self.reset_counter += 1
        return super().reset()

    def get_voltage(self, i_sup, t, *args, **kwargs):
        self.get_voltage_counter += 1
        self.i_sup = i_sup
        self.t = t
        self.args = args
        self.kwargs = kwargs
        return [self._u_nominal]


class DummyConverter(PowerElectronicConverter):

    voltages = Box(0, 1, shape=(1,), dtype=np.float64)
    currents = Box(-1, 1, shape=(1,), dtype=np.float64)
    action_space = Discrete(4)

    def __init__(self, tau=2E-4, dead_time=False, interlocking_time=0, action_space=None, voltages=None, currents=None, **kwargs):
        super().__init__(tau, dead_time, interlocking_time)
        self.action_space = action_space or self.action_space
        self.voltages = voltages or self.voltages
        self.currents = currents or self.currents
        self.reset_counter = 0
        self.convert_counter = 0
        self.switching_times = [tau]
        self.action = None
        self.action_set_time = None
        self.i_out = None
        self.last_i_out = None
        self.t = None
        self.kwargs = kwargs
        self.u_in = None

    def i_sup(self, i_out):
        self.last_i_out = i_out
        return i_out[0]

    def set_switching_times(self, switching_times):
        self.switching_times = switching_times

    def set_action(self, action, t):
        self.action_set_time = t
        self.action = action
        return [t + self._tau / 2, t + self._tau]

    def reset(self):
        self.reset_counter += 1
        return [0.0] * self.voltages.shape[0]

    def convert(self, i_out, t):
        self.i_out = i_out
        self.t = t
        self.convert_counter += 1
        self.u_in = [self.action] if type(self.action_space) is Discrete else self.action
        return self.u_in


class DummyElectricMotor(ElectricMotor):

    # defined test values
    _default_motor_parameter = permex_motor_parameter['motor_parameter']
    _default_limits = dict(omega=16, torque=26, u=15, i=26, i_0=26, i_1=21, u_0=15)
    _default_nominal_values = dict(omega=14, torque=20, u=15, i=22, i_0=22, i_1=20)
    HAS_JACOBIAN = True
    electrical_jac_return = None
    CURRENTS_IDX = [0, 1]
    CURRENTS = ['i_0', 'i_1']
    VOLTAGES = ['u_0']

    def __init__(self, tau=1e-5, **kwargs):
        self.kwargs = kwargs
        self.reset_counter = 0
        self.u_in = None
        super().__init__(**kwargs)

    def electrical_ode(self, state, u_in, omega, *_):
        self.u_in = u_in
        return state - u_in

    def reset(self, state_space, state_positions):
        self.reset_counter += 1
        return super().reset(state_space, state_positions)

    def torque(self, currents):
        return np.prod(currents)

    def i_in(self, state):
        return [np.sum(state)]

    def electrical_jacobian(self, state, u_in, omega, *_):
        return self.electrical_jac_return


class PowerElectronicConverterWrapper(cv.PowerElectronicConverter):

    def __init__(self, subconverter, **kwargs):
        super().__init__(**kwargs)
        self._converter = subconverter
        self.action_space = self._converter.action_space
        self.currents = self._converter.currents
        self.voltages = self._converter.voltages

        self.reset_calls = 0
        self.set_action_calls = 0
        self.last_action = None
        self.last_t = None
        self.last_i_out = None
        self.last_u = None
        self.last_i_sup = None

    def reset(self):
        self.reset_calls += 1
        return self._converter.reset()

    def set_action(self, action, t):
        self.last_action = action
        self.last_t = t
        return self._converter.set_action(action, t)

    def convert(self, i_out, t):
        self.last_i_out = i_out
        self.last_t = t
        self.last_u = self._converter.convert(i_out, t)
        return self.last_u

    def i_sup(self, i_out):
        self.last_i_out = i_out
        self.last_i_sup = self._converter.i_sup(i_out)
        return self.last_i_sup


class DummyScipyOdeSolver(ode):
    """
    Dummy class for ScipyOdeSolver
    """
    # defined test values
    _kwargs = {'nsteps': 5}
    _integrator = 'dop853'
    _y = np.zeros(2)
    _y_init = np.array([1, 6])
    _t = 0
    _tau = 1e-3
    _t_init = 0.1
    jac = None

    # counter
    _init_counter = 0
    _set_integrator_counter = 0
    _set_initial_value_counter = 0
    _set_f_params_counter = 0
    _set_jac_params_counter = 0
    _integrate_counter = 0

    def __init__(self, system_equation, jacobian_):
        self._init_counter += 1
        assert system_equation == system
        assert jacobian_ == jacobian
        super().__init__(system_equation, jacobian_)

    def set_integrator(self, integrator, **args):
        self._set_integrator_counter += 1
        assert integrator == self._integrator
        assert args == self._kwargs
        return super().set_integrator(integrator, **args)

    def set_initial_value(self, y, t=0.0):
        self._set_initial_value_counter += 1
        assert all(y == self._y_init)
        assert t == self._t_init

    def set_f_params(self, *args):
        self._set_f_params_counter += 1
        assert args == (2,)
        super().set_f_params(2)

    def set_jac_params(self, *args):
        self._set_jac_params_counter += 1
        assert args == (2,)
        super().set_jac_params(*args)

    def integrate(self, t, *_):
        self._integrate_counter += 1
        assert t == self._t_init + self._tau
        return self._y_init * 2


class DummyLoad(MechanicalLoad):
    """
    dummy class for mechanical load
    """
    state_names = ['omega', 'position']
    limits = dict(omega=15, position=10)
    nominal_values = dict(omega=15, position=10)
    mechanical_state = None
    t = None
    mechanical_ode_return = None
    mechanical_jac_return = None
    omega_range = None
    HAS_JACOBIAN = True

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.reset_counter = 0
        super().__init__(**kwargs)

    def reset(self, state_space, state_positions, nominal_state,  *_, **__):
        self.reset_counter += 1
        return np.zeros(2)

    def mechanical_ode(self, t, mechanical_state, torque):
        self.mechanical_state = mechanical_state
        self.t = t
        self.mechanical_ode_return = np.array([torque, -torque])
        return self.mechanical_ode_return

    def mechanical_jacobian(self, t, mechanical_state, torque):
        self.mechanical_state = mechanical_state
        self.t = t
        self.mechanical_ode_return = np.array([torque, -torque])
        return self.mechanical_jac_return

    def get_state_space(self, omega_range):
        self.omega_range = omega_range
        return {'omega': 0, 'position': -1}, {'omega': 1, 'position': -1}


class DummyNoise(NoiseGenerator):
    """
    dummy class for noise generator
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.reset_counter = 0
        super().__init__()

    def reset(self):
        return np.ones_like(self._state_variables, dtype=float) * 0.36

    def noise(self, *_, **__):
        return np.ones_like(self._state_variables, dtype=float) * 0.42


class DummyOdeSolver(OdeSolver):
    """
    Dummy class for ode solver
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__()

    def integrate(self, t):
        self.last_y = self._y
        self._y = self._y + t - self._t
        self._t = t
        return self._y


class DummyConstraint(Constraint):

    def __init__(self, violation_degree=0.0):
        super().__init__()
        self.modules_set = False
        self.violation_degree = violation_degree

    def __call__(self, state):
        return self.violation_degree

    def set_modules(self, ps):
        super().set_modules(ps)
        self.modules_set = True


class DummyConstraintMonitor(ConstraintMonitor):

    def __init__(self, no_of_dummy_constraints=1):
        constraints = [DummyConstraint() for _ in range(no_of_dummy_constraints)]
        super().__init__(additional_constraints=constraints)


class DummySCMLSystem(SCMLSystem):
    """
    dummy class for SCMLSystem
    """
    # defined test values
    OMEGA_IDX = 0
    TORQUE_IDX = 1
    CURRENTS_IDX = []
    VOLTAGES_IDX = []
    U_SUP_IDX = -1

    _limits = {}
    _nominal_state = {}
    _supply = None
    _converter = None
    _electrical_motor = None
    _mechanical_load = None

    _state_names = ['omega_me', 'torque', 'u', 'i', 'u_sup']
    _state_length = 5

    # counter
    _set_limits_counter = 0
    _set_nominal_state_counter = 0

    def _set_limits(self):
        self._set_limits_counter += 1

    def _set_nominal_state(self):
        self._set_nominal_state_counter += 1

    def _build_state_space(self, state_names):
        assert state_names == self._state_names
        return None

    def _build_state_names(self):
        return self._state_names

    def _set_indices(self):
        pass

    def simulate(self, action, *_, **__):
        return np.ones(self._state_length) * 0.46

    def _system_equation(self, t, state, u_in, **__):
        return np.ones(self._state_length) * 0.87

    def reset(self, *_):
        return np.ones(self._state_length) * 0.12

    def _forward_transform(self, quantities, motor_state):
        return quantities

    def _build_state(self, motor_state, torque, u_in, u_sup):
        pass

    def _action_transformation(self, action):
        return action


class DummyRandom:
    _expected_low = None
    _expected_high = None
    _expected_left = None
    _expected_mode = None
    _expected_right = None
    _expected_values = None
    _expected_probabilities = None
    _expected_loc = None
    _expected_scale = None
    _expected_size = None

    # counter
    _monkey_random_rand_counter = 0
    _monkey_random_triangular_counter = 0
    _monkey_random_randint_counter = 0
    _monkey_random_choice_counter = 0
    _monkey_random_normal_counter = 0

    def __init__(self, exp_low=None, exp_high=None, exp_left=None, exp_right=None, exp_mode=None, exp_values=None,
                 exp_probabilities=None, exp_loc=None, exp_scale=None, exp_size=None):
        """
        set expected values
        :param exp_low: expected lower value
        :param exp_high: expected upper value
        :param exp_mode: expected mode value
        :param exp_right: expected right value
        :param exp_left: expected left value
        :param exp_values: expected values for choice
        :param exp_probabilities: expected probabilities for choice
        :param exp_loc: expected loc value
        :param exp_scale: expected scale value
        :param exp_size: expected size value
        """
        self._expected_low = exp_low
        self._expected_high = exp_high
        self._expected_mode = exp_mode
        self._expected_left = exp_left
        self._expected_right = exp_right
        self._expected_values = exp_values
        self._expected_probabilities = exp_probabilities
        self._expected_loc = exp_loc
        self._expected_scale = exp_scale
        self._expected_size = exp_size

    def monkey_random_rand(self):
        self._monkey_random_rand_counter += 1
        """
        mock function for np.random.rand()
        :return:
        """
        return 0.25

    def monkey_random_triangular(self, left, mode, right):
        self._monkey_random_triangular_counter += 1
        if self._expected_left is not None:
            assert left == self._expected_left
        if self._expected_high is not None:
            assert right == self._expected_right
        if self._expected_mode is not None:
            assert mode == self._expected_mode
        """
        mock function for np.random.triangular()
        :return:
        """
        return 0.45

    def monkey_random_randint(self, low, high):
        if self._expected_low is not None:
            assert low == self._expected_low
        if self._expected_high is not None:
            assert high == self._expected_high
        self._monkey_random_randint_counter += 1
        """
        mock function for random.randint()
        :param low:
        :param high:
        :return:
        """
        return 7

    def monkey_random_choice(self, a, p):
        self._monkey_random_choice_counter += 1
        assert len(a) == len(p)
        if self._expected_values is not None:
            assert a == self._expected_values
        if self._expected_probabilities is not None:
            assert p == self._expected_probabilities
        return a[0]

    def monkey_random_normal(self, loc=0, scale=1.0, size=None):
        if self._expected_loc is not None:
            assert loc == self._expected_loc
        if self._expected_scale is not None:
            assert scale == self._expected_scale
        if self._expected_size is not None:
            assert size == self._expected_size
        else:
            size = 1
        self._monkey_random_normal_counter += 1
        result = np.array([0.1, -0.2, 0.6, 0.1, -0.5, -0.3, -1.7, 0.1, -0.2, 0.4])
        return result[:size]


class DummyElectricMotorEnvironment(ElectricMotorEnvironment):
    """Dummy environment to test pre implemented callbacks. Extend for further testing cases"""
    
    def __init__(self, reference_generator=None, callbacks=(), **kwargs):
        reference_generator = reference_generator or DummyReferenceGenerator()
        super().__init__(DummyPhysicalSystem(), reference_generator, DummyRewardFunction(), callbacks=callbacks)
    
    def step(self):
        self._call_callbacks('on_step_begin', 0, 0)
        self._call_callbacks('on_step_end', 0, 0, 0, 0, 0)
            
    def reset(self):
        self._call_callbacks('on_reset_begin')
        self._call_callbacks('on_reset_end', 0, 0)
        
    def close(self):
        self._call_callbacks(self._callbacks, 'on_close')


class DummyCallback(Callback):
    
    def __init__(self):
        super().__init__()
        self.reset_begin = 0
        self.reset_end = 0
        self.step_begin = 0
        self.step_end = 0
        self.close = 0
    
    def on_reset_begin(self):
        self.reset_begin += 1

    def on_reset_end(self, *_):
        self.reset_end += 1

    def on_step_begin(self, *_):
        self.step_begin += 1

    def on_step_end(self, *_):
        self.step_end += 1

    def on_close(self):
        self.close += 1
