from gym_electric_motor.envs.gym_dcm import *
from gym_electric_motor.envs.gym_pmsm import *
from gym_electric_motor.envs.gym_synrm import *
from gym_electric_motor.envs.gym_im import *
from gym_electric_motor.envs.gym_dcm.dc_extex_motor_env import *
import gym_electric_motor.envs.gym_dcm.dc_extex_motor_env as extexenv
import gym_electric_motor.envs.gym_dcm.dc_permex_motor_env as permexenv
import gym_electric_motor.envs.gym_dcm.dc_series_motor_env as seriesenv
import gym_electric_motor.envs.gym_dcm.dc_shunt_motor_env as shuntenv
import gym_electric_motor.envs.gym_pmsm.perm_mag_syn_motor_env as pmsmenv
import gym_electric_motor.envs.gym_synrm.syn_reluctance_motor_env as synrmenv
import gym_electric_motor.envs.gym_im.squirrel_cage_induc_motor_env as scimenv
import gym_electric_motor.envs.gym_im.doubly_fed_induc_motor_env as dfimenv

from ..testing_utils import *
from ..conf import *
from numpy.random import seed
import pytest

# region first tests


g_disc_converter = ['Disc-1QC', 'Disc-2QC', 'Disc-4QC']
g_cont_converter = ['Cont-1QC', 'Cont-2QC', 'Cont-4QC']
g_converter = ['Disc-1QC', 'Disc-2QC', 'Disc-4QC', 'Cont-1QC', 'Cont-2QC', 'Cont-4QC']
g_reference_generator = ['WienerProcessReference', 'SinusReference', 'StepReference']
g_reward_functions = ['WSE']


@pytest.mark.parametrize("converter_type", g_cont_converter)
@pytest.mark.parametrize("reward_function_type", g_reward_functions)
@pytest.mark.parametrize("reference_generator_type", g_reference_generator)
@pytest.mark.parametrize("tau_test", [1E-5, 1E-4, 1E-3])
@pytest.mark.parametrize(
    "motor_type, state_filter, motor_class, initializer", [
        ('DcPermEx', ['omega', 'u', 'i'], ContDcPermanentlyExcitedMotorEnvironment, permex_initializer),
        ('DcSeries', ['torque'], ContDcSeriesMotorEnvironment, series_initializer),
        ('DcShunt', ['omega'], ContDcShuntMotorEnvironment, shunt_initializer),
        ('DcExtEx', ['torque'], ContDcExternallyExcitedMotorEnvironment, extex_initializer)
    ]
)
def test_cont_dc_motor_environments(
        motor_type, converter_type, reward_function_type, reference_generator_type, tau_test, state_filter,
        motor_class, turn_off_windows, initializer
):
    """
    test cont dc motor environments

    :param motor_type: motor name (string)
    :param converter_type: converter name (string)
    :param reward_function_type: reward function name (string)
    :param reference_generator_type: reference generator name (string)
    :param tau_test: sampling time
    :param state_filter: states shown in the observation
    :param motor_class: class of motor
    :return:
    """
    # setup the parameter
    conv_parameter = converter_parameter.copy()
    conv_parameter.update({'tau': tau_test})
    tau = conv_parameter['tau']

    solver_kwargs = {}
    motor_parameter = test_motor_parameter[motor_type]['motor_parameter']
    nominal_values = test_motor_parameter[motor_type]['nominal_values']  # dict
    limit_values = test_motor_parameter[motor_type]['limit_values']  # dict
    u_sup = motor_parameter['u_sup']
    reward_weights = test_motor_parameter[motor_type]['reward_weights']
    # setup converter parameter
    kwargs = {}
    if motor_type == 'DcExtEx':
        converter_types = 'Cont-Multi'
        kwargs.update({'subconverters': [converter_type, converter_type]})
    else:
        converter_types = converter_type
    # different initializations
    env_default = motor_class()
    env_1 = motor_class(tau=tau,
                        converter=converter_types,
                        reward_function=reward_function_type,
                        reference_generator=reference_generator_type,
                        visualization=[],
                        state_filter=state_filter,
                        ode_solver='euler',
                        load='PolyStaticLoad',
                        solver_kwargs=solver_kwargs,
                        motor_parameter=motor_parameter,
                        nominal_values=nominal_values,
                        limit_values=limit_values,
                        load_parameter=load_parameter['parameter'],
                        dead_time=conv_parameter['dead_time'],
                        interlocking_time=conv_parameter['interlocking_time'],
                        u_sup=u_sup,
                        reward_weights=reward_weights,
                        motor_initializer=initializer,
                        **kwargs
                        )

    _physical_system = setup_physical_system(motor_type, converter_type)
    reference_generator = make_module(ReferenceGenerator, reference_generator_type,
                                      reference_state='omega')
    reward_function = make_module(RewardFunction, reward_function_type,
                                  reward_weights=reward_weights)
    converter = setup_dc_converter(converter_type, motor_type)  # not used so far
    voltage_supply = IdealVoltageSupply(u_sup)
    load = PolynomialStaticLoad(load_parameter['parameter'])
    solver = EulerSolver()
    env_2 = motor_class(tau=tau, converter=converter,
                        load=load, solver=solver,
                        supply=voltage_supply,
                        reference_generator=reference_generator,
                        reward_function=reward_function,
                        state_filter=state_filter,
                        motor_initializer=initializer,)
    # test the different initializations
    envs = [env_default, env_1, env_2]
    for index, env in enumerate(envs):
        env.reset()
        action_space = env.action_space
        for k in range(10):
            action = action_space.sample()
            obs, reward, done, _ = env.step(action)
            reward_range = env.reward_function.reward_range
            if not done:
                assert reward_range[0] <= reward <= reward_range[1], "Reward out of range " + str([reward_range[0],
                                                                                                  reward,
                                                                                                  reward_range[1],
                                                                                                  index])
            else:
                assert reward == env.reward_function._violation_reward
                break
            env.render()
        env.close()


@pytest.mark.parametrize("converter_type", g_disc_converter)
@pytest.mark.parametrize("reward_function_type", g_reward_functions)
@pytest.mark.parametrize("reference_generator_type", g_reference_generator)
@pytest.mark.parametrize("tau_test", [1E-5, 1E-4, 1E-3])
@pytest.mark.parametrize("motor_type, state_filter, motor_class, initializer",
                         [('DcPermEx', ['omega', 'u', 'i'], DiscDcPermanentlyExcitedMotorEnvironment, permex_initializer),
                          ('DcSeries', ['torque'], DiscDcSeriesMotorEnvironment, series_initializer),
                          ('DcShunt', ['omega'], DiscDcShuntMotorEnvironment, shunt_initializer),
                          ('DcExtEx', ['torque'], DiscDcExternallyExcitedMotorEnvironment, extex_initializer)])
def test_disc_dc_motor_environments(motor_type, converter_type, reward_function_type,
                                    reference_generator_type, tau_test, state_filter,
                                    motor_class, turn_off_windows, initializer):
    """
    test disc dc motor environments
    :param motor_type: motor name (stirng)
    :param converter_type: converter name (string)
    :param reward_function_type: reward function name (string)
    :param reference_generator_type: reference generator name (string)
    :param tau_test: sampling time
    :param state_filter: states shown in the observation
    :param motor_class: class of motor
    :return:
    """
    # setup the parameter
    conv_parameter = converter_parameter.copy()
    conv_parameter.update({'tau': tau_test})
    tau = conv_parameter['tau']

    solver_kwargs = {}
    motor_parameter = test_motor_parameter[motor_type]['motor_parameter']
    nominal_values = test_motor_parameter[motor_type]['nominal_values']  # dict
    limit_values = test_motor_parameter[motor_type]['limit_values']  # dict
    u_sup = motor_parameter['u_sup']
    reward_weights = test_motor_parameter[motor_type]['reward_weights']
    # setup converter parameter
    kwargs = {}
    if motor_type == 'DcExtEx':
        converter_types = 'Disc-Multi'
        kwargs.update({'subconverters': [converter_type, converter_type]})
    else:
        converter_types = converter_type
    # different initializations
    env_default = motor_class()
    env_1 = motor_class(tau=tau,
                        converter=converter_types,
                        reward_function=reward_function_type,
                        reference_generator=reference_generator_type,
                        visualization=[],
                        state_filter=state_filter,
                        ode_solver='euler',
                        load='PolyStaticLoad',
                        solver_kwargs=solver_kwargs,
                        motor_parameter=motor_parameter,
                        nominal_values=nominal_values,
                        limit_values=limit_values,
                        load_parameter=load_parameter['parameter'],
                        dead_time=conv_parameter['dead_time'],
                        interlocking_time=conv_parameter['interlocking_time'],
                        u_sup=u_sup,
                        reward_weights=reward_weights,
                        motor_initializer=initializer,
                        **kwargs
                        )

    _physical_system = setup_physical_system(motor_type, converter_type)
    reference_generator = make_module(ReferenceGenerator, reference_generator_type,
                                      reference_state='omega')
    reward_function = make_module(RewardFunction, reward_function_type,
                                  reward_weights=reward_weights)
    converter = setup_dc_converter(converter_type, motor_type)  # not used so far
    voltage_supply = IdealVoltageSupply(u_sup)
    load = PolynomialStaticLoad(load_parameter['parameter'])
    solver = EulerSolver()
    env_2 = motor_class(tau=tau, converter=converter,
                        load=load, solver=solver,
                        supply=voltage_supply,
                        reference_generator=reference_generator,
                        reward_function=reward_function,
                        state_filter=state_filter,
                        motor_initializer=initializer,)
    # test the different initializations
    envs = [env_default, env_1, env_2]
    for index, env in enumerate(envs):
        env.reset()
        action_space = env.action_space
        for k in range(10):
            action = action_space.sample()
            obs, reward, done, _ = env.step(action)
            reward_range = env.reward_function.reward_range
            if not done:
                assert reward_range[0] <= reward <= reward_range[1], "Reward out of range " + str([reward_range[0],
                                                                                                   reward,
                                                                                                   reward_range[1],
                                                                                                   index])
            else:
                reward == env.reward_function._violation_reward
                break
            env.render()
        env.close()


@pytest.mark.parametrize("reward_function_type", g_reward_functions)
@pytest.mark.parametrize("reference_generator_type", g_reference_generator)
@pytest.mark.parametrize("tau_test", [1E-5, 1E-4, 1E-3])
@pytest.mark.parametrize("motor_type, motor_class, time_type, initializer",
                         [('PMSM', ContPermanentMagnetSynchronousMotorEnvironment, 'Cont', pmsm_initializer),
                          ('PMSM', DiscPermanentMagnetSynchronousMotorEnvironment, 'Disc', pmsm_initializer),
                          ('SynRM', ContSynchronousReluctanceMotorEnvironment, 'Cont', synrm_initializer),
                          ('SynRM', DiscSynchronousReluctanceMotorEnvironment, 'Disc', synrm_initializer),
                          ('SCIM', ContSquirrelCageInductionMotorEnvironment, 'Cont', sci_initializer),
                          ('SCIM', DiscSquirrelCageInductionMotorEnvironment, 'Disc', sci_initializer),
                          ('DFIM', ContDoublyFedInductionMotorEnvironment, 'Cont', dfim_initializer),
                          ('DFIM', DiscDoublyFedInductionMotorEnvironment, 'Disc', dfim_initializer),
                          ])
def test_threephase_environments(reward_function_type, reference_generator_type, tau_test, motor_type, motor_class,
                                  time_type, turn_off_windows, initializer):
    kwargs = {}
    if motor_type == "DFIM":
        converter_type = time_type + '-Multi'
        kwargs.update({'subconverters': [time_type + '-B6C', time_type + '-B6C']})
    else:
        converter_type = time_type + '-B6C'
    # setup the parameter
    conv_parameter = converter_parameter.copy()
    conv_parameter.update({'tau': tau_test})
    tau = conv_parameter['tau']

    solver_kwargs = {}
    motor_parameter = test_motor_parameter[motor_type]['motor_parameter']
    nominal_values = test_motor_parameter[motor_type]['nominal_values']  # dict
    limit_values = test_motor_parameter[motor_type]['limit_values']  # dict
    u_sup = motor_parameter['u_sup']
    reward_weights = test_motor_parameter[motor_type]['reward_weights']
    # different initializations, using euler solver to accelerate testing
    env_default = motor_class(ode_solver='euler')
    env_1 = motor_class(tau=tau,
                        converter=converter_type,
                        reward_function=reward_function_type,
                        reference_generator=reference_generator_type,
                        ode_solver='euler',
                        load='PolyStaticLoad',
                        solver_kwargs=solver_kwargs,
                        motor_parameter=motor_parameter,
                        nominal_values=nominal_values,
                        limit_values=limit_values,
                        constraints=['all_states'],
                        load_parameter=load_parameter['parameter'],
                        dead_time=conv_parameter['dead_time'],
                        interlocking_time=conv_parameter['interlocking_time'],
                        u_sup=u_sup,
                        reward_weights=reward_weights,
                        motor_initializer=initializer,
                        plots=['omega'])


    _physical_system = setup_physical_system(motor_type, converter_type, three_phase=True, **kwargs)
    reference_generator = make_module(ReferenceGenerator, reference_generator_type,
                                      reference_state='omega')
    reward_function = make_module(RewardFunction, reward_function_type,
                                  reward_weights=reward_weights)
    converter = setup_dc_converter(converter_type, motor_type, **kwargs)  # not used so far
    voltage_supply = IdealVoltageSupply(u_sup)
    load = PolynomialStaticLoad(load_parameter['parameter'])
    solver = EulerSolver()
    env_2 = motor_class(tau=tau, converter=converter,
                        constraints=['all_states'],
                        load=load, solver=solver,
                        supply=voltage_supply,
                        reference_generator=reference_generator,
                        ode_solver='euler',
                        reward_function=reward_function,
                        motor_initializer=initializer)
    # test the different initializations
    envs = [env_default, env_1, env_2]
    seed(123)
    for index, env in enumerate(envs):
        env.reset()
        action_space = env.action_space
        observation_space = env.observation_space
        reward_range = env.reward_function.reward_range
        for k in range(25):
            action = action_space.sample()
            obs, reward, done, _ = env.step(action)
            if env is not env_default:  # limits are not observed in the default case
                if not done:
                    assert all(observation_space[0].low <= obs[0]) and all(observation_space[0].high >= obs[0]), \
                        "State out of limits " + str(obs)
                    assert reward_range[0] <= reward <= reward_range[1], "Reward out of range " + str([reward_range[0],
                                                                                                       reward,
                                                                                                       reward_range[1],
                                                                                                       index])
                else:
                    assert any(observation_space[0].high > abs(obs[0])), "State out of limits " + str(obs)
                    if reward_function_type == 'WSE':
                        assert reward == -1 / (1 - 0.9)
                    else:
                        assert reward == 0
                    env.reset()
            else:
                if done:
                    assert reward == -1/(1-0.9)
                    env.reset()
            env.render()
        env.close()

# endregion

# region new tests


class TestEnvironments:
    """
    class for testing the environments
    """
    _motor = None
    _motor_class = None
    _reward_function = None
    _reference_generator = None
    _reward_weights = {'omega': 1}
    kwargs = {'test_value': 42}

    # counter
    reward_init_counter = 0
    motor_init_counter = 0
    reference_init_counter = 0

    def monkey_super_init(
            self, physical_system=None, reference_generator=None, reward_function=None, constraints = (), **kwargs
    ):
        """
        mock function for super().__init__()
        :param physical_system: instantiated physical system
        :param reference_generator: instantiated reference generator
        :param reward_function: instantiated reward function
        :param kwargs:
        :return:
        """
        assert type(physical_system) == DummyPhysicalSystem, 'Physical system is no DummySCMLSystem'
        if self._reward_function is None:
            assert isinstance(reward_function,
                              DummyRewardFunction), 'Reward function is no instance of DummyRewardfunction'
        else:
            assert reward_function == self._reward_function, 'Reward function is not the expected instance'

        if self._reference_generator is None:
            assert isinstance(reference_generator,
                              DummyReferenceGenerator), 'Reference generator is no instance of DummyReferenceGenerator'
        else:
            assert reference_generator == self._reference_generator, 'Reference generator is not the expected instance'

        assert self.kwargs == kwargs, 'Unexpected additional arguments.'

    def monkey_motor_system(self, motor=None, **kwargs):
        """
        mock function for electric motor system
        :param motor:
        :param kwargs:
        :return:
        """
        self.motor_init_counter += 1
        assert kwargs == self.kwargs, 'Unexpected additional arguments. Keep in mind None and {}.'
        assert motor == self._motor, ' Motor is not the expected instance'
        return DummyPhysicalSystem()

    def monkey_wiener_process(self, **kwargs):
        """
        mock function for wiener process reference generator
        :param kwargs:
        :return:
        """
        self.reference_init_counter += 1
        assert kwargs == self.kwargs, 'Unexpected additional arguments. Keep in mind None and {}.'
        return DummyReferenceGenerator()

    def monkey_weighted_sum_of_errors(self, reward_weights, **kwargs):
        """
        mock function for WSE reward function
        :param reward_weights: given weights for calculating the reward
        :param kwargs:
        :return:
        """
        self.reward_init_counter += 1
        if self._reward_function is None:
            assert reward_weights == dict(torque=1), 'reward weight 1 for torque and 0 otherwise is expected'
        else:
            assert reward_weights == dict(torque=1), 'reward weight 1 for torque and 0 otherwise is expected'
        for key in kwargs.keys():
            assert kwargs[key] == self.kwargs[key], 'Unexpected additional arguments. Keep in mind None and {}.'
        return DummyRewardFunction()

    @pytest.mark.parametrize("reward_function, reward_init_counter",
                             [(None, 1), ('WSE', 0), ('WSE', 0), (DummyRewardFunction(), 0)])
    @pytest.mark.parametrize("reference_generator, reference_init_counter",
                             [(None, 1), ('GWN', 0), (DummyReferenceGenerator(), 0)])
    @pytest.mark.parametrize("motor, motor_class, motor_file, motor_system",
                             [('DcExtEx', extexenv.DcExternallyExcitedMotorEnvironment, extexenv, "DcMotorSystem"),
                              ('DcPermEx', permexenv.DcPermanentlyExcitedMotorEnvironment, permexenv, "DcMotorSystem"),
                              ('DcSeries', seriesenv.DcSeriesMotorEnvironment, seriesenv, "DcMotorSystem"),
                              ('DcShunt', shuntenv.DcShuntMotorEnvironment, shuntenv, "DcMotorSystem"),
                              ('PMSM', pmsmenv.PermanentMagnetSynchronousMotorEnvironment, pmsmenv,
                               "SynchronousMotorSystem"),
                              ('SynRM', synrmenv.SynchronousReluctanceMotorEnvironment, synrmenv,
                               "SynchronousMotorSystem"),
                              ('SCIM', scimenv.SquirrelCageInductionMotorEnvironment, scimenv,
                              "SquirrelCageInductionMotorSystem"),
                              ('DFIM', dfimenv.DoublyFedInductionMotorEnvironment, dfimenv,
                              "DoublyFedInductionMotorSystem"),
                              ])
    def test_dc_environments(self, monkeypatch, motor, motor_class, motor_file, motor_system, reward_function,
                             reward_init_counter, reference_generator, reference_init_counter):
        """
        test the initialization of dc motor environments
        :param monkeypatch:
        :param motor: motor name
        :param motor_class: motor environment name
        :param motor_file: name of imported motor file
        :param motor_system: name of motor system class
        :param reward_function: name of reward function
        :param reward_init_counter: expected result of counter
        :param reference_generator: name of reference generator
        :param reference_init_counter: expected result of counter
        :return:
        """
        # setup test scenario
        self._motor = motor
        self._reward_function = reward_function
        self._reference_generator = reference_generator
        monkeypatch.setattr(ElectricMotorEnvironment, "__init__", self.monkey_super_init)
        monkeypatch.setattr(motor_file, motor_system, self.monkey_motor_system)
        monkeypatch.setattr(motor_file, "WienerProcessReferenceGenerator", self.monkey_wiener_process)
        monkeypatch.setattr(motor_file, "WeightedSumOfErrors", self.monkey_weighted_sum_of_errors)
        reward_weights = dict(torque=1)
        self.kwargs.update(
            {'reward_weights': reward_weights, 'load_parameter': dict(a=0.01, b=0.05, c=0.1, j_load=0.1)}
        )
        # call function to test
        test_object = motor_class(reward_function=self._reward_function,
                                  reference_generator=reference_generator,
                                  **self.kwargs)
        # verify the expected results
        assert reward_init_counter == self.reward_init_counter, \
            'Initialization of reward generator not called as often as expected.'
        assert self.motor_init_counter == 1, 'Initialization of electric motor not called as often as expected.'
        assert reference_init_counter == self.reference_init_counter, \
            'Initialization of reference generator not called as often as expected.'

# endregion
