import gym_electric_motor.envs
from gym_electric_motor.envs.gym_dcm import *
from gym_electric_motor.envs.gym_pmsm import *
from gym_electric_motor.envs.gym_synrm import *
from .functions_used_for_testing import *
from numpy.random import randint, seed
import pytest


g_disc_converter = ['Disc-1QC', 'Disc-2QC', 'Disc-4QC']
g_cont_converter = ['Cont-1QC', 'Cont-2QC', 'Cont-4QC']
g_converter = ['Disc-1QC', 'Disc-2QC', 'Disc-4QC', 'Cont-1QC', 'Cont-2QC', 'Cont-4QC']
g_reference_generator = ['WienerProcessReference', 'SinusReference', 'StepReference']
g_reward_functions = ['WSE', 'SWSE']


@pytest.mark.parametrize("converter_type", g_cont_converter)
@pytest.mark.parametrize("reward_function_type", g_reward_functions)
@pytest.mark.parametrize("reference_generator_type", g_reference_generator)
@pytest.mark.parametrize("tau_test", [1E-5, 1E-4, 1E-3])
@pytest.mark.parametrize("motor_type, state_filter, motor_class",
                         [('DcPermEx', ['omega', 'u', 'i'], ContDcPermanentlyExcitedMotorEnvironment),
                          ('DcSeries', ['torque'], ContDcSeriesMotorEnvironment),
                          ('DcShunt', ['omega'], ContDcShuntMotorEnvironment),
                          ('DcExtEx', ['torque'], ContDcExternallyExcitedMotorEnvironment)])
def test_cont_dc_motor_environments(motor_type, converter_type, reward_function_type,
                                    reference_generator_type, tau_test, state_filter, motor_class, turn_off_windows):
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

    solver_kwargs = {'method': 'RK45'}
    motor_parameter = test_motor_parameter[motor_type]['motor_parameter']
    nominal_values = test_motor_parameter[motor_type]['nominal_values']  # dict
    limit_values = test_motor_parameter[motor_type]['limit_values']  # dict
    u_sup = motor_parameter['u_sup']
    observed_states = ['omega', 'i']
    reward_weights = test_motor_parameter[motor_type]['reward_weights']
    # setup converter parameter
    kwargs = {}
    if motor_type == 'DcExtEx':
        converter_types = 'Cont-Double'
        kwargs.update({'subconverters': [converter_type, converter_type]})
    else:
        converter_types = converter_type
    # different initializations
    env_default = motor_class()
    env_1 = motor_class(tau=tau,
                        converter=converter_types,
                        reward_function=reward_function_type,
                        reference_generator=reference_generator_type,
                        visualization=None,
                        state_filter=state_filter,
                        ode_solver='scipy.solve_ivp',
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
                        observed_states=observed_states,
                        **kwargs
                        )

    _physical_system = setup_physical_system(motor_type, converter_type)
    reference_generator = make_module(ReferenceGenerator, reference_generator_type,
                                      reference_state='omega')
    reward_function = make_module(RewardFunction, reward_function_type, observed_states=observed_states,
                                  reward_weights=reward_weights)
    converter = setup_dc_converter(converter_type, motor_type)  # not used so far
    voltage_supply = IdealVoltageSupply(u_sup)
    load = PolynomialStaticLoad(load_parameter['parameter'])
    solver = ScipySolveIvpSolver(method='RK45')
    env_2 = motor_class(tau=tau, converter=converter,
                        load=load, solver=solver,
                        supply=voltage_supply,
                        reference_generator=reference_generator,
                        reward_function=reward_function,
                        state_filter=state_filter)
    # test the different initializations
    envs = [env_default, env_1, env_2]
    for index, env in enumerate(envs):
        env.reset()
        action_space = env.action_space
        for k in range(10):
            action = action_space.sample()
            obs, reward, done, _ = env.step(action)
            reward_range = env.reward_function.reward_range
            assert reward_range[0] <= reward <= reward_range[1], "Reward out of range " + str([reward_range[0], reward,
                                                                                               reward_range[1], index])
            env.render()
        env.close()


@pytest.mark.parametrize("plotted_variables",
                         [['torque', 'omega', 'u_sup'], ['all'], ['none'], ['omega'], ['omega', 'u', 'u_a', 'u_e']])
@pytest.mark.parametrize("visu_period", [5E-1, 1, 5])
@pytest.mark.parametrize("update_period", [1E-2, 1E-1])
@pytest.mark.parametrize("visualization", [None, 'MotorDashboard', 'instance'])
@pytest.mark.parametrize("motor_class",
                         [ContDcPermanentlyExcitedMotorEnvironment,
                          ContDcSeriesMotorEnvironment,
                          ContDcShuntMotorEnvironment,
                          ContDcExternallyExcitedMotorEnvironment,
                          DiscDcPermanentlyExcitedMotorEnvironment,
                          DiscDcSeriesMotorEnvironment,
                          DiscDcShuntMotorEnvironment,
                          DiscDcExternallyExcitedMotorEnvironment])
def test_env_visualization(visualization, plotted_variables, visu_period, update_period, motor_class, turn_off_windows):
    """
    test the visualization of motor states in the environments
    :param visualization:
    :param plotted_variables:
    :param visu_period:
    :param update_period:
    :param motor_class:
    :return:
    """
    # setup Dashboard
    if visualization == 'instance':
        visualization = MotorDashboard(visu_period=visu_period,
                                       update_period=update_period,
                                       plotted_variables=plotted_variables)
    env = motor_class(visualization=visualization,
                      visu_period=visu_period,
                      update_period=update_period,
                      plotted_variables=plotted_variables)
    if visualization is not None:
        assert visu_period == env._visualization._visu_period
        assert update_period == env._visualization._update_period
    env.reset()
    # test step function for random input
    for k in range(100):
        action = env.action_space.sample()
        env.step(action)
        env.render()
    env.close()


@pytest.mark.parametrize("converter_type", g_disc_converter)
@pytest.mark.parametrize("reward_function_type", g_reward_functions)
@pytest.mark.parametrize("reference_generator_type", g_reference_generator)
@pytest.mark.parametrize("tau_test", [1E-5, 1E-4, 1E-3])
@pytest.mark.parametrize("motor_type, state_filter, motor_class",
                         [('DcPermEx', ['omega', 'u', 'i'], DiscDcPermanentlyExcitedMotorEnvironment),
                          ('DcSeries', ['torque'], DiscDcSeriesMotorEnvironment),
                          ('DcShunt', ['omega'], DiscDcShuntMotorEnvironment),
                          ('DcExtEx', ['torque'], DiscDcExternallyExcitedMotorEnvironment)])
def test_disc_dc_motor_environments(motor_type, converter_type, reward_function_type,
                                    reference_generator_type, tau_test, state_filter, motor_class, turn_off_windows):
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

    solver_kwargs = {'method': 'RK45'}
    motor_parameter = test_motor_parameter[motor_type]['motor_parameter']
    nominal_values = test_motor_parameter[motor_type]['nominal_values']  # dict
    limit_values = test_motor_parameter[motor_type]['limit_values']  # dict
    u_sup = motor_parameter['u_sup']
    observed_states = ['omega', 'i']
    reward_weights = test_motor_parameter[motor_type]['reward_weights']
    # setup converter parameter
    kwargs = {}
    if motor_type == 'DcExtEx':
        converter_types = 'Disc-Double'
        kwargs.update({'subconverters': [converter_type, converter_type]})
    else:
        converter_types = converter_type
    # different initializations
    env_default = motor_class()
    env_1 = motor_class(tau=tau,
                        converter=converter_types,
                        reward_function=reward_function_type,
                        reference_generator=reference_generator_type,
                        visualization=None,
                        state_filter=state_filter,
                        ode_solver='scipy.solve_ivp',
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
                        observed_states=observed_states,
                        **kwargs
                        )

    _physical_system = setup_physical_system(motor_type, converter_type)
    reference_generator = make_module(ReferenceGenerator, reference_generator_type,
                                      reference_state='omega')
    reward_function = make_module(RewardFunction, reward_function_type, observed_states=observed_states,
                                  reward_weights=reward_weights)
    converter = setup_dc_converter(converter_type, motor_type)  # not used so far
    voltage_supply = IdealVoltageSupply(u_sup)
    load = PolynomialStaticLoad(load_parameter['parameter'])
    solver = ScipySolveIvpSolver(method='RK45')
    env_2 = motor_class(tau=tau, converter=converter,
                        load=load, solver=solver,
                        supply=voltage_supply,
                        reference_generator=reference_generator,
                        reward_function=reward_function,
                        state_filter=state_filter)
    # test the different initializations
    envs = [env_default, env_1, env_2]
    for index, env in enumerate(envs):
        env.reset()
        action_space = env.action_space
        for k in range(10):
            action = action_space.sample()
            obs, reward, done, _ = env.step(action)
            reward_range = env.reward_function.reward_range
            assert reward_range[0] <= reward <= reward_range[1], "Reward out of range " + str([reward_range[0], reward,
                                                                                               reward_range[1], index])
            env.render()
        env.close()


@pytest.mark.parametrize("reward_function_type", g_reward_functions)
@pytest.mark.parametrize("reference_generator_type", g_reference_generator)
@pytest.mark.parametrize("tau_test", [1E-5, 1E-4, 1E-3])
@pytest.mark.parametrize("motor_type, motor_class, time_type",
                         [('PMSM', ContPermanentMagnetSynchronousMotorEnvironment, 'Cont'),
                          ('SynRM', ContSynchronousReluctanceMotorEnvironment, 'Cont'),
                          ('PMSM', DiscPermanentMagnetSynchronousMotorEnvironment, 'Disc'),
                          ('SynRM', DiscSynchronousReluctanceMotorEnvironment, 'Disc')])
def test_synchronous_environments(reward_function_type, reference_generator_type, tau_test, motor_type, motor_class,
                                  time_type, turn_off_windows):
    converter_type = time_type + '-B6C'
    # setup the parameter
    conv_parameter = converter_parameter.copy()
    conv_parameter.update({'tau': tau_test})
    tau = conv_parameter['tau']

    solver_kwargs = {'method': 'RK45'}
    motor_parameter = test_motor_parameter[motor_type]['motor_parameter']
    nominal_values = test_motor_parameter[motor_type]['nominal_values']  # dict
    limit_values = test_motor_parameter[motor_type]['limit_values']  # dict
    u_sup = motor_parameter['u_sup']
    observed_states = ['all']
    reward_weights = test_motor_parameter[motor_type]['reward_weights']
    # different initializations
    env_default = motor_class()
    env_1 = motor_class(tau=tau,
                        converter=converter_type,
                        reward_function=reward_function_type,
                        reference_generator=reference_generator_type,
                        visualization='MotorDashboard',
                        ode_solver='scipy.solve_ivp',
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
                        observed_states=observed_states)

    _physical_system = setup_physical_system(motor_type, converter_type, True)
    reference_generator = make_module(ReferenceGenerator, reference_generator_type,
                                      reference_state='omega')
    reward_function = make_module(RewardFunction, reward_function_type, observed_states=observed_states,
                                  reward_weights=reward_weights)
    converter = setup_dc_converter(converter_type, motor_type)  # not used so far
    voltage_supply = IdealVoltageSupply(u_sup)
    load = PolynomialStaticLoad(load_parameter['parameter'])
    solver = ScipySolveIvpSolver(method='RK45')
    env_2 = motor_class(tau=tau, converter=converter,
                        load=load, solver=solver,
                        supply=voltage_supply,
                        reference_generator=reference_generator,
                        reward_function=reward_function)
    # test the different initializations
    envs = [env_default, env_1, env_2]
    seed(123)
    for index, env in enumerate(envs):
        env.reset()
        action_space = env.action_space
        observation_space = env.observation_space
        reward_range = env.reward_function.reward_range
        for k in range(25):
            if time_type == 'Cont':
                action = np.array(
                    [np.sin(k * tau / 2E-2), np.sin(k * tau / 2E-2 + np.pi / 3), np.sin(k * tau / 2E-2 - np.pi / 3)])
            else:
                action = randint(action_space.n)
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
            env.render()

        env.close()


