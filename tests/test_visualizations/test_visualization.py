from gym_electric_motor.visualization.motor_dashboard import *
from ..testing_utils import *
from gym_electric_motor.reward_functions import *
from gym_electric_motor.utils import make_module
import pytest


# region first version tests

#@pytest.mark.parametrize("motor_type", ['DcPermEx', 'DcShunt', 'DcExtEx'])
#@pytest.mark.parametrize("converter_type", ['Disc-2QC', 'Disc-2QC', 'Disc-4QC', 'Cont-1QC', 'Cont-2QC', 'Cont-4QC'])
#@pytest.mark.parametrize("plotted_variables", [['torque', 'omega', 'u_sup'], ['all'], ['none'], ['omega'],
#                                             ['omega', 'u', 'u_a', 'u_e']])
#def test_visualization(motor_type, converter_type, plotted_variables, turn_off_windows):
"""
    test initialization and basic functions for all motor and converters
    :param motor_type: motor name (string)
    :param converter_type: converter name (string)
    :param plotted_variables: shown variables (list/True/False)
    :return:
    """
"""    # set parameters
    update_period = 1E-2
    visu_period = 0.1
    # setup physical system
    physical_system = setup_physical_system(motor_type, converter_type)
    # setup reference generator
    reference_generator = setup_reference_generator('SinusReference', physical_system)
    # setup reward function
    reward_function = make_module(RewardFunction, 'WSE')
    reward_function.set_modules(physical_system, reference_generator)
    # initializations of dashboards in different ways
    dashboard_default = MotorDashboard()
    dashboard_make = make_module(ElectricMotorVisualization, 'MotorDashboard', visu_period=visu_period,
                                 update_period=update_period,
                                 plotted_variables=plotted_variables)
    dashboard_1 = MotorDashboard(visu_period=visu_period, update_period=update_period,
                                 plotted_variables=plotted_variables)

    # setup state, reference, reward for testing
    len_state = len(physical_system.state_names)
    reference = physical_system.state_space.low
    reward = -0.5
    dashboards = [dashboard_default, dashboard_make, dashboard_1]

    # test references given in each step
    for dashboard in dashboards:
        if plotted_variables == ['none'] and dashboard is not dashboard_default:
            with pytest.warns(Warning):
                dashboard.set_modules(physical_system, reference_generator, reward_function)
                dashboard.reset()
                dashboard.step(physical_system.state_space.sample(), physical_system.state_space.sample(), 0)
        else:
            dashboard.set_modules(physical_system, reference_generator, reward_function)
            dashboard.reset()
            dashboard.step(physical_system.state_space.sample(), physical_system.state_space.sample(), 0)

        for k in range(2):
            state = np.ones(len_state) * np.sin(k / 1000) / 2 + 0.5
            dashboard.step(state, reference, reward)
        dashboard.close()


@pytest.mark.parametrize("motor_type", ['DcExtEx'])
@pytest.mark.parametrize("converter_type", ['Cont-4QC'])
@pytest.mark.parametrize("plotted_variables", ['none', ['no_useful_reference']])
def test_visualization_plotted_variables(motor_type, converter_type, plotted_variables, turn_off_windows):
    """"""
    test initialization and basic functions for all motor and converters
    :param motor_type: motor name (string)
    :param converter_type: converter name (string)
    :param plotted_variables: shown variables (list/True/False)
    :return:
    """"""
    # setup physical system
    physical_system = setup_physical_system(motor_type, converter_type)
    # setup reference generator
    reference_generator = setup_reference_generator('SinusReference', physical_system)
    # setup reward function
    reward_function = make_module(RewardFunction, 'WSE')
    reward_function.set_modules(physical_system, reference_generator)
    # initializations of dashboards in different ways
    dashboard_default = MotorDashboard(plotted_variables=plotted_variables)
    # test for warning
    with pytest.warns(Warning):
        dashboard_default.set_modules(physical_system, reference_generator, reward_function)
        dashboard_default.reset()
        dashboard_default.step(physical_system.state_space.sample(), physical_system.state_space.sample(), 0)
    dashboard_default.close()


@pytest.mark.parametrize("motor_type", ['DcExtEx'])
@pytest.mark.parametrize("converter_type", ['Cont-4QC'])
@pytest.mark.parametrize("plotted_variables", [['all']])
@pytest.mark.parametrize("update_period", [1E-3, 1E-2, 5E-3, 1E-1])
@pytest.mark.parametrize("visu_period", [1E-2, 1E-1, 1])
def test_visualization_parameter(motor_type, converter_type, plotted_variables, update_period, visu_period,
                                 turn_off_windows):
    """"""
    test visu period and update period
    :param motor_type: motor name (string)
    :param converter_type: converter name (string)
    :param plotted_variables: shown variables (list/True/False)
    :param update_period: update period from dashboard
    :param visu_period: visu period from dashboard
    :return:
    """"""
    # setup physical system
    physical_system = setup_physical_system(motor_type, converter_type)
    # setup reference generator
    reference_generator = setup_reference_generator('StepReference', physical_system)
    # setup reward function
    reward_function = make_module(RewardFunction, 'WSE')
    reward_function.set_modules(physical_system, reference_generator)
    # initializations of dashboards in different ways
    dashboard = MotorDashboard(plotted_variables=plotted_variables,
                               update_period=update_period,
                               visu_period=visu_period)
    # setup the modules
    dashboard.set_modules(physical_system, reference_generator, reward_function)
    # test for given references in the reset
    dashboard.reset()
    reward = 0
    len_state = len(physical_system.state_names)
    for k in range(150):
        state = np.ones(len_state) * np.sin(k / 1000) / 2 + 0.5
        reference = np.ones(len_state) * np.cos(k / 2000) / 2 + 0.5
        dashboard.step(state, reference, reward)
    dashboard.close()


@pytest.mark.parametrize("motor_type", ['DcExtEx'])
@pytest.mark.parametrize("converter_type", ['Cont-4QC'])
@pytest.mark.parametrize("plotted_variables", [['all']])
@pytest.mark.parametrize("update_period", [1E-1, 1E-2])
@pytest.mark.parametrize("visu_period", [1, 0.5])
def test_visualization_reset(motor_type, converter_type, plotted_variables, update_period, visu_period,
                             turn_off_windows):
    """"""
    test reset with given references
    :param motor_type: motor name (string)
    :param converter_type: converter name (string)
    :param plotted_variables: shown variables (list/True/False)
    :param update_period: update period from dashboard
    :param visu_period: visu period from dashboard
    :return:
    """"""
    # setup physical system
    physical_system = setup_physical_system(motor_type, converter_type)
    # setup reference generator
    reference_generator = setup_reference_generator('WienerProcessReference', physical_system)
    # setup reward function
    reward_function = make_module(RewardFunction, 'WSE')
    reward_function.set_modules(physical_system, reference_generator)
    # initializations of dashboards in different ways
    dashboard = MotorDashboard(plotted_variables=plotted_variables,
                               update_period=update_period,
                               visu_period=visu_period)
    # setup the modules
    dashboard.set_modules(physical_system, reference_generator, reward_function)
    # test for given references in the reset
    len_state = len(physical_system.state_names)
    # setup different pre defined references
    len_reference = 1000
    reference_basic = np.ones((len_state, len_reference))
    references_1 = reference_basic
    references_1[0, :] *= physical_system.limits[0] * 0.8
    references_1[1, :] = None
    references_2 = reference_basic[:len_state, :] * 0.5
    references_2[2, :] = None
    references_3 = reference_basic
    reset_args = [references_1, references_2, references_3, None]

    # test for given references in the reset
    for reset_arg in reset_args:
        dashboard.reset(reset_arg)
        for k in range(50):
            state = np.ones(len_state) * np.sin(k / 1000) / 2 + 0.5
            dashboard.step(state, None, None)
    # test reset() and step function with step reference
    dashboard.reset()
    for k in range(50):
        state = np.ones(len_state) * np.sin(k / 1000) / 2 + 0.5
        reference_generator.get_reference_observation()
        reference = reference_generator.get_reference()
        dashboard.step(state, reference, None)
    dashboard.close()

# endregion
"""