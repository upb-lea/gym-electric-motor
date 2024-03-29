from classic_controllers import Controller
from externally_referenced_state_plot import ExternallyReferencedStatePlot
import gym_electric_motor as gem
from gym_electric_motor.visualization import MotorDashboard

if __name__ == '__main__':

    """
    motor type:     'PermExDc'  Permanently Excited DC Motor
                    'ExtExDc'   Externally Excited MC Motor
                    'SeriesDc'  DC Series Motor
                    'ShuntDc'   DC Shunt Motor
                    
    control type:   'SC'         Speed Control
                    'TC'         Torque Control
                    'CC'         Current Control
                    
    action_type:    'Cont'      Continuous Action Space
                    'Finite'    Discrete Action Space
    """

    motor_type = 'PermExDc'
    control_type = 'TC'
    action_type = 'Cont'

    motor = action_type + '-' + control_type + '-' + motor_type + '-v0'

    if motor_type in ['PermExDc', 'SeriesDc']:
        states = ['omega', 'torque', 'i', 'u']
    elif motor_type == 'ShuntDc':
        states = ['omega', 'torque', 'i_a', 'i_e', 'u']
    elif motor_type == 'ExtExDc':
        states = ['omega', 'torque', 'i_a', 'i_e', 'u_a', 'u_e']
    else:
        raise KeyError(motor_type + ' is not available')

    # definition of the plotted variables
    external_ref_plots = [ExternallyReferencedStatePlot(state) for state in states]

    # initialize the gym-electric-motor environment
    env = gem.make(motor, visualization=MotorDashboard(additional_plots=external_ref_plots), render_mode="figure_once")
    """
        initialize the controller

        Args:
            environment                     gym-electric-motor environment
            external_ref_plots (optional)   plots of the environment, to plot all reference values
            stages (optional)               structure of the controller
            automated_gain (optional)       if True (default), the controller will be tuned automatically
            a (optional)                    tuning parameter of the symmetrical optimum (default: 4)
    
    """
    visualization = MotorDashboard(additional_plots=external_ref_plots)
    controller = Controller.make(env, external_ref_plots=external_ref_plots)

    (state, reference), _ = env.reset(seed = None)
    # simulate the environment
    for i in range(10001):
        action = controller.control(state, reference)
        (state, reference), reward, terminated, truncated, _ = env.step(action)
        if terminated:
            env.reset()
            controller.reset()

    env.close()
