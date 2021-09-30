from classic_controllers import Controller
from externally_referenced_state_plot import ExternallyReferencedStatePlot
from external_plot import ExternalPlot
import gym_electric_motor as gem
from gym_electric_motor.visualization import MotorDashboard
import numpy as np

if __name__ == '__main__':

    """
        motor type:     'SCIM'      Squirrel Cage Induction Motor

        control type:   'SC'        Speed Control
                        'TC'        Torque Control
                        'CC'        Current Control

        action_type:    'AbcCont'      Continuous Action Space
    """

    motor_type = 'SCIM'
    control_type = 'SC'
    action_type = 'AbcCont'

    env_id = action_type + '-' + control_type + '-' + motor_type + '-v0'

    # definition of the plotted variables
    states = ['omega', 'torque', 'i_sd', 'i_sq', 'u_sd', 'u_sq']
    external_ref_plots = [ExternallyReferencedStatePlot(state) for state in states]
    external_plot = [ExternalPlot(referenced=control_type != 'CC'), ExternalPlot(min=-np.pi, max=np.pi)]
    external_ref_plots += external_plot

    # initialize the gym-electric-motor environment
    env = gem.make(env_id, visualization=MotorDashboard(additional_plots=external_ref_plots))

    """
        initialize the controller

        Args:
            environment                     gym-electric-motor environment
            external_plots (optional)       plots of the environment, to plot all reference values
            stages (optional)               structure of the controller
            automated_gain (optional)       if True (default), the controller will be tune automatically

    """
    controller = Controller.make(env, external_plot=external_ref_plots)

    state, reference = env.reset()

    # simulate the environment
    for i in range(10001):
        env.render()
        action = controller.control(state, reference)
        (state, reference), reward, done, _ = env.step(action)
        if done:
            env.reset()
            controller.reset()
    env.close()
