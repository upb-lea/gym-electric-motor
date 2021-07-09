from classic_controllers import Controller
from externally_referenced_state_plot import ExternallyReferencedStatePlot
import gym_electric_motor as gem
from gym_electric_motor.visualization import MotorDashboard
import numpy as np

if __name__ == '__main__':

    """
        motor type:     'PMSM'      Permanent Magnet Synchronous Motor
                        'SynRM'     Synchronous Reluctance Motor
                        
        control type:   'S'         Speed Control
                        'T'         Torque Control
                        'C'         Current Control

        action_type:    'AbcCont'   Continuous Action Space in ABC-Coordinates
                        'Finite'    Discrete Action Space
    """

    motor_type = 'PMSM'
    control_type = 'T'
    action_type = 'AbcCont'

    env_id = action_type + '-' + control_type + 'C-' + motor_type + '-v0'

    # definition of the motor parameters
    limit_values = dict(omega=12e3 * np.pi / 30, torque=100, i=280, u=320)
    nominal_values = dict(omega=10e3 * np.pi / 30, torque=95.0, i=240, epsilon=np.pi, u=300)
    motor_parameter = dict(p=3, l_d=0.37e-3, l_q=1.2e-3, j_rotor=0.03883, r_s=18e-3, psi_p=45e-3)

    # definition of the plotted variables
    external_ref_plots = [ExternallyReferencedStatePlot(state) for state in ['omega', 'torque', 'i_sd', 'i_sq', 'u_sd', 'u_sq']]

    # initialize the gym-electric-motor environment
    env = gem.make(env_id, visualization=MotorDashboard(additional_plots=external_ref_plots),
                   motor=dict(limit_values=limit_values, nominal_values=nominal_values, motor_parameter=motor_parameter))

    """
    initialize the controller
    
    Args:
        environment                     gym-electric-motor environment
        external_ref_plots (optional)   plots of the environment, to plot all reference values
        stages (optional)               structure of the controller
        automated_gain (optional)       if True (default), the controller will be tune automatically
        a (optional)                    tuning parameter of the Symmetrical Optimum (default: 4)
        
        additionally for TC or SC:
        torque_control (optional)       mode of the torque controller, 'interpolate' (default), 'analytical' or 'online'
        plot_torque(optional)           plot some graphs of the torque controller (default: True)
        plot_modulation (optional)      plot some graphs of the modulation controller (default: False)
        
    """

    controller = Controller.make(env, external_ref_plots=external_ref_plots, torque_control='analytical')

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
