from gym_electric_motor.envs.motors import ActionType, ControlType, Motor, MotorType
from classic_controllers import Controller
from externally_referenced_state_plot import ExternallyReferencedStatePlot
import gym_electric_motor as gem
from gym_electric_motor.visualization import MotorDashboard


if __name__ == "__main__":
    """
        motor type:     'PMSM'      Permanent Magnet Synchronous Motor
                        'SynRM'     Synchronous Reluctance Motor
                        
        control type:   'SC'         Speed Control
                        'TC'         Torque Control
                        'CC'         Current Control

        action_type:    'Cont'   Continuous Action Space in ABC-Coordinates
                        'Finite'    Discrete Action Space
    """
    
    """
    motor_type = "PMSM"
    control_type = "TC"
    action_type = "Cont"
     """

    motor = Motor(
        MotorType.PermanentMagnetSynchronousMotor,
        ControlType.TorqueControl,
        ActionType.Continuous
        )

    # definition of the plotted variables
    external_ref_plots = [
        ExternallyReferencedStatePlot(state) for state in ["omega", "torque", "i_sd", "i_sq", "u_sd", "u_sq"]
    ]

    motor_dashboard = MotorDashboard(additional_plots=external_ref_plots)
    # initialize the gym-electric-motor environment
    env = gem.make(
        motor.env_id(), visualization=motor_dashboard
    )

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

    controller = Controller.make(
        env, external_ref_plots=external_ref_plots, torque_control="analytical"
    )

    (state, reference), _ = env.reset()

    # simulate the environment
    for i in range(10001):
        action = controller.control(state, reference)
        (state, reference), reward, terminated, truncated, _ = env.step(action)
        if terminated:
            env.reset()
            controller.reset()
   
    motor_dashboard.show_and_hold()
    env.close()
