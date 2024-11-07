from gym_electric_motor.envs.motors import ActionType, ControlType, Motor, MotorType
from classic_controllers import Controller
from externally_referenced_state_plot import ExternallyReferencedStatePlot
import gym_electric_motor as gem
from gym_electric_motor.visualization import MotorDashboard
import numpy as np

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
    control_type = "SC"
    action_type = "Cont"
    """

    motor = Motor(MotorType.PermanentMagnetSynchronousMotor,
                  ControlType.SpeedControl,
                  ActionType.Continuous)

    # definition of the motor parameters
    psi_p = 0 if "SynRM" in motor.env_id() else 45e-3
    limit_values = dict(omega=12e3 * np.pi / 30, torque=100, i=280, u=320)
    nominal_values = dict(
        omega=10e3 * np.pi / 30, torque=95.0, i=240, epsilon=np.pi, u=300
    )
    motor_parameter = dict(
        p=3, l_d=0.37e-3, l_q=1.2e-3, j_rotor=0.03883, r_s=18e-3, psi_p=psi_p
    )

    # definition of the plotted variables
    external_ref_plots = [
        ExternallyReferencedStatePlot(state)
        for state in ["omega", "torque", "i_sd", "i_sq", "u_sd", "u_sq"]
    ]

    motor_dashboard = MotorDashboard(additional_plots=external_ref_plots)
    # initialize the gym-electric-motor environment
    env = gem.make(
        motor.env_id(),
        visualization=MotorDashboard(additional_plots=external_ref_plots),
        motor=dict(
            limit_values=limit_values,
            nominal_values=nominal_values,
            motor_parameter=motor_parameter,
        ),
        render_mode="figure",
    )

    """
    initialize the controller
    
    Args:
        environment                     gym-electric-motor environment
        external_ref_plots (optional)   plots of the environment, to plot all reference values
        automated_gain (optional)       if True (default), the controller will be tune automatically
        a (optional)                    tuning parameter of the Symmetrical Optimum (default: 4)
        
        stages (optional)               Each controller stage is defined in a dict. The key controller_type specifies
                                        which type of controller is used for the stage.  In addition, parameters of the
                                        controller can be passed like e.g. the p-gain and i-gain for the PI controller
                                        (see example below).
                                        The stages are grouped in an array in ascending order. For environments with
                                        current control only an array with the corresponding current controllers is
                                        needed (see example below).  For a higher-level torque or speed control, these
                                        controllers are passed in an additional controller. Note that the
                                        TorqueToCurrent controller is added automatically (see example below).
                                        
                                        Examples:
                                        
                                        controller stage:
                                        d_current_controller = {'controller_type': 'pi-controller', 'p_gain': 2,
                                                                'i_gain': 30}
                                        
                                        AbcCont currrent control:
                                            stages = [d_current_controller, q_current_controller]
                                        
                                        Finite current control:
                                            stages = [a_current_controller, b_current_controller, c_current_controller]
                                        
                                        AbcCont torque control:
                                            stages = [[d_current_controller, q_current_controller]]  (no overlaid
                                            controller, because the torque to current stage is added automatically)
                                        
                                        Finite torque control:      
                                            stages = [[a_current_controller, b_current_controller, c_current_controller]]
                                        
                                        AbcCont speed control:
                                            stages = [[d_current_controller, q_current_controller], [speed_controller]]
                                            
                                        Finite speed control:      
                                            stages = [[a_current_controller, b_current_controller, c_current_controller],
                                                      [speed_controller]]
        
        additionally for TC or SC:
        torque_control (optional)       mode of the torque controller, 'interpolate' (default), 'analytical' or 'online'
        plot_torque(optional)           plot some graphs of the torque controller (default: True)
        plot_modulation (optional)      plot some graphs of the modulation controller (default: False)
        
    """

    current_d_controller = {
        "controller_type": "pi_controller",
        "p_gain": 1,
        "i_gain": 500,
    }
    current_q_controller = {
        "controller_type": "pi_controller",
        "p_gain": 3,
        "i_gain": 1400,
    }
    speed_controller = {
        "controller_type": "pi_controller",
        "p_gain": 12,
        "i_gain": 1300,
    }

    current_controller = [current_d_controller, current_q_controller]
    overlaid_controller = [speed_controller]

    stages = [current_controller, overlaid_controller]

    controller = Controller.make(
        env,
        stages=stages,
        external_ref_plots=external_ref_plots,
        torque_control="analytical",
    )

    (state, reference), _ = env.reset()

    # simulate the environment
    for i in range(10001):
        action = controller.control(state, reference)
        (state, reference), reward, terminated, truncated, _ = env.step(action)
        if terminated:
            env.reset()
            controller.reset()
 
    env.close()
