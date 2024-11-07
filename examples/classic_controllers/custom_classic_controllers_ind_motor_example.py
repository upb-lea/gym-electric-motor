from gym_electric_motor.envs.motors import ActionType, ControlType, Motor, MotorType
from classic_controllers import Controller
from externally_referenced_state_plot import ExternallyReferencedStatePlot
from external_plot import ExternalPlot
import gym_electric_motor as gem
from gym_electric_motor.visualization import MotorDashboard
import numpy as np

if __name__ == "__main__":
    """
        motor type:     'SCIM'      Squirrel Cage Induction Motor

        control type:   'SC'        Speed Control
                        'TC'        Torque Control
                        'CC'        Current Control

        action_type:    'Cont'      Continuous Action Space
    """
    
    """
    motor_type = "SCIM"
    control_type = "SC"
    action_type = "Cont"
    """

    motor = Motor(
        MotorType.SquirrelCageInductionMotor,
        ControlType.SpeedControl,
        ActionType.Continuous,
    )

    # definition of the plotted variables
    states = ["omega", "torque", "i_sd", "i_sq", "u_sd", "u_sq"]
    external_ref_plots = [ExternallyReferencedStatePlot(state) for state in states]
    external_plot = [
        ExternalPlot(referenced=ControlType.SpeedControl != "CC"),
        ExternalPlot(min=-np.pi, max=np.pi),
    ]
    external_ref_plots += external_plot

    motor_dashboard = MotorDashboard(additional_plots=external_ref_plots)

    # initialize the gym-electric-motor environment
    env = gem.make(
        motor.env_id(), 
        visualization=motor_dashboard
    )

    """
        initialize the controller

        Args:
            environment                     gym-electric-motor environment
            external_plots (optional)       plots of the environment, to plot all reference values
            stages (optional)               structure of the controller
            automated_gain (optional)       if True (default), the controller will be tune automatically

    """

    current_controller = [
        {"controller_type": "pi_controller", "p_gain": 40, "i_gain": 15000},
        {"controller_type": "pi_controller", "p_gain": 25, "i_gain": 10000},
    ]
    speed_controller = [
        {"controller_type": "pi_controller", "p_gain": 1, "i_gain": 100}
    ]
    stages = [current_controller, speed_controller]

    controller = Controller.make(env, stages=stages, external_plot=external_ref_plots)

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
