from classic_controllers import Controller
from externally_referenced_state_plot import ExternallyReferencedStatePlot
import gym_electric_motor as gem
from gym_electric_motor.envs.motors import ActionType, ControlType, Motor, MotorType
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.visualization.render_modes import RenderMode

if __name__ == "__main__":
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

    # following manual controller design addresses an ExtExDc. Other motor types require different controller stages
    """
    motor_type = "ExtExDc"
    control_type = "CC"
    action_type = "Cont"
    """

    motor = Motor(MotorType.ExternallyExcitedDcMotor,
        ControlType.CurrentControl,
        ActionType.Continuous,)


    # definition of the plotted variables
    external_ref_plots = [ExternallyReferencedStatePlot(state) for state in motor.states()]

    motor_dashboard = MotorDashboard(additional_plots=external_ref_plots, render_mode=RenderMode.Figure)

    # initialize the gym-electric-motor environment
    env = gem.make(
        motor.env_id(),
        visualization=motor_dashboard
        
    )

    """
        initialize the controller

        Args:
            environment                     gym-electric-motor environment
            external_ref_plots (optional)   plots of the environment, to plot all reference values
            automated_gain (optional)       if True (default), the controller will be tune automatically
            a (optional)                    tuning parameter of the symmetrical optimum (default: 4)
            
            stages (optional)               Each controller stage is defined in a dict. The key controller_type
                                            specifies which type of controller is used for the stage.  In addition,
                                            parameters of the controller can be passed like, e.g., the P-gain and I-gain
                                            for the PI controller. The stages are grouped in an array in ascending
                                            order. For the ExtExDc an additional current controller is needed, which is
                                            added in a separate array.
    """

    current_a_controller = {
        "controller_type": "pi_controller",
        "p_gain": 0.3,
        "i_gain": 50,
    }
    speed_controller = {"controller_type": "pi_controller", "p_gain": 1, "i_gain": 40}
    current_e_controller = {
        "controller_type": "pi_controller",
        "p_gain": 5,
        "i_gain": 300,
    }

    stages_a = [current_a_controller, speed_controller]
    stages_e = [current_e_controller]

    stages = [stages_a, stages_e]

    controller = Controller.make(
        env, external_ref_plots=external_ref_plots, stages=stages
    )

    (state, reference), _ = env.reset()

    # simulate the environment
    for i in range(10001):
        action = controller.control(state, reference)
        (state, reference), reward, terminated, truncated, _ = env.step(action)
        if terminated:
            env.reset()
            controller.reset()

    motor_dashboard.show()
    env.close()
