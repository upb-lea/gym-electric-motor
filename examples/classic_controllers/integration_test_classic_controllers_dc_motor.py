from classic_controllers import Controller
from externally_referenced_state_plot import ExternallyReferencedStatePlot
import gym_electric_motor as gem
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.reference_generators import SinusoidalReferenceGenerator
import time
from gym_electric_motor.helper import *

if __name__ == "__main__":
    """
    motor type:     'PermExDc'  Permanently Excited DC Motor
                    'ExtExDc'   Externally Excited DC Motor
                    'SeriesDc'  DC Series Motor
                    'ShuntDc'   DC Shunt Motor

    control type:   'SC'         Speed Control
                    'TC'         Torque Control
                    'CC'         Current Control

    action_type:    'Cont'      Continuous Action Space
                    'Finite'    Discrete Action Space
    """

    # motor_type = "PermExDc"
    # control_type = "SC"
    # action_type = "Cont"

    motor = Motor(
        MotorType.PermanentlyExcitedDcMotor,
        ControlType.SpeedControl,
        ActionType.Continuous,
    )

    # definition of the plotted variables
    external_ref_plots = [
        ExternallyReferencedStatePlot(state) for state in motor.get_state_names()
    ]

    # definition of the reference generator

    ref_generator = SinusoidalReferenceGenerator(
        amplitude_range=(1, 1),
        frequency_range=(5, 5),
        offset_range=(0, 0),
        episode_lengths=(10001, 10001),
    )
    motor_dashboard = MotorDashboard(additional_plots=external_ref_plots)
    # initialize the gym-electric-motor environment
    env = gem.make(
        motor.get_env_id(),
        visualization=motor_dashboard,
        scale_plots=True,
        render_mode="figure",
        reference_generator=ref_generator,
    )
    motor_dashboard.set_env(env)

    env.metadata["filename_prefix"] = "integration-test"
    env.metadata["filename_suffix"] = ""
    env.metadata["save_figure_on_close"] = True
    env.metadata["hold_figure_on_close"] = False
    """
        initialize the controller

        Args:
            environment                     gym-electric-motor environment
            external_ref_plots (optional)   plots of the environment, to plot all reference values
            stages (optional)               structure of the controller
            automated_gain (optional)       if True (default), the controller will be tuned automatically
            a (optional)                    tuning parameter of the symmetrical optimum (default: 4)
    
    """
    controller = Controller.make(env, external_ref_plots=external_ref_plots)

    (state, reference), _ = env.reset(seed=1337)
    # simulate the environment
    for i in range(10001):
        action = controller.control(state, reference)
        # if i % 100 == 0:
        #   (state, reference), reward, terminated, truncated, _ = env.step(env.action_space.sample())
        # else:
        (state, reference), reward, terminated, truncated, _ = env.step(action)

        # viz.render()

        if terminated:
            env.reset()

            controller.reset()

    env.close()
    motor_dashboard.show()
    motor_dashboard.save_to_file("test")
