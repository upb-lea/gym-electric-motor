import os
import sys

import gym_electric_motor as gem
from gem_controllers import GemController
from gym_electric_motor.envs.motors import ActionType, ControlType, Motor, MotorType
from gym_electric_motor.observers import StateObserver
from gym_electric_motor.reference_generators import SinusoidalReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard, RenderMode

path = os.getcwd() + "/examples/classic_controllers"
sys.path.append(path)
from classic_controllers import Controller  # noqa: E402
from externally_referenced_state_plot import ExternallyReferencedStatePlot  # noqa: E402

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

    motor = Motor(
        MotorType.PermanentlyExcitedDcMotor,
        ControlType.SpeedControl,
        ActionType.Continuous,
    )

    # definition of the plotted variables
    external_ref_plots = [ExternallyReferencedStatePlot(state) for state in motor.states()]

    # definition of the reference generator

    ref_generator = SinusoidalReferenceGenerator(
        amplitude_range=(1, 1),
        frequency_range=(5, 5),
        offset_range=(0, 0),
        episode_lengths=(10001, 10001),
    )
    motor_dashboard = MotorDashboard(additional_plots=external_ref_plots, render_mode=RenderMode.FigureOnce)
    # initialize the gym-electric-motor environment
    env = gem.make(
        motor.env_id(),
        visualization=motor_dashboard,
        scale_plots=True,
        reference_generator=ref_generator,
    )

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
    # controller = GemController.make(env, env_id=motor.env_id())

    (state, reference), _ = env.reset(seed=1337)
    print("state_names: ", motor.states())

    state_observer = StateObserver(env)
    # simulate the environment
    for i in range(100):
        action = controller.control(state, reference)
        # if i % 100 == 0:
        #   (state, reference), reward, terminated, truncated, _ = env.step(env.action_space.sample())
        # else:
        (state, reference), reward, terminated, truncated, _ = env.step(action)
        torque = state_observer.observe("torque")
        print(f"Toque: {torque}")
        # viz.render()

        if terminated:
            env.reset()

            controller.reset()

    env.close()

    motor_dashboard.show_and_hold()