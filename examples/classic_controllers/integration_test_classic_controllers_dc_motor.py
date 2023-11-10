from classic_controllers import Controller
from externally_referenced_state_plot import ExternallyReferencedStatePlot
import gym_electric_motor as gem
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.reference_generators import SinusoidalReferenceGenerator
import time
from enum import Enum
from dataclasses import dataclass

MotorType = Enum(
    "MotorType",
    ["PermanentlyExcitedDcMotor", "ExternallyExcitedDcMotor", "SeriesDc", "ShuntDc"],
)
MotorType.PermanentlyExcitedDcMotor.env_id_tag = "PermExDc"
MotorType.ExternallyExcitedDcMotor.env_id_tag = "ExtExDc"
MotorType.PermanentlyExcitedDcMotor.states = ["omega", "torque", "i", "u"]
MotorType.ExternallyExcitedDcMotor.states = [
    "omega",
    "torque",
    "i_a",
    "i_e",
    "u_a",
    "u_e",
]
MotorType.SeriesDc.states = ["omega", "torque", "i", "u"]
MotorType.ShuntDc.states = ["omega", "torque", "i_a", "i_e", "u"]


ControlType = Enum("ControlType", ["SpeedControl", "TorqueControl", "CurrentControl"])
ControlType.SpeedControl.env_id_tag = "SC"
ControlType.TorqueControl.env_id_tag = "TC"
ControlType.CurrentControl.env_id_tag = "CC"

ActionType = Enum("ActionType", ["Continuous", "Finite"])
ActionType.Continuous.env_id_tag = "Cont"


# We need this helper functions to be backwards compatible with env_id string
def _get_env_id_tag(t) -> str:
    if hasattr(t, "env_id_tag"):
        return t.env_id_tag
    else:
        return t.name


@dataclass
class Motor:
    motor_type: MotorType
    control_type: ControlType
    action_type: ActionType

    def get_env_id(self) -> str:
        return (
            _get_env_id_tag(self.action_type)
            + "-"
            + _get_env_id_tag(self.control_type)
            + "-"
            + _get_env_id_tag(self.motor_type)
            + "-v0"
        )

    def get_state_names(self) -> list[str]:
        return self.motor_type.states


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

    # initialize the gym-electric-motor environment
    env = gem.make(
        motor.get_env_id(),
        visualization=MotorDashboard(additional_plots=external_ref_plots),
        scale_plots=True,
        render_mode="figure",
        reference_generator=ref_generator,
    )

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
    s = 0
    for i in range(10001):
        s += 1
        action = controller.control(state, reference)
        # if i % 100 == 0:
        #   (state, reference), reward, terminated, truncated, _ = env.step(env.action_space.sample())
        # else:
        (state, reference), reward, terminated, truncated, _ = env.step(action)
        print(f"step{s}")
        # viz.render()

        if terminated:
            env.reset()
            controller.reset()

    env.close()
