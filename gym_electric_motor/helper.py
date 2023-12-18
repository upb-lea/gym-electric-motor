from enum import Enum
from dataclasses import dataclass

RenderMode = Enum("RenderMode", ["Figure", "FigureOnce"])


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
