from dataclasses import dataclass
from enum import Enum

MotorType = Enum(
    "MotorType",
    ["PermanentlyExcitedDcMotor", "ExternallyExcitedDcMotor", "SeriesDc", "ShuntDc", "ExternallyExcitedSynchronousMotor", "DoublyFedInductionMotor", "SquirrelCageInductionMotor", "PermanentMagnetSynchronousMotor", "SynchronousReluctanceMotor"],
)
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
MotorType.ExternallyExcitedSynchronousMotor.states = ["omega" , "torque", "i_sd", "i_sq", "i_a", "i_b", "i_c", "i_e", "u_sd", "u_sq", "u_a", "u_b", "u_c", "u_e"]
MotorType.DoublyFedInductionMotor.states = [
    "omega" , "torque",
    "i_sa", "i_sb", "i_sc", "i_sd", "i_sq",
    "u_sa", "u_sb", "u_sc", "u_sd", "u_sq",
    "i_ra", "i_rb", "i_rc", "i_rd", "i_rq",
    "u_ra", "u_rb", "u_rc", "u_rd", "u_rq",
    "epsilon"
]
MotorType.SquirrelCageInductionMotor.states = [
    "omega" , "torque",
    "i_sa", "i_sb", "i_sc", "i_sd", "i_sq",
    "u_sa", "u_sb", "u_sc", "u_sd", "u_sq",
    "epsilon"
]
MotorType.PermanentMagnetSynchronousMotor.states = [
    "omega" , "torque", "i_sd", "i_sq", "i_a", "i_b", "i_c", "u_sd", "u_sq", "u_a", "u_b", "u_c"
]
MotorType.SynchronousReluctanceMotor.states = [
    "omega" , "torque", "i_sd", "i_sq", "i_a", "i_b", "i_c", "u_sd", "u_sq", "u_a", "u_b", "u_c"
]


# add env_id_tag if you dont want to use enum name as env_id
MotorType.PermanentlyExcitedDcMotor.env_id_tag = "PermExDc"
MotorType.ExternallyExcitedDcMotor.env_id_tag = "ExtExDc"
MotorType.ExternallyExcitedSynchronousMotor.env_id_tag = "EESM"
MotorType.DoublyFedInductionMotor.env_id_tag = "DFIM"
MotorType.SquirrelCageInductionMotor.env_id_tag = "SCIM"
MotorType.PermanentMagnetSynchronousMotor.env_id_tag = "PMSM"
MotorType.SynchronousReluctanceMotor.env_id_tag = "SynRM"

ControlType = Enum("ControlType", ["SpeedControl", "TorqueControl", "CurrentControl"])
ControlType.SpeedControl.env_id_tag = "SC"
ControlType.TorqueControl.env_id_tag = "TC"
ControlType.CurrentControl.env_id_tag = "CC"

ActionType = Enum("ActionType", ["Continuous", "Finite"])
ActionType.Continuous.env_id_tag = "Cont"


# Check if we added an env_id_tag and use this instead of the enum name
def _to_env_id(t) -> str:
    if hasattr(t, "env_id_tag"):
        return t.env_id_tag
    else:
        return t.name


@dataclass
class Motor:
    motor_type: MotorType
    control_type: ControlType
    action_type: ActionType

    def env_id(self) -> str:
        return (
            _to_env_id(self.action_type)
            + "-"
            + _to_env_id(self.control_type)
            + "-"
            + _to_env_id(self.motor_type)
            + "-v0"
        )

    def states(self) -> list[str]:
        return self.motor_type.states
