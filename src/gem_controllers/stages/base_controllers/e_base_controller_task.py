from enum import Enum


class EBaseControllerTask(Enum):
    CC = "CC"
    CurrentControl = "CC"
    TC = "TC"
    TorqueControl = "TC"
    SC = "SC"
    SpeedControl = "SC"
    FC = "FC"
    FluxControl = "FC"

    @staticmethod
    def equals(value, base_controller_task):
        if isinstance(value, str):
            return EBaseControllerTask(value) == base_controller_task
        else:
            return value == base_controller_task

    @staticmethod
    def get_control_task(value):
        if isinstance(value, str):
            return EBaseControllerTask(value)
        elif isinstance(value, EBaseControllerTask):
            return value
        else:
            raise Exception(
                f"The type of the control task has to be string or EBaseControllerTask but is {type(value)}."
            )
