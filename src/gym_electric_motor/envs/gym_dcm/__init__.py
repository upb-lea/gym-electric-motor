from .extex_dc_motor_env import (
    ContCurrentControlDcExternallyExcitedMotorEnv,
    ContSpeedControlDcExternallyExcitedMotorEnv,
    ContTorqueControlDcExternallyExcitedMotorEnv,
    FiniteCurrentControlDcExternallyExcitedMotorEnv,
    FiniteSpeedControlDcExternallyExcitedMotorEnv,
    FiniteTorqueControlDcExternallyExcitedMotorEnv,
)
from .permex_dc_motor_env import (
    ContCurrentControlDcPermanentlyExcitedMotorEnv,
    ContSpeedControlDcPermanentlyExcitedMotorEnv,
    ContTorqueControlDcPermanentlyExcitedMotorEnv,
    FiniteCurrentControlDcPermanentlyExcitedMotorEnv,
    FiniteSpeedControlDcPermanentlyExcitedMotorEnv,
    FiniteTorqueControlDcPermanentlyExcitedMotorEnv,
)
from .series_dc_motor_env import (
    ContCurrentControlDcSeriesMotorEnv,
    ContSpeedControlDcSeriesMotorEnv,
    ContTorqueControlDcSeriesMotorEnv,
    FiniteCurrentControlDcSeriesMotorEnv,
    FiniteSpeedControlDcSeriesMotorEnv,
    FiniteTorqueControlDcSeriesMotorEnv,
)
from .shunt_dc_motor_env import (
    ContCurrentControlDcShuntMotorEnv,
    ContSpeedControlDcShuntMotorEnv,
    ContTorqueControlDcShuntMotorEnv,
    FiniteCurrentControlDcShuntMotorEnv,
    FiniteSpeedControlDcShuntMotorEnv,
    FiniteTorqueControlDcShuntMotorEnv,
)
