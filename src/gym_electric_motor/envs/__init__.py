from .gym_dcm.extex_dc_motor_env import (
    ContCurrentControlDcExternallyExcitedMotorEnv,
    ContSpeedControlDcExternallyExcitedMotorEnv,
    ContTorqueControlDcExternallyExcitedMotorEnv,
    FiniteCurrentControlDcExternallyExcitedMotorEnv,
    FiniteSpeedControlDcExternallyExcitedMotorEnv,
    FiniteTorqueControlDcExternallyExcitedMotorEnv,
)

# Version 1
from .gym_dcm.permex_dc_motor_env import (
    ContCurrentControlDcPermanentlyExcitedMotorEnv,
    ContSpeedControlDcPermanentlyExcitedMotorEnv,
    ContTorqueControlDcPermanentlyExcitedMotorEnv,
    FiniteCurrentControlDcPermanentlyExcitedMotorEnv,
    FiniteSpeedControlDcPermanentlyExcitedMotorEnv,
    FiniteTorqueControlDcPermanentlyExcitedMotorEnv,
)
from .gym_dcm.series_dc_motor_env import (
    ContCurrentControlDcSeriesMotorEnv,
    ContSpeedControlDcSeriesMotorEnv,
    ContTorqueControlDcSeriesMotorEnv,
    FiniteCurrentControlDcSeriesMotorEnv,
    FiniteSpeedControlDcSeriesMotorEnv,
    FiniteTorqueControlDcSeriesMotorEnv,
)
from .gym_dcm.shunt_dc_motor_env import (
    ContCurrentControlDcShuntMotorEnv,
    ContSpeedControlDcShuntMotorEnv,
    ContTorqueControlDcShuntMotorEnv,
    FiniteCurrentControlDcShuntMotorEnv,
    FiniteSpeedControlDcShuntMotorEnv,
    FiniteTorqueControlDcShuntMotorEnv,
)
from .gym_eesm.cont_cc_eesm_env import (
    ContCurrentControlExternallyExcitedSynchronousMotorEnv,
)
from .gym_eesm.cont_sc_eesm_env import (
    ContSpeedControlExternallyExcitedSynchronousMotorEnv,
)
from .gym_eesm.cont_tc_eesm_env import (
    ContTorqueControlExternallyExcitedSynchronousMotorEnv,
)
from .gym_eesm.finite_cc_eesm_env import (
    FiniteCurrentControlExternallyExcitedSynchronousMotorEnv,
)
from .gym_eesm.finite_sc_eesm_env import (
    FiniteSpeedControlExternallyExcitedSynchronousMotorEnv,
)
from .gym_eesm.finite_tc_eesm_env import (
    FiniteTorqueControlExternallyExcitedSynchronousMotorEnv,
)
from .gym_im import (
    ContCurrentControlDoublyFedInductionMotorEnv,
    ContCurrentControlSquirrelCageInductionMotorEnv,
    ContSpeedControlDoublyFedInductionMotorEnv,
    ContSpeedControlSquirrelCageInductionMotorEnv,
    ContTorqueControlDoublyFedInductionMotorEnv,
    ContTorqueControlSquirrelCageInductionMotorEnv,
    FiniteCurrentControlDoublyFedInductionMotorEnv,
    FiniteCurrentControlSquirrelCageInductionMotorEnv,
    FiniteSpeedControlDoublyFedInductionMotorEnv,
    FiniteSpeedControlSquirrelCageInductionMotorEnv,
    FiniteTorqueControlDoublyFedInductionMotorEnv,
    FiniteTorqueControlSquirrelCageInductionMotorEnv,
)
from .gym_pmsm.cont_cc_pmsm_env import (
    ContCurrentControlPermanentMagnetSynchronousMotorEnv,
)
from .gym_pmsm.cont_sc_pmsm_env import (
    ContSpeedControlPermanentMagnetSynchronousMotorEnv,
)
from .gym_pmsm.cont_tc_pmsm_env import (
    ContTorqueControlPermanentMagnetSynchronousMotorEnv,
)
from .gym_pmsm.finite_cc_pmsm_env import (
    FiniteCurrentControlPermanentMagnetSynchronousMotorEnv,
)
from .gym_pmsm.finite_sc_pmsm_env import (
    FiniteSpeedControlPermanentMagnetSynchronousMotorEnv,
)
from .gym_pmsm.finite_tc_pmsm_env import (
    FiniteTorqueControlPermanentMagnetSynchronousMotorEnv,
)
from .gym_synrm.cont_cc_synrm_env import ContCurrentControlSynchronousReluctanceMotorEnv
from .gym_synrm.cont_sc_synrm_env import ContSpeedControlSynchronousReluctanceMotorEnv
from .gym_synrm.cont_tc_synrm_env import ContTorqueControlSynchronousReluctanceMotorEnv
from .gym_synrm.finite_cc_synrm_env import (
    FiniteCurrentControlSynchronousReluctanceMotorEnv,
)
from .gym_synrm.finite_sc_synrm_env import (
    FiniteSpeedControlSynchronousReluctanceMotorEnv,
)
from .gym_synrm.finite_tc_synrm_env import (
    FiniteTorqueControlSynchronousReluctanceMotorEnv,
)
from .motors import ActionType, ControlType, Motor, MotorType
