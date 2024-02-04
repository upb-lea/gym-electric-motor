from .motors import Motor, MotorType, ControlType, ActionType

# Version 1
from .gym_dcm.permex_dc_motor_env import ContSpeedControlDcPermanentlyExcitedMotorEnv
from .gym_dcm.permex_dc_motor_env import FiniteSpeedControlDcPermanentlyExcitedMotorEnv
from .gym_dcm.permex_dc_motor_env import ContTorqueControlDcPermanentlyExcitedMotorEnv
from .gym_dcm.permex_dc_motor_env import FiniteTorqueControlDcPermanentlyExcitedMotorEnv
from .gym_dcm.permex_dc_motor_env import ContCurrentControlDcPermanentlyExcitedMotorEnv
from .gym_dcm.permex_dc_motor_env import (
    FiniteCurrentControlDcPermanentlyExcitedMotorEnv,
)

from .gym_dcm.extex_dc_motor_env import ContSpeedControlDcExternallyExcitedMotorEnv
from .gym_dcm.extex_dc_motor_env import FiniteSpeedControlDcExternallyExcitedMotorEnv
from .gym_dcm.extex_dc_motor_env import ContTorqueControlDcExternallyExcitedMotorEnv
from .gym_dcm.extex_dc_motor_env import FiniteTorqueControlDcExternallyExcitedMotorEnv
from .gym_dcm.extex_dc_motor_env import ContCurrentControlDcExternallyExcitedMotorEnv
from .gym_dcm.extex_dc_motor_env import FiniteCurrentControlDcExternallyExcitedMotorEnv

from .gym_dcm.series_dc_motor_env import ContSpeedControlDcSeriesMotorEnv
from .gym_dcm.series_dc_motor_env import FiniteSpeedControlDcSeriesMotorEnv
from .gym_dcm.series_dc_motor_env import ContTorqueControlDcSeriesMotorEnv
from .gym_dcm.series_dc_motor_env import FiniteTorqueControlDcSeriesMotorEnv
from .gym_dcm.series_dc_motor_env import ContCurrentControlDcSeriesMotorEnv
from .gym_dcm.series_dc_motor_env import FiniteCurrentControlDcSeriesMotorEnv

from .gym_dcm.shunt_dc_motor_env import ContSpeedControlDcShuntMotorEnv
from .gym_dcm.shunt_dc_motor_env import FiniteSpeedControlDcShuntMotorEnv
from .gym_dcm.shunt_dc_motor_env import ContTorqueControlDcShuntMotorEnv
from .gym_dcm.shunt_dc_motor_env import FiniteTorqueControlDcShuntMotorEnv
from .gym_dcm.shunt_dc_motor_env import ContCurrentControlDcShuntMotorEnv
from .gym_dcm.shunt_dc_motor_env import FiniteCurrentControlDcShuntMotorEnv

from .gym_pmsm.finite_sc_pmsm_env import (
    FiniteSpeedControlPermanentMagnetSynchronousMotorEnv,
)
from .gym_pmsm.finite_cc_pmsm_env import (
    FiniteCurrentControlPermanentMagnetSynchronousMotorEnv,
)
from .gym_pmsm.finite_tc_pmsm_env import (
    FiniteTorqueControlPermanentMagnetSynchronousMotorEnv,
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

from .gym_eesm.finite_sc_eesm_env import (
    FiniteSpeedControlExternallyExcitedSynchronousMotorEnv,
)
from .gym_eesm.finite_cc_eesm_env import (
    FiniteCurrentControlExternallyExcitedSynchronousMotorEnv,
)
from .gym_eesm.finite_tc_eesm_env import (
    FiniteTorqueControlExternallyExcitedSynchronousMotorEnv,
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

from .gym_synrm.finite_sc_synrm_env import (
    FiniteSpeedControlSynchronousReluctanceMotorEnv,
)
from .gym_synrm.finite_cc_synrm_env import (
    FiniteCurrentControlSynchronousReluctanceMotorEnv,
)
from .gym_synrm.finite_tc_synrm_env import (
    FiniteTorqueControlSynchronousReluctanceMotorEnv,
)
from .gym_synrm.cont_tc_synrm_env import ContTorqueControlSynchronousReluctanceMotorEnv
from .gym_synrm.cont_cc_synrm_env import ContCurrentControlSynchronousReluctanceMotorEnv
from .gym_synrm.cont_sc_synrm_env import ContSpeedControlSynchronousReluctanceMotorEnv

from .gym_im import ContSpeedControlSquirrelCageInductionMotorEnv
from .gym_im import ContCurrentControlSquirrelCageInductionMotorEnv
from .gym_im import ContTorqueControlSquirrelCageInductionMotorEnv
from .gym_im import FiniteSpeedControlSquirrelCageInductionMotorEnv
from .gym_im import FiniteCurrentControlSquirrelCageInductionMotorEnv
from .gym_im import FiniteTorqueControlSquirrelCageInductionMotorEnv

from .gym_im import ContSpeedControlDoublyFedInductionMotorEnv
from .gym_im import ContCurrentControlDoublyFedInductionMotorEnv
from .gym_im import ContTorqueControlDoublyFedInductionMotorEnv
from .gym_im import FiniteSpeedControlDoublyFedInductionMotorEnv
from .gym_im import FiniteCurrentControlDoublyFedInductionMotorEnv
from .gym_im import FiniteTorqueControlDoublyFedInductionMotorEnv