from .core import ReferenceGenerator
from .core import PhysicalSystem
from .core import RewardFunction
from .core import ElectricMotorVisualization
from .core import ConstraintMonitor
from .random_component import RandomComponent
from .constraints import Constraint, LimitConstraint
from .utils import make, register_superclass

register_superclass(RewardFunction)
register_superclass(ElectricMotorVisualization)
register_superclass(ReferenceGenerator)
register_superclass(PhysicalSystem)

import gym_electric_motor.reference_generators
import gym_electric_motor.reward_functions
import gym_electric_motor.visualization
import gym_electric_motor.physical_systems
import gym_electric_motor.envs
from gym.envs.registration import register
import gym
from packaging import version
# Add all superclasses of the modules to the registry.

# Deactivate the order enforce wrapper that is put around a created env per default from gym-version 0.21.0 onwards
registration_kwargs = dict(order_enforce=False) if version.parse(gym.__version__) >= version.parse('0.21.0') else dict()

envs_path = 'gym_electric_motor.envs:'

# Permanently Excited DC Motor Environments
register(
    id='Finite-SC-PermExDc-v0',
    entry_point=envs_path+'FiniteSpeedControlDcPermanentlyExcitedMotorEnv',
    **registration_kwargs
)
register(
    id='Cont-SC-PermExDc-v0',
    entry_point=envs_path+'ContSpeedControlDcPermanentlyExcitedMotorEnv',
    **registration_kwargs
)
register(
    id='Finite-TC-PermExDc-v0',
    entry_point=envs_path+'FiniteTorqueControlDcPermanentlyExcitedMotorEnv',
    **registration_kwargs
)
register(
    id='Cont-TC-PermExDc-v0',
    entry_point=envs_path+'ContTorqueControlDcPermanentlyExcitedMotorEnv',
    **registration_kwargs
)
register(
    id='Finite-CC-PermExDc-v0',
    entry_point=envs_path+'FiniteCurrentControlDcPermanentlyExcitedMotorEnv',
    **registration_kwargs
)
register(
    id='Cont-CC-PermExDc-v0',
    entry_point=envs_path+'ContCurrentControlDcPermanentlyExcitedMotorEnv',
    **registration_kwargs
)

# Externally Excited DC Motor Environments
register(
    id='Finite-SC-ExtExDc-v0',
    entry_point=envs_path+'FiniteSpeedControlDcExternallyExcitedMotorEnv',
    **registration_kwargs
)
register(
    id='Cont-SC-ExtExDc-v0',
    entry_point=envs_path+'ContSpeedControlDcExternallyExcitedMotorEnv',
    **registration_kwargs
)
register(
    id='Finite-TC-ExtExDc-v0',
    entry_point=envs_path+'FiniteTorqueControlDcExternallyExcitedMotorEnv',
    **registration_kwargs
)
register(
    id='Cont-TC-ExtExDc-v0',
    entry_point=envs_path+'ContTorqueControlDcExternallyExcitedMotorEnv',
    **registration_kwargs
)
register(
    id='Finite-CC-ExtExDc-v0',
    entry_point=envs_path+'FiniteCurrentControlDcExternallyExcitedMotorEnv',
    **registration_kwargs
)
register(
    id='Cont-CC-ExtExDc-v0',
    entry_point=envs_path+'ContCurrentControlDcExternallyExcitedMotorEnv',
    **registration_kwargs
)

# Series DC Motor Environments
register(
    id='Finite-SC-SeriesDc-v0',
    entry_point=envs_path+'FiniteSpeedControlDcSeriesMotorEnv',
    **registration_kwargs
)
register(
    id='Cont-SC-SeriesDc-v0',
    entry_point=envs_path+'ContSpeedControlDcSeriesMotorEnv',
    **registration_kwargs
)
register(
    id='Finite-TC-SeriesDc-v0',
    entry_point=envs_path+'FiniteTorqueControlDcSeriesMotorEnv',
    **registration_kwargs
)
register(
    id='Cont-TC-SeriesDc-v0',
    entry_point=envs_path+'ContTorqueControlDcSeriesMotorEnv',
    **registration_kwargs
)
register(
    id='Finite-CC-SeriesDc-v0',
    entry_point=envs_path+'FiniteCurrentControlDcSeriesMotorEnv',
    **registration_kwargs
)
register(
    id='Cont-CC-SeriesDc-v0',
    entry_point=envs_path+'ContCurrentControlDcSeriesMotorEnv',
    **registration_kwargs
)

# Shunt DC Motor Environments
register(
    id='Finite-SC-ShuntDc-v0',
    entry_point=envs_path+'FiniteSpeedControlDcShuntMotorEnv',
    **registration_kwargs
)
register(
    id='Cont-SC-ShuntDc-v0',
    entry_point=envs_path+'ContSpeedControlDcShuntMotorEnv',
    **registration_kwargs
)
register(
    id='Finite-TC-ShuntDc-v0',
    entry_point=envs_path+'FiniteTorqueControlDcShuntMotorEnv',
    **registration_kwargs
)
register(
    id='Cont-TC-ShuntDc-v0',
    entry_point=envs_path+'ContTorqueControlDcShuntMotorEnv',
    **registration_kwargs
)
register(
    id='Finite-CC-ShuntDc-v0',
    entry_point=envs_path+'FiniteCurrentControlDcShuntMotorEnv',
    **registration_kwargs
)
register(
    id='Cont-CC-ShuntDc-v0',
    entry_point=envs_path+'ContCurrentControlDcShuntMotorEnv',
    **registration_kwargs
)

# Permanent Magnet Synchronous Motor Environments
register(
    id='Finite-SC-PMSM-v0',
    entry_point=envs_path+'FiniteSpeedControlPermanentMagnetSynchronousMotorEnv',
    **registration_kwargs
)
register(
    id='Finite-TC-PMSM-v0',
    entry_point=envs_path+'FiniteTorqueControlPermanentMagnetSynchronousMotorEnv',
    **registration_kwargs
)
register(
    id='Finite-CC-PMSM-v0',
    entry_point=envs_path+'FiniteCurrentControlPermanentMagnetSynchronousMotorEnv',
    **registration_kwargs
)
register(
    id='AbcCont-CC-PMSM-v0',
    entry_point=envs_path+'AbcContCurrentControlPermanentMagnetSynchronousMotorEnv',
    **registration_kwargs
)
register(
    id='AbcCont-TC-PMSM-v0',
    entry_point=envs_path+'AbcContTorqueControlPermanentMagnetSynchronousMotorEnv',
    **registration_kwargs
)
register(
    id='AbcCont-SC-PMSM-v0',
    entry_point=envs_path+'AbcContSpeedControlPermanentMagnetSynchronousMotorEnv',
    **registration_kwargs
)
register(
    id='DqCont-CC-PMSM-v0',
    entry_point=envs_path+'DqContCurrentControlPermanentMagnetSynchronousMotorEnv',
    **registration_kwargs
)
register(
    id='DqCont-TC-PMSM-v0',
    entry_point=envs_path+'DqContTorqueControlPermanentMagnetSynchronousMotorEnv',
    **registration_kwargs
)
register(
    id='DqCont-SC-PMSM-v0',
    entry_point=envs_path+'DqContSpeedControlPermanentMagnetSynchronousMotorEnv',
    **registration_kwargs
)

# Synchronous Reluctance Motor Environments
register(
    id='Finite-SC-SynRM-v0',
    entry_point=envs_path+'FiniteSpeedControlSynchronousReluctanceMotorEnv',
    **registration_kwargs
)
register(
    id='Finite-TC-SynRM-v0',
    entry_point=envs_path+'FiniteTorqueControlSynchronousReluctanceMotorEnv',
    **registration_kwargs
)
register(
    id='Finite-CC-SynRM-v0',
    entry_point=envs_path+'FiniteCurrentControlSynchronousReluctanceMotorEnv',
    **registration_kwargs
)
register(
    id='AbcCont-CC-SynRM-v0',
    entry_point=envs_path+'AbcContCurrentControlSynchronousReluctanceMotorEnv',
    **registration_kwargs
)
register(
    id='AbcCont-TC-SynRM-v0',
    entry_point=envs_path+'AbcContTorqueControlSynchronousReluctanceMotorEnv',
    **registration_kwargs
)
register(
    id='AbcCont-SC-SynRM-v0',
    entry_point=envs_path+'AbcContSpeedControlSynchronousReluctanceMotorEnv',
    **registration_kwargs
)
register(
    id='DqCont-CC-SynRM-v0',
    entry_point=envs_path+'DqContCurrentControlSynchronousReluctanceMotorEnv',
    **registration_kwargs
)
register(
    id='DqCont-TC-SynRM-v0',
    entry_point=envs_path+'DqContTorqueControlSynchronousReluctanceMotorEnv',
    **registration_kwargs
)
register(
    id='DqCont-SC-SynRM-v0',
    entry_point=envs_path+'DqContSpeedControlSynchronousReluctanceMotorEnv',
    **registration_kwargs
)

# Squirrel Cage Induction Motor Environments
register(
    id='Finite-SC-SCIM-v0',
    entry_point=envs_path+'FiniteSpeedControlSquirrelCageInductionMotorEnv',
    **registration_kwargs
)
register(
    id='Finite-TC-SCIM-v0',
    entry_point=envs_path+'FiniteTorqueControlSquirrelCageInductionMotorEnv',
    **registration_kwargs
)
register(
    id='Finite-CC-SCIM-v0',
    entry_point=envs_path+'FiniteCurrentControlSquirrelCageInductionMotorEnv',
    **registration_kwargs
)
register(
    id='AbcCont-CC-SCIM-v0',
    entry_point=envs_path+'AbcContCurrentControlSquirrelCageInductionMotorEnv',
    **registration_kwargs
)
register(
    id='AbcCont-TC-SCIM-v0',
    entry_point=envs_path+'AbcContTorqueControlSquirrelCageInductionMotorEnv',
    **registration_kwargs
)
register(
    id='AbcCont-SC-SCIM-v0',
    entry_point=envs_path+'AbcContSpeedControlSquirrelCageInductionMotorEnv',
    **registration_kwargs
)
register(
    id='DqCont-CC-SCIM-v0',
    entry_point=envs_path+'DqContCurrentControlSquirrelCageInductionMotorEnv',
    **registration_kwargs
)
register(
    id='DqCont-TC-SCIM-v0',
    entry_point=envs_path+'DqContTorqueControlSquirrelCageInductionMotorEnv',
    **registration_kwargs
)
register(
    id='DqCont-SC-SCIM-v0',
    entry_point=envs_path+'DqContSpeedControlSquirrelCageInductionMotorEnv',
    **registration_kwargs
)

# Doubly Fed Induction Motor Environments
register(
    id='Finite-SC-DFIM-v0',
    entry_point=envs_path+'FiniteSpeedControlDoublyFedInductionMotorEnv',
    **registration_kwargs
)
register(
    id='Finite-TC-DFIM-v0',
    entry_point=envs_path+'FiniteTorqueControlDoublyFedInductionMotorEnv',
    **registration_kwargs
)
register(
    id='Finite-CC-DFIM-v0',
    entry_point=envs_path+'FiniteCurrentControlDoublyFedInductionMotorEnv',
    **registration_kwargs
)
register(
    id='AbcCont-CC-DFIM-v0',
    entry_point=envs_path+'AbcContCurrentControlDoublyFedInductionMotorEnv',
    **registration_kwargs
)
register(
    id='AbcCont-TC-DFIM-v0',
    entry_point=envs_path+'AbcContTorqueControlDoublyFedInductionMotorEnv',
    **registration_kwargs
)
register(
    id='AbcCont-SC-DFIM-v0',
    entry_point=envs_path+'AbcContSpeedControlDoublyFedInductionMotorEnv',
    **registration_kwargs
)
register(
    id='DqCont-CC-DFIM-v0',
    entry_point=envs_path+'DqContCurrentControlDoublyFedInductionMotorEnv',
    **registration_kwargs
)
register(
    id='DqCont-TC-DFIM-v0',
    entry_point=envs_path+'DqContTorqueControlDoublyFedInductionMotorEnv',
    **registration_kwargs
)
register(
    id='DqCont-SC-DFIM-v0',
    entry_point=envs_path+'DqContSpeedControlDoublyFedInductionMotorEnv',
    **registration_kwargs
)
