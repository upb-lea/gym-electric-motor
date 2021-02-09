from .core import ReferenceGenerator
from .core import PhysicalSystem
from .core import RewardFunction
from .core import ElectricMotorVisualization
from .core import ConstraintMonitor
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

# Add all superclasses of the modules to the registry.


envs_path = 'gym_electric_motor.envs:'

# Permanently Excited DC Motor Environments
register(
    id='Finite-SC-PermExDc-v0',
    entry_point=envs_path+'FiniteSpeedControlDcPermanentlyExcitedMotorEnv'
)
register(
    id='Cont-SC-PermExDc-v0',
    entry_point=envs_path+'ContSpeedControlDcPermanentlyExcitedMotorEnv'
)
register(
    id='Finite-TC-PermExDc-v0',
    entry_point=envs_path+'FiniteTorqueControlDcPermanentlyExcitedMotorEnv'
)
register(
    id='Cont-TC-PermExDc-v0',
    entry_point=envs_path+'ContTorqueControlDcPermanentlyExcitedMotorEnv'
)
register(
    id='Finite-CC-PermExDc-v0',
    entry_point=envs_path+'FiniteCurrentControlDcPermanentlyExcitedMotorEnv'
)
register(
    id='Cont-CC-PermExDc-v0',
    entry_point=envs_path+'ContCurrentControlDcPermanentlyExcitedMotorEnv'
)

# Externally Excited DC Motor Environments
register(
    id='Finite-SC-ExtExDc-v0',
    entry_point=envs_path+'FiniteSpeedControlDcExternallyExcitedMotorEnv'
)
register(
    id='Cont-SC-ExtExDc-v0',
    entry_point=envs_path+'ContSpeedControlDcExternallyExcitedMotorEnv'
)
register(
    id='Finite-TC-ExtExDc-v0',
    entry_point=envs_path+'FiniteTorqueControlDcExternallyExcitedMotorEnv'
)
register(
    id='Cont-TC-ExtExDc-v0',
    entry_point=envs_path+'ContTorqueControlDcExternallyExcitedMotorEnv'
)
register(
    id='Finite-CC-ExtExDc-v0',
    entry_point=envs_path+'FiniteCurrentControlDcExternallyExcitedMotorEnv'
)
register(
    id='Cont-CC-ExtExDc-v0',
    entry_point=envs_path+'ContCurrentControlDcExternallyExcitedMotorEnv'
)

# Series DC Motor Environments
register(
    id='Finite-SC-SeriesDc-v0',
    entry_point=envs_path+'FiniteSpeedControlDcSeriesMotorEnv'
)
register(
    id='Cont-SC-SeriesDc-v0',
    entry_point=envs_path+'ContSpeedControlDcSeriesMotorEnv'
)
register(
    id='Finite-TC-SeriesDc-v0',
    entry_point=envs_path+'FiniteTorqueControlDcSeriesMotorEnv'
)
register(
    id='Cont-TC-SeriesDc-v0',
    entry_point=envs_path+'ContTorqueControlDcSeriesMotorEnv'
)
register(
    id='Finite-CC-SeriesDc-v0',
    entry_point=envs_path+'FiniteCurrentControlDcSeriesMotorEnv'
)
register(
    id='Cont-CC-SeriesDc-v0',
    entry_point=envs_path+'ContCurrentControlDcSeriesMotorEnv'
)

# Shunt DC Motor Environments
register(
    id='Finite-SC-ShuntDc-v0',
    entry_point=envs_path+'FiniteSpeedControlDcShuntMotorEnv'
)
register(
    id='Cont-SC-ShuntDc-v0',
    entry_point=envs_path+'ContSpeedControlDcShuntMotorEnv'
)
register(
    id='Finite-TC-ShuntDc-v0',
    entry_point=envs_path+'FiniteTorqueControlDcShuntMotorEnv'
)
register(
    id='Cont-TC-ShuntDc-v0',
    entry_point=envs_path+'ContTorqueControlDcShuntMotorEnv'
)
register(
    id='Finite-CC-ShuntDc-v0',
    entry_point=envs_path+'FiniteCurrentControlDcShuntMotorEnv'
)
register(
    id='Cont-CC-ShuntDc-v0',
    entry_point=envs_path+'ContCurrentControlDcShuntMotorEnv'
)

# Permanent Magnet Synchronous Motor Environments
register(
    id='Finite-SC-PMSM-v0',
    entry_point=envs_path+'FiniteSpeedControlPermanentMagnetSynchronousMotorEnv'
)
register(
    id='Finite-TC-PMSM-v0',
    entry_point=envs_path+'FiniteTorqueControlPermanentMagnetSynchronousMotorEnv'
)
register(
    id='Finite-CC-PMSM-v0',
    entry_point=envs_path+'FiniteCurrentControlPermanentMagnetSynchronousMotorEnv'
)
register(
    id='AbcCont-CC-PMSM-v0',
    entry_point=envs_path+'AbcContCurrentControlPermanentMagnetSynchronousMotorEnv'
)
register(
    id='AbcCont-TC-PMSM-v0',
    entry_point=envs_path+'AbcContTorqueControlPermanentMagnetSynchronousMotorEnv'
)
register(
    id='AbcCont-SC-PMSM-v0',
    entry_point=envs_path+'AbcContSpeedControlPermanentMagnetSynchronousMotorEnv'
)
register(
    id='DqCont-CC-PMSM-v0',
    entry_point=envs_path+'DqContCurrentControlPermanentMagnetSynchronousMotorEnv'
)
register(
    id='DqCont-TC-PMSM-v0',
    entry_point=envs_path+'DqContTorqueControlPermanentMagnetSynchronousMotorEnv'
)
register(
    id='DqCont-SC-PMSM-v0',
    entry_point=envs_path+'DqContSpeedControlPermanentMagnetSynchronousMotorEnv'
)

# Synchronous Reluctance Motor Environments
register(
    id='Finite-SC-SynRM-v0',
    entry_point=envs_path+'FiniteSpeedControlPermanentMagnetSynchronousMotorEnv'
)
register(
    id='Finite-TC-SynRM-v0',
    entry_point=envs_path+'FiniteTorqueControlPermanentMagnetSynchronousMotorEnv'
)
register(
    id='Finite-CC-SynRM-v0',
    entry_point=envs_path+'FiniteCurrentControlPermanentMagnetSynchronousMotorEnv'
)
register(
    id='AbcCont-CC-SynRM-v0',
    entry_point=envs_path+'AbcContCurrentControlPermanentMagnetSynchronousMotorEnv'
)
register(
    id='AbcCont-TC-SynRM-v0',
    entry_point=envs_path+'AbcContTorqueControlPermanentMagnetSynchronousMotorEnv'
)
register(
    id='AbcCont-SC-SynRM-v0',
    entry_point=envs_path+'AbcContSpeedControlPermanentMagnetSynchronousMotorEnv'
)
register(
    id='DqCont-CC-SynRM-v0',
    entry_point=envs_path+'DqContCurrentControlPermanentMagnetSynchronousMotorEnv'
)
register(
    id='DqCont-TC-SynRM-v0',
    entry_point=envs_path+'DqContTorqueControlPermanentMagnetSynchronousMotorEnv'
)
register(
    id='DqCont-SC-SynRM-v0',
    entry_point=envs_path+'DqContSpeedControlPermanentMagnetSynchronousMotorEnv'
)


register(id='SCIMCont-v1',
         entry_point=envs_path+'ContSquirrelCageInductionMotorEnvironment')
register(id='SCIMDisc-v1',
         entry_point=envs_path+'DiscSquirrelCageInductionMotorEnvironment')
register(id='DFIMCont-v1',
         entry_point=envs_path+'ContDoublyFedInductionMotorEnvironment')
register(id='DFIMDisc-v1',
         entry_point=envs_path+'DiscDoublyFedInductionMotorEnvironment')
