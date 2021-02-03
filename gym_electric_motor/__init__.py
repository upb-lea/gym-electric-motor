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

# Version 1
register(id='Finite-SC-PermExDc-v0',
         entry_point=envs_path+'FiniteSpeedControlDcPermanentlyExcitedMotorEnv')
register(id='Cont-SC-PermExDc-v0',
         entry_point=envs_path+'ContSpeedControlDcPermanentlyExcitedMotorEnv')
register(id='Finite-TC-PermExDc-v0',
         entry_point=envs_path+'FiniteTorqueControlDcPermanentlyExcitedMotorEnv')
register(id='Cont-TC-PermExDc-v0',
         entry_point=envs_path+'ContTorqueControlDcPermanentlyExcitedMotorEnv')
register(id='Finite-CC-PermExDc-v0',
         entry_point=envs_path+'FiniteCurrentControlDcPermanentlyExcitedMotorEnv')
register(id='Cont-CC-PermExDc-v0',
         entry_point=envs_path+'ContCurrentControlDcPermanentlyExcitedMotorEnv')

register(id='Disc-SC-ExtExDc-v0',
         entry_point=envs_path+'DiscSpeedControlDcExternallyExcitedMotorEnv')
register(id='Cont-SC-ExtExDc-v0',
         entry_point=envs_path+'ContSpeedControlDcExternallyExcitedMotorEnv')
register(id='Disc-TC-ExtExDc-v0',
         entry_point=envs_path+'DiscTorqueControlDcExternallyExcitedMotorEnv')
register(id='Cont-TC-ExtExDc-v0',
         entry_point=envs_path+'ContTorqueControlDcExternallyExcitedMotorEnv')
register(id='Disc-CC-ExtExDc-v0',
         entry_point=envs_path+'DiscCurrentControlDcExternallyExcitedMotorEnv')
register(id='Cont-CC-ExtExDc-v0',
         entry_point=envs_path+'ContCurrentControlDcExternallyExcitedMotorEnv')

register(id='Disc-SC-SeriesDc-v0',
         entry_point=envs_path+'DiscSpeedControlDcSeriesMotorEnv')
register(id='Cont-SC-SeriesDc-v0',
         entry_point=envs_path+'ContSpeedControlDcSeriesMotorEnv')
register(id='Disc-TC-SeriesDc-v0',
         entry_point=envs_path+'DiscTorqueControlDcSeriesMotorEnv')
register(id='Cont-TC-SeriesDc-v0',
         entry_point=envs_path+'ContTorqueControlDcSeriesMotorEnv')
register(id='Disc-CC-SeriesDc-v0',
         entry_point=envs_path+'DiscCurrentControlDcSeriesMotorEnv')
register(id='Cont-CC-SeriesDc-v0',
         entry_point=envs_path+'ContCurrentControlDcSeriesMotorEnv')

register(id='Disc-SC-ShuntDc-v0',
         entry_point=envs_path+'DiscSpeedControlDcShuntMotorEnv')
register(id='Cont-SC-ShuntDc-v0',
         entry_point=envs_path+'ContSpeedControlDcShuntMotorEnv')
register(id='Disc-TC-ShuntDc-v0',
         entry_point=envs_path+'DiscTorqueControlDcShuntMotorEnv')
register(id='Cont-TC-ShuntDc-v0',
         entry_point=envs_path+'ContTorqueControlDcShuntMotorEnv')
register(id='Disc-CC-ShuntDc-v0',
         entry_point=envs_path+'DiscCurrentControlDcShuntMotorEnv')
register(id='Cont-CC-ShuntDc-v0',
         entry_point=envs_path+'ContCurrentControlDcShuntMotorEnv')

register(id='PMSMCont-v1',
         entry_point=envs_path+'ContPermanentMagnetSynchronousMotorEnvironment')
register(id='PMSMDisc-v1',
         entry_point=envs_path+'DiscPermanentMagnetSynchronousMotorEnvironment')
register(id='SynRMCont-v1',
         entry_point=envs_path+'ContSynchronousReluctanceMotorEnvironment')
register(id='SynRMDisc-v1',
         entry_point=envs_path+'DiscSynchronousReluctanceMotorEnvironment')
register(id='SCIMCont-v1',
         entry_point=envs_path+'ContSquirrelCageInductionMotorEnvironment')
register(id='SCIMDisc-v1',
         entry_point=envs_path+'DiscSquirrelCageInductionMotorEnvironment')
register(id='DFIMCont-v1',
         entry_point=envs_path+'ContDoublyFedInductionMotorEnvironment')
register(id='DFIMDisc-v1',
         entry_point=envs_path+'DiscDoublyFedInductionMotorEnvironment')
