from .core import ReferenceGenerator
from .core import PhysicalSystem
from .core import RewardFunction
from .core import ElectricMotorVisualization
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

"""register(id='emotor-asm-cont-v0',
         entry_point=envs_path+'ASMContinuousControlEnv')
register(id='emotor-asm-disc-v0',
         entry_point=envs_path+'ASMContinuousControlEnv')"""

register(id='emotor-pmsm-cont-v0',
         entry_point=envs_path+'PmsmCont')
register(id='emotor-pmsm-disc-v0',
         entry_point=envs_path+'PmsmDisc')

# Version 1
register(id='emotor-dc-permex-cont-v1',
         entry_point=envs_path+'ContDcPermanentlyExcitedMotorEnvironment')
register(id='emotor-dc-permex-disc-v1',
         entry_point=envs_path+'DiscDcPermanentlyExcitedMotorEnvironment')
register(id='emotor-dc-extex-cont-v1',
         entry_point=envs_path+'ContDcExternallyExcitedMotorEnvironment')
register(id='emotor-dc-extex-disc-v1',
         entry_point=envs_path+'DiscDcExternallyExcitedMotorEnvironment')
register(id='emotor-dc-series-cont-v1',
         entry_point=envs_path+'ContDcSeriesMotorEnvironment')
register(id='emotor-dc-series-disc-v1',
         entry_point=envs_path+'DiscDcSeriesMotorEnvironment')
register(id='emotor-dc-shunt-cont-v1',
         entry_point=envs_path+'ContDcShuntMotorEnvironment')
register(id='emotor-dc-shunt-disc-v1',
         entry_point=envs_path+'DiscDcShuntMotorEnvironment')
register(id='emotor-pmsm-cont-v1',
         entry_point=envs_path+'ContPermanentMagnetSynchronousMotorEnvironment')
register(id='emotor-pmsm-disc-v1',
         entry_point=envs_path+'DiscPermanentMagnetSynchronousMotorEnvironment')
register(id='emotor-synrm-cont-v1',
         entry_point=envs_path+'ContSynchronousReluctanceMotorEnvironment')
register(id='emotor-synrm-disc-v1',
         entry_point=envs_path+'DiscSynchronousReluctanceMotorEnvironment')
