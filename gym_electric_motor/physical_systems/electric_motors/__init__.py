# Electric Motor Base Class
from .electric_motor import ElectricMotor

# DC Motors
from .dc_motor import DcMotor
from .dc_externally_excited_motor import DcExternallyExcitedMotor
from .dc_permanently_excited_motor import DcPermanentlyExcitedMotor
from .dc_series_motor import DcSeriesMotor
from .dc_shunt_motor import DcShuntMotor

# Three Phase Motors
from .three_phase_motor import ThreePhaseMotor

# Synchronous Motors
from .synchronous_motor import SynchronousMotor
from .synchronous_reluctance_motor import SynchronousReluctanceMotor
from .permanent_magnet_synchronous_motor import PermanentMagnetSynchronousMotor

# Induction Motors
from .induction_motor import InductionMotor
from .squirrel_cage_induction_motor import SquirrelCageInductionMotor
from .doubly_fed_induction_motor import DoublyFedInductionMotor

