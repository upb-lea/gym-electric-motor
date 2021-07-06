from .physical_systems import DcMotorSystem, SynchronousMotorSystem, SquirrelCageInductionMotorSystem, DoublyFedInductionMotorSystem
from .converters import PowerElectronicConverter, FiniteOneQuadrantConverter, FiniteTwoQuadrantConverter, \
    FiniteFourQuadrantConverter, FiniteMultiConverter, FiniteB6BridgeConverter, ContOneQuadrantConverter, \
    ContTwoQuadrantConverter, ContFourQuadrantConverter, ContMultiConverter, ContB6BridgeConverter, NoConverter

from .electric_motors import DcExternallyExcitedMotor, DcSeriesMotor, DcPermanentlyExcitedMotor, DcShuntMotor, \
    PermanentMagnetSynchronousMotor, ElectricMotor, SynchronousReluctanceMotor, SquirrelCageInductionMotor, \
    DoublyFedInductionMotor

from .mechanical_loads import MechanicalLoad, PolynomialStaticLoad, ExternalSpeedLoad, ConstantSpeedLoad, \
    OrnsteinUhlenbeckLoad

from .solvers import OdeSolver, EulerSolver, ScipyOdeIntSolver, ScipySolveIvpSolver, ScipyOdeSolver

from .noise_generators import NoiseGenerator, GaussianWhiteNoiseGenerator

from .voltage_supplies import VoltageSupply, IdealVoltageSupply, RCVoltageSupply, AC1PhaseSupply, AC3PhaseSupply


from ..utils import register_class, register_superclass
from .. import PhysicalSystem

register_superclass(PowerElectronicConverter)
register_superclass(MechanicalLoad)
register_superclass(ElectricMotor)
register_superclass(OdeSolver)
register_superclass(NoiseGenerator)
register_superclass(VoltageSupply)


register_class(DcMotorSystem, PhysicalSystem, 'DcMotorSystem')
register_class(SynchronousMotorSystem, PhysicalSystem, 'SyncMotorSystem')
register_class(SquirrelCageInductionMotorSystem, PhysicalSystem, 'SquirrelCageInductionMotorSystem')
register_class(DoublyFedInductionMotorSystem, PhysicalSystem, 'DoublyFedInductionMotorSystem')

register_class(FiniteOneQuadrantConverter, PowerElectronicConverter, 'Finite-1QC')
register_class(ContOneQuadrantConverter, PowerElectronicConverter, 'Cont-1QC')
register_class(FiniteTwoQuadrantConverter, PowerElectronicConverter, 'Finite-2QC')
register_class(ContTwoQuadrantConverter, PowerElectronicConverter, 'Cont-2QC')
register_class(FiniteFourQuadrantConverter, PowerElectronicConverter, 'Finite-4QC')
register_class(ContFourQuadrantConverter, PowerElectronicConverter, 'Cont-4QC')
register_class(FiniteMultiConverter, PowerElectronicConverter, 'Finite-Multi')
register_class(ContMultiConverter, PowerElectronicConverter, 'Cont-Multi')
register_class(FiniteB6BridgeConverter, PowerElectronicConverter, 'Finite-B6C')
register_class(ContB6BridgeConverter, PowerElectronicConverter, 'Cont-B6C')
register_class(NoConverter, PowerElectronicConverter, 'NoConverter')

register_class(PolynomialStaticLoad, MechanicalLoad, 'PolyStaticLoad')
register_class(ConstantSpeedLoad, MechanicalLoad, 'ConstSpeedLoad')
register_class(ExternalSpeedLoad, MechanicalLoad, 'ExtSpeedLoad')

register_class(GaussianWhiteNoiseGenerator, NoiseGenerator, 'GWN')

register_class(EulerSolver, OdeSolver, 'euler')
register_class(ScipyOdeSolver, OdeSolver, 'scipy.ode')
register_class(ScipySolveIvpSolver, OdeSolver, 'scipy.solve_ivp')
register_class(ScipyOdeIntSolver, OdeSolver, 'scipy.odeint')

register_class(DcSeriesMotor, ElectricMotor, 'DcSeries')
register_class(DcPermanentlyExcitedMotor, ElectricMotor, 'DcPermEx')
register_class(DcExternallyExcitedMotor, ElectricMotor, 'DcExtEx')
register_class(DcShuntMotor, ElectricMotor, 'DcShunt')
register_class(PermanentMagnetSynchronousMotor, ElectricMotor, 'PMSM')
register_class(SynchronousReluctanceMotor, ElectricMotor, 'SynRM')
register_class(SquirrelCageInductionMotor, ElectricMotor, 'SCIM')
register_class(DoublyFedInductionMotor, ElectricMotor, 'DFIM')

register_class(IdealVoltageSupply, VoltageSupply, 'IdealVoltageSupply')
register_class(RCVoltageSupply, VoltageSupply, 'RCVoltageSupply')
register_class(AC1PhaseSupply, VoltageSupply, 'AC1PhaseSupply')
register_class(AC3PhaseSupply, VoltageSupply, 'AC3PhaseSupply')




