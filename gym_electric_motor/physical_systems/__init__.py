from .physical_systems import DcMotorSystem, SynchronousMotorSystem
from .converters import PowerElectronicConverter, DiscOneQuadrantConverter, DiscTwoQuadrantConverter, \
    DiscFourQuadrantConverter, DiscDoubleConverter, DiscB6BridgeConverter, ContOneQuadrantConverter, \
    ContTwoQuadrantConverter, ContFourQuadrantConverter, ContDoubleConverter, ContB6BridgeConverter

from .electric_motors import DcExternallyExcitedMotor, DcSeriesMotor, DcPermanentlyExcitedMotor, DcShuntMotor, \
    PermanentMagnetSynchronousMotor, ElectricMotor, SynchronousReluctanceMotor

from .mechanical_loads import MechanicalLoad, PolynomialStaticLoad

from .solvers import OdeSolver, EulerSolver, ScipyOdeIntSolver, ScipySolveIvpSolver, ScipyOdeSolver

from .noise_generators import NoiseGenerator, GaussianWhiteNoiseGenerator

from .voltage_supplies import VoltageSupply, IdealVoltageSupply


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

register_class(DiscOneQuadrantConverter, PowerElectronicConverter, 'Disc-1QC')
register_class(ContOneQuadrantConverter, PowerElectronicConverter, 'Cont-1QC')
register_class(DiscTwoQuadrantConverter, PowerElectronicConverter, 'Disc-2QC')
register_class(ContTwoQuadrantConverter, PowerElectronicConverter, 'Cont-2QC')
register_class(DiscFourQuadrantConverter, PowerElectronicConverter, 'Disc-4QC')
register_class(ContFourQuadrantConverter, PowerElectronicConverter, 'Cont-4QC')
register_class(DiscDoubleConverter, PowerElectronicConverter, 'Disc-Double')
register_class(ContDoubleConverter, PowerElectronicConverter, 'Cont-Double')
register_class(DiscB6BridgeConverter, PowerElectronicConverter, 'Disc-B6C')
register_class(ContB6BridgeConverter, PowerElectronicConverter, 'Cont-B6C')

register_class(PolynomialStaticLoad, MechanicalLoad, 'PolyStaticLoad')

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

register_class(IdealVoltageSupply, VoltageSupply, 'IdealVoltageSupply')


