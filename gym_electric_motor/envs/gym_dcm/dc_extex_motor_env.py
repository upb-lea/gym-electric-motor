from ...core import ElectricMotorEnvironment
from ...physical_systems.physical_systems import DcMotorSystem
from ...reference_generators import WienerProcessReferenceGenerator
from ...reward_functions import WeightedSumOfErrors


class DcExternallyExcitedMotorEnvironment(ElectricMotorEnvironment):

    def __init__(self, motor='DcExtEx', reward_function=None, reference_generator=None, physical_system=None,
                 constraints=('i_a', 'i_e'), **kwargs):
        """
        Args:
            motor(ElectricMotor): Electric Motor used in the PhysicalSystem
            reward_function(RewardFunction): Reward Function for the environment
            reference_generator(ReferenceGenerator): Reference Generator for the environment
            kwargs(dict): Further kwargs to pass to the superclass and the submodules
        """
        physical_system = physical_system or DcMotorSystem(motor=motor, **kwargs)
        reference_generator = reference_generator or WienerProcessReferenceGenerator(**kwargs)
        reward_function = reward_function or WeightedSumOfErrors(**kwargs)
        super().__init__(
            physical_system, reference_generator=reference_generator, reward_function=reward_function,
            constraints=constraints, **kwargs
        )


class DiscDcExternallyExcitedMotorEnvironment(DcExternallyExcitedMotorEnvironment):
    """
    Description:
        Environment to simulate a discretely controlled externally excited DC Motor

    Key:
        `DcExtExDisc-v1`

    Default Modules:

        Physical System:
            SCMLSystem/DcMotorSystem with:
                | IdealVoltageSupply
                | DiscDoubleConverter(subconverters=('Disc-4QC', 'Disc-1QC'))
                | DcExternallyExcitedMotor
                | PolynomialStaticLoad
                | GaussianWhiteNoiseGenerator
                | EulerSolver
                | tau=1e-5

        Reference Generator:
            WienerProcessReferenceGenerator
                Reference Quantity. 'omega'

        Reward Function:
            WeightedSumOfErrors(reward_weights= {'omega': 1 })

        Visualization:
            ElectricMotorVisualization (Dummy for no Visualization)

    State Variables:
        ``['omega' , 'torque', 'i_a', 'i_e', 'u_a', 'u_e', 'u_sup']``

    Observation Space:
        Type: Tuple(State_Space, Reference_Space)

    State Space:
        Box(low=[-1, -1, -1, -1, -1, -1, 0], high=[1, 1, 1, 1, 1, 1, 1])

    Reference Space:
        Box(low=[-1], high=[1])

    Action Space:
        Type: Discrete(8)

    Starting State:
        Zeros on all state variables.

    Episode Termination:
        Termination if current limits are violated. The terminal reward -10 is used as reward.
        (Have a look at the reward functions.)
    """

    def __init__(self, tau=1e-5, converter='Disc-Multi', subconverters=('Disc-4QC', 'Disc-1QC'), **kwargs):
        # Docstring in Base Class
        super().__init__(tau=tau, converter=converter, subconverters=subconverters, **kwargs)


class ContDcExternallyExcitedMotorEnvironment(DcExternallyExcitedMotorEnvironment):
    """
    Description:
        Environment to simulate a continuously controlled externally excited DC Motor

    Key:
        `DcExtExCont-v1`

    Default Modules:

        Physical System:
            SCMLSystem/DcMotorSystem with:
                | IdealVoltageSupply
                | ContDoubleConverter(subconverters=('Cont-4QC', 'Cont-1QC'))
                | DcExternallyExcitedMotor
                | PolynomialStaticLoad
                | GaussianWhiteNoiseGenerator
                | EulerSolver.
                | tau=1e-4

        Reference Generator:
            WienerProcessReferenceGenerator
                Reference Quantity. 'omega'

        Reward Function:
            WeightedSumOfErrors(reward_weights= {'omega': 1 })

        Visualization:
            ElectricMotorVisualization (Dummy for no Visualization)

    State Names:
        ``['omega' , 'torque', 'i_a', 'i_e', 'u_a', 'u_e', 'u_sup']``

    Observation Space:
        Type: Tuple(State_Space, Reference_Space)

    State Space:
        Box(low=[-1, -1, -1, -1, -1, -1, 0], high=[1, 1, 1, 1, 1, 1, 1])

    Reference Space:
        Box(low=[-1], high=[1])

    Action Space:
        Type: Box(low=[-1,0], high=[1,1])

    Starting State:
        Zeros on all state variables.

    Episode Termination:
        Termination if current limits are violated. The terminal reward -10 is used as reward.
        (Have a look at the reward functions.)

    """

    def __init__(self, tau=1e-4, converter='Cont-Multi', subconverters=('Cont-4QC', 'Cont-1QC'), **kwargs):
        # Docstring in Base Class
        super().__init__(tau=tau, converter=converter, subconverters=subconverters, **kwargs)
