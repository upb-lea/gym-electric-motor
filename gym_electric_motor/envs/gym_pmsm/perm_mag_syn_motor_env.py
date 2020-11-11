from ...core import ElectricMotorEnvironment
from ...physical_systems.physical_systems import SynchronousMotorSystem
from ...reference_generators import WienerProcessReferenceGenerator
from ...reward_functions import WeightedSumOfErrors
from ...constraints import SquaredConstraint


class PermanentMagnetSynchronousMotorEnvironment(ElectricMotorEnvironment):

    def __init__(self, motor='PMSM', reward_function=None, reference_generator=None, constraints=None, **kwargs):
        """
        Args:
            motor(ElectricMotor): Electric Motor used in the PhysicalSystem
            reward_function(RewardFunction): Reward Function for the environment
            reference_generator(ReferenceGenerator): Reference Generator for the environment
            kwargs(dict): Further kwargs tot pass to the superclass and the submodules
        """
        physical_system = SynchronousMotorSystem(motor=motor, **kwargs)
        reference_generator = reference_generator or WienerProcessReferenceGenerator(**kwargs)
        reward_function = reward_function or WeightedSumOfErrors(**kwargs)
        constraints_ = constraints if constraints is not None \
            else ('i_a', 'i_b', 'i_c', SquaredConstraint(('i_sd', 'i_sq')))
        super().__init__(
            physical_system, reference_generator=reference_generator, reward_function=reward_function,
            constraints=constraints_, **kwargs
        )


class DiscPermanentMagnetSynchronousMotorEnvironment(PermanentMagnetSynchronousMotorEnvironment):
    """
    Description:
        Environment to simulate a discretely controlled Permanent Magnet Synchronous Motor (PMSM).

    Key:
        `PMSMDisc-v1`

    Default Modules:

        Physical System:
            SCMLSystem/DcMotorSystem with:
                | IdealVoltageSupply
                | DiscB6BridgeConverter
                | PermanentMagnetSynchronousMotor
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
        ``['omega' , 'torque', 'i_sa', 'i_sb', 'i_sc', 'i_sd', 'i_sq',``
        ``'u_sa', 'u_sb', 'u_sc', 'u_sd', 'u_sq','epsilon', 'u_sup']``

    Observation Space:
        Type: Tuple(State_Space, Reference_Space)

    State Space:
        Box(low=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], high=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    Reference Space:
        Box(low=[-1], high=[1])

    Action Space:
        Type: Discrete(8)


    Starting State:
        Zeros on all state variables.

    Episode Termination:
        Termination if current limits are violated. The terminal reward -10 is used as reward.
        (Have a look at the reward functions.)

    u_sup and u_nominal must be the same
    """
    def __init__(self, tau=1e-5, converter='Disc-B6C', **kwargs):
        super().__init__(tau=tau, converter=converter, **kwargs)


class ContPermanentMagnetSynchronousMotorEnvironment(PermanentMagnetSynchronousMotorEnvironment):
    """
    Description:
        Environment to simulate a continuously controlled Permanent Magnet Synchronous Motor (PMSM).

    Key:
        `PMSMCont-v1`

    Default Modules:

        Physical System:
            SCMLSystem/DcMotorSystem with:
                | IdealVoltageSupply
                | ContinuousB6BridgeConverter
                | PermanentMagnetSynchronousMotor
                | PolynomialStaticLoad
                | GaussianWhiteNoiseGenerator
                | EulerSolver
                | tau=1e-4

        Reference Generator:
            WienerProcessReferenceGenerator
                Reference Quantity. 'omega'

        Reward Function:
            WeightedSumOfErrors(reward_weights= {'omega': 1 })

        Visualization:
            ElectricMotorVisualization (Dummy for no Visualization)

    State Variables:
        ``['omega' , 'torque', 'i_sa', 'i_sb', 'i_sc', 'i_sd',``
        ``'i_sq', 'u_sa', 'u_sb', 'u_sc', 'u_sd', 'u_sq','epsilon', 'u_sup']``

    Observation Space:
        Type: Tuple(State_Space, Reference_Space)

    State Space:
        Box(low=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], high=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    Reference Space:
        Box(low=[-1], high=[1])

    Action Space:
        Type: Box(low=[-1, -1, -1], high=[1, 1, 1])

    Starting State:
        Zeros on all state variables.

    Episode Termination:
        Termination if current limits are violated. The terminal reward -10 is used as reward.
        (Have a look at the reward functions.)

    u_sup and u_nominal must be the same
    """
    def __init__(self, tau=1e-4, converter='Cont-B6C', **kwargs):
        super().__init__(tau=tau, converter=converter, **kwargs)
