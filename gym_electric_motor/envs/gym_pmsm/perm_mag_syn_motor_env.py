from ...core import ElectricMotorEnvironment
from ...physical_systems.physical_systems import SynchronousMotorSystem
from ...reference_generators import WienerProcessReferenceGenerator
from ...reward_functions import WeightedSumOfErrors


class PermanentMagnetSynchronousMotorEnvironment(ElectricMotorEnvironment):

    def __init__(self, motor='PMSM', reward_function=None, reference_generator=None, **kwargs):
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
        super().__init__(
            physical_system, reference_generator=reference_generator, reward_function=reward_function, **kwargs
        )


class DiscPermanentMagnetSynchronousMotorEnvironment(PermanentMagnetSynchronousMotorEnvironment):
    """
    Description:
        Environment to simulate a discretely controlled Permanent Magnet Synchronous Motor (PMSM).

    Key:
        `emotor-pmsm-disc-v1`

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
        ``['omega' , 'torque', 'i_a', 'i_b', 'i_c', 'u_a', 'u_b', 'u_c', 'epsilon', 'u_sup']``

    Observation Space:
        Type: Tuple(State_Space, Reference_Space)

    State Space:
        Box(low=[-1, -1, -1, -1, -1, -1, 0], high=[1, 1, 1, 1, 1, 1, 1])

    Reference Space:
        Box(low=[-1], high=[1])

    Action Space:
        Type: Discrete(8)

    Reward:
        .. math::
            reward = (\omega - \omega^*) / (2 * \omega_{lim})

    Starting State:
        Zeros on all state variables.

    Episode Termination:
        Current limits (i_a ,i_b, i_c) are observed and the reference generation is continuous.
        Therefore, an episode ends only, when current limits have been violated.

    Limit Violation Reward:
        .. math::
            limit~violation~reward = -1 / (1- \gamma ) = -10 (Default: \gamma = 0.9)

    u_sup and u_nominal must be the same
    """
    def __init__(self, tau=1e-5, converter='Disc-B6C', **kwargs):
        super().__init__(tau=tau, converter=converter, **kwargs)


class ContPermanentMagnetSynchronousMotorEnvironment(PermanentMagnetSynchronousMotorEnvironment):
    """
    Description:
        Environment to simulate a continuously controlled Permanent Magnet Synchronous Motor (PMSM).

    Key:
        `emotor-pmsm-cont-v1`

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
        ``['omega' , 'torque', 'i_a', 'i_b', 'i_c', 'u_a', 'u_b', 'u_c', 'epsilon', 'u_sup']``

    Observation Space:
        Type: Tuple(State_Space, Reference_Space)

    State Space:
        Box(low=[-1, -1, -1, -1, -1, -1, 0], high=[1, 1, 1, 1, 1, 1, 1])

    Reference Space:
        Box(low=[-1], high=[1])

    Action Space:
        Type: Box(low=[-1, -1, -1], high=[1, 1, 1])

    Reward:
        .. math::
            reward = (\omega - \omega^*) / (2 * \omega_{lim})

    Starting State:
        Zeros on all state variables.

    Episode Termination:
        Current limits (i_a ,i_b, i_c) are observed and the reference generation is continuous.
        Therefore, an episode ends only, when current limits have been violated.

    Limit Violation Reward:
        .. math::
            limit~violation~reward = -1 / (1- \gamma ) = -10 (Default: \gamma = 0.9)

    u_sup and u_nominal must be the same
    """
    def __init__(self, tau=1e-4, converter='Cont-B6C', **kwargs):
        super().__init__(tau=tau, converter=converter, **kwargs)
