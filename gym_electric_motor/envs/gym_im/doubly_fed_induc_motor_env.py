from ...core import ElectricMotorEnvironment
from ...physical_systems.physical_systems import DoublyFedInductionMotorSystem
from ...reference_generators import WienerProcessReferenceGenerator
from ...reward_functions import WeightedSumOfErrors


class DoublyFedInductionMotorEnvironment(ElectricMotorEnvironment):

    def __init__(self, motor='DFIM', reward_function=None, reference_generator=None, **kwargs):
        """
        Args:
            motor(ElectricMotor): Electric Motor used in the PhysicalSystem
            reward_function(RewardFunction): Reward Function for the environment
            reference_generator(ReferenceGenerator): Reference Generator for the environment
            kwargs(dict): Further kwargs tot pass to the superclass and the submodules
        """
        physical_system = DoublyFedInductionMotorSystem(motor=motor, **kwargs)
        reference_generator = reference_generator or WienerProcessReferenceGenerator(**kwargs)
        reward_function = reward_function or WeightedSumOfErrors(**kwargs)
        super().__init__(
            physical_system, reference_generator=reference_generator, reward_function=reward_function, **kwargs
        )


class DiscDoublyFedInductionMotorEnvironment(DoublyFedInductionMotorEnvironment):
    """
    Description:
        Environment to simulate a discretely controlled Doubly-Fed Induction Motor (SCIM).

    Key:
        `DFIMDisc-v1`

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
        ``['omega' , 'torque', 'i_sa', 'i_sb', 'i_sc', 'i_sd', 'i_sq', 'u_sa', 'u_sb', 'u_sc', 'u_sd', 'u_sq',``
        ``'i_ra', 'i_rb', 'i_rc', 'i_rd', 'i_rq', 'u_ra', 'u_rb', 'u_rc', 'u_rd', 'u_rq', 'epsilon', 'u_sup']``

    Observation Space:
        Type: Tuple(State_Space, Reference_Space)

    State Space:
        Box(low=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
        high=[1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1])

    Reference Space:
        Box(low=[-1], high=[1])

    Action Space:
        Type: MultiDiscrete([8, 8])

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
    def __init__(self, tau=1e-5, converter='Disc-Multi', subconverters=('Disc-B6C', 'Disc-B6C'), **kwargs):
        super().__init__(tau=tau, converter=converter, subconverters=subconverters, **kwargs)


class ContDoublyFedInductionMotorEnvironment(DoublyFedInductionMotorEnvironment):
    """
    Description:
        Environment to simulate a continuously controlled Doubly-Fed Induction Motor (SCIM).

    Key:
        `DFIMCont-v1`

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
        ``['omega' , 'torque', 'i_sa', 'i_sb', 'i_sc', 'i_sd', 'i_sq', 'u_sa', 'u_sb', 'u_sc', 'u_sd', 'u_sq',``
        ``'i_ra', 'i_rb', 'i_rc', 'i_rd', 'i_rq', 'u_ra', 'u_rb', 'u_rc', 'u_rd', 'u_rq', 'epsilon', 'u_sup']``

    Observation Space:
        Type: Tuple(State_Space, Reference_Space)

    State Space:
        Box(low=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
        high=[1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1])

    Reference Space:
        Box(low=[-1], high=[1])

    Action Space:
        Type: Box(low=[-1, -1, -1, -1, -1, -1], high=[1, 1, 1, 1, 1, 1])

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
    def __init__(self, tau=1e-4, converter='Cont-Multi', subconverters=('Cont-B6C', 'Cont-B6C'), **kwargs):
        super().__init__(tau=tau, converter=converter, subconverters=subconverters, **kwargs)
