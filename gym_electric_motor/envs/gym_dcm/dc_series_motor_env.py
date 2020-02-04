from ...core import ElectricMotorEnvironment
from ...physical_systems.physical_systems import DcMotorSystem
from ...reference_generators.wiener_process_reference_generator import WienerProcessReferenceGenerator
from ...reward_functions import WeightedSumOfErrors


class DcSeriesMotorEnvironment(ElectricMotorEnvironment):

    def __init__(self, motor='DcSeries', reward_function=None, reference_generator=None, **kwargs):
        """
        Args:
            motor(ElectricMotor): Electric Motor used in the PhysicalSystem
            reward_function(RewardFunction): Reward Function for the environment
            reference_generator(ReferenceGenerator): Reference Generator for the environment
            kwargs(dict): Further kwargs to pass to the superclass and the submodules
        """
        physical_system = DcMotorSystem(motor=motor, **kwargs)
        reference_generator = reference_generator or WienerProcessReferenceGenerator(**kwargs)
        reward_function = reward_function or WeightedSumOfErrors(**kwargs)
        super().__init__(
            physical_system, reference_generator=reference_generator, reward_function=reward_function, **kwargs
        )


class DiscDcSeriesMotorEnvironment(DcSeriesMotorEnvironment):
    """
    Description:
        Environment to simulate a discretely controlled DC Series Motor.

    Key:
        `emotor-dc-series-disc-v1`

    Default Modules:

        Physical System:
            SCMLSystem/DcMotorSystem with:
                | IdealVoltageSupply
                | DiscreteOneQuadrantConverter
                | DcSeriesMotor
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
        ``['omega' , 'torque', 'i', 'u', 'u_sup']``

    Observation Space:
        Type: Tuple(State_Space, Reference_Space)

    State Space:
        Box(low=[0, 0, 0, 0, 0], high=[1, 1, 1, 1, 1])

    Reference Space:
        Box(low=[0], high=[1])

    Action Space:
        Type: Discrete(2)

    Reward:
        .. math::
            reward = (\omega - \omega^*) / (2 * \omega_{lim})

    Starting State:
        Zeros on all state variables.

    Episode Termination:
        Current limit (i) is observed and the reference generation is continuous.
        Therefore, an episode ends only, when current limits have been violated.

    Limit Violation Reward:
        .. math::
            limit~violation~reward = -1 / (1- \gamma ) = -10 (Default: \gamma = 0.9)
    """
    def __init__(self, tau=1e-5, converter='Disc-1QC', **kwargs):
        # Docstring in Base Class
        super().__init__(tau=tau, converter=converter, **kwargs)


class ContDcSeriesMotorEnvironment(DcSeriesMotorEnvironment):
    """
    Description:
        Environment to simulate a continuously controlled DC Series Motor.

    Key:
        `emotor-dc-series-cont-v1`

    Default Modules:

        Physical System:
            SCMLSystem/DcMotorSystem with:
                | IdealVoltageSupply
                | ContOneQuadrantConverter
                | DcSeriesMotor
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
        ```['omega' , 'torque', 'i', 'u', 'u_sup']```

    Observation Space:
        Type: Tuple(State_Space, Reference_Space)

    State Space:
        Box(low=[0, 0, 0, 0, 0], high=[1, 1, 1, 1, 1])

    Reference Space:
        Box(low=[0], high=[1])

    Action Space:
        Type: Box(low=[0], high=[1])

    Reward:
        .. math::
            reward = (\omega - \omega^*) / (2 * \omega_{lim})

    Starting State:
        Zeros on all state variables.

    Episode Termination:
        Current limit (i) is observed and the reference generation is continuous.
        Therefore, an episode ends only, when current limits have been violated.

    Limit Violation Reward:
        .. math::
            limit~violation~reward = -1 / (1- \gamma ) = -10 (Default: \gamma = 0.9)
    """
    def __init__(self, tau=1e-4, converter='Cont-1QC', **kwargs):
        # Docstring in Base Class
        super().__init__(tau=tau, converter=converter, **kwargs)
