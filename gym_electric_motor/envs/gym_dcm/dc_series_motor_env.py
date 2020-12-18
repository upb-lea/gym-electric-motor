from ...core import ElectricMotorEnvironment
from ...physical_systems.physical_systems import DcMotorSystem
from ...reference_generators.wiener_process_reference_generator import WienerProcessReferenceGenerator
from ...reward_functions import WeightedSumOfErrors


class DcSeriesMotorEnvironment(ElectricMotorEnvironment):

    def __init__(self, motor='DcSeries', reward_function=None, reference_generator=None, constraints=('i',), **kwargs):
        """
        Args:
            motor(ElectricMotor): Electric Motor used in the PhysicalSystem
            reward_function(RewardFunction): Reward Function for the environment
            reference_generator(ReferenceGenerator): Reference Generator for the environment
            kwargs(dict): Further kwargs to pass to the superclass and the submodules
        """
        kwargs['load_parameter'] = kwargs.get('load_parameter', dict(a=0.01, b=0.05, c=0.1, j_load=0.1))
        physical_system = DcMotorSystem(motor=motor, **kwargs)
        reference_generator = reference_generator or WienerProcessReferenceGenerator(**kwargs)
        reward_function = reward_function or WeightedSumOfErrors(**kwargs)
        super().__init__(
            physical_system, reference_generator=reference_generator, reward_function=reward_function,
            constraints=constraints, **kwargs
        )


class DiscDcSeriesMotorEnvironment(DcSeriesMotorEnvironment):
    """
    Description:
        Environment to simulate a discretely controlled DC Series Motor.

    Key:
        `DcSeriesDisc-v1`

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

    Starting State:
        Zeros on all state variables.

    Episode Termination:
        Termination if current limits are violated. The terminal reward -10 is used as reward.
        (Have a look at the reward functions.)
    """
    def __init__(self, tau=1e-5, converter='Disc-1QC', **kwargs):
        # Docstring in Base Class
        super().__init__(tau=tau, converter=converter, **kwargs)


class ContDcSeriesMotorEnvironment(DcSeriesMotorEnvironment):
    """
    Description:
        Environment to simulate a continuously controlled DC Series Motor.

    Key:
        `DcSeriesCont-v1`

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

    Starting State:
        Zeros on all state variables.

    Episode Termination:
        Termination if current limits are violated. The terminal reward -10 is used as reward.
        (Have a look at the reward functions.)
    """
    def __init__(self, tau=1e-4, converter='Cont-1QC', **kwargs):
        # Docstring in Base Class
        super().__init__(tau=tau, converter=converter, **kwargs)
