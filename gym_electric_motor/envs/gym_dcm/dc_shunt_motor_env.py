from ...core import ElectricMotorEnvironment
from ...physical_systems.physical_systems import DcMotorSystem
from ...reference_generators import WienerProcessReferenceGenerator
from ...reward_functions import WeightedSumOfErrors


class DcShuntMotorEnvironment(ElectricMotorEnvironment):

    def __init__(self, motor='DcShunt', reward_function=None, reference_generator=None, constraints=('i_a', 'i_e'),
                 **kwargs):
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


class DiscDcShuntMotorEnvironment(DcShuntMotorEnvironment):
    """
    Description:
        Environment to simulate a discretely controlled DC Shunt Motor.

    Key:
        `DcShuntDisc-v1`

    Default Modules:

        Physical System:
            SCMLSystem/DcMotorSystem with:
                | IdealVoltageSupply
                | DiscreteTwoQuadrantConverter
                | DcShuntMotor
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
        ``['omega' , 'torque', 'i_a', 'i_e', 'u', 'u_sup']``

    Observation Space:
        Type: Tuple(State_Space, Reference_Space)

    State Space:
        Box(low=[0, -1, -1, -1, 0], high=[1, 1, 1, 1, 1])

    Reference Space:
        Box(low=[0], high=[1])

    Action Space:
        Type: Discrete(3)

    Starting State:
        Zeros on all state variables.

    Episode Termination:
        Termination if current limits are violated. The terminal reward -10 is used as reward.
        (Have a look at the reward functions.)
    """
    def __init__(self, tau=1e-5, converter='Disc-2QC', **kwargs):
        # Docstring in Base Class
        super().__init__(tau=tau, converter=converter, **kwargs)


class ContDcShuntMotorEnvironment(DcShuntMotorEnvironment):
    """
    Description:
        Environment to simulate a continuously controlled DC Shunt Motor.

    Key:
        `DcShuntCont-v1`

    Default Modules:

        Physical System:
            SCMLSystem/DcMotorSystem with:
                IdealVoltageSupply
                ContTwoQuadrantConverter
                DcShuntMotor
                PolynomialStaticLoad
                GaussianWhiteNoiseGenerator
                EulerSolver
                tau=1e-4

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
        Box(low=[0, -1, -1, -1, 0], high=[1, 1, 1, 1, 1])

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
    def __init__(self, tau=1e-4, converter='Cont-2QC', **kwargs):
        # Docstring in Base Class
        super().__init__(tau=tau, converter=converter, **kwargs)
