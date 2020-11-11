from gym_electric_motor.core import ElectricMotorEnvironment
from gym_electric_motor.physical_systems.physical_systems import DcMotorSystem
from gym_electric_motor.reference_generators.wiener_process_reference_generator import WienerProcessReferenceGenerator
from gym_electric_motor.reward_functions import WeightedSumOfErrors


class DcPermanentlyExcitedMotorEnvironment(ElectricMotorEnvironment):

    def __init__(self, motor='DcPermEx', reward_function=None, reference_generator=None, constraints=('i',), **kwargs):
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
            physical_system, reference_generator=reference_generator, reward_function=reward_function,
            constraints=constraints, **kwargs
        )


class DiscDcPermanentlyExcitedMotorEnvironment(DcPermanentlyExcitedMotorEnvironment):
    """
    Description:
        Environment to simulate a discretely controlled permanently excited DC Motor

    Key:
        `DcPermExDisc-v1`

    Default Modules:

        Physical System:
            SCMLSystem/DcMotorSystem with:
                | IdealVoltageSupply
                | DiscFourQuadrantConverter
                | DcPermanentlyExcitedMotor
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
        Box(low=[-1, -1, -1, -1, 0], high=[1, 1, 1, 1, 1])

    Reference Space:
        Box(low=[-1], high=[1])

    Action Space:
        Type: Box(low=[-1], high=[1])

    Starting State:
        Zeros on all state variables.

    Episode Termination:
        Termination if current limits are violated. The terminal reward -10 is used as reward.
        (Have a look at the reward functions.)
    """
    def __init__(self, tau=1e-5, converter='Disc-4QC', **kwargs):
        # Docstring in Base Class
        super().__init__(tau=tau, converter=converter, **kwargs)


class ContDcPermanentlyExcitedMotorEnvironment(DcPermanentlyExcitedMotorEnvironment):
    """
    Description:
        Environment to simulate a continuously controlled permanently excited DC Motor

    Key:
        `DcPermExCont-v1`

    Default Modules:

        Physical System:
            SCMLSystem/DcMotorSystem with:
                | IdealVoltageSupply
                | ContFourQuadrantConverter
                | DcPermanentlyExcitedMotor
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
        ``['omega' , 'torque', 'i', 'u', 'u_sup']``

    Observation Space:
        Type: Tuple(State_Space, Reference_Space)

    State Space:
        Box(low=[-1, -1, -1, -1, 0], high=[1, 1, 1, 1, 1])

    Reference Space:
        Box(low=[-1], high=[1])

    Action Space:
        Type: Box(low=[-1], high=[1])

    Starting State:
        Zeros on all state variables.

    Episode Termination:
        Termination if current limits are violated. The terminal reward -10 is used as reward.
        (Have a look at the reward functions.)

    """
    def __init__(self, tau=1e-4, converter='Cont-4QC', **kwargs):
        # Docstring in Base Class
        super().__init__(tau=tau, converter=converter, **kwargs)
