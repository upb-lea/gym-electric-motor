from gym_electric_motor.core import ElectricMotorEnvironment, ReferenceGenerator, RewardFunction, \
    ElectricMotorVisualization
from gym_electric_motor.physical_systems.physical_systems import DcMotorSystem
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.reference_generators.wiener_process_reference_generator import WienerProcessReferenceGenerator
from gym_electric_motor import physical_systems as ps
from gym_electric_motor.reward_functions import WeightedSumOfErrors
from gym_electric_motor.utils import initialize


class DiscSpeedControlDcPermanentlyExcitedMotorEnv(ElectricMotorEnvironment):
    """
        Description:
            Environment to simulate a discretely speed controlled permanently excited DC Motor

        Key:
            `Disc-SC-DcPermEx-v0`

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
                WeightedSumOfErrors: reward_weights 'omega' = 1

            Visualization:
                *MotorDashboard* - Plots: omega, action

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
    def __init__(self, supply=None, converter=None, motor=None, load=None, ode_solver=None, noise_generator=None,
                 reward_function=None, reference_generator=None, visualization=None,
                 constraints=('i',), calc_jacobian=True, tau=1e-5):
        """
        Args:
            motor(ElectricMotor): Electric Motor used in the PhysicalSystem
            reward_function(RewardFunction): Reward Function for the environment
            reference_generator(ReferenceGenerator): Reference Generator for the environment
        """

        physical_system = DcMotorSystem(
            supply=initialize(ps.VoltageSupply, supply, ps.IdealVoltageSupply, dict(u_nominal=420.0)),
            converter=initialize(ps.PowerElectronicConverter, converter, ps.DiscFourQuadrantConverter, dict()),
            motor=initialize(ps.ElectricMotor, motor, ps.DcPermanentlyExcitedMotor, dict()),
            load=initialize(ps.MechanicalLoad, load, ps.PolynomialStaticLoad, dict(
                load_parameter=dict(a=0.01, b=0.01, c=0.0)
            )),
            ode_solver=initialize(ps.OdeSolver, ode_solver, ps.EulerSolver, dict()),
            noise_generator=initialize(ps.NoiseGenerator, noise_generator, ps.NoiseGenerator, dict()),
            calc_jacobian=calc_jacobian,
            tau=tau
        )
        reference_generator = initialize(
            ReferenceGenerator, reference_generator, WienerProcessReferenceGenerator, dict(reference_state='omega')
        )
        reward_function = initialize(
            RewardFunction, reward_function, WeightedSumOfErrors, dict(reward_weights=dict(omega=1.0))
        )
        visualization = initialize(
            ElectricMotorVisualization, visualization, MotorDashboard, dict(state_plots=('omega',), action_plots=(0,)))
        super().__init__(
            physical_system=physical_system, reference_generator=reference_generator, reward_function=reward_function,
            constraints=constraints, visualization=visualization
        )


class ContSpeedControlDcPermanentlyExcitedMotorEnv(ElectricMotorEnvironment):
    """
        Description:
            Environment to simulate a discretely speed controlled permanently excited DC Motor

        Key:
            `Disc-SC-DcPermEx-v0`

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
                WeightedSumOfErrors: reward_weights 'omega' = 1

            Visualization:
                *MotorDashboard* - Plots: omega, action

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
    def __init__(self, supply=None, converter=None, motor=None, load=None, ode_solver=None, noise_generator=None,
                 reward_function=None, reference_generator=None, visualization=None,
                 constraints=('i',), calc_jacobian=True, tau=1e-4):
        """
        Args:
            motor(ElectricMotor): Electric Motor used in the PhysicalSystem
            reward_function(RewardFunction): Reward Function for the environment
            reference_generator(ReferenceGenerator): Reference Generator for the environment
        """

        physical_system = DcMotorSystem(
            supply=initialize(ps.VoltageSupply, supply, ps.IdealVoltageSupply, dict(u_nominal=420.0)),
            converter=initialize(ps.PowerElectronicConverter, converter, ps.ContFourQuadrantConverter, dict()),
            motor=initialize(ps.ElectricMotor, motor, ps.DcPermanentlyExcitedMotor, dict()),
            load=initialize(ps.MechanicalLoad, load, ps.PolynomialStaticLoad, dict(
                load_parameter=dict(a=0.01, b=0.01, c=0.0)
            )),
            ode_solver=initialize(ps.OdeSolver, ode_solver, ps.EulerSolver, dict()),
            noise_generator=initialize(ps.NoiseGenerator, noise_generator, ps.NoiseGenerator, dict()),
            calc_jacobian=calc_jacobian,
            tau=tau
        )
        reference_generator = initialize(
            ReferenceGenerator, reference_generator, WienerProcessReferenceGenerator, dict(reference_state='omega')
        )
        reward_function = initialize(
            RewardFunction, reward_function, WeightedSumOfErrors, dict(reward_weights=dict(omega=1.0))
        )
        visualization = initialize(
            ElectricMotorVisualization, visualization, MotorDashboard, dict(state_plots=('omega',), action_plots=(0,)))
        super().__init__(
            physical_system=physical_system, reference_generator=reference_generator, reward_function=reward_function,
            constraints=constraints, visualization=visualization
        )


class DiscTorqueControlDcPermanentlyExcitedMotorEnv(ElectricMotorEnvironment):
    """
        Description:
            Environment to simulate a discretely speed controlled permanently excited DC Motor

        Key:
            `Disc-SC-DcPermEx-v0`

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
                WeightedSumOfErrors: reward_weights 'omega' = 1

            Visualization:
                *MotorDashboard* - Plots: omega, action

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
    def __init__(self, supply=None, converter=None, motor=None, load=None, ode_solver=None, noise_generator=None,
                 reward_function=None, reference_generator=None, visualization=None,
                 constraints=('i',), calc_jacobian=True, tau=1e-5):
        """
        Args:
            motor(ElectricMotor): Electric Motor used in the PhysicalSystem
            reward_function(RewardFunction): Reward Function for the environment
            reference_generator(ReferenceGenerator): Reference Generator for the environment
        """

        physical_system = DcMotorSystem(
            supply=initialize(ps.VoltageSupply, supply, ps.IdealVoltageSupply, dict(u_nominal=420.0)),
            converter=initialize(ps.PowerElectronicConverter, converter, ps.DiscFourQuadrantConverter, dict()),
            motor=initialize(ps.ElectricMotor, motor, ps.DcPermanentlyExcitedMotor, dict()),
            load=initialize(ps.MechanicalLoad, load, ps.ConstantSpeedLoad, dict(omega_fixed=100.0)),
            ode_solver=initialize(ps.OdeSolver, ode_solver, ps.EulerSolver, dict()),
            noise_generator=initialize(ps.NoiseGenerator, noise_generator, ps.NoiseGenerator, dict()),
            calc_jacobian=calc_jacobian,
            tau=tau
        )
        reference_generator = initialize(
            ReferenceGenerator, reference_generator, WienerProcessReferenceGenerator, dict(reference_state='torque')
        )
        reward_function = initialize(
            RewardFunction, reward_function, WeightedSumOfErrors, dict(reward_weights=dict(torque=1.0))
        )
        visualization = initialize(
            ElectricMotorVisualization, visualization, MotorDashboard, dict(state_plots=('torque',), action_plots=(0,)))
        super().__init__(
            physical_system=physical_system, reference_generator=reference_generator, reward_function=reward_function,
            constraints=constraints, visualization=visualization
        )


class ContTorqueControlDcPermanentlyExcitedMotorEnv(ElectricMotorEnvironment):
    """
        Description:
            Environment to simulate a discretely speed controlled permanently excited DC Motor

        Key:
            `Disc-SC-DcPermEx-v0`

        Default Modules:

            Physical System:
                SCMLSystem/DcMotorSystem with:
                    | IdealVoltageSupply
                    | DiscFourQuadrantConverter
                    | DcPermanentlyExcitedMotor
                    | PolynomialStaticLoad
                    | GaussianWhiteNoiseGenerator
                    | EulerSolver
                    | tau=1e-4

            Reference Generator:
                WienerProcessReferenceGenerator
                    Reference Quantity. 'omega'

            Reward Function:
                WeightedSumOfErrors: reward_weights 'omega' = 1

            Visualization:
                *MotorDashboard* - Plots: omega, action

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
    def __init__(self, supply=None, converter=None, motor=None, load=None, ode_solver=None, noise_generator=None,
                 reward_function=None, reference_generator=None, visualization=None,
                 constraints=('i',), calc_jacobian=True, tau=1e-4):
        """
        Args:
            motor(ElectricMotor): Electric Motor used in the PhysicalSystem
            reward_function(RewardFunction): Reward Function for the environment
            reference_generator(ReferenceGenerator): Reference Generator for the environment
        """

        physical_system = DcMotorSystem(
            supply=initialize(ps.VoltageSupply, supply, ps.IdealVoltageSupply, dict(u_nominal=420.0)),
            converter=initialize(ps.PowerElectronicConverter, converter, ps.ContFourQuadrantConverter, dict()),
            motor=initialize(ps.ElectricMotor, motor, ps.DcPermanentlyExcitedMotor, dict()),
            load=initialize(ps.MechanicalLoad, load, ps.ConstantSpeedLoad, dict(omega_fixed=100.0)),
            ode_solver=initialize(ps.OdeSolver, ode_solver, ps.EulerSolver, dict()),
            noise_generator=initialize(ps.NoiseGenerator, noise_generator, ps.NoiseGenerator, dict()),
            calc_jacobian=calc_jacobian,
            tau=tau
        )
        reference_generator = initialize(
            ReferenceGenerator, reference_generator, WienerProcessReferenceGenerator, dict(reference_state='torque')
        )
        reward_function = initialize(
            RewardFunction, reward_function, WeightedSumOfErrors, dict(reward_weights=dict(torque=1.0))
        )
        visualization = initialize(
            ElectricMotorVisualization, visualization, MotorDashboard, dict(state_plots=('torque',), action_plots=(0,)))
        super().__init__(
            physical_system=physical_system, reference_generator=reference_generator, reward_function=reward_function,
            constraints=constraints, visualization=visualization
        )


class DiscCurrentControlDcPermanentlyExcitedMotorEnv(ElectricMotorEnvironment):
    """
        Description:
            Environment to simulate a discretely speed controlled permanently excited DC Motor

        Key:
            `Disc-SC-DcPermEx-v0`

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
                WeightedSumOfErrors: reward_weights 'omega' = 1

            Visualization:
                *MotorDashboard* - Plots: omega, action

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
    def __init__(self, supply=None, converter=None, motor=None, load=None, ode_solver=None, noise_generator=None,
                 reward_function=None, reference_generator=None, visualization=None,
                 constraints=('i',), calc_jacobian=True, tau=1e-5):
        """
        Args:
            motor(ElectricMotor): Electric Motor used in the PhysicalSystem
            reward_function(RewardFunction): Reward Function for the environment
            reference_generator(ReferenceGenerator): Reference Generator for the environment
        """

        physical_system = DcMotorSystem(
            supply=initialize(ps.VoltageSupply, supply, ps.IdealVoltageSupply, dict(u_nominal=420.0)),
            converter=initialize(ps.PowerElectronicConverter, converter, ps.DiscFourQuadrantConverter, dict()),
            motor=initialize(ps.ElectricMotor, motor, ps.DcPermanentlyExcitedMotor, dict()),
            load=initialize(ps.MechanicalLoad, load, ps.ConstantSpeedLoad, dict(omega_fixed=100.0)),
            ode_solver=initialize(ps.OdeSolver, ode_solver, ps.EulerSolver, dict()),
            noise_generator=initialize(ps.NoiseGenerator, noise_generator, ps.NoiseGenerator, dict()),
            calc_jacobian=calc_jacobian,
            tau=tau
        )
        reference_generator = initialize(
            ReferenceGenerator, reference_generator, WienerProcessReferenceGenerator, dict(reference_state='i')
        )
        reward_function = initialize(
            RewardFunction, reward_function, WeightedSumOfErrors, dict(reward_weights=dict(i=1.0))
        )
        visualization = initialize(
            ElectricMotorVisualization, visualization, MotorDashboard, dict(state_plots=('i',), action_plots=(0,)))
        super().__init__(
            physical_system=physical_system, reference_generator=reference_generator, reward_function=reward_function,
            constraints=constraints, visualization=visualization
        )


class ContCurrentControlDcPermanentlyExcitedMotorEnv(ElectricMotorEnvironment):
    """
        Description:
            Environment to simulate a discretely speed controlled permanently excited DC Motor

        Key:
            `Cont-CC-DcPermEx-v0`

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
                WeightedSumOfErrors: reward_weights 'omega' = 1

            Visualization:
                *MotorDashboard* - Plots: omega, action

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
    def __init__(self, supply=None, converter=None, motor=None, load=None, ode_solver=None, noise_generator=None,
                 reward_function=None, reference_generator=None, visualization=None,
                 constraints=('i',), calc_jacobian=True, tau=1e-4):
        """
        Args:
            motor(ElectricMotor): Electric Motor used in the PhysicalSystem
            reward_function(RewardFunction): Reward Function for the environment
            reference_generator(ReferenceGenerator): Reference Generator for the environment
        """

        physical_system = DcMotorSystem(
            supply=initialize(ps.VoltageSupply, supply, ps.IdealVoltageSupply, dict(u_nominal=420.0)),
            converter=initialize(ps.PowerElectronicConverter, converter, ps.ContFourQuadrantConverter, dict()),
            motor=initialize(ps.ElectricMotor, motor, ps.DcPermanentlyExcitedMotor, dict()),
            load=initialize(ps.MechanicalLoad, load, ps.ConstantSpeedLoad, dict(omega_fixed=100)),
            ode_solver=initialize(ps.OdeSolver, ode_solver, ps.EulerSolver, dict()),
            noise_generator=initialize(ps.NoiseGenerator, noise_generator, ps.NoiseGenerator, dict()),
            calc_jacobian=calc_jacobian,
            tau=tau
        )
        reference_generator = initialize(
            ReferenceGenerator, reference_generator, WienerProcessReferenceGenerator, dict(reference_state='i')
        )
        reward_function = initialize(
            RewardFunction, reward_function, WeightedSumOfErrors, dict(reward_weights=dict(i=1.0))
        )
        visualization = initialize(
            ElectricMotorVisualization, visualization, MotorDashboard, dict(state_plots=('i',), action_plots=(0,)))
        super().__init__(
            physical_system=physical_system, reference_generator=reference_generator, reward_function=reward_function,
            constraints=constraints, visualization=visualization
        )
