from gym_electric_motor import physical_systems as ps
from gym_electric_motor.constraints import LimitConstraint, SquaredConstraint
from gym_electric_motor.core import (
    ElectricMotorEnvironment,
    ElectricMotorVisualization,
    ReferenceGenerator,
    RewardFunction,
)
from gym_electric_motor.physical_systems.physical_systems import (
    ExternallyExcitedSynchronousMotorSystem,
    SynchronousMotorSystem,
)
from gym_electric_motor.reference_generators import WienerProcessReferenceGenerator
from gym_electric_motor.reward_functions import WeightedSumOfErrors
from gym_electric_motor.utils import initialize
from gym_electric_motor.visualization import MotorDashboard


class ContSpeedControlExternallyExcitedSynchronousMotorEnv(ElectricMotorEnvironment):
    """
    Description:
        Environment to simulate a continuous control set speed controlled externally excited synchronous motor.

    Key:
        ``'Cont-SC-EESM-v0'``

    Default Components:
        - Supply: :py:class:`.IdealVoltageSupply`
        - Converter: :py:class:`.ContMultiConverter`(:py:class:`.ContB6BridgeConverter`, :py:class:`.ContFourQuadrantConverter`)
        - Motor: :py:class:`.ExternallyExcitedSynchronousMotor`
        - Load: :py:class:`.PolynomialStaticLoad`
        - Ode-Solver: :py:class:`.ScipyOdeSolver`

        - Reference Generator: :py:class:`.WienerProcessReferenceGenerator` *Reference Quantity:* ``'omega'``

        - Reward Function: :py:class:`.WeightedSumOfErrors` reward_weights: ``'omega' = 1.0``

        - Visualization: :py:class:`.MotorDashboard` current and action plots

        - Constraints: :py:class:`.SquaredConstraint` on the currents  ``'i_sd', 'i_sq'``, :py:class:`.LimitConstraint` on the current  ``'i_e''``

    State Variables:
        ``['omega' , 'torque', 'i_sd', 'i_sq', 'i_a', 'i_b', 'i_c', 'i_e', 'u_sd', 'u_sq', 'u_a', 'u_b', 'u_c', 'u_e', 'u_sup']``

    Referenced Variables:
        ``['omega']``

    Control Cycle Time:
        tau = 1e-4 seconds

    Observation Space:
        Type: Tuple(State_Space, Reference_Space)

    State Space:
        Box(low=15 * [-1], high=15 * [1])

    Reference Space:
        Box(low=[-1], high=[1])

    Action Space:
        Box(low=[-1, -1, -1, -1], high=[1, 1, 1, 1])

    Initial State:
        Zeros on all state variables.

    Example:
        >>> import gym_electric_motor as gem
        >>> from gym_electric_motor.reference_generators import LaplaceProcessReferenceGenerator
        >>>
        >>> # Select a different ode_solver with default parameters by passing a keystring
        >>> my_overridden_solver = 'scipy.solve_ivp'
        >>>
        >>> # Update the default arguments to the voltage supply by passing a parameter dict
        >>> my_changed_voltage_supply_args = {'u_nominal': 400.0}
        >>>
        >>> # Replace the reference generator by passing a new instance
        >>> my_new_ref_gen_instance = LaplaceProcessReferenceGenerator(
        ...     reference_state='omega',
        ...     sigma_range=(1e-3, 1e-2)
        ... )
        >>> env = gem.make(
        ...     'Cont-SC-EESM-v0',
        ...     voltage_supply=my_changed_voltage_supply_args,
        ...     ode_solver=my_overridden_solver,
        ...     reference_generator=my_new_ref_gen_instance
        ... )
        >>> terminated = True
        >>> for _ in range(1000):
        >>>     if terminated:
        >>>         state, reference = env.reset()
        >>>     (state, reference), reward, terminated, truncated, _ = env.step(env.action_space.sample())
    """

    def __init__(
        self,
        supply=None,
        converter=None,
        motor=None,
        load=None,
        ode_solver=None,
        reward_function=None,
        reference_generator=None,
        visualization=None,
        state_filter=None,
        callbacks=(),
        constraints=(SquaredConstraint(("i_sq", "i_sd")), LimitConstraint(("i_e",))),
        calc_jacobian=True,
        tau=1e-4,
        physical_system_wrappers=(),
        **kwargs,
    ):
        """
        Args:
            supply(env-arg): Specification of the :py:class:`.VoltageSupply` for the environment
            converter(env-arg): Specification of the :py:class:`.PowerElectronicConverter` for the environment
            motor(env-arg): Specification of the :py:class:`.ElectricMotor` for the environment
            load(env-arg): Specification of the :py:class:`.MechanicalLoad` for the environment
            ode_solver(env-arg): Specification of the :py:class:`.OdeSolver` for the environment
            reward_function(env-arg): Specification of the :py:class:`.RewardFunction` for the environment
            reference_generator(env-arg): Specification of the :py:class:`.ReferenceGenerator` for the environment
            visualization(env-arg): Specification of the :py:class:`.ElectricMotorVisualization` for the environment
            constraints(iterable(str/Constraint)): All Constraints of the environment. \n
                - str: A LimitConstraints for states (episode terminates, if the quantity exceeds the limit)
                 can be directly specified by passing the state name here (e.g., 'i', 'omega') \n
                - instance of Constraint: More complex constraints (e.g., the SquaredConstraint can be initialized and
                 passed to the environment.
            calc_jacobian(bool): Flag, if the jacobian of the environment shall be taken into account during the
                simulation. This may lead to speed improvements. Default: True
            tau(float): Duration of one control step in seconds. Default: 1e-4.
            state_filter(list(str)): List of states that shall be returned to the agent. Default: None (no filter)
            callbacks(list(Callback)): Callbacks for user interaction. Default: ()
            physical_system_wrappers(list(PhysicalSystemWrapper)): List of Physical System Wrappers to modify the
            actions to and states from the physical system before they are used in the environment. Default: ()

        Note on the env-arg type:
            All parameters of type env-arg can be selected as one of the following types:

            **instance:** Pass an already instantiated object derived from the corresponding base class
            (e.g., ``reward_function=MyRewardFunction()``). This is directly used in the environment.

            **dict:** Pass a dict to update the default parameters of the default type.
            (e.g., ``visualization=dict(state_plots=('omega', 'u'))``)

            **str:** Pass a string out of the registered classes to select a different class for the component.
            This class is then initialized with its default parameters.
            The available strings can be looked up in the documentation. (e.g., ``converter='Finite-2QC'``)
        """
        default_subconverters = (
            ps.ContB6BridgeConverter(),
            ps.ContFourQuadrantConverter(),
        )
        physical_system = ExternallyExcitedSynchronousMotorSystem(
            supply=initialize(ps.VoltageSupply, supply, ps.IdealVoltageSupply, dict(u_nominal=200.0)),
            converter=initialize(
                ps.PowerElectronicConverter,
                converter,
                ps.ContMultiConverter,
                dict(subconverters=default_subconverters),
            ),
            motor=initialize(ps.ElectricMotor, motor, ps.ExternallyExcitedSynchronousMotor, dict()),
            load=initialize(
                ps.MechanicalLoad,
                load,
                ps.PolynomialStaticLoad,
                dict(load_parameter=dict(a=0.01, b=0.01, c=0.0)),
            ),
            ode_solver=initialize(ps.OdeSolver, ode_solver, ps.ScipyOdeSolver, dict()),
            calc_jacobian=calc_jacobian,
            tau=tau,
        )
        reference_generator = initialize(
            ReferenceGenerator,
            reference_generator,
            WienerProcessReferenceGenerator,
            dict(reference_state="omega"),
        )
        reward_function = initialize(
            RewardFunction,
            reward_function,
            WeightedSumOfErrors,
            dict(reward_weights=dict(omega=1.0)),
        )
        visualization = initialize(
            ElectricMotorVisualization,
            visualization,
            MotorDashboard,
            dict(state_plots=("omega",), action_plots="all"),
        )
        super().__init__(
            physical_system=physical_system,
            reference_generator=reference_generator,
            reward_function=reward_function,
            constraints=constraints,
            visualization=visualization,
            state_filter=state_filter,
            callbacks=callbacks,
            physical_system_wrappers=physical_system_wrappers,
            **kwargs,
        )
