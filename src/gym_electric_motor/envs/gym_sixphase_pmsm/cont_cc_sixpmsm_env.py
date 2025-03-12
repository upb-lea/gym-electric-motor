
from gym_electric_motor import physical_systems as ps
from gym_electric_motor.constraints import SquaredConstraint
from gym_electric_motor.core import (
    ElectricMotorEnvironment,
    ElectricMotorVisualization,
    ReferenceGenerator,
    RewardFunction,
)
from gym_electric_motor.physical_systems.physical_systems import SixPhasePMSM
from gym_electric_motor.reference_generators import (
    MultipleReferenceGenerator,
    WienerProcessReferenceGenerator,
)
from gym_electric_motor.reward_functions import WeightedSumOfErrors
from gym_electric_motor.utils import initialize
from gym_electric_motor.visualization import MotorDashboard


class ContCurrentControlSixPhasePermanentMagnetSynchronousMotorEnv(ElectricMotorEnvironment):
   
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
        constraints=(SquaredConstraint(("i_sq", "i_sd", "i_sx", "i_sy")),),
        calc_jacobian=True,
        tau=1e-4,
        physical_system_wrappers=(),
        **kwargs,
    ):
 
        default_subgenerators = (
            WienerProcessReferenceGenerator(reference_state="i_sd"),
            WienerProcessReferenceGenerator(reference_state="i_sq"),
            WienerProcessReferenceGenerator(reference_state="i_sx"),
            WienerProcessReferenceGenerator(reference_state="i_sy")
        )

        physical_system = SixPhasePMSM(
            supply=initialize(ps.VoltageSupply, supply, ps.IdealVoltageSupply, dict(u_nominal=300.0)),
            converter=initialize(ps.PowerElectronicConverter, converter, ps.ContB6BridgeConverter, dict()),
            motor=initialize(ps.ElectricMotor, motor, ps.electric_motors.SixPhasePMSM, dict()),
            load=initialize(ps.MechanicalLoad, load, ps.PolynomialStaticLoad, dict(load_parameter=dict(a=0.01, b=0.01, c=0.0))),
            ode_solver=initialize(ps.OdeSolver, ode_solver, ps.ScipyOdeSolver, dict()),
            calc_jacobian=calc_jacobian,
            tau=tau,
        )
        reference_generator = initialize(
            ReferenceGenerator,
            reference_generator,
            MultipleReferenceGenerator,
            dict(sub_generators=default_subgenerators),
        )
        reward_function = initialize(
            RewardFunction,
            reward_function,
            WeightedSumOfErrors,
            dict(reward_weights=dict(i_sd=0.5, i_sq=0.5, i_sx=0.5, i_sy=0.5,)),
        )
        visualization = initialize(
            ElectricMotorVisualization,
            visualization,
            MotorDashboard,
            dict(state_plots=("i_sd", "i_sq"), action_plots="all"),
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