from classic_controllers import Controller
from externally_referenced_state_plot import ExternallyReferencedStatePlot

import gym_electric_motor as gem
import gym_electric_motor.physical_systems as ps
from gym_electric_motor.core import Signal
from gym_electric_motor.reference_generators import WienerProcessReferenceGenerator
from gym_electric_motor.reward_functions import WeightedSumOfErrors
from gym_electric_motor.visualization import MotorDashboard


class Block:
    def connect():
        pass

    def step():
        pass


class MyIdealVoltageSupply(Block):
    u_sup: Signal

    supply = ps.IdealVoltageSupply(u_nominal=60.0)

    def connect(self):
        return self.u_sup

    def step(self):
        self.u_sup.value = self.supply.get_voltage(0, 0)[0]


class MyConverter(Block):
    u_sup: Signal
    u: Signal
    i: Signal

    converter = ps.ContFourQuadrantConverter()

    def connect(self):
        return self.u, self.i

    def step(self):
        self.u.value = self.u_sup.value
        self.i.value = self.converter.i_in(0, self.u.value)


class MyMotor(Block):
    u: Signal
    omega: Signal
    torque: Signal
    i: Signal

    motor = ps.DcPermanentlyExcitedMotor()

    def __init__(self):
        self.motor._update_model()
        self.motor._update_limits()

    def connect(self):
        return self.omega, self.torque, self.i

    def step(self):
        self.motor._update_model()
        self.omega.value = self.motor.electrical_ode([0, 0, 0], [self.u.value], 0)[0]
        self.torque.value = self.motor.torque([self.i.value])
        self.i.value = self.motor.i_in([0, 0, 0, 0, 0, 0])


class MyLoad(Block):
    omega: Signal
    torque: Signal

    load = ps.ConstantSpeedLoad(omega_fixed=100.0)

    def connect(self):
        return self.torque

    def step(self):
        self.torque.value = self.load.torque(0, self.omega.value)


class MyEnv(gem.SimulationEnvironment):
    supply = ps.IdealVoltageSupply(u_nominal=60.0)
    converter = ps.ContFourQuadrantConverter()
    motor = ps.DcPermanentlyExcitedMotor()
    load = ps.ConstantSpeedLoad(omega_fixed=100.0)
    ode_solver = ps.ScipyOdeSolver()
    reference_generator = WienerProcessReferenceGenerator(
        reference_state="torque", sigma_range=(1e-2, 1e-1)
    )
    reward_function = WeightedSumOfErrors(reward_weights=dict(torque=1.0))

    tau = 1e-4
    calc_jacobian = True

    def __init__(self) -> None:
        self.motor.inp["u"] = self.supply.out["u"]
        self

    def step(self, action):
        state = self._physical_system.simulate(action)


if __name__ == "__main__":
    """
    motor type:     'PermExDc'  Permanently Excited DC Motor
                    'ExtExDc'   Externally Excited MC Motor
                    'SeriesDc'  DC Series Motor
                    'ShuntDc'   DC Shunt Motor
                    
    control type:   'SC'         Speed Control
                    'TC'         Torque Control
                    'CC'         Current Control
                    
    action_type:    'Cont'      Continuous Action Space
                    'Finite'    Discrete Action Space
    """

    motor_type = "PermExDc"
    control_type = "TC"
    action_type = "Cont"

    motor = action_type + "-" + control_type + "-" + motor_type + "-v0"

    if motor_type in ["PermExDc", "SeriesDc"]:
        states = ["omega", "torque", "i", "u"]
    elif motor_type == "ShuntDc":
        states = ["omega", "torque", "i_a", "i_e", "u"]
    elif motor_type == "ExtExDc":
        states = ["omega", "torque", "i_a", "i_e", "u_a", "u_e"]
    else:
        raise KeyError(motor_type + " is not available")

    # definition of the plotted variables
    external_ref_plots = [ExternallyReferencedStatePlot(state) for state in states]

    # initialize the gym-electric-motor environment
    env = gem.make(
        motor,
        visualization=MotorDashboard(additional_plots=external_ref_plots),
        render_mode="figure_once",
    )
    """
        initialize the controller

        Args:
            environment                     gym-electric-motor environment
            external_ref_plots (optional)   plots of the environment, to plot all reference values
            stages (optional)               structure of the controller
            automated_gain (optional)       if True (default), the controller will be tuned automatically
            a (optional)                    tuning parameter of the symmetrical optimum (default: 4)
    
    """
    visualization = MotorDashboard(additional_plots=external_ref_plots)
    controller = Controller.make(env, external_ref_plots=external_ref_plots)

    (state, reference), _ = env.reset(seed=None)
    ws = env.workspace
    # simulate the environment
    for i in range(100):
        action = controller.control(state, reference)
        (state, reference), reward, terminated, truncated, _ = env.step(action)

        # print(ws)
        if terminated:
            env.reset()
            controller.reset()

    env.close()
