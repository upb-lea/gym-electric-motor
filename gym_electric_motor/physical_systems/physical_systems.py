import numpy as np
from gym.spaces import Box

from ..core import PhysicalSystem
from ..physical_systems import electric_motors as em, mechanical_loads as ml, converters as cv, \
    voltage_supplies as vs, noise_generators as ng, solvers as sv
from ..utils import instantiate, set_state_array


class SCMLSystem(PhysicalSystem):
    """
    The SCML(Supply-Converter-Motor-Load)-System is used for the simulation of a technical setting consisting of these
    components as well as a noise generator and a solver for the electrical ODE of the motor and mechanical ODE of the
    load.
    """
    OMEGA_IDX = 0
    TORQUE_IDX = 1
    CURRENTS_IDX = []
    VOLTAGES_IDX = []
    U_SUP_IDX = -1

    @property
    def limits(self):
        return self._limits

    @property
    def nominal_state(self):
        return self._nominal_state

    @property
    def supply(self):
        """
        The voltage supply instance in the physical system
        """
        return self._supply

    @property
    def converter(self):
        """
        The power electronic converter instance in the system
        """
        return self._converter

    @property
    def electrical_motor(self):
        """
        The electrical motor instance of the system
        """
        return self._electrical_motor

    @property
    def mechanical_load(self):
        """
        The mechanical load instance in the system
        """
        return self._mechanical_load

    def __init__(
        self, converter, motor, load=None, supply='IdealVoltageSupply', ode_solver='euler', solver_kwargs=None,
        noise_generator=None, tau=1e-4, **kwargs
    ):
        """
        Args:
            converter(PowerElectronicConverter): Converter for the physical system
            motor(ElectricMotor): Motor of the system
            load(MechanicalLoad): Mechanical Load of the System
            supply(VoltageSupply): Voltage Supply
            ode_solver(OdeSolver): Ode Solver to use in this setting
            solver_kwargs(dict): Special keyword arguments to be passed to the solver
            noise_generator(NoiseGenerator):  Noise generator
            tau(float): discrete time step of the system
            kwargs(dict): Further arguments to pass to the modules while instantiation
        """
        self._converter = instantiate(cv.PowerElectronicConverter, converter, tau=tau, **kwargs)
        self._electrical_motor = instantiate(em.ElectricMotor, motor, tau=tau, **kwargs)
        load = load or ml.PolynomialStaticLoad(tau=tau, **kwargs)
        self._mechanical_load = instantiate(ml.MechanicalLoad, load, tau=tau, **kwargs)
        if 'u_sup' in kwargs.keys():
            u_sup = kwargs['u_sup']
        else:
            u_sup = self._electrical_motor.limits['u']
        self._supply = instantiate(vs.VoltageSupply, supply, u_nominal=u_sup, tau=tau, **kwargs)
        noise_generator = noise_generator or ng.GaussianWhiteNoiseGenerator(tau=tau, **kwargs)
        self._noise_generator = instantiate(ng.NoiseGenerator, noise_generator, **kwargs)
        state_names = self._build_state_names()
        self._noise_generator.set_state_names(state_names)
        solver_kwargs = solver_kwargs or {}
        self._ode_solver = instantiate(sv.OdeSolver, ode_solver, **solver_kwargs)
        self._ode_solver.set_system_equation(self._system_equation)
        self._mechanical_load.set_j_rotor(self._electrical_motor.motor_parameter['j_rotor'])
        self._t = 0
        self._set_indices()
        state_space = self._build_state_space(state_names)
        super().__init__(self._converter.action_space, state_space, state_names, tau)
        self._limits = np.zeros_like(state_names, dtype=float)
        self._nominal_state = np.zeros_like(state_names, dtype=float)
        self._set_limits()
        self._set_nominal_state()
        self._noise_generator.set_signal_power_level(self._nominal_state)

    def _set_limits(self):
        """
        Method to set the physical limits from the modules.
        """
        for ind, state in enumerate(self._state_names):
            motor_lim = self._electrical_motor.limits.get(state, np.inf)
            mechanical_lim = self._mechanical_load.limits.get(state, np.inf)
            self._limits[ind] = min(motor_lim, mechanical_lim)
        self._limits[self._state_positions['u_sup']] = self.supply.u_nominal

    def _set_nominal_state(self):
        """
        Method to set the nominal values from the modules.
        """
        for ind, state in enumerate(self._state_names):
            motor_nom = self._electrical_motor.nominal_values.get(state, np.inf)
            mechanical_nom = self._mechanical_load.nominal_values.get(state, np.inf)
            self._nominal_state[ind] = min(motor_nom, mechanical_nom)
        self._nominal_state[self._state_positions['u_sup']] = self.supply.u_nominal

    def _build_state_space(self, state_names):
        """
        Method to build the normalized state space (i.e. the maximum and minimum possible values for each state variable
        normalized by the limits).

        Args:
            state_names(list(str)): list of the names of each state.
        """
        raise NotImplementedError

    def _build_state_names(self):
        """
        Setting of the state names in the physical system.
        """
        raise NotImplementedError

    def _set_indices(self):
        """
        Setting of indices to faster access the arrays during integration.
        """
        self._omega_ode_idx = self._mechanical_load.OMEGA_IDX
        self._load_ode_idx = list(range(len(self._mechanical_load.state_names)))
        self._ode_currents_idx = list(range(
            self._load_ode_idx[-1] + 1, self._load_ode_idx[-1] + 1 + len(self._electrical_motor.CURRENTS)
        ))
        self._motor_ode_idx = self._ode_currents_idx

    def simulate(self, action, *_, **__):
        # Docstring of superclass
        transformed_action = self._action_transformation(action)
        state = self._ode_solver.y
        i_in = self._backward_transform(self._electrical_motor.i_in(state[self._ode_currents_idx]), state)
        switching_times = self._converter.set_action(transformed_action, self._t)
        i_sup = self._converter.i_sup(i_in)
        u_sup = self._supply.get_voltage(self._t, i_sup)

        for t in switching_times[:-1]:
            u_in = self._converter.convert(i_in, self._ode_solver.t)
            u_in = [u * u_sup for u in u_in]
            u_transformed = self._forward_transform(u_in, state)
            self._ode_solver.set_f_params(u_transformed)
            state = self._ode_solver.integrate(t)
            i_in = self._backward_transform(self._electrical_motor.i_in(state[self._ode_currents_idx]), state)
        u_in = self._converter.convert(i_in, self._ode_solver.t)
        u_in = [u * u_sup for u in u_in]
        u_transformed = self._forward_transform(u_in, state)
        self._ode_solver.set_f_params(u_transformed)
        ode_state = self._ode_solver.integrate(self._t + self._tau)
        self._t = self._ode_solver.t
        self._k += 1
        torque = self._electrical_motor.torque(ode_state[self._motor_ode_idx])
        noise = self._noise_generator.noise()
        return (self._build_state(ode_state, torque, u_in, u_sup) + noise) / self._limits

    def _system_equation(self, t, state, u_in, **__):
        """
        Systems differential equation system.

        It is a concatenation of the motors electrical ode system and the mechanical ode system.

        Args:
            t(float): Current systems time
            state(ndarray(float)): Current systems ODE-State
            u_in(list(float)): Input voltages from the converter

        Returns:
            ndarray(float): The derivatives of the ODE-State. Based on this, the Ode Solver calculates the next state.
        """
        motor_derivative = self._electrical_motor.electrical_ode(
            state[self._motor_ode_idx], u_in, state[self._omega_ode_idx]
        )
        torque = self._electrical_motor.torque(state[self._motor_ode_idx])
        load_derivative = self._mechanical_load.mechanical_ode(t, state[self._load_ode_idx], torque)
        return np.concatenate((load_derivative, motor_derivative))

    def reset(self, *_):
        """
        Reset all the systems modules to an initial state.

        Returns:
             The new state of the system.
        """
        motor_state = self._electrical_motor.reset()
        load_state = self._mechanical_load.reset()
        state = np.concatenate((load_state, motor_state))
        u_sup = self.supply.reset()
        u_in = self.converter.reset()
        u_in = [u * u_sup for u in u_in]
        torque = self.electrical_motor.torque(motor_state)
        noise = self._noise_generator.reset()
        self._t = 0
        self._k = 0
        self._ode_solver.set_initial_value(state, self._t)
        return (self._build_state(state, torque, u_in, u_sup) + noise) / self._limits

    def _forward_transform(self, quantities, motor_state):
        """
        Transformation to transform from the physical systems state to the ode-state
        """
        return quantities

    def _backward_transform(self, quantities, motor_state):
        """
        Transformation to transform from the ode-state to the systems-state
        """
        return quantities

    def _build_state(self, motor_state, torque, u_in, u_sup):
        """
        Based on the ode-state and the further input quantities the new systems state is built.
        """
        raise NotImplementedError

    @staticmethod
    def _action_transformation(action):
        """
        Placeholder for the option to use different representations for the synchronous motor in the future.
        """
        return action


class DcMotorSystem(SCMLSystem):
    """
    SCML-System that can be used for all DC Motors.
    """
    def _set_indices(self):
        # Docstring of superclass
        super()._set_indices()
        self.OMEGA_IDX = self.mechanical_load.OMEGA_IDX
        self.TORQUE_IDX = len(self.mechanical_load.state_names)
        currents_lower = self.TORQUE_IDX + 1
        currents_upper = currents_lower + len(self._electrical_motor.CURRENTS)
        self.CURRENTS_IDX = list(range(currents_lower, currents_upper))
        voltages_lower = currents_upper
        voltages_upper = voltages_lower + len(self._electrical_motor.VOLTAGES)
        self.VOLTAGES_IDX = list(range(voltages_lower, voltages_upper))
        self.U_SUP_IDX = voltages_upper

    def _build_state_names(self):
        # Docstring of superclass
        return (
            self._mechanical_load.state_names
            + ['torque']
            + self._electrical_motor.CURRENTS
            + self._electrical_motor.VOLTAGES
            + ['u_sup']
        )

    def _build_state_space(self, state_names):
        # Docstring of superclass
        low, high = self._electrical_motor.get_state_space(self._converter.currents, self._converter.voltages)
        low_mechanical, high_mechanical = self._mechanical_load.get_state_space((low['omega'], high['omega']))
        low.update(low_mechanical)
        high.update(high_mechanical)
        high['u_sup'] = self._supply.supply_range[1] / self._supply.u_nominal
        if self._supply.supply_range[0] != self._supply.supply_range[1]:
            low['u_sup'] = self._supply.supply_range[0] / self._supply.u_nominal
        else:
            low['u_sup'] = 0
        low = set_state_array(low, state_names)
        high = set_state_array(high, state_names)
        return Box(low, high)

    def _build_state(self, motor_state, torque, u_in, u_sup):
        # Docstring of superclass
        state = np.zeros_like(self.state_names, dtype=float)
        state[:len(self._mechanical_load.state_names)] = motor_state[:len(self._mechanical_load.state_names)]
        state[self.TORQUE_IDX] = torque
        state[self.CURRENTS_IDX] = motor_state[self._electrical_motor.CURRENTS_IDX]
        state[self.VOLTAGES_IDX] = u_in
        state[self.U_SUP_IDX] = u_sup
        return state


class SynchronousMotorSystem(SCMLSystem):
    """
    SCML-System that can be used with all Synchronous Motors
    """
    def _build_state_space(self, state_names):
        # Docstring of superclass
        low = -1 * np.ones_like(state_names, dtype=float)
        low[self.U_SUP_IDX] = 0.0
        high = np.ones_like(state_names, dtype=float)
        return Box(low, high)

    def _build_state_names(self):
        # Docstring of superclass
        return (
            self._mechanical_load.state_names
            + ['torque']
            + ['i_a'] + ['i_b'] + ['i_c']
            + ['u_a'] + ['u_b'] + ['u_c']
            + ['epsilon']
            + ['u_sup']
        )

    def _set_indices(self):
        # Docstring of superclass
        super()._set_indices()
        self._motor_ode_idx += [self._motor_ode_idx[-1] + 1]
        self._ode_currents_idx = self._motor_ode_idx[:-1]
        self.OMEGA_IDX = self.mechanical_load.OMEGA_IDX
        self.TORQUE_IDX = len(self.mechanical_load.state_names)
        currents_lower = self.TORQUE_IDX + 1
        currents_upper = currents_lower + 3
        self.CURRENTS_IDX = list(range(currents_lower, currents_upper))
        voltages_lower = currents_upper
        voltages_upper = voltages_lower + 3
        self.VOLTAGES_IDX = list(range(voltages_lower, voltages_upper))
        self.EPSILON_IDX = voltages_upper
        self.U_SUP_IDX = self.EPSILON_IDX + 1
        self._ode_epsilon_idx = self._motor_ode_idx[-1]

    def _forward_transform(self, quantities, motor_state):
        # Docstring of superclass
        motor_quantity = self._electrical_motor.q_inv(
            self._electrical_motor.t_23(quantities), motor_state[self._ode_epsilon_idx]
        )
        return motor_quantity[::-1]

    def _backward_transform(self, quantities, motor_state):
        # Docstring of superclass
        return list(self._electrical_motor.t_32(
            self._electrical_motor.q(quantities[::-1], motor_state[self._ode_epsilon_idx])
        ))

    def _build_state(self, motor_state, torque, u_in, u_sup):
        # Docstring of superclass
        mechanical_state = motor_state[self._load_ode_idx]
        currents = list(
            self._backward_transform(motor_state[self._ode_currents_idx], motor_state)
        )
        epsilon = motor_state[self._ode_epsilon_idx] % (2 * np.pi)
        if epsilon > np.pi:
            epsilon -= 2 * np.pi

        return np.array(
            list(mechanical_state)
            + [torque]
            + currents
            + u_in
            + [epsilon]
            + [u_sup]
        )
