import numpy as np
from gym.spaces import Box
import warnings

import gym_electric_motor as gem
from ..random_component import RandomComponent
from ..core import PhysicalSystem
from ..utils import set_state_array


class SCMLSystem(PhysicalSystem, RandomComponent):
    """
    The SCML(Supply-Converter-Motor-Load)-System is used for the simulation of
    a technical setting consisting of these components as well as a noise
    generator and a solver for the electrical ODE of the motor and mechanical
    ODE of the load.
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
        """The voltage supply instance in the physical system"""
        return self._supply

    @property
    def converter(self):
        """The power electronic converter instance in the system"""
        return self._converter

    @property
    def electrical_motor(self):
        """The electrical motor instance of the system"""
        return self._electrical_motor

    @property
    def mechanical_load(self):
        """The mechanical load instance in the system"""
        return self._mechanical_load

    def __init__(self, converter, motor, load, supply, ode_solver, noise_generator=None, tau=1e-4, calc_jacobian=None):
        """
        Args:
            converter(PowerElectronicConverter): Converter for the physical system
            motor(ElectricMotor): Motor of the system
            load(MechanicalLoad): Mechanical Load of the System
            supply(VoltageSupply): Voltage Supply
            ode_solver(OdeSolver): Ode Solver to use in this setting
            noise_generator(NoiseGenerator):  Noise generator
            tau(float): discrete time step of the system
            calc_jacobian(bool): If True, the jacobian matrices will be taken into account for the ode-solvers.
                Default: The jacobians are used, if available
        """
        RandomComponent.__init__(self)
        self._converter = converter
        self._electrical_motor = motor
        self._mechanical_load = load
        self._supply = supply
        self._noise_generator = noise_generator
        state_names = self._build_state_names()
        self._noise_generator.set_state_names(state_names)
        self._ode_solver = ode_solver
        if calc_jacobian is None:
            calc_jacobian = self._electrical_motor.HAS_JACOBIAN and self._mechanical_load.HAS_JACOBIAN
        if calc_jacobian and self._electrical_motor.HAS_JACOBIAN and self._mechanical_load.HAS_JACOBIAN:
            jac = self._system_jacobian
        else:
            jac = None
        if calc_jacobian and jac is None:
            warnings.warn('Jacobian Matrix is not provided for either the motor or the load Model')

        self._ode_solver.set_system_equation(self._system_equation, jac)
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
        self.system_state = np.zeros_like(state_names, dtype=float)
        self._system_eq_placeholder = None
        self._motor_deriv_size = None
        self._load_deriv_size = None
        self._components = [
            self._supply, self._converter, self._electrical_motor, self._mechanical_load, self._ode_solver,
            self._noise_generator
        ]

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
        self.OMEGA_IDX = self.mechanical_load.OMEGA_IDX
        self.TORQUE_IDX = len(self.mechanical_load.state_names)
        currents_lower = self.TORQUE_IDX + 1
        currents_upper = currents_lower + len(self._electrical_motor.CURRENTS)
        self.CURRENTS_IDX = list(range(currents_lower, currents_upper))
        voltages_lower = currents_upper
        voltages_upper = voltages_lower + len(self._electrical_motor.VOLTAGES)
        self.VOLTAGES_IDX = list(range(voltages_lower, voltages_upper))
        self.U_SUP_IDX = list(range(voltages_upper, voltages_upper + self._supply.voltage_len))

    def seed(self, seed=None):
        RandomComponent.seed(self, seed)
        sub_seeds = self.seed_sequence.spawn(len(self._components))
        for component, sub_seed in zip(self._components, sub_seeds):
            if isinstance(component, gem.RandomComponent):
                component.seed(sub_seed)

    def simulate(self, action, *_, **__):
        # Docstring of superclass
        ode_state = self._ode_solver.y
        i_in = self._electrical_motor.i_in(ode_state[self._ode_currents_idx])
        switching_times = self._converter.set_action(action, self._t)

        for t in switching_times[:-1]:
            i_sup = self._converter.i_sup(i_in)
            u_sup = self._supply.get_voltage(self._t, i_sup)
            u_in = self._converter.convert(i_in, self._ode_solver.t)
            u_in = [u * u_s for u in u_in for u_s in u_sup]
            self._ode_solver.set_f_params(u_in)
            ode_state = self._ode_solver.integrate(t)
            i_in = self._electrical_motor.i_in(ode_state[self._ode_currents_idx])
        
        i_sup = self._converter.i_sup(i_in)
        u_sup = self._supply.get_voltage(self._t, i_sup)
        u_in = self._converter.convert(i_in, self._ode_solver.t)
        u_in = [u * u_s for u in u_in for u_s in u_sup]
        self._ode_solver.set_f_params(u_in)
        ode_state = self._ode_solver.integrate(self._t + self._tau)
        self._t = self._ode_solver.t
        self._k += 1
        torque = self._electrical_motor.torque(ode_state[self._motor_ode_idx])
        noise = self._noise_generator.noise()

        n_mech_states = len(self.mechanical_load.state_names)
        motor_state = ode_state[n_mech_states:]
        self.system_state[:n_mech_states] = ode_state[:n_mech_states]
        self.system_state[self.TORQUE_IDX] = torque
        self.system_state[self.CURRENTS_IDX] = \
            motor_state[self._electrical_motor.CURRENTS_IDX]
        self.system_state[self.VOLTAGES_IDX] = u_in
        self.system_state[self.U_SUP_IDX] = u_sup
        return (self.system_state + noise) / self._limits

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
        if self._system_eq_placeholder is None:

            motor_derivative = self._electrical_motor.electrical_ode(
                state[self._motor_ode_idx], u_in, state[self._omega_ode_idx]
            )
            torque = self._electrical_motor.torque(state[self._motor_ode_idx])
            load_derivative = self._mechanical_load.mechanical_ode(t, state[
                self._load_ode_idx], torque)
            self._system_eq_placeholder = np.concatenate((load_derivative,
                                                          motor_derivative))
            self._motor_deriv_size = motor_derivative.size
            self._load_deriv_size = load_derivative.size
        else:
            self._system_eq_placeholder[:self._load_deriv_size] = \
                self._mechanical_load.mechanical_ode(
                    t, state[self._load_ode_idx],
                    self._electrical_motor.torque(state[self._motor_ode_idx])
                ).ravel()
            self._system_eq_placeholder[self._load_deriv_size:] = \
                self._electrical_motor.electrical_ode(
                    state[self._motor_ode_idx], u_in, state[self._omega_ode_idx]
                ).ravel()

        return self._system_eq_placeholder

    def _system_jacobian(self, t, state, u_in, **__):
        motor_jac, el_state_over_omega, torque_over_el_state = self._electrical_motor.electrical_jacobian(
            state[self._motor_ode_idx], u_in, state[self._omega_ode_idx]
        )
        torque = self._electrical_motor.torque(state[self._motor_ode_idx])
        load_jac, load_over_torque = self._mechanical_load.mechanical_jacobian(
            t, state[self._load_ode_idx], torque
        )
        system_jac = np.zeros((state.shape[0], state.shape[0]))
        system_jac[:load_jac.shape[0], :load_jac.shape[1]] = load_jac
        system_jac[-motor_jac.shape[0]:, -motor_jac.shape[1]:] = motor_jac
        system_jac[-motor_jac.shape[0]:, [self._omega_ode_idx]] = el_state_over_omega.reshape((-1, 1))
        system_jac[:load_jac.shape[0], load_jac.shape[1]:] = np.matmul(
            load_over_torque.reshape(-1, 1), torque_over_el_state.reshape(1, -1)
        )
        return system_jac

    def reset(self, *_):
        """
        Reset all the systems modules to an initial state.

        Returns:
             The new state of the system.
        """
        self.next_generator()
        motor_state = self._electrical_motor.reset(
            state_space=self.state_space,
            state_positions=self.state_positions)
        mechanical_state = self._mechanical_load.reset(
            state_space=self.state_space,
            state_positions=self.state_positions,
            nominal_state=self.nominal_state)
        ode_state = np.concatenate((mechanical_state, motor_state))
        u_sup = self.supply.reset()
        u_in = self.converter.reset()
        u_in = [u * u_s for u in u_in for u_s in u_sup]
        torque = self.electrical_motor.torque(motor_state)
        noise = self._noise_generator.reset()
        self._t = 0
        self._k = 0
        self._ode_solver.set_initial_value(ode_state, self._t)
        system_state = np.concatenate((
            ode_state[:len(self._mechanical_load.state_names)],
            [torque],
            motor_state[self._electrical_motor.CURRENTS_IDX],
            u_in,
            u_sup
        ))
        return (system_state + noise) / self._limits


class DcMotorSystem(SCMLSystem):
    """
    SCML-System that can be used for all DC Motors.
    """

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
        return Box(low, high, dtype=np.float64)


class ThreePhaseMotorSystem(SCMLSystem):
    """
    SCML-System that implements the basic transformations needed for three phase drives.
    """
    def abc_to_alphabeta_space(self, abc_quantities):
        """
        Transformation from abc to alphabeta space

        Args:
            abc_quantities: Three quantities in abc-space (e.g. (u_a, u_b, u_c) or (i_a, i_b, i_c))

        Returns:
            (quantity_alpha, quantity_beta): The quantities in the alphabeta-space
        """
        alphabeta_quantity = self._electrical_motor.t_23(abc_quantities)
        return alphabeta_quantity

    def alphabeta_to_abc_space(self, alphabeta_quantities):
        """
        Transformation from dq to abc space

        Args:
            alphabeta_quantities: Two quantities in alphabeta-space (e.g. (u_alpha, u_beta) or (i_alpha, i_beta))

        Returns:
            (quantity_a, quantity_b, quantity_c): The quantities in the abc-space
        """
        return self._electrical_motor.t_32(alphabeta_quantities)

    def abc_to_dq_space(self, abc_quantities, epsilon_el, normed_epsilon=False):
        """
        Transformation from abc to dq space

        Args:
            abc_quantities: Three quantities in abc-space (e.g. (u_a, u_b, u_c) or (i_a, i_b, i_c))
            epsilon_el: Electrical angle of the motor
            normed_epsilon(bool): True, if epsilon is normed to [-1,1] else in [-pi, pi] (default)

        Returns:
            (quantity_d, quantity_q): The quantities in the dq-space
        """
        if normed_epsilon:
            epsilon_el *= np.pi
        dq_quantity = self._electrical_motor.q_inv(self._electrical_motor.t_23(abc_quantities), epsilon_el)
        return dq_quantity

    def dq_to_abc_space(self, dq_quantities, epsilon_el, normed_epsilon=False):
        """
        Transformation from dq to abc space

        Args:
            dq_quantities: Three quantities in dq-space (e.g. (u_d, u_q) or (i_d, i_q))
            epsilon_el: Electrical angle of the motor
            normed_epsilon(bool): True, if epsilon is normed to [-1,1] else in [-pi, pi] (default)

        Returns:
            (quantity_a, quantity_b, quantity_c): The quantities in the abc-space
        """
        if normed_epsilon:
            epsilon_el *= np.pi
        return self._electrical_motor.t_32(self._electrical_motor.q(dq_quantities, epsilon_el))

    def alphabeta_to_dq_space(self, alphabeta_quantities, epsilon_el, normed_epsilon=False):
        """
        Transformation from alphabeta to dq space

        Args:
            alphabeta_quantities: Two quantities in alphabeta-space (e.g. (u_alpha, u_beta) or (i_alpha, i_beta))
            epsilon_el: Electrical angle of the motor
            normed_epsilon(bool): True, if epsilon is normed to [-1,1] else in [-pi, pi] (default)

        Returns:
            (quantity_d, quantity_q): The quantities in the dq-space
        """
        if normed_epsilon:
            epsilon_el *= np.pi
        dq_quantity = self._electrical_motor.q_inv(alphabeta_quantities, epsilon_el)
        return dq_quantity

    def dq_to_alphabeta_space(self, dq_quantities, epsilon_el, normed_epsilon=False):
        """
        Transformation from dq to alphabeta space

        Args:
            dq_quantities: Two quantities in dq-space (e.g. (u_d, u_q) or (i_d, i_q))
            epsilon_el: Electrical angle of the motor
            normed_epsilon(bool): True, if epsilon is normed to [-1,1] else in [-pi, pi] (default)

        Returns:
            (quantity_alpha, quantity_beta): The quantities in the alphabeta-space
        """
        if normed_epsilon:
            epsilon_el *= np.pi
        return self._electrical_motor.q(dq_quantities, epsilon_el)


class SynchronousMotorSystem(ThreePhaseMotorSystem):
    """
    SCML-System that can be used with all Synchronous Motors
    """

    def __init__(self, control_space='abc', **kwargs):
        """
        Args:
            control_space(str):('abc' or 'dq') Choose, if actions the actions space is in dq or abc space
            kwargs: Further arguments to pass tp SCMLSystem
        """
        super().__init__(**kwargs)
        self.control_space = control_space
        if control_space == 'dq':
            assert type(self._converter.action_space) == Box, \
                'dq-control space is only available for Continuous Controlled Converters'
            self._action_space = Box(-1, 1, shape=(2,), dtype=np.float64)

    def _build_state_space(self, state_names):
        # Docstring of superclass
        low = -1 * np.ones_like(state_names, dtype=float)
        low[self.U_SUP_IDX] = 0.0
        high = np.ones_like(state_names, dtype=float)
        return Box(low, high, dtype=np.float64)

    def _build_state_names(self):
        # Docstring of superclass
        return (
            self._mechanical_load.state_names +['torque',
                                                'i_a', 'i_b', 'i_c', 'i_sd', 'i_sq',
                                                'u_a', 'u_b', 'u_c', 'u_sd', 'u_sq',
                                                'epsilon', 'u_sup',
                                                ]
        )

    def _set_indices(self):
        # Docstring of superclass
        self._omega_ode_idx = self._mechanical_load.OMEGA_IDX
        self._load_ode_idx = list(range(len(self._mechanical_load.state_names)))
        self._ode_currents_idx = list(range(
            self._load_ode_idx[-1] + 1, self._load_ode_idx[-1] + 1 + len(self._electrical_motor.CURRENTS)
        ))
        self._motor_ode_idx = self._ode_currents_idx
        self._motor_ode_idx += [self._motor_ode_idx[-1] + 1]
        self._ode_currents_idx = self._motor_ode_idx[:-1]
        self.OMEGA_IDX = self.mechanical_load.OMEGA_IDX
        self.TORQUE_IDX = len(self.mechanical_load.state_names)
        currents_lower = self.TORQUE_IDX + 1
        currents_upper = currents_lower + 5
        self.CURRENTS_IDX = list(range(currents_lower, currents_upper))
        voltages_lower = currents_upper
        voltages_upper = voltages_lower + 5
        self.VOLTAGES_IDX = list(range(voltages_lower, voltages_upper))
        self.EPSILON_IDX = voltages_upper
        self.U_SUP_IDX = list(range(self.EPSILON_IDX + 1, self.EPSILON_IDX + 1 + self._supply.voltage_len))
        self._ode_epsilon_idx = self._motor_ode_idx[-1]

    def simulate(self, action, *_, **__):
        # Docstring of superclass
        ode_state = self._ode_solver.y
        eps = ode_state[self._ode_epsilon_idx]
        if self.control_space == 'dq':
            action = self.dq_to_abc_space(action, eps)
        i_in = self.dq_to_abc_space(self._electrical_motor.i_in(ode_state[self._ode_currents_idx]), eps)
        switching_times = self._converter.set_action(action, self._t)

        for t in switching_times[:-1]:
            i_sup = self._converter.i_sup(i_in)
            u_sup = self._supply.get_voltage(self._t, i_sup)
            u_in = self._converter.convert(i_in, self._ode_solver.t)
            u_in = [u * u_s for u in u_in for u_s in u_sup]
            u_dq = self.abc_to_dq_space(u_in, eps)
            self._ode_solver.set_f_params(u_dq)
            ode_state = self._ode_solver.integrate(t)
            eps = ode_state[self._ode_epsilon_idx]
            i_in = self.dq_to_abc_space(self._electrical_motor.i_in(ode_state[self._ode_currents_idx]), eps)

        i_sup = self._converter.i_sup(i_in)
        u_sup = self._supply.get_voltage(self._t, i_sup)
        u_in = self._converter.convert(i_in, self._ode_solver.t)
        u_in = [u * u_s for u in u_in for u_s in u_sup]
        u_dq = self.abc_to_dq_space(u_in, eps)
        self._ode_solver.set_f_params(u_dq)
        ode_state = self._ode_solver.integrate(self._t + self._tau)
        self._t = self._ode_solver.t
        self._k += 1
        torque = self._electrical_motor.torque(ode_state[self._motor_ode_idx])
        noise = self._noise_generator.noise()
        mechanical_state = ode_state[self._load_ode_idx]
        i_dq = ode_state[self._ode_currents_idx]
        i_abc = list(
            self.dq_to_abc_space(i_dq, eps)
        )
        eps = ode_state[self._ode_epsilon_idx] % (2 * np.pi)
        if eps > np.pi:
            eps -= 2 * np.pi

        system_state = np.concatenate((
            mechanical_state,
            [torque],
            i_abc, i_dq,
            u_in, u_dq,
            [eps],
            u_sup
        ))
        return (system_state + noise) / self._limits

    def reset(self, *_):
        # Docstring of superclass
        motor_state = self._electrical_motor.reset(
            state_space=self.state_space,
            state_positions=self.state_positions)
        mechanical_state = self._mechanical_load.reset(
            state_positions=self.state_positions,
            state_space=self.state_space,
            nominal_state=self.nominal_state)
        ode_state = np.concatenate((mechanical_state, motor_state))
        u_sup = self.supply.reset()
        eps = ode_state[self._ode_epsilon_idx]
        if eps > np.pi:
            eps -= 2 * np.pi
        u_abc = self.converter.reset()
        u_abc = [u * u_s for u in u_abc for u_s in u_sup]
        u_dq = self.abc_to_dq_space(u_abc, eps)
        i_dq = ode_state[self._ode_currents_idx]
        i_abc = self.dq_to_abc_space(i_dq, eps)
        torque = self.electrical_motor.torque(motor_state)
        noise = self._noise_generator.reset()
        self._t = 0
        self._k = 0
        self._ode_solver.set_initial_value(ode_state, self._t)
        system_state = np.concatenate((
            mechanical_state,
            [torque],
            i_abc, i_dq,
            u_abc, u_dq,
            [eps],
            u_sup,
        ))
        return (system_state + noise) / self._limits


class SquirrelCageInductionMotorSystem(ThreePhaseMotorSystem):
    """
    SCML-System for the Squirrel Cage Induction Motor
    """
    def __init__(self, control_space='abc', ode_solver='scipy.ode', **kwargs):
        """
        Args:
            control_space(str):('abc' or 'dq') Choose, if actions the actions space is in dq or abc space
            kwargs: Further arguments to pass tp SCMLSystem
        """
        super().__init__(ode_solver=ode_solver, **kwargs)
        self.control_space = control_space
        if control_space == 'dq':
            self._action_space = Box(-1, 1, shape=(2,), dtype=np.float64)

    def _build_state_space(self, state_names):
        # Docstring of superclass
        low = -1 * np.ones_like(state_names, dtype=float)
        low[self.U_SUP_IDX] = 0.0
        high = np.ones_like(state_names, dtype=float)
        return Box(low, high, dtype=np.float64)

    def _build_state_names(self):
        # Docstring of superclass
        return (
            self._mechanical_load.state_names + ['torque',
                                                 'i_sa', 'i_sb', 'i_sc', 'i_sd', 'i_sq',
                                                 'u_sa', 'u_sb', 'u_sc', 'u_sd', 'u_sq',
                                                 'epsilon', 'u_sup',
                                                 ]
        )

    def _set_indices(self):
        # Docstring of superclass
        super()._set_indices()
        self._motor_ode_idx += range(self._motor_ode_idx[-1] + 1, self._motor_ode_idx[-1] + 1 + len(self._electrical_motor.FLUXES))
        self._motor_ode_idx += [self._motor_ode_idx[-1] + 1]

        self._ode_currents_idx = self._motor_ode_idx[self._electrical_motor.I_SALPHA_IDX:self._electrical_motor.I_SBETA_IDX + 1]
        self._ode_flux_idx = self._motor_ode_idx[self._electrical_motor.PSI_RALPHA_IDX:self._electrical_motor.PSI_RBETA_IDX + 1]

        self.OMEGA_IDX = self.mechanical_load.OMEGA_IDX
        self.TORQUE_IDX = len(self.mechanical_load.state_names)
        currents_lower = self.TORQUE_IDX + 1
        currents_upper = currents_lower + 5
        self.CURRENTS_IDX = list(range(currents_lower, currents_upper))
        voltages_lower = currents_upper
        voltages_upper = voltages_lower + 5
        self.VOLTAGES_IDX = list(range(voltages_lower, voltages_upper))
        self.EPSILON_IDX = voltages_upper
        self.U_SUP_IDX = list(range(self.EPSILON_IDX + 1, self.EPSILON_IDX + 1 + self._supply.voltage_len))
        self._ode_epsilon_idx = self._motor_ode_idx[-1]

    def calculate_field_angle(self, state):
        psi_ralpha = state[self._ode_flux_idx[0]]
        psi_rbeta = state[self._ode_flux_idx[1]]
        eps_fs = np.arctan2(psi_rbeta, psi_ralpha)
        return eps_fs

    def simulate(self, action, *_, **__):
        # Docstring of superclass
        ode_state = self._ode_solver.y

        eps_fs = self.calculate_field_angle(ode_state)

        if self.control_space == 'dq':
            action = self.dq_to_abc_space(action, eps_fs)

        i_in = self.alphabeta_to_abc_space(self._electrical_motor.i_in(ode_state[self._ode_currents_idx]))
        switching_times = self._converter.set_action(action, self._t)

        for t in switching_times[:-1]:
            i_sup = self._converter.i_sup(i_in)
            u_sup = self._supply.get_voltage(self._t, i_sup)
            u_in = self._converter.convert(i_in, self._ode_solver.t)
            u_in = [u * u_s for u in u_in for u_s in u_sup]
            u_alphabeta = self.abc_to_alphabeta_space(u_in)
            self._ode_solver.set_f_params(u_alphabeta)
            ode_state = self._ode_solver.integrate(t)
            eps_fs = self.calculate_field_angle(ode_state)
            i_in = self.alphabeta_to_abc_space(self._electrical_motor.i_in(ode_state[self._ode_currents_idx]))

        i_sup = self._converter.i_sup(i_in)
        u_sup = self._supply.get_voltage(self._t, i_sup)
        u_in = self._converter.convert(i_in, self._ode_solver.t)
        u_in = [u * u_s for u in u_in for u_s in u_sup]
        u_dq = self.abc_to_dq_space(u_in, eps_fs)
        u_alphabeta = self.abc_to_alphabeta_space(u_in)
        self._ode_solver.set_f_params(u_alphabeta)
        ode_state = self._ode_solver.integrate(self._t + self._tau)
        self._t = self._ode_solver.t
        self._k += 1
        torque = self._electrical_motor.torque(ode_state[self._motor_ode_idx])
        noise = self._noise_generator.noise()
        mechanical_state = ode_state[self._load_ode_idx]
        i_dq = self.alphabeta_to_dq_space(ode_state[self._ode_currents_idx], eps_fs)
        i_abc = list(self.dq_to_abc_space(i_dq, eps_fs))

        eps = ode_state[self._ode_epsilon_idx] % (2 * np.pi)
        if eps > np.pi:
            eps -= 2 * np.pi

        system_state = np.concatenate((
            mechanical_state, [torque],
            i_abc, i_dq,
            u_in, u_dq,
            [eps],
            u_sup
        ))
        return (system_state + noise) / self._limits

    def reset(self, *_):
        # Docstring of superclass
        mechanical_state = self._mechanical_load.reset(
            state_positions=self.state_positions,
            state_space=self.state_space,
            nominal_state=self.nominal_state)
        motor_state = self._electrical_motor.reset(
            state_space=self.state_space,
            state_positions=self.state_positions,
            omega=mechanical_state)
        ode_state = np.concatenate((mechanical_state, motor_state))
        u_sup = self.supply.reset()

        eps = ode_state[self._ode_epsilon_idx]
        eps_fs = self.calculate_field_angle(ode_state)

        if eps > np.pi:
            eps -= 2 * np.pi

        u_abc = self.converter.reset()
        u_abc = [u * u_s for u in u_abc for u_s in u_sup]
        u_dq = self.abc_to_dq_space(u_abc, eps_fs)
        i_dq = self.alphabeta_to_dq_space(ode_state[self._ode_currents_idx], eps_fs)
        i_abc = self.dq_to_abc_space(i_dq, eps_fs)
        torque = self.electrical_motor.torque(motor_state)
        noise = self._noise_generator.reset()
        self._t = 0
        self._k = 0
        self._ode_solver.set_initial_value(ode_state, self._t)
        system_state = np.concatenate([
            mechanical_state, [torque],
            i_abc, i_dq,
            u_abc, u_dq,
            [eps],
            u_sup
        ])
        return (system_state + noise) / self._limits


class DoublyFedInductionMotorSystem(ThreePhaseMotorSystem):
    """
    SCML-System for the Doubly Fed Induction Motor
    """
    def __init__(self, control_space='abc', ode_solver='scipy.ode', **kwargs):
        """
        Args:
            control_space(str):('abc' or 'dq') Choose, if actions the actions space is in dq or abc space
            kwargs: Further arguments to pass tp SCMLSystem
        """
        super().__init__(ode_solver=ode_solver, **kwargs)
        self.control_space = control_space
        if control_space == 'dq':
            self._action_space = Box(-1, 1, shape=(4,), dtype=np.float64)

        self.stator_voltage_space_idx = 0
        self.stator_voltage_low_idx = 0
        self.stator_voltage_high_idx = \
            self.stator_voltage_low_idx \
            + self._converter.subsignal_voltage_space_dims[self.stator_voltage_space_idx]

        self.rotor_voltage_space_idx = 1
        self.rotor_voltage_low_idx = self.stator_voltage_high_idx
        self.rotor_voltage_high_idx = \
            self.rotor_voltage_low_idx \
            + self._converter.subsignal_voltage_space_dims[self.rotor_voltage_space_idx]

    def _set_limits(self):
        """
        Method to set the physical limits from the modules.
        """
        for ind, state in enumerate(self._state_names):
            motor_lim = self._electrical_motor.limits.get(state, np.inf)
            mechanical_lim = self._mechanical_load.limits.get(state, np.inf)
            self._limits[ind] = min(motor_lim, mechanical_lim)
        self._limits[self._state_positions['u_sup']] = self.supply.u_nominal

    def _build_state_space(self, state_names):
        # Docstring of superclass
        low = -1 * np.ones_like(state_names, dtype=float)
        low[self.U_SUP_IDX] = 0.0
        high = np.ones_like(state_names, dtype=float)
        return Box(low, high, dtype=np.float64)

    def _build_state_names(self):
        # Docstring of superclass
        names_l = \
            self._mechanical_load.state_names \
            + [
                'torque',
                'i_sa', 'i_sb', 'i_sc', 'i_sd', 'i_sq',
                'i_ra', 'i_rb', 'i_rc', 'i_rd', 'i_rq',
                'u_sa', 'u_sb', 'u_sc', 'u_sd', 'u_sq',
                'u_ra', 'u_rb', 'u_rc', 'u_rd', 'u_rq',
                'epsilon', 'u_sup',
            ]
        return names_l

    def _set_indices(self):
        # Docstring of superclass
        super()._set_indices()
        self._motor_ode_idx += range(self._motor_ode_idx[-1] + 1, self._motor_ode_idx[-1] + 1 + len(self._electrical_motor.FLUXES))
        self._motor_ode_idx += [self._motor_ode_idx[-1] + 1]

        self._ode_currents_idx = self._motor_ode_idx[self._electrical_motor.I_SALPHA_IDX:self._electrical_motor.I_SBETA_IDX + 1]
        self._ode_flux_idx = self._motor_ode_idx[self._electrical_motor.PSI_RALPHA_IDX:self._electrical_motor.PSI_RBETA_IDX + 1]

        self.OMEGA_IDX = self.mechanical_load.OMEGA_IDX
        self.TORQUE_IDX = len(self.mechanical_load.state_names)
        currents_lower = self.TORQUE_IDX + 1
        currents_upper = currents_lower + 10
        self.CURRENTS_IDX = list(range(currents_lower, currents_upper))
        voltages_lower = currents_upper
        voltages_upper = voltages_lower + 10
        self.VOLTAGES_IDX = list(range(voltages_lower, voltages_upper))
        self.EPSILON_IDX = voltages_upper
        self.U_SUP_IDX = list(range(self.EPSILON_IDX + 1, self.EPSILON_IDX + 1 + self._supply.voltage_len))

        self._ode_epsilon_idx = self._motor_ode_idx[-1]

    def calculate_field_angle(self, state):
        # field angle is calculated from states
        psi_ralpha = state[self._motor_ode_idx[self._electrical_motor.PSI_RALPHA_IDX]]
        psi_rbeta = state[self._motor_ode_idx[self._electrical_motor.PSI_RBETA_IDX]]
        eps_fs = np.arctan2(psi_rbeta, psi_ralpha)
        return eps_fs

    def calculate_rotor_current(self, state):
        # rotor current is calculated from states
        mp = self._electrical_motor.motor_parameter
        l_r = mp['l_m'] + mp['l_sigr']

        i_salpha = state[self._motor_ode_idx[self._electrical_motor.I_SALPHA_IDX]]
        i_sbeta = state[self._motor_ode_idx[self._electrical_motor.I_SBETA_IDX]]
        psi_ralpha = state[self._motor_ode_idx[self._electrical_motor.PSI_RALPHA_IDX]]
        psi_rbeta = state[self._motor_ode_idx[self._electrical_motor.PSI_RBETA_IDX]]

        i_ralpha = 1 / l_r * psi_ralpha - mp['l_m'] / l_r * i_salpha
        i_rbeta = 1 / l_r * psi_rbeta - mp['l_m'] / l_r * i_sbeta
        return [i_ralpha, i_rbeta]

    def simulate(self, action, *_, **__):
        # Docstring of superclass

        # Coordinate Systems used here:
        # alphabeta refers to the stator-fixed two-phase reference frame
        # gammadelta refers to the rotor-fixed two-phase reference frame
        # abc refers to the stator-fixed three-phase reference frame
        # def refers to the rotor-fixed three-phase reference frame
        # dq refers to the field-oriented (two-phase) reference frame
        # e.g. u_rdef is the rotor voltage representation in the rotor-fixed three-phase reference frame
        # u_rabc ist the rotor voltage representation in the stator-fixed three-phase reference frame

        ode_state = self._ode_solver.y

        eps_field = self.calculate_field_angle(ode_state)
        eps_el = ode_state[self._ode_epsilon_idx]

        # convert dq input voltage to abc
        if self.control_space == 'dq':
            stator_input_len = len(self._electrical_motor.STATOR_VOLTAGES)
            rotor_input_len = len(self._electrical_motor.ROTOR_VOLTAGES)
            action_stator = action[:stator_input_len]
            action_rotor = action[stator_input_len:stator_input_len + rotor_input_len]
            action_stator = self.dq_to_abc_space(action_stator, eps_field)
            action_rotor = self.dq_to_abc_space(action_rotor, eps_field-eps_el)
            action = np.concatenate((action_stator, action_rotor)).tolist()

        i_sabc = self.alphabeta_to_abc_space(self._electrical_motor.i_in(ode_state[self._ode_currents_idx]))
        i_rdef = self.alphabeta_to_abc_space(self.calculate_rotor_current(ode_state))
        switching_times = self._converter.set_action(action, self._t)

        for t in switching_times[:-1]:
            i_sup = self._converter.i_sup(np.concatenate((i_sabc, i_rdef)))
            u_sup = self._supply.get_voltage(self._t, i_sup)
            u_in = self._converter.convert(np.concatenate([i_sabc, i_rdef]).tolist(), self._ode_solver.t)
            u_in = [u * u_s for u in u_in for u_s in u_sup]
            u_sabc = u_in[self.stator_voltage_low_idx:self.stator_voltage_high_idx]
            u_rdef = u_in[self.rotor_voltage_low_idx:self.rotor_voltage_high_idx]
            u_rdq = self.abc_to_dq_space(u_rdef, eps_field-eps_el)
            u_salphabeta = self.abc_to_alphabeta_space(u_sabc)
            u_ralphabeta = self.dq_to_alphabeta_space(u_rdq, eps_field)

            u_sr_alphabeta = np.array([u_salphabeta, u_ralphabeta])
            self._ode_solver.set_f_params(u_sr_alphabeta)
            ode_state = self._ode_solver.integrate(t)

            eps_field = self.calculate_field_angle(ode_state)
            eps_el = ode_state[self._ode_epsilon_idx]
            i_sabc = self.alphabeta_to_abc_space(self._electrical_motor.i_in(ode_state[self._ode_currents_idx]))
            i_rdef = self.alphabeta_to_abc_space(self.calculate_rotor_current(ode_state))
            
        i_sup = self._converter.i_sup(np.concatenate((i_sabc, i_rdef)))
        u_sup = self._supply.get_voltage(self._t, i_sup)
        u_in = self._converter.convert(np.concatenate([i_sabc, i_rdef]).tolist(), self._ode_solver.t)
        u_in = [u * u_s for u in u_in for u_s in u_sup]
        u_sabc = u_in[self.stator_voltage_low_idx:self.stator_voltage_high_idx]
        u_rdef = u_in[self.rotor_voltage_low_idx:self.rotor_voltage_high_idx]
        u_sdq = self.abc_to_dq_space(u_sabc, eps_field)
        u_rdq = self.abc_to_dq_space(u_rdef, eps_field-eps_el)
        u_salphabeta = self.abc_to_alphabeta_space(u_sabc)
        u_ralphabeta = self.dq_to_alphabeta_space(u_rdq, eps_field)

        u_sr_alphabeta = np.array([u_salphabeta, u_ralphabeta])
        self._ode_solver.set_f_params(u_sr_alphabeta)
        ode_state = self._ode_solver.integrate(self._t + self._tau)
        self._t = self._ode_solver.t
        self._k += 1
        torque = self._electrical_motor.torque(ode_state[self._motor_ode_idx])
        noise = self._noise_generator.noise()
        mechanical_state = ode_state[self._load_ode_idx]

        i_sdq = self.alphabeta_to_dq_space(ode_state[self._ode_currents_idx], eps_field)
        i_sabc = list(self.dq_to_abc_space(i_sdq, eps_field))

        i_rdq = self.alphabeta_to_dq_space(self.calculate_rotor_current(ode_state), eps_field)
        i_rdef = list(self.dq_to_abc_space(i_rdq, eps_field-eps_el))

        eps_el = ode_state[self._ode_epsilon_idx] % (2 * np.pi)
        if eps_el > np.pi:
            eps_el -= 2 * np.pi

        system_state = np.concatenate((
            mechanical_state,
            [torque],
            i_sabc, i_sdq,
            i_rdef, i_rdq,
            u_sabc, u_sdq,
            u_rdef, u_rdq,
            [eps_el],
            u_sup,
        ))
        return (system_state + noise) / self._limits

    def reset(self, *_):
        # Docstring of superclass
        mechanical_state = self._mechanical_load.reset(
            state_positions=self.state_positions,
            state_space=self.state_space,
            nominal_state=self.nominal_state)
        motor_state = self._electrical_motor.reset(
            state_space=self.state_space,
            state_positions=self.state_positions,
            omega=mechanical_state)
        ode_state = np.concatenate((mechanical_state, motor_state))
        u_sup = self.supply.reset()

        eps_el = ode_state[self._ode_epsilon_idx]
        eps_field = self.calculate_field_angle(ode_state)

        if eps_el > np.pi:
            eps_el -= 2 * np.pi

        if eps_field > np.pi:
            eps_field -= 2 * np.pi

        u_sr_abcdef = self.converter.reset()
        u_sr_abcdef = [u * u_s for u in u_sr_abcdef for u_s in u_sup]
        u_sabc = u_sr_abcdef[self.stator_voltage_low_idx:self.stator_voltage_high_idx]
        u_rdef = u_sr_abcdef[self.rotor_voltage_low_idx:self.rotor_voltage_high_idx]
        u_sdq = self.abc_to_dq_space(u_sabc, eps_field)
        u_rdq = self.abc_to_dq_space(u_rdef, eps_field-eps_el)

        i_sdq = self.alphabeta_to_dq_space(ode_state[self._ode_currents_idx], eps_field)
        i_sabc = self.dq_to_abc_space(i_sdq, eps_field)

        i_rdq = self.alphabeta_to_dq_space(self.calculate_rotor_current(ode_state), eps_field-eps_el)
        i_rdef = self.dq_to_abc_space(i_rdq, eps_field-eps_el)

        torque = self.electrical_motor.torque(motor_state)
        noise = self._noise_generator.reset()
        self._t = 0
        self._k = 0
        self._ode_solver.set_initial_value(ode_state, self._t)
        system_state = np.concatenate([
            mechanical_state, [torque],
            i_sabc, i_sdq,
            i_rdef, i_rdq,
            u_sabc, u_sdq,
            u_rdef, u_rdq,
            [eps_el],
            u_sup
        ])
        return (system_state + noise) / self._limits
