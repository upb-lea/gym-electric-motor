from gym.spaces import Discrete, Box
from gym_electric_motor.physical_systems.electric_motors import DcShuntMotor, DcExternallyExcitedMotor, \
    DcPermanentlyExcitedMotor, DcSeriesMotor
from gym_electric_motor.physical_systems import SynchronousMotorSystem
import math
import numpy as np


class Controller:

    @classmethod
    def make(cls, controller_type, environment, **controller_kwargs):
        assert controller_type in _controllers.keys(), f'Controller {controller_type} unknown'
        controller = _controllers[controller_type](environment, **controller_kwargs)
        return controller

    def control(self, state, reference):
        raise NotImplementedError

    def reset(self):
        pass


class OnOffController(Controller):

    def __init__(self, environment, state_idx=None, reference_idx=0):
        action_space = environment.action_space
        assert type(action_space) is Discrete, 'Not suitable action space for On off controller'
        self._high_action = 1
        if action_space.n in [3, 4]:
            self._low_action = 2
        else:
            self._low_action = 0
        self._referenced_state = state_idx or np.argmax(environment.reference_generator.referenced_states)
        self._ref_idx = reference_idx

    def control(self, state, reference):
        if state[self._referenced_state] < reference[self._ref_idx]:
            return self._high_action
        else:
            return self._low_action


class ThreePointController(Controller):

    def __init__(self, environment, hysteresis=0.01, state_idx=None, reference_idx=0):
        action_space = environment.action_space
        assert type(action_space) is Discrete, 'Not suitable action space for three point controller'
        self._hysteresis = hysteresis
        self._high_action = 1
        self._idle_action = 0
        self._ref_idx = reference_idx
        if action_space.n in [3, 4]:
            self._low_action = 2
        else:
            self._low_action = 0
        self._referenced_state = state_idx or np.argmax(environment.reference_generator.referenced_states)

    def control(self, state, reference):
        if state[self._referenced_state] < reference[self._ref_idx] - self._hysteresis:
            return self._high_action
        elif state[self._referenced_state] > reference[self._ref_idx] + self._hysteresis:
            return self._low_action
        else:
            return self._idle_action


class PController(Controller):

    def __init__(self, environment, k_p=10, controller_no=0, state_idx=None, reference_idx=0):
        action_space = environment.action_space
        assert type(action_space) is Box, 'No suitable action space for P Controller'
        self._k_p = k_p
        self._controller_no = controller_no
        self._action_min = action_space.low[controller_no]
        self._action_max = action_space.high[controller_no]
        self._ref_idx = reference_idx
        self._referenced_state = state_idx or np.argmax(environment.reference_generator.referenced_states)

    def control(self, state, reference):
        return np.array([
            max(
                self._action_min,
                min(
                    self._action_max,
                    self._k_p * (reference[self._ref_idx] - state[self._referenced_state])
                )
            )
        ])


class PIController(PController):

    def __init__(self, environment, k_p=10, k_i=0.01, controller_no=0, reference_idx=0):
        super().__init__(environment, k_p, controller_no, reference_idx)
        self._k_i = k_i
        self._tau = environment.physical_system.tau
        self._integrated_value = 0

    def control(self, state, reference):
        diff = reference[self._ref_idx] - state[self._referenced_state]
        self._integrated_value += diff * self._tau
        return np.array([
            max(
                self._action_min,
                min(
                    self._action_max,
                    self._k_p * (reference[0] - state[self._referenced_state])
                    + self._k_i / self._tau * self._integrated_value
                )
            )
        ])

    def reset(self, **__):
        self._integrated_value = 0


class PmsmOnOffController(Controller):

    def __init__(self, environment, state_idx=None, ref_idx=0):
        t32 = environment.physical_system.electrical_motor.t_32
        q = environment.physical_system.electrical_motor.q
        t23 = environment.physical_system.electrical_motor.t_23
        q_inv = environment.physical_system.electrical_motor.q_inv
        self._forward_transformation = lambda quantities, eps: q_inv(t23(quantities), eps)[::-1]
        self._backward_transformation = (
            lambda quantities, eps: t32(q(quantities[::-1], eps))
        )
        self._l_q = environment.physical_system.electrical_motor.motor_parameter['l_q']
        self._epsilon_idx = environment.physical_system.EPSILON_IDX
        self._currents_idx = environment.physical_system.CURRENTS_IDX
        self._ref_idx = ref_idx
        self._omega_idx = environment.physical_system.state_positions['omega']
        self._u_sup = environment.physical_system.supply.u_nominal
        self._referenced_state = state_idx or np.argmax(environment.reference_generator.referenced_states)
        self._limits = environment.physical_system.electrical_motor.limits

    def control(self, state, reference):
        if state[self._referenced_state] < reference[self._ref_idx]:
            u_q = 1
        else:
            u_q = -1
        epsilon = np.pi * state[self._epsilon_idx]
        u_d = 0
        u_a, u_b, u_c = self._backward_transformation((u_q, u_d), epsilon)
        return 4 * (u_a > 0) + 2 * (u_b > 0) + (u_c > 0)


class SynRmOnOffController(PmsmOnOffController):

    def control(self, state, reference):
        if state[self._referenced_state] < reference[self._ref_idx]:
            u_q = 1
            u_d = 1
        else:
            u_q = -1
            u_d = -1
        epsilon = state[self._epsilon_idx]
        u_a, u_b, u_c = self._backward_transformation((u_q, u_d), epsilon)
        return 4 * u_a > 0 + 2 * u_b > 0 + u_c > 0


class CascadedPIController(Controller):

    def __init__(self, environment, ref_idx=0):
        self._omega_idx = environment.physical_system.OMEGA_IDX
        self._currents_idx = environment.physical_system.CURRENTS_IDX
        self._voltages_idx = environment.physical_system.VOLTAGES_IDX
        self._u_a_idx = self._voltages_idx[0]
        self._i_a_idx = self._currents_idx[0]

        if len(self._currents_idx) > 1:
            self._i_e_idx = environment.physical_system.state_positions['i_e']
        else:
            self._i_e_idx = environment.physical_system.state_positions['i']
        if len(self._voltages_idx) > 1:
            self._u_e_idx = environment.physical_system.state_positions['u_e']
        else:
            self._u_e_idx = None
        self._limits = environment.physical_system.limits
        self._ref_idx = ref_idx
        self._tau = environment.physical_system.tau
        mp = environment.physical_system.electrical_motor.motor_parameter
        t_motor = mp['l_a'] / mp['r_a']
        t_t = 3 / 2 * self._tau
        r_motor = mp['r_a']
        self._i_a_max = 0
        self._i_a_min = 0
        self._u_a_max = 0
        self._u_a_min = 0
        self._integrated_values = [0, 0]
        self._converter_voltages = environment.physical_system.converter.voltages
        self._i_a_max = self._limits[self._i_a_idx] * environment.physical_system.state_space.high[self._i_a_idx]
        self._i_a_min = self._limits[self._i_a_idx] * environment.physical_system.state_space.low[self._i_a_idx]
        if 'psi_e' in mp.keys():
            self._psi_e = mp['psi_e']
            self._i_e_max_prime = None
        elif 'l_e_prime' in mp.keys():
            self._psi_e = None
            self._i_e_max_prime = self._limits[self._currents_idx[-1]] * mp['l_e_prime']
        else:
            raise Exception('Motor Parameter Error. No psi_e and no l_e_prime entry found in motor parameters')
        self._u_a_max = self._limits[self._u_a_idx] * environment.physical_system.state_space.high[self._u_a_idx]
        self._u_a_min = self._limits[self._u_a_idx] * environment.physical_system.state_space.low[self._u_a_idx]
        # compute motor type specific parameter
        # use inner_ and outer_gain_adjustment to adjust the integral part gains for better control behaviour
        # Gains chosen as given in "Elektrische Antriebe - Regelung von Antriebssystemen", D. Schröder, 2009
        if type(environment.physical_system.electrical_motor) == DcPermanentlyExcitedMotor:
            inner_gain_adjustment = 1e-3
            outer_gain_adjustment = 1e-3
        elif type(environment.physical_system.electrical_motor) == DcSeriesMotor:
            t_motor = (mp['l_a'] + mp['l_e']) / (mp['r_a'] + mp['r_e'])
            r_motor = (mp['r_a'] + mp['r_e'])

            inner_gain_adjustment = 1
            outer_gain_adjustment = 1

        elif type(environment.physical_system.electrical_motor) == DcExternallyExcitedMotor:
            inner_gain_adjustment = 1E-4
            outer_gain_adjustment = 1E-3

        elif type(environment.physical_system.electrical_motor) == DcShuntMotor:
            inner_gain_adjustment = 1E-2
            outer_gain_adjustment = 1
        else:
            raise Exception('Unknown Motor')

        # set up gains for the controller
        # Integral gains are multiplied by the sampling time to simplify the computation during control

        t_sigma = min(t_motor, t_t)
        t_1 = max(t_motor, t_t)
        v_s = 1 / r_motor
        # Integral Inner loop
        self._k_i_i = 1 / (2 * t_sigma * v_s) * self._tau * inner_gain_adjustment
        # Proportional Inner loop
        self._k_p_i = t_1 / (2 * t_sigma * v_s)
        # Integral Outer loop
        j = environment.physical_system.mechanical_load.j_total
        self._k_i_o = (
                j / (32 * t_sigma ** 2)
                * self._tau * outer_gain_adjustment
        )
        # Proportional Outer loop
        self._k_p_o = j / (4 * t_sigma)

    def control(self, state, reference):
        # denormalize quantities
        omega = state[self._omega_idx] * self._limits[self._omega_idx]
        omega_ref = reference[self._ref_idx] * self._limits[self._omega_idx]
        i_a = state[self._i_a_idx] * self._limits[self._i_a_idx]

        psi_e = self._psi_e or state[self._i_e_idx] * self._i_e_max_prime

        # outer control loop
        d_omega = omega_ref - omega
        if psi_e != 0:
            temp = self._integrated_values[0] + d_omega * self._k_i_o / psi_e  # integral part
            i_a_des = temp + d_omega * self._k_p_o / psi_e
        else:
            i_a_des = math.copysign(1, d_omega) * self._i_a_max
            temp = self._integrated_values[0]

        # hold current constraints, anti wind-up
        if i_a_des > self._i_a_max or i_a_des < self._i_a_min:
            i_a_des = min(max(i_a_des, self._i_a_min), self._i_a_max)
        else:
            self._integrated_values[0] = temp

        d_i_a = i_a_des - i_a
        # inner control loop
        temp = self._integrated_values[1] + d_i_a * self._k_i_i  # integral part
        d_u_a = temp + d_i_a * self._k_p_i
        u_a_0 = omega * psi_e
        u_a = d_u_a + u_a_0

        # hold voltage limits, anti wind-up
        if u_a > self._u_a_max or u_a < self._u_a_min:
            u_a = min(max(u_a, self._u_a_min), self._u_a_max)
        else:
            self._integrated_values[1] = temp

        # normalize the desired output voltage to a duty cycle referring to the supply voltage
        # Assumption: u_sup = u_N is made
        des_duty_cycle = u_a / self._limits[self._u_a_idx]
        duty_cycle = min(
            max(des_duty_cycle, self._u_a_min / self._limits[self._u_a_idx]),
            self._u_a_max / self._limits[self._u_a_idx])
        return np.array([duty_cycle])


class FOCController(Controller):

    def __init__(self, environment, ref_idx=0, weight=1):
        assert type(environment.physical_system) is SynchronousMotorSystem
        self._ref_idx = ref_idx
        self._weight = weight
        self._omega_idx = environment.physical_system.OMEGA_IDX
        self._currents_idx = environment.physical_system.CURRENTS_IDX
        self._voltages_idx = environment.physical_system.VOLTAGES_IDX
        self._epsilon_idx = environment.physical_system.EPSILON_IDX
        self._limits = environment.physical_system.limits
        self._tau = environment.physical_system.tau
        t32 = environment.physical_system.electrical_motor.t_32
        q = environment.physical_system.electrical_motor.q
        t23 = environment.physical_system.electrical_motor.t_23
        q_inv = environment.physical_system.electrical_motor.q_inv
        self._forward_transformation = lambda quantities, eps: q_inv(t23(quantities), eps)[::-1]
        self._backward_transformation = (
            lambda quantities, eps: t32(q(quantities[::-1], eps))
        )
        self._motor_parameter = environment.physical_system.electrical_motor.motor_parameter
        mp = self._motor_parameter
        # current controller i_d
        t_motor_d = mp['l_d'] / mp['r_s']
        tau = environment.physical_system.tau
        t_t = 3 / 2 * tau
        t_1_d = max(t_motor_d, t_t)
        t_sigma_d = min(t_motor_d, t_t)
        v_s_d = 1 / mp['r_s']

        # current controller i_q
        t_motor_q = mp['l_q'] / mp['r_s']
        t_1_q = max(t_motor_q, t_t)
        t_sigma_q = min(t_motor_q, t_t)
        v_s_q = 1 / mp['r_s']

        # outer speed controller
        t_2 = 2 * t_sigma_q
        t_1_s = environment.physical_system.mechanical_load.j_total
        v_s_s = 3 / 2 * mp['p'] * mp['psi_p']

        self._k_i_t = 2 * t_1_s / v_s_s * tau  # integral gain speed controller.
        self._k_p_t = t_1_s / (2 * t_2 * v_s_s)  # prop. gain speed controller
        self._k_i_d = 1 / (2 * t_sigma_d * v_s_d) * tau  # integral gain i_sd controller.
        self._k_p_d = t_1_d / (2 * t_sigma_d * v_s_d)  # prop. gain i_sd controller
        self._k_i_q = 1 / (2 * t_sigma_q * v_s_q) * tau  # integral gain i_sq controller.
        self._k_p_q = t_1_q / (2 * t_sigma_q * v_s_q)  # prop. gain i_sq controller

        # specify max values for normalisation and anti wind up
        # an anti wind up scheme is necessary for good control behaviour to limit the integral parts in case of
        # limit violations of the desired input voltage

        # maximum speed without flux weakening
        self._omega_1 = (
                self._limits[self._voltages_idx][0] / mp['l_q'] / np.sqrt(self._limits[self._currents_idx][0]) ** 2
                + mp['psi_p'] ** 2 / mp['l_q'] ** 2
        )
        self._integrated_values = [0, 0, 0]

    def reset(self):
        self._integrated_values = [0, 0, 0]

    def control(self, state, reference):
        """
        Field oriented control from the lecture "controlled three phase drives, chapter 5"
        """

        # extract quantities from state
        mp = self._motor_parameter
        omega = state[self._omega_idx] * self._limits[self._omega_idx]
        omega_ref = reference[self._ref_idx] * self._limits[self._omega_idx]

        u = state[self._voltages_idx] * self._limits[self._voltages_idx]
        epsilon = state[self._epsilon_idx] * self._limits[self._epsilon_idx]
        i = state[self._currents_idx] * self._limits[self._currents_idx]
        # transformation from a/b/c to alpha/beta and d/q

        i_qd = self._forward_transformation(i, epsilon)

        # compute u_d_0 and u_q_0
        u_d_0 = omega * mp['l_q'] * i_qd[0]
        u_q_0 = omega * (mp['psi_p'] + mp['l_d'] * i_qd[1])
        d_omega = omega_ref - omega

        # compute T* (Torque reference) and i*_sq (q-axis current reference)
        temp = self._integrated_values[0] + d_omega * self._k_i_t  # integral part
        t_des = temp + d_omega * self._k_p_t  # proportional part
        i_sq_des = 2 * t_des / (3 * mp['p'] * mp['psi_p'])
        # anti wind-up
        if i_sq_des > self._limits[self._currents_idx[0]] * self._weight\
                or i_sq_des < -self._limits[self._currents_idx[0]] * self._weight:
            i_sq_des = min(
                max(i_sq_des, -self._limits[self._currents_idx[0]] * self._weight),
                self._limits[self._currents_idx[0]] * self._weight
            )
        else:
            self._integrated_values[0] = temp

        if abs(omega_ref) < self._omega_1:
            i_sd_des = 0
        else:
            i_sd_des = (
                (self._limits[self._voltages_idx[0]] / omega_ref) ** 2
                - (mp['l_q'] * self._limits[self._currents_idx[0]]) ** 2 - mp['psi_p'] ** 2
                / (2 * mp['psi_p'] * mp['l_d']))

        # transform back to abc-domain
        currents = self._backward_transformation((i_sq_des, i_sd_des), epsilon)

        # test if current limits are violated
        if np.max(np.abs(currents)) > self._limits[self._currents_idx[0]]:
            clipping = self._limits[self._currents_idx]
            currents = np.clip(currents, -clipping, clipping)
            array = self._forward_transformation(currents, epsilon)
            i_sd_des = array[1]
            i_sq_des = array[0]

        # compute du*_sq, du*_sd
        d_i_sd = i_sd_des - i_qd[1]
        d_i_sq = i_sq_des - i_qd[0]
        temp_u_sd = self._integrated_values[1] + d_i_sd * self._k_i_d  # integral part
        temp_u_sq = self._integrated_values[2] + d_i_sq * self._k_i_q  # integral part
        d_u_sd_des = temp_u_sd + d_i_sd * self._k_p_d
        d_u_sq_des = temp_u_sq + d_i_sq * self._k_p_q
        # anti-wind-up u_sd
        if d_u_sd_des > self._limits[self._voltages_idx[0]] * self._weight - u_d_0 or \
                d_u_sd_des < -self._limits[self._voltages_idx[0]] * self._weight - u_d_0:
            d_u_sd_des = np.clip(d_u_sd_des, -self._limits[self._voltages_idx[0]] * self._weight - u_d_0,
                                 self._limits[self._voltages_idx[0]] * self._weight - u_d_0)
        else:
            self._integrated_values[1] = temp_u_sd
        # anti-wind-up u_sq
        if d_u_sq_des > self._limits[self._voltages_idx[0]] * self._weight - u_q_0 or \
                d_u_sq_des < -self._limits[self._voltages_idx[0]] * self._weight - u_q_0:
            d_u_sq_des = np.clip(d_u_sq_des, -self._limits[self._voltages_idx[0]] * self._weight - u_q_0,
                                 self._limits[self._voltages_idx[0]] * self._weight - u_q_0)
        else:
            self._integrated_values[2] = temp_u_sq

        # compute u*_sq, u*_sd, epsilon + d_epsilon due to delay of the controller
        u_sd_des = u_d_0 + d_u_sd_des
        u_sq_des = d_u_sq_des + u_q_0
        epsilon_shift = epsilon + 3 / 2 * self._tau * omega

        # from d/q to alpha/beta and a/b/c
        u_qd_des = np.array([u_sq_des, u_sd_des])
        voltages = self._backward_transformation(u_qd_des, epsilon_shift)

        # normalise inputs
        result = np.clip(voltages / self._limits[self._voltages_idx[0]], -1, 1)
        return result


class PmsmPController(Controller):

    def __init__(self, environment, state_idx=None, ref_idx=0, k_p=1):
        self._k_p = k_p
        t32 = environment.physical_system.electrical_motor.t_32
        q = environment.physical_system.electrical_motor.q
        t23 = environment.physical_system.electrical_motor.t_23
        q_inv = environment.physical_system.electrical_motor.q_inv
        self._forward_transformation = lambda quantities, eps: q_inv(t23(quantities), eps)[::-1]
        self._backward_transformation = (
            lambda quantities, eps: t32(q(quantities[::-1], eps))
        )
        self._epsilon_idx = environment.physical_system.EPSILON_IDX
        self._currents_idx = environment.physical_system.CURRENTS_IDX
        self._ref_idx = ref_idx
        self._referenced_state = state_idx or np.argmax(environment.reference_generator.referenced_states)
        self._phase = 0

    def control(self, state, reference):
        u_q = min(1, max(-1, self._k_p * reference[self._ref_idx] - state[self._referenced_state]))
        epsilon = np.pi * state[self._epsilon_idx]

        u_d = 0
        u_a, u_b, u_c = self._backward_transformation((u_q, u_d), epsilon)
        return [u_a, u_b, u_c]

    def reset(self):
        self._phase = 0


class ThreePhaseSteadyState(Controller):

    def __init__(self, environment, omega_el=15):
        self._omega_el = omega_el
        self._tau = environment.physical_system.tau
        self._k = 0
        t = np.linspace(0, 2 * np.pi / abs(omega_el), 1 / abs(omega_el * self._tau))
        self._u_a = np.sin(omega_el * t)
        self._u_b = np.sin(omega_el * t - 2/3 * np.pi)
        self._u_c = np.sin(omega_el * t + 2/3 * np.pi)

    def reset(self):
        self._k = -1

    def control(self, state, reference):
        self._k += 1
        length = len(self._u_a)
        return self._u_a[self._k % length], self._u_b[self._k % length], self._u_c[self._k % length],


_controllers = {
    'on_off': OnOffController,
    'three_point': ThreePointController,
    'p_controller': PController,
    'pi_controller': PIController,
    'pmsm_on_off': PmsmOnOffController,
    'synrm_on_off': SynRmOnOffController,
    'cascaded_pi': CascadedPIController,
    'foc_controller': FOCController,
    'pmsm_p_controller': PmsmPController,
    'three_phase_steadystate': ThreePhaseSteadyState
}
