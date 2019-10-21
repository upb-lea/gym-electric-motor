import numpy as np
from gym_electric_motor.envs.gym_pmsm.models.pmsm_model import PmsmModel


class Controller(object):

    """
    Class including several controllers.
    
    Initialize a specific controller at the beginning
    call the function "control" to apply the controller
    
    Control=Controller(controller_type='PI_controller',param={'K_i':K_i,'K_p':K_p,'tau':tau})
    Control.control(state,ref,k) # k is the current time step
    The param dict can also include the motor parameter.

    The integration is paused if limits are exceeded in the controller as anti-wind-up.

    Args:
        param: Parameter Dictionary with keys as described below.

    Param Keys:
        cumulated_values: Summation of all previous errors for the integral part
        control: function that should be used to apply the controller
        control_param: including all necessary parameters for the control, also motor params for adaptive controller
            always
        tau: sampling and update frequency of the system
        K_p: gain of the proportional part of the PI Controller
        K_i: gain of the integral part of the PI Controller
        hysteresis: hysteresis value for the hysteresis controller

    Some controllers include further parameters as the cascaded ones for example need further gains and motor parameter
    for the feed forward control
    """

    def __init__(self, controller_type='PI_controller', param=None):
        """
        Initialisation of a controller

        Args:
            controller_type: selected controller type
            param: parameters of the controller and motor
        """
        assert controller_type in _select.keys(), 'Controller not available'
        self.controller_type, self.cumulated_values, self.controller_params = _select[controller_type]
        self.action_space = None
        if param is not None:
            # Use motor parameter to set controller parameter
            name_split = param['env_name'].split('-')
            if param is not None and name_split[1] == 'pmsm':
                if self.controller_params is None:
                    self.controller_params = param
                else:
                    self.controller_params.update(param)

                # Determine limit values of the quantities
                self.controller_params['omega_max'] = self.controller_params['omega_N']\
                                                      * self.controller_params['safety_margin']
                self.controller_params['u_max'] = self.controller_params['u_N'] * self.controller_params['safety_margin'] / 2
                self.controller_params['i_max'] = self.controller_params['i_N'] * self.controller_params['safety_margin']
                self.controller_params['epsilon_max'] = 2 * np.pi * self.controller_params['safety_margin']
            elif param is not None and name_split[1] == 'dc':

                # set inital sampling time t0 1E-4
                if self.controller_params is None:
                    self.controller_params = {'tau': 1E-4}
                else:
                    self.controller_params.update({'tau': 1E-4})

                # update controller parameters  with motor parameters
                self.controller_params.update(param)

        # use special setup functions for some controller
        if controller_type == 'cascaded_PI_controller':
            self.setup_cascaded_controller(param)
        elif controller_type == 'foc_controller':
            self.setup_foc_controller(param)

        # set the control function that will be taken
        self.control = lambda state_in, ref_in, k, *args: self.controller_type(self, state_in, ref_in, k, *args)

    # region setup parameter

    def setup_cascaded_controller(self, params=None):
        """
        Set the parameter of the cascaded controller for DC motors
        The controller can be used for the armature voltage of the externally excited motor and all other DC motors
        Args:
            params: dict, containing the motor params
        """

        if params is None:  # use default parameter
            tau = 1e-4
            self.controller_params = {'tau': tau}
            self.controller_params.update({'K_p_i': 1, 'K_i_i': 1 * tau, 'K_p_o': 1, 'K_i_o': 1 * tau})
            print("Use default parameters, all Gains 1")
        else:  # use motor parameters to adapt the controller
            self.controller_params = params
            self.controller_params['omega_max'] = self.controller_params['omega_N'] \
                * self.controller_params['safety_margin']
            T_motor = params['l_a'] / params['r_a']
            T_t = 3 / 2 * params['tau']
            r_motor = params['r_a']

        # compute motor type specific parameter
        # use inner_ and outer_gain_adjustment to adjust the integral part gains for better control behaviour
        # Gains choosen as given in "Elektrische Antriebe - Regelung von Antriebssystemem", D. Schröder, 2009
        if params['env_name'] == 'emotor-dc-permex-cont-v0':
            self._i_a_max = self.controller_params['i_N'] * self.controller_params['safety_margin'] \
                * params['converter_current'][1]
            self._i_a_min = self.controller_params['i_N'] * self.controller_params['safety_margin'] \
                * params['converter_current'][0]
            self.psi_e = self.controller_params['psi_e']
            self._u_a_max = self.controller_params['u_N'] * params['converter_voltage'][1]
            self._u_a_min = self.controller_params['u_N'] * params['converter_voltage'][0]
            self.controller_type = Controller._cascaded_PI_controller_Permex
            inner_gain_adjustment = 1E-3
            outer_gain_adjustment = 1E-3

        elif params['env_name'] == 'emotor-dc-series-cont-v0':
            T_motor = (params['l_a'] + params['l_e'])/(params['r_a'] + params['r_e'])
            r_motor = (params['r_a'] + params['r_e'])

            self._i_a_max = self.controller_params['i_N'] * self.controller_params['safety_margin']\
                * params['converter_current'][1]
            self._i_a_min = self.controller_params['i_N'] * self.controller_params['safety_margin']\
                * params['converter_current'][0]
            self._u_a_max = self.controller_params['u_N'] * params['converter_voltage'][1]
            self._u_a_min = self.controller_params['u_N'] * params['converter_voltage'][0]
            self._i_e_max_prime = self.controller_params['i_N'] * self.controller_params['safety_margin'] * \
                                  self.controller_params['l_e_prime']
            inner_gain_adjustment = 1
            outer_gain_adjustment = 1

        elif params['env_name'] == 'emotor-dc-extex-cont-v0':
            self._i_a_max = self.controller_params['i_a_N'] * self.controller_params['safety_margin']\
                            * params['converter_current'][1]
            self._i_a_min = self.controller_params['i_a_N'] * self.controller_params['safety_margin']\
                            * params['converter_current'][0]
            self._u_a_max = self.controller_params['u_a_N'] * params['converter_voltage'][1]
            self._u_a_min = self.controller_params['u_a_N'] * params['converter_voltage'][0]
            self._i_e_max_prime = self.controller_params['i_e_N'] * self.controller_params['safety_margin'] * \
                                  self.controller_params['l_e_prime']
            inner_gain_adjustment = 1E-4
            outer_gain_adjustment = 1E-3

        elif params['env_name'] == 'emotor-dc-shunt-cont-v0':
            self._i_a_max = self.controller_params['i_a_N'] * self.controller_params['safety_margin'] \
                * params['converter_current'][1]
            self._i_a_min = self.controller_params['i_a_N'] * self.controller_params['safety_margin'] \
                * params['converter_current'][0]
            self._u_a_max = self.controller_params['u_sup'] * params['converter_voltage'][1]
            self._u_a_min = self.controller_params['u_sup'] * params['converter_voltage'][0]
            self._i_e_max_prime = self.controller_params['i_e_N'] * self.controller_params['safety_margin'] \
                            * self.controller_params['l_e_prime']
            inner_gain_adjustment = 1E-2
            outer_gain_adjustment = 1

        else:
            print("Wrong Motor")

        # set up gains for the controller
        # Integral gains are multiplied by the sampling time to samplify the computation during control
        if params is not None:
            T_sigma = min(T_motor, T_t)
            T_1 = max(T_motor, T_t)
            V_S = 1 / r_motor
            # Integral Inner loop
            self.controller_params['K_i_i'] = 1 / (2 * T_sigma * V_S) * params['tau'] * inner_gain_adjustment
            # Proportional Inner loop
            self.controller_params['K_p_i'] = T_1 / (2 * T_sigma * V_S)
            # Integral Outer loop
            self.controller_params['K_i_o'] = params['j'] / (32 * T_sigma ** 2) * params['tau'] * outer_gain_adjustment
            # Proportional Outer loop
            self.controller_params['K_p_o'] = params['j'] / (4 * T_sigma)

    def setup_foc_controller(self, params=None):
        """
        Set the parameters for the field oriented control for the PMSM
        Use results from "Elektrische Antriebe - Regelung von Antriebssystemen", D. Schröder, 2009
        The controller parameters are based on motor parameters. Rotor fixed d-q-coordinates are used for controller.
        The inner loop contains a current controller and the outer loop a speed controller.
        Args:
            params: dict with {omega_N, u_N, i_N, epsilon_N, safety_margin, L_q, L_d, p, Psi_p, tau}
        """

        if params is None:  # set default parameter
            tau = 1e-4
            safety_margin = 1.3
            self.controller_params = {'tau': tau}
            self.controller_params.update({'K_i_T': 1 * tau, 'K_p_T': 1, 'K_i_d': 1 * tau, 'K_p_d': 1, 'K_i_q': 1 * tau,
                                           'K_p_q': 1})
            self.controller_params['omega_max'] = 100 * safety_margin
            self.controller_params['u_max'] = 400 * safety_margin
            self.controller_params['i_max'] = 100 * safety_margin
            self.controller_params['epsilon_max'] = 2 * np.pi * safety_margin
            self.controller_params['L_d'] = 79E-3
            self.controller_params['L_q'] = 113E-3
            self.controller_params['Psi_p'] = 5E-3
            self.controller_params['p'] = 2
            self.controller_params['safety_margin'] = safety_margin
            print("Use default parameter")
        else:
            self.controller_params.update(params)

            # current controller i_d
            T_motor_d = params['L_d'] / params['R_s']
            T_t = 3 / 2 * params['tau']
            T_1_d = max(T_motor_d, T_t)
            T_sigma_d = min(T_motor_d, T_t)
            V_S_d = 1 / params['R_s']

            # current controller i_q
            T_motor_q = params['L_q'] / params['R_s']
            T_1_q = max(T_motor_q, T_t)
            T_sigma_q = min(T_motor_q, T_t)
            V_S_q = 1 / params['R_s']

            # outer speed controller
            T_2 = 2 * T_sigma_q
            T_1_s = params['J_rotor']
            V_S_s = 3 / 2 * params['p'] * params['Psi_p']

            self.controller_params['K_i_T'] = 2 * T_1_s / V_S_s * params['tau']  # integral gain speed contr.
            self.controller_params['K_p_T'] = T_1_s / (2 * T_2 * V_S_s)  # prop. gain speed controller
            self.controller_params['K_i_d'] = 1 / (2 * T_sigma_d * V_S_d) * params['tau']  # integral gain i_sd contr.
            self.controller_params['K_p_d'] = T_1_d / (2 * T_sigma_d * V_S_d)  # prop. gain i_sd controller
            self.controller_params['K_i_q'] = 1 / (2 * T_sigma_q * V_S_q) * params['tau']  # integral gain i_sq contr.
            self.controller_params['K_p_q'] = T_1_q / (2 * T_sigma_q * V_S_q)  # prop. gain i_sq controller

            # specify max values for normalisation and anti wind up
            # an anti wind up scheme is necessary for good control behaviour to limit the integral parts in case of
            # limit violations of the desired input voltage
            self.controller_params['omega_max'] = params['omega_N'] * params['safety_margin']
            self.controller_params['u_max'] = params['u_N'] * params['safety_margin']
            self.controller_params['i_max'] = params['i_N'] * params['safety_margin']
            self.controller_params['epsilon_max'] = 2 * np.pi * params['safety_margin']

            # maximum speed without flux weakening
            self.controller_params['omega_1'] = self.controller_params['u_max'] / self.controller_params['L_q']\
                                                /np.sqrt(self.controller_params['i_max']**2
                                                         + self.controller_params['Psi_p']**2
                                                         /self.controller_params['L_q']**2)

    # endregion

    # region continuous DC controller
    def _P_controller(self, state, ref, k):
        """
        Implementation of saturated P controller

        Args:
            state: current measured system state
            ref: reference for the system
            k: current time step

        Returns:
            input action (voltage)
        """
        action = np.array([self.controller_params['K_p'] * (ref - state)])
        action = np.clip(action, self.action_space.low[0], self.action_space.high[0])
        return action

    def _PI_controller(self, state=0, ref=0, k=0):
        """
        Implementation of a simple PI controller with a back step integration, no anti wind up

        Args:
            state: current measured system state
            ref: reference for the system
            k: current time step

        Returns:
            input action (voltage)
        """
        action = self.controller_params['K_p'] * (ref - state) + self.cumulated_values
        if np.any(action > self.action_space.high):
            action = self.action_space.high[0]
        elif np.any(action < self.action_space.low):
            action = self.action_space.low[0]
        self.cumulated_values += self.controller_params['K_i'] * self.controller_params['tau'] * (ref - state)
        return np.array([min(max(action, self.action_space.low[0]), self.action_space.high[0])])

    def _cascaded_PI_controller(self, state, ref, k):
        """
        cascaded controller with two PI controllers for inner current and outer speed control for DC motors
        Args:
            :param state: [omega, i_a, i_e]  # use [omega, i, i] for series
            :param ref: reference
            :param k: current time step

        Returns:
            :return: normalised input voltage
        """
        OMEGA_IDX = 0
        i_a_IDX = 1
        i_e_IDX = 2

        # denormalize quantities
        omega = state[OMEGA_IDX] * self.controller_params['omega_max']
        omega_ref = ref * self.controller_params['omega_max']
        i_a = state[i_a_IDX] * self._i_a_max
        psi_e = state[i_e_IDX] * self._i_e_max_prime

        # outer control loop
        d_omega = omega_ref-omega
        if psi_e != 0:
            temp = self.cumulated_values[0] + d_omega * self.controller_params['K_i_o'] / psi_e  # integral part
            i_a_des = temp + d_omega * self.controller_params['K_p_o'] / psi_e
        else:
            i_a_des = np.sign(d_omega) * self._i_a_max
            temp = self.cumulated_values[0]

        # hold current constraints, anti wind-up
        if i_a_des > self._i_a_max or i_a_des < self._i_a_min:
            i_a_des = np.clip(i_a_des, self._i_a_min, self._i_a_max)
        else:
            self.cumulated_values[0] = temp

        d_i_a = i_a_des - i_a
        # inner control loop
        temp = self.cumulated_values[1] + d_i_a * self.controller_params['K_i_i']  # integral part
        d_u_a = temp + d_i_a * self.controller_params['K_p_i']
        u_a_0 = omega * psi_e
        u_a = d_u_a + u_a_0

        # hold voltage limits, anti wind-up
        if u_a > self._u_a_max or u_a < self._u_a_min:
            u_a = np.clip(u_a, self._u_a_min, self._u_a_max)
        else:
            self.cumulated_values[1] = temp

        # normalise the desired output voltage to a duty cycle referring to the supply voltage
        # Assumption: u_sup = u_N is made
        des_duty_cycle = u_a / self.controller_params['u_sup']
        duty_cycle = np.clip(des_duty_cycle, self.controller_params['converter_voltage'][0],
                             self.controller_params['converter_voltage'][1])
        return np.array([duty_cycle])

    def _cascaded_PI_controller_Permex(self, state, ref, k, *args):
        """
        cascaded controller with two PI controllers for current and speed control for PermEx motor
        Args:
            :param state: current system state
            :param ref: reference
            :param k: current time step

        Returns:
            :return: normalised input voltage
        """
        OMEGA_IDX = 0
        i_IDX = 1

        # denormalize quantities
        omega = state[OMEGA_IDX] * self.controller_params['omega_max']
        omega_ref = ref * self.controller_params['omega_max']
        i_a = state[i_IDX] * self._i_a_max

        # outer control loop
        d_omega = omega_ref-omega
        temp = self.cumulated_values[0] + d_omega * self.controller_params['K_i_o'] / self.psi_e  # integral part
        i_a_des = temp + d_omega * self.controller_params['K_p_o'] / self.psi_e

        # hold current constraints, anti wind-up
        if i_a_des > self._i_a_max or i_a_des < self._i_a_min:
            i_a_des = np.clip(i_a_des, self._i_a_min, self._i_a_max)
        else:
            self.cumulated_values[0] = temp

        d_i_a = i_a_des - i_a

        # inner control loop
        temp = self.cumulated_values[1] + d_i_a * self.controller_params['K_i_i']  # integral part
        d_u_a = temp + d_i_a * self.controller_params['K_p_i']
        u_a_0 = omega * self.psi_e
        u_a = d_u_a + u_a_0
        # hold voltage limits, anti wind-up
        if u_a > self._u_a_max or u_a < self._u_a_min:
            u_a = np.clip(u_a, self._u_a_min, self._u_a_max)
        else:
            self.cumulated_values[1] = temp

        # normalise the desired output voltage to a duty cycle referring to the supply voltage
        # Assumption: u_sup = u_N is made
        des_duty_cycle = u_a / self.controller_params['u_sup']
        duty_cycle = np.clip(des_duty_cycle, self.controller_params['converter_voltage'][0],
                             self.controller_params['converter_voltage'][1])
        return np.array([duty_cycle])

    # endregion

    # region discrete DC controller

    def _on_off(self, state, ref, k):
        """
        On or Off controller depending on the current state and the reference
        Args:
            :param state: current measured system state
            :param ref: reference for the system
            :param k: current time step

        Returns:
            :return: input action (voltage)
        """
        action = 1 if state < ref else 0  # Hint: modified else branch to 0 or 2 for some converters
        return action

    def _three_point(self, state, ref, *_):
        """
        Implementation of a hysteresis controller

        Args:
            :param state: current measured system state
            :param ref: reference for the system
            :param k: current time step

        Returns:
            :return: input action (voltage)
        """
        action = 1 if state - ref < -self.controller_params['hysteresis']\
            else 2 if state - ref > self.controller_params['hysteresis'] \
            else 0
        return action
    # endregion

    # region discrete PMSM controller
    def _pmsm_hysteresis(self, state, ref, k):
        """
        Hysteresis controller for PMSM with feed forward control for u_d

        :param state: state/observation from the motor
        :param ref: current reference value
        :param k: current time step

        :return: switching command for the converter
        """

        # indizes in the observation array
        OMEGA_IDX = 0
        I_A_IDX = 2
        I_B_IDX = 3
        I_C_IDX = 4
        EPSILON_IDX = 8
        CURRENTS = [I_A_IDX, I_B_IDX, I_C_IDX]

        # denormalization
        omega = state[OMEGA_IDX] * self.controller_params['omega_max']
        i = state[CURRENTS] * self.controller_params['i_max']
        epsilon = state[EPSILON_IDX] * self.controller_params['epsilon_max']

        # transformation to dq-coordinates
        i_dq = PmsmModel.q_inv(PmsmModel.t_23(i), epsilon)

        # feed forward control
        u_d_0 = omega * self.controller_params['L_q'] * i_dq[1] / self.controller_params['u_N']

        # hysteresis control
        state = state[OMEGA_IDX]
        if state < ref - self.controller_params['hysteresis']:
            u_q = 1
        elif state > ref + self.controller_params['hysteresis']:
            u_q = -1
        else:
            u_q = 0
        # transformation back to abc-coordinates
        u_a, u_b, u_c = PmsmModel.t_32(PmsmModel.q((u_d_0, u_q), epsilon))
        return 4 * (u_a > 0) + 2 * (u_b > 0) + (u_c > 0)

    def _pmsm_on(self, state, ref, k, *args):
        """
        On or Off controller for the PMSM

        :param state: state/observation from the motor
        :param ref: current reference value
        :param k: current time step
        :param args: additional arguments as the angle epsilon
        :return:
        """
        if ref > state[0]:
            u_q = 1
            u_d = 0
        else:
            u_q = -1
            u_d = 0
        u_a, u_b, u_c = PmsmModel.t_32(PmsmModel.q((u_d, u_q), args[0] * self.controller_params['safety_margin']))
        return 4 * (u_a > 0) + 2 * (u_b > 0) + (u_c > 0)
    # endregion

    # region continuous PMSM controller
    def _foc_controller(self, state, ref, k, *args):
        """
        Field oriented control from the lecture "controlled three phase drives, chapter 5"
        Args:
            state: current system state
            ref: references
            k: current time steps
            args: not used in this function

        Returns:
            normalised input voltages
        """

        weight = 1  # weight for maximum values for the anti-wind-up from abc-values to dq-values

        # indices in the state array
        OMEGA_IDX = 0
        I_A_IDX = 2
        I_B_IDX = 3
        I_C_IDX = 4
        U_A_IDX = 5
        U_B_IDX = 6
        U_C_IDX = 7
        EPSILON_IDX = 8
        CURRENTS = [I_A_IDX, I_B_IDX, I_C_IDX]
        VOLTAGES = [U_A_IDX, U_B_IDX, U_C_IDX]

        # extract quantities from state
        omega = state[OMEGA_IDX] * self.controller_params['omega_max']
        omega_ref = ref * self.controller_params['omega_max']
        i = state[CURRENTS] * self.controller_params['i_max']
        u = state[VOLTAGES] * self.controller_params['u_max']
        epsilon = state[EPSILON_IDX] * self.controller_params['epsilon_max'] * self.controller_params['p']

        # transformation from a/b/c to alpha/beta and d/q
        u_alphabeta = PmsmModel.t_23(u)
        i_alphabeta = PmsmModel.t_23(i)

        u_dq = PmsmModel.q_inv(u_alphabeta, epsilon)
        i_dq = PmsmModel.q_inv(i_alphabeta, epsilon)

        # compute u_d_0 and u_q_0
        u_d_0 = omega * self.controller_params['L_q'] * i_dq[1]
        u_q_0 = omega * (self.controller_params['Psi_p'] + self.controller_params['L_d'] * i_dq[0])
        d_omega = omega_ref - omega

        # compute T* (Torque reference) and i*_sq (q-axis current reference)
        temp = self.cumulated_values[0] + d_omega * self.controller_params['K_i_T']  # integral part
        T_des = temp + d_omega * self.controller_params['K_p_T']  # proportional part
        i_sq_des = 2 * T_des / (3 * self.controller_params['p'] * self.controller_params['Psi_p'])
        # anti wind-up
        if i_sq_des > self.controller_params['i_max'] * weight or i_sq_des < -self.controller_params['i_max'] * weight:
            i_sq_des = np.clip(i_sq_des, -self.controller_params['i_max'] * weight,
                               self.controller_params['i_max'] * weight)
        else:
            self.cumulated_values[0] = temp

        if np.abs(omega_ref) < self.controller_params['omega_1']:
            i_sd_des = 0
        else:
            i_sd_des = ((self.controller_params['u_max'] / omega_ref)**2 -
                        (self.controller_params['L_q'] * self.controller_params['i_max'])**2
                        - self.controller_params['Psi_p']**2)\
                       / (2 * self.controller_params['Psi_p'] * self.controller_params['L_d'])

        # transform back to abc-domain
        currents = np.matmul(PmsmModel.t32, PmsmModel.q(np.array([i_sd_des, i_sq_des]), epsilon))

        # test if current limits are violated
        if np.max(np.abs(currents)) > self.controller_params['i_max']:
            clipping = self.controller_params['i_max'] * np.ones(3)
            currents = np.clip(currents, -clipping, clipping)
            array = PmsmModel.q_inv(PmsmModel.t_23(currents), epsilon)
            i_sd_des = array[0]
            i_sq_des = array[1]

         # compute du*_sq, du*_sd
        d_i_sd = i_sd_des - i_dq[0]
        d_i_sq = i_sq_des - i_dq[1]
        temp_u_sd = self.cumulated_values[1] + d_i_sd * self.controller_params['K_i_d']  # integral part
        temp_u_sq = self.cumulated_values[2] + d_i_sq * self.controller_params['K_i_q']  # integral part
        d_u_sd_des = temp_u_sd + d_i_sd * self.controller_params['K_p_d']
        d_u_sq_des = temp_u_sq + d_i_sq * self.controller_params['K_p_q']
        # anti-wind-up u_sd
        if d_u_sd_des > self.controller_params['u_max'] * weight - u_d_0 or \
                d_u_sd_des < -self.controller_params['u_max'] * weight - u_d_0:
            d_u_sd_des = np.clip(d_u_sd_des, -self.controller_params['u_max'] * weight - u_d_0,
                                 self.controller_params['u_max'] * weight - u_d_0)
        else:
            self.cumulated_values[1] = temp_u_sd
        # anti-wind-up u_sq
        if d_u_sq_des > self.controller_params['u_max'] * weight - u_q_0 or \
                d_u_sq_des < -self.controller_params['u_max'] * weight - u_q_0:
            d_u_sq_des = np.clip(d_u_sq_des, -self.controller_params['u_max'] * weight - u_q_0,
                                 self.controller_params['u_max'] * weight - u_q_0)
        else:
            self.cumulated_values[2] = temp_u_sq

        # compute u*_sq, u*_sd, epsilon + depsilon due to delay of the controller
        u_sd_des = u_d_0 + d_u_sd_des
        u_sq_des = d_u_sq_des + u_q_0
        epsilon_shift = epsilon + 3 / 2 * self.controller_params['tau'] * omega * self.controller_params['p']

        # from d/q to alpha/beta and a/b/c
        u_dq_des = np.array([u_sd_des, u_sq_des])
        voltages = np.matmul(PmsmModel.t32, PmsmModel.q(u_dq_des, epsilon_shift))

        # normalise inputs
        result = np.clip(voltages / self.controller_params['u_max'] * self.controller_params['safety_margin'], -1, 1)
        return result
    # endregion

_select = {
    'PI_controller': (Controller._PI_controller, 0, {'K_i': 10, 'K_p': 15}),
    'P_controller':  (Controller._P_controller, 0, {'K_p': 50}),
    'cascaded_PI_controller': (Controller._cascaded_PI_controller, np.zeros(2), None),  # i...inner, o...outer
    'three_point': (Controller._three_point, 0, {'hysteresis': 0.001}),
    'on_off': (Controller._on_off, 0, {}),
    'pmsm_hysteresis': (Controller._pmsm_hysteresis, 0, {'hysteresis': 0.01}),
    'pmsm_on': (Controller._pmsm_on, 0, {}),
    'foc_controller': (Controller._foc_controller, np.zeros(3), None)
}
