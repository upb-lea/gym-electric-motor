import numpy as np
from scipy.optimize import root_scalar
from .conf import default_motor_parameter
import warnings
import math


class PmsmModel:
    """
        Technical Model of a Permanent Magnet Synchronous Motor (PMSM).

        Description:
            This class contains the technical properties of a PMSM.
            This includes the differential equation (model), the motor parameters like inductances and resistances,
            the torque equation, the motor bandwidth and its limits.
            The motor will be represented in dq-coordinates.

        Source:

        State Variables:
            +-----+--------+---------------------------------------------------+
            |Index| State  | Description                                       |
            +=====+========+===================================================+
            |0    | omega  | mechanical angular velocity of the motor in rad/s |
            +-----+--------+---------------------------------------------------+
            |1    | i_q    | current flowing into branch a                     |
            +-----+--------+---------------------------------------------------+
            |2    | i_d    | current flowing into branch b                     |
            +-----+--------+---------------------------------------------------+
            |3    | epsilon| rotation angle of the rotor                       |
            +-----+--------+---------------------------------------------------+
    """
    OMEGA_IDX = 0
    I_Q_IDX = 1
    I_D_IDX = 2
    EPSILON_IDX = 3

    # transformation matrix from abc to alpha-beta representation
    t23 = np.array([2/3]) * np.array([
        [1, -0.5,            -0.5],
        [0, 0.5 * np.sqrt(3), -0.5 * np.sqrt(3)]
    ])

    # transformation matrix from alpha-beta to abc representation
    t32 = np.array([
        [1, 0],
        [-0.5, 0.5 * np.sqrt(3)],
        [-0.5, -0.5 * np.sqrt(3)]
    ])

    @property
    def motor_parameter(self):
        return self._motor_parameter

    def __init__(self, motor_parameter=None, load_fct=lambda omega: 0.01, load_j=0):
        """
        Initialize the motor.  Set its parameters, determine constants, initialize the load model

        Args:
            motor_parameter: motor parameters
            load_fct:  load function
            load_j: jacobian of the load function
        """
        # Set Motor param to motor parameter, if they are not None otherwise to 0
        motor_param = motor_parameter if motor_parameter is not None else 0
        try:
            if type(motor_param) is dict:
                self._motor_parameter = motor_param
            elif type(motor_param) is int:
                self._motor_parameter = default_motor_parameter[motor_param]
            else:
                raise TypeError
        except TypeError:
            warnings.warn('Invalid Type for motor parameter, fell back to default set 0')
            self._motor_parameter = default_motor_parameter[0]
        except IndexError:
            warnings.warn('Invalid Index for motor parameter, fell back to default set 0')
            self._motor_parameter = default_motor_parameter[0]
        # Set operating region borders
        mp = self.motor_parameter
        # Solve Quadratic equation over omega_el to find border of first operation region
        a = mp['L_q'] ** 2 * mp['i_N'] ** 2 + mp['Psi_p'] ** 2
        b = 2 * mp['R_s'] * mp['i_N'] * mp['Psi_p']
        c = mp['R_s'] ** 2 * mp['i_N'] ** 2 - mp['u_N'] ** 2
        self._omega_1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        # For the second
        i_0 = mp['Psi_p'] / mp['L_d']
        self._k = i_0 / mp['i_N']
        if self._k < 1:
            i_d_2 = -i_0
            i_q_2 = np.sqrt(mp['i_N']**2-i_d_2**2)
            u_q_2 = mp['R_s'] * i_q_2
            u_d_2 = np.sqrt(mp['u_N']**2 - u_q_2**2)
            self._omega_2 = (mp['R_s'] * i_d_2 - u_d_2) / (mp['L_q'] * i_q_2)
        else:
            self._omega_2 = np.inf
        self._model_constants = None
        self._jac_constants = None
        self._load_j = load_j
        self.u_sup = self.motor_parameter['u_sup']
        self._update_model()
        self._update_limits(load_fct)

    @staticmethod
    def t_23(quantities):
        """
        Transformation of PMSM-quantities like currents or voltages from abc representation to alpha-beta representation

        Args:
            quantities: The properties in the abc representation like ''[u_a, u_b, u_c]''

        Returns:
            The converted quantities in the alpha-beta representation like ''[u_alpha, u_beta]''
        """
        return np.matmul(PmsmModel.t23, quantities)

    @staticmethod
    def t_32(quantities):
        """
        Transformation of PMSM-quantities like currents or voltages from alpha-beta representation to abc representation

        Args:
            quantities: The properties in the alpha-beta representation like ''[u_alpha, u_beta]''

        Returns:
            The converted quantities in the abc representation like ''[u_a, u_b, u_c]''
        """
        return np.matmul(PmsmModel.t32, quantities)

    @staticmethod
    def q(quantities, epsilon):
        """
        Transformation of the dq-representation into alpha-beta using the electrical angle

        Args:
            quantities: Array of two quantities in dq-representation. Example [i_d, i_q]
            epsilon: Current electrical angle of the motor

        Returns:
            Array of the two quantities converted to alpha-beta-representation. Example [u_alpha, u_beta]
        """
        cos = math.cos(epsilon)
        sin = math.sin(epsilon)
        return cos * quantities[0] - sin * quantities[1], sin * quantities[0] + cos * quantities[1]

    @staticmethod
    def q_inv(quantities, epsilon):
        """
        Transformation of the alpha-beta-representation into dq using the elctrical angle

        Args:
            quantities: Array of two quantities in alpha-beta-representation. Example [u_alpha, u_beta]
            epsilon: Current electrical angle of the motor

        Returns:
            Array of the two quantities converted to dq-representation. Example [u_d, u_q]

        Note:
            The transformation from alpha-beta to dq is just its inverse conversion with negated epsilon.
            So this method calls q(quantities, -epsilon).
        """
        return PmsmModel.q(quantities, -epsilon)

    def q_me(self, quantities, epsilon):
        """
        Transformation of the dq-representation into alpha-beta using the mechanical angle

        Args:
            quantities: Array of two quantities in dq-representation. Example [i_d, i_q]
            epsilon: Current mechanical angle of the motor

        Returns:
            Array of the two quantities converted to alpha-beta-representation. Example [u_alpha, u_beta]
        """
        return self.q(quantities, epsilon * self._motor_parameter['p'])

    def q_inv_me(self, quantities, epsilon):
        """
        Transformation of the alpha-beta-representation into dq using the mechanical angle

        Args:
            quantities: Array of two quantities in alpha-beta-representation. Example [u_alpha, u_beta]
            epsilon: Current mechanical angle of the motor

        Returns:
            Array of the two quantities converted to dq-representation. Example [u_d, u_q]

        Note:
            The transformation from alpha-beta to dq is just its inverse conversion with negated epsilon.
            So this method calls q(quantities, -epsilon).
        """
        return self.q_me(quantities, -epsilon)

    def bandwidth(self):
        """
        Calculation of the motors q and d components bandwidth.
        This is especially important for the reference generation in the environment.

        Returns:
            The bandwidth of the q and the d component of the motor.
        """
        return (self._motor_parameter['R_s'] / self._motor_parameter['L_q'],
                self._motor_parameter['R_s'] / self._motor_parameter['L_d'])

    def _update_model(self):
        mp = self._motor_parameter
        self._model_constants = np.array([
            # omega                  i_q                          i_d         u_q u_d omega*i_q             omega*i_d       load_fct
            [0,                      1.5 * mp['p'] * mp['Psi_p'], 0,           0, 0,  0,                    0,                   -1],
            [-mp['Psi_p'] * mp['p'], -mp['R_s'],                  0,           1, 0,  0,                    -mp['L_d'] * mp['p'], 0],
            [0,                      0,                           -mp['R_s'],  0, 1,  mp['L_q'] * mp['p'],  0,                    0],
            [1,                      0,                           0,           0, 0,  0,                    0,                    0]
        ])
        self._model_constants[self.OMEGA_IDX] /= (mp['J_rotor'] + self._load_j)
        self._model_constants[self.I_Q_IDX] /= mp['L_q']
        self._model_constants[self.I_D_IDX] /= mp['L_d']

    def _update_limits(self, load_fct):
        """
        Update the maximal parameters for the motor.

        If a maximum value of a motor state is not given in advance(state_N == 0), then it will be calculated to its
        maximum physical possible value.

        Args:
            load_fct: The load function of the system with the signature load_fct(self, omega)

        """
        mp = self._motor_parameter
        if mp['i_N'] == 0:
            self._motor_parameter['i_N'] = mp['u_N'] / mp['R_s']
        if mp['torque_N'] == 0:
            self._motor_parameter['torque_N'] = 1.5 * mp['p'] * mp['Psi_p'] * self._motor_parameter['i_N']
        # If the torque(omega/p) will never reach zero set omega max to the value when omega_dot equals 0.001 * omega
        try:
            omega_max = root_scalar(
                lambda omega: self.max_torque(omega) - load_fct(omega),
                bracket=[1e-4, 100000.0]
            ).root
        except ValueError:
            omega_max = root_scalar(
                lambda omega: (self.max_torque(omega)-load_fct(omega)) / (mp['J_rotor'] + self._load_j) - 0.001 * omega,
                bracket=[1e-4, 100000.0]
            ).root

        if self._motor_parameter['omega_N'] == 0:
            self._motor_parameter['omega_N'] = omega_max
        else:
            self._motor_parameter['omega_N'] = min(omega_max, self._motor_parameter['omega_N'])

    def max_torque(self, omega):
        """
        Calculate the maximum possible torque for a given speed omega.

        Args:
            omega: angular velocity

        Returns:
            Maximum possible torque at angular velocity omega.
        """
        mp = self._motor_parameter
        sign = 1 if omega >= 0 else -1
        omega = sign * omega
        i_q_max, i_d_max = self.i_max(omega)
        return sign * self.torque([omega, i_q_max, i_d_max, 0])

    def i_max(self, omega):
        mp = self.motor_parameter
        omega_el = omega * mp['p']

        # Determine the operating region
        if mp['u_N'] < mp['R_s'] * mp['i_N']:
            # Motor cannot exceed current limits due to voltage limitations
            return mp['u_N'] / mp['R_s'], 0
        # Equation is solvable with omega > 0 if u_n > R_s * i_N
        elif omega_el < self._omega_1:
            # Motor works at current limit, but no flux weakening current is required
            return mp['i_N'], 0
        elif self._k > 1 or omega_el < self._omega_2:
            # Motor works at current and voltage limits
            i_q_max = (
                np.sin(np.arctan2(mp['R_s'], -  omega_el * mp['L_d'])) * mp['u_N']
                + omega_el * mp['L_d'] / mp['R_s'] * np.cos(np.arctan2(mp['R_s'], -omega_el * mp['L_d']))
                - omega_el * mp['Psi_p']
            ) / (mp['R_s'] + omega_el**2 * mp['L_d'] * mp['L_q'] / mp['R_s'])
            i_d_max = np.sqrt(mp['i_N']**2-i_q_max**2)
            return i_q_max, i_d_max
        else:
            a = mp['R_s']**2 + omega_el**2 * mp['L_q']**2
            b = 2 * mp['R_s'] * mp['Psi_p'] * omega_el * mp['L_q'] / mp['L_d']
            c = mp['R_s']**2 * mp['Psi_p']**2 / mp['L_d']**2 - mp['u_N']**2
            return (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a), -mp['Psi_p'] / mp['L_d']

    def torque(self, state):
        """
        Calculate the torque generated at a given state.

        Args:
            state: State array of the Motor in the form ''[omega, i_q, i_d, epsilon]''

        Returns:
            The active torque for the given state.

        """
        mp = self._motor_parameter
        return 1.5 * mp['p'] * (mp['Psi_p'] + (mp['L_d'] - mp['L_q']) * state[self.I_D_IDX]) * state[self.I_Q_IDX]

    def model(self, state, t_load, u_qd):
        """
        The differential equation of the PMSM.

        Args:
            state: The current state of the motor. [omega, i_q, i_d, epsilon]
            t_load: The mechanical load
            u_qd: The input voltages [u_q, u_d]

        Returns:
            The derivatives of the state vector d/dt([omega, i_a, i_d, epsilon])
        """
        return np.matmul(self._model_constants, np.array([
            # omega   i_q       i_d       u_q      u_d      omega * i_q          omega * i_d          load_fct
            state[self.OMEGA_IDX],
            state[self.I_Q_IDX],
            state[self.I_D_IDX],
            u_qd[0],
            u_qd[1],
            state[self.OMEGA_IDX] * state[self.I_Q_IDX],
            state[self.OMEGA_IDX] * state[self.I_D_IDX],
            t_load
        ]))

    def jac(self, state, dt_load_dt, u_in):
        """
        Not implemented yet, maybe obsolete.
        Calculate the Jacobian matrix for the system at given state, t_load and u_in.

        Args:
            state: state of the electric motor
            dt_load_dt: derivative of the load equation
            u_in: input voltage

        Returns:
            Jacobian matrix for the system at given state, t_load and u_in.
        """
        raise NotImplementedError
