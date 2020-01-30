import numpy as np
from scipy.optimize import root_scalar
from .conf import default_motor_parameter
import warnings


class _DcMotor(object):
    """
        The _DcMotor and its subclasses implement the technical system of a DC motor. \n

        This includes the system equations, the motor parameters of the equivalent circuit diagram,
        as well as limits and bandwidth.
        The state array of the base motor is **[ omega, i_a, i_e ]**.
        It has got two input voltages **[u_a, u_e]**.

    """

    # Indices for array accesses
    OMEGA_IDX = 0
    I_A_IDX = 1
    I_E_IDX = 2
    U_A_IDX = 3
    U_E_IDX = 4

    @property
    def motor_parameter(self):
        return self._motor_parameter

    def __init__(self, load_eq, motor_parameter=None, load_j=0, motor_type='DcExtEx'):
        """
        Basic setting of all the common motor parameters.

        Args:
            load_eq: Load equation of the load model
            motor_parameter(dict or int): Motor parameter set. This can either be an integer or a dictionary.
                                    If an integer is passed the motor parameter file from the conf.py is loaded.
                                    If a dict is passed this will become the parameter file.

        """
        motor_param = motor_parameter if motor_parameter is not None else 0
        try:
            if type(motor_param) is dict:
                self._motor_parameter = motor_param
            elif type(motor_param) is int:
                self._motor_parameter = default_motor_parameter[motor_type][motor_param]
            else:
                raise TypeError
        except TypeError:
            warnings.warn('Invalid Type for motor parameter, fell back to default set 0')
            self._motor_parameter = default_motor_parameter[0]
        except IndexError:
            warnings.warn('Invalid Index for motor parameter, fell back to default set 0')
            self._motor_parameter = default_motor_parameter[0]

        self._model_constants = None
        # :Matrix that contains the constant parameters of the systems jacobian for faster computation
        self._jac_constants = None
        # :The last input voltage at the motor
        self.u_in = 0.0
        # Write parameters to local variables for speedup
        self.u_sup = self.motor_parameter['u_sup']
        if 'l_e_prime' in self.motor_parameter:
            self.l_e_prime = self.motor_parameter['l_e_prime']
        if 'psi_e' in self.motor_parameter:
            self.psi_e = self.motor_parameter['psi_e']
        self._load_j = load_j
        self._update_model()
        self.u_in = 0.0
        self.u_sup = self.motor_parameter['u_sup']
        # Calculate maximum possible values for the state variables
        self._update_limits(load_eq)

    def induced_voltage(self, state):
        """
        The induced voltage of the armature circuit.

        Args:
            state: The current state array of the motor

        Returns:
            The voltage induced by the armature in Volt
        """
        return self.l_e_prime * state[self.I_E_IDX] * state[self.OMEGA_IDX]

    def update_params(self, motor_parameter):
        """
        Changes motor parameters and changes directly the internally model parameters, too.

        Args:
            motor_parameter: Motor parameters that shall be changed
        """
        self._motor_parameter.update(motor_parameter)
        self.l_e_prime = self._motor_parameter['l_e_prime']
        if self._motor_parameter['torque_N'] == 0:
            self._motor_parameter['torque_N'] = self.torque(
                [self._motor_parameter['omega_N'], self._motor_parameter['i_a_N'], self._motor_parameter['i_e_N']])
        self._update_model()

    def _update_model(self):
        """
        Update the motors model parameters with the motor parameters.

        Called internally when the motor parameters are changed or the motor is initialized.
        """
        self.l_e_prime = self.motor_parameter['l_e_prime']
        self._model_constants = np.array([
            [0, 0, self.motor_parameter['l_e_prime'], 0, 0, 0, -1],
            [-self.motor_parameter['r_a'], 0, 0, -self.motor_parameter['l_e_prime'], 1, 0, 0],
            [0, -self.motor_parameter['r_e'], 0, 0, 0, 1, 0]
        ])
        self._model_constants[self.OMEGA_IDX] /= (self._motor_parameter['j'] + self._load_j)
        self._model_constants[self.I_A_IDX] /= self._motor_parameter['l_a']
        self._model_constants[self.I_E_IDX] /= self._motor_parameter['l_e']
        self._jac_constants = np.array([
            [self._motor_parameter['l_e_prime'], self._motor_parameter['l_e_prime'], -1],
            [-self._motor_parameter['r_a'], -self._motor_parameter['l_e_prime'], - self._motor_parameter['l_e_prime']],
            [0, -self._motor_parameter['r_e'], 0]
        ])
        self._jac_constants[self.OMEGA_IDX] /= (self._motor_parameter['j'] + self._load_j)
        self._jac_constants[self.I_A_IDX] /= self._motor_parameter['l_a']
        self._jac_constants[self.I_E_IDX] /= self._motor_parameter['l_e']

    def torque(self, state):
        """
        The torque equation of the motor.

        Args:
            state: The current state array of the motor

        Returns:
            The current torque of the motor based on the state.
        """
        return self.l_e_prime * state[self.I_A_IDX] * state[self.I_E_IDX]

    def torque_max(self, omega):
        """
        The speed-torque equation of the motor.

        Calculate the maximum possible torque for a given omega for nominal currents.

        Args:
            omega: The speed of the motor in rad/s
        Returns:
            The maximal possible torque in Nm for the given speed
        """
        return min(self.u_sup**2 / (4 * self.motor_parameter['r_a'] * omega),  self.motor_parameter['torque_N'])

    def i_a_max(self, omega):
        """
        The maximum possible current through the armature circuit for a given speed omega for nominal supply voltage.

        Args:
            omega: The speed of the motor in rad/s

        Returns:
            The maximum possible current i_a in Ampere for the given speed
        """
        return max(0, min(self.motor_parameter['i_a_N'], self.u_sup *
                   (1 - self.l_e_prime * omega / self.motor_parameter['r_e']) / self.motor_parameter['r_a']))

    def i_in(self, state):
        """
        The current flowing into the motor in Amperes.

        Args:
            state: The current state array of the motor

        Returns:
            The current flowing into the motor.
        """
        return state[self.I_A_IDX], state[self.I_E_IDX]

    def model(self, state, t_load, u_in):
        """
        The differential system equation.

        Args:
            state: The state array of the motor.
            t_load: The load calculated by the load model
            u_in: The input voltages at the terminals [u_a, u_e]

        Returns:
            The derivative of the state variables omega, i_a, i_e
        """
        self.u_in = u_in
        return np.matmul(self._model_constants, np.array([
            state[self.I_A_IDX],
            state[self.I_E_IDX],
            state[self.I_A_IDX] * state[self.I_E_IDX],
            state[self.OMEGA_IDX] * state[self.I_E_IDX],
            u_in[0],
            u_in[1],
            t_load
        ]))

    def jac(self, state, dtorque_domega):
        """
        The jacobian of the model.

        Args:
            state: The state values of the motor
            dtorque_domega: The derivative of the load by omega. Calculated by the load model
        Returns:
            The jacobian matrix of the current motor state
        """
        return self._jac_constants * np.array([
            [state[self.I_E_IDX], state[self.I_A_IDX], dtorque_domega],
            [1, state[self.OMEGA_IDX], state[self.I_E_IDX]],
            [0, 1, 0]
        ])

    def bandwidth(self):
        """
        Calculate the bandwidth of the circuits based on the motor parameters

        Returns:
            Tuple of the armature circuit and excitation circuit bandwidth
        """
        return (self.motor_parameter['r_a'] / self.motor_parameter['l_a'],
                self.motor_parameter['r_e'] / self.motor_parameter['l_e'])

    def _update_limits(self, load_fct):
        """
        Calculate for all the missing maximal values the physical maximal possible values considering nominal voltages.

        Args:
            load_fct: load function of the mechanical model. Must be of type load_fct(omega)
        """
        if self._motor_parameter['i_a_N'] == 0.0:
            self._motor_parameter['i_a_N'] = self._motor_parameter['u_a_N'] / self._motor_parameter['r_a']
        if self.motor_parameter['i_e_N'] == 0.0:
            self.motor_parameter['i_e_N'] = self.motor_parameter['u_e_N'] / self.motor_parameter['r_e']
        if self.motor_parameter['torque_N'] == 0.0:
            self.motor_parameter['torque_N'] = self.torque(
                [0, self.motor_parameter['i_a_N'], self.motor_parameter['i_e_N']]
            )

        # If the torque(omega) will never reach zero set omega max to the value when omega_dot equals 0.001 * omega
        try:
            max_omega = root_scalar(
                lambda omega: self.torque_max(omega) - load_fct(omega),
                bracket=[1e-4, 100000.0]).root
        except ValueError:
            max_omega = root_scalar(
                lambda omega: self.torque_max(omega) - load_fct(omega)
                              / (self.motor_parameter['J_rotor'] + self._load_j) - 0.001 * omega,
                bracket=[1e-4, 100000.0]).root

        if self._motor_parameter['omega_N'] == 0:
            self._motor_parameter['omega_N'] = max_omega
        else:
            self.motor_parameter['omega_N'] = min(max_omega, self.motor_parameter['omega_N'])


class DcShuntMotor(_DcMotor):
    """
    Class that models a DC shunt motor.
    The state array of this motor is: [ omega, torque, i_a, i_e ].
    It has got one input voltage: u_in
    """
    def __init__(self, load_fct, motor_parameter=None, load_j=0):
        super().__init__(load_fct, motor_parameter, load_j, motor_type='DcShunt')
        self.u_in = 0
        if self.motor_parameter['torque_N'] == 0:
            self.motor_parameter['torque_N'] = self.torque(np.array([self.motor_parameter['omega_N'],
                                                                     self.motor_parameter['i_a_N'],
                                                                     self.motor_parameter['i_e_N']
                                                                     ]))

    def bandwidth(self):
        return max(super().bandwidth())

    def torque(self, state):
        return super().torque(state)

    def i_in(self, state):
        return state[self.I_A_IDX] + state[self.I_E_IDX]

    def model(self, state, t_load, u_in):
        dots = super().model(state, t_load, (u_in, u_in))
        self.u_in = u_in
        return dots

    def jac(self, state, dtorque_domega):
        return super().jac(state, dtorque_domega)

    def torque_max(self, omega):
        """
        The speed-torque equation of the motor.

        Calculate the maximum possible torque for a given omega for nominal currents.

        Args:
            omega: The speed of the motor in rad/s
        Returns:
            The maximal possible torque in Nm for the given speed
        """
        return min(self.l_e_prime * self.u_sup**2
                   * (
                       (1 - self.l_e_prime / self.motor_parameter['r_e'] * omega)
                       / (self.motor_parameter['r_a'] + self.motor_parameter['r_e'])
                   ),
                   self.motor_parameter['torque_N'])

    def _update_limits(self, load_fct):
        """
        Calculate for all the missing maximal values the physical maximal possible values considering nominal voltages.

        Args:
            load_fct: load function of the mechanical model. Must be of type load_fct(omega)
        """
        if self.motor_parameter['i_a_N'] == 0.0:
            self.motor_parameter['i_a_N'] = self.motor_parameter['u_N'] / self.motor_parameter['r_a']
        if self.motor_parameter['i_e_N'] == 0.0:
            self.motor_parameter['i_e_N'] = self.motor_parameter['u_N'] / self.motor_parameter['r_e']
        if self.motor_parameter['torque_N'] == 0.0:
            self.motor_parameter['torque_N'] = self.torque(
                [0, self.motor_parameter['i_a_N'], self.motor_parameter['i_e_N']]
            )

        # If the torque(omega) will never reach zero set omega max to the value when omega_dot equals 0.001 * omega
        try:
            max_omega = root_scalar(
                lambda omega: self.torque_max(omega) - load_fct(omega),
                bracket=[1e-4, 100000.0]).root
        except ValueError:
            max_omega = root_scalar(
                lambda omega: (self.torque_max(omega) - load_fct(omega))
                              / (self.motor_parameter['J_rotor'] + self._load_j) - 0.001 * omega,
                bracket=[1e-4, 100000.0]).root
        if self.motor_parameter['omega_N'] == 0:
            self.motor_parameter['omega_N'] = max_omega
        else:
            self.motor_parameter['omega_N'] = min(max_omega, self.motor_parameter['omega_N'])


class DcSeriesMotor(_DcMotor):
    """
    Class that models a DC series motor.
    The state array of this motor is: [ omega, torque, i ].
    It has got one input voltage: u_in
    """
    I_IDX = 1

    def __init__(self, load_fct, motor_parameter=None, load_j=0):
        super().__init__(load_fct, motor_parameter, load_j, motor_type='DcSeries')
        self.u_in = 0
        self._update_model()
        if self.motor_parameter['torque_N'] == 0:
            self.motor_parameter['torque_N'] = self.torque(np.array([self.motor_parameter['omega_N'],
                                                                     self.motor_parameter['i_N'],
                                                                     self.motor_parameter['i_N']
                                                                     ]))

    def induced_voltage(self, state):
        return self.l_e_prime * state[self.I_IDX] * state[self.OMEGA_IDX]

    def update_params(self, motor_params):
        self._motor_parameter.update(motor_params)
        self.l_e_prime = self.motor_parameter['l_e_prime']
        if self._motor_parameter['torque_N'] == 0:
            self._motor_parameter['torque_N'] = self.torque(
                [self._motor_parameter['omega_N'], self._motor_parameter['i_N'], self._motor_parameter['i_N']]
            )
        self._update_model()

    def _update_model(self):
        self._model_constants = np.array([
            [0, 0, self.motor_parameter['l_e_prime'], 0, -1],
            [-self.motor_parameter['r_a'] - self.motor_parameter['r_e'], -self.motor_parameter['l_e_prime'], 0, 1, 0]
        ])
        self._model_constants[self.OMEGA_IDX] /= (self._motor_parameter['j'] + self._load_j)
        self._model_constants[self.I_IDX] /= (self._motor_parameter['l_a'] + self._motor_parameter['l_e'])

        self._jac_constants = np.array([
            [2 * self._motor_parameter['l_e_prime'], -1],
            [-self._motor_parameter['l_e_prime'] - self._motor_parameter['r_a'] - self._motor_parameter['r_e'],
             - self._motor_parameter['l_e_prime']]
        ])
        self._jac_constants[self.OMEGA_IDX] /= (self._motor_parameter['j'] + self._load_j)
        self._jac_constants[self.I_IDX] /= (self._motor_parameter['l_a'] + self._motor_parameter['l_e'])

    def torque(self, state):
        return super().torque([state[self.OMEGA_IDX], state[self.I_IDX], state[self.I_IDX]])

    def torque_max(self, omega):
        return min(
            self.l_e_prime
            * (self.u_sup / (self.motor_parameter['r_a'] + self.motor_parameter['r_e'] + self.l_e_prime * omega))**2,
            self.motor_parameter['torque_N'])

    def i_a_max(self, omega):
        return self.u_sup / (self.motor_parameter['r_a'] + self.motor_parameter['r_e'] + self.l_e_prime * omega)

    def model(self, state, t_load, u_in):
        self.u_in = u_in
        return np.matmul(self._model_constants, np.array([
            state[self.I_IDX],
            state[self.OMEGA_IDX] * state[self.I_IDX],
            state[self.I_IDX] ** 2,
            u_in,
            t_load
        ]))

    def i_in(self, state):
        return state[self.I_IDX]

    def jac(self, state, load_jac):
        return self._jac_constants * np.array([
            [1, load_jac],
            [state[self.OMEGA_IDX], state[self.I_IDX]]
        ])

    def bandwidth(self):
        return ((self.motor_parameter['r_a'] + self.motor_parameter['r_e'])
                / (self.motor_parameter['l_a'] + self.motor_parameter['l_e']))

    def _update_limits(self, load_fct):
        """
        Calculate for all the missing maximal values the physical maximal possible values considering nominal voltages.

        Args:
            load_fct: load function of the mechanical model. Must be of type load_fct(omega)
        """
        if self.motor_parameter['i_N'] == 0.0:
            self.motor_parameter['i_N'] = \
                self.motor_parameter['u_N'] / (self.motor_parameter['r_a'] + self.motor_parameter['r_e'])
        if self.motor_parameter['torque_N'] == 0.0:
            self.motor_parameter['torque_N'] = self.torque([0, self.motor_parameter['i_N']])

        # If the torque(omega) will never reach zero set omega max to the value when omega_dot equals 0.001 * omega
        try:
            max_omega = root_scalar(
                lambda omega: self.torque_max(omega) - load_fct(omega),
                bracket=[1e-4, 100000.0]).root
        except ValueError:
            max_omega = root_scalar(
                lambda omega: (self.torque_max(omega) - load_fct(omega))
                              / (self.motor_parameter['j'] + self._load_j) - 0.001 * omega,
                bracket=[1e-4, 100000.0]).root
        if self._motor_parameter['omega_N'] == 0:
            self._motor_parameter['omega_N'] = max_omega
        else:
            self.motor_parameter['omega_N'] = min(max_omega, self.motor_parameter['omega_N'])


class DcExternallyExcited(_DcMotor):
    """
    The externally excited motor is basically the same as the base motor
    """
    def __init__(self, load_fct, motor_params=None, load_j=0):
        super().__init__(load_fct, motor_params, load_j, motor_type='DcExtEx')
        self.motor_parameter['torque_N'] = self.torque(np.array([self.motor_parameter['omega_N'],
                                                                 self.motor_parameter['i_a_N'],
                                                                 self.motor_parameter['i_e_N']
                                                                 ]))


class DcPermanentlyExcited(_DcMotor):
    """
    Class that models a permanently excited DC motor.
    The state array of this motor is: [ omega, torque, i ].
    It has got one input voltage: u_in
    """
    I_IDX = 1

    def __init__(self, load_fct, motor_parameter=None, load_j=0):
        super().__init__(load_fct, motor_parameter, load_j, motor_type='DcPermEx')
        self.u_in = 0.0

    def induced_voltage(self, state):
        return self.psi_e * state[self.OMEGA_IDX]

    def torque(self, state):
        return self.psi_e * state[self.I_IDX]

    def torque_max(self, omega):
        return min(
            self.motor_parameter['torque_N'],
            self.psi_e * (self.u_sup - self.psi_e * omega) / self.motor_parameter['r_a']
        )

    def i_a_max(self, omega):
        return min(self.motor_parameter['i_N'], (self.u_sup - self.psi_e * omega) / self.motor_parameter['r_a'])

    def update_params(self, motor_params):
        self._motor_parameter.update(motor_params)
        self.l_e_prime = self.motor_parameter['l_e_prime']
        if self._motor_parameter['torque_N'] == 0:
            self._motor_parameter['torque_N'] = self.torque(
                [self._motor_parameter['omega_N'], self._motor_parameter['i_N']])
        self._update_model()

    def _update_model(self):
        self._model_constants = np.array([
            [0.0, self.motor_parameter['psi_e'], -1.0, 0.0],
            [-self.motor_parameter['psi_e'], -self.motor_parameter['r_a'], 0.0, 1.0]
        ])
        self._model_constants[self.OMEGA_IDX] /= (self._motor_parameter['j'] + self._load_j)
        self._model_constants[self.I_IDX] /= self._motor_parameter['l_a']

        self._jac_constants = np.array([
            [-1.0, self.motor_parameter['psi_e']],
            [-self.motor_parameter['psi_e'], -self.motor_parameter['r_a']]
        ])
        self._jac_constants[self.OMEGA_IDX] /= (self._motor_parameter['j'] + self._load_j)
        self._jac_constants[self.I_IDX] /= self._motor_parameter['l_a']

    def i_in(self, state):
        return state[self.I_IDX]

    def model(self, state, t_load, u_in):
        self.u_in = u_in
        return np.matmul(self._model_constants, np.array([state[self.OMEGA_IDX], state[self.I_IDX], t_load, u_in]))

    def jac(self, state, dtorque_domega):
        """
        The Jacobian equation of the motor.

        Args:
            state: the current motor state
            dtorque_domega: the derivative of the load function over omega

        Returns:
            The solution for the Jacobian equation for the Motor at the given state
        """
        return self._jac_constants * np.array([[1.0, dtorque_domega],
                                               [1.0, 1.0]])

    def bandwidth(self):
        return self.motor_parameter['r_a'] / self.motor_parameter['l_a']

    def _update_limits(self, load_fct):
        """
        Calculate for all the missing maximal values the physical maximal possible values considering nominal voltages.

        Args:
            load_fct: load function of the mechanical model. Must be of type load_fct(omega)
        """
        if self.motor_parameter['i_N'] == 0.0:
            self.motor_parameter['i_N'] = self.motor_parameter['u_N'] / self.motor_parameter['r_a']
        if self.motor_parameter['torque_N'] == 0.0:
            self.motor_parameter['torque_N'] = self.torque([0, self.motor_parameter['i_N']])

            self.motor_parameter['omega_N'] = root_scalar(
                lambda omega: self.motor_parameter['psi_e'] * self.motor_parameter['i_N']
                              * (self.motor_parameter['u_N'] - self.motor_parameter['psi_e'] * omega)
                              / self.motor_parameter['r_a'] - load_fct(omega),
                bracket=(1e-4, 10000.0)).root
        # If the torque(omega) will never reach zero set omega max to the value when omega_dot equals 0.001 * omega
        try:
            max_omega = root_scalar(
                lambda omega: self.torque_max(omega) - load_fct(omega),
                bracket=[1e-4, 100000.0]).root
        except ValueError:
            max_omega = root_scalar(
                lambda omega: (self.torque_max(omega) - load_fct(omega))
                / (self.motor_parameter['J_rotor'] + self._load_j) - 0.001 * omega,
                bracket=[1e-4, 100000.0]).root
        if self._motor_parameter['omega_N'] == 0:
            self._motor_parameter['omega_N'] = max_omega
        else:
            self.motor_parameter['omega_N'] = min(max_omega, self.motor_parameter['omega_N'])


def make(model='Series', load_fct=lambda omega: 0.0, motor_parameter=None, load_j=0):
    """
    Dc Motor Factory function.

    Args:
        model: Define the Dc Motor Type ['Shunt', 'Series', 'ExtEx', 'PermEx']
        load_fct: The load function of the load model
        motor_parameter: Motor parameters to update
        load_j: Loads Moment of inertia in kgm²

    Returns:
        An instantiated Dc Motor
    """
    models = ['DcShunt', 'DcSeries', 'DcExtEx', 'DcPermEx']
    assert model in models, "No Model: " + model + "\n Must be in: " + str(models)
    typ = {
        'DcSeries': DcSeriesMotor,
        'DcShunt': DcShuntMotor,
        'DcExtEx': DcExternallyExcited,
        'DcPermEx': DcPermanentlyExcited
    }[model]
    return typ(load_fct, motor_parameter, load_j)
