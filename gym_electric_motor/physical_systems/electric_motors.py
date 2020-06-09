import numpy as np
import math


class ElectricMotor:
    """
        Base class for all technical electrical motor models.

        A motor consists of the ode-state. These are the dynamic quantities of its ODE.
        For example:
            ODE-State of a DC-shunt motor: `` [i_a, i_e ] ``
                * i_a: Anchor circuit current
                * i_e: Exciting circuit current

        Each electric motor can be parametrized by a dictionary of motor parameters, the nominal state dictionary
        and the limit dictionary.
    """

    #: Parameter indicating if the class is implementing the optional jacobian function
    HAS_JACOBIAN = False

    #: CURRENTS_IDX(list(int)): Indices for accessing all motor currents.
    CURRENTS_IDX = []

    #: CURRENTS(list(str)): List of the motor currents names
    CURRENTS = []
    #: VOLTAGES(list(str)): List of the motor input voltages names
    VOLTAGES = []

    #: _default_motor_parameter(dict): Default parameter dictionary for the motor
    _default_motor_parameter = {}
    #: _default_nominal_values(dict(float)): Default nominal motor state array
    _default_nominal_values = {}
    #: _default_limits(dict(float)): Default motor limits (0 for unbounded limits)
    _default_limits = {}

    @property
    def nominal_values(self):
        """
        Readonly motors nominal values.

        Returns:
            dict(float): Current nominal values of the motor.
        """
        return self._nominal_values

    @property
    def limits(self):
        """
        Readonly motors limit state array. Entries are set to the maximum physical possible values
        in case of unspecified limits.

        Returns:
            dict(float): Limits of the motor.
        """
        return self._limits

    @property
    def motor_parameter(self):
        """
        Returns:
             dict(float): The motors parameter dictionary
        """
        return self._motor_parameter

    def __init__(self, motor_parameter=None, nominal_values=None, limit_values=None, **__):
        """
        :param  motor_parameter: Motor parameter dictionary. Contents specified for each motor.
        :param nominal_values: Nominal values for the motor quantities.
        :param limit_values: Limits for the motor quantities.
        """
        motor_parameter = motor_parameter or {}
        self._motor_parameter = self._default_motor_parameter.copy()
        self._motor_parameter.update(motor_parameter)
        limit_values = limit_values or {}
        self._limits = self._default_limits.copy()
        self._limits.update(limit_values)
        nominal_values = nominal_values or {}
        self._nominal_values = self._default_nominal_values.copy()
        self._nominal_values.update(nominal_values)

    def electrical_ode(self, state, u_in, omega, *_):
        """
        Calculation of the derivatives of each motor state variable for the given inputs / The motors ODE-System.

        Args:
            state(ndarray(float)): The motors state.
            u_in(list(float)): The motors input voltages.
            omega(float): Angular velocity of the motor

        Returns:
             ndarray(float): Derivatives of the motors ODE-system for the given inputs.
        """
        raise NotImplementedError

    def electrical_jacobian(self, state, u_in, omega, *_):
        """
        Calculation of the jacobian of each motor ODE for the given inputs / The motors ODE-System.

        Overriding this method is optional for each subclass. If it is overridden, the parameter HAS_JACOBIAN must also
        be set to True. Otherwise, the jacobian will not be called.

        Args:
            state(ndarray(float)): The motors state.
            u_in(list(float)): The motors input voltages.
            omega(float): Angular velocity of the motor

        Returns:
             Tuple(ndarray, ndarray, ndarray):
                [0]: Derivatives of all electrical motor states over all electrical motor states shape:(states x states)
                [1]: Derivatives of all electrical motor states over omega shape:(states,)
                [2]: Derivative of Torque over all motor states shape:(states,)
        """
        pass

    def torque(self, currents):
        """
        Torque equation of the motor.

        Args:
            currents(numpy.ndarray(float)): Motor currents to calculate the Torque.

        Returns:
            float: Motor torque for the given state.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the motors state to a new initial state. (Default 0)

        Returns:
            numpy.ndarray(float): The initial motors state.
        """
        return np.zeros(len(self.CURRENTS), dtype=float)

    def i_in(self, state):
        """
        Args:
            state(ndarray(float)): ODE state of the motor

        Returns:
             list(float): List of all currents flowing into the motor.
        """
        raise NotImplementedError

    def _update_limits(self, limits_d={}, nominal_d={}):
        """Replace missing limits and nominal values with physical maximums.

        Args:
            limits_d(dict): Mapping: quantitity to its limit if not specified
        """

        # omega is replaced the same way for all motor types
        limits_d.update(dict(omega=self._default_limits['omega']))

        for qty, lim in limits_d.items():
            if self._limits.get(qty, 0) == 0:
                self._limits[qty] = lim

        for entry in self._limits.keys():
            if self._nominal_values.get(entry, 0) == 0:
                self._nominal_values[entry] = nominal_d.get(entry, None) or \
                                              self._limits[entry]


class DcMotor(ElectricMotor):
    """
        The DcMotor and its subclasses implement the technical system of a dc motor.

        This includes the system equations, the motor parameters of the equivalent circuit diagram,
        as well as limits.

        =====================  ==========  ============= ===========================================
        Motor Parameter        Unit        Default Value Description
        =====================  ==========  ============= ===========================================
        r_a                    Ohm         0.78          Armature circuit resistance
        r_e                    Ohm         25            Exciting circuit resistance
        l_a                    H           6.3e-3        Armature circuit inductance
        l_e                    H           1.2           Exciting circuit inductance
        l_e_prime              H           0.0094        Effective excitation inductance
        j_rotor                kg/m^2      0.017         Moment of inertia of the rotor
        =====================  ==========  ============= ===========================================

        =============== ====== =============================================
        Motor Currents  Unit   Description
        =============== ====== =============================================
        i_a             A      Armature circuit current
        i_e             A      Exciting circuit current
        =============== ====== =============================================
        =============== ====== =============================================
        Motor Voltages  Unit   Description
        =============== ====== =============================================
        u_a             V      Armature circuit voltage
        u_e             v      Exciting circuit voltage
        =============== ====== =============================================

        ======== ===========================================================
        Limits / Nominal Value Dictionary Entries:
        -------- -----------------------------------------------------------
        Entry    Description
        ======== ===========================================================
        i_a      Armature current
        i_e      Exciting current
        omega    Angular Velocity
        torque   Motor generated torque
        u_a      Armature Voltage
        u_e      Exciting Voltage
        ======== ===========================================================
    """

    # Indices for array accesses
    I_A_IDX = 0
    I_E_IDX = 1
    CURRENTS_IDX = [0, 1]
    CURRENTS = ['i_a', 'i_e']
    VOLTAGES = ['u_a', 'u_e']
    _default_motor_parameter = {
        'r_a': 0.78, 'r_e': 25, 'l_a': 6.3e-3, 'l_e': 1.2, 'l_e_prime': 0.0094, 'j_rotor': 0.017,
    }
    _default_nominal_values = {'omega': 368, 'torque': 0.0, 'i_a': 50, 'i_e': 1.2, 'u': 420}
    _default_limits = {'omega': 500, 'torque': 0.0, 'i_a': 75, 'i_e': 2, 'u': 420}

    def __init__(self, motor_parameter=None, nominal_values=None, limit_values=None, **__):
        # Docstring of superclass
        super().__init__(motor_parameter, nominal_values, limit_values)
        #: Matrix that contains the constant parameters of the systems equation for faster computation
        self._model_constants = None
        self._update_model()
        self._update_limits()

    def _update_model(self):
        """
        Update the motors model parameters with the motor parameters.

        Called internally when the motor parameters are changed or the motor is initialized.
        """
        mp = self._motor_parameter
        self._model_constants = np.array([
            [-mp['r_a'], 0, -mp['l_e_prime'], 1, 0],
            [0, -mp['r_e'], 0, 0, 1]
        ])
        self._model_constants[self.I_A_IDX] = self._model_constants[self.I_A_IDX] / mp['l_a']
        self._model_constants[self.I_E_IDX] = self._model_constants[self.I_E_IDX] / mp['l_e']

    def torque(self, currents):
        # Docstring of superclass
        return self._motor_parameter['l_e_prime'] * currents[self.I_A_IDX] * currents[self.I_E_IDX]

    def i_in(self, currents):
        # Docstring of superclass
        return list(currents)

    def electrical_ode(self, state, u_in, omega, *_):
        # Docstring of superclass
        return np.matmul(self._model_constants, np.array([
            state[self.I_A_IDX],
            state[self.I_E_IDX],
            omega * state[self.I_E_IDX],
            u_in[0],
            u_in[1],
        ]))

    def get_state_space(self, input_currents, input_voltages):
        """
        Calculate the possible normalized state space for the motor as a tuple of dictionaries "low" and "high".

        Args:
            input_currents: Tuple of the two converters possible output currents.
            input_voltages: Tuple of the two converters possible output voltages.

        Returns:
             tuple(dict,dict): Dictionaries defining if positive and negative values are possible for each motors state.
        """
        a_converter = 0
        e_converter = 1
        low = {
            'omega': -1 if input_voltages.low[a_converter] == -1
            or input_voltages.low[e_converter] == -1 else 0,
            'torque': -1 if input_currents.low[a_converter] == -1
            or input_currents.low[e_converter] == -1 else 0,
            'i_a': -1 if input_currents.low[a_converter] == -1 else 0,
            'i_e': -1 if input_currents.low[e_converter] == -1 else 0,
            'u_a': -1 if input_voltages.low[a_converter] == -1 else 0,
            'u_e': -1 if input_voltages.low[e_converter] == -1 else 0,
        }
        high = {
            'omega': 1,
            'torque': 1,
            'i_a': 1,
            'i_e': 1,
            'u_a': 1,
            'u_e': 1
        }
        return low, high

    def _update_limits(self, limits_d={}):
        # Docstring of superclass

        # torque is replaced the same way for all DC motors
        limits_d.update(dict(torque=self.torque([self._limits[state] for state
                                                 in self.CURRENTS])))
        super()._update_limits(limits_d)


class DcShuntMotor(DcMotor):
    """
        The DcShuntMotor is a DC motor with parallel armature and exciting circuit connected to one input voltage.

        =====================  ==========  ============= ===========================================
        Motor Parameter        Unit        Default Value Description
        =====================  ==========  ============= ===========================================
        r_a                    Ohm         0.78          Armature circuit resistance
        r_e                    Ohm         25            Exciting circuit resistance
        l_a                    H           6.3e-3        Armature circuit inductance
        l_e                    H           1.2           Exciting circuit inductance
        l_e_prime              H           0.0094        Effective excitation inductance
        j_rotor                kg/m^2      0.017         Moment of inertia of the rotor
        =====================  ==========  ============= ===========================================

        =============== ====== =============================================
        Motor Currents  Unit   Description
        =============== ====== =============================================
        i_a             A      Armature circuit current
        i_e             A      Exciting circuit current
        =============== ====== =============================================
        =============== ====== =============================================
        Motor Voltages  Unit   Description
        =============== ====== =============================================
        u               V      Voltage applied to both circuits
        =============== ====== =============================================

        ======== ===========================================================
        Limits / Nominal Value Dictionary Entries:
        -------- -----------------------------------------------------------
        Entry    Description
        ======== ===========================================================
        i_a      Armature current
        i_e      Exciting current
        omega    Angular Velocity
        torque   Motor generated torque
        u        Voltage
        ======== ===========================================================
    """
    HAS_JACOBIAN = True
    VOLTAGES = ['u']

    _default_nominal_values = {'omega': 368, 'torque': 0.0, 'i_a': 50, 'i_e': 1.2, 'u': 420}
    _default_limits = {'omega': 500, 'torque': 0.0, 'i_a': 75, 'i_e': 2, 'u': 420}

    def i_in(self, state):
        # Docstring of superclass
        return [state[self.I_A_IDX] + state[self.I_E_IDX]]

    def electrical_ode(self, state, u_in, omega, *_):
        # Docstring of superclass
        return super().electrical_ode(state, (u_in[0], u_in[0]), omega)

    def electrical_jacobian(self, state, u_in, omega, *_):
        mp = self._motor_parameter
        return (
            np.array([
                [-mp['r_a'] / mp['l_a'], -mp['l_e_prime'] / mp['l_a'] * omega],
                [0, -mp['r_e'] / mp['l_e']]
            ]),
            np.array([-mp['l_e_prime'] * state[self.I_E_IDX] / mp['l_a'], 0]),
            np.array([mp['l_e_prime'] * state[self.I_E_IDX], mp['l_e_prime'] * state[self.I_A_IDX]])
        )

    def get_state_space(self, input_currents, input_voltages):
        """
        Calculate the possible normalized state space for the motor as a tuple of dictionaries "low" and "high".

        Args:
            input_currents: The converters possible output currents.
            input_voltages: The converters possible output voltages.

        Returns:
             tuple(dict,dict): Dictionaries defining if positive and negative values are possible for each motors state.
        """
        lower_limit = 0

        low = {
            'omega': 0,
            'torque': -1 if input_currents.low[0] == -1 else 0,
            'i_a': -1 if input_currents.low[0] == -1 else 0,
            'i_e': -1 if input_currents.low[0] == -1 else 0,
            'u': -1 if input_voltages.low[0] == -1 else 0,
        }
        high = {
            'omega': 1,
            'torque': 1,
            'i_a': 1,
            'i_e': 1,
            'u': 1,
        }
        return low, high

    def _update_limits(self):
        # Docstring of superclass

        # R_a might be 0, protect against that
        r_a = 1 if self._motor_parameter['r_a'] == 0 else self._motor_parameter['r_a']

        limit_agenda = \
            {'u': self._default_limits['u'],
             'i_a': self._limits.get('i', None) or
                    self._limits['u'] / r_a,
             'i_e': self._limits.get('i', None) or
                    self._limits['u'] / self.motor_parameter['r_e'],
             }

        super()._update_limits(limit_agenda)


class DcSeriesMotor(DcMotor):
    """
        The DcSeriesMotor is a DcMotor with an armature and exciting circuit connected in series to one input voltage.

        =====================  ==========  ============= ===========================================
        Motor Parameter        Unit        Default Value Description
        =====================  ==========  ============= ===========================================
        r_a                    Ohm         2.78          Armature circuit resistance
        r_e                    Ohm         1.0           Exciting circuit resistance
        l_a                    H           6.3e-3        Armature circuit inductance
        l_e                    H           1.6e-3        Exciting circuit inductance
        l_e_prime              H           0.05          Effective excitation inductance
        j_rotor                kg/m^2      0.017         Moment of inertia of the rotor
        =====================  ==========  ============= ===========================================

        =============== ====== =============================================
        Motor Currents  Unit   Description
        =============== ====== =============================================
        i               A      Circuit current
        =============== ====== =============================================
        =============== ====== =============================================
        Motor Voltages  Unit   Description
        =============== ====== =============================================
        u               V      Circuit voltage
        =============== ====== =============================================

        ======== ===========================================================
        Limits / Nominal Value Dictionary Entries:
        -------- -----------------------------------------------------------
        Entry    Description
        ======== ===========================================================
        i        Circuit Current
        omega    Angular Velocity
        torque   Motor generated torque
        u        Circuit Voltage
        ======== ===========================================================
    """
    HAS_JACOBIAN = True
    I_IDX = 0
    CURRENTS_IDX = [0]
    CURRENTS = ['i']
    VOLTAGES = ['u']

    _default_motor_parameter = {
        'r_a': 2.78, 'r_e': 1.0, 'l_a': 6.3e-3, 'l_e': 1.6e-3,
        'l_e_prime': 0.05, 'j_rotor': 0.017,
    }
    _default_nominal_values = dict(omega=80, torque=0.0, i=50, u=420)
    _default_limits = dict(omega=100, torque=0.0, i=100, u=420)

    def _update_model(self):
        # Docstring of superclass
        mp = self._motor_parameter
        self._model_constants = np.array([
            [-mp['r_a'] - mp['r_e'], -mp['l_e_prime'], 1]
        ])
        self._model_constants[self.I_IDX] = self._model_constants[self.I_IDX] / (mp['l_a'] + mp['l_e'])

    def torque(self, currents):
        # Docstring of superclass
        return super().torque([currents[self.I_IDX], currents[self.I_IDX]])

    def electrical_ode(self, state, u_in, omega, *_):
        # Docstring of superclass
        return np.matmul(
            self._model_constants,
            np.array([
                state[self.I_IDX],
                omega * state[self.I_IDX],
                u_in[0]
            ])
        )

    def i_in(self, state):
        # Docstring of superclass
        return state[self.CURRENTS_IDX]

    def _update_limits(self):
        # Docstring of superclass

        # R_a might be 0, protect against that
        r_a = 1 if self._motor_parameter['r_a'] == 0 else self._motor_parameter['r_a']
        limits_agenda = {
            'u': self._default_limits['u'],
            'i': self._limits['u'] / (r_a + self._motor_parameter['r_e']),
        }
        super()._update_limits(limits_agenda)

    def get_state_space(self, input_currents, input_voltages):
        # Docstring of superclass
        lower_limit = 0
        low = {
            'omega': 0,
            'torque': 0,
            'i': -1 if input_currents.low[0] == -1 else 0,
            'u': -1 if input_voltages.low[0] == -1 else 0,
        }
        high = {
            'omega': 1,
            'torque': 1,
            'i': 1,
            'u': 1,
        }
        return low, high

    def electrical_jacobian(self, state, u_in, omega, *_):
        mp = self._motor_parameter
        return (
            np.array([[-(mp['r_a'] + mp['r_e'] + mp['l_e_prime'] * omega) / (mp['l_a'] + mp['l_e'])]]),
            np.array([-mp['l_e_prime'] * state[self.I_IDX] / (mp['l_a'] + mp['l_e'])]),
            np.array([2 * mp['l_e_prime'] * state[self.I_IDX]])
        )


class DcPermanentlyExcitedMotor(DcMotor):
    """
        The DcPermanentlyExcitedMotor is a DcMotor with a Permanent Magnet instead of the excitation circuit.

        =====================  ==========  ============= ===========================================
        Motor Parameter        Unit        Default Value Description
        =====================  ==========  ============= ===========================================
        r_a                    Ohm         25.0          Armature circuit resistance
        l_a                    H           3.438e-2      Armature circuit inductance
        psi_e                  Wb          18            Magnetic Flux of the permanent magnet
        j_rotor                kg/m^2      0.017         Moment of inertia of the rotor
        =====================  ==========  ============= ===========================================
        =============== ====== =============================================
        Motor Currents  Unit   Description
        =============== ====== =============================================
        i               A      Circuit current
        =============== ====== =============================================
        =============== ====== =============================================
        Motor Voltages  Unit   Description
        =============== ====== =============================================
        u               V      Circuit voltage
        =============== ====== =============================================

        ======== ===========================================================
        Limits / Nominal Value Dictionary Entries:
        -------- -----------------------------------------------------------
        Entry    Description
        ======== ===========================================================
        i        Circuit Current
        omega    Angular Velocity
        torque   Motor generated torque
        u        Circuit Voltage
        ======== ===========================================================
    """
    I_IDX = 0
    CURRENTS_IDX = [0]
    CURRENTS = ['i']
    VOLTAGES = ['u']
    HAS_JACOBIAN = True

    _default_motor_parameter = {
        'r_a': 25.0, 'l_a': 3.438e-2, 'psi_e': 18, 'j_rotor': 0.017
    }
    _default_nominal_values = dict(omega=22, torque=0.0, i=16, u=400)
    _default_limits = dict(omega=50, torque=0.0, i=25, u=400)

    # placeholder for omega, currents and u_in
    _ode_placeholder = np.zeros(2 + len(CURRENTS_IDX), dtype=np.float64)

    def torque(self, state):
        # Docstring of superclass
        return self._motor_parameter['psi_e'] * state[self.I_IDX]

    def _update_model(self):
        # Docstring of superclass
        mp = self._motor_parameter
        self._model_constants = np.array([
            [-mp['psi_e'], -mp['r_a'], 1.0]
        ])
        self._model_constants[self.I_IDX] /= mp['l_a']

    def i_in(self, state):
        # Docstring of superclass
        return state[self.CURRENTS_IDX]

    def electrical_ode(self, state, u_in, omega, *_):
        # Docstring of superclass
        self._ode_placeholder[:] = [omega] + np.atleast_1d(state[self.I_IDX]).tolist()\
                                   + [u_in[0]]
        return np.matmul(self._model_constants, self._ode_placeholder)

    def electrical_jacobian(self, state, u_in, omega, *_):
        mp = self._motor_parameter
        return (
            np.array([[-mp['r_a'] / mp['l_a']]]),
            np.array([-mp['psi_e'] / mp['l_a']]),
            np.array([mp['psi_e']])
        )

    def _update_limits(self):
        # Docstring of superclass

        # R_a might be 0, protect against that
        r_a = 1 if self._motor_parameter['r_a'] == 0 else self._motor_parameter['r_a']

        limits_agenda = {
            'u': self._default_limits['u'],
            'i': self._limits['u'] / r_a,
        }
        super()._update_limits(limits_agenda)

    def get_state_space(self, input_currents, input_voltages):
        # Docstring of superclass
        lower_limit = 0
        low = {
            'omega': -1 if input_voltages.low[0] == -1 else 0,
            'torque': -1 if input_currents.low[0] == -1 else 0,
            'i': -1 if input_currents.low[0] == -1 else 0,
            'u': -1 if input_voltages.low[0] == -1 else 0,
        }
        high = {
            'omega': 1,
            'torque': 1,
            'i': 1,
            'u': 1,
        }
        return low, high


class DcExternallyExcitedMotor(DcMotor):
    # Equals DC Base Motor
    HAS_JACOBIAN = True

    def electrical_jacobian(self, state, u_in, omega, *_):
        mp = self._motor_parameter
        return (
            np.array([
                [-mp['r_a'] / mp['l_a'], -mp['l_e_prime'] / mp['l_a'] * omega],
                [0, -mp['r_e'] / mp['l_e']]
            ]),
            np.array([-mp['l_e_prime'] * state[self.I_E_IDX] / mp['l_a'], 0]),
            np.array([mp['l_e_prime'] * state[self.I_E_IDX], mp['l_e_prime'] * state[self.I_A_IDX]])
        )

    def _update_limits(self):
        # Docstring of superclass

        # R_a might be 0, protect against that
        r_a = 1 if self._motor_parameter['r_a'] == 0 else self._motor_parameter['r_a']

        limit_agenda = \
            {'u_a': self._default_limits['u'],
             'u_e': self._default_limits['u'],
             'i_a': self._limits.get('i', None) or
                    self._limits['u'] / r_a,
             'i_e': self._limits.get('i', None) or
                    self._limits['u'] / self.motor_parameter['r_e'],
             }
        super()._update_limits(limit_agenda)


class ThreePhaseMotor(ElectricMotor):

    """
            The ThreePhaseMotor and its subclasses implement the technical system of Three Phase Motors.

            This includes the system equations, the motor parameters of the equivalent circuit diagram,
            as well as limits and bandwidth.
    """
    # transformation matrix from abc to alpha-beta representation
    _t23 = 2 / 3 * np.array([
        [1, -0.5, -0.5],
        [0, 0.5 * np.sqrt(3), -0.5 * np.sqrt(3)]
    ])

    # transformation matrix from alpha-beta to abc representation
    _t32 = np.array([
        [1, 0],
        [-0.5, 0.5 * np.sqrt(3)],
        [-0.5, -0.5 * np.sqrt(3)]
    ])

    @staticmethod
    def t_23(quantities):
        """
        Transformation from abc representation to alpha-beta representation

        Args:
            quantities: The properties in the abc representation like ''[u_a, u_b, u_c]''

        Returns:
            The converted quantities in the alpha-beta representation like ''[u_alpha, u_beta]''
        """
        return np.matmul(ThreePhaseMotor._t23, quantities)

    @staticmethod
    def t_32(quantities):
        """
        Transformation from alpha-beta representation to abc representation

        Args:
            quantities: The properties in the alpha-beta representation like ``[u_alpha, u_beta]``

        Returns:
            The converted quantities in the abc representation like ``[u_a, u_b, u_c]``
        """
        return np.matmul(ThreePhaseMotor._t32, quantities)

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
        Transformation of the alpha-beta-representation into dq using the electrical angle

        Args:
            quantities: Array of two quantities in alpha-beta-representation. Example [u_alpha, u_beta]
            epsilon: Current electrical angle of the motor

        Returns:
            Array of the two quantities converted to dq-representation. Example [u_d, u_q]

        Note:
            The transformation from alpha-beta to dq is just its inverse conversion with negated epsilon.
            So this method calls q(quantities, -epsilon).
        """
        return SynchronousMotor.q(quantities, -epsilon)

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

    def _torque_limit(self):
        """
        Returns:
             Maximal possible torque for the given limits in self._limits
        """
        raise NotImplementedError()

    def _update_limits(self, limits_d={}, nominal_d={}):
        # Docstring of superclass
        super()._update_limits(limits_d, nominal_d)
        super()._update_limits(dict(torque=self._torque_limit()))


class SynchronousMotor(ThreePhaseMotor):
    """
        The SynchronousMotor and its subclasses implement the technical system of a three phase synchronous motor.

        This includes the system equations, the motor parameters of the equivalent circuit diagram,
        as well as limits and bandwidth.

        =====================  ==========  ============= ===========================================
        Motor Parameter        Unit        Default Value Description
        =====================  ==========  ============= ===========================================
        r_s                    Ohm         0.78          Stator resistance
        l_q                    H           6.3e-3        Quadrature axis inductance
        l_d                    H           1.2           Direct axis inductance
        psi_p                  Wb          0.0094        Effective excitation flux (PMSM only)
        p                      1           2             Pole pair number
        j_rotor                kg/m^2      0.017         Moment of inertia of the rotor
        =====================  ==========  ============= ===========================================

        =============== ====== =============================================
        Motor Currents  Unit   Description
        =============== ====== =============================================
        i_sq            A      Quadrature axis current
        i_sd            A      Direct axis current
        i_a             A      Current through branch a
        i_b             A      Current through branch b
        i_c             A      Current through branch c
        i_alpha         A      Current in alpha axis
        i_beta          A      Current in beta axis
        =============== ====== =============================================
        =============== ====== =============================================
        Motor Voltages  Unit   Description
        =============== ====== =============================================
        u_sq            A      Quadrature axis voltage
        u_sd            A      Direct axis voltage
        u_a             A      Voltage through branch a
        u_b             A      Voltage through branch b
        u_c             A      Voltage through branch c
        u_alpha         A      Voltage in alpha axis
        u_beta          A      Voltage in beta axis
        =============== ====== =============================================

        ======== ===========================================================
        Limits / Nominal Value Dictionary Entries:
        -------- -----------------------------------------------------------
        Entry    Description
        ======== ===========================================================
        i        General current limit / nominal value
        i_a      Current in phase a
        i_b      Current in phase b
        i_c      Current in phase c
        i_alpha  Current in alpha axis
        i_beta   Current in beta axis
        i_sd     Current in direct axis
        i_sq     Current in quadrature axis
        omega    Mechanical angular Velocity
        epsilon  Electrical rotational angle
        torque   Motor generated torque
        u_a      Voltage in phase a
        u_b      Voltage in phase b
        u_c      Voltage in phase c
        u_alpha  Voltage in alpha axis
        u_beta   Voltage in beta axis
        u_sd     Voltage in direct axis
        u_sq     Voltage in quadrature axis
        ======== ===========================================================


        Note:
            The voltage limits should be the amplitude of the phase voltage (:math:`\hat{u}_S`).
            Typically the rms value for the line voltage (:math:`U_L`) is given.
            :math:`\hat{u}_S=\sqrt{2/3}~U_L`

            The current limits should be the amplitude of the phase current (:math:`\hat{i}_S`).
            Typically the rms value for the phase current (:math:`I_S`) is given.
            :math:`\hat{i}_S = \sqrt{2}~I_S`

            If not specified, nominal values are equal to their corresponding limit values.
            Furthermore, if specific limits/nominal values (e.g. i_a) are not specified they are inferred from
            the general limits/nominal values (e.g. i)
        """
    I_SQ_IDX = 0
    I_SD_IDX = 1
    EPSILON_IDX = 2
    CURRENTS_IDX = [0, 1]
    CURRENTS = ['i_sq', 'i_sd']
    VOLTAGES = ['u_sq', 'u_sd']

    _model_constants = None

    def __init__(self, motor_parameter=None, nominal_values=None,
                 limit_values=None, **kwargs):
        # Docstring of superclass
        nominal_values = nominal_values or {}
        limit_values = limit_values or {}
        super().__init__(motor_parameter, nominal_values, limit_values)
        self._update_model()
        self._update_limits()

    @property
    def motor_parameter(self):
        # Docstring of superclass
        return self._motor_parameter

    def reset(self):
        # Docstring of superclass
        return np.zeros(len(self.CURRENTS) + 1)

    def torque(self, state):
        # Docstring of superclass
        raise NotImplementedError

    def _update_model(self):
        """
        Set motor parameters into a matrix for faster computation
        """
        raise NotImplementedError

    def electrical_ode(self, state, u_qd, omega, *_):
        """
        The differential equation of the Synchronous Motor.

        Args:
            state: The current state of the motor. [i_sq, i_sd, epsilon]
            omega: The mechanical load
            u_qd: The input voltages [u_sq, u_sd]

        Returns:
            The derivatives of the state vector d/dt([i_sq, i_sd, epsilon])
        """
        return np.matmul(self._model_constants, np.array([
            omega,
            state[self.I_SQ_IDX],
            state[self.I_SD_IDX],
            u_qd[0],
            u_qd[1],
            omega * state[self.I_SQ_IDX],
            omega * state[self.I_SD_IDX],
        ]))

    def i_in(self, state):
        # Docstring of superclass
        return state[self.CURRENTS_IDX]

    def _update_limits(self):
        # Docstring of superclass

        voltage_limit = 0.5 * self._limits['u']
        voltage_nominal = 0.5 * self._nominal_values['u']

        limits_agenda = {}
        nominal_agenda = {}
        for u, i in zip(self.IO_VOLTAGES, self.IO_CURRENTS):
            limits_agenda[u] = voltage_limit
            nominal_agenda[u] = voltage_nominal
            limits_agenda[i] = self._limits.get('i', None) or \
                               self._limits[u] / self._motor_parameter['r_s']
            nominal_agenda[i] = self._nominal_values.get('i', None) or \
                                self._nominal_values[u] / self._motor_parameter['r_s']

        super()._update_limits(limits_agenda, nominal_agenda)


class SynchronousReluctanceMotor(SynchronousMotor):
    """
        =====================  ==========  ============= ===========================================
        Motor Parameter        Unit        Default Value Description
        =====================  ==========  ============= ===========================================
        r_s                    Ohm         0.78          Stator resistance
        l_q                    H           6.3e-3        Quadrature axis inductance
        l_d                    H           1.2           Direct axis inductance
        p                      1           2             Pole pair number
        j_rotor                kg/m^2      0.017         Moment of inertia of the rotor
        =====================  ==========  ============= ===========================================

        =============== ====== =============================================
        Motor Currents  Unit   Description
        =============== ====== =============================================
        i_sq            A      Quadrature axis current
        i_sd            A      Direct axis current
        i_a             A      Current through branch a
        i_b             A      Current through branch b
        i_c             A      Current through branch c
        i_alpha         A      Current in alpha axis
        i_beta          A      Current in beta axis
        =============== ====== =============================================
        =============== ====== =============================================
        Motor Voltages  Unit   Description
        =============== ====== =============================================
        u_sq            V      Quadrature axis voltage
        u_sd            V      Direct axis voltage
        u_a             V      Voltage through branch a
        u_b             V      Voltage through branch b
        u_c             V      Voltage through branch c
        u_alpha         V      Voltage in alpha axis
        u_beta          V      Voltage in beta axis
        =============== ====== =============================================

        ======== ===========================================================
        Limits / Nominal Value Dictionary Entries:
        -------- -----------------------------------------------------------
        Entry    Description
        ======== ===========================================================
        i        General current limit / nominal value
        i_a      Current in phase a
        i_b      Current in phase b
        i_c      Current in phase c
        i_alpha  Current in alpha axis
        i_beta   Current in beta axis
        i_sd     Current in direct axis
        i_sq     Current in quadrature axis
        omega    Mechanical angular Velocity
        epsilon  Electrical rotational angle
        torque   Motor generated torque
        u_a      Voltage in phase a
        u_b      Voltage in phase b
        u_c      Voltage in phase c
        u_alpha  Voltage in alpha axis
        u_beta   Voltage in beta axis
        u_sd     Voltage in direct axis
        u_sq     Voltage in quadrature axis
        ======== ===========================================================


        Note:
            The voltage limits should be the amplitude of the phase voltage (:math:`\hat{u}_S`).
            Typically the rms value for the line voltage (:math:`U_L`) is given.
            :math:`\hat{u}_S=\sqrt{2/3}~U_L`

            The current limits should be the amplitude of the phase current (:math:`\hat{i}_S`).
            Typically the rms value for the phase current (:math:`I_S`) is given.
            :math:`\hat{i}_S = \sqrt{2}~I_S`

            If not specified, nominal values are equal to their corresponding limit values.
            Furthermore, if specific limits/nominal values (e.g. i_a) are not specified they are inferred from
            the general limits/nominal values (e.g. i)

    """
    HAS_JACOBIAN = True
    _default_motor_parameter = {'p': 2, 'l_d': 73.2e-3, 'l_q': 7.3e-3, 'j_rotor': 2.45e-3, 'r_s': 0.3256}
    _default_nominal_values = {
        'i': 54, 'torque': 0, 'omega': 523.0, 'epsilon': np.pi, 'u': 600
    }
    _default_limits = {'i': 70, 'torque': 0, 'omega': 600.0, 'epsilon': np.pi, 'u': 600}

    IO_VOLTAGES = ['u_a', 'u_b', 'u_c', 'u_sd', 'u_sq']
    IO_CURRENTS = ['i_a', 'i_b', 'i_c', 'i_sd', 'i_sq']

    def _update_model(self):
        # Docstring of superclass

        mp = self._motor_parameter
        self._model_constants = np.array([
            # omega, i_sq, i_sd, u_sq, u_sd, omega * i_sq, omega * i_sd
            [0, -mp['r_s'], 0, 1, 0, 0, -mp['l_d'] * mp['p']],
            [0, 0, -mp['r_s'], 0, 1, mp['l_q'] * mp['p'], 0],
            [mp['p'], 0, 0, 0, 0, 0, 0]
        ])
        self._model_constants[self.I_SQ_IDX] = self._model_constants[self.I_SQ_IDX] / mp['l_q']
        self._model_constants[self.I_SD_IDX] = self._model_constants[self.I_SD_IDX] / mp['l_d']

    def _torque_limit(self):
        # Docstring of superclass
        return self.torque([self._limits['i_sq'] / np.sqrt(2), self._limits['i_sd'] / np.sqrt(2), 0])

    def torque(self, currents):
        # Docstring of superclass
        mp = self._motor_parameter
        return 1.5 * mp['p'] * ((mp['l_d'] - mp['l_q']) * currents[self.I_SD_IDX]) * currents[self.I_SQ_IDX]

    def electrical_jacobian(self, state, u_in, omega, *_):
        mp = self._motor_parameter
        return (
            np.array([
                [-mp['r_s'] / mp['l_q'], -mp['l_d']/mp['l_q']*omega, 0],
                [mp['l_q'] / mp['l_d']*omega, - mp['r_s'] / mp['l_d'], 0],
                [0, 0, 0]
            ]),
            np.array([
                - mp['p'] * mp['l_d'] / mp['l_q'] * state[self.I_SD_IDX],
                mp['p'] * mp['l_q'] / mp['l_d'] * state[self.I_SQ_IDX],
                mp['p']
            ]),
            np.array([
                1.5 * mp['p'] * (mp['l_d'] - mp['l_q']) * state[self.I_SD_IDX],
                1.5 * mp['p'] * (mp['l_d'] - mp['l_q']) * state[self.I_SQ_IDX],
                0
            ])
        )


class PermanentMagnetSynchronousMotor(SynchronousMotor):
    """
        =====================  ==========  ============= ===========================================
        Motor Parameter        Unit        Default Value Description
        =====================  ==========  ============= ===========================================
        r_s                    Ohm         0.78          Stator resistance
        l_q                    H           6.3e-3        Quadrature axis inductance
        l_d                    H           1.2           Direct axis inductance
        p                      1           2             Pole pair number
        j_rotor                kg/m^2      0.017         Moment of inertia of the rotor
        =====================  ==========  ============= ===========================================

        =============== ====== =============================================
        Motor Currents  Unit   Description
        =============== ====== =============================================
        i_sq            A      Quadrature axis current
        i_sd            A      Direct axis current
        i_a             A      Current through branch a
        i_b             A      Current through branch b
        i_c             A      Current through branch c
        i_alpha         A      Current in alpha axis
        i_beta          A      Current in beta axis
        =============== ====== =============================================
        =============== ====== =============================================
        Motor Voltages  Unit   Description
        =============== ====== =============================================
        u_sq            V      Quadrature axis voltage
        u_sd            V      Direct axis voltage
        u_a             V      Voltage through branch a
        u_b             V      Voltage through branch b
        u_c             V      Voltage through branch c
        u_alpha         V      Voltage in alpha axis
        u_beta          V      Voltage in beta axis
        =============== ====== =============================================

        ======== ===========================================================
        Limits / Nominal Value Dictionary Entries:
        -------- -----------------------------------------------------------
        Entry    Description
        ======== ===========================================================
        i        General current limit / nominal value
        i_a      Current in phase a
        i_b      Current in phase b
        i_c      Current in phase c
        i_alpha  Current in alpha axis
        i_beta   Current in beta axis
        i_sd     Current in direct axis
        i_sq     Current in quadrature axis
        omega    Mechanical angular Velocity
        torque   Motor generated torque
        epsilon  Electrical rotational angle
        u_a      Voltage in phase a
        u_b      Voltage in phase b
        u_c      Voltage in phase c
        u_alpha  Voltage in alpha axis
        u_beta   Voltage in beta axis
        u_sd     Voltage in direct axis
        u_sq     Voltage in quadrature axis
        ======== ===========================================================


        Note:
            The voltage limits should be the amplitude of the phase voltage (:math:`\hat{u}_S`).
            Typically the rms value for the line voltage (:math:`U_L`) is given.
            :math:`\hat{u}_S=\sqrt{2/3}~U_L`

            The current limits should be the amplitude of the phase current (:math:`\hat{i}_S`).
            Typically the rms value for the phase current (:math:`I_S`) is given.
            :math:`\hat{i}_S = \sqrt{2}~I_S`

            If not specified, nominal values are equal to their corresponding limit values.
            Furthermore, if specific limits/nominal values (e.g. i_a) are not specified they are inferred from
            the general limits/nominal values (e.g. i)
    """
    _default_motor_parameter = {
        'p': 2,
        'l_d': 79e-3,
        'l_q': 113e-3,
        'j_rotor': 2.45e-3,
        'r_s': 4.9,
        'psi_p': 0.165,
    }
    HAS_JACOBIAN = True
    _default_limits = dict(omega=80, torque=0.0, i=20, epsilon=math.pi, u=600)
    _default_nominal_values = dict(omega=75, torque=0.0, i=12, epsilon=math.pi, u=600)

    IO_VOLTAGES = ['u_a', 'u_b', 'u_c', 'u_sd', 'u_sq']
    IO_CURRENTS = ['i_a', 'i_b', 'i_c', 'i_sd', 'i_sq']

    def _update_model(self):
        # Docstring of superclass
        mp = self._motor_parameter
        self._model_constants = np.array([
            # omega,                 i_q,        i_d,        u_q, u_d, omega * i_q,         omega * i_d
            [-mp['psi_p'] * mp['p'], -mp['r_s'], 0,          1,   0,   0,                   -mp['l_d'] * mp['p']],
            [0,                      0,          -mp['r_s'], 0,   1,   mp['l_q'] * mp['p'], 0],
            [mp['p'],                0,          0,          0,   0,   0,                   0]
        ])

        self._model_constants[self.I_SQ_IDX] = self._model_constants[self.I_SQ_IDX] / mp['l_q']
        self._model_constants[self.I_SD_IDX] = self._model_constants[self.I_SD_IDX] / mp['l_d']

    def _torque_limit(self):
        # Docstring of superclass
        return self.torque([self._limits['i_sq'], 0, 0])

    def torque(self, currents):
        # Docstring of superclass
        mp = self._motor_parameter
        return 1.5 * mp['p'] * (mp['psi_p'] + (mp['l_d'] - mp['l_q']) * currents[self.I_SD_IDX])*currents[self.I_SQ_IDX]

    def electrical_jacobian(self, state, u_in, omega, *args):
        mp = self._motor_parameter
        return (
            np.array([ # dx'/dx
                [-mp['r_s'] / mp['l_q'], -mp['l_d']/mp['l_q']*omega, 0],
                [mp['l_q'] / mp['l_d']*omega, -mp['r_s']/mp['l_d'], 0],
                [0, 0, 0]
            ]),
            np.array([ # dx'/dw
                -mp['p'] * mp['l_d'] / mp['l_q'] * state[self.I_SD_IDX] - mp['p'] * mp['psi_p'] / mp['l_q'],
                mp['p'] * mp['l_q'] / mp['l_d'] * state[self.I_SQ_IDX],
                mp['p']
            ]),
            np.array([ # dT/dx
                1.5 * mp['p'] * (mp['psi_p'] + (mp['l_d'] - mp['l_q']) * state[self.I_SD_IDX]),
                1.5 * mp['p'] * (mp['l_d'] - mp['l_q']) * state[self.I_SQ_IDX],
                0
            ])
        )


class InductionMotor(ThreePhaseMotor):
    """
        The InductionMotor and its subclasses implement the technical system of a three phase induction motor.

        This includes the system equations, the motor parameters of the equivalent circuit diagram,
        as well as limits and bandwidth.

        =====================  ==========  ============= ===========================================
        Motor Parameter        Unit        Default Value Description
        =====================  ==========  ============= ===========================================
        r_s                    Ohm         2.9338        Stator resistance
        r_r                    Ohm         1.355         Rotor resistance
        l_m                    H           143.75e-3     Main inductance
        l_sigs                 H           5.87e-3       Stator-side stray inductance
        l_sigr                 H           5.87e-3       Rotor-side stray inductance
        p                      1           2             Pole pair number
        j_rotor                kg/m^2      0.0011        Moment of inertia of the rotor
        =====================  ==========  ============= ===========================================

        =============== ====== =============================================
        Motor Currents  Unit   Description
        =============== ====== =============================================
        i_sq            A      Quadrature axis current
        i_sd            A      Direct axis current
        i_sa            A      Current through branch a
        i_sb            A      Current through branch b
        i_sc            A      Current through branch c
        i_salpha        A      Current in alpha axis
        i_sbeta         A      Current in beta axis
        =============== ====== =============================================
        =============== ====== =============================================
        Motor Voltages  Unit   Description
        =============== ====== =============================================
        u_sq            V      Quadrature axis voltage
        u_sd            V      Direct axis voltage
        u_sa            V      Voltage through branch a
        u_sb            V      Voltage through branch b
        u_sc            V      Voltage through branch c
        u_salpha        V      Voltage in alpha axis
        u_sbeta         V      Voltage in beta axis
        =============== ====== =============================================

        ======== ===========================================================
        Limits / Nominal Value Dictionary Entries:
        -------- -----------------------------------------------------------
        Entry    Description
        ======== ===========================================================
        i        General current limit / nominal value
        i_sa      Current in phase a
        i_sb      Current in phase b
        i_sc      Current in phase c
        i_salpha  Current in alpha axis
        i_sbeta   Current in beta axis
        i_sd     Current in direct axis
        i_sq     Current in quadrature axis
        omega    Mechanical angular Velocity
        torque   Motor generated torque
        u_sa      Voltage in phase a
        u_sb      Voltage in phase b
        u_sc      Voltage in phase c
        u_salpha  Voltage in alpha axis
        u_sbeta   Voltage in beta axis
        u_sd     Voltage in direct axis
        u_sq     Voltage in quadrature axis
        ======== ===========================================================

        Note:
            The voltage limits should be the amplitude of the phase voltage (:math:`\hat{u}_S`).
            Typically the rms value for the line voltage (:math:`U_L`) is given.
            :math:`\hat{u}_S=\sqrt{2/3}~U_L`

            The current limits should be the amplitude of the phase current (:math:`\hat{i}_S`).
            Typically the rms value for the phase current (:math:`I_S`) is given.
            :math:`\hat{i}_S = \sqrt{2}~I_S`

            If not specified, nominal values are equal to their corresponding limit values.
            Furthermore, if specific limits/nominal values (e.g. i_a) are not specified they are inferred from
            the general limits/nominal values (e.g. i)
        """
    I_SALPHA_IDX = 0
    I_SBETA_IDX = 1
    PSI_RALPHA_IDX = 2
    PSI_RBETA_IDX = 3
    EPSILON_IDX = 4

    CURRENTS_IDX = [0, 1]
    FLUX_IDX = [2, 3]
    CURRENTS = ['i_salpha', 'i_sbeta']
    FLUXES = ['psi_ralpha', 'psi_rbeta']
    STATOR_VOLTAGES = ['u_salpha', 'u_sbeta']

    IO_VOLTAGES = ['u_sa', 'u_sb', 'u_sc', 'u_salpha', 'u_sbeta', 'u_sd', 'u_sq']
    IO_CURRENTS = ['i_sa', 'i_sb', 'i_sc', 'i_salpha', 'i_sbeta', 'i_sd', 'i_sq']

    HAS_JACOBIAN = True
    _default_motor_parameter = {
        'p': 2,
        'l_m': 143.75e-3,
        'l_ssig': 5.87e-3,
        'l_rsig': 5.87e-3,
        'j_rotor': 1.1e-3,
        'r_s': 2.9338,
        'r_r': 1.355,
    }

    _default_limits = dict(omega=350, torque=0.0, i=5.5, epsilon=math.pi, u=560)
    _default_nominal_values = dict(omega=314, torque=0.0, i=3.9, epsilon=math.pi, u=560)
    _model_constants = None

    @property
    def motor_parameter(self):
        # Docstring of superclass
        return self._motor_parameter

    def __init__(self, motor_parameter=None, nominal_values=None, limit_values=None, **__):
        # Docstring of superclass

        # convert placeholder i and u to actual IO quantities
        _nominal_values = self._default_nominal_values.copy()
        _nominal_values.update({u: _nominal_values['u'] for u in self.IO_VOLTAGES})
        _nominal_values.update({i: _nominal_values['i'] for i in self.IO_CURRENTS})
        del _nominal_values['u'], _nominal_values['i']
        _nominal_values.update(nominal_values or {})
        # same for limits
        _limit_values = self._default_limits.copy()
        _limit_values.update({u: _limit_values['u'] for u in self.IO_VOLTAGES})
        _limit_values.update({i: _limit_values['i'] for i in self.IO_CURRENTS})
        del _limit_values['u'], _limit_values['i']
        _limit_values.update(limit_values or {})

        super().__init__(motor_parameter, nominal_values, limit_values)
        self._update_model()
        self._update_limits(_limit_values, _nominal_values)

    def reset(self):
        # Docstring of superclass
        return np.zeros(len(self.CURRENTS) + len(self.FLUXES) + 1)

    def electrical_ode(self, state, u_sr_alphabeta, omega, *args):
        """
        The differential equation of the Induction Motor.

        Args:
            state: The current state of the motor. [i_salpha, i_sbeta, psi_ralpha, psi_rbeta, epsilon]
            omega: The mechanical load
            u_sr_alphabeta: The input voltages [u_salpha, u_sbeta, u_ralpha, u_rbeta]

        Returns:
            The derivatives of the state vector d/dt( [i_salpha, i_sbeta, psi_ralpha, psi_rbeta, epsilon])
        """
        return np.matmul(self._model_constants, np.array([
            # omega, i_alpha, i_beta, psi_ralpha, psi_rbeta, omega * psi_ralpha, omega * psi_rbeta, u_salpha, u_sbeta, u_ralpha, u_rbeta,
            omega,
            state[self.I_SALPHA_IDX],
            state[self.I_SBETA_IDX],
            state[self.PSI_RALPHA_IDX],
            state[self.PSI_RBETA_IDX],
            omega * state[self.PSI_RALPHA_IDX],
            omega * state[self.PSI_RBETA_IDX],
            u_sr_alphabeta[0, 0],
            u_sr_alphabeta[0, 1],
            u_sr_alphabeta[1, 0],
            u_sr_alphabeta[1, 1],
        ]))

    def i_in(self, state):
        # Docstring of superclass
        return state[self.CURRENTS_IDX]

    def _torque_limit(self):
        # Docstring of superclass
        mp = self._motor_parameter
        return 1.5 * mp['p'] * mp['l_m'] ** 2/(mp['l_m']+mp['l_rsig']) * self._limits['i_sd'] * self._limits['i_sq'] / 2

    def torque(self, states):
        # Docstring of superclass
        mp = self._motor_parameter
        return 1.5 * mp['p'] * mp['l_m']/(mp['l_m'] + mp['l_rsig']) * (states[self.PSI_RALPHA_IDX] * states[self.I_SBETA_IDX] - states[self.PSI_RBETA_IDX] * states[self.I_SALPHA_IDX])

    def _update_model(self):
        # Docstring of superclass
        mp = self._motor_parameter
        l_s = mp['l_m']+mp['l_ssig']
        l_r = mp['l_m']+mp['l_rsig']
        sigma = (l_s*l_r-mp['l_m']**2) /(l_s*l_r)
        tau_r = l_r / mp['r_r']
        tau_sig = sigma * l_s / (mp['r_s'] + mp['r_r'] * (mp['l_m']**2) / (l_r**2))

        self._model_constants = np.array([
            # omega,  i_alpha,         i_beta,          psi_ralpha,                               psi_rbeta,                              omega * psi_ralpha,                  omega * psi_rbeta,                  u_salpha,        u_sbeta,       u_ralpha,                        u_rbeta,
            [0,       -1/tau_sig,      0,               mp['l_m']*mp['r_r']/(sigma*l_s * l_r**2), 0,                                      0,                                   +mp['l_m']*mp['p']/(sigma*l_r*l_s), 1/(sigma * l_s), 0,             -mp['l_m']/ (sigma * l_r * l_s), 0,                               ],  # i_ralpha_dot
            [0,       0,               -1/tau_sig,      0,                                        mp['l_m']*mp['r_r']/(sigma*l_s*l_r**2), -mp['l_m']*mp['p']/(sigma*l_r*l_s),  0,                                  0,               1/(sigma*l_s),  0,                              -mp['l_m']/ (sigma * l_r * l_s), ],  # i_rbeta_dot
            [0,       mp['l_m']/tau_r, 0,               -1/tau_r,                                 0,                                      0,                                   -mp['p'],                           0,               0,              1,                              0,                               ],  # psi_ralpha_dot
            [0,       0,               mp['l_m']/tau_r, 0,                                        -1/tau_r,                               mp['p'],                             0,                                  0,               0,              0,                              1,                               ],  # psi_rbeta_dot
            [mp['p'], 0,               0,               0,                                        0,                                      0,                                   0,                                  0,               0,              0,                              0,                               ],  # epsilon_dot
        ])

    def electrical_jacobian(self, state, u_in, omega, *args):
        mp = self._motor_parameter
        l_s = mp['l_m'] + mp['l_ssig']
        l_r = mp['l_m'] + mp['l_rsig']
        sigma = (l_s * l_r - mp['l_m'] ** 2) / (l_s * l_r)
        tau_r = l_r / mp['r_r']
        tau_sig = sigma * l_s / (mp['r_s'] + mp['r_r'] * (mp['l_m'] ** 2) / (l_r ** 2))

        return (
            np.array([ # dx'/dx
                # i_alpha          i_beta               psi_alpha                                    psi_beta                                   epsilon
                [-1/tau_sig,        0,                  mp['l_m']*mp['r_r']/(sigma*l_s * l_r**2),    omega * mp['l_m']*mp['p']/(sigma*l_r*l_s), 0],
                [0,                 - 1 / tau_sig,      - omega * mp['l_m']*mp['p']/(sigma*l_r*l_s), mp['l_m']*mp['r_r']/(sigma*l_s * l_r**2),  0],
                [mp['l_m'] / tau_r, 0,                  - 1 / tau_r,                                 - omega * mp['p'],                         0],
                [0,                  mp['l_m'] / tau_r, omega * mp['p'],                             - 1 / tau_r,                               0],
                [0,                 0,                  0,                                           0,                                         0]
            ]),
            np.array([ # dx'/dw
                mp['l_m'] * mp['p'] / (sigma*l_r*l_s) * state[self.PSI_RBETA_IDX],
                - mp['l_m'] * mp['p'] / (sigma*l_r*l_s) * state[self.PSI_RALPHA_IDX],
                - mp['p'] * state[self.PSI_RBETA_IDX],
                mp['p'] * state[self.PSI_RALPHA_IDX],
                mp['p']
            ]),
            np.array([ # dT/dx
                - state[self.PSI_RBETA_IDX] * 3 / 2 * mp['p'] * mp['l_m'] / l_r,
                state[self.PSI_RALPHA_IDX] * 3 / 2 * mp['p'] * mp['l_m'] / l_r,
                state[self.I_SBETA_IDX] * 3 / 2 * mp['p'] * mp['l_m'] / l_r,
                - state[self.I_SALPHA_IDX] * 3 / 2 * mp['p'] * mp['l_m'] / l_r,
                0
            ])
        )


class SquirrelCageInductionMotor(InductionMotor):
    """
        =====================  ==========  ============= ===========================================
        Motor Parameter        Unit        Default Value Description
        =====================  ==========  ============= ===========================================
        r_s                    Ohm         2.9338        Stator resistance
        r_r                    Ohm         1.355         Rotor resistance
        l_m                    H           143.75e-3     Main inductance
        l_ssig                 H           5.87e-3       Stator-side stray inductance
        l_rsig                 H           5.87e-3       Rotor-side stray inductance
        p                      1           2             Pole pair number
        j_rotor                kg/m^2      0.0011        Moment of inertia of the rotor
        =====================  ==========  ============= ===========================================

        =============== ====== =============================================
        Motor Currents  Unit   Description
        =============== ====== =============================================
        i_sq            A      Quadrature axis current
        i_sd            A      Direct axis current
        i_sa            A      Stator current through branch a
        i_sb            A      Stator current through branch b
        i_sc            A      Stator current through branch c
        i_salpha        A      Stator current in alpha direction
        i_sbeta         A      Stator current in beta direction
        =============== ====== =============================================
        =============== ====== =============================================
        Rotor flux      Unit   Description
        =============== ====== =============================================
        psi_rq          Vs     Quadrature axis of the rotor oriented flux
        psi_rd          Vs     Direct axis of the rotor oriented flux
        psi_ra          Vs     Rotor oriented flux in branch a
        psi_rb          Vs     Rotor oriented flux in branch b
        psi_rc          Vs     Rotor oriented flux in branch c
        psi_ralpha      Vs     Rotor oriented flux in alpha direction
        psi_rbeta       Vs     Rotor oriented flux in beta direction
        =============== ====== =============================================
        =============== ====== =============================================
        Motor Voltages  Unit   Description
        =============== ====== =============================================
        u_sq            V      Quadrature axis voltage
        u_sd            V      Direct axis voltage
        u_sa            V      Stator voltage through branch a
        u_sb            V      Stator voltage through branch b
        u_sc            V      Stator voltage through branch c
        u_salpha        V      Stator voltage in alpha axis
        u_sbeta         V      Stator voltage in beta axis
        =============== ====== =============================================

        ======== ===========================================================
        Limits / Nominal Value Dictionary Entries:
        -------- -----------------------------------------------------------
        Entry    Description
        ======== ===========================================================
        i        General current limit / nominal value
        i_sa      Current in phase a
        i_sb      Current in phase b
        i_sc      Current in phase c
        i_salpha  Current in alpha axis
        i_sbeta   Current in beta axis
        i_sd     Current in direct axis
        i_sq     Current in quadrature axis
        omega    Mechanical angular Velocity
        torque   Motor generated torque
        u_sa      Voltage in phase a
        u_sb      Voltage in phase b
        u_sc      Voltage in phase c
        u_salpha  Voltage in alpha axis
        u_sbeta   Voltage in beta axis
        u_sd     Voltage in direct axis
        u_sq     Voltage in quadrature axis
        ======== ===========================================================


        Note:
            The voltage limits should be the amplitude of the phase voltage (:math:`\hat{u}_S`).
            Typically the rms value for the line voltage (:math:`U_L`) is given.
            :math:`\hat{u}_S=\sqrt{2/3}~U_L`

            The current limits should be the amplitude of the phase current (:math:`\hat{i}_S`).
            Typically the rms value for the phase current (:math:`I_S`) is given.
            :math:`\hat{i}_S = \sqrt{2}~I_S`

            If not specified, nominal values are equal to their corresponding limit values.
            Furthermore, if specific limits/nominal values (e.g. i_a) are not specified they are inferred from
            the general limits/nominal values (e.g. i)
        """
    _default_motor_parameter = {
        'p': 2,
        'l_m': 143.75e-3,
        'l_ssig': 5.87e-3,
        'l_rsig': 5.87e-3,
        'j_rotor': 1.1e-3,
        'r_s': 2.9338,
        'r_r': 1.355,
    }

    _default_limits = dict(omega=350, torque=0.0, i=5.5, epsilon=math.pi, u=560)
    _default_nominal_values = dict(omega=314, torque=0.0, i=3.9, epsilon=math.pi, u=560)

    def electrical_ode(self, state, u_salphabeta, omega, *args):
        """
        The differential equation of the SCIM.
        Sets u_ralpha = u_rbeta = 0 before calling the respective super function.
        """
        u_ralphabeta = np.zeros_like(u_salphabeta)
        u_sr_aphabeta = np.array([u_salphabeta, u_ralphabeta])

        return super().electrical_ode(state, u_sr_aphabeta, omega, *args)

    def _update_limits(self, limit_values={}, nominal_values={}):
        # Docstring of superclass

        # todo: this function is redundant wrt DoublyFedInductinoMotor
        voltage_limit = 0.5 * self._limits['u']
        voltage_nominal = 0.5 * self._nominal_values['u']

        limits_agenda = {}
        nominal_agenda = {}
        for u, i in zip(self.IO_VOLTAGES, self.IO_CURRENTS):
            limits_agenda[u] = voltage_limit
            nominal_agenda[u] = voltage_nominal
            limits_agenda[i] = self._limits.get('i', None) or \
                               self._limits[u] / self._motor_parameter['r_s']
            nominal_agenda[i] = self._nominal_values.get('i', None) or \
                                self._nominal_values[u] / self._motor_parameter['r_s']
        # overwrite default limits and nominal values with func args
        limits_agenda.update(limit_values)
        nominal_agenda.update(nominal_values)
        super()._update_limits(limits_agenda, nominal_agenda)


class DoublyFedInductionMotor(InductionMotor):
    """
        =====================  ==========  ============= ===========================================
        Motor Parameter        Unit        Default Value Description
        =====================  ==========  ============= ===========================================
        r_s                    Ohm         12e-3        Stator resistance
        r_r                    Ohm         21e-3         Rotor resistance
        l_m                    H           13.5e-3        Main inductance
        l_ssig                 H           0.2e-3        Stator-side stray inductance
        l_rsig                 H           0.1e-3        Rotor-side stray inductance
        p                      1           2             Pole pair number
        j_rotor                kg/m^2      1e3        Moment of inertia of the rotor
        =====================  ==========  ============= ===========================================

        =============== ====== =============================================
        Motor Currents  Unit   Description
        =============== ====== =============================================
        i_sq            A      Quadrature axis current
        i_sd            A      Direct axis current
        i_sa            A      Current through branch a
        i_sb            A      Current through branch b
        i_sc            A      Current through branch c
        i_salpha        A      Current in alpha axis
        i_sbeta         A      Current in beta axis
        =============== ====== =============================================
        =============== ====== =============================================
        Rotor flux      Unit   Description
        =============== ====== =============================================
        psi_rq          Vs     Quadrature axis of the rotor oriented flux
        psi_rd          Vs     Direct axis of the rotor oriented flux
        psi_ra          Vs     Rotor oriented flux in branch a
        psi_rb          Vs     Rotor oriented flux in branch b
        psi_rc          Vs     Rotor oriented flux in branch c
        psi_ralpha      Vs     Rotor oriented flux in alpha direction
        psi_rbeta       Vs     Rotor oriented flux in beta direction
        =============== ====== =============================================
        =============== ====== =============================================
        Motor Voltages  Unit   Description
        =============== ====== =============================================
        u_sq            V      Quadrature axis voltage
        u_sd            V      Direct axis voltage
        u_sa            V      Stator voltage through branch a
        u_sb            V      Stator voltage through branch b
        u_sc            V      Stator voltage through branch c
        u_salpha        V      Stator voltage in alpha axis
        u_sbeta         V      Stator voltage in beta axis
        u_ralpha        V      Rotor voltage in alpha axis
        u_rbeta         V      Rotor voltage in beta axis
        =============== ====== =============================================

        ======== ===========================================================
        Limits / Nominal Value Dictionary Entries:
        -------- -----------------------------------------------------------
        Entry    Description
        ======== ===========================================================
        i         General current limit / nominal value
        i_sa      Current in phase a
        i_sb      Current in phase b
        i_sc      Current in phase c
        i_salpha  Current in alpha axis
        i_sbeta   Current in beta axis
        i_sd      Current in direct axis
        i_sq      Current in quadrature axis
        omega     Mechanical angular Velocity
        torque    Motor generated torque
        u_sa      Voltage in phase a
        u_sb      Voltage in phase b
        u_sc      Voltage in phase c
        u_salpha  Voltage in alpha axis
        u_sbeta   Voltage in beta axis
        u_sd      Voltage in direct axis
        u_sq      Voltage in quadrature axis
        u_ralpha  Rotor voltage in alpha axis
        u_rbeta   Rotor voltage in beta axis
        ======== ===========================================================


        Note:
            The voltage limits should be the amplitude of the phase voltage (:math:`\hat{u}_S`).
            Typically the rms value for the line voltage (:math:`U_L`) is given.
            :math:`\hat{u}_S=\sqrt{2/3}~U_L`

            The current limits should be the amplitude of the phase current (:math:`\hat{i}_S`).
            Typically the rms value for the phase current (:math:`I_S`) is given.
            :math:`\hat{i}_S = \sqrt{2}~I_S`

            If not specified, nominal values are equal to their corresponding limit values.
            Furthermore, if specific limits/nominal values (e.g. i_a) are not specified they are inferred from
            the general limits/nominal values (e.g. i)
        """

    ROTOR_VOLTAGES = ['u_ralpha', 'u_rbeta']
    ROTOR_CURRENTS = ['i_ralpha', 'i_rbeta']

    IO_ROTOR_VOLTAGES = ['u_ra', 'u_rb', 'u_rc', 'u_rq', 'u_rd']
    IO_ROTOR_CURRENTS = ['i_ra', 'i_rb', 'i_rc', 'i_rq', 'i_rd']

    _default_motor_parameter = {
        'p': 2,
        'l_m': 13.5e-3,
        'l_ssig': 0.2e-3,
        'l_rsig': 0.1e-3,
        'j_rotor': 1e3,
        'r_s': 12e-3,
        'r_r': 21e-3,
    }

    _default_limits = dict(omega=160, torque=0.0, i=1900, epsilon=math.pi, u=1200)
    _default_nominal_values = dict(omega=157.08, torque=0.0, i=1900, epsilon=math.pi, u=1200)

    def __init__(self, **kwargs):
        self.IO_VOLTAGES += self.IO_ROTOR_VOLTAGES
        self.IO_CURRENTS += self.IO_ROTOR_CURRENTS
        super().__init__(**kwargs)

    def _update_limits(self, limit_values={}, nominal_values={}):
        # Docstring of superclass

        voltage_limit = 0.5 * self._limits['u']
        voltage_nominal = 0.5 * self._nominal_values['u']

        limits_agenda = {}
        nominal_agenda = {}
        for u, i in zip(self.IO_VOLTAGES+self.ROTOR_VOLTAGES,
                        self.IO_CURRENTS+self.ROTOR_CURRENTS):
            limits_agenda[u] = voltage_limit
            nominal_agenda[u] = voltage_nominal
            limits_agenda[i] = self._limits.get('i', None) or \
                               self._limits[u] / self._motor_parameter['r_r']
            nominal_agenda[i] = self._nominal_values.get('i', None) or \
                                self._nominal_values[u] / \
                                self._motor_parameter['r_r']
        # overwrite default limits and nominal values with func args
        limits_agenda.update(limit_values)
        nominal_agenda.update(nominal_values)
        super()._update_limits(limits_agenda, nominal_agenda)
