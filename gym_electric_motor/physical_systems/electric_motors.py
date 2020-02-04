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

    def electrical_ode(self, currents, u_in, omega, *_):
        """
        Calculation of the derivatives of each motor state variable for the given inputs / The motors ODE-System.

        Args:
            currents(ndarray(float)): The motors state.
            u_in(list(float)): The motors input voltages.
            omega(float): Angular velocity of the motor

        Returns:
             ndarray(float): Derivatives of the motors ODE-system for the given inputs.
        """
        raise NotImplementedError

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
        self._model_constants[self.I_A_IDX] /= mp['l_a']
        self._model_constants[self.I_E_IDX] /= mp['l_e']

    def torque(self, currents):
        # Docstring of superclass
        return self._motor_parameter['l_e_prime'] * currents[self.I_A_IDX] * currents[self.I_E_IDX]

    def i_in(self, currents):
        # Docstring of superclass
        return list(currents)

    def electrical_ode(self, currents, u_in, omega, *_):
        # Docstring of superclass
        return np.matmul(self._model_constants, np.array([
            currents[self.I_A_IDX],
            currents[self.I_E_IDX],
            omega * currents[self.I_E_IDX],
            u_in[0],
            u_in[1],
        ]))

    def _update_limits(self):
        """
        Calculate for all the missing maximal and nominal values the physical maximal possible values.
        """
        if self._limits.get('u_a', 0) == 0:
            self._limits['u_a'] = self._default_limits['u']
        if self._limits.get('u_e', 0) == 0:
            self._limits['u_e'] = self._default_limits['u']
        if self._limits.get('i_a', 0) == 0.0:
            self._limits['i_a'] = self._limits.get('i', None) or self._limits['u_a'] / self._motor_parameter['r_a']
        if self._nominal_values.get('i_a', 0) == 0:
            self._nominal_values['i_a'] = self._limits['i_a']
        if self._limits.get('i_e', 0) == 0.0:
            self._limits['i_e'] = self._limits.get('i', None) or self._limits['u_e'] / self.motor_parameter['r_e']
        if self._nominal_values.get('i_e', 0) == 0:
            self._nominal_values['i_e'] = self._limits['i_e']
        if self._limits.get('torque', 0) == 0.0:
            motor_limit_state = [self._limits[state] for state in self.CURRENTS]
            self._limits['torque'] = self.torque(motor_limit_state)
        if self._nominal_values.get('torque', 0) == 0:
            self._nominal_values['torque'] = self._limits['torque']
        if self._limits.get('omega', 0) == 0.0:
            self._limits['omega'] = self._default_limits['omega']
        if self._nominal_values.get('omega', 0) == 0:
            self._nominal_values['omega'] = self._limits['omega']

        for entry in self._limits.keys():
            if self._nominal_values.get(entry, 0) == 0:
                self._nominal_values[entry] = self._limits[entry]

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
        lower_limit = 0
        low = {
            'omega': -1 if input_voltages[a_converter][lower_limit] == -1
            or input_voltages[e_converter][lower_limit] == -1 else 0,
            'torque': -1 if input_currents[a_converter][lower_limit] == -1
            or input_currents[e_converter][lower_limit] == -1 else 0,
            'i_a': -1 if input_currents[a_converter][lower_limit] == -1 else 0,
            'i_e': -1 if input_currents[e_converter][lower_limit] == -1 else 0,
            'u_a': -1 if input_voltages[a_converter][lower_limit] == -1 else 0,
            'u_e': -1 if input_voltages[e_converter][lower_limit] == -1 else 0,
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


class DcShuntMotor(DcMotor):
    """
        The DcShuntMotor is a DC Motor with parallel armature and exciting circuit connected to one input voltage.

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

    VOLTAGES = ['u']

    _default_nominal_values = {'omega': 368, 'torque': 0.0, 'i_a': 50, 'i_e': 1.2, 'u': 420}
    _default_limits = {'omega': 500, 'torque': 0.0, 'i_a': 75, 'i_e': 2, 'u': 420}

    def i_in(self, state):
        # Docstring of superclass
        return [state[self.I_A_IDX] + state[self.I_E_IDX]]

    def electrical_ode(self, currents, u_in, omega, *_):
        # Docstring of superclass
        return super().electrical_ode(currents, (u_in[0], u_in[0]), omega)

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
            'torque': -1 if input_currents[lower_limit] == -1 else 0,
            'i_a': -1 if input_currents[lower_limit] == -1 else 0,
            'i_e': -1 if input_currents[lower_limit] == -1 else 0,
            'u': -1 if input_voltages[lower_limit] == -1 else 0,
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
        """
        Calculate for all the missing maximal and nominal values the physical maximal possible values.
        """
        if self._limits.get('u', 0) == 0:
            self._limits['u'] = self._default_limits['u']
        if self._limits.get('i_a', 0) == 0.0:
            self._limits['i_a'] = self._limits.get('i', None) or self._limits['u_a'] / self._motor_parameter['r_a']
        if self._nominal_values.get('i_a', 0) == 0:
            self._nominal_values['i_a'] = self._limits['i_a']
        if self._limits.get('i_e', 0) == 0.0:
            self._limits['i_e'] = self._limits.get('i', None) or self._limits['u_e'] / self.motor_parameter['r_e']
        if self._nominal_values.get('i_e', 0) == 0:
            self._nominal_values['i_e'] = self._limits['i_e']
        if self._limits.get('torque', 0) == 0.0:
            motor_limit_state = [self._limits[state] for state in self.CURRENTS]
            self._limits['torque'] = self.torque(motor_limit_state)
        if self._nominal_values.get('torque', 0) == 0:
            self._nominal_values['torque'] = self._limits['torque']
        if self._limits.get('omega', 0) == 0.0:
            self._limits['omega'] = self._default_limits['omega']
        if self._nominal_values.get('omega', 0) == 0:
            self._nominal_values['omega'] = self._limits['omega']

        for entry in self._limits.keys():
            if self._nominal_values.get(entry, 0) == 0:
                self._nominal_values[entry] = self._limits[entry]


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
    I_IDX = 0
    CURRENTS_IDX = [0]
    CURRENTS = ['i']
    VOLTAGES = ['u']

    _default_motor_parameter = {
        'r_a': 2.78, 'r_e': 1.0, 'l_a': 6.3e-3, 'l_e': 1.6e-3, 'l_e_prime': 0.05, 'j_rotor': 0.017,
    }
    _default_nominal_values = dict(omega=80, torque=0.0, i=50, u=420)
    _default_limits = dict(omega=100, torque=0.0, i=100, u=420)

    def _update_model(self):
        # Docstring of superclass
        mp = self._motor_parameter
        self._model_constants = np.array([
            [-mp['r_a'] - mp['r_e'], -mp['l_e_prime'], 1]
        ])
        self._model_constants[self.I_IDX] /= (mp['l_a'] + mp['l_e'])

    def torque(self, currents):
        # Docstring of superclass
        return super().torque([currents[self.I_IDX], currents[self.I_IDX]])

    def electrical_ode(self, currents, u_in, omega, *_):
        # Docstring of superclass
        return np.matmul(
            self._model_constants,
            np.array([
                currents[self.I_IDX],
                omega * currents[self.I_IDX],
                u_in[0]
            ])
        )

    def i_in(self, state):
        # Docstring of superclass
        return state[self.CURRENTS_IDX]

    def _update_limits(self):
        # Docstring of superclass
        if self._limits.get('u', 0) == 0:
            self._limits['u'] = self._default_limits['u']
        if self._limits.get('i', 0) == 0.0:
            self._limits['i'] = self._limits['u'] / (self._motor_parameter['r_a'] + self._motor_parameter['r_e'])
        if self._limits.get('torque', 0) == 0.0:
            motor_limit_state = [self._limits[state] for state in self.CURRENTS]
            self._limits['torque'] = self.torque(motor_limit_state)
        if self._nominal_values.get('torque', 0) == 0:
            self._nominal_values['torque'] = self._limits['torque']
        if self._limits.get('omega', 0) == 0.0:
            self._limits['omega'] = self._default_limits['omega']
        if self._nominal_values.get('omega', 0) == 0:
            self._nominal_values['omega'] = self._limits['omega']
        for entry in self._limits.keys():
            if self._nominal_values.get(entry, 0) == 0:
                self._nominal_values[entry] = self._limits[entry]

    def get_state_space(self, input_currents, input_voltages):
        # Docstring of superclass
        lower_limit = 0
        low = {
            'omega': 0,
            'torque': 0,
            'i': -1 if input_currents[lower_limit] == -1 else 0,
            'u': -1 if input_voltages[lower_limit] == -1 else 0,
        }
        high = {
            'omega': 1,
            'torque': 1,
            'i': 1,
            'u': 1,
        }
        return low, high


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

    _default_motor_parameter = {
        'r_a': 25.0, 'l_a': 3.438e-2, 'psi_e': 18, 'j_rotor': 0.017
    }
    _default_nominal_values = dict(omega=22, torque=0.0, i=16, u=400)
    _default_limits = dict(omega=50, torque=0.0, i=25, u=400)

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
        return np.matmul(self._model_constants, np.array([omega, state[self.I_IDX], u_in[0]]))

    def _update_limits(self):
        # Docstring of superclass
        if self._limits.get('u', 0) == 0:
            self._limits['u'] = self._default_limits['u']
        if self._limits.get('i', 0) == 0.0:
            self._limits['i'] = self._limits['u'] / self._motor_parameter['r_a']
        if self._limits.get('torque', 0) == 0.0:
            motor_limit_state = [self._limits[state] for state in self.CURRENTS]
            self._limits['torque'] = self.torque(motor_limit_state)
        if self._nominal_values.get('torque', 0) == 0:
            self._nominal_values['torque'] = self._limits['torque']
        if self._limits.get('omega', 0) == 0.0:
            self._limits['omega'] = self._default_limits['omega']
        if self._nominal_values.get('omega', 0) == 0:
            self._nominal_values['omega'] = self._limits['omega']
        for entry in self._limits.keys():
            if self._nominal_values.get(entry, 0) == 0:
                self._nominal_values[entry] = self._limits[entry]

    def get_state_space(self, input_currents, input_voltages):
        # Docstring of superclass
        lower_limit = 0
        low = {
            'omega': -1 if input_voltages[lower_limit] == -1 else 0,
            'torque': -1 if input_currents[lower_limit] == -1 else 0,
            'i': -1 if input_currents[lower_limit] == -1 else 0,
            'u': -1 if input_voltages[lower_limit] == -1 else 0,
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
    pass


class SynchronousMotor(ElectricMotor):
    """
        The SynchronousMotor and its subclasses implement the technical system of a Three Phase Synchronous motor.

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

    _model_constants = None

    @property
    def motor_parameter(self):
        # Docstring of superclass
        return self._motor_parameter

    @staticmethod
    def t_23(quantities):
        """
        Transformation from abc representation to alpha-beta representation

        Args:
            quantities: The properties in the abc representation like ''[u_a, u_b, u_c]''

        Returns:
            The converted quantities in the alpha-beta representation like ''[u_alpha, u_beta]''
        """
        return np.matmul(SynchronousMotor._t23, quantities)

    @staticmethod
    def t_32(quantities):
        """
        Transformation from alpha-beta representation to abc representation

        Args:
            quantities: The properties in the alpha-beta representation like ``[u_alpha, u_beta]``

        Returns:
            The converted quantities in the abc representation like ``[u_a, u_b, u_c]``
        """
        return np.matmul(SynchronousMotor._t32, quantities)

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

    def __init__(self, motor_parameter=None, nominal_values=None, limit_values=None, **__):
        # Docstring of superclass
        nominal_values = nominal_values or {}
        limit_values = limit_values or {}
        super().__init__(motor_parameter, nominal_values, limit_values)
        self._update_model()
        self._update_limits()

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

    def _torque_limit(self):
        """
        Returns:
             Maximal possible torque for the given limits in self._limits
        """
        raise NotImplementedError

    def _update_limits(self):
        """
        Calculate for all the missing maximal and nominal values the physical maximal possible values.
        """
        mp = self._motor_parameter
        if self._limits.get('u_a', 0) == 0:
            self._limits['u_a'] = .5 * self._limits['u']
        if self._limits.get('u_b', 0) == 0:
            self._limits['u_b'] = .5 * self._limits['u']
        if self._limits.get('u_c', 0) == 0:
            self._limits['u_c'] = .5 * self._limits['u']
        if self._limits.get('u_alpha', 0) == 0:
            self._limits['u_alpha'] = .5 * self._limits['u']
        if self._limits.get('u_beta', 0) == 0:
            self._limits['u_beta'] = 0.5 * self._limits['u']
        if self._limits.get('u_sd', 0) == 0:
            self._limits['u_sd'] = .5 * self._limits['u']
        if self._limits.get('u_sq', 0) == 0:
            self._limits['u_sq'] = .5 * self._limits['u']

        if self._limits.get('i_alpha', 0) == 0:
            self._limits['i_alpha'] = self._limits.get('i', None) or self._limits[
                'u_alpha'] / mp['r_s']
        if self._limits.get('i_beta', 0) == 0:
            self._limits['i_beta'] = self._limits.get('i', None) or self._limits['u_beta'] / \
                                     mp['r_s']
        if self._limits.get('i_a', 0) == 0:
            self._limits['i_a'] = self._limits.get('i', None) or self._limits['u_a'] / mp['r_s']
        if self._limits.get('i_b', 0) == 0:
            self._limits['i_b'] = self._limits.get('i', None) or self._limits['u_b'] / mp['r_s']
        if self._limits.get('i_c', 0) == 0:
            self._limits['i_c'] = self._limits.get('i', None) or self._limits['u_c'] / mp['r_s']
        if self._limits.get('i_sd', 0) == 0:
            self._limits['i_sd'] = self._limits.get('i', None) or self._limits['u_sd'] / mp[
                'r_s']
        if self._limits.get('i_sq', 0.0) == 0:
            self._limits['i_sq'] = self._limits.get('i', None) or self._limits['u_sq'] / mp[
                'r_s']

        if self._limits['torque'] == 0:
            self._limits['torque'] = self._torque_limit()

        if self._limits['omega'] == 0:
            self._limits['omega'] = self._default_limits['omega']

        if 'u' not in self._nominal_values.keys():
            self._nominal_values.update({'u': self._limits['u']})
        if 'i' not in self._nominal_values.keys():
            self._nominal_values.update({'i': self._limits['i']})

        if self._nominal_values.get('u_a', 0) == 0:
            self._nominal_values['u_a'] = .5 * self._nominal_values['u']
        if self._nominal_values.get('u_a', 0) == 0:
            if self._nominal_values.get('u_b', 0) == 0:
                self._nominal_values['u_b'] = .5 * self._nominal_values['u']
        if self._nominal_values.get('u_c', 0) == 0:
            self._nominal_values['u_c'] = .5 * self._nominal_values['u']
        if self._nominal_values.get('u_alpha', 0) == 0:
            self._nominal_values['u_alpha'] = .5 * self._nominal_values['u']
        if self._nominal_values.get('u_beta', 0) == 0:
            self._nominal_values['u_beta'] = .5 * self._nominal_values['u']
        if self._nominal_values.get('u_sd', 0) == 0:
            self._nominal_values['u_sd'] = .5 * self._nominal_values['u']
        if self._nominal_values.get('u_sq', 0) == 0:
            self._nominal_values['u_sq'] = .5 * self._nominal_values['u']

        if self._nominal_values.get('i_alpha', 0) == 0:
            self._nominal_values['i_alpha'] = self._nominal_values.get('i', None) \
                                              or self._nominal_values['u_alpha'] / mp['r_s']
        if self._nominal_values.get('i_beta', 0) == 0:
            self._nominal_values['i_beta'] = self._nominal_values.get('i', None) \
                                             or self._nominal_values['u_beta'] / mp['r_s']
        if self._nominal_values.get('i_a', 0) == 0:
            self._nominal_values['i_a'] = self._nominal_values.get('i', None) or self._nominal_values['u_a'] / mp['r_s']
        if self._nominal_values.get('i_b', 0) == 0:
            self._nominal_values['i_b'] = self._nominal_values.get('i', None) or self._nominal_values['u_b'] / mp['r_s']
        if self._nominal_values.get('i_c', 0) == 0:
            self._nominal_values['i_c'] = self._nominal_values.get('i', None) \
                                          or self._nominal_values['u_c'] / mp['r_s']
        if self._nominal_values.get('i_sd', 0) == 0:
            self._nominal_values['i_sd'] = self._nominal_values.get('i', None) \
                                           or self._nominal_values['u_sd'] / mp['r_s']
        if self._nominal_values.get('i_sq', 0.0) == 0:
            self._nominal_values['i_sq'] = self._nominal_values.get('i', None) \
                                           or self._nominal_values['u_sq'] / mp['r_s']

        for entry in self._limits.keys():
            if self._nominal_values.get(entry, 0) == 0:
                self._nominal_values[entry] = self._limits[entry]

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
    _default_motor_parameter = {'p': 2, 'l_d': 73.2e-3, 'l_q': 7.3e-3, 'j_rotor': 2.45e-3, 'r_s': 0.3256}
    _default_nominal_values = {
        'i': 54, 'torque': 0, 'omega': 523.0, 'epsilon': np.pi, 'u': 600
    }
    _default_limits = {
        'i': 70, 'torque': 0, 'omega': 600.0, 'epsilon': np.pi, 'u': 600
    }

    def _update_model(self):
        # Docstring of superclass
        mp = self._motor_parameter
        self._model_constants = np.array([
            # omega, i_sq, i_sd, u_sq, u_sd, omega * i_sq, omega * i_sd
            [0, -mp['r_s'], 0, 1, 0, 0, -mp['l_d'] * mp['p']],
            [0, 0, -mp['r_s'], 0, 1, mp['l_q'] * mp['p'], 0],
            [mp['p'], 0, 0, 0, 0, 0, 0]
        ])
        self._model_constants[self.I_SQ_IDX] /= mp['l_q']
        self._model_constants[self.I_SD_IDX] /= mp['l_d']

    def _torque_limit(self):
        # Docstring of superclass
        return self.torque([self._limits['i_sq'] / np.sqrt(2), self._limits['i_sd'] / np.sqrt(2), 0])

    def torque(self, currents):
        # Docstring of superclass
        mp = self._motor_parameter
        return 1.5 * mp['p'] * ((mp['l_d'] - mp['l_q']) * currents[self.I_SD_IDX]) * currents[self.I_SQ_IDX]


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
    _default_motor_parameter = {
        'p': 2,
        'l_d': 79e-3,
        'l_q': 113e-3,
        'j_rotor': 2.45e-3,
        'r_s': 4.9,
        'psi_p': 0.165,
    }

    _default_limits = dict(omega=80, torque=0.0, i=20, epsilon=math.pi, u=600)
    _default_nominal_values = dict(omega=75, torque=0.0, i=12, epsilon=math.pi, u=600)

    def _update_model(self):
        # Docstring of superclass
        mp = self._motor_parameter
        self._model_constants = np.array([
            # omega,                 i_q,        i_d,     u_q, u_d, omega * i_q,        omega * i_d
            [-mp['psi_p'] * mp['p'], -mp['r_s'], 0, 1, 0, 0, -mp['l_d'] * mp['p']],
            [0, 0, -mp['r_s'], 0, 1, mp['l_q'] * mp['p'], 0],
            [mp['p'], 0, 0, 0, 0, 0, 0]
        ])

        self._model_constants[self.I_SQ_IDX] /= mp['l_q']
        self._model_constants[self.I_SD_IDX] /= mp['l_d']

    def _torque_limit(self):
        # Docstring of superclass
        return self.torque([self._limits['i_sq'], 0, 0])

    def torque(self, currents):
        # Docstring of superclass
        mp = self._motor_parameter
        return 1.5 * mp['p'] * (
                mp['psi_p'] + (mp['l_d'] - mp['l_q']) * currents[self.I_SD_IDX]
        ) * currents[self.I_SQ_IDX]
