import numpy as np

from .three_phase_motor import ThreePhaseMotor


class SynchronousMotor(ThreePhaseMotor):
    """The SynchronousMotor and its subclasses implement the technical system of a three phase synchronous motor.

    This includes the system equations, the motor parameters of the equivalent circuit diagram,
    as well as limits and bandwidth.

    =====================  ==========  ============= ===========================================
    Motor Parameter        Unit        Default Value Description
    =====================  ==========  ============= ===========================================
    r_s                    Ohm         0.78          Stator resistance
    l_d                    H           1.2           Direct axis inductance
    l_q                    H           6.3e-3        Quadrature axis inductance
    psi_p                  Wb          0.0094        Effective excitation flux (PMSM only)
    p                      1           2             Pole pair number
    j_rotor                kg/m^2      0.017         Moment of inertia of the rotor
    =====================  ==========  ============= ===========================================

    =============== ====== =============================================
    Motor Currents  Unit   Description
    =============== ====== =============================================
    i_sd            A      Direct axis current
    i_sq            A      Quadrature axis current
    i_a             A      Current through branch a
    i_b             A      Current through branch b
    i_c             A      Current through branch c
    i_alpha         A      Current in alpha axis
    i_beta          A      Current in beta axis
    =============== ====== =============================================
    =============== ====== =============================================
    Motor Voltages  Unit   Description
    =============== ====== =============================================
    u_sd            A      Direct axis voltage
    u_sq            A      Quadrature axis voltage
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

    I_SD_IDX = 0
    I_SQ_IDX = 1
    EPSILON_IDX = 2
    CURRENTS_IDX = [0, 1]
    CURRENTS = ['i_sd', 'i_sq']
    VOLTAGES = ['u_sd', 'u_sq']

    _model_constants = None

    _initializer = None

    def __init__(self, motor_parameter=None, nominal_values=None, limit_values=None, motor_initializer=None):
        # Docstring of superclass
        nominal_values = nominal_values or {}
        limit_values = limit_values or {}
        super().__init__(motor_parameter, nominal_values,
                         limit_values, motor_initializer)
        self._update_model()
        self._update_limits()

    @property
    def motor_parameter(self):
        # Docstring of superclass
        return self._motor_parameter

    @property
    def initializer(self):
        # Docstring of superclass
        return self._initializer

    def _torque_limit(self):
        raise NotImplementedError

    def reset(self, state_space, state_positions, **__):
        # Docstring of superclass
        if self._initializer and self._initializer['states']:
            self.initialize(state_space, state_positions)
            return np.asarray(list(self._initial_states.values()))
        else:
            return np.zeros(len(self.CURRENTS) + 1)

    def torque(self, state):
        # Docstring of superclass
        raise NotImplementedError

    def _update_model(self):
        """
        Set motor parameters into a matrix for faster computation
        """
        raise NotImplementedError

    def electrical_ode(self, state, u_dq, omega, *_):
        """
        The differential equation of the Synchronous Motor.

        Args:
            state: The current state of the motor. [i_sd, i_sq, epsilon]
            omega: The mechanical load
            u_qd: The input voltages [u_sd, u_sq]

        Returns:
            The derivatives of the state vector d/dt([i_sd, i_sq, epsilon])
        """
        return np.matmul(self._model_constants, np.array([
            omega,
            state[self.I_SD_IDX],
            state[self.I_SQ_IDX],
            u_dq[0],
            u_dq[1],
            omega * state[self.I_SD_IDX],
            omega * state[self.I_SQ_IDX],
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
            limits_agenda[i] = self._limits.get('i', None) \
                or self._limits[u] / self._motor_parameter['r_s']
            nominal_agenda[i] = self._nominal_values.get('i', None) \
                or self._nominal_values[u] / self._motor_parameter['r_s']
        super()._update_limits(limits_agenda, nominal_agenda)
