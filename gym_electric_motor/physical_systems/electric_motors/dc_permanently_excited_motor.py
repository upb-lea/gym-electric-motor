import numpy as np

from .dc_motor import DcMotor


class DcPermanentlyExcitedMotor(DcMotor):
    """The DcPermanentlyExcitedMotor is a DcMotor with a Permanent Magnet instead of the excitation circuit.

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

    # Motor parameter, nominal values and limits are based on the following DC Motor:
    # https://www.heinzmann-electric-motors.com/en/products/dc-motors/pmg-132-dc-motor
    _default_motor_parameter = {
        'r_a': 16e-3, 'l_a': 19e-6, 'psi_e': 0.165, 'j_rotor': 0.025
    }
    _default_nominal_values = dict(omega=300, torque=16.0, i=97, u=60)
    _default_limits = dict(omega=400, torque=38.0, i=210, u=60)
    _default_initializer = {
        'states': {'i': 0.0},
        'interval': None,
        'random_init': None,
        'random_params': (None, None)
    }

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
        self._ode_placeholder[:] = [omega] + np.atleast_1d(state[self.I_IDX]).tolist() + [u_in[0]]
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
