import numpy as np

from .dc_motor import DcMotor


class DcSeriesMotor(DcMotor):
    """The DcSeriesMotor is a DcMotor with an armature and exciting circuit connected in series to one input voltage.

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

    # Motor parameter, nominal values and limits are based on the following DC Motor:
    # https://www.heinzmann-electric-motors.com/en/products/dc-motors/pmg-132-dc-motor
    _default_motor_parameter = {
        'r_a': 16e-3, 'r_e': 48e-3, 'l_a': 19e-6, 'l_e_prime': 1.7e-3, 'l_e': 5.4e-3, 'j_rotor': 0.0025
    }
    _default_nominal_values = dict(omega=300, torque=16.0, i=97, i_a=97, i_e=97, u=60, u_a=60, u_e=60)
    _default_limits = dict(omega=400, torque=38.0, i=210, i_a=210, i_e=210, u=60, u_a=60, u_e=60)
    _default_initializer = {'states': {'i': 0.0},
                            'interval': None,
                            'random_init': None,
                            'random_params': (None, None)}

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
            np.array([[-(mp['r_a'] + mp['r_e'] + mp['l_e_prime'] * omega) / (
                    mp['l_a'] + mp['l_e'])]]),
            np.array([-mp['l_e_prime'] * state[self.I_IDX] / (
                    mp['l_a'] + mp['l_e'])]),
            np.array([2 * mp['l_e_prime'] * state[self.I_IDX]])
        )
