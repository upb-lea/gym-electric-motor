import numpy as np

from .dc_motor import DcMotor


class DcShuntMotor(DcMotor):
    """The DcShuntMotor is a DC motor with parallel armature and exciting circuit connected to one input voltage.

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

    # Motor parameter, nominal values and limits are based on the following DC Motor:
    # https://www.heinzmann-electric-motors.com/en/products/dc-motors/pmg-132-dc-motor
    _default_motor_parameter = {
        'r_a': 16e-3, 'r_e': 4e-1, 'l_a': 19e-6, 'l_e_prime': 1.7e-3, 'l_e': 5.4e-3, 'j_rotor': 0.0025
    }
    _default_nominal_values = dict(omega=300, torque=16.0, i=97, i_a=97, i_e=97, u=60, u_a=60, u_e=60)
    _default_limits = dict(omega=400, torque=38.0, i=210, i_a=210, i_e=210, u=60, u_a=60, u_e=60)
    _default_initializer = {
        'states': {'i_a': 0.0, 'i_e': 0.0},
        'interval': None,
        'random_init': None,
        'random_params': (None, None)
    }

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

    def _update_limits(self, limits_d=None, nominal_d=None):
        # Docstring of superclass

        # R_a might be 0, protect against that
        r_a = 1 if self._motor_parameter['r_a'] == 0 else self._motor_parameter['r_a']

        limit_agenda = {
            'u': self._default_limits['u'],
            'i_a': self._limits.get('i', None) or self._limits['u'] / r_a,
            'i_e': self._limits.get('i', None) or self._limits['u'] / self.motor_parameter['r_e'],
        }

        super()._update_limits(limit_agenda)
