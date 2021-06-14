import math
import numpy as np

from .synchronous_motor import SynchronousMotor


class PermanentMagnetSynchronousMotor(SynchronousMotor):
    """
    =====================  ==========  ============= ===========================================
    Motor Parameter        Unit        Default Value Description
    =====================  ==========  ============= ===========================================
    r_s                    Ohm         0.78          Stator resistance
    l_d                    H           1.2           Direct axis inductance
    l_q                    H           6.3e-3        Quadrature axis inductance
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
    u_sd            V      Direct axis voltage
    u_sq            V      Quadrature axis voltage
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

    #### Parameters taken from DOI: 10.1109/TPEL.2020.3006779 (A. Brosch, S. Hanke, O. Wallscheid, J. Boecker)
    #### and DOI: 10.1109/IEMDC.2019.8785122 (S. Hanke, O. Wallscheid, J. Boecker)
    _default_motor_parameter = {
        'p': 3,
        'l_d': 0.37e-3,
        'l_q': 1.2e-3,
        'j_rotor': 0.3883,
        'r_s': 18e-3,
        'psi_p': 66e-3,
    }
    HAS_JACOBIAN = True
    _default_limits = dict(omega=12e3 * np.pi / 30, torque=0.0, i=400, epsilon=math.pi, u=300)
    _default_nominal_values = dict(omega=3e3 * np.pi / 30, torque=0.0, i=240, epsilon=math.pi, u=300)
    _default_initializer = {
        'states':  {'i_sq': 0.0, 'i_sd': 0.0, 'epsilon': 0.0},
        'interval': None,
        'random_init': None,
        'random_params': (None, None)
    }

    IO_VOLTAGES = ['u_a', 'u_b', 'u_c', 'u_sd', 'u_sq']
    IO_CURRENTS = ['i_a', 'i_b', 'i_c', 'i_sd', 'i_sq']

    def _update_model(self):
        # Docstring of superclass
        mp = self._motor_parameter
        self._model_constants = np.array([
            #                 omega,        i_d,        i_q, u_d, u_q,          omega * i_d,          omega * i_q
            [                     0, -mp['r_s'],          0,   1,   0,                    0, mp['l_q'] * mp['p']],
            [-mp['psi_p'] * mp['p'],          0, -mp['r_s'],   0,   1, -mp['l_d'] * mp['p'],                   0],
            [               mp['p'],          0,          0,   0,   0,                    0,                   0],
        ])

        self._model_constants[self.I_SD_IDX] = self._model_constants[self.I_SD_IDX] / mp['l_d']
        self._model_constants[self.I_SQ_IDX] = self._model_constants[self.I_SQ_IDX] / mp['l_q']

    def _torque_limit(self):
        # Docstring of superclass
        mp = self._motor_parameter
        if mp['l_d'] == mp['l_q']:
            return self.torque([0, self._limits['i_sq'], 0])
        else:
            i_n = self.nominal_values['i']
            _p = mp['psi_p'] / (2 * (mp['l_d'] - mp['l_q']))
            _q = - i_n ** 2 / 2
            i_d_opt = - _p / 2 - np.sqrt( (_p / 2) ** 2 - _q)
            i_q_opt = np.sqrt(i_n ** 2 - i_d_opt ** 2)
            return self.torque([i_d_opt, i_q_opt, 0])

    def torque(self, currents):
        # Docstring of superclass
        mp = self._motor_parameter
        return 1.5 * mp['p'] * (mp['psi_p'] + (mp['l_d'] - mp['l_q']) * currents[self.I_SD_IDX]) * currents[self.I_SQ_IDX]

    def electrical_jacobian(self, state, u_in, omega, *args):
        mp = self._motor_parameter
        return (
            np.array([ # dx'/dx
                [-mp['r_s'] / mp['l_d'], mp['l_q']/mp['l_d'] * omega * mp['p'], 0],
                [-mp['l_d'] / mp['l_q'] * omega * mp['p'], - mp['r_s'] / mp['l_q'], 0],
                [0, 0, 0]
            ]),
            np.array([ # dx'/dw
                mp['p'] * mp['l_q'] / mp['l_d'] * state[self.I_SQ_IDX],
                - mp['p'] * mp['l_d'] / mp['l_q'] * state[self.I_SD_IDX] - mp['p'] * mp['psi_p'] / mp['l_q'],
                mp['p']
            ]),
            np.array([ # dT/dx
                1.5 * mp['p'] * (mp['l_d'] - mp['l_q']) * state[self.I_SQ_IDX],
                1.5 * mp['p'] * (mp['psi_p'] + (mp['l_d'] - mp['l_q']) * state[self.I_SD_IDX]),
                0
            ])
        )
