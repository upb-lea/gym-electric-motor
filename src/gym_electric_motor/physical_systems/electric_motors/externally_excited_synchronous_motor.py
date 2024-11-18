import math

import numpy as np

from .synchronous_motor import SynchronousMotor


class ExternallyExcitedSynchronousMotor(SynchronousMotor):
    """
    =====================  ==========  ============= =================================================
    Motor Parameter        Unit        Default Value Description
    =====================  ==========  ============= =================================================
    r_s                    mOhm        15.55         Stator resistance
    r_e                    mOhm        7.2           Excitation resistance
    l_d                    mH          1.66          Direct axis inductance
    l_q                    mH          0.35          Quadrature axis inductance
    l_m                    mH          1.589         Mutual inductance
    l_e                    mH          1.74          Excitation inductance
    p                      1           3             Pole pair number
    k                      1           65.21         Transfer ratio of number of windings (N_s / N_e)
    j_rotor                kg/m^2      0.3883        Moment of inertia of the rotor
    =====================  ==========  ============= =================================================

    =============== ====== ===========================================================================
    Motor Currents  Unit   Description
    =============== ====== ===========================================================================
    i_sd            A      Direct axis current
    i_sq            A      Quadrature axis current
    i_e             A      Excitation current
    i_a             A      Current through branch a
    i_b             A      Current through branch b
    i_c             A      Current through branch c
    i_alpha         A      Current in alpha axis
    i_beta          A      Current in beta axis
    =============== ====== ===========================================================================
    =============== ====== ===========================================================================
    Motor Voltages  Unit   Description
    =============== ====== ===========================================================================
    u_sd            V      Direct axis voltage
    u_sq            V      Quadrature axis voltage
    u_e             V      Exciting voltage
    u_a             V      Voltage through branch a
    u_b             V      Voltage through branch b
    u_c             V      Voltage through branch c
    u_alpha         V      Voltage in alpha axis
    u_beta          V      Voltage in beta axis
    =============== ====== ===========================================================================

    ======== =========================================================================================
    Limits / Nominal Value Dictionary Entries:
    -------- -----------------------------------------------------------------------------------------
    Entry    Description
    ======== =========================================================================================
    i        General current limit / nominal value
    i_a      Current in phase a
    i_b      Current in phase b
    i_c      Current in phase c
    i_alpha  Current in alpha axis
    i_beta   Current in beta axis
    i_sd     Current in direct axis
    i_sq     Current in quadrature axis
    i_e      Current in excitation circuit
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
    u_e      Voltage in excitation circuit
    ======== =========================================================================================


    Note:
        The voltage limits should be the peak-to-peak value of the phase voltage (:math:`\hat{u}_S`).
        A phase voltage denotes the potential difference from a line to the neutral point in contrast to the line voltage between two lines.
        Typically the RMS value for the line voltage (:math:`U_L`) is given as
        :math:`\hat{u}_S=\sqrt{2/3}~U_L`

        The current limits should be the peak-to-peak value of the phase current (:math:`\hat{i}_S`).
        Typically the RMS value for the phase current (:math:`I_S`) is given as
        :math:`\hat{i}_S = \sqrt{2}~I_S`

        If not specified, nominal values are equal to their corresponding limit values.
        Furthermore, if specific limits/nominal values (e.g. i_a) are not specified they are inferred from
        the general limits/nominal values (e.g. i)
    """

    I_SD_IDX = 0
    I_SQ_IDX = 1
    I_E_IDX = 2
    EPSILON_IDX = 3
    CURRENTS_IDX = [0, 1, 2]
    CURRENTS = ["i_sd", "i_sq", "i_e"]
    VOLTAGES = ["u_sd", "u_sq", "u_e"]

    #### Parameters taken from DOI: 10.1109/ICELMACH.2014.6960287 (C. D. Nguyen; W. Hofmann)
    _default_motor_parameter = {
        'p': 3,
        'l_d': 1.66e-3,
        'l_q': 0.35e-3,
        'l_m': 1.589e-3,
        'l_e': 1.74e-3,
        'j_rotor': 0.3883,
        'r_s': 15.55e-3,
        'r_e': 7.2e-3,
        'k': 65.21,
    }
    HAS_JACOBIAN = True
    _default_limits = dict(omega=12e3 * np.pi / 30, torque=0.0, i=150, i_e=150, epsilon=math.pi, u=320)
    _default_nominal_values = dict(omega=4.3e3 * np.pi / 30, torque=0.0, i=120, i_e=150, epsilon=math.pi, u=320)
    _default_initializer = {
        "states": {"i_sq": 0.0, "i_sd": 0.0, "i_e": 0.0, "epsilon": 0.0},
        "interval": None,
        "random_init": None,
        "random_params": (None, None),
    }

    IO_VOLTAGES = ["u_a", "u_b", "u_c", "u_sd", "u_sq", "u_e"]
    IO_CURRENTS = ["i_a", "i_b", "i_c", "i_sd", "i_sq", "i_e"]

    def _update_model(self):
        # Docstring of superclass
        mp = self._motor_parameter

        # Transform rotor quantities to stator side (uppercase index denotes transformation to stator side)
        mp['r_E'] = mp['k'] ** 2 * 3/2 * mp['r_e']
        mp['l_M'] = mp['k'] * 3/2 * mp['l_m']
        mp['l_E'] = mp['k'] ** 2 * 3/2 * mp['l_e']

        mp['i_k_rs'] = 2 / 3 / mp['k']  # ratio (i_E / i_e)
        mp['sigma'] = 1 - mp['l_M'] ** 2 / (mp['l_d'] * mp['l_E'])

        # fmt: off

        self._model_constants = np.array([
            #                 omega,                                               i_d,        i_q,                                                              i_e,                              u_d, u_q,                                                    u_e,          omega * i_d,                                                  omega * i_q,                         omega * i_e
            [                     0,                          -mp["r_s"] / mp["sigma"],          0, mp["l_M"] * mp["r_E"] / (mp["sigma"] * mp["l_E"]) * mp["i_k_rs"],                        1 / mp["sigma"],   0, -mp["l_M"] * mp["k"] / (mp["sigma"] * mp["l_E"]),                    0,                            mp["l_q"] * mp["p"] / mp["sigma"],                                   0],
            [                     0,                                                 0, -mp["r_s"],                                                                0,                                      0,   1,                                                0, -mp["l_d"] * mp["p"],                                                            0, -mp["p"] * mp["l_M"] * mp["i_k_rs"]],
            [                     0, mp["l_M"] * mp["r_s"] / (mp["sigma"] * mp["l_d"]),          0,                          -mp["r_E"] / mp["sigma"] * mp["i_k_rs"], -mp["l_M"] / (mp["sigma"] * mp["l_d"]),   0,                            mp["k"] / mp["sigma"],                    0, -mp["p"] * mp["l_M"] * mp["l_q"] / (mp["sigma"] * mp["l_d"]),                                   0],
            [               mp["p"],                                                 0,          0,                                                                0,                                      0,   0,                                                0,                    0,                                                            0,                                   0],
        ])

        self._model_constants[self.I_SD_IDX] = self._model_constants[self.I_SD_IDX] / mp["l_d"]
        self._model_constants[self.I_SQ_IDX] = self._model_constants[self.I_SQ_IDX] / mp["l_q"]
        self._model_constants[self.I_E_IDX] = self._model_constants[self.I_E_IDX] / mp["l_E"] / mp["i_k_rs"]

        # fmt: on

    def electrical_ode(self, state, u_dqe, omega, *_):
        """
        The differential equation of the Synchronous Motor.

        Args:
            state: The current state of the motor. [i_sd, i_sq, epsilon]
            omega: The mechanical load
            u_qd: The input voltages [u_sd, u_sq]

        Returns:
            The derivatives of the state vector d/dt([i_sd, i_sq, epsilon])
        """

        return np.matmul(
            self._model_constants,
            np.array(
                [
                    omega,
                    state[self.I_SD_IDX],
                    state[self.I_SQ_IDX],
                    state[self.I_E_IDX],
                    u_dqe[0],
                    u_dqe[1],
                    u_dqe[2],
                    omega * state[self.I_SD_IDX],
                    omega * state[self.I_SQ_IDX],
                    omega * state[self.I_E_IDX]
                ]
            )
        )

    def _torque_limit(self):
        # Docstring of superclass
        mp = self._motor_parameter
        if mp["l_d"] == mp["l_q"]:
            return self.torque([0, self._limits["i_sq"], self._limits["i_e"], 0])
        else:
            i_n = self.nominal_values["i"]
            _p = mp["l_M"] * i_n / (2 * (mp["l_d"] - mp["l_q"]))
            _q = - i_n ** 2 / 2
            if mp["l_d"] < mp["l_q"]:
                i_d_opt = - _p / 2 - np.sqrt((_p / 2) ** 2 - _q)
            else:
                i_d_opt = - _p / 2 + np.sqrt((_p / 2) ** 2 - _q)
            i_q_opt = np.sqrt(i_n ** 2 - i_d_opt ** 2)
        return self.torque([i_d_opt, i_q_opt, self._limits["i_e"], 0])

    def torque(self, currents):
        # Docstring of superclass
        mp = self._motor_parameter
        return 1.5 * mp["p"] * (mp["l_M"] * currents[self.I_E_IDX] * mp["i_k_rs"] + (mp["l_d"] - mp["l_q"]) * currents[self.I_SD_IDX]) * currents[self.I_SQ_IDX]

    def electrical_jacobian(self, state, u_in, omega, *args):

        mp = self._motor_parameter

        return (
            np.array([ # dx'/dx
                [                                      -mp["r_s"] / (mp["l_d"] * mp["sigma"]),                                         mp["l_q"] / (mp["sigma"] * mp["l_d"]) * omega * mp["p"], mp["l_M"] * mp["r_E"] / (mp["sigma"] * mp["l_d"] * mp["l_E"]) * mp["i_k_rs"], 0],
                [                                    -mp["l_d"] / mp["l_q"] * omega * mp["p"],                                                                          -mp["r_s"] / mp["l_q"],                      -omega * mp["p"] * mp["l_M"] / mp["l_q"] * mp["i_k_rs"], 0],
                [mp["l_M"] * mp["r_s"] / (mp["sigma"] * mp["l_d"] * mp["l_E"] * mp["i_k_rs"]), -omega * mp["p"] * mp["l_M"] * mp["l_q"] / (mp["sigma"] * mp["l_d"] * mp["l_E"] * mp["i_k_rs"]),                                       -mp["r_E"] / (mp["sigma"] * mp["l_E"]), 0],
                [                                                                           0,                                                                                               0,                                                                            0, 0],
            ]),
            np.array([ # dx'/dw
                mp["p"] * mp["l_q"] / (mp["l_d"] * mp["sigma"]) * state[self.I_SQ_IDX],
                -mp["p"] * mp["l_d"] / mp["l_q"] * state[self.I_SD_IDX] - mp["p"] * mp["l_M"] / mp["l_q"] * state[self.I_E_IDX] * mp["i_k_rs"],
                -mp["p"] * mp["l_M"] * mp["l_q"] / (mp["sigma"] * mp["l_d"] * mp["l_E"] * mp["i_k_rs"]) * state[self.I_SQ_IDX],
                mp["p"],
            ]),
            np.array([ # dT/dx
                1.5 * mp["p"] * (mp["l_d"] - mp["l_q"]) * state[self.I_SQ_IDX],
                1.5 * mp["p"] * (mp["l_M"] * state[self.I_E_IDX] * mp["i_k_rs"] + (mp["l_d"] - mp["l_q"]) * state[self.I_SD_IDX]),
                1.5 * mp["p"] * mp["l_M"] * mp["i_k_rs"] * state[self.I_SQ_IDX],
                0,
            ])
        )
        # fmt: on
