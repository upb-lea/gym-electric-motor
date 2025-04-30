import math

import numpy as np

from .six_phase_motor import SixPhaseMotor


class SixPhasePMSM(SixPhaseMotor):
    """
    =====================  ==========  ============= ===========================================
    Motor Parameter        Unit        Default Value Description
    =====================  ==========  ============= ===========================================
    r_s                    Ohm         64.3e-3       Stator resistance
    l_d                    H           125e-6        Direct axis inductance
    l_q                    H           126e-6        Quadrature axis inductance
    l_x                    H           39e-6         x-axis inductance
    l_y                    H           35e-6         y-axis inductance
    p                      1            5            Pole pair number
    psi_PM                 Vs          4.7e-3        flux linkage of the permanent magnets
    =====================  ==========  ============= ===========================================

    The default parameters originate from:
    L. Broghammer et al., "Reinforcement Learning Control of Six-Phase Permanent Magnet Synchronous Machines,"
    2023 13th International Electric Drives Production Conference (EDPC), Regensburg, Germany, 2023,pp. 1-8,
    doi: 10.1109/EDPC60603.2023.10372153.

    =============== ====== =============================================
    Motor Currents  Unit   Description
    =============== ====== =============================================
    i_sd            A      Direct axis current
    i_sq            A      Quadrature axis current
    i_sx            A      Stator current in the counter-rotating system
    i_sy            A      Stator current in the counter-rotating system
    i_salpha        A      Stator current in alpha direction
    i_sbeta         A      Stator current in beta direction
    i_sX            A      Stator current in X direction
    i_sY            A      Stator current in Y direction
    i_sa1           A      Stator current in phase a1
    i_sa2           A      Stator current in phase a2
    i_sb1           A      Stator current in phase b1
    i_sb2           A      Stator current in phase b2
    i_sc1           A      Stator current in phase c1
    i_sc2           A      Stator current in phase c2

    =============== ====== =============================================
    =============== ====== =============================================
    Motor Voltages  Unit   Description
    =============== ====== =============================================
    u_sd            V      Direct axis voltage
    u_sq            V      Quadrature axis voltage
    u_sx            V      voltage in the counter-rotating system
    u_sy            V      voltage in the counter-rotating system
    u_a1            V      Phase voltage in phase a1
    u_a2            V      Phase voltage in phase a2
    u_b1            V      Phase voltage in phase b1
    u_b2            V      Phase voltage in phase b2
    u_c1            V      Phase voltage in phase c1
    u_c2            V      Phase voltage in phase c2
    =============== ====== =============================================

    ======== ===========================================================
    Limits / Nominal Value Dictionary Entries:
    -------- -----------------------------------------------------------
    Entry    Description
    ======== ===========================================================
    i        General current limit / nominal value
    ======== ===========================================================
        
    """
    I_SD_IDX = 0
    I_SQ_IDX = 1
    I_SX_IDX = 2
    I_SY_IDX = 3
    CURRENTS_IDX = [0, 1, 2, 3]
    CURRENTS = ["i_sd", "i_sq", "i_sx", "i_sy"]
    VOLTAGES = ["u_sd", "u_sq", "u_sx", "u_sy"]

    @property
    def motor_parameter(self):
        # Docstring of superclass
        return self._motor_parameter

    @property
    def initializer(self):
        # Docstring of superclass
        return self._initializer

   #### Parameters taken from  https://ieeexplore.ieee.org/document/10372153
    _default_motor_parameter = {"p": 5, "l_d": 125e-6, "l_q": 126e-6, "l_x": 39e-6, "l_y": 35e-6, "r_s": 64.3e-3, "psi_PM": 4.7e-3,  "j_rotor": 0.0110,}
    HAS_JACOBIAN = True
    _default_limits =dict(omega=4e3 * np.pi / 30, torque=10.0, i=400, epsilon=math.pi, u=300)
    _default_nominal_values = dict(omega=3e3 * np.pi / 30, torque=0.0, i=240, epsilon=math.pi, u=300)
    _default_initializer = {
        "states": {"i_sd": 0.0, "i_sq": 0.0, "i_sx": 0.0, "i_sy": 0.0, "epsilon": 0.0},
        "interval": None,
        "random_init": None,
        "random_params": (None, None),
    }
    IO_VOLTAGES = ["u_a1", "u_b1", "u_c1", "u_a2", "u_b2", "u_c2", "u_sd", "u_sq", "u_sx", "u_sy"]
    IO_CURRENTS = ["i_a1", "i_b1", "i_a2", "i_b2", "i_c2",  "i_c1", "i_sd", "i_sq", "i_sx", "i_sy"]
    _model_constants = None

    _initializer = None

    def __init__(
        self,
        motor_parameter=None,
        nominal_values=None,
        limit_values=None,
        motor_initializer=None,
    ):
        # Docstring of superclass
        nominal_values = nominal_values or {}
        limit_values = limit_values or {}
        super().__init__(motor_parameter, nominal_values, limit_values, motor_initializer)
        self._update_model()
        self._update_limits()
    
    def _update_model(self):
        """Updates the motor's model parameters with the motor parameters.

        Called internally when the motor parameters are changed or the motor is initialized.
        """
        mp = self._motor_parameter
        self._model_constants = np.array([
            #        omega,         i_d,         i_q,         i_x,        i_y, u_d, u_q, u_x, u_y, omega * i_d, omega * i_q, omega * i_x,  omega * i_y
            [            0,   -mp['r_s'],          0,           0,          0,   1,   0,   0,   0,           0,   mp['l_q'],           0,          0],
            [-mp['psi_PM'],            0,  -mp['r_s'],          0,          0,   0,   1,   0,   0,   -mp['l_d'],          0,           0,          0],
            [            0,            0,           0, -mp['r_s'],          0,   0,   0,   1,   0,            0,          0,           0, -mp['l_y']],
            [            0,            0,           0,          0, -mp['r_s'],   0,   0,   0,   1,            0,          0,   mp['l_x'],          0],
            [      mp['p'],            0,           0,          0,          0,   0,   0,   0,   0,            0,          0,           0,          0],

        ])
        self._model_constants[self.I_SD_IDX] = self._model_constants[self.I_SD_IDX] / mp["l_d"]
        self._model_constants[self.I_SQ_IDX] = self._model_constants[self.I_SQ_IDX] / mp["l_q"]
        self._model_constants[self.I_SX_IDX] = self._model_constants[self.I_SX_IDX] / mp["l_x"]
        self._model_constants[self.I_SY_IDX] = self._model_constants[self.I_SY_IDX] / mp["l_y"]


    def electrical_ode(self, state, u_dqxy, omega, *_):
        """
        The differential equation of the Six phase PMSM.

        Args:
            state: The current state of the motor. [i_sd, i_sq, i_sx, i_sy, epsilon]
            omega: electrical rotational speed
            u_qdxy: The input voltages [u_sd, u_sq, u_sx, u_sy]

        Returns:
            The derivatives of the state vector d/dt([i_sd, i_sq, i_sx, i_sy, epsilon])
        """
        return np.matmul(
            self._model_constants,
            np.array(
                [
                    omega,
                    state[self.I_SD_IDX],
                    state[self.I_SQ_IDX],
                    state[self.I_SX_IDX],
                    state[self.I_SY_IDX],
                    u_dqxy[0],
                    u_dqxy[1],
                    u_dqxy[2],
                    u_dqxy[3],
                    omega * state[self.I_SD_IDX],
                    omega * state[self.I_SQ_IDX],
                    omega * state[self.I_SX_IDX],
                    omega * state[self.I_SY_IDX],
                ]
            ),
        )
    def electrical_jacobian(self, state, u_in, omega, *args):
        mp = self._motor_parameter
        return (
            np.array(
                [  # dx'/dx
                    # i_sd          i_sq               i_sx                                    i_sy                                   epsilon
                    [
                        -mp["r_s"] / mp["l_d"],
                        mp["l_q"] / mp["l_d"] * omega, 
                        0, 
                        0, 
                        0
                    ],
                    [
                        -mp["l_d"] / mp["l_q"] * omega,
                        -mp["r_s"] / mp["l_q"],
                        0,
                        0,
                        0
                    ],
                    [
                        0,
                        0,
                        -mp["r_s"] / mp["l_x"],
                        -mp["l_y"] / mp["l_x"] * omega,
                        0
                    ],
                    [
                        0,
                        0,
                        mp["l_x"] / mp["l_y"] * omega,
                        -mp["r_s"] / mp["l_y"],
                        0
                    ],
                    [0, 0, 0, 0, 0],
                ]
            ),
            np.array(
                [  # dx'/dw
                    mp["l_q"] / mp["l_d"],
                    -mp["l_d"] / mp["l_q"],
                    -mp["l_y"] / mp["l_x"],
                    mp["l_x"] / mp["l_y"],
                    mp["p"],
                ]
            ),
            np.array(
                [  # dT/dx
                    1.5 * mp["p"] * (mp["l_d"] - mp["l_q"]) * state[self.I_SQ_IDX],
                    1.5 * mp["p"] * (mp["psi_PM"] + (mp["l_d"] - mp["l_q"]) * state[self.I_SD_IDX]),
                    0,
                    0,
                    0
                ]
            ),
        )

    def i_in(self, state):
        # Docstring of superclass
        return state[self.CURRENTS_IDX]
    
    def reset(self, state_space, state_positions, **__):
        # Docstring of superclass
        if self._initializer and self._initializer["states"]:
            self.initialize(state_space, state_positions)
            return np.asarray(list(self._initial_states.values()))
        else:
            return np.zeros(len(self.CURRENTS) + 1)
        
        
    #from pmsm
    def torque(self, currents):
        # Docstring of superclass
        mp = self._motor_parameter
        return (
            1.5 * mp["p"] * (mp["psi_PM"] + (mp["l_d"] - mp["l_q"]) * currents[self.I_SD_IDX]) * currents[self.I_SQ_IDX]
        )
    
    #torque limit ?
    
    def _update_limits(self):
        # Docstring of superclass

        voltage_limit = 0.5 * self._limits["u"]
        voltage_nominal = 0.5 * self._nominal_values["u"]

        limits_agenda = {}
        nominal_agenda = {}
        for u, i in zip(self.IO_VOLTAGES, self.IO_CURRENTS):
            limits_agenda[u] = voltage_limit
            nominal_agenda[u] = voltage_nominal
            limits_agenda[i] = self._limits.get("i", None) or self._limits[u] / self._motor_parameter["r_s"]
            nominal_agenda[i] = (
                self._nominal_values.get("i", None) or self._nominal_values[u] / self._motor_parameter["r_s"]
            )
        super()._update_limits(limits_agenda, nominal_agenda)