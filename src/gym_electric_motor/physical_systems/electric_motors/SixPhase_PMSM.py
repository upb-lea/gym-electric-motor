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

    =============== ====== =============================================
    Motor Currents  Unit   Description
    =============== ====== =============================================
    i_sd            A      Direct axis current
    i_sq            A      Quadrature axis current
    i_sx            A
    i_sy            A
    i_salpha        A      Stator current in alpha direction
    i_sbeta         A      Stator current in beta direction
    i_sX            A
    i_sY            A
    i_sa1           A
    i_sa2           A
    i_sb1           A
    i_sb2           A
    i_sc1           A
    i_sc2           A

    =============== ====== =============================================
    =============== ====== =============================================
    Motor Voltages  Unit   Description
    =============== ====== =============================================
    u_sd            V      Direct axis voltage
    =============== ====== =============================================

    ======== ===========================================================
    Limits / Nominal Value Dictionary Entries:
    -------- -----------------------------------------------------------
    Entry    Description
    ======== ===========================================================
    i        General current limit / nominal value
    ======== ===========================================================
        
    """
#### Parameters taken from  https://ieeexplore.ieee.org/document/10372153
    _default_motor_parameter = {
        "p": 5,
        "l_d": 125e-6,
        "l_q": 126e-6,
        "l_x": 39e-6,
        "l_y": 35e-6,
        "r_s": 64.3e-3,
        "psi_PM": 4.7e-3,
    }
    #_default_limits = ?
    #_default_nominal_values = ?
    #_model_constants = None
    #_default_initializer = {"states": {?},"interval": None,"random_init": None,"random_params": (None, None),}



    @property
    def motor_parameter(self):
        # Docstring of superclass
        return self._motor_parameter

    @property
    def initializer(self):
        # Docstring of superclass
        return self._initializer