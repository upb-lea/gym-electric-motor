import math
import numpy as np

from .induction_motor import InductionMotor


class DoublyFedInductionMotor(InductionMotor):
    """
        =====================  ==========  ============= ===========================================
        Motor Parameter        Unit        Default Value Description
        =====================  ==========  ============= ===========================================
        r_s                    Ohm         12e-3         Stator resistance
        r_r                    Ohm         21e-3         Rotor resistance
        l_m                    H           13.5e-3       Main inductance
        l_sigs                 H           0.2e-3        Stator-side stray inductance
        l_sigr                 H           0.1e-3        Rotor-side stray inductance
        p                      1           2             Pole pair number
        j_rotor                kg/m^2      1e3           Moment of inertia of the rotor
        =====================  ==========  ============= ===========================================
        =============== ====== =============================================
        Motor Currents  Unit   Description
        =============== ====== =============================================
        i_sd            A      Direct axis current
        i_sq            A      Quadrature axis current
        i_sa            A      Current through branch a
        i_sb            A      Current through branch b
        i_sc            A      Current through branch c
        i_salpha        A      Current in alpha axis
        i_sbeta         A      Current in beta axis
        =============== ====== =============================================
        =============== ====== =============================================
        Rotor flux      Unit   Description
        =============== ====== =============================================
        psi_rd          Vs     Direct axis of the rotor oriented flux
        psi_rq          Vs     Quadrature axis of the rotor oriented flux
        psi_ra          Vs     Rotor oriented flux in branch a
        psi_rb          Vs     Rotor oriented flux in branch b
        psi_rc          Vs     Rotor oriented flux in branch c
        psi_ralpha      Vs     Rotor oriented flux in alpha direction
        psi_rbeta       Vs     Rotor oriented flux in beta direction
        =============== ====== =============================================
        =============== ====== =============================================
        Motor Voltages  Unit   Description
        =============== ====== =============================================
        u_sd            V      Direct axis voltage
        u_sq            V      Quadrature axis voltage
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

    IO_ROTOR_VOLTAGES = ['u_ra', 'u_rb', 'u_rc', 'u_rd', 'u_rq']
    IO_ROTOR_CURRENTS = ['i_ra', 'i_rb', 'i_rc', 'i_rd', 'i_rq']

    #### Parameters taken from  DOI: 10.1016/j.jestch.2016.01.015 (N. Kumar, T. R. Chelliah, S. P. Srivastava)
    _default_motor_parameter = {
        'p': 2,
        'l_m': 297.5e-3,
        'l_sigs': 25.71e-3,
        'l_sigr': 25.71e-3,
        'j_rotor': 13.695e-3,
        'r_s': 4.42,
        'r_r': 3.51,
    }

    _default_limits = dict(omega=1800 * np.pi / 30, torque=0.0, i=9, epsilon=math.pi, u=720)
    _default_nominal_values = dict(omega=1650 * np.pi / 30, torque=0.0, i=7.5, epsilon=math.pi, u=720)
    _default_initializer = {'states':  {'i_salpha': 0.0, 'i_sbeta': 0.0,
                                        'psi_ralpha': 0.0, 'psi_rbeta': 0.0,
                                        'epsilon': 0.0},
                            'interval': None,
                            'random_init': None,
                            'random_params': (None, None)}

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
        super()._update_limits(limits_agenda, nominal_agenda)

    def _update_initial_limits(self, nominal_new={}, omega=None):
        # Docstring of superclass
        # draw a sample magnetic field angle from [-pi,pi]
        eps_mag = 2 * np.pi * np.random.random_sample() - np.pi
        flux_alphabeta_limits = self._flux_limit(omega=omega,
                                                 eps_mag=eps_mag,
                                                 u_q_max=self._nominal_values['u_sq'],
                                                 u_rq_max=self._nominal_values['u_rq'])
        flux_nominal_limits = {state: value for state, value in
                               zip(self.FLUXES, flux_alphabeta_limits)}
        super()._update_initial_limits(flux_nominal_limits)
