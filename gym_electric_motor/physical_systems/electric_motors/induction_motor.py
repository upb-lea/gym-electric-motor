import math
import numpy as np

from .three_phase_motor import ThreePhaseMotor


class InductionMotor(ThreePhaseMotor):
    """
        The InductionMotor and its subclasses implement the technical system of a three phase induction motor.

        This includes the system equations, the motor parameters of the equivalent circuit diagram,
        as well as limits and bandwidth.

        =====================  ==========  ============= ===========================================
        Motor Parameter        Unit        Default Value Description
        =====================  ==========  ============= ===========================================
        r_s                    Ohm         2.9338        Stator resistance
        r_r                    Ohm         1.355         Rotor resistance
        l_m                    H           143.75e-3     Main inductance
        l_sigs                 H           5.87e-3       Stator-side stray inductance
        l_sigr                 H           5.87e-3       Rotor-side stray inductance
        p                      1           2             Pole pair number
        j_rotor                kg/m^2      0.0011        Moment of inertia of the rotor
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
        Motor Voltages  Unit   Description
        =============== ====== =============================================
        u_sd            V      Direct axis voltage
        u_sq            V      Quadrature axis voltage
        u_sa            V      Voltage through branch a
        u_sb            V      Voltage through branch b
        u_sc            V      Voltage through branch c
        u_salpha        V      Voltage in alpha axis
        u_sbeta         V      Voltage in beta axis
        =============== ====== =============================================

        ======== ===========================================================
        Limits / Nominal Value Dictionary Entries:
        -------- -----------------------------------------------------------
        Entry    Description
        ======== ===========================================================
        i        General current limit / nominal value
        i_sa      Current in phase a
        i_sb      Current in phase b
        i_sc      Current in phase c
        i_salpha  Current in alpha axis
        i_sbeta   Current in beta axis
        i_sd     Current in direct axis
        i_sq     Current in quadrature axis
        omega    Mechanical angular Velocity
        torque   Motor generated torque
        u_sa      Voltage in phase a
        u_sb      Voltage in phase b
        u_sc      Voltage in phase c
        u_salpha  Voltage in alpha axis
        u_sbeta   Voltage in beta axis
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
    I_SALPHA_IDX = 0
    I_SBETA_IDX = 1
    PSI_RALPHA_IDX = 2
    PSI_RBETA_IDX = 3
    EPSILON_IDX = 4

    CURRENTS_IDX = [0, 1]
    FLUX_IDX = [2, 3]
    CURRENTS = ['i_salpha', 'i_sbeta']
    FLUXES = ['psi_ralpha', 'psi_rbeta']
    STATOR_VOLTAGES = ['u_salpha', 'u_sbeta']

    IO_VOLTAGES = ['u_sa', 'u_sb', 'u_sc', 'u_salpha', 'u_sbeta', 'u_sd',
                   'u_sq']
    IO_CURRENTS = ['i_sa', 'i_sb', 'i_sc', 'i_salpha', 'i_sbeta', 'i_sd',
                   'i_sq']

    HAS_JACOBIAN = True

    #  Parameters taken from  DOI: 10.1109/EPEPEMC.2018.8522008  (O. Wallscheid, M. Schenke, J. Boecker)
    _default_motor_parameter = {
        'p': 2,
        'l_m': 143.75e-3,
        'l_sigs': 5.87e-3,
        'l_sigr': 5.87e-3,
        'j_rotor': 1.1e-3,
        'r_s': 2.9338,
        'r_r': 1.355,
    }

    _default_limits = dict(omega=4e3 * np.pi / 30, torque=0.0, i=5.5, epsilon=math.pi, u=560)
    _default_nominal_values = dict(omega=3e3 * np.pi / 30, torque=0.0, i=3.9, epsilon=math.pi, u=560)
    _model_constants = None
    _default_initializer = {'states':  {'i_salpha': 0.0, 'i_sbeta': 0.0,
                                        'psi_ralpha': 0.0, 'psi_rbeta': 0.0,
                                        'epsilon': 0.0},
                            'interval': None,
                            'random_init': None,
                            'random_params': (None, None)}

    _initializer = None

    @property
    def motor_parameter(self):
        # Docstring of superclass
        return self._motor_parameter

    @property
    def initializer(self):
        # Docstring of superclass
        return self._initializer

    def __init__(
        self, motor_parameter=None, nominal_values=None, limit_values=None, motor_initializer=None,initial_limits=None,
    ):
        # Docstring of superclass
        # convert placeholder i and u to actual IO quantities
        _nominal_values = self._default_nominal_values.copy()
        _nominal_values.update({u: _nominal_values['u'] for u in self.IO_VOLTAGES})
        _nominal_values.update({i: _nominal_values['i'] for i in self.IO_CURRENTS})
        del _nominal_values['u'], _nominal_values['i']
        _nominal_values.update(nominal_values or {})
        # same for limits
        _limit_values = self._default_limits.copy()
        _limit_values.update({u: _limit_values['u'] for u in self.IO_VOLTAGES})
        _limit_values.update({i: _limit_values['i'] for i in self.IO_CURRENTS})
        del _limit_values['u'], _limit_values['i']
        _limit_values.update(limit_values or {})

        super().__init__(motor_parameter, nominal_values,
                         limit_values, motor_initializer, initial_limits)
        self._update_model()
        self._update_limits(_limit_values, _nominal_values)

    def reset(self,
              state_space,
              state_positions,
              omega=None):
        # Docstring of superclass
        if self._initializer and self._initializer['states']:
            self._update_initial_limits(omega=omega)
            self.initialize(state_space, state_positions)
            return np.asarray(list(self._initial_states.values()))
        else:
            return np.zeros(len(self.CURRENTS) + len(self.FLUXES) + 1)

    def electrical_ode(self, state, u_sr_alphabeta, omega, *args):
        """
        The differential equation of the Induction Motor.

        Args:
            state: The momentary state of the motor. [i_salpha, i_sbeta, psi_ralpha, psi_rbeta, epsilon]
            omega: The mechanical load
            u_sr_alphabeta: The input voltages [u_salpha, u_sbeta, u_ralpha, u_rbeta]

        Returns:
            The derivatives of the state vector d/dt( [i_salpha, i_sbeta, psi_ralpha, psi_rbeta, epsilon])
        """
        return np.matmul(self._model_constants, np.array([
            # omega, i_alpha, i_beta, psi_ralpha, psi_rbeta, omega * psi_ralpha, omega * psi_rbeta, u_salpha, u_sbeta, u_ralpha, u_rbeta,
            omega,
            state[self.I_SALPHA_IDX],
            state[self.I_SBETA_IDX],
            state[self.PSI_RALPHA_IDX],
            state[self.PSI_RBETA_IDX],
            omega * state[self.PSI_RALPHA_IDX],
            omega * state[self.PSI_RBETA_IDX],
            u_sr_alphabeta[0, 0],
            u_sr_alphabeta[0, 1],
            u_sr_alphabeta[1, 0],
            u_sr_alphabeta[1, 1],
        ]))

    def i_in(self, state):
        # Docstring of superclass
        return state[self.CURRENTS_IDX]

    def _torque_limit(self):
        # Docstring of superclass
        mp = self._motor_parameter
        return 1.5 * mp['p'] * mp['l_m'] ** 2/(mp['l_m']+mp['l_sigr']) * self._limits['i_sd'] * self._limits['i_sq'] / 2

    def torque(self, states):
        # Docstring of superclass
        mp = self._motor_parameter
        return \
            1.5 * mp['p'] * mp['l_m'] \
            / (mp['l_m'] + mp['l_sigr']) \
            * (
                states[self.PSI_RALPHA_IDX] * states[self.I_SBETA_IDX]
                - states[self.PSI_RBETA_IDX] * states[self.I_SALPHA_IDX]
            )

    def _flux_limit(self, omega=0, eps_mag=0, u_q_max=0.0, u_rq_max=0.0):
        """Calculate Flux limits for given current and magnetic-field angle

        Args:
            omega(float): speed given by mechanical load
            eps_mag(float): magnetic field angle
            u_q_max(float): maximal strator voltage in q-system
            u_rq_max(float): maximal rotor voltage in q-system

        Returns:
            maximal flux values(list) in alpha-beta-system
        """
        mp = self.motor_parameter
        l_s = mp['l_m'] + mp['l_sigs']
        l_r = mp['l_m'] + mp['l_sigr']
        l_mr = mp['l_m'] / l_r
        sigma = (l_s * l_r - mp['l_m'] ** 2) / (l_s * l_r)
        # limiting flux for a low omega
        if omega == 0:
            psi_d_max = mp['l_m'] * self._nominal_values['i_sd']
        else:
            i_d, i_q = self.q_inv([self._initial_states['i_salpha'],
                                  self._initial_states['i_sbeta']],
                                  eps_mag)
            psi_d_max = mp['p'] * omega * sigma * l_s * i_d + \
                        (mp['r_s'] + mp['r_r'] * l_mr**2) * i_q + \
                        u_q_max + \
                        l_mr * u_rq_max
            psi_d_max /= - mp['p'] * omega * l_mr
            # clipping flux and setting nominal limit
            psi_d_max = 0.9 * np.clip(psi_d_max, a_min=0, a_max=np.abs(mp['l_m'] * i_d))
        # returning flux in alpha, beta system
        return self.q([psi_d_max, 0], eps_mag)

    def _update_model(self):
        # Docstring of superclass
        mp = self._motor_parameter
        l_s = mp['l_m']+mp['l_sigs']
        l_r = mp['l_m']+mp['l_sigr']
        sigma = (l_s*l_r-mp['l_m']**2) /(l_s*l_r)
        tau_r = l_r / mp['r_r']
        tau_sig = sigma * l_s / (
                mp['r_s'] + mp['r_r'] * (mp['l_m'] ** 2) / (l_r ** 2))

        self._model_constants = np.array([
            # omega, i_alpha, i_beta, psi_ralpha, psi_rbeta, omega * psi_ralpha, omega * psi_rbeta, u_salpha, u_sbeta, u_ralpha, u_rbeta,
            [0, -1 / tau_sig, 0,mp['l_m'] * mp['r_r'] / (sigma * l_s * l_r ** 2), 0, 0,
             +mp['l_m'] * mp['p'] / (sigma * l_r * l_s), 1 / (sigma * l_s), 0,
             -mp['l_m'] / (sigma * l_r * l_s), 0, ],  # i_ralpha_dot
            [0, 0, -1 / tau_sig, 0,
             mp['l_m'] * mp['r_r'] / (sigma * l_s * l_r ** 2),
             -mp['l_m'] * mp['p'] / (sigma * l_r * l_s), 0, 0,
             1 / (sigma * l_s), 0, -mp['l_m'] / (sigma * l_r * l_s), ],
            # i_rbeta_dot
            [0, mp['l_m'] / tau_r, 0, -1 / tau_r, 0, 0, -mp['p'], 0, 0, 1,
             0, ],  # psi_ralpha_dot
            [0, 0, mp['l_m'] / tau_r, 0, -1 / tau_r, mp['p'], 0, 0, 0, 0, 1, ],
            # psi_rbeta_dot
            [mp['p'], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # epsilon_dot
        ])

    def electrical_jacobian(self, state, u_in, omega, *args):
        mp = self._motor_parameter
        l_s = mp['l_m'] + mp['l_sigs']
        l_r = mp['l_m'] + mp['l_sigr']
        sigma = (l_s * l_r - mp['l_m'] ** 2) / (l_s * l_r)
        tau_r = l_r / mp['r_r']
        tau_sig = sigma * l_s / (
                mp['r_s'] + mp['r_r'] * (mp['l_m'] ** 2) / (l_r ** 2))

        return (
            np.array([  # dx'/dx
                # i_alpha          i_beta               psi_alpha                                    psi_beta                                   epsilon
                [-1 / tau_sig, 0,
                 mp['l_m'] * mp['r_r'] / (sigma * l_s * l_r ** 2),
                 omega * mp['l_m'] * mp['p'] / (sigma * l_r * l_s), 0],
                [0, - 1 / tau_sig,
                 - omega * mp['l_m'] * mp['p'] / (sigma * l_r * l_s),
                 mp['l_m'] * mp['r_r'] / (sigma * l_s * l_r ** 2), 0],
                [mp['l_m'] / tau_r, 0, - 1 / tau_r, - omega * mp['p'], 0],
                [0, mp['l_m'] / tau_r, omega * mp['p'], - 1 / tau_r, 0],
                [0, 0, 0, 0, 0]
            ]),
            np.array([  # dx'/dw
                mp['l_m'] * mp['p'] / (sigma * l_r * l_s) * state[
                    self.PSI_RBETA_IDX],
                - mp['l_m'] * mp['p'] / (sigma * l_r * l_s) * state[
                    self.PSI_RALPHA_IDX],
                - mp['p'] * state[self.PSI_RBETA_IDX],
                mp['p'] * state[self.PSI_RALPHA_IDX],
                mp['p']
            ]),
            np.array([  # dT/dx
                - state[self.PSI_RBETA_IDX] * 3 / 2 * mp['p'] * mp[
                    'l_m'] / l_r,
                state[self.PSI_RALPHA_IDX] * 3 / 2 * mp['p'] * mp['l_m'] / l_r,
                state[self.I_SBETA_IDX] * 3 / 2 * mp['p'] * mp['l_m'] / l_r,
                - state[self.I_SALPHA_IDX] * 3 / 2 * mp['p'] * mp['l_m'] / l_r,
                0
            ])
        )
