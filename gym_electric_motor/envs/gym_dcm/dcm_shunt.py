from .dcm_base_env import _DCMBaseEnv
from gym.spaces import Box
import numpy as np


class _DCMShunt(_DCMBaseEnv):
    """
        **Description:**
            An environment for a discrete controlled DC Shunt Motor. The armature and excitation circuit are in
            parallel.

        **Observation:**

            Type: Dict(Box,Box,Box,Box,Box)

            +-------+-------------------+
            |Name   |Observation        |
            +=======+===================+
            |omega  |Angular Velocity   |
            +-------+-------------------+
            |torque |Electrical Torque  |
            +-------+-------------------+
            |i_a    |Armature Current   |
            +-------+-------------------+
            |i_e    |Excitation Current |
            +-------+-------------------+
            |u      |Input Voltage      |
            +-------+-------------------+
            |u_sup  |Supply Voltage     |
            +-------+-------------------+
            |References                 |
            +---------------------------+

            References in the same order as the state variables

        **Action Space:**
            Depends on the converter and if a continuous or discrete action should be used. It is the applied voltage.

        **Reward:**
            See DCM Base Environment

        **Starting State:**
            Random initial state, that fulfills some plausibility constraints

        **Episode Termination:**
            An episode terminates, when all the steps in the reference have been simulated or limits are violated.
    """

    # State array indices
    OMEGA_IDX = 0
    TORQUE_IDX = 1
    I_A_IDX = 2
    I_E_IDX = 3
    U_IDX = 4
    U_SUP_IDX = 5
    MOTOR_IDX = [OMEGA_IDX, I_A_IDX, I_E_IDX]

    def _step_integrate(self, action):
        raise NotImplementedError

    def __init__(self, zero_refs=(), **kwargs):
        super().__init__('DcShunt', state_vars=['omega', 'torque', 'i_a', 'i_e', 'u', 'u_sup'],
                         zero_refs=tuple(zero_refs)+('u',), **kwargs)

    def _set_limits(self):
        """
        Sum up all upper limits in an array in the same order as the states.
        """
        self._limits = np.array([
            self.motor_model.motor_parameter['omega_N'],
            self.motor_model.motor_parameter['torque_N'],
            self.motor_model.motor_parameter['i_a_N'],
            self.motor_model.motor_parameter['i_e_N'],
            self.motor_model.motor_parameter['u_N'],
            self.motor_model.motor_parameter['u_sup']  # supply voltage
        ]) * self._safety_margin

    def _set_observation_space(self):
        """
        Setting of observation space depending on the converter type.

        omega: Always positive
        Torque, i_a, i_e: Can be negative, when negative currents flow
        u: can be negative when negative voltages can be applied
        u_sup: always positive

        """
        # normalised values for the voltages and currents
        u_min, u_max = self.converter_model.voltages
        i_min, i_max = self.converter_model.currents
        # Set up the observation space state variables[omega, torque, i_a, i_e, u, u_sup]
        low = np.array([
            0.0,    # omega
            i_min,  # torque
            i_min,  # i_a
            i_min,  # i_e
            u_min,  # u
            0.0     # u_sup
        ])
        high = np.array([
            1,      # omega
            1,      # torque
            i_max,  # i_a
            i_max,  # i_e
            u_max,  # u
            u_max   # u_sup
        ])

        # Set the observation space for the references. Its ranges are equal to the range of the matching state var.
        for ref in range(len(self.state_vars)):
            if self._reward_weights[ref] <= 0 or self.state_vars[ref] in self._zero_refs:
                continue
            if self.state_vars[ref] == 'omega':
                mul_min = u_min
                mul_max = u_max
            else:
                mul_min = i_min
                mul_max = i_max
            low = np.concatenate((low, np.array(mul_min * np.ones(int(self._prediction_horizon + 1)))))
            high = np.concatenate((high, mul_max * np.ones(int(self._prediction_horizon + 1))))
        self.observation_space = Box(low, high, dtype=float)

    def _set_initial_value(self):
        """
        Set the initial value for the integration
        """
        self.system.set_initial_value(self._state[self.MOTOR_IDX], 0.0)

    def _generate_random_references(self):
        """
        This function generates random references for the next episode.

        First a random input sequence is build. Afterwards the system is
        fed with this input sequence and the resulting trajectories for all quantities are clipped to the limits and
        used as references. For all quantities, that should not have a reference, it is set to zero.
        """
        motor_param = self.motor_model.motor_parameter
        u = self._generate_random_control_sequence(self.motor_model.bandwidth(), motor_param['u_N'])
        i_a, i_e, torque, omega = np.zeros((4, len(u)))
        u_sup = np.ones(len(u)) * self.motor_model.u_sup  # supply voltage
        clip_low = self.observation_space.low[self.MOTOR_IDX] * self._limits[self.MOTOR_IDX] / self._safety_margin
        clip_high = self.observation_space.high[self.MOTOR_IDX] * self._limits[self.MOTOR_IDX] / self._safety_margin
        state = np.random.triangular(clip_low, self._state[self.MOTOR_IDX], clip_high)

        for k in range(1, len(u)):
            state = state \
                    + self.motor_model.model(state, self.load_model.load(state[self.OMEGA_IDX]), u[k-1]) * self._tau
            state = np.clip(state, clip_low, clip_high)
            omega[k], i_a[k], i_e[k] = state
            torque[k] = self.motor_model.torque(state)

        self._reference = np.array([omega, torque, i_a, i_e, u, u_sup], subok=True)
        lim = self._limits
        self._reference = self._reference / lim.reshape(len(lim), 1)

    def _set_initial_state(self):
        """
        Find a random initial state, that is plausible.
        Omega will be set anywhere in its limits.
        i_e will be set to a positive value
        i_a will be set to a discounted value, because the induced voltage in the motor limits it.
        """
        omega = np.random.rand() * self._limits[self.OMEGA_IDX] / self._safety_margin
        i_e = np.random.rand() * self._limits[self.I_E_IDX] / self._safety_margin
        i_a_max = min(self._limits[self.I_A_IDX] / self._safety_margin, self.motor_model.i_a_max(omega))
        i_a = ((self.observation_space.high[self.I_A_IDX] - self.observation_space.low[self.I_A_IDX]) *
               np.random.rand() + self.observation_space.low[self.I_A_IDX]) * i_a_max
        self._state = np.zeros_like(self._state)
        self._state[[self.OMEGA_IDX, self.I_A_IDX, self.I_E_IDX]] = omega, i_a, i_e
        self._state[self.TORQUE_IDX] = self.motor_model.torque(self._state[self.MOTOR_IDX])
        self._state = np.clip(self._state, -self._limits / self._safety_margin, self._limits / self._safety_margin)


class DCMShuntCont(_DCMShunt):
    def __init__(self, converter_type='2Q', tau=1e-4, **kwargs):
        super().__init__(converter_type=f'cont-{converter_type}', tau=tau, **kwargs)

    def step(self, action):
        """
        Clips the action to its limits and uses the base step function

        Args:
              action: desired switching state of the converter that should be applied

        Returns:
              Tuple(array(float), float, bool, dict):
                  **observation:**    The observation from the environment \n
                  **reward:**         The reward for the taken action \n
                  **bool:**           Flag if the episode has ended \n
                  **info:**           An always empty dictionary \n
        """
        return super().step(np.clip(action, self.action_space.low[0], self.action_space.high[0])[0])

    def _step_integrate(self, action):
        """
        The integration is done for one time period. The converter considers the dead time and interlocking time.

        Args:
            action: switching state of the converter that should be applied
        """
        self._state[self.U_SUP_IDX] = self._reference[self.U_SUP_IDX][self._k]*self._limits[self.U_SUP_IDX]
        i_in = self.motor_model.i_in(self._state)
        u_in = self.converter_model.convert(action, i_in, self.system.t) * self._state[self.U_SUP_IDX]
        self.system.set_f_params(u_in, self._noise[self.U_IDX, self._k])
        state = self.system.integrate(self.system.t + self._tau)
        self._state[self.MOTOR_IDX] = state
        self._state[self.TORQUE_IDX] = self.motor_model.torque(state)
        self._state[self.U_IDX] = u_in


class DCMShuntDisc(_DCMShunt):
    def __init__(self, converter_type='2Q', tau=1e-5, **kwargs):
        super().__init__(converter_type=f'disc-{converter_type}', tau=tau, **kwargs)

    def _step_integrate(self, action):
        """
        The integration is done for one time period. The converter considers the dead time and the interlocking time.

        Args:
            action: switching state of the converter that should be applied
        """
        # input voltage in interlocking time
        self._state[self.U_SUP_IDX] = self._reference[self.U_SUP_IDX][self._k] * self._limits[self.U_SUP_IDX]
        i_in = self.motor_model.i_in(self._state)
        u_in = self.converter_model.convert(action, i_in, self.system.t) * self._state[self.U_SUP_IDX]
        self.system.set_f_params(u_in, self._noise[self.U_IDX, self._k])
        state = self.system.integrate(self.system.t + self.converter_model.interlocking_time)

        # input voltage after switching
        i_in = self.motor_model.i_in(self._state)
        u_in = self.converter_model.convert(action, i_in, self.system.t) * self._state[self.U_SUP_IDX]
        self.system.set_f_params(u_in, self._noise[self.U_IDX, self._k])
        state = self.system.integrate(self.system.t+self.tau-self.converter_model.interlocking_time)
        self._state[self.MOTOR_IDX] = state
        self._state[self.TORQUE_IDX] = self.motor_model.torque(state)
        self._state[self.U_IDX] = u_in

