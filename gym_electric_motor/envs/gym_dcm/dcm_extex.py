import numpy as np
from gym import spaces

from .models import converter_models
from .dcm_base_env import _DCMBaseEnv
from gym.spaces import Box

OMEGA_IDX = 0
TORQUE_IDX = 1
I_A_IDX = 2
I_E_IDX = 3
U_A_IDX = 4
U_E_IDX = 5
U_SUP_IDX = 6
MOTOR_IDX = [OMEGA_IDX, I_A_IDX, I_E_IDX]
VOLTAGES = [U_A_IDX, U_E_IDX]


class _DCMExtEx(_DCMBaseEnv):
    """
        **Description:**
            Environment for a DC externally excited Motor.
        **Observation:**

            Type: Dict(Box,Box,Box,Box,Box,Box)

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
            |u_a    |Armature Voltage   |
            +-------+-------------------+
            |u_e    |Excitation Voltage |
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

    def _step_integrate(self, action):
        raise NotImplementedError

    def __init__(self, converter_type, zero_refs=(), **kwargs):
        super().__init__('DcExtEx', state_vars=['omega', 'torque', 'i_a', 'i_e', 'u_a', 'u_e','u_sup'],
                         zero_refs=tuple(zero_refs)+('u_a', 'u_e'),
                         converter_type=converter_type, **kwargs)
        self.converter_model = (converter_models.Converter.make(converter_type, self._tau), self.converter_model)

        # normalised values for the voltages and currents

        u_a_min, u_a_max = self.converter_model[0].voltages
        i_a_min, i_a_max = self.converter_model[0].currents
        u_e_min, u_e_max = self.converter_model[1].voltages
        i_e_min, i_e_max = self.converter_model[1].currents

        # Set up the observation space
        low = np.array([
            min(u_a_min, u_e_min),  # omega
            min(i_a_min, i_e_min),  # torque
            i_a_min,                # i_a
            i_e_min,                # i_e
            u_a_min,                # u_a
            u_e_min,                # u_e
            0.0                     # u_sup
        ])

        high = np.array([
            max(u_a_max, u_e_max),  # omega
            max(i_a_max, i_e_max),  # torque
            i_a_max,                # i_a
            i_e_max,                # i_e
            u_a_max,                # u_a
            u_e_max,                # u_e
            max(u_a_max, u_e_max)   # u_sup
        ])

        # Set the observation space for the references. Its ranges are equal to the range of the matching state var.
        for ref in range(len(self.state_vars)):
            if self._reward_weights[ref] <= 0 or self.state_vars[ref] in self._zero_refs:
                continue
            mul_min, mul_max = {
                'omega': (min(u_a_min, u_e_min), max(u_a_max, u_e_max)),
                'torque': (min(i_a_min, i_e_min), max(i_a_max, i_e_max)),
                'i_a': (i_a_min, i_a_max),
                'i_e': (i_e_min, i_e_max),
                'u_a': (u_a_min, u_a_max),
                'u_e': (u_e_min, u_e_max)
            }[self.state_vars[ref]]
            low = np.concatenate((low, np.array(mul_min * np.ones(int(self._prediction_horizon+1)))))
            high = np.concatenate((high, mul_max * np.ones(int(self._prediction_horizon+1))))
            self.observation_space = Box(low, high, dtype=np.float32)

    def _set_limits(self):
        """
        sum up all upper limits in an array in the same order as the states
        """
        self._limits = np.array([
            self.motor_model.motor_parameter['omega_N'],
            self.motor_model.motor_parameter['torque_N'],
            self.motor_model.motor_parameter['i_a_N'],
            self.motor_model.motor_parameter['i_e_N'],
            self.motor_model.motor_parameter['u_a_N'],
            self.motor_model.motor_parameter['u_e_N'],
            max(self.motor_model.motor_parameter['u_a_N'], self.motor_model.motor_parameter['u_e_N'])
        ]) * self._safety_margin  # enlarge nominal values to maximum values

    def _set_observation_space(self):
        pass

    def _set_initial_value(self):
        """
        Set the initial value for the integration.
        """
        self.system.set_initial_value(self._state[MOTOR_IDX], 0.0)

    def _system_eq(self, t, state, u_in, noise):
        """
        The differential equation of the system.

        This function is called by the integrator.

        Args:
            t: Current time of the system
            state: The current state as a numpy array.
            u_in: input voltage

        Returns:
            The solution of the system. The first derivatives of all the state variables.
        """

        t_load = self.load_model.load(state[OMEGA_IDX])  # ~1e-5s
        return self.motor_model.model(state, t_load, u_in + noise)  # 2e-4

    def _generate_random_references(self):
        """
        This function generates a random references. First a random input sequence is build. Afterwards the system is
        fed with this input sequence and the resulting trajectories for all quantities are clipped to the limits and
        used as references. For all quantities, that should not have a reference, it is set to zero.
        """
        motor_params = self.motor_model.motor_parameter
        u_a = self._generate_random_control_sequence(self.motor_model.bandwidth()[0], motor_params['u_a_N'])
        u_e = self._generate_random_control_sequence(self.motor_model.bandwidth()[1], motor_params['u_e_N'])
        i_a, i_e, torque, omega = np.zeros((4, len(u_a)))
        u_sup = np.ones(len(u_a)) * self.motor_model.u_sup
        clip_low = self.observation_space.low[MOTOR_IDX] * self._limits[MOTOR_IDX] / self._safety_margin
        clip_high = self.observation_space.high[MOTOR_IDX] * self._limits[MOTOR_IDX] / self._safety_margin
        state = np.random.triangular(clip_low, self._state[MOTOR_IDX], clip_high)
        for k in range(1, len(u_a)):
            state += self.motor_model.model(state, self.load_model.load(state[OMEGA_IDX]), (u_a[k-1], u_e[k-1]))\
                     * self._tau
            state = np.clip(state, clip_low, clip_high)
            omega[k], i_a[k], i_e[k] = state
            torque[k] = self.motor_model.torque(state)

        self._reference = np.array([omega, torque, i_a, i_e, u_a, u_e, u_sup], subok=True)
        lim = self._limits
        self._reference = self._reference / lim.reshape(len(lim), 1)

    def _generate_random_control_sequence(self, bw, maximum):
        ref_len = self.episode_length + self._prediction_horizon
        rands = np.random.randn(2, ref_len // 2)
        u = rands[0] + 1j * rands[1]
        bw_noise = np.random.rand()
        bw *= bw_noise
        delta_w = 2 * np.pi / ref_len / self._tau
        u[int(bw / delta_w) + 1:] = 0.0
        sigma = np.linspace(1, 0, int(bw / delta_w) + 1) ** np.random.randint(10)

        if len(sigma) < len(u):
            u[:len(sigma)] *= sigma
        else:
            u *= sigma[:len(u)]

        fourier = np.concatenate((np.random.randn(1), u, np.flip(np.conjugate(u))))
        u = np.fft.ifft(fourier).real

        power_noise = 2 * np.random.rand(1) + 0.5
        u = u * maximum / np.sqrt((u ** 2).sum() / ref_len) * power_noise
        leakage = np.random.rand(1)
        u = np.clip(u, self.converter_model[0].voltages[0] * maximum - leakage * maximum,
                    self.converter_model[0].voltages[1] * maximum + leakage * maximum)
        return u[:ref_len]

    def _maximal_bandwidth(self,max_band=20):
        """
         Computes the maximal allowed bandwidth, considering a user defined limit and the technical limit.
         First choose the minimal bandwidth of the externally excited motor

         Args:
            max_band: Maximal user defined value for the bandwidth

        Returns:
            Maximal bandwidth for the reference
         """
        return min(min(self.motor_model.bandwidth()), max_band)

    def _set_initial_state(self):
        """
        Find random an initial state, that is plausible.
        """
        omega = ((self.observation_space.high[OMEGA_IDX] - self.observation_space.low[OMEGA_IDX])
                 * np.random.rand() + self.observation_space.low[OMEGA_IDX]) * self._limits[OMEGA_IDX]\
                 / self._safety_margin
        i_e = ((self.observation_space.high[I_E_IDX] - self.observation_space.low[I_E_IDX])
               * np.random.rand() + self.observation_space.low[I_E_IDX]) * self._limits[I_E_IDX] / self._safety_margin
        i_a_max = self.motor_model.i_a_max(np.abs(omega))
        i_a = ((self.observation_space.high[I_A_IDX] - self.observation_space.low[I_A_IDX]) *
               np.random.rand() + self.observation_space.low[I_A_IDX]) * i_a_max
        self._state = np.zeros_like(self._state)
        self._state[[OMEGA_IDX, I_A_IDX, I_E_IDX]] = omega, i_a, i_e
        self._state[TORQUE_IDX] = self.motor_model.torque(self._state[MOTOR_IDX])
        self._state = np.clip(self._state, -self._limits / self._safety_margin, self._limits / self._safety_margin)

    def get_motor_param(self):
        """
        This function returns all motor parameters, sampling time, safety margin and converter limits

        Returns:
            All parameters
        """
        params = self.motor_parameter
        params['tau'] = self.tau
        params['safety_margin'] = self.safety_margin
        params['converter_voltage'] = self.converter_model[0].voltages
        params['converter_current'] = self.converter_model[0].currents
        return params


class DCMExtExDisc(_DCMExtEx):

    def __init__(self, converter_type='4Q', tau=1e-5, **kwargs):
        super().__init__(converter_type=f'disc-{converter_type}', tau=tau, **kwargs)
        self.action_space = spaces.MultiDiscrete([self.converter_model[0].action_space.n,
                                                  self.converter_model[1].action_space.n])

    def _step_integrate(self, action):
        """
        The integration is done for one time period. The converter considers the dead time and interlocking time.

        Args:
            action: switching state of the converter that should be applied
        """
        self._state[U_SUP_IDX] = self._reference[U_SUP_IDX][self._k] * self._limits[U_SUP_IDX]
        i_in = self.motor_model.i_in(self._state)
        u_in = np.array([self.converter_model[0].convert(action[0], i_in[0], self.system.t),
                         self.converter_model[1].convert(action[1], i_in[1], self.system.t)]) * self._state[U_SUP_IDX]
        self.system.set_f_params(u_in, self._noise[VOLTAGES, self._k])
        state = self.system.integrate(self.system.t + self.converter_model[0].interlocking_time)\
                + self._noise[self.MOTOR_IDX, self._k]

        # input voltage after switching
        i_in = self.motor_model.i_in(self._state)
        u_in = np.array([self.converter_model[0].convert(action[0], i_in[0], self.system.t),
                         self.converter_model[1].convert(action[1], i_in[1], self.system.t)]) * self._state[6]
        self.system.set_f_params(u_in, self._noise[VOLTAGES, self._k])
        state = self.system.integrate(self.system.t + self.tau - self.converter_model[0].interlocking_time)
        self._state[MOTOR_IDX] = state
        self._state[TORQUE_IDX] = self.motor_model.torque(state)
        self._state[VOLTAGES] = u_in

    def get_motor(self):
        return self.motor_model.motor_parameter


class DCMExtExCont(_DCMExtEx):

    def __init__(self, converter_type='4Q', tau=1e-4, **kwargs):
        super().__init__(converter_type=f'cont-{converter_type}', tau=tau, **kwargs)
        self.action_space = spaces.Box(
            np.array([self.converter_model[0].action_space.low[0], self.converter_model[1].action_space.low[0]]),
            np.array([self.converter_model[0].action_space.high[0], self.converter_model[1].action_space.high[0]])
        )

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
        return super().step(np.clip(action, self.action_space.low, self.action_space.high))

    def _step_integrate(self, action):
        """
        The integration is done for one time period. The converter considers the dead time and interlocking time.

        Args:
            action: switching state of the converter that should be applied
        """
        self._state[U_SUP_IDX] = self._reference[U_SUP_IDX][self._k]*self._limits[U_SUP_IDX]
        i_in = self.motor_model.i_in(self._state)
        u_in = np.array([self.converter_model[0].convert(action[0], i_in[0], self.system.t),
                         self.converter_model[1].convert(action[1], i_in[1], self.system.t)]) * self._state[U_SUP_IDX]
        self.system.set_f_params(u_in, self._noise[VOLTAGES, self._k])
        state = self.system.integrate(self.system.t + self._tau)
        self._state[MOTOR_IDX] = state
        self._state[TORQUE_IDX] = self.motor_model.torque(state)
        self._state[VOLTAGES] = u_in

