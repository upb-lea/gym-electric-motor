import numpy as np

from .dcm_base_env import _DCMBaseEnv
from gym.spaces import Box


class _DCMPermEx(_DCMBaseEnv):
    """
        **Description:**
            Environment for a controlled DC Permanently  Excited Motor.
        **Observation:**
            Type: Dict(Box,Box,Box,Box)

            +-------+-------------------+
            |Name   |Observation        |
            +=======+===================+
            |omega  |Angular Velocity   |
            +-------+-------------------+
            |torque |Electrical Torque  |
            +-------+-------------------+
            |i      |Armature Current   |
            +-------+-------------------+
            |u      |Armature Voltage   |
            +-------+-------------------+
            |u_sup  |Supply Voltage     |
            +-------+-------------------+
            | References                |
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

    OMEGA_IDX = 0
    TORQUE_IDX = 1
    I_IDX = 2
    U_IDX = 3
    U_SUP_IDX = 4
    MOTOR_IDX = [OMEGA_IDX, I_IDX]

    def __init__(self, zero_refs=(), **kwargs):
        super().__init__('DcPermEx', state_vars=['omega', 'torque', 'i', 'u', 'u_sup'],
                         zero_refs=tuple(zero_refs)+('u',), **kwargs)

    def _set_initial_state(self):
        """
        Set a plausible random initial state for the next episode.

        """
        # Set the speed somewhere in the Limits
        omega = ((self.observation_space.high[self.OMEGA_IDX] - self.observation_space.low[self.OMEGA_IDX])
                 * np.random.rand() + self.observation_space.low[self.OMEGA_IDX]) * self._limits[self.OMEGA_IDX]\
                 / self._safety_margin

        # Discount for the current due to the induced voltage
        i_max = self.motor_model.i_a_max(np.abs(omega))
        i = ((self.observation_space.high[self.I_IDX] - self.observation_space.low[self.I_IDX]) *
             np.random.rand() + self.observation_space.low[self.I_IDX]) * i_max

        # Write the state
        self._state = np.zeros_like(self._state)
        self._state[self.OMEGA_IDX] = omega
        self._state[self.I_IDX] = i
        self._state[self.TORQUE_IDX] = self.motor_model.torque([omega, i])
        self._state = np.clip(self._state, -self._limits / self._safety_margin, self._limits / self._safety_margin)

    def _set_limits(self):
        """
        Sum up all upper limits in an array in the same order as the states.
        """
        self._limits = np.array([
            self.motor_model.motor_parameter['omega_N'],
            self.motor_model.motor_parameter['torque_N'],
            self.motor_model.motor_parameter['i_N'],
            self.motor_model.motor_parameter['u_N'],
            self.motor_model.motor_parameter['u_N']
        ]) * self._safety_margin

    def _set_observation_space(self):
        """
        Set the observation space including all quantities and reference values.

        Possible values are -1, 0, 1 and the voltage limits also defines the values for the speed and in the same
        way the current defines the signs of the torque limits.
        """
        # normalised values for the voltages and currents
        u_min, u_max = self.converter_model.voltages
        i_min, i_max = self.converter_model.currents
        # Set up the observation space
        low = np.array([
            u_min,  # omega
            i_min,  # torque
            i_min,  # i
            u_min,  # u
            0.0     # u_sup
        ])
        high = np.array([
            u_max,  # omega
            i_max,  # torque
            i_max,  # i
            u_max,  # u
            u_max   # u_sup
        ])

        # Set the observation space for the references. Its ranges are equal to the range of the matching state var.
        for ref in range(len(self.state_vars)):
            if not self.reference_vars[ref]:
                continue
            mul_min, mul_max = {
                'omega': (u_min, u_max),
                'torque': (u_min, i_max),
                'i': (i_min, i_max),
                'u': (u_min, u_max)
            }[self.state_vars[ref]]
            low = np.concatenate((low, np.array(mul_min * np.ones(int(self._prediction_horizon+1)))))
            high = np.concatenate((high, mul_max * np.ones(int(self._prediction_horizon+1))))
            self.observation_space = Box(low, high, dtype=np.float32)

    def _set_initial_value(self):
        """
        Set the initial value for the integration.
        """
        self.system.set_initial_value(self._state[self.MOTOR_IDX], 0.0)

    def _generate_random_references(self):
        """
        This function generates random references for the next episode.

        First a random input sequence is build. Afterwards the system is
        fed with this input sequence and the resulting trajectories for all quantities are clipped to the limits and
        used as references. For all quantities, that should not have a reference, it is set to zero.
        """
        motor_params = self.motor_model.motor_parameter
        u = self._generate_random_control_sequence(self.motor_model.bandwidth(), motor_params['u_sup'])
        i, torque, omega = np.zeros((3, len(u)))
        u_sup = np.ones(len(u)) * self.motor_model.u_sup  # supply voltage
        state = np.random.triangular(
            self.observation_space.low[self.MOTOR_IDX] * self._limits[self.MOTOR_IDX] / self._safety_margin,
            self._state[self.MOTOR_IDX],
            self.observation_space.high[self.MOTOR_IDX] * self._limits[self.MOTOR_IDX] / self._safety_margin
        )

        for k in range(1, len(u)):
            state += self.motor_model.model(state, self.load_model.load(state[self.OMEGA_IDX]), u[k-1]) * self._tau
            state = np.clip(
                state,
                self.observation_space.low[self.MOTOR_IDX]*self._limits[self.MOTOR_IDX] / self._safety_margin,
                self.observation_space.high[self.MOTOR_IDX] * self._limits[self.MOTOR_IDX] / self._safety_margin
            )
            omega[k], i[k] = state
            torque[k] = self.motor_model.torque(state)
        self._reference = np.array([omega, torque, i, u, u_sup], subok=True)
        lim = self._limits
        self._reference = self._reference / lim.reshape(len(lim), 1)


class DCMPermExDisc(_DCMPermEx):
    def __init__(self, converter_type='4Q', tau=1e-5, **kwargs):
        super().__init__(converter_type=f'disc-{converter_type}', tau=tau, **kwargs)

    def _step_integrate(self, action):
        """
        The integration is done for one time period. The converter considers the dead time and the interlocking time.

        Args:
            action: switching state of the converter that should be applied
        """
        # input voltage in interlocking time
        self._state[self.U_SUP_IDX] = self._reference[self.U_SUP_IDX][self._k]*self._limits[self.U_SUP_IDX]
        i_in = self.motor_model.i_in(self._state)
        u_in = self.converter_model.convert(action, i_in, self.system.t) * self._state[self.U_SUP_IDX]
        self.system.set_f_params(u_in, self._noise[self.U_IDX, self._k])
        state = self.system.integrate(self.system.t + self.converter_model.interlocking_time)

        # input voltage after switching
        i_in = self.motor_model.i_in(self._state)
        u_in = self.converter_model.convert(action, i_in, self.system.t) * self._state[self.U_SUP_IDX]
        self.system.set_f_params(u_in, self._noise[self.U_IDX, self._k])
        state = self.system.integrate(self.system.t + self.tau - self.converter_model.interlocking_time)
        self._state[self.MOTOR_IDX] = state
        self._state[self.TORQUE_IDX] = self.motor_model.torque(state)
        self._state[self.U_IDX] = u_in


class DCMPermExCont(_DCMPermEx):
    def __init__(self, converter_type='4Q', tau=1e-4, **kwargs):
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
        The integration is done for one time period. The converter considers the dead time and the interlocking time.

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
