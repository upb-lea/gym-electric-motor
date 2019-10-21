import gym
import numpy as np
from .models.converter_models import Disc2Level3PhaseConverter, Cont2Level3PhaseConverter
from .models.load_models import Load
from .models.pmsm_model import PmsmModel
from .models import pmsm_model
from ..utils import EulerSolver
from scipy.integrate import ode
from ..dashboard import MotorDashboard

OMEGA_IDX = 0
TORQUE_IDX = 1
I_A_IDX = 2
I_B_IDX = 3
I_C_IDX = 4
U_A_IDX = 5
U_B_IDX = 6
U_C_IDX = 7
EPSILON_IDX = 8
U_SUP_IDX = 9
CURRENTS = [I_A_IDX, I_B_IDX, I_C_IDX]
VOLTAGES = [U_A_IDX, U_B_IDX, U_C_IDX]


class PmsmEnv(gym.Env):
    """
    Gym Environment for a Permanent Magnet Synchronous Motor (PMSM).

    **Description**:
        This class contains the environment of a PMSM.

    **Source:**

    **State Variables**:
        +-----+-------+-------------------------------------------------------------+
        |Index| State | Description                                                 |
        +=====+=======+=============================================================+
        | 0   | omega | mechanical angular velocity of the motor in rad/s           |
        +-----+-------+-------------------------------------------------------------+
        | 1   | torque| generated torque by the motor in Nm                         |
        +-----+-------+-------------------------------------------------------------+
        | 2   | i_a   | current flowing into branch a in A                          |
        +-----+-------+-------------------------------------------------------------+
        | 3   | i_b   | current flowing into branch b in A                          |
        +-----+-------+-------------------------------------------------------------+
        | 4   | i_c   | current flowing into branch c in A                          |
        +-----+-------+-------------------------------------------------------------+
        | 5   | u_a   | voltage applied between branch a and ground in V            |
        +-----+-------+-------------------------------------------------------------+
        | 6   | u_b   | voltage applied between branch b and ground in V            |
        +-----+-------+-------------------------------------------------------------+
        | 7   | u_c   | voltage applied between branch c and ground in V            |
        +-----+-------+-------------------------------------------------------------+
        | 8   |epsilon| mechanical angle of the rotor in rad                        |
        +-----+-------+-------------------------------------------------------------+
        | 9   | u_sup | supply voltage                                              |
        +-----+-------+-------------------------------------------------------------+

    **Observation**:

        Type: Box(10 + Number of reference variables * (prediction horizon + 1))

        The observation consists of the current normalized values for the state variables in the limit range [0, 1] or
        [-1, 1] concatenated with the current and future references of the reference variables that are no zero
        references.


        *Example*:
            Reference Variables: omega and i_a with prediction horizon 1
            ::
                [omega, torque, i_a, i_b, i_c, u_a, u_b, u_c, epsilon,  u_sup,
                omega_ref[k], omega_ref[k+1], i_a_ref[k], i_a_ref[k+1]]

    **Actions**:

        The actions are the direct switching states of the transistors for the converter for the discrete case.

        Type: Discrete(8)

        +---+---------+-----------+--------------+
        |Num|Switch A | Switch B  |     Switch C |
        +===+=========+===========+==============+
        |0  |lower    | lower     |     lower    |
        +---+---------+-----------+--------------+
        |1  |lower    | lower     |     upper    |
        +---+---------+-----------+--------------+
        |2  |lower    | upper     |     lower    |
        +---+---------+-----------+--------------+
        |3  |lower    | upper     |     upper    |
        +---+---------+-----------+--------------+
        |4  |upper    | lower     |     lower    |
        +---+---------+-----------+--------------+
        |5  |upper    | lower     |     upper    |
        +---+---------+-----------+--------------+
        |6  |upper    | upper     |     lower    |
        +---+---------+-----------+--------------+
        |7  |upper    | upper     |     upper    |
        +---+---------+-----------+--------------+

        A lower position means 0V and upper position means +u_dc.

        In the continuous environment the actions are the modulated switching states.
        Type: Box(3)

        +----+-------------------------------+
        | Num|Description                    |
        +====+===============================+
        | 0  |Duty Cycle for Switch A [-1, 1]|
        +----+-------------------------------+
        | 1  |Duty Cycle for Switch B [-1, 1]|
        +----+-------------------------------+
        | 2  |Duty Cycle for Switch C [-1, 1]|
        +----+-------------------------------+

    **Reward:**
        The reward is selectable between several reward functions. See below for further information.

    **Starting State:**
        The angular velocity is initialized randomly between [-1, 1]
        The Amplitude of the current vector is initialized randomly between [-1, 1]
        The Phase of the current vector is initialized randomly between [0, 2 * pi]
        By this the currents i_a, i_b, i_c are calculated.
        All other states are initialized with zero.

    **Episode Termination:**
        An episode terminates when episode_length steps have been taken.
        Furthermore, an episode might end, if limits of the motor are violated.
        This behaviour can be selected with the limit_observer.

    """

    # region properties

    @property
    def prediction_horizon(self):
        return self._prediction_horizon

    @property
    def reference_vars(self):
        return self._reference_vars

    # endregion

    def __init__(self, converter_type, zero_refs=(), tau=1e-4, episode_length=100000, load_parameter=None,
                 motor_parameter=None, reward_weight=(('omega', 1.0),), on_dashboard=('True',), integrator='euler',
                 nsteps=1, prediction_horizon=0, interlocking_time=0.0, noise_levels=0.0, reward_fct='swsae',
                 limit_observer='off', dead_time=True, safety_margin=1.3, gamma=0.9):
        """
        Basic setting of all the common motor parameters.

        Args:
            converter_type: Selection of discrete or continuous converter. Either 'Disc' or 'Cont'
            zero_refs(list(str)): Selection of reference variables that should get zero references
             to keep their value low.
            tau: sampling time
            episode_length: The episode length of the environment
            load_parameter: A dict of Load parameters that differ from the default ones. \
                           For details look into the load model.
            motor_parameter: A dict of motor parameters that differ from the default ones. \
                            For details look into the dc_motor model.
            reward_weight: Iterable of key/value pairs that specifies how the rewards in the environment
                          are weighted.
                          E.g. ::
                            (('omega', 0.9),('u_a', 0.1))
            on_dashboard(list(str) with entries that are names of state variables):
                             Set the dashboard variables that shall be displayed.
                             If on_dashboard is ['True'] then all variables will be displayed
                             If on_dashboard is ['False'] then no variables will be displayed
            integrator: Select integration method between 'dopri5' and 'euler'
            nsteps: Maximum number of steps the integrator takes
            prediction_horizon: number of future reference values included in the observation
            interlocking_time: interlocking time of the converter
            noise_levels: Setting of noise levels in percentage of signal power for each variable. \n
                             If the noise level is integer, then the noise level is equal for each state var. \n
                             If noise level is of type iterable(('state var', float))
                             then each state var has different noise. \n
                             Variables that are not in the iterable are assigned 0 noise
            reward_fct: Select the reward function between: (Each one normalised to [0,1] or [-1,0]) \n
                'swae': Absolute Error between references and state variables [-1,0] \n
                'swse': Squared Error between references and state  variables [-1,0]\n
                'swsae': Shifted absolute error / 1 + swae [0,1] \n
                'swsse': Shifted squared error  / 1 + swse [0,1]  \n

            limit_observer: Select the limit observing function. \n
                'off': No limits are observed. Episode goes on. \n
                'no_punish': Limits are observed, no punishment term for violation.
                    This function should be used with shifted reward functions. \n
                'const_punish': Limits are observed. Punishment in the form of -1 / (1-gamma) to
                    punish the agent with the maximum negative reward for the further steps.
                    This function should be used with non shifted reward functions.
            dead_time: specifies if dead time of one sampling interval will be considered
            gamma: Parameter for the const_punish limit observer. If this one is not chosen gamma is unused.
                Specifies the punishment height for limit violations. Should roughly equal the agents discount factor
                gamma.
        """

        # Set State variables and the inverse state var dictionary
        self._gamma = gamma
        self._safety_margin = safety_margin
        self.state_vars = np.array(['omega', 'torque', 'i_a', 'i_b', 'i_c', 'u_a', 'u_b', 'u_c', 'epsilon', 'u_sup'])
        self._state_var_pos = {}
        for val, key in enumerate(self.state_vars):
            self._state_var_pos[key] = val

        self._tau = tau
        self._episode_length = episode_length
        self._prediction_horizon = max(0, prediction_horizon)

        self._zero_refs = tuple(zero_refs) + ('u_a', 'u_b', 'u_c')
        # Flag array to select zero references faster
        self._zero_ref_flags = np.isin(self.state_vars, self._zero_refs)

        #: Set the converter, load and Motor
        if converter_type == 'Disc':
            self.converter = Disc2Level3PhaseConverter(interlocking_time, tau, dead_time)
        elif converter_type == 'Cont':
            self.converter = Cont2Level3PhaseConverter(interlocking_time, tau, dead_time)
        else:
            raise AssertionError(f'converter_type was {converter_type} and must be in ["Disc", "Cont"]')
        self.action_space = self.converter.action_space
        self.load = Load(load_parameter)
        self.motor = PmsmModel(motor_parameter, self.load.load, self.load.j_load)
        self._k = 0

        # Initialize the state arrays
        self._state = np.zeros_like(self.state_vars, dtype=float)
        self._dashboard = None
        self._references = np.zeros((len(self.state_vars), episode_length + self._prediction_horizon))
        self._noise = np.zeros_like(self._references)
        self._reward_weights = np.zeros(len(self.state_vars))
        for key, value in reward_weight:
            self._reward_weights[self._state_var_pos[key]] = value

        # Setting of noise levels in percentage of signal power for each variable
        if type(noise_levels) is float or type(noise_levels) is int:
            self._noise_levels = np.ones(len(self.state_vars)) * noise_levels
        else:
            self._noise_levels = np.zeros_like(self.state_vars)
            for key, value in noise_levels:
                self._noise_levels[self._state_var_pos[key]] = value

        # Set the dashboard variables into the array _on_dashboard
        self._on_dashboard = np.ones(len(self.state_vars), dtype=bool)
        if on_dashboard[0] == 'True':
            self._on_dashboard *= True
        elif on_dashboard[0] == 'False':
            self._on_dashboard *= False
        else:
            self._on_dashboard *= False
            for key in on_dashboard:
                self._on_dashboard[self._state_var_pos[key]] = True
        mp = self.motor.motor_parameter
        self._limits = np.array([
            mp['omega_N'],
            mp['torque_N'],
            mp['i_N'], mp['i_N'], mp['i_N'],
            # positive and negative voltages are possible due to a shifted supply voltage
            mp['u_N'] / 2, mp['u_N'] / 2, mp['u_N'] / 2,
            2 * np.pi,
            mp['u_N']
        ]) * self._safety_margin

        # Select integration method
        if integrator == 'euler':
            self.system = EulerSolver(self._system_eq, nsteps)
        else:
            self.system = ode(self._system_eq).set_integrator(integrator, nsteps=nsteps)

        self._limit_observer = self._limit_observers(limit_observer)
        self._reward_fct = self._reward_functions(reward_fct)
        self._resetDashboard = True
        self._reference_vars = (self._reward_weights > 0) & ~self._zero_ref_flags

        # observation space
        u_min, u_max = self.converter.voltages
        i_min, i_max = self.converter.currents
        obs_high = np.ones(len(self.state_vars))
        obs_low = np.array([-1,     # omega
                            -1,  # torque
                            i_min,  # i_a
                            i_min,  # i_b
                            i_min,  # i_c
                            u_min,  # u_a
                            u_min,  # u_b
                            u_min,  # u_c
                            0,      # epsilon
                            0])      # u_sup

        # Set the observation space for the references. Its ranges are equal to the range of the matching state var.
        for ref in range(len(self.state_vars)):
            if self._reward_weights[ref] <= 0 or self.state_vars[ref] in self._zero_refs:
                continue
            mul_min, mul_max = {
                'omega': (-1, 1),
                'torque': (u_min, u_max),
                'i_a': (i_min, i_max),
                'i_b': (i_min, i_max),
                'i_c': (i_min, i_max),
                'u_a': (u_min, u_max),
                'u_b': (u_min, u_max),
                'u_c': (u_min, u_max),
                'epsilon': (0, 1)
            }[self.state_vars[ref]]
            obs_low = np.concatenate((obs_low, np.array(mul_min * np.ones(int(self._prediction_horizon + 1)))))
            obs_high = np.concatenate((obs_high, mul_max * np.ones(int(self._prediction_horizon + 1))))

        self.observation_space = gym.spaces.Box(obs_low, obs_high)

    # region environment functions

    def reset(self):
        """
        Resets the environment for another episode.

        The step counter is reset to 0. \
        The starting state is initialized as described in the class documentation. \
        New References are generated either deterministic (sinusoidal, triangular, sawtooth or step) or randomly. \
        The dashboard is reset. \

        Returns:
              The observation of the initial state.
        """
        self._k = 0
        self._state = np.zeros_like(self._state)
        self._state[OMEGA_IDX] = np.random.triangular(-1, 0, 1)
        phase = 2 * np.pi * np.random.rand()
        amplitude = np.random.triangular(-1, 0, 1)

        self._state[I_A_IDX] = amplitude * np.cos(phase)
        self._state[I_B_IDX] = amplitude * np.cos(phase + np.pi / 3)
        self._state[I_C_IDX] = amplitude * np.cos(phase - np.pi / 3)

        self._generate_references()
        initial_motor = np.concatenate((
            [self._state[OMEGA_IDX]],
            self.motor.q_inv_me(self.motor.t_23(self._state[CURRENTS]), self._state[EPSILON_IDX]),
            [self._state[EPSILON_IDX]]
        )) * self._limits[[OMEGA_IDX, I_A_IDX, I_A_IDX, EPSILON_IDX]]
        self.system.set_initial_value(initial_motor, 0.0)
        self._resetDashboard = True
        self._noise = (
            np.sqrt(self._noise_levels/6) / self._safety_margin
            * np.random.randn(self._episode_length+1, len(self.state_vars))
        ).T
        observation_references = self._references[self._reference_vars, self._k:self._k + self._prediction_horizon + 1]
        observation = np.concatenate((self._state, observation_references.flatten()))
        return observation

    def step(self, action):
        """
        Clips the action to its limits and performs one step on the motor.

        This method performs one step by first calculating the input voltages at the terminals of the motor by the
        action on the converter.

        Args:
            action: The action from the Action space that will be performed on the motor

        Returns:
            Tuple(array(float), float, bool, dict):
                **observation:**    The observation from the environment \n
                **reward:**         The reward for the taken action \n
                **bool:**           Flag if the episode has ended \n
                **info:**           An always empty dictionary \n
        """
        state, u_in = self._step_integrate(action)
        i_dq = state[[self.motor.I_D_IDX, self.motor.I_Q_IDX]]
        i_abc = self.motor.t_32(self.motor.q_me(i_dq, state[self.motor.EPSILON_IDX]))
        self._state[OMEGA_IDX] = state[self.motor.OMEGA_IDX]
        self._state[TORQUE_IDX] = self.motor.torque(state)
        self._state[CURRENTS] = i_abc
        self._state[VOLTAGES] = u_in
        self._state[EPSILON_IDX] = (state[self.motor.EPSILON_IDX] % (2 * np.pi))
        self._state /= self._limits
        rew = self._reward_fct(self._state, self._references[:, self._k].T)
        done, punish = self._limit_observer(self._state)
        if done:
            rew = punish
        observation_references = self._references[self._reference_vars, self._k:self._k + self._prediction_horizon + 1]
        self._state += self._noise[:, self._k]  # Add measurement noise
        observation = np.concatenate((self._state, observation_references.flatten()))
        self._k += 1
        if not done:
            done = self._k == self._episode_length
        return observation, rew, done, {}

    def close(self):
        """
        Cleanup for the environment.

        When the environment is closed the dashboard will also be closed.
        This function does not need to be called explicitly.
        """
        if self._dashboard is not None:
            self._dashboard.close()

    def render(self, mode='human'):
        """
        Update the visualization step by step.

        Call this function once a cycle to update the visualization with the current values.
        """
        if not self._on_dashboard.any():
            return
        if self._dashboard is None:
            self._dashboard = MotorDashboard(
                self.state_vars[self._on_dashboard], self._tau,
                self.observation_space.low[:len(self.state_vars)][self._on_dashboard]
                * self._limits[self._on_dashboard],
                self.observation_space.high[:len(self.state_vars)][self._on_dashboard]
                * self._limits[self._on_dashboard],
                self._episode_length,
                self._safety_margin,
                self._reward_weights[self._on_dashboard] > 0
            )
        if self._resetDashboard:
            self._resetDashboard = False
            dash_limits = self._limits[self._on_dashboard]
            self._dashboard.reset(self._references[self._on_dashboard] * dash_limits.reshape((len(dash_limits), 1)))
        self._dashboard.step(self._state[self._on_dashboard] * self._limits[self._on_dashboard], self._k)

    # endregion

    def _step_integrate(self, action):
        """
        Implemented by the child classes.
        """
        raise NotImplementedError

    def _system_eq(self, t, state, u_in):
        """
        The differential equation of the system.

        This function is called by the ODE-Solver. \
        It is in the following Form: \
            d_state/dt = f(state)

        Args:
            t: The parameter for the current time. Required for the solvers. Not needed here.
            state: The motor state in the form: \n
                `[omega, i_q, i_d, epsilon]`
            u_in: Input voltages `[u_q, u_d]`

        Returns:
            The solution of the PMSM differential equation
        """
        t_load = self.load.load(state[self.motor.OMEGA_IDX])
        return self.motor.model(state, t_load, u_in)

    # region Limit Observers

    def _limit_observers(self, key):
        return {
            'off': self._no_observation,
            'no_punish': self._no_punish,
            'const_punish': self._const_punish,
        }[key]

    def _no_observation(self, _):
        """
        No limit violations are observed. No punishment and the episode continues even after limit violations.

        Args:
            state: Current state of the motor

        Returns:
            Tuple of a flag if the episode should be terminated (here always false)
            and the punishment for the reward (here always 0)
        """
        return False, 0.0

    def _const_punish(self, state):
        """
           Punishment, if constraints are violated and termination of the episode.

           The punishment equals -1 / (1 - self.gamma), which is equivalent to the by gamma discounted reward a learner
           would receive, if it receives always the minimum reward after the limit violation.

           This punishment is recommended, when taking a negative reward function.

           Args:
               state: Current state of the environment

           Returns:
               Tuple of a flag if the episode should be terminated and the punishment for the reward

        """
        if (state < self._limits).all() and (state > -self._limits).all():
            return False, 0.0
        else:
            return True, -1 * 1 / (1 - self._gamma)

    def _no_punish(self, state):
        """
        No reward punishment, only break the episode when limits are violated. Recommended for positive rewards.

        Args:
            state: Current state of the environment

        Returns:
            Tuple of a flag if the episode should be terminated and the punishment for the reward
        """
        if (state < self._limits).all() and (state > -self._limits).all():
            return False, 0.0
        else:
            return True, 0.0
    # endregion

    # region Reward Functions

    def _reward_functions(self, key):
        return {
            'swae': self._absolute_error,
            'swse':  self._squared_error,
            'swsae': self._shifted_absolute_error,
            'swsse': self._shifted_squared_error
        }[key]

    def _absolute_error(self, state, reference, *_):
        return -(self._reward_weights * np.abs(state - reference)).sum()

    def _squared_error(self, state, reference, *_):
        return -(self._reward_weights * (state - reference)**2).sum()

    def _shifted_squared_error(self, state, reference, *_):
        return 1 + self._squared_error(state, reference)

    def _shifted_absolute_error(self, state, reference, *_):
        return 1 + self._absolute_error(state, reference)

    # endregion

    # region Reference Generation

    def _reference_sin(self,  bandwidth=20):
        """
        Set sinus references for the state variables with a random amplitude, offset and phase shift

        Args:
            bandwidth: bandwidth of the system
        """
        x = np.arange(0, (self._episode_length + self._prediction_horizon))
        if self.observation_space.low[OMEGA_IDX] == 0.0:
            amplitude = np.random.rand() / 2
            offset = np.random.rand() * (1 - 2 * amplitude) + amplitude
        else:
            amplitude = np.random.rand()
            offset = (2 * np.random.rand() - 1) * (1 - amplitude)
        t_min, t_max = self._set_time_interval_reference('sin', bandwidth)  # specify range for period time
        t_s = np.random.rand() * (t_max - t_min) + t_min
        phase_shift = 2 * np.pi * np.random.rand()
        self._references = amplitude * np.sin(2 * np.pi / t_s * x * self._tau + phase_shift) + offset
        self._references = self._references * np.ones(
            (len(self.state_vars), 1)) / self._safety_margin  # limit to nominal values

    def _reference_rect(self, bandwidth=20):
        """
        Set rect references for the state variables with a random amplitude, offset and phase shift

        Args:
            bandwidth: bandwidth of the system
        """
        x = np.arange(0, (self._episode_length + self._prediction_horizon))
        if self.observation_space.low[0] == 0.0:
            amplitude = np.random.rand()
            offset = np.random.rand()*(1-amplitude)
        else:
            amplitude = 2 * np.random.rand() - 1
            offset = (-1 + np.random.rand() * (2 - np.abs(amplitude))) * np.sign(amplitude)

        t_min, t_max = self._set_time_interval_reference('rect', bandwidth)
        t_s = np.random.rand() * (t_max - t_min) + t_min  # specify range for period time
        t_on = np.random.rand() * t_s     # time period on amplitude + offset value
        t_off = t_s - t_on                # time period on offset value
        reference = np.zeros(self._episode_length + self._prediction_horizon)
        reference[x * self._tau % (t_on + t_off) > t_off] = amplitude
        reference += offset
        self._references = reference * np.ones((len(self.state_vars), 1)) / self._safety_margin

    def _reference_tri(self,  bandwidth=20):
        """
        set triangular reference with random amplitude, offset, times for rise and fall

        Args:
            bandwidth: bandwidth of the system
        """
        t_min, t_max = self._set_time_interval_reference('tri', bandwidth)  # specify range for period time
        t_s = np.random.rand() * (t_max - t_min) + t_min
        t_rise = np.random.rand() * t_s
        t_fall = t_s - t_rise

        if self.observation_space.low[0] == 0.0:
            amplitude = np.random.rand()
            offset = np.random.rand() * (1 - amplitude)
        else:
            amplitude = 2 * np.random.rand() - 1
            offset = (-1 + np.random.rand() * (2 - np.abs(amplitude))) * np.sign(amplitude)

        reference = np.ones(self._episode_length + self._prediction_horizon)
        for t in range(0, (self._episode_length + self._prediction_horizon)):
            # use a triangular function
            if (t * self._tau) % t_s <= t_rise:
                reference[t] = ((t * self._tau) % t_s) / t_rise * amplitude + offset
            else:
                reference[t] = -((t * self._tau) % t_s - t_s) / t_fall * amplitude + offset
        self._references = reference * np.ones((len(self.state_vars), 1)) / self._safety_margin

    def _reference_sawtooth(self,  bandwidth=20):
        """
        sawtooth signal generator with random time period and amplitude

        Args:
            bandwidth: bandwidth of the system
        """
        t_min, t_max = self._set_time_interval_reference('sawtooth', bandwidth)  # specify range for period time
        t_s = np.random.rand() * (t_max - t_min) + t_min
        if self.observation_space.low[OMEGA_IDX] == 0.0:
            amplitude = np.random.rand()
        else:
            amplitude = 2 * np.random.rand() - 1
        x = np.arange(self._episode_length + self._prediction_horizon, dtype=float)
        self._references = np.ones_like(x, dtype=float)
        self._references *= (x*self._tau) % t_s * amplitude / t_s
        # limit to nominal values
        self._references = self._references * np.ones((len(self.state_vars), 1))/self._safety_margin

    def _set_time_interval_reference(self, shape=None,  bandwidth=20):
        """
        This function returns the minimum and maximum time period specified by the bandwidth of the motor,
        episode length and individual modifications for each shape
        At least on time period of a shape should fit in an episode, but not to fast that the motor can not follow the
        reference properly.

        Args:
            shape: shape of the reference

        Returns:
            minimal and maximal time period
        """
        bw = self._maximal_bandwidth(bandwidth)  # Bandwidth of reference limited
        t_episode = (self._episode_length+self._prediction_horizon)*self._tau
        t_min = min(1 / bw, t_episode)
        t_max = max(1 / bw, t_episode)
        # In this part individual modifications can be made for each shape
        # Modify the values to get useful references
        if shape == 'sin':
            t_min = t_min
            t_max = t_max / 5
        elif shape == 'rect':
            t_min = t_min
            t_max = t_max / 3
        elif shape == 'tri':
            t_min = t_min
            t_max = t_max / 5
        elif shape == 'sawtooth':
            t_min = t_min
            t_max = t_max / 5
        else:
            t_min = t_min
            t_max = t_max / 5
        return min(t_min, t_max), max(t_min, t_max)  # to make sure that the order is correct

    def _maximal_bandwidth(self,  bandwidth=20):
        """
        Computes the maximal allowed bandwidth, considering a user defined limit and the technical limit.

        Args:
           bandwidth: maximal user defined value for the bandwidth

        Returns:
           maximal bandwidth for the reference
        """
        return min(min(self.motor.bandwidth()), bandwidth)

    def _generate_references(self, bandwidth=20):
        """
        Select which reference to generate. The shaped references (rect, sin, triangular, sawtooth) are equally propable
        with 12,5% and a random reference is generated with a probability of 50%

        Args:
            bandwidth: bandwidth of the system
        """
        val = np.random.rand()

        if val < 0.125:
            self._reference_rect(bandwidth)
        elif val < 0.25:
            self._reference_sin(bandwidth)
        elif val < 0.375:
            self._reference_tri(bandwidth)
        elif val < 0.5:
            self._reference_sawtooth(bandwidth)
        else:
            self._generate_random_references()
        u_sup = np.ones(self._episode_length + self._prediction_horizon) * self.motor.u_sup
        self._references[U_SUP_IDX] = np.clip(
            u_sup,
            0.0,
            self._limits[U_SUP_IDX],
        )/self._limits[U_SUP_IDX]

        self._references[self._zero_ref_flags] = np.zeros((
            len(self._zero_refs),
            self._episode_length + self._prediction_horizon,
        ))

    def _generate_random_references(self):
        """
        This function generates random references for the next episode.

        First a random input sequence is build. Afterwards the system is
        fed with this input sequence and the resulting trajectories for all quantities are clipped to the limits and
        used as references. For all quantities, that should not have a reference, it is set to zero.
        """
        u_q = self._generate_random_control_sequence(self.motor.bandwidth()[0]) * self.motor.u_sup
        u_d = self._generate_random_control_sequence(self.motor.bandwidth()[1]) * self.motor.u_sup
        i_q, i_d, i_a, i_b, i_c, torque, omega, epsilon = np.zeros((8, len(u_q)))
        i_a_0, i_b_0, i_c_0, epsilon_0 = self._state[CURRENTS + [EPSILON_IDX]]
        i_d_0, i_q_0 = self.motor.q_inv_me(PmsmModel.t_23([i_a_0, i_b_0, i_c_0]), epsilon_0)
        state = np.array([self._state[OMEGA_IDX], i_q_0, i_d_0, epsilon_0]) \
            * self._limits[[OMEGA_IDX, I_A_IDX, I_B_IDX, EPSILON_IDX]]
        limits = self._limits[[OMEGA_IDX, I_A_IDX, I_B_IDX]] / self._safety_margin
        clip_low = np.concatenate((-limits, np.array([0])))
        clip_high = np.concatenate((limits, np.array([np.inf])))
        for k in range(1, len(u_q)):
            state += self.motor.model(state, self.load.load(state[OMEGA_IDX]),
                                      (u_q[k-1], u_d[k-1])) * self._tau
            state = np.clip(state, clip_low, clip_high)
            omega[k], i_q[k], i_d[k], epsilon[k] = state
            torque[k] = self.motor.torque(state)

            i_a[k], i_b[k], i_c[k] = self.motor.t_32(self.motor.q_me([i_q[k], i_d[k]], epsilon[k]))
        u_sup = np.ones_like(omega) * self.motor.u_sup
        self._references = np.array(
            [omega, torque,
             i_a, i_b, i_c,
             np.zeros(len(u_q)), np.zeros(len(u_d)), np.zeros(len(u_d)),
             epsilon % (2*np.pi), u_sup], subok=True
        )
        self._references /= self._limits.reshape((len(self._limits), 1))
        self._references[self._zero_ref_flags] = np.zeros(
            (len(self._zero_refs), self._episode_length + self._prediction_horizon)
        )

    def _generate_random_control_sequence(self, bw):
        """
        Function that is called by the random reference generation in the motors to generate a random control sequence.

        Args:
            bw: Bandwidth for the control sequence

        Returns:
            A random control sequence that is following the bandwidth and power constraints at most.
        """
        ref_len = self._episode_length + self._prediction_horizon
        rands = np.random.randn(2, ref_len // 2)
        u = rands[0] + 1j * rands[1]
        bw_noise = np.random.rand() + 1
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

        power_noise = np.random.rand() + 1
        u = u / np.sqrt((u ** 2).sum() / ref_len) * power_noise
        u = np.clip(u, -1/self._safety_margin, 1/self._safety_margin)
        return u[:ref_len]

    # endregion

    def get_motor_param(self):
        """
        This function returns all motor parameters, sampling time, safety margin and converter limits.

        Returns:
            A dictionary containing the motor parameters as well as the sampling time tau, the safety margin and the
            converter limits.
        """
        params = self.motor.motor_parameter
        params['tau'] = self._tau
        params['safety_margin'] = self._safety_margin
        params['converter_voltage'] = self.converter.voltages
        params['converter_current'] = self.converter.currents
        params['episode_length'] = self._episode_length
        params['prediction_horizon'] = self._prediction_horizon
        params['tau'] = self._tau
        params['limits'] = self._limits.tolist()
        return params


class PmsmDisc(PmsmEnv):
    def __init__(self, tau=1e-5, **kwargs):
        super().__init__('Disc', tau=tau, **kwargs)

    def _step_integrate(self, action):
        """
        This function contains the integration for the discrete type in two steps. First, it is integrated over the
        interlocking time and second integrated over the rest of the interval such that the whole integration takes ones
        the sampling time.
        In both cases first the converter determines the applied voltage. Afterwards follows the transformation from
        a/b/c to d/q coordinates and the integration.

        Args:
            action: current action from the controller

        Returns:
            New system state and applied input voltage
        """
        self._state[U_SUP_IDX] = self._references[U_SUP_IDX, self._k] * self._limits[U_SUP_IDX]
        u_in = self.converter.convert(action, self._state[CURRENTS], self.system.t) * self._state[U_SUP_IDX]
        u_dq = self.motor.q_inv_me(self.motor.t_23(u_in + self._noise[VOLTAGES, self._k]),  self._state[EPSILON_IDX]
                                   * self._limits[EPSILON_IDX])
        self.system.set_f_params([u_dq[1], u_dq[0]])
        state = self.system.integrate(self.system.t + self.converter.interlocking_time)
        u_in = self.converter.convert(action, self._state[CURRENTS], self.system.t) * self._state[U_SUP_IDX]
        u_dq = self.motor.q_inv_me(self.motor.t_23(u_in + self._noise[VOLTAGES, self._k]),  self._state[EPSILON_IDX]
                                   * self._limits[EPSILON_IDX])
        self.system.set_f_params([u_dq[1], u_dq[0]])
        # integrate rest of a period
        state = self.system.integrate(self.system.t + self._tau - self.converter.interlocking_time)
        return state, u_in


class PmsmCont(PmsmEnv):
    def __init__(self, tau=1e-4, **kwargs):
        super().__init__('Cont', tau=tau, **kwargs)

    def step(self, action):
        return super().step(np.clip(action, self.converter.action_space.low, self.converter.action_space.high))

    def _step_integrate(self, action):
        """
        This function executes the integration over one sampling period. First, the converter determines the applied
        input voltage. Afterwards follows the transformation from a/b/c to d/q coordinates and the integration.

        Args:
            action: Current action from the controller

        Returns:
            New system state and applied input voltage
        """

        # action from the controller and input currents
        self._state[U_SUP_IDX] = self._references[U_SUP_IDX, self._k] * self._limits[U_SUP_IDX]
        u_in = self.converter.convert(action, self._state[CURRENTS]) * self._state[U_SUP_IDX]
        # Convert Input voltage to dq space
        u_dq = self.motor.q_inv_me(self.motor.t_23(u_in + self._noise[VOLTAGES, self._k]),  self._state[EPSILON_IDX]
                                   * self._limits[EPSILON_IDX])
        self.system.set_f_params([u_dq[1], u_dq[0]])
        # Simulate the system for a step
        state = self.system.integrate(self.system.t + self._tau)
        return state, u_in

