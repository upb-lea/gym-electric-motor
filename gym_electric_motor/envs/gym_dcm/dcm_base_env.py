import gym
from scipy.integrate import ode
import numpy as np
import json

from .models import dcmotor_model, converter_models, load_models
from ..dashboard import MotorDashboard
from ..utils import EulerSolver


class _DCMBaseEnv(gym.Env):
    """
            **Description:**
                An abstract environment for common functions of the DC motors

            **Observation:**
                Specified by the concrete motor. It is always a concatenation of the state variables, voltages, torque
                and next reference values.

            **Actions:**
                Depending on the converter type the action space may be discrete or continuous

                Type: Discrete(2 / 3 / 4)

                Num	Action: Depend on the converter

                    1Q Converter: (only positive voltage and positive current)
                        - 0: transistor block
                        - 1: positive DC-link voltage applied

                    2Q Converter: (only positive voltage and both current directions)
                        - 0: both transistors blocking
                        - 1: positive DC-link voltage applied
                        - 2: 0V applied

                    4Q Converter (both voltage and current directions)
                        - 0:	short circuit with upper transistors, 0V applied
                        - 1:	positive DC-link voltage
                        - 2:    negative DC-link voltage
                        - 3:    short circuit with lower transistors, 0V applied

                Type: Box()
                    Defines the duty cycle for the transistors.\n
                    [0,  1]: 1Q and 2Q\n
                    [-1, 1]: 4Q

                    For an externally excited motor it is a two dimensional box from [-1, 1] or [0, 1]

            **Reward:**
                The reward is the cumulative squared error (se) or the cumulative absolute error (ae) between the
                current value and the current reference of the state variables. Both are also available in a shifted
                form with an added on such that the reward is positive. More details are given below.
                The variables are normalised by their maximal values and weighted by the reward_weights.

            **Starting State:**
                All observations are assigned a random value.

            **Episode Termination**:
                An episode terminates, when all the steps in the reference have been simulated
                or a limit has been violated.

            **Attributes:**
                +----------------------------+----------------------------------------------------------+
                | **Name**                   |  **Description**                                         |
                +============================+==========================================================+
                | **state_vars**             | Names of all the quantities that can be observed         |
                +----------------------------+----------------------------------------------------------+
                | **state_var_positions**    | Inverse dict of the state vars. Mapping of key to index. |
                +----------------------------+----------------------------------------------------------+
                | **limits**                 | Maximum allowed values of the state variables            |
                +----------------------------+----------------------------------------------------------+
                | **reward_weights**         | Ratio of the weight of the state variable for the reward |
                +----------------------------+----------------------------------------------------------+
                | **on_dashboard**           | Flag indicating if the state var is shown on dashboard   |
                +----------------------------+----------------------------------------------------------+
                | **noise_levels**           | Percentage of the noise power to the signal power        |
                +----------------------------+----------------------------------------------------------+
                | **zero_refs**              | State variables that get a fixed zero reference          |
                +----------------------------+----------------------------------------------------------+

    """

    OMEGA_IDX = 0
    MOTOR_IDX = None

    # region Properties

    @property
    def tau(self):
        """
        Returns:
            the step size of the environment Default: 1e-5 for discrete / 1e-4 for continuous action space
        """
        return self._tau

    @property
    def episode_length(self):
        """
        Returns:
            The length of the current episode
        """
        return self._episode_length

    @episode_length.setter
    def episode_length(self, episode_length):
        """
        Set the length of the episode in the environment. Must be larger than the prediction horizon.
        """
        self._episode_length = max(self._prediction_horizon + 1, episode_length)

    @property
    def k(self):
        """
        Returns:
             The current step in the running episode
        """
        return self._k

    @property
    def limit_observer(self):
        return self._limit_observer

    @property
    def safety_margin(self):
        return self._safety_margin

    @property
    def prediction_horizon(self):
        return self._prediction_horizon

    @property
    def motor_parameter(self):
        """

        Returns:
            motor parameter with calculated limits
        """
        params = self.motor_model.motor_parameter
        params['safety_margin'] = self.safety_margin
        params['episode_length'] = self._episode_length
        params['prediction_horizon'] = self._prediction_horizon
        params['tau'] = self._tau
        params['limits'] = self._limits.tolist()
        return params

    @property
    def _reward(self):
        return self._reward_function

    # endregion

    def __init__(self, motor_type, state_vars, zero_refs, converter_type, tau, episode_length=10000, load_parameter=None,
                 motor_parameter=None, reward_weight=(('omega', 1.0),), on_dashboard=('omega',), integrator='euler',
                 nsteps=1, prediction_horizon=0, interlocking_time=0.0, noise_levels=0.0, reward_fct='swsae',
                 limit_observer='off', safety_margin=1.3, gamma=0.9, dead_time=True):
        """
        Basic setting of all the common motor parameters.

        Args:
            motor_type: Can be 'dc-series', 'dc-shunt', 'dc-extex' or 'dc-permex'. Set by the child classes.
            state_vars: State variables of the DC motor. Set by the child classes.
            zero_refs: State variables that get zero references. (E.g. to punish high control power)
            motor_parameter: A dict of motor parameters that differ from the default ones. \n
                            For details look into the dc_motor model.
            load_parameter: A dict of load parameters that differ from the default ones. \n
                            For details look into the load model.
            converter_type: The specific converter type.'{disc/cont}-{1Q/2Q/4Q}'. For details look into converter
            tau: The step size or sampling time of the environment.
            episode_length: The episode length of the environment
            reward_weight: Iterable of key/value pairs that specifies how the rewards in the environment
                          are weighted.
                          E.g. ::
                            (('omega', 0.9),('u', 0.1))
            on_dashboard: Iterable that specifies the variables on the dashboard.
                            E.g.::
                                ['omega','u']
            integrator: Select which integrator to choose from 'euler', 'dopri5'
            nsteps: Maximum allowed number of steps for the integrator.
            prediction_horizon: The length of future reference points that are shown to the agents
            interlocking_time: interlocking time of the converter
            noise_levels: Noise levels of the state variables in percentage of the signal power.
            reward_fct: Select the reward function between: (Each one normalised to [0,1] or [-1,0]) \n
                'swae': Absolute Error between references and state variables [-1,0] \n
                'swse': Squared Error between references and state  variables [-1,0]\n
                'swsae': Shifted absolute error / 1 + swae [0,1] \n
                'swsse': Shifted squared error  / 1 + swse [0,1]  \n

            limit_observer: Select the limit observing function. \n
                'off': No limits are observed. Episode goes on. \n
                'no_punish': Limits are observed, no punishment term for violation. This function should be used with
                shifted reward functions. \n
                'const_punish': Limits are observed. Punishment in the form of -1 / (1-gamma) to punish the agent with
                the maximum negative reward for the further steps. This function should be used with non shifted reward
                functions.
            safety_margin: Ratio between maximal and nominal power of the motor parameters.
            gamma: Parameter for the punishment of a limit violation. Should equal agents gamma parameter.

        """

        self._gamma = gamma
        self._safety_margin = safety_margin
        self._reward_function, self.reward_range = self._reward_functions(reward_fct)
        self._limit_observer = self._limit_observers(limit_observer)
        self._tau = tau
        self._episode_length = episode_length
        self.state_vars = np.array(state_vars)

        #: dict(int): Inverse state vars. Dictionary to map state names to positions in the state arrays
        self._state_var_positions = {}
        for ind, val in enumerate(state_vars):
            self._state_var_positions[val] = ind
        self._prediction_horizon = max(0, prediction_horizon)
        self._zero_refs = zero_refs

        #: array(bool): True, if the state variable on the index is a zero_reference. For fast access
        self._zero_ref_flags = np.isin(self.state_vars, self._zero_refs)
        self.load_model = load_models.Load(load_parameter)
        self.motor_model = dcmotor_model.make(motor_type, self.load_model.load, motor_parameter)
        self.converter_model = converter_models.Converter.make(converter_type, self._tau, interlocking_time, dead_time)
        self._k = 0
        self._dashboard = None
        self._state = np.zeros(len(state_vars))
        self._reference = np.zeros((len(self.state_vars), episode_length + prediction_horizon))
        self._reward_weights = np.zeros(len(self._state))

        self.reference_vars = np.zeros_like(self.state_vars, dtype=bool)
        self._on_dashboard = np.ones_like(self.state_vars, dtype=bool)
        if on_dashboard[0] == 'True':
            self._on_dashboard *= True
        elif on_dashboard[0] == 'False':
            self._on_dashboard *= False
        else:
            self._on_dashboard *= False
            for key in on_dashboard:
                self._on_dashboard[self._state_var_positions[key]] = True
        for key, val in reward_weight:
            self._reward_weights[self._state_var_positions[key]] = val
        for i in range(len(state_vars)):
            if self._reward_weights[i] > 0 and self.state_vars[i] not in self._zero_refs:
                self.reference_vars[i] = True

        integrators = ['euler', 'dopri5']
        assert integrator in integrators, f'Integrator was {integrator}, but has to be in {integrators}'
        if integrator == 'euler':
            self.system = EulerSolver(self._system_eq, nsteps)
        else:
            self.system = ode(self._system_eq, self._system_jac).set_integrator(integrator, nsteps=nsteps)
        self.integrate = self.system.integrate
        self.action_space = self.converter_model.action_space
        self._limits = np.zeros(len(self.state_vars))
        self._set_limits()
        self._set_observation_space()
        self._noise_levels = np.zeros(len(state_vars))
        if type(noise_levels) is tuple:
            for state_var, noise_level in noise_levels:
                self._noise_levels[self._state_var_positions[state_var]] = noise_level
        else:
            self._noise_levels = np.ones(len(self.state_vars)) * noise_levels
        self._noise = None
        self._resetDashboard = True

    def seed(self, seed=None):
        """
        Seed the random generators in the environment

        Args:
            seed: The value to seed the random number generator with
        """
        np.random.seed(seed)

    def _set_observation_space(self):
        """
        Child classes need to write their concrete observation space into self.observation_space here
        """
        raise NotImplementedError

    def _set_limits(self):
        """
        Child classes need to write their concrete limits of the state variables into self._limits here
        """
        raise NotImplementedError

    def _step_integrate(self, action):
        """
        The integration is done for one time period. The converter considers the dead time and interlocking time.

        Args:
            action: switching state of the converter that should be applied
        """
        raise NotImplementedError

    def step(self, action):
        """
        Clips the action to its limits and performs one step of the environment.

        Args:
            action: The action from the action space that will be performed on the motor

        Returns:
            Tuple(array(float), float, bool, dict):
                **observation:**    The observation from the environment \n
                **reward:**         The reward for the taken action \n
                **bool:**           Flag if the episode has ended \n
                **info:**           An always empty dictionary \n
        """
        last_state = np.array(self._state, copy=True)
        self._step_integrate(action)
        rew = self._reward(self._state/self._limits, self._reference[:, self._k].T)
        done, punish = self.limit_observer(self._state)
        observation_references = self._reference[self.reference_vars, self._k:self._k + self._prediction_horizon + 1]
        # normalize the observation
        observation = np.concatenate((
            self._state/self._limits + self._noise[:, self._k], observation_references.flatten()
        ))
        self._k += 1
        if done == 0:  # Check if period is finished
            done = self._k == self._episode_length
        else:
            rew = punish
        return observation, rew, done, {}

    def _set_initial_value(self):
        """
        call self.system.set_initial_value(initial_state, 0.0) to reset the state to initial.
        """
        self.system.set_initial_value(self._state[self.MOTOR_IDX], 0.0)

    def reset(self):
        """
        Resets the environment.

        All state variables will be set to a random value in [-nominal value, nominal value].
        New references will be generated.

        Returns:
            The initial observation for the episode

        """
        self._k = 0
        # Set new state
        self._set_initial_state()
        # New References
        self._generate_references()
        # Reset Integrator
        self._set_initial_value()
        # Reset Dashboard Flag
        self._resetDashboard = True
        # Generate new gaussian noise for the state variables
        self._noise = (
            np.sqrt(self._noise_levels/6) / self._safety_margin
            * np.random.randn(self._episode_length+1, len(self.state_vars))
        ).T

        # Calculate initial observation
        observation_references = self._reference[self.reference_vars, self._k:self._k + self._prediction_horizon+1]
        observation = np.concatenate((self._state/self._limits, observation_references.flatten()))
        return observation

    def render(self, mode='human'):
        """
        Call this function once a cycle to update the visualization with the current values.
        """
        if not self._on_dashboard.any():
            return
        if self._dashboard is None:
            # First Call: No dashboard was initialised before
            self._dashboard = MotorDashboard(self.state_vars[self._on_dashboard], self._tau,
                                             self.observation_space.low[:len(self.state_vars)][self._on_dashboard]
                                             * self._limits[self._on_dashboard],
                                             self.observation_space.high[:len(self.state_vars)][self._on_dashboard]
                                             * self._limits[self._on_dashboard],
                                             self._episode_length,
                                             self._safety_margin,
                                             self._reward_weights[self._on_dashboard] > 0)
        if self._resetDashboard:
            self._resetDashboard = False
            self._dashboard.reset((self._reference[self._on_dashboard].T * self._limits[self._on_dashboard]).T)
        self._dashboard.step(self._state[self._on_dashboard], self._k)  # Update the plot in the dashboard

    def close(self):
        """
        When the environment is closed the dashboard will also be closed.
        This function does not need to be called explicitly.
        """
        if self._dashboard is not None:

            self._dashboard.close()

    def _system_eq(self, t, state, u_in, noise):
        """
        The differential equation of the whole system consisting of the converter, load and motor.

        This function is called by the integrator.

        Args:
            t: Current time of the system
            state: The current state as a numpy array.
            u_in: Applied input voltage

        Returns:
            The solution of the system. The first derivatives of all the state variables of the system.
        """
        t_load = self.load_model.load(state[self.OMEGA_IDX])
        return self.motor_model.model(state, t_load, u_in + noise)

    def _system_jac(self, t, state):
        """
        The Jacobian matrix of the systems equation.

        Args:
            t: Current time of the system.
            state: Current state

        Returns:
            The solution of the Jacobian matrix for the current state
        """
        load_jac = self.load_model.jac(state)
        return self.motor_model.jac(state, load_jac)

    # region Reference Generation

    def _reference_sin(self, bandwidth=20):
        """
        Set sinus references for the state variables with a random amplitude, offset and phase shift

        Args:
            bandwidth: bandwidth of the system
        """
        x = np.arange(0, (self._episode_length + self._prediction_horizon))
        if self.observation_space.low[0] == 0.0:
            amplitude = np.random.rand() / 2
            offset = np.random.rand() * (1 - 2*amplitude) + amplitude
        else:
            amplitude = np.random.rand()
            offset = (2 * np.random.rand() - 1) * (1 - amplitude)
        t_min, t_max = self._set_time_interval_reference('sin', bandwidth)  # specify range for period time
        t_s = np.random.rand() * (t_max - t_min) + t_min
        phase_shift = 2 * np.pi * np.random.rand()
        self._reference = amplitude * np.sin(2 * np.pi / t_s * x * self.tau + phase_shift) + offset
        self._reference = self._reference*np.ones((len(self.state_vars), 1))/self._safety_margin

    def _reference_rect(self, bandwidth=20):
        """
        Set rect references for the state variables with a random amplitude, offset and phase shift

        Args:
            bandwidth: bandwidth of the system
        """
        x = np.arange(0, (self._episode_length + self._prediction_horizon))
        if self.observation_space.low[self.OMEGA_IDX] == 0.0:
            amplitude = np.random.rand()
            offset = np.random.rand() * (1 - amplitude)
        else:
            amplitude = 2 * np.random.rand() - 1
            offset = (-1 + np.random.rand() * (2 - np.abs(amplitude))) * np.sign(amplitude)

        t_min, t_max = self._set_time_interval_reference('rect', bandwidth)
        # specify range for period time
        t_s = np.random.rand() * (t_max - t_min) + t_min
        # time period on amplitude + offset value
        t_on = np.random.rand() * t_s
        # time period on offset value
        t_off = t_s - t_on
        reference = np.zeros(self._episode_length + self._prediction_horizon)
        reference[x * self.tau % (t_on + t_off) > t_off] = amplitude
        reference += offset
        self._reference = reference * np.ones((len(self.state_vars), 1)) / self._safety_margin

    def _reference_tri(self, bandwidth=20):
        """
        Set triangular reference with random amplitude, offset, times for rise and fall for all state variables

        Args:
            bandwidth: bandwidth of the system
        """
        t_min, t_max = self._set_time_interval_reference('tri', bandwidth)  # specify range for period time
        t_s = np.random.rand() * (t_max-t_min) + t_min
        t_rise = np.random.rand() * t_s
        t_fall = t_s - t_rise

        if self.observation_space.low[self.OMEGA_IDX] == 0.0:
            amplitude = np.random.rand()
            offset = np.random.rand() * (1 - amplitude)
        else:
            amplitude = 2 * np.random.rand() - 1
            offset = (-1 + np.random.rand() * (2 - np.abs(amplitude))) * np.sign(amplitude)

        reference = np.ones(self._episode_length + self._prediction_horizon)
        for t in range(0, (self._episode_length + self._prediction_horizon)):
            # use a triangular function
            if (t*self.tau) % t_s <= t_rise:
                reference[t] = ((t * self.tau) % t_s) / t_rise * amplitude + offset
            else:
                reference[t] = -((t * self.tau) % t_s - t_s) / t_fall * amplitude + offset
        self._reference = reference*np.ones((len(self.state_vars), 1))/self._safety_margin

    def _reference_sawtooth(self, bandwidth=20):
        """
        Sawtooth signal generator with random time period and amplitude

        Args:
            bandwidth: bandwidth of the system
        """
        t_min, t_max = self._set_time_interval_reference('sawtooth', bandwidth)  # specify range for period time
        t_s = np.random.rand() * (t_max - t_min) + t_min
        if self.observation_space.low[self.OMEGA_IDX] == 0.0:
            amplitude = np.random.rand()
        else:
            amplitude = 2 * np.random.rand() - 1
        x = np.arange(self.episode_length + self._prediction_horizon, dtype=float)
        self._reference = np.ones_like(x, dtype=float)
        self._reference *= (x * self.tau) % t_s * amplitude / t_s
        self._reference = self._reference * np.ones((len(self.state_vars), 1)) / self._safety_margin

    def _generate_references(self, bandwidth=20):
        """
        Select which reference to generate. The shaped references (rect, sin, triangular, sawtooth) are equally probable
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
        # Set the supply voltage.
        # In this step an additive noise to the supply voltage can be implemented in the future.
        u_sup = np.ones(self.episode_length + self._prediction_horizon) * self.motor_model.u_sup \
                / self._limits[self._state_var_positions['u_sup']]
        self._reference[self._state_var_positions['u_sup']] = u_sup
        # Reset all zero references to zero.
        self._reference[self._zero_ref_flags] = np.zeros((len(self._zero_refs),
                                                         self.episode_length + self._prediction_horizon))

    def _generate_random_references(self):
        """
        Each subclass needs to define its own random reference generation here.
        """
        raise NotImplementedError()

    def _generate_random_control_sequence(self, bw, maximum):
        """
        Function that is called by the random reference generation in the motors to generate a random control sequence.

        A random control sequence is applied onto the system and generates the reference trajectories.

        Args:
            bw: Bandwidth for the control sequence
            maximum: Maximum value for the control sequence

        Returns:
            A random control sequence that is following the bandwidth and power constraints at most.
        """
        ref_len = self.episode_length + self._prediction_horizon
        rands = np.random.randn(2, ref_len // 2)
        u = rands[0] + 1j * rands[1]
        bw_noise = np.random.rand() * 0.5
        bw *= bw_noise
        delta_w = 2 * np.pi / ref_len / self._tau
        u[int(bw / delta_w) + 1:] = 0.0
        sigma = np.linspace(1, 0, int(bw / delta_w) + 1)

        if len(sigma) < len(u):
            u[:len(sigma)] *= sigma
        else:
            u *= sigma[:len(u)]

        fourier = np.concatenate((np.random.randn(1), u, np.flip(np.conjugate(u))))
        u = np.fft.ifft(fourier).real

        power_noise = np.random.rand() + 0.5
        u = u * maximum / np.sqrt((u ** 2).sum() / ref_len) * power_noise
        leakage = np.random.rand(1) * 0.1
        voltage_offset = maximum * ((self.converter_model.voltages[1] - self.converter_model.voltages[0])
                                    * np.random.rand() + self.converter_model.voltages[0])
        u += voltage_offset
        u = np.clip(u, (self.converter_model.voltages[0] - leakage) * maximum,
                    (self.converter_model.voltages[1] + leakage) * maximum)
        return u[:ref_len]

    def _set_time_interval_reference(self, shape=None, bandwidth=20):
        """
        This function returns the minimum and maximum time period specified by the bandwidth of the motor,
        episode length and individual modifications for each shape
        At least on time period of a shape should fit in an episode, but not to fast that the motor can not follow the
        reference properly.

        Args:
            shape: shape of the reference

        Returns:
            Minimal and maximal time period
        """
        bw = self._maximal_bandwidth(bandwidth)  # Bandwidth of reference limited
        t_episode = (self.episode_length+self._prediction_horizon)*self.tau
        t_min = min(1 / bw, t_episode)
        t_max = max(1 / bw, t_episode)
        # In this part individual modifications can be made for each shape
        # Modify the values to get useful references. Some testing necessary to find practical values.
        if shape == 'sin':
            t_min = t_min
            t_max = t_max / 3
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
            t_max = t_max/5
        return min(t_min, t_max), max(t_min, t_max)  # make sure that the order is correct

    def _maximal_bandwidth(self, bandwidth=20):
        """
        Computes the maximal allowed bandwidth, considering a user defined limit and the technical limit.

        Args:
            bandwidth: Maximal user defined value for the bandwidth

        Returns:
            Maximal bandwidth for the reference
        """
        return min(self.motor_model.bandwidth(), bandwidth)

    # endregion

    def _set_initial_state(self):
        """
        Defined in each motor itself. Sets the initial environment state.
        """
        raise NotImplementedError

    # region Reward Functions

    def _reward_functions(self, key):
        """
        Selector for the concrete reward function selected by the key string

        Returns:
            The selected reward function.
        """
        return {
            # (Reward Function, Reward Range)
            'swae': (self._absolute_error, (-1, 0)),
            'swse':  (self._squared_error, (-1, 0)),
            'swsae': (self._shifted_absolute_error, (0, 1)),
            'swsse': (self._shifted_squared_error, (0, 1)),
        }[key]

    def _absolute_error(self, state, reference):
        """
        The weighted, absolute error between the reference and state variables normalised to [-1,0]

        Args:
            state: the current state of the environment
            reference: the current reference values of the observation variables

        Returns:
            The reward value
        """
        return -(self._reward_weights * np.abs(state - reference)
                 / (self.observation_space.high[:len(self.state_vars)]
                    - self.observation_space.low[:len(self.state_vars)])
                 ).sum()

    def _squared_error(self, state, reference):
        """
        The weighted, squared absolute error between the reference and state variables normalised to [-1,0]

        Args:
            state: the current state of the environment
            reference: the current reference values of the observation variables

        Returns:
            The reward value
        """
        return -(self._reward_weights *
                 ((state - reference)
                     / (self.observation_space.high[:len(self.state_vars)]
                        - self.observation_space.low[:len(self.state_vars)])
                  )**2
                 ).sum()

    def _shifted_squared_error(self, state, reference):
        """
        The weighted, squared error between the reference and state variables normalised to [0,1]

        Args:
            state: the current state of the environment
            reference: the current reference values of the observation variables

        Returns:
            The reward value
        """
        return 1 + self._squared_error(state, reference)

    def _shifted_absolute_error(self, state, reference):
        """
        The weighted, absolute error between the reference and state variables normalised to [0,1]

        Args:
            state: the current state of the environment
            reference: the current reference values of the observation variables

        Returns:
            The reward value
        """
        return 1 + self._absolute_error(state, reference)
    # endregion

    # region Limit Observers

    def _limit_observers(self, key):
        """
        Selector for the concrete limit observer by the key string.

        Returns:
            The selected limit observer function.
        """
        return {
            'off': self._no_observation,
            'no_punish': self._no_punish,
            'const_punish': self._const_punish,
        }[key]

    def _no_punish(self, state):
        """
        No reward punishment, only break the episode when limits are violated. Recommended for positive rewards.

        Args:
            state: Current state of the environment

        Returns:
            Tuple of a flag if the episode should be terminated and the punishment for the reward
        """
        if self._limits_violated(state):
            return False, 0.0
        else:
            return True, 0.0

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
        if self._limits_violated(state):
            return False, 0.0
        else:
            # Terminate the episode if constraints are violated
            return True, -1 * 1 / (1 - self._gamma)

    def _limits_violated(self, state):
        """
        Check, if any limit is violated.

        Args:
            state: Current state of the environment

        Returns:
            True, if any limit is violated, false otherwise.

        """
        return (np.abs(state) <= self.observation_space.high[:len(self.state_vars)] * self._limits).all()

    def _no_observation(self, *_):
        """
        No limit violations are observed. No punishment and the episode continues even after limit violations.

        Args:
            state: Current state of the motor

        Returns:
            Tuple of a flag if the episode should be terminated (here always false)
            and the punishment for the reward (here always 0)
        """
        return False, 0.0
    # endregion

    def get_motor_param(self):
        """
        Returns:
            This function returns all motor parameters, sampling time, safety margin and converter limits
        """
        params = self.motor_parameter
        params['converter_voltage'] = self.converter_model.voltages
        params['converter_current'] = self.converter_model.currents
        return params
