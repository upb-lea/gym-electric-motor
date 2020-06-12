import numpy as np
from scipy.stats import truncnorm


class MechanicalLoad:
    """
    The MechanicalLoad is the base class for all the mechanical systems attached to the electrical motors rotor.

    It contains an mechanical ode system as well as the state names, limits and nominal values
    of the mechanical quantities. The only required state is 'omega' as the rotational speed of the motor shaft
    in rad/s.
    """

    @property
    def j_total(self):
        """
        Returns:
             float: Total moment of inertia affecting the motor shaft.
        """
        return self._j_total

    @property
    def state_names(self):
        """
        Returns:
            list(str): Names of the states in the mechanical-ODE.
        """
        return self._state_names

    @property
    def limits(self):
        """
        Returns:
            dict(float): Mapping of the motor states to their limit values.
        """
        return self._limits

    @property
    def nominal_values(self):
        """
        Returns:
              dict(float): Mapping of the motor states to their nominal values

        """
        return self._nominal_values

    @property
    def initializer(self):
        """
        Returns:
            dict: The motors initial state and additional initializer parameters
        """
        return self._initializer

    OMEGA_IDX = 0

    #: Parameter indicating if the class is implementing the optional jacobian function
    HAS_JACOBIAN = False
    #: _default_initial_state(dict): Default initial motor-state values
    _default_initializer = {}

    def __init__(self, state_names=None, j_load=0.0, load_initializer=None, **__):
        """
        Args:
            state_names(list(str)): List of the names of the states in the mechanical-ODE.
            j_load(float): Moment of inertia of the load affecting the motor shaft.
        """
        self._j_total = self._j_load = j_load
        self._state_names = list(state_names or ['omega'])
        self._limits = {}
        self._nominal_values = {}
        load_initializer = load_initializer or {}
        self._initializer = self._default_initializer.copy()
        self._initializer.update(load_initializer)
        self._initial_states = self._initializer['states']

    def initialize(self,
                   nominal_state,
                   state_space=None,
                   state_positions=None,
                   **__):
        """
        Initializes give state values. Values can be given as a constant or
        sampled random out of a statistical distribution. Initial value is in
        range of the nominal values or a given interval.

        Args:
            nominal_state(list): nominal values fpr each state given from
                                  physical system
            state_space(gym.Box): normalized state space boundaries
            state_positions(dict): indexes of system states
        Returns:
            ndarray(float): initial value for each state

        """
        # for order and organization purposes
        interval = self._initializer['interval']
        random_dist = self._initializer['random_init']
        random_params = self._initializer['random_params']

        # setting nominal values as interval limits
        state_idx = [state_positions[state] for state in self._initial_states.keys()]
        upper_bound = np.asarray(nominal_state[state_idx], dtype=float)
        lower_bound = upper_bound * np.asarray(state_space.low, dtype=float)
        # clip nominal boundaries to user defined
        if interval is not None:
            lower_bound = np.clip(lower_bound,
                                  a_min=
                                  np.asarray(self.interval, dtype=float).T[0],
                                  a_max=None)
            upper_bound = np.clip(upper_bound,
                                  a_min=None,
                                  a_max=
                                  np.asarray(self.interval, dtype=float).T[1])
        else:
            pass
        # if (interval is not None
        #         and isinstance(interval, (list, np.ndarray, tuple))):
        #     lower_bound = np.asarray(interval).T[0]
        #     upper_bound = np.asarray(interval).T[1]
        #     # clip bounds to nominal values
        #     nom_currents = np.asarray([nominal_values[s]
        #                                for s in initial_states.keys()])
        #     upper_bound = np.clip(upper_bound,
        #                           a_min=None,
        #                           a_max=nom_currents)
        #     lower_bound = np.clip(lower_bound,
        #                           a_min=np.zeros(len(initial_states.keys()),
        #                                          dtype=float),
        #                           a_max=None)
        # # use nominal limits as interval in state space
        # else:
        #     upper_bound = np.asarray([nominal_values[s]
        #                               for s in initial_states.keys()])
        #     lower_bound = np.zeros_like(upper_bound)

        # random initialization for each motor state (current, epsilon)

            if random_dist is not None:
                # initial_states = dict.fromkeys(initial_states.keys(), None)
                if random_dist == 'uniform':
                    initial_value = (upper_bound - lower_bound) * \
                                    np.random.random_sample(
                                        len(self._initial_states.keys())) + \
                                    lower_bound
                    random_states = \
                        {state: initial_value[idx]
                         for idx, state in
                         enumerate(self._initial_states.keys())}
                    self._initial_states.update(random_states)

                #               return np.ones(len(self._initial_states.keys()),
                #                             dtype=float) * initial_value
                elif random_dist in ['normal', 'gaussian']:
                    mue = random_params[0] or upper_bound / 2
                    sigma = random_params[1] or 1
                    a, b = (lower_bound - mue) / sigma, (
                                upper_bound - mue) / sigma
                    initial_value = truncnorm.rvs(a, b,
                                                  loc=mue,
                                                  scale=sigma,
                                                  size=len(
                                                      self._initial_states.keys()))
                    random_states = \
                        {state: initial_value[idx]
                         for idx, state in
                         enumerate(self._initial_states.keys())}
                    self._initial_states.update(random_states)
                #                return np.ones(len(self._initial_states.keys()),
                #                              dtype=float) * initial_value
                else:
                    # todo implement other distribution
                    raise NotImplementedError

            # constant initialization for each motor state (current, epsilon)
            elif self._initial_states is not None:
                initial_value = np.atleast_1d(
                    list(self._initial_states.values()))
                # check init_value meets interval boundaries
                if ((lower_bound <= initial_value).all()
                        and (initial_value <= upper_bound).all()):
                    initial_states_ = \
                        {state: initial_value[idx]
                         for idx, state in
                         enumerate(self._initial_states.keys())}
                    self._initial_states.update(initial_states_)
                    # return np.ones(len(initial_states.keys()),
                    #              dtype=float) * initial_value
                else:
                    raise Exception(
                        'Initialization Value have to be in nominal '
                        'boundaries')
            else:
                raise Exception('No matching Initialization Case')

    def reset(self,
              state_positions=None,
              state_space=None,
              nominal_state=None,
              **__):
        """
        Reset the motors state to a new initial state. (Default 0)

        Args:
            state_positions(dict): indexes of system states
        Returns:
            numpy.ndarray(float): The initial motor states.
        """
        if self._initializer:
            self.initialize(nominal_state, state_space, state_positions)
            return np.asarray(list(self._initial_states.values()))
        else:
            return np.zeros_like(self._state_names, dtype=float)

    def set_j_rotor(self, j_rotor):
        """
        Args:
            j_rotor(float): The moment of inertia of the rotor shaft of the motor.
        """
        self._j_total += j_rotor

    def mechanical_ode(self, t, mechanical_state, torque):
        """
        Calculation of the derivatives of the mechanical-ODE for each of the mechanical states.

        Args:
            t(float): Current time of the system.
            mechanical_state(ndarray(float)): Current state of the mechanical system.
            torque(float): Generated input torque by the electrical motor.

        Returns:
            ndarray(float): Derivatives of the mechanical state for the given input torque.
        """
        raise NotImplementedError

    def mechanical_jacobian(self, t, mechanical_state, torque):
        """
        Calculation of the jacobians of the mechanical-ODE for each of the mechanical state.

        Overriding this method is optional for each subclass. If it is overridden, the parameter HAS_JACOBIAN must also
        be set to True. Otherwise, the jacobian will not be called.

        Args:
            t(float): Current time of the system.
            mechanical_state(ndarray(float)): Current state of the mechanical system.
            torque(float): Generated input torque by the electrical motor.

        Returns:
            Tuple(ndarray, ndarray):
                [0]: Derivatives of the mechanical_state-odes over the mechanical_states shape:(states x states)
                [1]: Derivatives of the mechanical_state-odes over the torque shape:(states,)
        """
        pass

    def get_state_space(self, omega_range):
        """
        Args:
            omega_range(Tuple(int,int)): Lower and upper values the motor can generate for omega normalized to (-1, 1)

        Returns:
            Tuple(dict,dict): Lowest and highest possible values for all states normalized to (-1, 1)
        """
        return {'omega': omega_range[0]}, {'omega': omega_range[1]}


class PolynomialStaticLoad(MechanicalLoad):
    """
    Mechanical system that models the Mechanical-ODE based on a static polynomial load torque.

    Parameter dictionary entries:
        | a: Constant Load Torque coefficient (for modeling static friction)
        | b: Linear Load Torque coefficient (for modeling sliding friction)
        | c: Quadratic Load Torque coefficient (for modeling air resistances)
        | j_load: Moment of inertia of the mechanical system.
    """

    _load_parameter = dict(a=0.0, b=0.0, c=0., j_load=0)
    _default_initializer = {'states': {'omega': 0.0},
                            'interval': None,
                            'random_init': None,
                            'random_params': (None, None)}

    #: Parameter indicating if the class is implementing the optional jacobian function
    HAS_JACOBIAN = True

    @property
    def load_parameter(self):
        """
        Returns:
            dict(float): Parameter dictionary of the load.
        """
        return self._load_parameter

    def __init__(self, load_parameter=(), limits=None,
                 load_initializer=None, **__):
        """
        Args:
            load_parameter(dict(float)): Parameter dictionary.
        """
        self._load_parameter.update(load_parameter)
        super().__init__(j_load=self._load_parameter['j_load'],
                         load_initializer=load_initializer)
        self._limits.update(limits or {})
        self._a = self._load_parameter['a']
        self._b = self._load_parameter['b']
        self._c = self._load_parameter['c']

    def _static_load(self, omega, *_):
        """
        Calculation of the load torque for a given speed omega.
        """
        sign = 1 if omega > 0 else -1 if omega < 0 else 0
        return sign * (self._c * omega**2 + self._b * abs(omega) + self._a)

    def mechanical_ode(self, t, mechanical_state, torque):
        # Docstring of superclass
        return np.array([(torque - self._static_load(mechanical_state[self.OMEGA_IDX])) / self._j_total])

    def mechanical_jacobian(self, t, mechanical_state, torque):
        # Docstring of superclass
        sign = 1 if mechanical_state[self.OMEGA_IDX] > 0 else -1 if mechanical_state[self.OMEGA_IDX] < 0 else 0
        return np.array([[(-self._b * sign - 2 * self._c * mechanical_state[self.OMEGA_IDX])/self._j_total]]), \
            np.array([1/self._j_total])


class ConstantSpeedLoad(MechanicalLoad):
    """
       Constant speed mechanical load system which will always set the speed
       to a predefined value.
    """

    HAS_JACOBIAN = True

    @property
    def omega_fixed(self):
        """
        Returns:
            float: Constant value for omega in rad/s.
        """
        return self._omega

    def reset(self, *_, **__):
        # Docstring from superclass
        return np.array([self._omega])

    def __init__(self, omega_fixed=0, **__):
        """
        Args:
            omega_fixed(float)): Fix value for the speed in rad/s.
        """
        super().__init__()
        self._omega = omega_fixed

    def mechanical_ode(self, *_, **__):
        # Docstring of superclass
        return np.array([0])

    def mechanical_jacobian(self, t, mechanical_state, torque):
        # Docstring of superclass
        return np.array([0]), np.array([0])


class PositionalPolyStaticLoad(PolynomialStaticLoad):

    def __init__(self, load_parameter=None, limits=None):
        load_parameter = load_parameter or {}
        limits = limits or {}
        limits.setdefault('position', 1)
        load_parameter.set_default('gear_ratio', 1)
        load_parameter.set_default('meter_per_revolution', 0.05)
        super().__init__(load_parameter, limits)
        self._state_names = ['omega', 'position']

    def get_state_space(self, omega_range):
        lower, upper = super().get_state_space(omega_range)
        lower['position'] = 0
        upper['position'] = self._limits['position']




