import numpy as np
from scipy.stats import truncnorm
import warnings

from ...random_component import RandomComponent


class MechanicalLoad(RandomComponent):
    """The MechanicalLoad is the base class for all the mechanical systems attached
    to the electrical motors rotor.

    It contains an mechanical ode system as well as the state names, limits and
    nominal values of the mechanical quantities. The only required state is
    'omega' as the rotational speed of the motor shaft in rad/s.
    ConstantSpeedLoad can be initialized with the initializer as an
    class parameter by instantiation. ExternalSpeedLoad takes the first value
    of the SpeedProfile as initial value.

    Initialization is given by initializer(dict). Can be a constant state value
    or random value in given interval.
    dict should be like:
        { 'states'(dict): with state names and initial values
          'interval'(array like): boundaries for each state
                    (only for random init), shape(num states, 2)
          'random_init'(str): 'uniform' or 'normal'
          'random_params(tuple): mue(float), sigma(int)

    Example initializer(dict) for constant initialization:
        { 'states': {'omega': 16.0}}
    Example  initializer(dict) for random initialization:
        { 'random_init': 'normal'}
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

    def __init__(self, state_names=None, j_load=0.0, load_initializer=None):
        """
        Args:
            state_names(list(str)): List of the names of the states in the mechanical-ODE.
            j_load(float): Moment of inertia of the load affecting the motor shaft.
        """
        RandomComponent.__init__(self)
        self._j_total = self._j_load = j_load
        self._state_names = list(state_names or ['omega'])
        self._limits = {}
        self._nominal_values = {}
        load_initializer = load_initializer or {}
        self._initializer = self._default_initializer.copy()
        self._initializer.update(load_initializer)
        self._initial_states = self._initializer.get('states', {state: 0.0 for state in self._state_names})

    def initialize(self, state_space, state_positions, nominal_state, **__):
        """Initializes the state of the load on an episode start.

        Values can be given as a constant or sampled random out of a statistical distribution. Initial value is in
        range of the nominal values or a given interval.

        Args:
            nominal_state(list): nominal values for each state given from physical system
            state_space(gym.spaces.Box): normalized state space boundaries
            state_positions(dict): indexes of system states
        """
        # for order and organization purposes
        interval = self._initializer['interval']
        random_dist = self._initializer['random_init']
        random_params = self._initializer['random_params']
        if isinstance(nominal_state, (list, np.ndarray)):
            nominal_state = np.asarray(nominal_state, dtype=float)
        elif isinstance(self._nominal_values, dict):
            nominal_state = [nominal_state[state] for state in self._initial_states.keys()]
            nominal_state = np.asarray(nominal_state)
        # setting nominal values as interval limits
        state_idx = [state_positions[state] for state in self._initial_states.keys()]
        upper_bound = nominal_state[state_idx]
        lower_bound = upper_bound * np.asarray(state_space.low, dtype=float)[state_idx]
        # clip nominal boundaries to user defined
        if interval is not None:
            lower_bound = np.clip(lower_bound, a_min=np.asarray(interval, dtype=float).T[0], a_max=None)
            upper_bound = np.clip(upper_bound, a_min=None, a_max=np.asarray(interval, dtype=float).T[1])
        else:
            pass
        # random initialization for each load state (omega)
        if random_dist is not None:
            if random_dist == 'uniform':
                initial_value = (upper_bound - lower_bound) \
                    * self.random_generator.uniform(size=len(self._initial_states.keys())) \
                    + lower_bound
                random_states = {
                    state: initial_value[idx] for idx, state in enumerate(self._initial_states.keys())
                }
                self._initial_states.update(random_states)

            elif random_dist in ['normal', 'gaussian']:
                # specific input or middle of interval
                mue = random_params[0] \
                      or (upper_bound - lower_bound) / 2 + lower_bound
                sigma = random_params[1] or 1
                a = (lower_bound - mue) / sigma
                b = (upper_bound - mue) / sigma
                initial_value = truncnorm.rvs(
                    a, b, loc=mue, scale=sigma, size=(len(self._initial_states.keys())),
                    random_state=self.seed_sequence.pool[0]
                )
                random_states = {
                    state: initial_value[idx] for idx, state in enumerate(self._initial_states.keys())
                }
                self._initial_states.update(random_states)
            else:
                raise NotImplementedError
        # constant initialization for each motor state (current, epsilon)
        elif self._initial_states is not None:
            initial_value = np.atleast_1d(list(self._initial_states.values()))
            # check init_value meets interval boundaries
            if (lower_bound <= initial_value).all() and (initial_value <= upper_bound).all():
                initial_states_ = {
                    state: initial_value[idx] for idx, state in enumerate(self._initial_states.keys())
                }
                self._initial_states.update(initial_states_)
            else:
                raise Exception('Initialization Value have to be in nominal boundaries')
        else:
            raise Exception('No matching Initialization Case')

    def reset(self, state_space, state_positions, nominal_state, **__):
        """
        Reset the motors state to a new initial state. (Default 0)

        Args:
            nominal_state(list): nominal values for each state given from
                                  physical system
            state_space(gym.Box): normalized state space boundaries
            state_positions(dict): indexes of system states
        Returns:
            numpy.ndarray(float): The initial motor states.
        """
        self.next_generator()
        if self._initializer:
            self.initialize(state_space, state_positions, nominal_state)
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
