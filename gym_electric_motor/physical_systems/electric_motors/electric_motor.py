import numpy as np
from scipy.stats import truncnorm

from ...random_component import RandomComponent
from gym_electric_motor.utils import update_parameter_dict


class ElectricMotor(RandomComponent):
    """Base class for all technical electrical motor models.

        A motor consists of the ode-state. These are the dynamic quantities of its ODE.
        For example:
            ODE-State of a DC-shunt motor: `` [i_a, i_e ] ``
                * i_a: Anchor circuit current
                * i_e: Exciting circuit current

        Each electric motor can be parametrized by a dictionary of motor parameters,
        the nominal state dictionary and the limit dictionary.

        Initialization is given by initializer(dict). It can be constant state value
        or random value in given interval.
        dict should be like:
        { 'states'(dict): with state names and initital values
                  'interval'(array like): boundaries for each state
                            (only for random init), shape(num states, 2)
                  'random_init'(str): 'uniform' or 'normal'
                  'random_params(tuple): mue(float), sigma(int)
        Example initializer(dict) for constant initialization:
            { 'states': {'omega': 16.0}}
        Example  initializer(dict) for random initialization:
            { 'random_init': 'normal'}
    """

    #: Parameter indicating if the class is implementing the optional jacobian function
    HAS_JACOBIAN = False

    #: CURRENTS_IDX(list(int)): Indices for accessing all motor currents.
    CURRENTS_IDX = []

    #: CURRENTS(list(str)): List of the motor currents names
    CURRENTS = []
    #: VOLTAGES(list(str)): List of the motor input voltages names
    VOLTAGES = []

    #: _default_motor_parameter(dict): Default parameter dictionary for the motor
    _default_motor_parameter = {}
    #: _default_nominal_values(dict(float)): Default nominal motor state array
    _default_nominal_values = {}
    #: _default_limits(dict(float)): Default motor limits (0 for unbounded limits)
    _default_limits = {}
    #: _default_initial_state(dict): Default initial motor-state values
    _default_initializer = {
        'states': {},
        'interval': None,
        'random_init': None,
        'random_params': None
    }
    #: _default_initial_limits(dict): Default limit for initialization
    _default_initial_limits = {}

    @property
    def nominal_values(self):
        """
        Readonly motors nominal values.

        Returns:
            dict(float): Current nominal values of the motor.
        """
        return self._nominal_values

    @property
    def limits(self):
        """
        Readonly motors limit state array. Entries are set to the maximum physical possible values
        in case of unspecified limits.

        Returns:
            dict(float): Limits of the motor.
        """
        return self._limits

    @property
    def motor_parameter(self):
        """
        Returns:
             dict(float): The motors parameter dictionary
        """
        return self._motor_parameter

    @property
    def initializer(self):
        """
        Returns:
            dict: Motor initial state and additional initializer parameter
        """
        return self._initializer

    @property
    def initial_limits(self):
        """
        Returns:
            dict: nominal motor limits for choosing initial values
        """
        return self._initial_limits

    def __init__(
            self, motor_parameter=None, nominal_values=None, limit_values=None, motor_initializer=None,
            initial_limits=None
    ):
        """
        :param  motor_parameter: Motor parameter dictionary. Contents specified
                for each motor.
        :param  nominal_values: Nominal values for the motor quantities.
        :param  limit_values: Limits for the motor quantities.
        :param  motor_initializer: Initial motor states (currents)
                            ('constant', 'uniform', 'gaussian' sampled from
                             given interval or out of nominal motor values)
        :param initial_limits: limits for of the initial state-value
        """
        RandomComponent.__init__(self)
        motor_parameter = motor_parameter or {}
        self._motor_parameter = self._default_motor_parameter.copy()
        self._motor_parameter = update_parameter_dict(self._default_motor_parameter, motor_parameter)
        limit_values = limit_values or {}
        self._limits = update_parameter_dict(self._default_limits, limit_values)
        nominal_values = nominal_values or {}
        self._nominal_values = update_parameter_dict(self._default_nominal_values, nominal_values)
        motor_initializer = motor_initializer or {}
        self._initializer = update_parameter_dict(self._default_initializer, motor_initializer)
        self._initial_states = {}
        if self._initializer['states'] is not None:
            self._initial_states.update(self._initializer['states'])
        # intialize limits, in general they're not needed to be changed
        # during  training or episodes
        initial_limits = initial_limits or {}
        self._initial_limits = self._nominal_values.copy()
        self._initial_limits.update(initial_limits)
        # preventing wrong user input for the basic case
        assert isinstance(self._initializer, dict), 'wrong initializer'

    def electrical_ode(self, state, u_in, omega, *_):
        """Calculation of the derivatives of each motor state variable for the given inputs / The motors ODE-System.

        Args:
            state(ndarray(float)): The motors state.
            u_in(list(float)): The motors input voltages.
            omega(float): Angular velocity of the motor

        Returns:
             ndarray(float): Derivatives of the motors ODE-system for the given inputs.
        """
        raise NotImplementedError

    def electrical_jacobian(self, state, u_in, omega, *_):
        """
        Calculation of the jacobian of each motor ODE for the given inputs / The motors ODE-System.

        Overriding this method is optional for each subclass. If it is overridden, the parameter HAS_JACOBIAN must also
        be set to True. Otherwise, the jacobian will not be called.

        Args:
            state(ndarray(float)): The motors state.
            u_in(list(float)): The motors input voltages.
            omega(float): Angular velocity of the motor

        Returns:
             Tuple(ndarray, ndarray, ndarray):
                [0]: Derivatives of all electrical motor states over all electrical motor states shape:(states x states)
                [1]: Derivatives of all electrical motor states over omega shape:(states,)
                [2]: Derivative of Torque over all motor states shape:(states,)
        """
        pass

    def initialize(self, state_space, state_positions, **__):
        """
        Initializes given state values. Values can be given as a constant or
        sampled random out of a statistical distribution. Initial value is in
        range of the nominal values or a given interval. Values are written in
        initial_states attribute

        Args:
            state_space(gym.Box): normalized state space boundaries (given by physical system)
            state_positions(dict): indices of system states (given by physical system)
        """
        # for organization purposes
        interval = self._initializer['interval']
        random_dist = self._initializer['random_init']
        random_params = self._initializer['random_params']
        self._initial_states.update(self._default_initializer['states'])
        if self._initializer['states'] is not None:
            self._initial_states.update(self._initializer['states'])

        # different limits for InductionMotor
        if any(map(lambda state: state in self._initial_states.keys(),
                   ['psi_ralpha', 'psi_rbeta'])):
            nominal_values_ = [self._initial_limits[state]
                               for state in self._initial_states]
            upper_bound = np.asarray(np.abs(nominal_values_), dtype=float)
            # state space for Induction Envs based on documentation
            # ['i_salpha', 'i_sbeta', 'psi_ralpha', 'psi_rbeta', 'epsilon']
            # hardcoded for induction motors currently given in the toolbox
            state_space_low = np.array([-1, -1, -1, -1, -1])
            lower_bound = upper_bound * state_space_low
        else:
            if isinstance(self._nominal_values, dict):
                nominal_values_ = [self._nominal_values[state]
                                   for state in self._initial_states.keys()]
                nominal_values_ = np.asarray(nominal_values_)
            else:
                nominal_values_ = np.asarray(self._nominal_values)

            state_space_idx = [
                state_positions[state] for state in self._initial_states.keys()
            ]

            upper_bound = np.asarray(nominal_values_, dtype=float)
            lower_bound = upper_bound * np.asarray(state_space.low, dtype=float)[state_space_idx]
        # clip nominal boundaries to user defined
        if interval is not None:
            lower_bound = np.clip(
                lower_bound,
                a_min=np.asarray(interval, dtype=float).T[0],
                a_max=None
            )
            upper_bound = np.clip(
                upper_bound,
                a_min=None,
                a_max=np.asarray(interval, dtype=float).T[1]
            )
        # random initialization for each motor state (current, epsilon)
        if random_dist is not None:
            if random_dist == 'uniform':
                initial_value = (upper_bound - lower_bound) \
                    * self._random_generator.uniform(size=len(self._initial_states.keys())) \
                    + lower_bound
                # writing initial values in initial_states dict
                random_states = {
                    state: initial_value[idx] for idx, state in enumerate(self._initial_states.keys())
                }
                self._initial_states.update(random_states)

            elif random_dist in ['normal', 'gaussian']:
                # specific input or middle of interval
                mue = random_params[0] or (upper_bound - lower_bound) / 2 + lower_bound
                sigma = random_params[1] or 1
                a, b = (lower_bound - mue) / sigma, (upper_bound - mue) / sigma
                initial_value = truncnorm.rvs(
                    a, b, loc=mue, scale=sigma, size=(len(self._initial_states.keys())),
                    random_state=self.seed_sequence.pool[0]
                )
                # writing initial values in initial_states dict
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
            if ((lower_bound <= initial_value).all()
                    and (initial_value <= upper_bound).all()):
                initial_states_ = \
                    {state: initial_value[idx]
                     for idx, state in enumerate(self._initial_states.keys())}
                self._initial_states.update(initial_states_)
            else:
                raise Exception('Initialization value has to be within nominal boundaries')
        else:
            raise Exception('No matching Initialization Case')

    def reset(self, state_space, state_positions, **__):
        """Reset the motors state to a new initial state. (Default 0)

        Args:
            state_space(gym.Box): normalized state space boundaries
            state_positions(dict): indexes of system states
        Returns:
            numpy.ndarray(float): The initial motor states.
        """
        # check for valid initializer
        self.next_generator()
        if self._initializer and self._initializer['states']:
            self.initialize(state_space, state_positions)
            return np.asarray(list(self._initial_states.values()))
        else:
            return np.zeros(len(self.CURRENTS))

    def i_in(self, state):
        """
        Args:
            state(ndarray(float)): ODE state of the motor

        Returns:
             list(float): List of all currents flowing into the motor.
        """
        raise NotImplementedError

    def _update_limits(self, limits_d=None, nominal_d=None):
        """Replace missing limits and nominal values with physical maximums.

        Args:
            limits_d(dict): Mapping: quantity to its limit if not specified
        """
        if limits_d is None:
            limits_d = dict()
        if nominal_d is None:
            nominal_d = dict()
        # omega is replaced the same way for all motor types
        limits_d.update(dict(omega=self._default_limits['omega']))

        for qty, lim in limits_d.items():
            if self._limits.get(qty, 0) == 0:
                self._limits[qty] = lim

        for entry in self._limits.keys():
            if self._nominal_values.get(entry, 0) == 0:
                self._nominal_values[entry] = nominal_d.get(entry, self._limits[entry])

    def _update_initial_limits(self, nominal_new=None):
        """Complete initial states with further state limits

        Args:
            nominal_new(dict): new/further state limits
        """
        nominal_new = dict() if nominal_new is None else nominal_new
        self._initial_limits.update(nominal_new)
