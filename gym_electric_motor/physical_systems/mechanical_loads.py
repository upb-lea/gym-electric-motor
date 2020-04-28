import numpy as np


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

    OMEGA_IDX = 0

    #: Parameter indicating if the class is implementing the optional jacobian function
    HAS_JACOBIAN = False

    def __init__(self, state_names=None, j_load=0.0, **__):
        """
        Args:
            state_names(list(str)): List of the names of the states in the mechanical-ODE.
            j_load(float): Moment of inertia of the load affecting the motor shaft.
        """
        self._j_total = self._j_load = j_load
        self._state_names = list(state_names or ['omega'])
        self._limits = {}
        self._nominal_values = {}

    def reset(self, *_, **__):
        """
        Reset the motors mechanical-ODE to an initial state (default: all zeros).

        Returns:
            ndarray(float): Initial state of the mechanical-ODE.
        """
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

    #: Parameter indicating if the class is implementing the optional jacobian function
    HAS_JACOBIAN = True

    @property
    def load_parameter(self):
        """
        Returns:
            dict(float): Parameter dictionary of the load.
        """
        return self._load_parameter

    def __init__(self, load_parameter=(), limits=None, **__):
        """
        Args:
            load_parameter(dict(float)): Parameter dictionary.
        """
        self._load_parameter.update(load_parameter)
        super().__init__(j_load=self._load_parameter['j_load'])
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
       Constant speed mechanical load system which will always set the speed to a predefined value.
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




