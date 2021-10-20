import numpy as np

from .mechanical_load import MechanicalLoad
from gym_electric_motor.utils import update_parameter_dict


class PolynomialStaticLoad(MechanicalLoad):
    """ Mechanical system that models the Mechanical-ODE based on a static polynomial load torque.

    Parameter dictionary entries:
        - :math:`a / Nm`: Constant Load Torque coefficient (for modeling static friction)
        - :math:`b / (Nm s)`: Linear Load Torque coefficient (for modeling sliding friction)
        - :math:`c / (Nm s^2)`: Quadratic Load Torque coefficient (for modeling air resistances)
        - :math:`j_load / (kg m^2)` : Moment of inertia of the mechanical system.

    Usage Example:
        >>> import gym_electric_motor as gem
        >>> from gym_electric_motor.physical_systems.mechanical_loads import PolynomialStaticLoad
        >>>
        >>> # Create a custom PolynomialStaticLoad instance
        >>> my_poly_static_load = PolynomialStaticLoad(
        ...     load_parameter=dict(a=1e-3, b=1e-4, c=0.0, j_load=1e-3),
        ...     limits=dict(omega=150.0), # rad / s
        ... )
        >>>
        >>> env = gem.make(
        ...     'Cont-SC-ExtExDc-v0',
        ...     load=my_poly_static_load
        ... )
        >>> done = True
        >>> for _ in range(1000):
        >>>     if done:
        >>>         state, reference = env.reset()
        >>>     env.render()
        >>>     (state, reference), reward, done, _ = env.step(env.action_space.sample())

    """

    _load_parameter = dict(a=0.0, b=0.0, c=0., j_load=1e-5)
    _default_initializer = {
        'states': {'omega': 0.0},
        'interval': None,
        'random_init': None,
        'random_params': (None, None)
    }

    #: Time constant to smoothen the static load functions constant term "a" around 0 velocity
    # Steps of a lead to unstable behavior of the ode-solver.
    tau_decay = 1e-3

    #: Parameter indicating if the class is implementing the optional jacobian function
    HAS_JACOBIAN = True

    @property
    def load_parameter(self):
        """
        Returns:
            dict(float): Parameter dictionary of the load.
        """
        return self._load_parameter

    def __init__(self, load_parameter=None, limits=None, load_initializer=None):
        """
        Args:
            load_parameter(dict(float)): Parameter dictionary. Keys: ``'a', 'b', 'c', 'j_load'``
            limits(dict): dictionary to update the limits of the load-instance. Keys: ``'omega'``
            load_initializer(dict): Dictionary to parameterize the initializer.
        """
        load_parameter = load_parameter if load_parameter is not None else dict()
        self._load_parameter = update_parameter_dict(self._load_parameter, load_parameter)
        super().__init__(j_load=self._load_parameter['j_load'], load_initializer=load_initializer)
        self._limits.update(limits or {})
        self._a = self._load_parameter['a']
        self._b = self._load_parameter['b']
        self._c = self._load_parameter['c']

    def _static_load(self, omega):
        """Calculation of the load torque for a given speed omega."""
        sign = 1 if omega > 0 else -1 if omega < -0 else 0
        # Limit the constant load term 'a' for velocities around zero for a more stable integration
        a = sign * self._a \
            if abs(omega) > self._a / self._j_total * self.tau_decay \
            else self._j_total / self.tau_decay * omega
        return sign * self._c * omega**2 + self._b * omega + a

    def mechanical_ode(self, t, mechanical_state, torque):
        # Docstring of superclass
        omega = mechanical_state[self.OMEGA_IDX]
        static_torque = self._static_load(omega)
        total_torque = torque - static_torque
        return np.array([total_torque / self._j_total])

    def mechanical_jacobian(self, t, mechanical_state, torque):
        # Docstring of superclass
        omega = mechanical_state[self.OMEGA_IDX]
        sign = 1 if omega > 0 else -1 if omega < 0 else 0
        # Linear region of the constant load term 'a' ?
        a = 0 if abs(omega) > self._a * self.tau_decay / self._j_total else self._j_total / self.tau_decay
        return np.array([[(-self._b - 2 * sign * self._c * omega - a) / self._j_total]]), \
            np.array([1 / self._j_total])
