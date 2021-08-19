import numpy as np

from .mechanical_load import MechanicalLoad
from gym_electric_motor.utils import update_parameter_dict


class PolynomialStaticLoad(MechanicalLoad):
    """ Mechanical system that models the Mechanical-ODE based on a static polynomial load torque.

    Parameter dictionary entries:
        | a: Constant Load Torque coefficient (for modeling static friction)
        | b: Linear Load Torque coefficient (for modeling sliding friction)
        | c: Quadratic Load Torque coefficient (for modeling air resistances)
        | j_load: Moment of inertia of the mechanical system.
    """

    _load_parameter = dict(a=0.0, b=0.0, c=0., j_load=1e-5)
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

    def __init__(self, load_parameter=None, limits=None, load_initializer=None):
        """
        Args:
            load_parameter(dict(float)): Parameter dictionary.
            limits(dict):
            load_initializer(dict):
        """
        load_parameter = load_parameter if load_parameter is not None else dict()
        self._load_parameter = update_parameter_dict(self._load_parameter, load_parameter)
        super().__init__(j_load=self._load_parameter['j_load'], load_initializer=load_initializer)
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
