import numpy as np

from .mechanical_load import MechanicalLoad


class ConstantSpeedLoad(MechanicalLoad):
    """
       Constant speed mechanical load system which will always set the speed
       to a predefined value.
    """

    HAS_JACOBIAN = True
    _default_initializer = {'states': {'omega': 0.0},
                            'interval': None,
                            'random_init': None,
                            'random_params': (None, None)}

    @property
    def omega_fixed(self):
        """
        Returns:
            float: Constant value for omega in rad/s.
        """
        return self._omega

    def __init__(self, omega_fixed=0, load_initializer=None, **kwargs):
        """
        Args:
            omega_fixed(float)): Fix value for the speed in rad/s.
        """
        super().__init__(load_initializer=load_initializer, **kwargs)
        self._omega = omega_fixed or self._initializer['states']['omega']
        if omega_fixed:
            self._initializer['states']['omega'] = omega_fixed

    def mechanical_ode(self, *_, **__):
        # Docstring of superclass
        return np.array([0])

    def mechanical_jacobian(self, t, mechanical_state, torque):
        # Docstring of superclass
        return np.array([0]), np.array([0])
