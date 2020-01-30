import json
from numpy import array
from numpy.polynomial.polynomial import polyder
from .conf import default_load_parameter
import warnings


class Load(object):
    """
    The load represents the mechanical part of the motor.
    Is is a polynomial of degree 2, that can be defined by passing a dict with the following keys:
        'a': constant load\n
        'b': load growing linear with omega (eg friction)\n
        'c': load growing quadratically with omega (eg air resistance)\n
        'J_load': loads moment of inertia\n
    """

    def load(self, omega):
        """
        Function that is called by the system equation to determine the load.

        Args:
            omega: Current mechanical angular velocity of the motor in rad/s
        Returns:
            The load of the motor in Nm
        """
        if omega > 0:
            return self.load_array[0] + self.load_array[1] * omega + self.load_array[2] * omega ** 2
        elif omega < 0:
            return -self.load_array[0] + self.load_array[1] * omega - self.load_array[2] * omega ** 2
        else:
            return 0

    def jac(self, omega):
        """
        Jacobian Matrix of the load function. This function is called by some of the ode-solvers.

        Args:
            omega: Current mechanical angular velocity of the motor in rad/s

        Returns:
            The jacobian matrix of the load function at given speed
        """
        if omega > 0:
            return self._load_jac[0] + self._load_jac[1] * omega
        else:
            return -self._load_jac[0] + self._load_jac[1] * omega

    def __init__(self, load_parameter=None):
        """
        Args:
            load_parameter: dict containing the following keys: \n
                'c': Quadratic load coefficient \n
                'b': linear load coefficient \n
                'a': constant load coefficient \n
                'J_load': loads moment of inertia \n
                or an integer to select the default load parameters from conf.py
        """

        load_param = load_parameter or 0
        try:
            if type(load_param) is dict:
                self._load_parameter = load_parameter
            elif type(load_param) is int:
                self._load_parameter = default_load_parameter[load_param]
            else:
                raise TypeError
        except TypeError:
            warnings.warn('Invalid Type for motor parameter, fell back to default set 0')
            self._motor_parameter = default_load_parameter[0]
        except IndexError:
            warnings.warn('Invalid Index for motor parameter, fell back to default set 0')
        # Load parameters written to array for faster computation
        # This hard coding is used the keep the order, even if it is changed in the series
        self.load_array = [self._load_parameter['c'], self._load_parameter['b'], self._load_parameter['a']]
        self.j_load = self._load_parameter['J_load']
        # jac array predefined for faster computation
        self._load_jac = list(polyder(self.load_array))



