import math
import numpy as np

from .electric_motor import ElectricMotor


class ThreePhaseMotor(ElectricMotor):
    """
            The ThreePhaseMotor and its subclasses implement the technical system of Three Phase Motors.

            This includes the system equations, the motor parameters of the equivalent circuit diagram,
            as well as limits and bandwidth.
    """
    # transformation matrix from abc to alpha-beta representation
    _t23 = 2 / 3 * np.array([
        [1, -0.5, -0.5],
        [0, 0.5 * np.sqrt(3), -0.5 * np.sqrt(3)]
    ])

    # transformation matrix from alpha-beta to abc representation
    _t32 = np.array([
        [1, 0],
        [-0.5, 0.5 * np.sqrt(3)],
        [-0.5, -0.5 * np.sqrt(3)]
    ])

    @staticmethod
    def t_23(quantities):
        """
        Transformation from abc representation to alpha-beta representation

        Args:
            quantities: The properties in the abc representation like ''[u_a, u_b, u_c]''

        Returns:
            The converted quantities in the alpha-beta representation like ''[u_alpha, u_beta]''
        """
        return np.matmul(ThreePhaseMotor._t23, quantities)

    @staticmethod
    def t_32(quantities):
        """
        Transformation from alpha-beta representation to abc representation

        Args:
            quantities: The properties in the alpha-beta representation like ``[u_alpha, u_beta]``

        Returns:
            The converted quantities in the abc representation like ``[u_a, u_b, u_c]``
        """
        return np.matmul(ThreePhaseMotor._t32, quantities)

    @staticmethod
    def q(quantities, epsilon):
        """
        Transformation of the dq-representation into alpha-beta using the electrical angle

        Args:
            quantities: Array of two quantities in dq-representation. Example [i_d, i_q]
            epsilon: Current electrical angle of the motor

        Returns:
            Array of the two quantities converted to alpha-beta-representation. Example [u_alpha, u_beta]
        """
        cos = math.cos(epsilon)
        sin = math.sin(epsilon)
        return cos * quantities[0] - sin * quantities[1], sin * quantities[
            0] + cos * quantities[1]

    @staticmethod
    def q_inv(quantities, epsilon):
        """Transformation of the alpha-beta-representation into dq using the electrical angle

        Args:
            quantities: Array of two quantities in alpha-beta-representation. Example [u_alpha, u_beta]
            epsilon: Current electrical angle of the motor

        Returns:
            Array of the two quantities converted to dq-representation. Example [u_d, u_q]

        Note:
            The transformation from alpha-beta to dq is just its inverse conversion with negated epsilon.
            So this method calls q(quantities, -epsilon).
        """
        return ThreePhaseMotor.q(quantities, -epsilon)

    def q_me(self, quantities, epsilon):
        """
        Transformation of the dq-representation into alpha-beta using the mechanical angle

        Args:
            quantities: Array of two quantities in dq-representation. Example [i_d, i_q]
            epsilon: Current mechanical angle of the motor

        Returns:
            Array of the two quantities converted to alpha-beta-representation. Example [u_alpha, u_beta]
        """
        return self.q(quantities, epsilon * self._motor_parameter['p'])

    def q_inv_me(self, quantities, epsilon):
        """
        Transformation of the alpha-beta-representation into dq using the mechanical angle

        Args:
            quantities: Array of two quantities in alpha-beta-representation. Example [u_alpha, u_beta]
            epsilon: Current mechanical angle of the motor

        Returns:
            Array of the two quantities converted to dq-representation. Example [u_d, u_q]

        Note:
            The transformation from alpha-beta to dq is just its inverse conversion with negated epsilon.
            So this method calls q(quantities, -epsilon).
        """
        return self.q_me(quantities, -epsilon)

    def _torque_limit(self):
        """
        Returns:
             Maximal possible torque for the given limits in self._limits
        """
        raise NotImplementedError()

    def _update_limits(self, limits_d=None, nominal_d=None):
        # Docstring of superclass
        limits_d = limits_d if limits_d is not None else dict()
        nominal_d = nominal_d if nominal_d is not None else dict()

        super()._update_limits(limits_d, nominal_d)
        super()._update_limits(dict(torque=self._torque_limit()))

    def _update_initial_limits(self, nominal_new=None, **kwargs):
        # Docstring of superclass
        nominal_new = nominal_new if nominal_new is not None else dict()
        super()._update_initial_limits(self._nominal_values)
        super()._update_initial_limits(nominal_new)
